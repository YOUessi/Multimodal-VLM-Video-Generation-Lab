"""Batch47D: DEV-only multihop KLT crossfit controls and data firewall."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import dev_temporal_bridge_affine as bridge


def texture(seed: int = 731, h: int = 160, w: int = 224):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def shift(frame, dx, dy=0):
    h, w = frame.shape
    return cv2.warpAffine(frame, np.float32([[1, 0, dx], [0, 1, dy]]),
                          (w, h), flags=cv2.INTER_LINEAR)


def test_single_hop_known_translation():
    frames = [texture(), shift(texture(), 4, -3)]
    r = bridge.chained_cycle_matches(frames)
    assert r["detected"] > 100
    assert r["survivors"] > 60
    residual = r["dst"] - r["src"] - [4, -3]
    assert np.median(np.linalg.norm(residual, axis=1)) < .35
    assert len(r["hop_survivors"]) == 1


def test_multihop_translation_accumulates_without_redetection():
    initial = texture()
    frames = [shift(initial, i * 2, i) for i in range(7)]
    r = bridge.chained_cycle_matches(frames)
    assert r["survivors"] > 35
    assert len(r["hop_survivors"]) == 6
    assert all(a >= b for a, b in zip(r["hop_survivors"], r["hop_survivors"][1:]))
    assert max(r["hop_survivors"]) <= r["detected"]
    residual = r["dst"] - r["src"] - [12, 6]
    assert np.median(np.linalg.norm(residual, axis=1)) < .75


def test_flat_texture_yields_abstention_and_no_features():
    frames = [np.zeros((128, 160), np.uint8) for _ in range(4)]
    r = bridge.chained_cycle_matches(frames)
    assert r["survivors"] == r["detected"] == 0
    assert r["hop_survivors"] == [0, 0, 0]
    assert bridge.spatial.crossfit_affine(r["src"], r["dst"], 160, 128)["status"] != "ok"


@pytest.mark.parametrize("bad", [
    [],
    [np.zeros((64, 64), np.uint8)],
    [np.zeros((64, 64, 3), np.uint8), np.zeros((64, 64, 3), np.uint8)],
    [np.zeros((64, 64), np.float32), np.zeros((64, 64), np.float32)],
    [np.zeros((64, 64), np.uint8), np.zeros((64, 65), np.uint8)],
])
def test_bad_input_fails_closed(bad):
    with pytest.raises(ValueError, match="EQUAL_UINT8_GRAY"):
        bridge.chained_cycle_matches(bad)


def test_corrupted_second_frame_never_increases_survivor_count():
    a = texture()
    b = shift(a, 2, 0)
    c = texture(seed=777)
    d = shift(c, 2, 0)
    r = bridge.chained_cycle_matches([a, b, c, d])
    assert r["hop_survivors"][0] >= r["hop_survivors"][1]
    assert r["hop_survivors"][1] >= r["hop_survivors"][2]
    assert r["survivors"] <= r["detected"]


def make_selection(tmp_path):
    rows = [dict(split="dev", video_id=f"dev{i}", label=f"action{i}",
                 shard_index=i, archive_member=f"action{i}.mp4") for i in range(6)]
    rows.append(dict(split="test", video_id="TEST_PRIVATE_ID_DO_NOT_READ",
                     label="TEST_PRIVATE_CLASS", shard_index=999,
                     archive_member="private.mp4"))
    selection = tmp_path / "selected.json"
    selection.write_text(json.dumps({"videos": rows}))
    shards = tmp_path / "shards"
    shards.mkdir()
    import hashlib
    manifest = {"shards": {}}
    for i in range(6):
        payload = f"dev-mock-{i}".encode()
        (shards / f"k700_val_{i:03d}.tar.gz").write_bytes(payload)
        manifest["shards"][f"{i:03d}"] = {
            "archive_member": f"action{i}.mp4",
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    (tmp_path / "download_manifest.json").write_text(json.dumps(manifest))
    return selection, shards


def mock_manifest_pin(selection):
    return bridge.audit.file_sha256(selection.with_name("download_manifest.json"))


def test_invalid_pin_rejects_before_any_media_access(tmp_path, monkeypatch):
    selection, shards = make_selection(tmp_path)
    monkeypatch.setattr(bridge, "analyze_dev_clip",
                        lambda *args: pytest.fail("Opened video before pin check"))
    with pytest.raises(RuntimeError, match="PIN_MISMATCH"):
        bridge.run(selection, shards, tmp_path / "o.json")


def test_dev_only_selection_and_write_once(tmp_path, monkeypatch):
    selection, shards = make_selection(tmp_path)
    called = []

    def fake_analyze(row, *args):
        called.append(row["video_id"])
        return {"dev_video_id": row["video_id"], "action_class": row["label"],
                "pairs": []}

    monkeypatch.setattr(bridge, "analyze_dev_clip", fake_analyze)
    pin = bridge.audit.selection_sha256(selection)
    output = tmp_path / "out.json"
    r = bridge.run(selection, shards, output, expected_selection_sha256=pin,
                   expected_download_manifest_sha256=mock_manifest_pin(selection))
    assert called == [f"dev{i}" for i in range(6)]
    assert "TEST_PRIVATE" not in json.dumps(r)
    assert "TEST_PRIVATE" not in output.read_text()
    assert not (shards / "k700_val_999.tar.gz").exists()
    with pytest.raises(RuntimeError, match="OUTPUT_EXISTS_DIFFERENT"):
        bridge.run(selection, shards, output, expected_selection_sha256=pin,
                   expected_download_manifest_sha256=mock_manifest_pin(selection), width=320)


def test_missing_dev_shard_fails_before_any_decoding(tmp_path, monkeypatch):
    selection, shards = make_selection(tmp_path)
    (shards / "k700_val_003.tar.gz").unlink()
    monkeypatch.setattr(bridge, "analyze_dev_clip",
                        lambda *args: pytest.fail("Opened video despite missing shard"))
    pin = bridge.audit.selection_sha256(selection)
    with pytest.raises(FileNotFoundError, match="DEV_SHARDS_MISSING"):
        bridge.run(selection, shards, tmp_path / "out.json",
                   expected_selection_sha256=pin,
                   expected_download_manifest_sha256=mock_manifest_pin(selection))


@pytest.mark.parametrize("bad", [{"anchor_count": 1}, {"width": 30}])
def test_invalid_protocol_before_media_access(tmp_path, bad):
    selection, shards = make_selection(tmp_path)
    with pytest.raises(ValueError, match="INVALID_PROTOCOL"):
        bridge.run(selection, shards, tmp_path / "out.json", **bad)


def test_paired_summary_is_explicit_and_coverage_not_hidden():
    direct = {"spatial": {"status": "ok", "crossfit_error_median_px": 1.0},
              "survivors": 25, "detected": 100, "match_cpu_ms": 1.0}
    valid_bridged = {"spatial": {"status": "ok", "crossfit_error_median_px": 0.5},
                     "survivors": 40, "detected": 100, "match_cpu_ms": 3.0}
    absent = {"spatial": {"status": "insufficient_valid_spatial_folds"},
              "survivors": 2, "detected": 100, "match_cpu_ms": 2.0}
    pairs = [{"methods": {"direct_30": direct, "bridge_3x10": valid_bridged,
                          "bridge_10x3": absent, "bridge_30x1": absent}}]
    s = bridge._summarize(pairs)
    assert s["bridge_3x10"]["paired_with_direct_valid"] == 1
    assert s["bridge_3x10"]["direct_holdout_error_px_paired"] == 1.0
    assert s["bridge_3x10"]["method_holdout_error_px_paired"] == 0.5
    assert s["bridge_10x3"]["valid_spatial_crossfit"] == 0
    assert s["bridge_10x3"]["paired_with_direct_valid"] == 0
    assert s["bridge_10x3"]["method_holdout_error_px_paired"] is None


def test_video_sampling_contains_every_intermediate_hop(tmp_path):
    path = tmp_path / "short.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             25, (160, 120))
    if not writer.isOpened():
        pytest.skip("Missing mp4v decoder")
    for i in range(100):
        writer.write(np.full((120, 160, 3), i, dtype=np.uint8))
    writer.release()
    fps, anchors, frames = bridge._read_requested_frames(
        path, width=160, anchors=4, extent=30
    )
    assert fps > 0 and len(anchors) == 4
    assert all(a + 30 < 100 for a in anchors)
    assert all(a + delta in frames for a in anchors for delta in range(31))
    assert len(frames) <= 4 * 31


def test_frame_sequence_uses_predeclared_30_frame_extent():
    initial = texture()
    frames = {i: shift(initial, i, 0) for i in range(31)}
    result = bridge.evaluate_frame_sequence(frames, 0)
    assert list(result["methods"]) == list(bridge.METHODS)
    assert len(result["methods"]["direct_30"]["hop_survivors"]) == 1
    assert len(result["methods"]["bridge_3x10"]["hop_survivors"]) == 3
    assert len(result["methods"]["bridge_10x3"]["hop_survivors"]) == 10
    assert len(result["methods"]["bridge_30x1"]["hop_survivors"]) == 30


def test_manifest_mutation_rejected_before_media_access(tmp_path, monkeypatch):
    selection, shards = make_selection(tmp_path)
    valid_pin = mock_manifest_pin(selection)
    manifest_path = selection.with_name("download_manifest.json")
    d = json.loads(manifest_path.read_text())
    d["shards"]["001"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(d))
    monkeypatch.setattr(bridge, "analyze_dev_clip",
                        lambda *args: pytest.fail("Media opened after manifest drift"))
    with pytest.raises(RuntimeError, match="DOWNLOAD_MANIFEST_PIN_MISMATCH"):
        bridge.run(selection, shards, tmp_path / "out.json",
                   expected_selection_sha256=bridge.audit.selection_sha256(selection),
                   expected_download_manifest_sha256=valid_pin)


def test_modified_dev_shard_rejected_before_archive_extraction(tmp_path):
    selection, shards = make_selection(tmp_path)
    pin = mock_manifest_pin(selection)
    row = json.loads(selection.read_text())["videos"][0]
    with pytest.raises(RuntimeError, match="SHARD_PIN_MISMATCH"):
        bridge.analyze_dev_clip(row, shards, 8, 512, "0" * 64)
