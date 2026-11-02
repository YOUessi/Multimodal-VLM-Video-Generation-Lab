"""Predeclared DEV-only temporal sensitivity and forward/backward KLT controls."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import dev_temporal_affine_stability as temporal


def texture(seed=30, h=240, w=320):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def test_cycle_filter_recovers_pure_translation_with_many_valid_matches():
    src = texture()
    dst = cv2.warpAffine(src, np.float32([[1, 0, 4], [0, 1, -3]]), (320, 240))
    row = temporal.matches_with_cycle_filter(src, dst)
    assert row["detected"] > 100
    assert row["cycle_valid"] > 90
    assert row["cycle_valid"] <= row["forward_valid"] <= row["detected"]
    assert np.median(np.linalg.norm(row["dst_fb"] - row["src_fb"] - [4, -3], axis=1)) < 0.3


def test_nontexture_returns_no_false_global_support():
    flat = np.zeros((240, 320), dtype=np.uint8)
    r = temporal.matches_with_cycle_filter(flat, flat)
    assert r["forward_valid"] == r["cycle_valid"] == 0
    assert temporal.spatial.crossfit_affine(r["src_fb"], r["dst_fb"], 320, 240)["status"] != "ok"


@pytest.mark.parametrize("bad", [
    np.zeros((64, 64, 3), dtype=np.uint8),
    np.zeros((64, 64), dtype=np.float32),
])
def test_invalid_gray_format_fails_closed(bad):
    with pytest.raises(ValueError, match="UINT8_GRAY"):
        temporal.matches_with_cycle_filter(bad, bad)


def test_different_shapes_fail_closed():
    with pytest.raises(ValueError, match="UINT8_GRAY"):
        temporal.matches_with_cycle_filter(
            np.zeros((64, 64), np.uint8), np.zeros((64, 63), np.uint8))


def test_cycle_filter_rejects_nonreversible_appearance_in_occluded_patch():
    a = texture(seed=55)
    b = a.copy()
    b[50:200, 80:240] = texture(seed=77, h=150, w=160)
    m = temporal.matches_with_cycle_filter(a, b)
    assert m["cycle_valid"] <= m["forward_valid"]
    assert m["forward_valid"] > 50
    assert m["cycle_valid"] < m["forward_valid"]  # occlusion causes a noninvertible match


def make_selection(tmp_path):
    rows = [dict(split="dev", video_id=f"dev{i}", label=f"a{i}", shard_index=i,
                 archive_member=f"a{i}.mp4") for i in range(6)]
    rows += [dict(split="test", video_id="test_secret", label="private",
                  shard_index=999, archive_member="test_secret.mp4")]
    p = tmp_path / "selected.json"
    p.write_text(json.dumps({"videos": rows}))
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    for i in range(6):
        (shard_dir / f"k700_val_{i:03d}.tar.gz").write_bytes(b"mock-not-opened")
    return p, shard_dir


def test_hard_selection_pin_rejects_before_accessing_video(tmp_path, monkeypatch):
    selection, shards = make_selection(tmp_path)
    monkeypatch.setattr(temporal, "analyze_clip",
                        lambda *a: pytest.fail("No video access on pin mismatch"))
    with pytest.raises(RuntimeError, match="PIN_MISMATCH"):
        temporal.run(selection, shards, tmp_path / "out.json")


def test_dev_only_no_test_open_and_report_write_once(tmp_path, monkeypatch):
    selection, shards = make_selection(tmp_path)
    pin = temporal.audit.selection_sha256(selection)
    observed = []

    def fake_clip(row, *args):
        observed.append(row["video_id"])
        return {"dev_video_id": row["video_id"], "action_class": row["label"], "strides": []}

    monkeypatch.setattr(temporal, "analyze_clip", fake_clip)
    out = tmp_path / "out.json"
    report = temporal.run(selection, shards, out, expected_selection_sha256=pin)
    assert observed == [f"dev{i}" for i in range(6)]
    assert "test_secret" not in json.dumps(report)
    assert not (shards / "k700_val_999.tar.gz").exists()
    assert report["scope"].startswith("DEV_ONLY")
    with pytest.raises(RuntimeError, match="EXISTS_DIFFERENT"):
        temporal.run(selection, shards, out, expected_selection_sha256=pin,
                     strides=(1, 5))


def test_dev_media_missing_is_rejected_before_any_decode(tmp_path, monkeypatch):
    selection, shards = make_selection(tmp_path)
    (shards / "k700_val_004.tar.gz").unlink()
    monkeypatch.setattr(temporal, "analyze_clip",
                        lambda *a: pytest.fail("Decode before input validation"))
    with pytest.raises(FileNotFoundError, match="DEV_SHARDS_MISSING"):
        temporal.run(selection, shards, tmp_path / "o.json",
                     expected_selection_sha256=temporal.audit.selection_sha256(selection))


@pytest.mark.parametrize("kwargs", [
    {"strides": ()}, {"strides": (1, 1)}, {"strides": (0, 3)},
    {"anchor_count": 1}, {"width": 40},
])
def test_protocol_invalid_before_media(tmp_path, kwargs):
    selection, shards = make_selection(tmp_path)
    with pytest.raises(ValueError, match="INVALID_PROTOCOL"):
        temporal.run(selection, shards, tmp_path / "o.json", **kwargs)


def test_multiple_strides_share_anchor_frames_without_lookahead(tmp_path):
    video = tmp_path / "control.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (160, 120))
    if not writer.isOpened():
        pytest.skip("OpenCV mp4v not available")
    for i in range(80):
        frame = np.full((120, 160, 3), i * 2, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    fps, anchors, frames = temporal._sample_frames(video, (1, 3, 10, 30), 8, 160)
    assert len(anchors) == 8
    assert max(anchors) + 30 < 80
    assert len(frames) <= 8 * 5
    assert fps > 0
    assert all(frame.dtype == np.uint8 and frame.ndim == 2 for frame in frames.values())
