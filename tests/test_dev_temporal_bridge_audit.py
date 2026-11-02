"""Frozen Batch47D evidence assertions and strict DEV-only runner plan."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import dev_temporal_bridge_affine as bridge
import dev_temporal_bridge_audit as audit


def fixture_report(tmp_path, monkeypatch):
    rows = [{"split": "dev", "video_id": f"dev{i}", "label": f"action{i}",
             "shard_index": i, "archive_member": f"m{i}.mp4"} for i in range(6)]
    rows += [{"split": "test", "video_id": "secret", "label": "secret",
              "shard_index": 999, "archive_member": "private.mp4"}]
    sel = tmp_path / "selection.json"
    sel.write_text(json.dumps({"videos": rows}))
    pin = audit.common.selection_sha256(sel)
    monkeypatch.setattr(bridge, "EXPECTED_SELECTION_SHA256", pin)
    manifest = tmp_path / "download_manifest.json"
    manifest.write_text(json.dumps({"shards": {f"{i:03d}": {
        "archive_member": f"m{i}.mp4", "sha256": f"{i:064x}"
    } for i in range(6)}}))
    manifest_pin = audit.common.file_sha256(manifest)
    monkeypatch.setattr(bridge, "EXPECTED_DOWNLOAD_MANIFEST_SHA256", manifest_pin)

    def one(method):
        step = bridge.METHODS[method]
        return {"detected": 30, "survivors": 24, "hop_survivors": [24] * (30 // step),
                "spatial": {"status": "insufficient_valid_spatial_folds",
                            "match_count": 24},
                "match_cpu_ms": 1.5, "crossfit_cpu_ms": 0.5}

    pairs = [{"anchor_frame": i, "methods": {n: one(n) for n in bridge.METHODS}}
             for i in range(8)]
    scenes = [{"dev_video_id": f"dev{i}", "action_class": f"action{i}",
               "anchors": list(range(8)), "pairs": pairs} for i in range(6)]
    d = {"scope": "DEV_ONLY_NO_TEST_MEDIA_NO_PRIVATE_ORACLE",
         "selection_sha256": pin,
         "download_manifest_sha256": manifest_pin,
         "source_sha256": audit.common.file_sha256(Path(bridge.__file__)),
         "settings": {"extent_frames": 30, "anchor_count": 8,
                      "methods": bridge.METHODS},
         "scenes": scenes, "overall": bridge._summarize(pairs * 6)}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(d))
    return path, sel, d


def test_valid_frozen_dev_report_passes(tmp_path, monkeypatch):
    path, sel, _ = fixture_report(tmp_path, monkeypatch)
    assert audit.verify(path, sel)["pairs"] == 48


def test_report_source_mutation_rejected(tmp_path, monkeypatch):
    path, sel, d = fixture_report(tmp_path, monkeypatch)
    d["source_sha256"] = "0" * 64
    path.write_text(json.dumps(d))
    with pytest.raises(RuntimeError, match="SOURCE_MISMATCH"):
        audit.verify(path, sel)


def test_test_id_in_dev_report_rejected(tmp_path, monkeypatch):
    path, sel, d = fixture_report(tmp_path, monkeypatch)
    d["scenes"][0]["dev_video_id"] = "secret"
    path.write_text(json.dumps(d))
    with pytest.raises(RuntimeError, match="DEV_VIDEO_ORDER_OR_ID"):
        audit.verify(path, sel)


def test_reacquisition_after_abstention_is_rejected(tmp_path, monkeypatch):
    path, sel, d = fixture_report(tmp_path, monkeypatch)
    d["scenes"][0]["pairs"][0]["methods"]["bridge_10x3"]["hop_survivors"] = [
        25, 24, 23, 24, 23, 22, 22, 22, 22, 22
    ]
    path.write_text(json.dumps(d))
    with pytest.raises(RuntimeError, match="SURVIVOR_INVARIANT"):
        audit.verify(path, sel)


def test_private_oracle_or_test_scoring_disallowed_in_frozen_plan():
    p = ROOT / "automation/plans/batch47d_temporal_bridge_dev_only_2026-11-02.json"
    plan = json.loads(p.read_text())
    assert plan["enabled"] and plan["max_stages_per_tick"] == 2
    assert len(plan["stages"]) == 4
    assert all(s["max_attempts"] == 1 and not s.get("gpu", False)
               for s in plan["stages"])
    payload = json.dumps(plan["stages"])
    assert "oracle-private" not in payload
    assert "predict-test" not in payload and "score-test" not in payload
    assert "test_results.json" not in payload
    assert "dev_temporal_bridge_affine.py" in payload
    assert "dev_temporal_bridge_audit.py" in payload
    assert "batch47d_frozen_bridge_dev.json" in payload


def test_legacy_plan_left_intact_and_wrapper_uses_new_plan():
    legacy = json.loads((ROOT / "automation/plan.json").read_text())
    assert any(s["id"] == "R00_SELECT_AND_DOWNLOAD_12_FRESH_CLASS_SHARDS"
               for s in legacy["stages"])
    wrapper = (ROOT / "automation/runner_batch47d.py").read_text()
    assert "batch47d_temporal_bridge_dev_only_2026-11-02.json" in wrapper
    assert "runner.PLAN" in wrapper


def test_download_manifest_report_pin_mismatch_rejected(tmp_path, monkeypatch):
    path, sel, d = fixture_report(tmp_path, monkeypatch)
    d["download_manifest_sha256"] = "0" * 64
    path.write_text(json.dumps(d))
    with pytest.raises(RuntimeError, match="DOWNLOAD_MANIFEST_MISMATCH"):
        audit.verify(path, sel)
