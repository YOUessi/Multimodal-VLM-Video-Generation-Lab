"""Unit and real-MP4 regression for affine camera shift and independent freeze."""
from __future__ import annotations
import inspect
import json
import pathlib
import sys
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import affine_domain_data as data
import affine_camera_tracker as geometry
import affine_motion_eval as evaluation
import domain_shift_motion_data as legacy


def config():
    return json.loads((ROOT / "configs/affine_camera_motion_v1.json").read_text())


def test_affine_camera_independent_seeds_and_predeclared_protocol():
    cfg = config()
    assert cfg["dev_seed"] != cfg["test_seed"]
    assert (cfg["dev_seed"], cfg["test_seed"]) != (722149, 828494)
    assert cfg["dataset_dir"].startswith("data/")
    assert cfg["output_dir"].startswith("outputs/")
    assert cfg["dataset_dir"] != cfg["output_dir"]
    assert cfg["predeclare_no_test_tuning"]
    assert cfg["not_natural_video"]
    assert cfg["dev_pairs"] == 40 and cfg["test_pairs"] == 100
    assert set(data.MODES) == {"camera_pan", "camera_rotation", "camera_zoom", "color_ambiguity"}
    assert set(evaluation.METHODS) == {"plain", "phase", "affine"}


def test_scene_nuisances_and_ambiguity_increase_tracking_difficulty():
    props = [data.scene(981239, mode, "up_right") for mode in data.MODES]
    assert props[0]["pan_velocity"] != [0, 0]
    assert props[1]["rotation_per_frame"] != 0
    assert props[2]["zoom_per_frame"] != 0
    assert props[3]["distractor_color"] == props[3]["target_color"]
    assert props[3]["distractor_shape"] != props[3]["target_shape"]
    assert props[3]["variant"] == "occlusion"
    for p in props:
        assert p["camera_velocity"] == [0, 0]


def test_affine_world_to_screen_and_inverse_correction_signs():
    p = data.scene(131, "camera_pan", "stationary")
    p["pan_velocity"] = [3, -2]
    transform = data.affine_for_frame(p, 9)
    a = [110.0, 115.0]
    b = transform @ np.array([a[0], a[1], 1.])
    corrected = geometry.corrected_delta([a, b.tolist()], transform)
    assert np.allclose(corrected, [0, 0], atol=1e-3)
    # Without camera correction stationary in world is moving on screen.
    assert np.linalg.norm(b - np.array(a)) > 20


def test_feature_geometry_uses_only_pixels_not_gold_parameters():
    src = inspect.getsource(geometry.estimate_background_affine)
    for forbidden in ("gold_direction", "camera_params", "scene_seed", "world_target_velocity"):
        assert forbidden not in src
    assert list(inspect.signature(geometry.tracker).parameters) == ["path", "question", "cfg", "method"]


def test_camera_rotation_real_mp4_recovers_background_transform(tmp_path):
    p = data.scene(13993, "camera_rotation", "stationary")
    p["rotation_per_frame"] = 1.7
    p["pan_velocity"] = [1, -1]
    frames = data.render(p, 10, 256, 256)
    path = tmp_path / "rotation.mp4"
    legacy.write_video(path, frames, 6)
    result = geometry.estimate_background_affine(path, config())
    assert result["accepted"], result
    actual = np.array(result["matrix"])
    expected = data.affine_for_frame(p, 9)
    at = np.array([80.0, 80.0, 1.0])
    assert np.linalg.norm(actual @ at - expected @ at) < 6.0, result


def test_camera_zoom_real_mp4_recovers_background_transform(tmp_path):
    p = data.scene(63817, "camera_zoom", "stationary")
    p["zoom_per_frame"] = -0.016
    p["pan_velocity"] = [-1, 1]
    frames = data.render(p, 10, 256, 256)
    path = tmp_path / "zoom.mp4"
    legacy.write_video(path, frames, 6)
    result = geometry.estimate_background_affine(path, config())
    assert result["accepted"], result
    at = np.array([67.0, 152.0, 1.0])
    assert np.linalg.norm(np.asarray(result["matrix"]) @ at
                          - data.affine_for_frame(p, 9) @ at) < 6.0


def test_pair_clustered_bootstrap_not_independent_reverse_clips():
    original = {"a_f": False, "a_r": False, "b_f": False, "b_r": False}
    improved = dict(original)
    improved["a_f"] = improved["a_r"] = True
    groups = {"a_f": "a", "a_r": "a", "b_f": "b", "b_r": "b"}
    stat = evaluation.base.bootstrap_scene(original, improved, groups, seed=13)
    assert stat["delta_pp"] == 50
    assert stat["unit"].startswith("independent")


def test_small_real_mp4_independent_freeze_manifest_and_immutability(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    cfg = config()
    cfg.update({"dataset_dir": "data/small_fixture", "dev_pairs": 20, "test_pairs": 20})
    doc = data.build(cfg)
    assert doc["total_video_clips"] == 80
    assert doc["dev_test_content_overlap"] == 0
    assert len({r["content_sha256"] for r in doc["rows"]}) == 80
    assert len({s["scene_seed"] for s in doc["source_pairs"]}) == 40
    assert data.build(cfg)["source_sha256"] == doc["source_sha256"]
    cfg["test_seed"] += 1
    with pytest.raises(RuntimeError, match="CONFIG_DRIFT"):
        data.build(cfg)
    assert not (tmp_path / "outputs" / "affine_camera_motion_v1" / "manifest.json").exists()


def test_abstain_when_registration_not_acceptable(tmp_path, monkeypatch):
    # A synthetic unit stub proves fallback cannot convert uncertain affine
    # geometry into a confident incorrect VLM override.
    fake = {
        "target_points": [[0,0],[4,4],[10,10]], "target_color": "red",
        "screen_delta_xy": [10, 10], "delta_xy": [10,10],
        "missing": 0, "reason": "OK", "direction": "down_right",
        "registration": None
    }
    monkeypatch.setattr(geometry, "frozen_tracker", lambda *args, **kwargs: dict(fake))
    monkeypatch.setattr(geometry, "estimate_background_affine", lambda *args, **kwargs:
                        {"accepted": False, "reason": "WEAK_RANSAC_INLIERS"})
    row = geometry.tracker(tmp_path / "unknown.mp4", "red circle", config(), "affine")
    assert row["direction"] is None and row["delta_xy"] is None
    assert geometry.gated_label(row, config())[0] is None


def test_vlm_report_markdown_has_six_separate_columns():
    # Regression for a presentation-only frozen Batch33 results table typo.
    result = {"base": 4, "hybrids": {
        "plain": {"correct": 7}, "phase": {"correct": 8}, "affine": {"correct": 9}}}
    row = evaluation.vlm_table_row("2b", 5, result, 20)
    assert row == "| 2b | 5 | 4/20 | 7/20 | 8/20 | 9/20 |"
    assert len(row.split("|")) == 8
