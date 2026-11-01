"""Regression and prospective guarantees for world parallax/identity study."""
from __future__ import annotations

import inspect
import json
import pathlib
import sys

import cv2
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import parallax_identity_data as generator
import parallax_identity_eval as study
import shape_aware_tracker as identity
import domain_shift_motion_data as legacy


def cfg():
    return json.loads((ROOT / "configs/parallax_identity_v1.json").read_text())


def test_independent_frozen_protocol_and_private_paths():
    c = cfg()
    assert c["predeclare_no_test_tuning"] and c["not_natural_video"]
    assert c["study_prior_batch33_scored_no_tuning"]
    assert c["dataset_dir"] == "data/parallax_identity_v1"
    assert c["output_dir"] == "outputs/parallax_identity_v1"
    assert c["dev_pairs"] == 40 and c["test_pairs"] == 100
    assert c["dev_seed"] != c["test_seed"]
    assert c["identity_min_background_inlier_fraction"] >= .70
    assert set(study.METHODS) == {"plain", "affine", "shape_affine"}
    assert len(generator.MODES) * len(generator.DIRECTIONS) == 20


def test_question_shape_requires_unique_semantic_identifier():
    assert identity.question_shape("Which way does the RED triangle move?") == "triangle"
    with pytest.raises(ValueError):
        identity.question_shape("Which way is the red ball going?")
    with pytest.raises(ValueError):
        identity.question_shape("Does the red circle beat the blue square?")


@pytest.mark.parametrize("shape", ["circle", "square", "triangle"])
def test_simple_contours_identify_target_shape(shape):
    arr = np.zeros((120, 120), np.uint8)
    contour_colors = {
        "circle": lambda: cv2.circle(arr, (60, 60), 18, 255, -1),
        "square": lambda: cv2.rectangle(arr, (42, 42), (78, 78), 255, -1),
        "triangle": lambda: cv2.fillPoly(
            arr, [np.array([[60, 39], [38, 82], [82, 82]], np.int32)], 255),
    }
    contour_colors[shape]()
    found, _ = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(found) == 1
    assert identity.contour_shape(found[0], cfg()["identity_polygon_epsilon"]) == shape


def test_same_hue_distinct_shape_semantics_can_select_correct_target():
    c = cfg()
    img = np.full((256, 256, 3), 168, np.uint8)
    legacy.draw_object(img, 85, 125, "red", "square", 13)
    legacy.draw_object(img, 172, 125, "red", "circle", 13)
    selected = identity.semantic_candidates(img, "red", "circle", c)
    assert len(selected) == 1
    assert abs(selected[0]["xy"][0] - 172) <= 1


def test_indistinguishable_same_color_same_shape_does_not_guess(tmp_path):
    c = cfg()
    frames = []
    for t in range(10):
        arr = np.full((256, 256, 3), 166, np.uint8)
        legacy.draw_object(arr, 71 + t, 95, "blue", "circle", 13)
        legacy.draw_object(arr, 172 - t, 164, "blue", "circle", 13)
        frames.append(arr)
    file = tmp_path / "two_identical.mp4"
    legacy.write_video(file, frames, 6)
    ans = identity.video_shape_trace(file, "Where does the blue circle move?", c)
    assert ans["delta_xy"] is None
    assert "SAME_APPEARANCE_TARGET_IDENTITY_AMBIGUOUS" in ans["reason"]


def test_missing_middle_frame_is_abstention(tmp_path):
    c = cfg()
    frames = []
    for t in range(10):
        arr = np.full((256, 256, 3), 165, np.uint8)
        if not 4 <= t <= 6:
            legacy.draw_object(arr, 115 + 2*t, 121, "green", "triangle", 14)
        frames.append(arr)
    file = tmp_path / "occluded.mp4"
    legacy.write_video(file, frames, 6)
    ans = identity.video_shape_trace(file, "Which way does the green triangle move?", c)
    assert ans["delta_xy"] is None
    assert ans["missing"] > 0


def test_parallax_renderer_has_multiple_depth_motion_planes():
    p = generator.world_scene(1479, "parallax_dual_depth", "stationary")
    assert p["far_depth_factor"] < p["object_depth_factor"] < p["near_depth_factor"]
    images = generator.frames_for_scene(p, 10, 256, 256)
    assert len(images) == 10
    assert images[0].shape == (256, 256, 3)
    assert not np.array_equal(images[0], images[-1])
    assert np.var(images[0]) > 0


def test_generation_seeds_independent_and_deterministic():
    a = generator.world_scene(98765, "same_shape_ambiguity", "down_left")
    b = generator.world_scene(98765, "same_shape_ambiguity", "down_left")
    c = generator.world_scene(98766, "same_shape_ambiguity", "down_left")
    assert a == b and a != c
    assert a["target_color"] == a["distractor_color"]
    assert a["target_shape"] == a["distractor_shape"]
    assert generator.OPPOSITE["down_left"] == "up_right"


def test_shape_tracker_never_imports_private_generator_or_reads_truth():
    source = inspect.getsource(identity)
    assert "parallax_identity_data" not in source
    assert "gold_direction" not in source
    assert "gold_letter" not in source
    assert "manifest.json" not in source


def test_full_benchmark_model_scoring_requires_complete_frozen_outputs(tmp_path, monkeypatch):
    c = cfg()
    fake_root = tmp_path
    (fake_root / c["dataset_dir"]).mkdir(parents=True)
    (fake_root / c["output_dir"]).mkdir(parents=True)
    monkeypatch.setattr(study, "ROOT", fake_root)
    # No manifest or files: report MUST fail, not fabricate a score.
    with pytest.raises((FileNotFoundError, RuntimeError)):
        study.report(c)


def test_new_benchmark_uses_true_forward_reverse_pairs_and_freeze(tmp_path, monkeypatch):
    c = cfg()
    # Use balanced miniature splits and a temp source: this test never touches
    # the prospective TEST media or score of the real future experiment.
    c["dev_pairs"], c["test_pairs"] = 20, 20
    c["dataset_dir"] = "data/prospective_contract_temp"
    monkeypatch.setattr(generator, "ROOT", tmp_path)
    first = generator.build(c)
    assert len(first["rows"]) == 80
    assert len(first["pairs"]) == 40
    assert not ({r["content_sha256"] for r in first["rows"] if r["split"] == "dev"} &
                {r["content_sha256"] for r in first["rows"] if r["split"] == "test"})
    assert first["source_sha256"] == generator.build(c)["source_sha256"]
    groups = {}
    for r in first["rows"]:
        groups.setdefault(r["pair_id"], []).append(r)
    assert all(len(g) == 2 and
               generator.OPPOSITE[g[0]["gold_direction"]] == g[1]["gold_direction"]
               for g in groups.values())
    c["test_seed"] += 1
    with pytest.raises(RuntimeError, match="CONFIG_DRIFT"):
        generator.build(c)
