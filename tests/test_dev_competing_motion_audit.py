"""Regression contract for DEV-only two-hypothesis affine ambiguity research."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import dev_competing_motion_audit as layer
import dev_counterfactual_motion_layers as control


def sample_points(n=120, seed=19):
    rng = np.random.default_rng(seed)
    xy = np.column_stack((rng.uniform(20, 492, n),
                          rng.uniform(20, 268, n))).astype(np.float32)
    return xy


def warp(points, shift):
    return points + np.asarray(shift, dtype=np.float32)[None, :]


def test_synthetic_two_coherent_layers_are_detected():
    points = sample_points()
    dest = np.empty_like(points)
    dest[:85] = warp(points[:85], (-11, 9))
    dest[85:] = warp(points[85:], (6, -4))
    primary = np.array([[1, 0, -11], [0, 1, 9]], dtype=np.float64)
    result = layer.competing_motion_evidence(points, dest, 288, 512, primary)
    assert result["detected_competing_layer"]
    assert result["alternate_inliers"] >= 30
    assert result["alternate_quadrants"] >= 2
    assert result["median_model_disagreement_px"] > 10
    assert result["status"] == "competing_models"


def test_single_motion_model_has_no_independent_competitor():
    points = sample_points()
    result = layer.competing_motion_evidence(
        points, warp(points, (6, -4)), 288, 512,
        np.array([[1, 0, 6], [0, 1, -4]], dtype=np.float32))
    assert result["primary_valid"]
    assert not result["detected_competing_layer"]
    assert result["status"] == "insufficient_nonprimary_support"


def test_fit_primary_when_absent_and_detects_multiple_layers():
    points = sample_points()
    dest = np.empty_like(points)
    dest[:85] = warp(points[:85], (-11, 9))
    dest[85:] = warp(points[85:], (6, -4))
    result = layer.competing_motion_evidence(points, dest, 288, 512)
    assert result["primary_valid"]
    assert result["detected_competing_layer"]


def test_insufficient_matches_explicitly_abstain():
    points = sample_points(n=8)
    result = layer.competing_motion_evidence(points, warp(points, (2, 1)), 288, 512)
    assert result["status"] == "insufficient_matches"
    assert not result["primary_valid"]
    assert not result["detected_competing_layer"]


@pytest.mark.parametrize("bad_source,bad_dest", [
    (np.zeros((4, 3)), np.zeros((4, 3))),
    (np.zeros((4, 2)), np.zeros((5, 2))),
    (np.array([[float("nan"), 4]], np.float32), np.array([[1, 4]], np.float32)),
    (np.array([[-5, 2]], np.float32), np.array([[10, 2]], np.float32)),
])
def test_rejects_invalid_correspondences(bad_source, bad_dest):
    with pytest.raises(ValueError, match="COMPETING_MOTION"):
        layer.competing_motion_evidence(bad_source, bad_dest, 288, 512)


def test_secondary_evidence_single_quadrant_does_not_claim_global_ambiguity():
    all_points = sample_points(110)
    # Inject a coherent alternate layer localized to top-left only.
    selected = np.where((all_points[:, 0] < 256) & (all_points[:, 1] < 144))[0]
    assert len(selected) >= 12
    dest = warp(all_points, (6, -4))
    dest[selected] = warp(all_points[selected], (-11, 9))
    primary = np.array([[1, 0, 6], [0, 1, -4]], np.float32)
    result = layer.competing_motion_evidence(all_points, dest, 288, 512, primary)
    assert result["alternate_inliers"] >= 12
    assert result["alternate_quadrants"] == 1
    assert not result["detected_competing_layer"]


def test_invalid_primary_fit_does_not_pass_safety_claim():
    points = sample_points()
    bad = np.array([[float("nan"), 0., 0.], [0., 1., 0.]])
    r = layer.competing_motion_evidence(points, warp(points, (1, 2)), 288, 512, bad)
    assert r["status"] == "no_valid_primary_fit"
    assert not r["detected_competing_layer"]


@pytest.mark.parametrize("condition", control.CONDITIONS)
def test_realistic_synthetic_pair_has_oracle_separate_from_decision(condition):
    rng = np.random.default_rng(991)
    frame = rng.integers(0, 256, (288, 512), dtype=np.uint8)
    source, dest, truth, grid = control.synthesize(frame, condition, seed=128)
    before = frame.copy()
    base, matrix = control.estimate_pair(source, dest)
    tracks = layer.bridge.chained_cycle_matches([source, dest])
    found = layer.competing_motion_evidence(
        tracks["src"], tracks["dst"], 288, 512, matrix)
    candidate_accept = base["gate_accept"] and not found["detected_competing_layer"]
    scored = control.score_known_affine(matrix, truth, grid)
    assert isinstance(candidate_accept, bool)
    assert int(candidate_accept) <= int(base["gate_accept"])
    assert scored["synthetic_bg_wrong"] in (None, True, False)
    assert np.array_equal(frame, before)


def test_summary_separates_abstention_from_precision():
    rows = [
        dict(fit_valid=True, detected_competing_layer=False, old_gate_accept=True,
             ambiguity_gate_accept=True, synthetic_bg_wrong=False, extra_cpu_ms=2.),
        dict(fit_valid=True, detected_competing_layer=True, old_gate_accept=True,
             ambiguity_gate_accept=False, synthetic_bg_wrong=True, extra_cpu_ms=6.),
        dict(fit_valid=False, detected_competing_layer=False, old_gate_accept=False,
             ambiguity_gate_accept=False, synthetic_bg_wrong=None, extra_cpu_ms=1.),
    ]
    s = layer.summarize(rows)
    assert s["old_gate_accepted"] == 2
    assert s["new_gate_accepted"] == 1
    assert s["old_wrong_accepted"] == 1
    assert s["new_wrong_accepted"] == 0
    assert s["new_gate_coverage"] == pytest.approx(1 / 3)
    assert s["median_extra_cpu_ms"] == pytest.approx(2)


def test_zero_acceptance_conditional_error_is_undefined():
    r = layer.summarize([dict(
        fit_valid=False, detected_competing_layer=False, old_gate_accept=False,
        ambiguity_gate_accept=False, synthetic_bg_wrong=None, extra_cpu_ms=1.)])
    assert r["new_wrong_fraction_among_accepted"] is None
    assert r["new_gate_coverage"] == 0.0


def test_source_untrusted_before_any_dev_or_test_clip_access(tmp_path, monkeypatch):
    bad_selection = tmp_path / "not-approved.json"
    bad_selection.write_text('{"videos": []}')
    monkeypatch.setattr(control, "_clip_frames",
                        lambda *args: pytest.fail("MEDIA_READ_BEFORE_PIN_CHECK"))
    with pytest.raises(RuntimeError, match="SELECTION_PIN_MISMATCH"):
        layer.run(bad_selection, tmp_path / "shards", tmp_path / "report.json")


def test_oracle_is_scoring_only_not_a_gate_input():
    source = inspect.getsource(layer.run)
    assert source.index("competing_motion_evidence(") < source.index(
        "control.score_known_affine(")
    assert "pins.dev_rows" in source
    assert "bridge.EXPECTED_SELECTION_SHA256" in source
    assert "bridge.EXPECTED_DOWNLOAD_MANIFEST_SHA256" in source
    assert "test_videos" not in source.lower()
