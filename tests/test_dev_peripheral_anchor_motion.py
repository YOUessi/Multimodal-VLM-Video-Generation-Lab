"""Tests for exploratory DEV-only boundary-prior motion diagnostic."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import dev_peripheral_anchor_motion as ring
import dev_counterfactual_motion_layers as control

H, W = 288, 512
BG = np.array([[1., 0., 6.], [0., 1., -4.]], np.float32)
FG = np.array([[1., 0., -11.], [0., 1., 9.]], np.float32)


def grid():
    x, y = np.meshgrid(np.arange(12, 510, 20), np.arange(9, 285, 15))
    return np.stack([x.ravel(), y.ravel()], axis=-1).astype(np.float32)


def move(x, matrix):
    return (x @ matrix[:, :2].T + matrix[:, 2]).astype(np.float32)


def test_fully_supported_true_global_translation_agrees():
    src = grid()
    r = ring.peripheral_motion_evidence(src, move(src, BG), H, W, BG)
    assert r["status"] == "periphery_agrees_with_primary"
    assert r["agrees_with_primary"]
    assert r["peripheral_supported"]
    assert r["ring_quadrants"] == 4
    assert r["ring_disagreement_px"] < 0.05
    assert r["background_identified"] is False


def test_center_dominated_foreground_is_refuted_by_edge_model():
    src = grid()
    dest = move(src, FG)
    peripheral = ring._peripheral_mask(src, H, W)
    assert sum(peripheral) >= 16
    dest[peripheral] = move(src[peripheral], BG)
    r = ring.peripheral_motion_evidence(src, dest, H, W, FG)
    assert r["peripheral_supported"]
    assert r["status"] == "periphery_disagrees_with_primary"
    assert r["ring_disagreement_px"] > 10
    assert not r["agrees_with_primary"]


def test_full_screen_foreground_shows_identification_limit():
    src = grid()
    r = ring.peripheral_motion_evidence(src, move(src, FG), H, W, FG)
    assert r["agrees_with_primary"]
    assert not r["background_identified"]
    # Identical observed correspondences coexist with arbitrary latent
    # background truth BG; no model of those pixels can infer BG.
    assert np.median(np.linalg.norm(move(src, FG) - move(src, BG), axis=1)) > 10


def test_insufficient_matches_abstains():
    src = grid()[:7]
    r = ring.peripheral_motion_evidence(src, move(src, BG), H, W, BG)
    assert r["status"] == "insufficient_correspondences"
    assert not r["agrees_with_primary"]


def test_center_only_matches_abstain_with_explicit_reason():
    src = grid()
    src = src[~ring._peripheral_mask(src, H, W)]
    r = ring.peripheral_motion_evidence(src, move(src, BG), H, W, BG)
    assert r["ring_matches"] == 0
    assert r["status"] == "insufficient_peripheral_matches"


def test_single_quadrant_periphery_is_not_global():
    src = grid()
    src = src[ring._peripheral_mask(src, H, W)]
    src = src[(src[:, 0] < 256) & (src[:, 1] < 144)]
    assert len(src) >= 16
    r = ring.peripheral_motion_evidence(src, move(src, BG), H, W, BG)
    assert r["status"] == "periphery_not_spatially_distributed"
    assert not r["agrees_with_primary"]


def test_primary_none_or_nonfinite_rejected():
    src = grid()
    for primary in [None, np.array([[np.nan, 0, 0], [0, 1, 0]])]:
        r = ring.peripheral_motion_evidence(src, move(src, BG), H, W, primary)
        assert r["status"] == "invalid_primary"
        assert not r["agrees_with_primary"]


@pytest.mark.parametrize("bad_source,bad_dest", [
    (np.zeros((12, 3)), np.zeros((12, 3))),
    (np.zeros((12, 2)), np.zeros((13, 2))),
    (np.array([[float("inf"), 2.]]), np.array([[1., 2.]])),
    (np.array([[-1., 2.]]), np.array([[1., 2.]])),
])
def test_invalid_correspondences_rejected(bad_source, bad_dest):
    with pytest.raises(ValueError):
        ring.peripheral_motion_evidence(bad_source, bad_dest, H, W, BG)


def test_hidden_background_synthesis_is_repeatable_and_full_screen():
    frame = np.full((H, W), 123, dtype=np.uint8)
    a, b, truth, grid_pts = ring.synthesize_hidden_background(frame, 444)
    a2, b2, truth2, grid2 = ring.synthesize_hidden_background(frame, 444)
    assert a.shape == frame.shape and b.shape == frame.shape
    assert np.array_equal(a, a2) and np.array_equal(b, b2)
    assert np.array_equal(truth, truth2) and np.array_equal(grid_pts, grid2)
    assert (truth[:, 2] == [control.DX, control.DY]).all()
    assert not np.array_equal(frame, a)
    assert np.array_equal(frame, np.full((H, W), 123, dtype=np.uint8))


@pytest.mark.parametrize("condition", control.CONDITIONS)
def test_exploratory_condition_decision_before_scoring(condition):
    rng = np.random.default_rng(32)
    frame = rng.integers(0, 256, (H, W), dtype=np.uint8)
    source, dest, truth, truth_grid = control.synthesize(frame, condition, 180)
    old, primary = control.estimate_pair(source, dest)
    tracks = ring.bridge.chained_cycle_matches([source, dest])
    evidence = ring.peripheral_motion_evidence(
        tracks["src"], tracks["dst"], H, W, primary
    )
    newer = bool(old["gate_accept"] and evidence["agrees_with_primary"])
    score = control.score_known_affine(primary, truth, truth_grid)
    assert int(newer) <= int(old["gate_accept"])
    assert score["synthetic_bg_wrong"] in (None, True, False)
    assert evidence["background_identified"] is False


def test_missing_pin_prevents_media_access(tmp_path, monkeypatch):
    bad = tmp_path / "selection.json"
    bad.write_text('{"videos":[]}')
    monkeypatch.setattr(control, "_clip_frames",
                        lambda *args: pytest.fail("MEDIA_READ_BEFORE_PIN"))
    with pytest.raises(RuntimeError, match="SELECTION_PIN_MISMATCH"):
        ring.run(bad, tmp_path / "shards", tmp_path / "report.json")


def test_oracle_scoring_after_gate_and_dev_filter():
    source = inspect.getsource(ring.run)
    assert source.index("peripheral_motion_evidence(") < source.index(
        "control.score_known_affine("
    )
    assert "audit.dev_rows" in source
    assert "bridge.EXPECTED_SELECTION_SHA256" in source
    assert "bridge.EXPECTED_DOWNLOAD_MANIFEST_SHA256" in source
    assert "test_videos" not in source.lower()


def test_abstention_not_counted_as_correct_prediction():
    rows = [
        dict(old_accept=True, new_accept=True, synthetic_bg_wrong=False,
             background_visible=True, peripheral_supported=True,
             status="periphery_agrees_with_primary", extra_cpu_ms=2),
        dict(old_accept=True, new_accept=False, synthetic_bg_wrong=True,
             background_visible=True, peripheral_supported=True,
             status="periphery_disagrees_with_primary", extra_cpu_ms=3),
        dict(old_accept=False, new_accept=False, synthetic_bg_wrong=None,
             background_visible=True, peripheral_supported=False,
             status="insufficient_correspondences", extra_cpu_ms=1),
    ]
    got = ring.summarize(rows)
    assert got["new_accepted"] == 1
    assert got["old_latent_or_visible_wrong_accepted"] == 1
    assert got["new_latent_or_visible_wrong_accepted"] == 0
    assert got["new_coverage"] == pytest.approx(1 / 3)
    assert got["median_extra_cpu_ms"] == pytest.approx(2)


def test_zero_coverage_precision_undefined():
    got = ring.summarize([
        dict(old_accept=False, new_accept=False, synthetic_bg_wrong=None,
             background_visible=True, peripheral_supported=False,
             status="insufficient_correspondences", extra_cpu_ms=1)
    ])
    assert got["new_accepted"] == 0
    assert got["new_wrong_among_accepted"] is None
