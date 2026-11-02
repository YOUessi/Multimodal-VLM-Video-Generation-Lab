"""DEV-only experimental provenance and identifiability regression tests."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import dev_temporal_anchor_provenance as method

H, W = 288, 512


def textured(seed=5):
    gen = np.random.default_rng(seed)
    return cv2.GaussianBlur(gen.integers(
        0, 256, (H, W), dtype=np.uint8), (3, 3), .7)


def grid():
    x, y = np.meshgrid(np.arange(12, W - 12, 25),
                       np.arange(12, H - 12, 20))
    return np.stack([x.ravel(), y.ravel()], axis=1).astype(np.float32)


def test_clean_translation_is_consistent_and_supported():
    triple, _, _, observable = method.synthesize_triplet(
        textured(), "clean_global", 144)
    got = method.provenance_evidence(triple, method.BG)
    assert observable and got["supported"] and got["agrees_with_primary"]
    assert got["background_identified"] is False
    assert got["survived_both_hops"] <= got["survived_first_hop"]
    assert got["survived_first_hop"] <= got["detected"]
    assert got["status"] == "provenance_agrees"


def test_clean_translation_conflicting_primary_is_rejected():
    triple, _, _, _ = method.synthesize_triplet(textured(), "clean_global", 5)
    got = method.provenance_evidence(triple, method.FG)
    assert got["supported"] and not got["agrees_with_primary"]
    assert got["status"] == "provenance_disagrees"
    assert got["disagreement_px"] > method.MAX_DISAGREEMENT_PX


def test_fully_occluded_but_visually_identical_foreground_not_solved():
    triple, truth, grid_points, visible = method.synthesize_triplet(
        textured(), "full_screen_observationally_equivalent", 4)
    assert not visible
    got = method.provenance_evidence(triple, method.FG)
    assert got["agrees_with_primary"]
    assert got["background_identified"] is False
    err = np.median(np.linalg.norm(
        grid_points @ (method.FG[:, :2] - truth[:, :2]).T
        + (method.FG[:, 2] - truth[:, 2]), axis=1))
    assert err > 10


def test_unrelated_fullscreen_layer_has_no_semantic_background_proof():
    triple, _, _, visible = method.synthesize_triplet(
        textured(), "full_screen_unrelated_foreground", 81)
    assert not visible
    got = method.provenance_evidence(triple, method.BG)
    assert got["background_identified"] is False


@pytest.mark.parametrize("condition", method.CONDITIONS)
def test_synthetic_controls_deterministic_and_readonly(condition):
    frame = textured(33)
    before = frame.copy()
    a, bg, grid, visible = method.synthesize_triplet(frame, condition, 31)
    b, bg2, grid2, visible2 = method.synthesize_triplet(frame, condition, 31)
    assert len(a) == 3 and all(x.shape == frame.shape for x in a)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert np.array_equal(bg, bg2) and np.array_equal(grid, grid2)
    assert np.array_equal(frame, before)
    assert visible == visible2


def test_provenance_cannot_claim_background_proved():
    triple, _, _, _ = method.synthesize_triplet(textured(), "clean_global", 19)
    got = method.provenance_evidence(triple, method.BG)
    assert got["background_identified"] is False


@pytest.mark.parametrize("bad", [
    (),
    (np.zeros((H, W), dtype=np.uint8),) * 2,
    (np.zeros((30, 30), dtype=np.uint8),) * 3,
    (np.zeros((H, W, 3), dtype=np.uint8),) * 3,
    (np.zeros((H, W), dtype=np.float32),) * 3,
    (np.zeros((H, W), dtype=np.uint8),
     np.zeros((H + 1, W), dtype=np.uint8),
     np.zeros((H, W), dtype=np.uint8)),
])
def test_bad_triplet_fail_closed(bad):
    with pytest.raises(ValueError, match="BAD_TRIPLET"):
        method.provenance_evidence(bad, method.BG)


def test_invalid_primary_rejected():
    triple = (textured(),) * 3
    for p in [np.ones((3, 3)), np.array([[np.nan, 0, 0], [0, 1, 0]])]:
        with pytest.raises(ValueError, match="INVALID_PRIMARY"):
            method.provenance_evidence(triple, p)


def test_none_primary_abstains():
    triple = (textured(),) * 3
    got = method.provenance_evidence(triple, None)
    assert got["status"] == "missing_primary"
    assert not got["agrees_with_primary"]


def test_zero_corners_abstains(monkeypatch):
    monkeypatch.setattr(method.cv2, "goodFeaturesToTrack", lambda *a, **k: None)
    triple = (textured(),) * 3
    got = method.provenance_evidence(triple, method.BG)
    assert got["status"] == "no_initial_anchors"
    assert not got["supported"]


def test_first_hop_no_survivors_abstains(monkeypatch):
    monkeypatch.setattr(method, "_hop", lambda a, b, p: (
        p.copy(), np.zeros(len(p), dtype=bool)))
    triple = (textured(),) * 3
    got = method.provenance_evidence(triple, method.BG)
    assert got["status"] == "insufficient_first_hop"


def test_nonspatial_support_abstains(monkeypatch):
    monkeypatch.setattr(method, "_quadrants", lambda *a: 1)
    triple = (textured(),) * 3
    got = method.provenance_evidence(triple, method.BG)
    assert got["status"] in ("anchors_not_spatially_distributed",
                             "insufficient_two_hop_anchors")


def test_summarize_keeps_abstention_out_of_accuracy():
    rows = [
        {"background_observable": True, "old_accept": True,
         "ring_accept": False, "provenance_accept": False,
         "synthetic_bg_wrong": True, "supported": True,
         "extra_cpu_ms": 2, "survived_both_hops": 25},
        {"background_observable": True, "old_accept": True,
         "ring_accept": True, "provenance_accept": True,
         "synthetic_bg_wrong": False, "supported": True,
         "extra_cpu_ms": 4, "survived_both_hops": 30},
        {"background_observable": False, "old_accept": False,
         "ring_accept": False, "provenance_accept": False,
         "synthetic_bg_wrong": None, "supported": False,
         "extra_cpu_ms": 6, "survived_both_hops": 0},
    ]
    got = method.summarize(rows)
    assert got["baseline"]["accepted"] == 2
    assert got["baseline"]["wrong_accepted"] == 1
    assert got["provenance"]["accepted"] == 1
    assert got["provenance"]["wrong_accepted"] == 0
    assert got["provenance"]["coverage"] == pytest.approx(1 / 3)
    assert got["median_extra_cpu_ms"] == pytest.approx(4)
    assert got["latent_background"] == 1


def test_no_accepted_means_precision_undefined():
    row = {"background_observable": False, "old_accept": False,
           "ring_accept": False, "provenance_accept": False,
           "synthetic_bg_wrong": None, "supported": False,
           "extra_cpu_ms": 0., "survived_both_hops": 0}
    got = method.summarize([row])
    assert got["provenance"]["wrong_among_accepted"] is None


def test_pin_before_media_is_accessed(tmp_path, monkeypatch):
    path = tmp_path / "selection.json"
    path.write_text('{"videos": []}', encoding="utf8")
    monkeypatch.setattr(method.control, "_clip_frames",
                        lambda *a, **k: pytest.fail("MEDIA_READ_UNPINNED"))
    with pytest.raises(RuntimeError, match="SELECTION_PIN_MISMATCH"):
        method.run(path, tmp_path / "missing", tmp_path / "result.json")


def test_dev_only_and_oracle_after_gate_is_staticaly_enforced():
    code = inspect.getsource(method.run)
    assert "audit.dev_rows" in code
    assert "bridge.EXPECTED_SELECTION_SHA256" in code
    assert "bridge.EXPECTED_DOWNLOAD_MANIFEST_SHA256" in code
    assert code.index("provenance_evidence(triplet") < code.index(
        "control.score_known_affine(primary")
    assert "test_videos" not in code.lower()


def test_bad_control_and_small_grayscale_rejected():
    with pytest.raises(ValueError):
        method.synthesize_triplet(textured(), "unknown", 13)
    with pytest.raises(ValueError):
        method.synthesize_triplet(np.zeros((20, 20), dtype=np.uint8),
                                  "clean_global", 13)


def test_periphery_ring_adversary_has_visible_background():
    triple, _, _, observable = method.synthesize_triplet(
        textured(), "peripheral_foreground", 33)
    assert observable
    # Synthetic video is the same seed but has foreground cover at the edge.
    assert not np.array_equal(triple[0], triple[1])
    assert not np.array_equal(triple[1], triple[2])


def test_known_texture_adversary_still_fools_three_frame_provenance():
    """Falsification: FB-tracked history is NOT proof of static identity.

    On deterministic synthetic texture, most previously unoccluded anchors
    can be matched onto a newly moving peripheral layer; all three gates
    may accept a motion that disagrees with visible synthetic background.
    This is a permanent negative regression, not a target to tune away.
    """
    triple, true_bg, grid_points, visible = method.synthesize_triplet(
        textured(5), "peripheral_foreground", 43)
    old, primary = method.control.estimate_pair(triple[1], triple[2])
    tracks = method.bridge.chained_cycle_matches(list(triple[1:]))
    ring = method.peripheral.peripheral_motion_evidence(
        tracks["src"], tracks["dst"], H, W, primary)
    provenance = method.provenance_evidence(triple, primary)
    assert visible and old["gate_accept"]
    assert ring["agrees_with_primary"] and provenance["agrees_with_primary"]
    assert provenance["background_identified"] is False
    # Oracle consulted ONLY after all acceptance decisions above.
    synthetic_truth = method.control.score_known_affine(
        primary, true_bg, grid_points)
    assert synthetic_truth["synthetic_bg_wrong"] is True
