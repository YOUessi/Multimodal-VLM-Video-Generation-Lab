"""Batch68 contract and counterfactual tests: all synthetic, never TEST."""
from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest

from experiments import batch68_dev_causal_flow as m
from experiments import batch62_dev_consensus_veto as b62
from experiments import batch64_dev_rgb_identity as b64


def noise(seed=9):
    return np.random.default_rng(seed).integers(0, 256, (256, 256), dtype=np.uint8)


def shift(frame, dx, dy):
    mat = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, mat, (256, 256), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)


@pytest.mark.parametrize("shape,dtype", [
    ((256, 256, 1), np.uint8), ((255, 256, 3), np.uint8),
    ((256, 256, 3), np.float32), ((256, 256), np.uint8),
])
def test_rgb_shape_fail_closed(shape, dtype):
    with pytest.raises(RuntimeError, match="BATCH68_RGB_SHAPE"):
        m.gray(np.zeros(shape, dtype=dtype))


def test_valid_rgb_gray_conversion():
    frame = cv2.cvtColor(noise(), cv2.COLOR_GRAY2RGB)
    assert np.array_equal(m.gray(frame), cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))


@pytest.mark.parametrize("point", [None, [1, 100], [0, 0], [255, 255],
                                    [150, 253], [-1, 50]])
def test_edges_or_missing_abstain(point):
    assert m.valid_point(point) is None


@pytest.mark.parametrize("point", [[float("inf"), 4], [float("nan"), 50],
                                    [True, 20], [50, "forty"]])
def test_invalid_finite_xy_rejected(point):
    with pytest.raises(RuntimeError):
        m.valid_point(point)


def test_flow_identity_has_small_residual_and_cycle_error():
    a = noise()
    e = m.lk_evidence(a, a, [101, 111], [101, 111], None)
    assert e is not None
    assert e["tracker_flow_residual_px"] < 0.25
    assert e["flow_cycle_error_px"] < 0.25
    assert "candidate_minus_global_2d_px" not in e


@pytest.mark.parametrize("dx,dy", [(3, 2), (-3, 4), (0, -4)])
def test_flow_translation_tracks_moving_pixels(dx, dy):
    a = noise()
    b = shift(a, dx, dy)
    e = m.lk_evidence(a, b, [118, 107], [118+dx, 107+dy],
                      {"dx": dx, "dy": dy})
    assert e is not None
    assert e["tracker_flow_residual_px"] < 1.5
    assert e["flow_cycle_error_px"] < 1.5
    assert e["candidate_minus_global_2d_px"] == pytest.approx(0.0)


def test_incorrect_track_has_large_one_step_residual():
    a = noise()
    e = m.lk_evidence(a, a, [100, 100], [120, 100], {"dx": 0, "dy": 0})
    assert e is not None
    assert e["tracker_flow_residual_px"] > 10
    assert e["candidate_minus_global_2d_px"] == pytest.approx(20.0)


def test_bg_relative_proxy_static_in_image_translated_scene():
    a = noise()
    b = shift(a, 5, -2)
    proxy = m.phase_translation(a, b)
    assert proxy is not None
    assert proxy["dx"] == pytest.approx(5, abs=1.0)
    assert proxy["dy"] == pytest.approx(-2, abs=1.0)


def test_phase_image_validation_and_flat_fail_closed():
    with pytest.raises(RuntimeError, match="BATCH68_PHASE_SHAPE"):
        m.phase_translation(np.zeros((50, 50), dtype=np.uint8), noise())
    assert m.phase_translation(np.full((256, 256), 42, dtype=np.uint8),
                               np.full((256, 256), 42, dtype=np.uint8)) is None


@pytest.mark.parametrize("prev,pred", [
    (None, [100, 100]), ([100, 100], None),
    ([2, 100], [100, 100]), ([100, 100], [254, 100]),
])
def test_lk_missing_or_border_abstains(prev, pred):
    a = noise()
    assert m.lk_evidence(a, a, prev, pred, None) is None


def test_lk_wrong_frame_shape_fails_closed():
    with pytest.raises(RuntimeError, match="BATCH68_LK_SHAPE"):
        m.lk_evidence(np.zeros((255, 256), np.uint8), noise(),
                      [100, 100], [100, 100], None)


def test_template_and_lk_gate_only_veto():
    tr = {b62.BASE: [None, [100, 100]],
          "template_recover4": [None, [101, 100]]}
    e = {"tracker_flow_residual_px": 0.5, "flow_cycle_error_px": 0.4}
    assert m.accepted(tr, None, 1, "baseline")
    assert m.accepted(tr, None, 1, "template_4px")
    for variant in m.POLICIES:
        assert m.accepted(tr, e, 1, variant)
        assert not m.accepted(tr, None, 1, variant)
    tr["template_recover4"][1] = [180, 180]
    assert m.accepted(tr, e, 1, "lk_4px_fb2px")
    assert not m.accepted(tr, e, 1, "template_lk_4px_fb2px")
    tr[b62.BASE][1] = None
    assert not m.accepted(tr, e, 1, "lk_4px_fb2px")


@pytest.mark.parametrize("residual,cycle,strict,loose", [
    (0.1, 0.2, True, True),
    (3.1, 1.4, False, True),
    (0.4, 1.6, False, True),
    (5.0, 0.0, False, False),
    (1.0, 2.1, False, False),
])
def test_fixed_gate_thresholds(residual, cycle, strict, loose):
    tr = {b62.BASE: [None, [100, 100]],
          "template_recover4": [None, [101, 100]]}
    e = {"tracker_flow_residual_px": residual,
         "flow_cycle_error_px": cycle}
    assert m.accepted(tr, e, 1, "lk_2px_fb1px") is strict
    assert m.accepted(tr, e, 1, "lk_4px_fb2px") is loose
    assert m.accepted(tr, e, 1, "template_lk_2px_fb1px") is strict


def test_unknown_policy_fails_closed():
    with pytest.raises(RuntimeError, match="BATCH68_UNKNOWN_POLICY"):
        m.accepted({}, None, 1, "test_optimized_gate")
    assert len(m.POLICIES) == 4
    assert len(m.VARIANTS) == 6


def toy():
    gr = noise()
    frames = np.stack([cv2.cvtColor(gr, cv2.COLOR_GRAY2RGB)] * 3)
    q = []
    tracks = {}
    labels = {}
    for j in range(4):
        qid = f"q{j}"
        q.append({"id": qid, "query_time": 0, "query_xy": [100, 100]})
        tracks[qid] = {
            b62.BASE: [[100, 100], [100, 100], [100, 100]],
            "template_recover4": [[100, 100], [100, 100], [100, 100]],
        }
        labels[qid] = {"xy": [[100, 100]] * 3,
                        "occluded": [False, False, j == 0]}
    return frames, q, {"nframes": 3, "tracks": tracks}, {"tracks": labels}


def test_toy_feature_shape_and_baseline_replay():
    frames, q, pred, oracle = toy()
    extracted = m.per_scene_features(frames, q, pred)
    assert set(extracted["query"]) == set(pred["tracks"])
    assert len(extracted["phase"]) == 3
    for row in extracted["query"].values():
        assert len(row) == 3 and row[0] is None
        assert row[1]["tracker_flow_residual_px"] < 0.5
    scored = m.score_scene(pred, q, oracle, extracted)
    assert scored["baseline"]["invisible"] == 1
    assert scored["baseline"]["visible"] == 7
    assert scored["baseline"]["invisible_false_accept"] == 1
    assert scored["baseline"]["correct8"] == 7
    assert scored["lk_4px_fb2px"]["correct8"] == 7


def test_query_population_mismatch_fails_closed():
    frames, q, pred, _ = toy()
    with pytest.raises(RuntimeError, match="BATCH68_PUBLIC_QUERY_POPULATION"):
        m.per_scene_features(frames, q[:3], pred)


def test_bad_prequery_prediction_fails_closed():
    frames, q, pred, _ = toy()
    q[0]["query_time"] = 1
    with pytest.raises(RuntimeError, match="BATCH68_TRACK_TIME_INVALID"):
        m.per_scene_features(frames, q, pred)


def test_frame_data_invalid_fails_closed():
    frames, q, pred, _ = toy()
    with pytest.raises(RuntimeError, match="BATCH68_FRAMES_INVALID"):
        m.per_scene_features(frames.astype(np.float32), q, pred)


def test_bad_scores_including_prequery_fail_closed():
    frames, q, pred, oracle = toy()
    evidence = m.per_scene_features(frames, q, pred)
    evidence["query"]["q0"][0] = {"tracker_flow_residual_px": 0.0}
    with pytest.raises(RuntimeError, match="BATCH68_PREQUERY_EVIDENCE"):
        m.score_scene(pred, q, oracle, evidence)


def test_pin_json_is_immutable_and_verifiable(tmp_path):
    path = tmp_path / "features.json"
    sha = b64.pin_write(path, {"split": "dev", "values": [1, 2, 3]})
    assert len(sha) == 64
    assert b62.pinned_json(path)["values"] == [1, 2, 3]
    with pytest.raises(RuntimeError, match="NO_OVERWRITE"):
        b64.pin_write(path, {"tamper": True})
    path.write_text('{"tamper":true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="PIN_DRIFT"):
        b62.pinned_json(path)


def test_methods_are_prognostic_not_claimed_physical_world_motion():
    assert m.PHASE_DOWNSAMPLE == (128, 128)
    assert m.LK_LEVEL == 2
    assert m.PHASE_MIN_RESPONSE > 0
    assert all(v[1] > 0 and v[2] > 0 for v in m.POLICIES.values())
