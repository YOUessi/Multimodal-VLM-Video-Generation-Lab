"""Targeted structural-identity safety tests. Synthetic examples contain no TEST data."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from experiments import batch65_dev_structural_identity as m
from experiments import batch62_dev_consensus_veto as b62


def gray_with_patch(x=100, y=100, seed=7):
    gray = np.full((256, 256), 128, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    pattern = rng.integers(10, 246, (11, 11), dtype=np.uint8)
    gray[y-5:y+6, x-5:x+6] = pattern
    return gray


def rgb(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def test_query_patch_shape_and_nonflat():
    g = gray_with_patch()
    p = m.query_template(g, [100.0, 100.0])
    assert p is not None and p.shape == (11, 11)
    assert m.query_template(np.full_like(g, 90), [100, 100]) is None


@pytest.mark.parametrize("xy", [[0, 15], [255, 10], [-1, 3], [2, 2], None])
def test_query_edge_abstains(xy):
    assert m.query_template(gray_with_patch(), xy) is None


def test_identical_patch_structural_confident():
    g = gray_with_patch()
    ref = m.query_template(g, [100, 100])
    ev = m.structural_evidence(g, ref, [100, 100])
    assert ev["candidate_ncc"] >= 0.99
    assert ev["distinctiveness_margin"] >= 0.1
    assert m.structural_evidence(g, ref, [1, 1]) is None


def test_nonlocal_duplicate_does_not_claim_uniqueness():
    g = gray_with_patch()
    g[90-5:90+6, 125-5:125+6] = g[100-5:100+6, 100-5:100+6]
    ref = m.query_template(g, [100, 100])
    ev = m.structural_evidence(g, ref, [100, 100])
    assert ev["candidate_ncc"] >= 0.99
    # 25-pixel-separated duplicate lies outside the search window and is not
    # evidence against local distinctiveness. Do not claim global uniqueness!
    assert ev["distinctiveness_margin"] > 0.1


def test_local_duplicate_detects_ambiguity():
    g = gray_with_patch()
    ref = m.query_template(g, [100, 100])
    # Put exact duplicate 10 px away (do not overwrite the center's patch).
    g[100-5:100+6, 112-5:112+6] = ref
    # At least some rival high score; periodic synthetic content can overlap,
    # therefore assert rival score rather than fragile exact margin.
    ev = m.structural_evidence(g, ref, [100, 100])
    assert ev["rival_ncc"] > 0.9
    assert ev["distinctiveness_margin"] < 0.1


def test_constant_template_abstains_from_evidence():
    assert m.structural_evidence(gray_with_patch(), None, [100, 100]) is None


@pytest.mark.parametrize("bad", [[255, 255], None, [-1, 3], [float("inf"), 1]])
def test_invalid_candidate_never_yields_acceptance(bad):
    g = gray_with_patch()
    if bad == [float("inf"), 1]:
        with pytest.raises(RuntimeError, match="NONFINITE_XY"):
            m.structural_evidence(g, m.query_template(g, [100, 100]), bad)
    else:
        assert m.structural_evidence(g, m.query_template(g, [100, 100]), bad) is None


@pytest.mark.parametrize("badframe", [
    np.zeros((256, 256), dtype=np.uint8),
    np.zeros((256, 256, 4), dtype=np.uint8),
    np.zeros((256, 256, 3), dtype=np.float32),
])
def test_invalid_rgb_frame_fails_closed(badframe):
    with pytest.raises(RuntimeError, match="BATCH65_INVALID_RGB_SHAPE"):
        m.grayscale_uint8(badframe)


def test_policy_subset_and_missing_evidence():
    tr = {b62.BASE: [None, [100., 100.]],
          "template_recover4": [None, [101., 100.]]}
    assert m.accepted(tr, None, 1, "baseline")
    assert m.accepted(tr, None, 1, "template_4px")
    for variant in m.POLICIES:
        assert not m.accepted(tr, None, 1, variant)
    evidence = {"candidate_ncc": 0.80, "distinctiveness_margin": 0.15}
    for variant in m.POLICIES:
        assert m.accepted(tr, evidence, 1, variant)
    tr["template_recover4"][1] = [200., 200.]
    assert not m.accepted(tr, evidence, 1, "template_ncc_050")
    assert m.accepted(tr, evidence, 1, "ncc_050")


def test_explicit_variants_are_fixed():
    assert len(m.VARIANTS) == 6
    with pytest.raises(RuntimeError, match="BATCH65_UNKNOWN_VARIANT"):
        m.accepted({}, {}, 0, "threshold_fit_on_test")


def test_prequery_nulls_and_four_query_validation():
    gray = gray_with_patch()
    frame = rgb(gray)
    arr = np.stack((frame, frame, frame), axis=0)
    q = {"id": "q1", "query_xy": [100., 100.], "query_time": 0}
    record = {"nframes": 3, "tracks": {
        "q1": {b62.BASE: [[100., 100.], [100., 100.], [100., 100.]]}}}
    with pytest.raises(RuntimeError, match="BATCH65_PUBLIC_QUERY_DRIFT"):
        m.per_scene_features(arr, [q], record)
    for j in range(2, 5):
        name = f"q{j}"
        record["tracks"][name] = {b62.BASE: [None, [100., 100.], [100., 100.]]}
    qs = [q] + [dict(q, id=f"q{j}") for j in range(2, 5)]
    vals = m.per_scene_features(arr, qs, record)
    assert len(vals) == 4 and vals["q1"][0] is None
    assert vals["q1"][1]["candidate_ncc"] >= 0.99


def test_pin_write_exclusive_and_drift_detected(tmp_path):
    from experiments import batch64_dev_rgb_identity as b64
    p = tmp_path / "artifact.json"
    digest = b64.pin_write(p, {"a": [1, 2]})
    assert len(digest) == 64
    assert b62.pinned_json(p) == {"a": [1, 2]}
    with pytest.raises(RuntimeError, match="NO_OVERWRITE"):
        b64.pin_write(p, {"a": [2, 3]})
    p.write_text(json.dumps({"a": [2, 3]}))
    with pytest.raises(RuntimeError, match="PIN_DRIFT"):
        b62.pinned_json(p)


def test_scoring_refuses_missing_frozen_feature_population():
    with pytest.raises(RuntimeError, match="BATCH65_SCORING_QUERY_MISMATCH"):
        m.score_scene({"nframes": 2}, [], {"tracks": {}}, {})


def test_source_not_interpreting_image_plane_as_camera():
    assert any("3D camera motion" in c for c in m.evaluate.__code__.co_consts if isinstance(c, str))
