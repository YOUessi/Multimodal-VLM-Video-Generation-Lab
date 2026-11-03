"""Synthetic and firewall regression tests for Batch69; never use TEST outcomes for selection."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import batch69_prospective_occlusion as m


def spec(regime="clean", seed=690100):
    return {"id": "dev_00_00", "split": "dev", "regime": regime,
            "seed": seed, "nframes": m.FRAMES, "query_time": 0}


def test_unique_prespecified_splits_and_regimes():
    specs = m.specifications()
    assert len(specs) == 6 * (m.DEV_PER_REGIME + m.TEST_PER_REGIME)
    assert len({row["seed"] for row in specs}) == len(specs)
    assert len({row["id"] for row in specs}) == len(specs)
    assert {row["split"] for row in specs} == {"dev", "test"}
    assert {row["regime"] for row in specs} == set(m.REGIMES)


@pytest.mark.parametrize("regime", m.REGIMES)
def test_render_is_deterministic_and_oracle_valid(regime):
    a, labels = m.render(spec(regime))
    b, labels2 = m.render(spec(regime))
    assert a.shape == (m.FRAMES, 256, 256)
    assert a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert labels == labels2
    assert len(labels["xy"]) == len(labels["occluded"]) == m.FRAMES
    assert not labels["occluded"][0]
    if "occluder" in regime:
        assert any(labels["occluded"])
    else:
        assert not any(labels["occluded"])


@pytest.mark.parametrize("regime", ["clean", "camera_pan", "pan_shared_motion_occluder"])
def test_world_screen_decomposition_is_2d_not_3d(regime):
    scene = spec(regime)
    _, labels = m.render(scene)
    for t in (0, 1, 17, 39):
        xy = labels["xy"][t]
        w = labels["world_xy"][t]
        c = labels["background_translation_xy"][t]
        assert abs(xy[0] - w[0] - c[0]) <= 0.51
        assert abs(xy[1] - w[1] - c[1]) <= 0.51


@pytest.mark.parametrize("regime", ["shared_motion_occluder", "pan_shared_motion_occluder"])
def test_adversarial_shared_motion_long_occlusion(regime):
    _, labels = m.render(spec(regime))
    assert all(labels["occluded"][i] for i in range(10, 33))
    assert not labels["occluded"][9]
    assert not labels["occluded"][33]


def test_crossing_occluder_has_real_visibility_transition():
    _, labels = m.render(spec("crossing_occluder"))
    occ = labels["occluded"]
    assert not occ[0]
    assert sum(occ) >= 5
    assert sum(occ) < m.FRAMES-1


def test_invalid_non_float_tracker_point_fails_closed():
    o = {"xy": [[1, 2]]*m.FRAMES, "occluded": [False]*m.FRAMES,
         "world_xy": [[1, 2]]*m.FRAMES,
         "background_translation_xy": [[0, 0]]*m.FRAMES}
    traces = {k: [[1, 2]]*m.FRAMES for k in m.METHODS}
    traces[m.BASE][1] = [float("nan"), 1]
    with pytest.raises(RuntimeError, match="NONFINITE_TRACK"):
        m.score_scene(traces, o)


def test_all_correct_visible_and_accepted_hidden_is_counted_as_error():
    _, oracle = m.render(spec("shared_motion_occluder"))
    tracks = {k: copy.deepcopy(oracle["xy"]) for k in m.METHODS}
    scored = m.score_scene(tracks, oracle)[m.BASE]
    assert scored["correct_visible"] == scored["visible"]
    assert scored["hidden_accepted"] == scored["hidden"]
    assert scored["max_contiguous_hidden_accept"] >= 23
    assert scored["occluded_false_accept_rate"] == 1.0
    assert scored["pck8_full_visible"] == 1.0
    assert scored["world_relative_support"] == scored["visible"]


def test_perfect_abstention_when_hidden_preserves_visible_coverage():
    _, oracle = m.render(spec("shared_motion_occluder"))
    seq = [None if occ and idx else xy for idx, (xy, occ) in
           enumerate(zip(oracle["xy"], oracle["occluded"]))]
    tracks = {k: copy.deepcopy(seq) for k in m.METHODS}
    scored = m.score_scene(tracks, oracle)[m.BASE]
    assert scored["hidden_accepted"] == 0
    assert scored["max_contiguous_hidden_accept"] == 0
    assert scored["visible_coverage"] == 1.0


def test_lookahead_oracle_cannot_relabel_prediction_as_abstention():
    _, oracle = m.render(spec("shared_motion_occluder"))
    tracks = {k: copy.deepcopy(oracle["xy"]) for k in m.METHODS}
    before = m.score_scene(tracks, oracle)[m.BASE]["hidden_accepted"]
    tracks[m.BASE][10] = None
    after = m.score_scene(tracks, oracle)[m.BASE]["hidden_accepted"]
    assert before - after == 1
    # Verification that all other methods and the oracle remain unchanged.
    assert m.score_scene(tracks, oracle)["klt_fb"]["hidden_accepted"] == before


def test_consensus_only_vetoes_existing_point():
    assert m.agree([[1, 2]], [[1, 2]], 0) == [1, 2]
    assert m.agree([[1, 2]], [[5.1, 2]], 0) is None
    assert m.agree([[1, 2]], [None], 0) is None
    assert m.agree([None], [[1, 2]], 0) is None


def test_aggregation_separates_regimes_and_splits():
    _, truth = m.render(spec("clean"))
    tracks = {k: truth["xy"] for k in m.METHODS}
    out = m.score_scene(tracks, truth)
    total = m.aggregate([{"id": "x", "regime": "clean", "metrics": out}])
    assert total[m.BASE]["overall"]["correct_visible"] == m.FRAMES-1
    assert "camera_pan" not in total[m.BASE]["by_regime"]


def test_immutable_json_outputs_have_sha_and_detect_tamper(tmp_path):
    dest = tmp_path / "x.json"
    checksum = m.save_json(dest, {"value": 8})
    assert m.read_pinned(dest) == {"value": 8}
    assert checksum == m.digest(dest.read_bytes())
    with pytest.raises(FileExistsError):
        m.save_json(dest, {"value": 9})
    dest.write_text('{"value": 9}')
    with pytest.raises(RuntimeError, match="HASH_DRIFT"):
        m.read_pinned(dest)


def test_manifest_checks_source_seeds_and_group_counts():
    rows = [{"frames_sha256": "a"*64, **x} for x in m.specifications()]
    doc = {"kind": "BATCH69_SYNTHETIC_PRE_REGISTERED_V1",
           "protocol": "fixed-untuned-five-methods-v1",
           "source_sha": "abcdef", "scenes": rows}
    m.assert_manifest(doc)
    rows[0]["seed"] = 99
    with pytest.raises(RuntimeError, match="SPLIT_OR_SEED_DRIFT"):
        m.assert_manifest(doc)


def test_predictions_do_not_read_private_oracle(tmp_path, monkeypatch):
    row = spec("clean")
    scene = {"frames_sha256": "abc", **row,
             "query_xy": [70, 96]}
    monkeypatch.setattr(m, "specifications", lambda: [row])
    doc = {"kind": "BATCH69_SYNTHETIC_PRE_REGISTERED_V1",
           "protocol": "fixed-untuned-five-methods-v1",
           "source_sha": "frozen",
           "config_sha256": m.digest(m.CONFIG.read_bytes()),
           "scenes": [scene]}
    m.save_json(tmp_path/"manifest.json", doc)
    # Replace only expensive vision routines with deterministic stub tracks.
    frames = [np.zeros((256, 256), np.uint8) for _ in range(m.FRAMES)]
    monkeypatch.setattr(m, "load_public", lambda data, row: frames)
    monkeypatch.setattr(m, "assert_clean_source", lambda sha: None)
    monkeypatch.setattr(m.core, "estimate_affines", lambda f, c:
                        ([None]*len(f), [], 0))
    monkeypatch.setattr(m.core, "predict_query", lambda f, q, transforms, c:
                        ({k: [[70, 96]]*len(f) for k in m.METHODS}, {}))
    original = Path.open
    def forbidden(self, *args, **kwargs):
        if "private" in self.parts:
            raise AssertionError("PREDICTION_OPENED_PRIVATE_ORACLE")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", forbidden)
    pred = m.predict("dev", tmp_path, tmp_path/"results")
    assert len(pred["predictions"]) == 1
    assert pred["private_oracle_read_during_prediction"] is False


def test_test_predict_requires_pinned_dev_score(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "specifications", lambda: [])
    doc = {"kind": "BATCH69_SYNTHETIC_PRE_REGISTERED_V1",
           "protocol": "fixed-untuned-five-methods-v1",
           "source_sha": "frozen",
           "config_sha256": m.digest(m.CONFIG.read_bytes()), "scenes": []}
    m.save_json(tmp_path/"manifest.json", doc)
    monkeypatch.setattr(m, "assert_clean_source", lambda sha: None)
    with pytest.raises(RuntimeError, match="MISSING_PIN"):
        m.predict("test", tmp_path, tmp_path/"out")


def test_unsafe_oracle_trace_or_split_is_rejected(tmp_path):
    wrong = {"xy": [], "occluded": [],
             "world_xy": [], "background_translation_xy": []}
    with pytest.raises(RuntimeError, match="ORACLE_LENGTH_DRIFT"):
        m.score_scene({k: [] for k in m.METHODS}, wrong)
    with pytest.raises(RuntimeError, match="UNKNOWN_SPLIT"):
        m.predict("holdout", tmp_path, tmp_path)
    with pytest.raises(RuntimeError, match="UNKNOWN_SPLIT"):
        m.score("holdout", tmp_path, tmp_path)
