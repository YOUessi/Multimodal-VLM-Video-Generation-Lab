"""Pre-registration tests for paired counterfactual ambiguity, data seals and labels."""
from pathlib import Path
import copy
import math

import numpy as np
import pytest

from experiments import batch71_counterfactual_identity as m


def specimen(regime="static_twins", seed=710111):
    return {
        "id": "dev_00_00", "split": "dev", "regime": regime,
        "seed": seed, "query_time": 0, "nframes": m.FRAMES,
    }


def oracle_tracks(keep_visible=True):
    _, world_a, world_b = m.render(specimen())
    trace = copy.deepcopy(world_a["xy"])
    if not keep_visible:
        trace = [trace[0]] + [None]*(m.FRAMES-1)
    return world_a, world_b, {k: copy.deepcopy(trace) for k in m.METHODS}


def test_population_is_fresh_pair_disjoint_and_independently_counted():
    rows = m.specs()
    assert len(rows) == 15
    assert len({r["id"] for r in rows}) == 15
    assert len({r["seed"] for r in rows}) == 15
    assert sum(r["split"] == "dev" for r in rows) == 9
    assert sum(r["split"] == "test" for r in rows) == 6
    assert {r["regime"] for r in rows} == set(m.REGIMES)
    assert len(m.METHODS) == 5


@pytest.mark.parametrize("regime", m.REGIMES)
def test_deterministic_frames_and_latent_worlds(regime):
    a = m.render(specimen(regime))
    b = m.render(specimen(regime))
    assert a[0].shape == (96, 256, 256)
    assert a[0].dtype == np.uint8
    assert np.array_equal(a[0], b[0])
    assert a[1] == b[1] and a[2] == b[2]


@pytest.mark.parametrize("regime", m.REGIMES)
def test_identical_query_and_divergent_post_occlusion_identity(regime):
    frames, a, b = m.render(specimen(regime))
    assert a["xy"][0] == b["xy"][0]
    assert a["xy"][:m.HIDDEN_STOP] == b["xy"][:m.HIDDEN_STOP]
    assert a["occluded"] == b["occluded"]
    assert a["occluded"] == [
        m.HIDDEN_START <= t < m.HIDDEN_STOP for t in range(m.FRAMES)]
    for t in range(m.HIDDEN_STOP, m.FRAMES):
        assert math.dist(a["xy"][t], b["xy"][t]) > 2*m.PCK
    # One actual public image sequence is shared by both identity hypotheses.
    assert frames[0, int(a["xy"][0][1]), int(a["xy"][0][0])] == (
        frames[0, int(b["xy"][0][1]), int(b["xy"][0][0])])


def test_opaque_occluder_overwrites_both_target_and_distractor():
    frames, a, b = m.render(specimen())
    t = 40
    for truth in (a, b):
        x, y = map(int, truth["xy"][t])
        assert frames[t, y, x] == 110
    assert frames[t, 110, 130] == 110


def test_camera_shift_is_2d_only_and_does_not_resolve_identity():
    _, a, b = m.render(specimen("planar_pan_twins"))
    assert a["shift_xy"][75] != [0., 0.]
    assert b["shift_xy"][75] == a["shift_xy"][75]
    _, static, _ = m.render(specimen())
    assert static["shift_xy"][75] == [0., 0.]


def test_perfect_left_tracking_cannot_be_perfect_in_both_worlds():
    a, b, traces = oracle_tracks()
    v = m.score_pair(traces, a, b)[m.METHODS[0]]
    assert v["pre_correct"] == v["pre_visible"] == 17
    assert v["hidden_accepted"] == v["hidden"] == 46
    assert v["max_hidden_run"] == 46
    assert v["post_accepted"] == v["post_visible"] == 32
    assert v["post_correct_a"] == 32 and v["post_correct_b"] == 0
    assert v["post_accepted_wrong_both"] == 0
    agg = m.aggregate([{"metrics": m.score_pair(traces, a, b)}])[m.METHODS[0]]
    assert agg["post_pck8_random_world"] == 0.5
    assert agg["post_pck8_ceiling_given_acceptance"] == 0.5
    assert agg["post_wrong_accepted_fraction_random_world"] == 0.5
    assert agg["hidden_false_accept_rate"] == 1.


def test_perfect_right_tracking_symmetric_after_occlusion():
    a, b, traces = oracle_tracks()
    for method in m.METHODS:
        traces[method][m.HIDDEN_STOP:] = copy.deepcopy(b["xy"][m.HIDDEN_STOP:])
    v = m.score_pair(traces, a, b)[m.METHODS[0]]
    assert (v["post_correct_a"], v["post_correct_b"]) == (0, 32)


def test_no_tracking_after_occlusion_has_zero_identity_coverage():
    a, b, traces = oracle_tracks(keep_visible=False)
    v = m.score_pair(traces, a, b)[m.METHODS[0]]
    assert v["post_accepted"] == 0
    assert v["hidden_accepted"] == 0
    agg = m.aggregate([{"metrics": m.score_pair(traces, a, b)}])[m.METHODS[0]]
    assert agg["post_coverage"] == 0
    assert agg["post_wrong_accepted_fraction_random_world"] is None


def test_prediction_far_from_both_has_error_beyond_half_bound():
    a, b, traces = oracle_tracks()
    for method in m.METHODS:
        traces[method][m.HIDDEN_STOP:] = [
            [5., 5.] for _ in range(m.FRAMES - m.HIDDEN_STOP)]
    q = m.score_pair(traces, a, b)[m.METHODS[0]]
    assert q["post_accepted_wrong_both"] == 32
    agg = m.aggregate([{"metrics": m.score_pair(traces, a, b)}])[m.METHODS[0]]
    assert agg["post_pck8_random_world"] == 0
    assert agg["post_wrong_accepted_fraction_random_world"] == 1


def test_invalid_trace_nan_and_length_fail_closed():
    a, b, traces = oracle_tracks()
    traces[m.METHODS[0]][70] = [float("nan"), 10.]
    with pytest.raises(RuntimeError, match="INVALID_PREDICTED_POINT"):
        m.score_pair(traces, a, b)
    traces[m.METHODS[0]] = traces[m.METHODS[0]][:-1]
    with pytest.raises(RuntimeError, match="TRACE_LENGTH_DRIFT"):
        m.score_pair(traces, a, b)


def test_counterfactual_visibility_mismatch_fails_closed():
    a, b, traces = oracle_tracks()
    b["occluded"][30] = False
    with pytest.raises(RuntimeError, match="VISIBILITY_WORLD_DRIFT"):
        m.score_pair(traces, a, b)


def test_wrong_split_rejected_without_reading_files():
    with pytest.raises(RuntimeError, match="INVALID_SPLIT"):
        m.predict("heldout")
    with pytest.raises(RuntimeError, match="INVALID_SPLIT"):
        m.score("heldout")


def test_manifest_seed_and_protocol_are_immutable():
    rows = [{**s, "frames_sha256": "a"*64, "query_xy": [71., 117.]}
            for s in m.specs()]
    man = {"kind": m.MANIFEST_KIND, "protocol": m.PROTOCOL,
           "source_sha": "frozen", "scenes": rows}
    m.assert_manifest(man)
    man["scenes"][0]["seed"] += 1
    with pytest.raises(RuntimeError, match="SPLIT_SEED_OR_REGIME_DRIFT"):
        m.assert_manifest(man)
    man["scenes"][0]["seed"] -= 1
    man["protocol"] = "altered"
    with pytest.raises(RuntimeError, match="INVALID_MANIFEST"):
        m.assert_manifest(man)


def test_hash_pin_write_once_and_tamper_detection(tmp_path):
    p = tmp_path/"immutable.json"
    d = {"value": 42}
    digest = m.b.save_json(p, d)
    assert m.b.digest(p.read_bytes()) == digest
    assert m.b.read_pinned(p) == d
    with pytest.raises(FileExistsError):
        m.b.save_json(p, {"value": 41})
    p.write_text('{"value":0}')
    with pytest.raises(RuntimeError, match="HASH_DRIFT"):
        m.b.read_pinned(p)


def test_predict_does_not_read_either_private_world(tmp_path, monkeypatch):
    scene = specimen()
    frames, a, _ = m.render(scene)
    row = {**scene, "frames_sha256": "a"*64, "query_xy": a["xy"][0]}
    monkeypatch.setattr(m, "specs", lambda: [scene])
    man = {"kind": m.MANIFEST_KIND, "protocol": m.PROTOCOL,
           "source_sha": "pinned",
           "config_sha256": m.b.digest(m.CONFIG.read_bytes()),
           "scenes": [row]}
    m.b.save_json(tmp_path/"manifest.json", man)
    monkeypatch.setattr(m, "source_guard", lambda x: None)
    monkeypatch.setattr(m, "load_frames", lambda data, rec: [
        frames[i] for i in range(m.FRAMES)])
    monkeypatch.setattr(m.b.core, "estimate_affines",
                        lambda frames, cfg: ([None]*len(frames), [], 0))
    monkeypatch.setattr(m.b.core, "predict_query",
                        lambda fs, q, af, cfg:
                        ({k: [list(q["query_xy"])]*len(fs) for k in m.METHODS}, {}))
    real_open = Path.open

    def prevent_oracle_read(self, *args, **kwargs):
        if "private" in self.parts:
            raise AssertionError("BATCH71_PRIVATE_ORACLE_READ_DURING_PREDICT")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", prevent_oracle_read)
    pred = m.predict("dev", data=tmp_path, output=tmp_path/"output")
    assert len(pred["predictions"]) == 1
    assert pred["one_prediction_per_counterfactual_pair"]
    assert pred["private_oracle_read_during_prediction"] is False


def test_test_prediction_requires_previous_dev_score(tmp_path, monkeypatch):
    scene = specimen()
    monkeypatch.setattr(m, "specs", lambda: [scene])
    man = {"kind": m.MANIFEST_KIND, "protocol": m.PROTOCOL,
           "source_sha": "pinned",
           "config_sha256": m.b.digest(m.CONFIG.read_bytes()),
           "scenes": [{**scene, "query_xy": [71., 117.],
                       "frames_sha256": "a"*64}]}
    m.b.save_json(tmp_path/"manifest.json", man)
    monkeypatch.setattr(m, "source_guard", lambda x: None)
    with pytest.raises(RuntimeError, match="MISSING_PIN"):
        m.predict("test", data=tmp_path, output=tmp_path/"output")


def test_aggregate_counts_pairs_not_as_independent_two_video_samples():
    a, b, tracks = oracle_tracks()
    metrics = m.score_pair(tracks, a, b)
    pair = {"metrics": metrics}
    one = m.aggregate([pair])[m.METHODS[0]]
    two = m.aggregate([pair, pair])[m.METHODS[0]]
    assert one["post_visible"] == 32
    assert two["post_visible"] == 64
    assert one["post_pck8_random_world"] == two["post_pck8_random_world"]
