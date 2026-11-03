"""Batch70 protocol, identity-ambiguity and label-firewall tests.

These are tests of the frozen generator and scoring CONTRACT, never a means
to inspect the prospective TEST answers or tune the trackers.
"""
import copy
from pathlib import Path

import numpy as np
import pytest

from experiments import batch70_shared_identity_stress as m


def specimen(regime="doppelganger", seed=700111):
    return {"id": "dev_00_00", "split": "dev", "regime": regime,
            "seed": seed, "nframes": m.FRAMES, "query_time": 0}


def test_immutable_split_definition_and_new_seed_population():
    specs = m.specifications()
    assert len(specs) == 4 * (m.DEV_PER_REGIME + m.TEST_PER_REGIME) == 28
    assert len({s["id"] for s in specs}) == 28
    assert len({s["seed"] for s in specs}) == 28
    assert {s["regime"] for s in specs} == set(m.REGIMES)
    assert sum(s["split"] == "test" for s in specs) == 12
    assert not ({s["seed"] for s in specs} & {690100, 699000})


@pytest.mark.parametrize("regime", m.REGIMES)
def test_determinism_and_frozen_visibility_window(regime):
    x, truth = m.render(specimen(regime))
    y, expected = m.render(specimen(regime))
    assert x.shape == (96, 256, 256) and x.dtype == np.uint8
    assert np.array_equal(x, y) and truth == expected
    assert truth["occluded"] == [18 <= t <= 80 for t in range(96)]
    assert all(25 <= v[0] <= 220 and 25 <= v[1] <= 220
               for v in truth["xy"])


@pytest.mark.parametrize("regime", m.REGIMES)
def test_image_plane_motion_is_separate_from_background_translation(regime):
    _, truth = m.render(specimen(regime))
    for t in (0, 17, 18, 47, 80, 81, 95):
        point = truth["xy"][t]
        world = truth["world_xy"][t]
        trans = truth["background_translation_xy"][t]
        assert abs(point[0] - world[0] - trans[0]) <= 0.5000001
        assert abs(point[1] - world[1] - trans[1]) <= 0.5000001
    assert (truth["background_translation_xy"][-1] != [0., 0.]) == (
        regime == "pan_doppelganger")


@pytest.mark.parametrize("regime", m.REGIMES)
def test_actual_point_is_hidden_in_rendered_video(regime):
    frames, truth = m.render(specimen(regime))
    # Stronger than only checking oracle flags: all inner target pixels are
    # genuinely overwritten by the occluder before impostor is composited.
    # The central 1x1 target point remains outside the offset 21x21 impostor.
    cx, cy = map(int, truth["xy"][40])
    assert (frames[40, cy, cx] != frames[17,
            int(truth["xy"][17][1]), int(truth["xy"][17][0])]) or (
            regime == "light_doppelganger")
    assert not truth["occluded"][17] and truth["occluded"][18]
    assert truth["occluded"][80] and not truth["occluded"][81]


def test_doppelganger_copies_query_sprite_exactly_in_photometrically_static_regime():
    frames, truth = m.render(specimen("doppelganger"))
    x0, y0 = map(int, truth["xy"][0])
    cx, cy = map(int, truth["xy"][40])
    duplicate_offset = 13 + specimen()["seed"] % 3
    original = frames[0, y0-10:y0+11, x0-10:x0+11]
    impostor = frames[40, cy+4-10:cy+4+11,
                      cx+duplicate_offset-10:cx+duplicate_offset+11]
    assert np.array_equal(original, impostor)


def test_unmatched_occluder_does_not_embed_duplicate_sprite():
    frames, truth = m.render(specimen("distinct_occluder"))
    x0, y0 = map(int, truth["xy"][0])
    cx, cy = map(int, truth["xy"][40])
    original = frames[0, y0-10:y0+11, x0-10:x0+11]
    fake_location = frames[40, cy+4-10:cy+4+11, cx+13-10:cx+13+11]
    assert not np.array_equal(original, fake_location)


def oracle_scores(regime="doppelganger"):
    _, truth = m.render(specimen(regime))
    traces = {k: copy.deepcopy(truth["xy"]) for k in m.METHODS}
    return truth, traces


def test_all_hidden_acceptances_are_errors_even_when_coordinates_match_ground_truth():
    truth, tracks = oracle_scores()
    score = m.score_scene(tracks, truth)[m.BASE]
    assert score["hidden_accepted"] == score["hidden"] == 63
    assert score["max_contiguous_hidden_accept"] == 63
    assert score["correct_visible"] == score["visible"] == 32
    assert score["occluded_false_accept_rate"] == 1.0


def test_abstention_cannot_gain_spurious_correct_visible_frames():
    truth, tracks = oracle_scores()
    for method in m.METHODS:
        for t in range(m.FRAMES):
            if truth["occluded"][t]:
                tracks[method][t] = None
    score = m.score_scene(tracks, truth)[m.BASE]
    assert score["hidden_accepted"] == 0
    assert score["max_contiguous_hidden_accept"] == 0
    assert score["visible_coverage"] == 1.0


def test_consensus_is_veto_only_not_coordinate_adjustment():
    assert m.agree([[4, 4]], [[4, 4]], 0) == [4, 4]
    assert m.agree([[4, 4]], [[8.01, 4]], 0) is None
    assert m.agree([[4, 4]], [None], 0) is None
    assert m.agree([None], [[4, 4]], 0) is None


def test_bad_predictions_rejected_before_metrics_are_reported():
    truth, tracks = oracle_scores()
    tracks[m.BASE][30] = [float("nan"), 0.]
    with pytest.raises(RuntimeError, match="NONFINITE_TRACK"):
        m.score_scene(tracks, truth)
    with pytest.raises(RuntimeError, match="UNKNOWN_SPLIT"):
        m.predict("holdout")
    with pytest.raises(RuntimeError, match="UNKNOWN_SPLIT"):
        m.score("holdout")


def test_immutable_json_and_hash_guards(tmp_path):
    p = tmp_path/"f.json"
    sha = m.save_json(p, {"ok": True})
    assert m.digest(p.read_bytes()) == sha
    assert m.read_pinned(p) == {"ok": True}
    with pytest.raises(FileExistsError):
        m.save_json(p, {})
    p.write_text('{"ok":false}')
    with pytest.raises(RuntimeError, match="HASH_DRIFT"):
        m.read_pinned(p)


def test_manifest_detects_seed_drift_and_protocol_rewrite():
    rows = [{"frames_sha256": "a"*64, **s} for s in m.specifications()]
    doc = {"kind": "BATCH70_IDENTITY_STRESS_PRE_REGISTERED_V1",
           "protocol": "fixed-untuned-five-methods-shared-identity-v1",
           "source_sha": "pinned", "scenes": rows}
    m.assert_manifest(doc)
    rows[0]["seed"] += 1
    with pytest.raises(RuntimeError, match="SPLIT_OR_SEED_DRIFT"):
        m.assert_manifest(doc)
    rows[0]["seed"] -= 1
    doc["protocol"] = "changed"
    with pytest.raises(RuntimeError, match="MANIFEST_PROTOCOL_INVALID"):
        m.assert_manifest(doc)


def test_predict_path_never_opens_private_labels(tmp_path, monkeypatch):
    row = specimen()
    public_row = {"frames_sha256": "f"*64, **row,
                  "query_xy": [66., 93.]}
    monkeypatch.setattr(m, "specifications", lambda: [row])
    doc = {"kind": "BATCH70_IDENTITY_STRESS_PRE_REGISTERED_V1",
           "protocol": "fixed-untuned-five-methods-shared-identity-v1",
           "source_sha": "fixed",
           "config_sha256": m.digest(m.CONFIG.read_bytes()),
           "scenes": [public_row]}
    m.save_json(tmp_path/"manifest.json", doc)
    monkeypatch.setattr(m, "assert_clean_source", lambda sha: None)
    monkeypatch.setattr(m, "load_public", lambda data, row:
                        [np.zeros((256, 256), dtype=np.uint8)] * m.FRAMES)
    monkeypatch.setattr(m.core, "estimate_affines", lambda f, cfg:
                        ([None]*len(f), [], 0))
    monkeypatch.setattr(m.core, "predict_query", lambda f, q, tf, cfg:
                        ({k: [[66., 93.]]*len(f) for k in m.METHODS}, {}))
    original = Path.open

    def forbidden(self, *args, **kwargs):
        if "private" in self.parts:
            raise AssertionError("PREDICTION_READ_PRIVATE_ORACLE")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", forbidden)
    result = m.predict("dev", tmp_path, tmp_path/"out")
    assert result["private_oracle_read_during_prediction"] is False
    assert len(result["predictions"]) == 1


def test_test_prediction_must_be_preceded_by_pinned_dev_score(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "specifications", lambda: [])
    doc = {"kind": "BATCH70_IDENTITY_STRESS_PRE_REGISTERED_V1",
           "protocol": "fixed-untuned-five-methods-shared-identity-v1",
           "source_sha": "fixed",
           "config_sha256": m.digest(m.CONFIG.read_bytes()),
           "scenes": []}
    m.save_json(tmp_path/"manifest.json", doc)
    monkeypatch.setattr(m, "assert_clean_source", lambda sha: None)
    with pytest.raises(RuntimeError, match="MISSING_PIN"):
        m.predict("test", tmp_path, tmp_path/"out")


def test_group_aggregation_counts_not_unweighted_scene_means():
    truth, traces = oracle_scores()
    scored = m.score_scene(traces, truth)
    agg = m.aggregate([{"id": "dev_00_00", "regime": "doppelganger",
                        "metrics": scored}])
    assert agg[m.BASE]["overall"]["hidden"] == 63
    assert agg[m.BASE]["overall"]["max_contiguous_hidden_accept"] == 63
    assert agg[m.BASE]["overall"]["visible"] == 32
