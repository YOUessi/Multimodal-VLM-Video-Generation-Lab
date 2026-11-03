"""Batch73: test observable-evidence / counterfactual-identity contract."""
import copy
import math
from pathlib import Path

import numpy as np
import pytest

from experiments import batch73_singleton_impostor as m


def specimen(regime="exact_singleton", world="present"):
    return dict(id=f"dev_00_00_{world}", pair_id="dev_00_00",
                seed=731103, split="dev", regime=regime,
                world=world, nframes=m.FRAMES, query_time=0)


def test_grouped_specs_disjoint_worlds_splits_and_seeds():
    rows = m.specs()
    assert len(rows) == len(m.REGIMES)*2*(m.DEV_PAIRS+m.TEST_PAIRS)
    assert sum(s["split"] == "test" for s in rows) == 2*len(m.REGIMES)*m.TEST_PAIRS
    assert len(set(s["pair_id"] for s in rows)) == len(rows)//2
    assert len(set(s["seed"] for s in rows)) == len(rows)//2
    for pair_id in set(s["pair_id"] for s in rows):
        a, b = [x for x in rows if x["pair_id"] == pair_id]
        assert (a["world"], b["world"]) == m.WORLDS
        assert a["seed"] == b["seed"]


@pytest.mark.parametrize("regime", m.REGIMES)
def test_paired_pixels_bit_identical_but_private_truth_disagrees(regime):
    pa, a = m.render(specimen(regime, "present"))
    pb, b = m.render(specimen(regime, "absent"))
    assert np.array_equal(pa, pb)
    assert a["xy"][0] == b["xy"][0]
    assert a["visible"][:m.OCC_STOP] == b["visible"][:m.OCC_STOP]
    assert all(a["visible"][m.OCC_STOP:])
    assert not any(b["visible"][m.OCC_STOP:])
    assert all(x is None for x in b["xy"][m.OCC_STOP:])
    assert np.all(pa >= 0) and pa.dtype == np.uint8
    assert pa.shape == (96, 256, 256)


@pytest.mark.parametrize("regime", m.REGIMES)
def test_generator_is_deterministic_and_query_visible(regime):
    x, a = m.render(specimen(regime))
    y, b = m.render(specimen(regime))
    assert np.array_equal(x, y)
    assert a == b
    assert a["visible"][0]
    assert a["phase"][0] == "pre"
    assert a["phase"][17] == "pre"
    assert a["phase"][18] == "opaque"
    assert a["phase"][63] == "opaque"
    assert a["phase"][64] == "post"
    assert not any(a["visible"][18:64])


@pytest.mark.parametrize("regime", m.REGIMES)
def test_counterfactual_template_predictions_identical(regime):
    frames_a, a = m.render(specimen(regime, "present"))
    frames_b, b = m.render(specimen(regime, "absent"))
    for margin in (True, False):
        ta = m.old.identity_match(list(frames_a), a["xy"][0], margin)
        tb = m.old.identity_match(list(frames_b), b["xy"][0], margin)
        assert ta == tb


def test_exact_singleton_can_fool_uniqueness_margin():
    frames, a = m.render(specimen())
    trace = m.old.identity_match(list(frames), a["xy"][0], True)
    assert sum(p is not None for p in trace[m.OCC_STOP:]) >= 28
    _, absent = m.render(specimen("exact_singleton", "absent"))
    result = m.score_trace(trace, absent)
    assert result["post_absent_accepted"] >= 28
    assert result["post_present"] == 0
    assert result["post_absent"] == 32
    assert result["longest_post_absent_error"] >= 28


def test_scoring_visible_vs_absent_distinct():
    _, a = m.render(specimen())
    _, b = m.render(specimen("exact_singleton", "absent"))
    correct = [[float(v) for v in x] if x is not None else None for x in a["xy"]]
    # Imagined always-positive tracker: correctly located in present world
    # but becomes an identity error in the pixel-identical absent world.
    for t in range(m.OCC_START, m.OCC_STOP):
        correct[t] = [120., 118.]
    present = m.score_trace(correct, a)
    absent = m.score_trace(correct, b)
    assert present["pre_correct"] == 17
    assert present["post_present_correct"] == 32
    assert present["hidden_accepted"] == 46
    assert absent["post_absent_accepted"] == 32
    assert absent["post_present_correct"] == 0
    assert absent["longest_post_absent_error"] == 32
    assert absent["post_absent_accepted"] == present["post_present_accepted"]


def test_clean_abstention_is_not_reacquisition():
    _, truth = m.render(specimen())
    trace = [truth["xy"][0]] + [None]*(m.FRAMES-1)
    stats = m.score_trace(trace, truth)
    assert stats["post_present_correct"] == 0
    assert stats["post_present_accepted"] == 0
    assert stats["first_correct_delay"] is None
    assert stats["hidden_accepted"] == 0


def test_score_rejects_malformed_nan_predictions():
    _, truth = m.render(specimen())
    trace = [None]*m.FRAMES
    with pytest.raises(RuntimeError, match="LENGTH_DRIFT"):
        m.score_trace(trace[:-1], truth)
    trace[77] = [math.nan, 18.]
    with pytest.raises(RuntimeError, match="INVALID_TRACE"):
        m.score_trace(trace, truth)
    trace[77] = "not a pair"
    with pytest.raises(RuntimeError, match="INVALID_TRACE"):
        m.score_trace(trace, truth)


def test_score_rejects_oracle_phase_and_visibility_inconsistency():
    _, oracle = m.render(specimen())
    trace = [None]*m.FRAMES
    q = copy.deepcopy(oracle)
    q["phase"][19] = "post"
    with pytest.raises(RuntimeError, match="PHASE_DRIFT"):
        m.score_trace(trace, q)
    q = copy.deepcopy(oracle)
    q["visible"][66] = False
    with pytest.raises(RuntimeError, match="VISIBILITY_ORACLE_DRIFT"):
        m.score_trace(trace, q)
    q = copy.deepcopy(oracle)
    q["visible"][12] = False
    q["xy"][12] = None
    with pytest.raises(RuntimeError, match="PRE_VISIBILITY_DRIFT"):
        m.score_trace(trace, q)


def test_paired_bound_for_same_observable_predictions():
    _, a = m.render(specimen())
    _, b = m.render(specimen("exact_singleton", "absent"))
    trace = [[70.,118.]] + [None]*(m.FRAMES-1)
    for t in range(m.OCC_STOP, m.FRAMES):
        trace[t] = a["xy"][t]
    stats_a = m.score_trace(trace, a)
    stats_b = m.score_trace(trace, b)
    rows = [{ "metrics": {key: stats for key in m.METHODS}}
            for stats in (stats_a, stats_b)]
    for method, v in m.totals(rows).items():
        assert v["post_present_correct"] == 32
        assert v["post_absent_accepted"] == 32
        assert v["post_present_accepted"] == 32
        assert 2*v["post_present_correct"] <= (
            v["post_present_accepted"]+v["post_absent_accepted"])


def test_manifest_rejects_seed_world_and_hash_drift():
    man = dict(kind=m.KIND, protocol=m.PROTOCOL, source_sha="frozen",
               scenes=[{**s, "frames_sha256": "0"*64, "query_xy": [70.,118.]}
                       for s in m.specs()])
    m.assert_manifest(man)
    man["scenes"][0]["seed"] += 1
    with pytest.raises(RuntimeError, match="SPLIT_SEED_OR_WORLD_DRIFT"):
        m.assert_manifest(man)
    man["scenes"][0]["seed"] -= 1
    man["scenes"][0]["world"] = "absent"
    with pytest.raises(RuntimeError, match="SPLIT_SEED_OR_WORLD_DRIFT"):
        m.assert_manifest(man)
    man["scenes"][0]["world"] = "present"
    man["scenes"][0]["frames_sha256"] = "broken"
    with pytest.raises(RuntimeError, match="PUBLIC_PIN_MISSING"):
        m.assert_manifest(man)


def test_split_validation():
    with pytest.raises(ValueError, match="INVALID_SPLIT"):
        m.predict("holdout")
    with pytest.raises(ValueError, match="INVALID_SPLIT"):
        m.score("holdout")


def test_frozen_prediction_stage_never_opens_private_oracle(tmp_path, monkeypatch):
    spec = specimen()
    frames, truth = m.render(spec)
    row = {**spec, "query_xy": truth["xy"][0],
           "frames_sha256": "a"*64}
    monkeypatch.setattr(m, "specs", lambda: [spec])
    m.b.save_json(tmp_path/"manifest.json",
                  dict(kind=m.KIND, protocol=m.PROTOCOL,
                       source_sha="test", config_sha256="test",
                       scenes=[row]))
    monkeypatch.setattr(m, "guard", lambda man: None)
    monkeypatch.setattr(m, "load_frames", lambda data, row: list(frames))
    monkeypatch.setattr(m.b.core, "estimate_affines",
                        lambda frames, cfg: ([None]*len(frames), [], None))
    monkeypatch.setattr(m.b.core, "predict_query",
                        lambda frames, q, aff, cfg:
                        ({k: [None]*len(frames) for k in m.METHODS}, {}))
    # Single-world fixture deliberately triggers paired-world assertion
    # AFTER tracing; unauthorized private/reads would fail first.
    orig_open = Path.open
    def denied(self, *args, **kwargs):
        if "private" in self.parts:
            raise AssertionError("PRIVATE_LABEL_READ_IN_PREDICT")
        return orig_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(RuntimeError, match="PIXEL_IDENTICAL_TRACE_DRIFT"):
        m.predict("dev", tmp_path, tmp_path/"output")


def test_write_once_and_hash_tampering(tmp_path):
    p = tmp_path/"fixed.json"
    m.b.save_json(p, {"v": 1})
    with pytest.raises(FileExistsError):
        m.b.save_json(p, {"v": 2})
    p.write_text('{"v":2}')
    with pytest.raises(RuntimeError, match="HASH_DRIFT"):
        m.b.read_pinned(p)
