"""Batch75 read-only DEV reacquisition audit regressions."""
from __future__ import annotations

import copy
import json

import pytest

from experiments import batch75_dev_reacquisition_audit as m


def example():
    occ = [False, True, True, False, False, True, False, False, False]
    xy = [[0., 0.] for _ in occ]
    pred = [None, None, [1., 1.], [30., 0.], [2., 0.],
            None, None, [1., 1.], [3., 1.]]
    accepted = [False, False, True, True, True, False, False, True, True]
    return occ, xy, pred, accepted


def test_disjoint_recovery_events_and_wrong_first_accept():
    es = m.post_occlusion_events(*example(), 0)
    assert len(es) == 2
    a, b = es
    assert (a["start"], a["end_occlusion_exclusive"], a["occluded_frames"]) == (1, 3, 2)
    assert a["visible_window_frames"] == 2
    assert a["occluded_false_accepted"] == 1
    assert a["first_accepted_status"] == "wrong"
    assert a["first_correct_lag_frames"] == 1
    assert a["correct_post_visible_frames"] == 1
    assert a["wrong_post_visible_frames"] == 1
    assert a["longest_wrong_visible_run"] == 1
    assert b["start"] == 5
    assert b["visible_window_frames"] == 3
    assert b["first_correct_lag_frames"] == 1
    assert b["first_accepted_status"] == "correct"
    assert b["wrong_post_visible_frames"] == 0


def test_rates_and_denominators():
    d = m.summarize(m.aggregate(m.post_occlusion_events(*example(), 0)))
    assert d["occlusion_events"] == 2
    assert d["reacquisition_evaluable_events"] == 2
    assert d["reacquired_events"] == 2
    assert d["first_accept_wrong_events"] == 1
    assert d["false_accepted_occluded_frames"] == 1
    assert d["post_visible_frames"] == 5
    assert d["accepted_post_visible_frames"] == 4
    assert d["correct_post_visible_frames"] == 3
    assert d["wrong_post_visible_frames"] == 1
    assert d["post_visible_coverage"] == pytest.approx(4 / 5)
    assert d["post_visible_correct_pck8"] == pytest.approx(3 / 5)
    assert d["post_visible_wrong_accept_fraction"] == pytest.approx(1 / 4)
    assert d["event_any_correct_recovery_rate"] == 1.0
    assert d["recovered_event_median_lag_frames"] == 1.0


def test_right_censored_is_not_false_reacquisition_failure():
    occ = [False, False, True, True]
    pred = [None] * 4
    acc = [False] * 4
    es = m.post_occlusion_events(occ, [None]*4, pred, acc, 0)
    assert len(es) == 1
    assert es[0]["right_censored"] is True
    d = m.summarize(m.aggregate(es))
    assert d["right_censored_events"] == 1
    assert d["reacquisition_evaluable_events"] == 0
    assert d["event_any_correct_recovery_rate"] is None
    assert d["post_visible_coverage"] is None


def test_no_occlusion_is_unsupported_not_perfect_safety():
    es = m.post_occlusion_events([False]*5, [None]*5, [None]*5,
                                 [False]*5, 0)
    assert es == []
    d = m.summarize(m.aggregate(es))
    assert d["event_any_correct_recovery_rate"] is None
    assert d["post_visible_wrong_accept_fraction"] is None
    assert d["reacquisition_evaluable_events"] == 0


def test_first_accepted_wrong_but_later_correct_counts_both():
    occ = [False, True, False, False]
    gt = [None, None, [0., 0.], [0., 0.]]
    pred = [None, None, [20., 0.], [0., 0.]]
    es = m.post_occlusion_events(occ, gt, pred,
                                 [False, False, True, True], 0)
    e = es[0]
    assert e["first_accepted_status"] == "wrong"
    assert e["first_correct_lag_frames"] == 1
    assert m.aggregate(es)["reacquired_events"] == 1


def test_incorrect_accepted_visible_is_not_called_identity_switch():
    occ = [False, True, False, False]
    gt = [None, None, [0., 0.], [0., 0.]]
    pred = [None, None, [20., 0.], [21., 0.]]
    es = m.post_occlusion_events(occ, gt, pred,
                                 [False, False, True, True], 0)
    d = m.summarize(m.aggregate(es))
    assert d["first_accept_wrong_events"] == 1
    assert d["reacquired_events"] == 0
    assert d["worst_wrong_visible_run"] == 2


def test_acceptance_gap_resets_wrong_streak():
    es = m.post_occlusion_events(
        [False, True, False, False, False],
        [None, None, [0., 0.], [0., 0.], [0., 0.]],
        [None, None, [20., 0.], None, [22., 0.]],
        [False, False, True, False, True], 0)
    assert es[0]["longest_wrong_visible_run"] == 1


def test_query_frame_never_counted():
    occ = [True, True, False]
    es = m.post_occlusion_events(occ, [None]*3, [None]*3,
                                 [True, False, False], 0)
    assert len(es) == 1
    assert es[0]["occluded_frames"] == 1
    assert es[0]["occluded_false_accepted"] == 0


@pytest.mark.parametrize("t0", [True, -1, 9, 0.0, None])
def test_invalid_query_time_rejected(t0):
    o, g, p, a = example()
    with pytest.raises(RuntimeError, match="BATCH75_INVALID_TRACE_LENGTH"):
        m.post_occlusion_events(o, g, p, a, t0)


@pytest.mark.parametrize("radius", [0, -1, float("nan"), float("inf"), True, "8"])
def test_invalid_radius_rejected(radius):
    with pytest.raises(RuntimeError, match="BATCH75_INVALID_RADIUS"):
        m.post_occlusion_events(*example(), 0, radius)


@pytest.mark.parametrize("field", ["occluded", "ground_truth", "predictions", "accepted"])
def test_mismatched_trace_lengths_rejected(field):
    o, g, p, a = example()
    mapp = {"occluded": o, "ground_truth": g, "predictions": p, "accepted": a}
    mapp[field] = mapp[field][:-1]
    with pytest.raises(RuntimeError, match="BATCH75_INVALID_TRACE_LENGTH"):
        m.post_occlusion_events(**mapp, query_time=0)


@pytest.mark.parametrize("field", ["occluded", "accepted"])
@pytest.mark.parametrize("invalid", [1, 0, None, 0.5])
def test_non_boolean_rejected(field, invalid):
    o, g, p, a = example()
    q = o if field == "occluded" else a
    q[3] = invalid
    with pytest.raises(RuntimeError, match="BATCH75_BOOLEAN_TRACE_REQUIRED"):
        m.post_occlusion_events(o, g, p, a, 0)


def test_accepted_null_prediction_rejected():
    o, g, p, a = example()
    p[3] = None
    with pytest.raises(RuntimeError, match="BATCH75_ACCEPTED_NULL_PREDICTION"):
        m.post_occlusion_events(o, g, p, a, 0)


@pytest.mark.parametrize("bad", [None, [float("nan"), 2.], [1], "bad"])
def test_accepted_visible_bad_location_rejected(bad):
    o, g, p, a = example()
    g[3] = bad
    with pytest.raises(RuntimeError):
        m.post_occlusion_events(o, g, p, a, 0)


def test_results_are_not_saved_with_rewrite(tmp_path):
    f = tmp_path / "audit.json"
    x = {"split": "dev", "value": 1}
    sha = m.write_immutable(f, x)
    assert len(sha) == 64
    assert f.with_suffix(".json.sha256").read_text().strip() == sha
    with pytest.raises(RuntimeError, match="BATCH75_OUTPUT_ALREADY_EXISTS"):
        m.write_immutable(f, x)
    assert json.loads(f.read_text()) == x


def test_existing_pin_without_data_fails_closed(tmp_path):
    f = tmp_path / "report.json"
    f.with_suffix(".json.sha256").write_text("deadbeef")
    with pytest.raises(RuntimeError, match="BATCH75_OUTPUT_ALREADY_EXISTS"):
        m.write_immutable(f, {"a": 2})


def test_source_declares_dev_only_and_has_no_future_test_scoring():
    assert m.VARIANTS == ("baseline", "template_4px", "affine_3px", "both_4px_3px")
    assert m.RADIUS == 8.0
    assert m.base.SCENES[-1] == "kinetics_35"


def test_real_historical_dev_replay_counts_when_present():
    # Real labels from HISTORICAL DEV only. This is not a new TEST evaluation.
    if not (m.base.IN / "dev_predictions.json").exists():
        pytest.skip("historical frozen DEV prediction data unavailable")
    report = m.load_and_score()
    assert report["split"] == "dev"
    assert report["hidden_test_oracle_read"] is False
    assert report["test_score_executions"] == 0
    assert report["queries"] == 144
    for variant, expected in (
        ("baseline", 317),
        ("template_4px", 162),
        ("affine_3px", 294),
        ("both_4px_3px", 159),
    ):
        assert report["overall"][variant]["false_accepted_occluded_frames"] == expected
        assert report["overall"][variant]["occlusion_events"] == 214
