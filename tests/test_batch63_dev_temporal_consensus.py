"""Batch63 causal temporal-veto, paired-risk and DEV-firewall regressions."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import batch62_dev_consensus_veto as b62
from experiments import batch63_dev_temporal_consensus as b63


def trace(evidence=(True, True, True, True, True), affine=None):
    if affine is None:
        affine = evidence
    return {
        b62.BASE: [[2, 3] for _ in evidence],
        "template_recover4": [[2, 3] if good else [30, 30] for good in evidence],
        "affine_stabilized_klt": [[2, 3] if good else [30, 30] for good in affine],
    }


@pytest.mark.parametrize("window,expected", [
    (1, [False, True, True, True, True]),
    (2, [False, False, True, True, True]),
    (3, [False, False, False, True, True]),
    (4, [False, False, False, False, True]),
])
def test_causal_delay_no_query_credit(window, expected):
    tr = trace()
    assert b63.causal_flags(tr, 0, 5, "template_4px", window) == expected


def test_query_time_does_not_give_free_evidence():
    tr = trace()
    for method in tr:
        tr[method][:2] = [None, None]
    assert b63.causal_flags(tr, 2, 5, "template_4px", 2) == [
        False, False, False, False, True
    ]


def test_gap_resets_streak():
    tr = trace((True, True, False, True, True, True))
    assert b63.causal_flags(tr, 0, 6, "template_4px", 2) == [
        False, False, False, False, True, True
    ]


def test_single_frame_replays_existing_gate():
    tr = trace((True, False, True, False, True))
    got = b63.causal_flags(tr, 0, 5, "template_4px", 1)
    assert got == [False] + [b62.pass_veto(tr, i, "template_4px") for i in range(1, 5)]


def test_baseline_replays_original_when_supporter_missing():
    tr = trace((False, False, False, False, False))
    assert b63.causal_flags(tr, 0, 5, "baseline", 1) == [False, True, True, True, True]


def test_affine_and_dual_evidence():
    tr = trace((True, True, True, True, True), (True, False, True, True, True))
    assert b63.causal_flags(tr, 0, 5, "affine_3px", 2) == [
        False, False, False, True, True
    ]
    assert b63.causal_flags(tr, 0, 5, "both_4px_3px", 2) == [
        False, False, False, True, True
    ]


def test_missing_base_is_always_rejected():
    tr = trace()
    tr[b62.BASE][3] = None
    assert b63.causal_flags(tr, 0, 5, "template_4px", 2) == [
        False, False, True, False, False
    ]


def test_causal_no_future_leak():
    tr = trace((True, True, True, True, True))
    before = b63.causal_flags(tr, 0, 5, "template_4px", 2)
    tr["template_recover4"][4] = [99, 99]
    after = b63.causal_flags(tr, 0, 5, "template_4px", 2)
    assert after[:4] == before[:4]
    assert not after[4]


def test_trace_unmodified():
    tr = trace((True, True, False, True, True))
    before = json.dumps(tr, sort_keys=True)
    b63.causal_flags(tr, 0, 5, "template_4px", 2)
    assert json.dumps(tr, sort_keys=True) == before


@pytest.mark.parametrize("time,n", [(-1, 5), (5, 5), (4, 5), (0, 1), (True, 5),
                                      (0, True), (0, -5)])
def test_bad_query(time, n):
    with pytest.raises(RuntimeError, match="INVALID_QUERY"):
        b63.causal_flags(trace(), time, n, "template_4px", 2)


@pytest.mark.parametrize("evidence,window", [
    ("unknown", 2), ("template_4px", 0), ("template_4px", True),
    ("template_4px", -1), ("template_4px", 1.1)])
def test_bad_gate(evidence, window):
    with pytest.raises(RuntimeError, match="INVALID_GATE"):
        b63.causal_flags(trace(), 0, 5, evidence, window)


def test_bad_trace_shape():
    tr = trace()
    tr["template_recover4"].pop()
    with pytest.raises(RuntimeError, match="INVALID_TRACE"):
        b63.causal_flags(tr, 0, 5, "template_4px", 2)


def test_prequery_leak_rejected():
    tr = trace()
    with pytest.raises(RuntimeError, match="PREQUERY_LEAK"):
        b63.causal_flags(tr, 2, 5, "template_4px", 2)


def test_long_occlusion_false_accept_run():
    assert b63.false_accept_runs([False, True, True, True, False, True, True],
                                 [False, True, True, True, False, True, True], 0) == [3, 2]


def test_visibility_breaks_false_accept_run():
    assert b63.false_accept_runs([False, True, True, True],
                                 [False, True, False, True], 0) == [1, 1]


def test_episodes_reject_wrong_length():
    with pytest.raises(RuntimeError, match="EPISODE_SHAPE"):
        b63.false_accept_runs([True], [True, False], 0)


def test_episodes_reject_bad_visibility():
    with pytest.raises(RuntimeError, match="BAD_VISIBILITY"):
        b63.false_accept_runs([False, True], [False, 1], 0)


def toy(n=5):
    pred = {"name": "kinetics_00", "nframes": n,
            "tracks": {f"p{i}": trace((True,) * n) for i in range(4)}}
    q = [{"id": f"p{i}", "query_time": 0} for i in range(4)]
    oracle = {"tracks": {f"p{i}": {
        "xy": [[2, 3] for _ in range(n)],
        "occluded": [False, False, False, True, True]
    } for i in range(4)}}
    return pred, q, oracle


def test_score_temporal_veto_counts_and_episodes():
    pred, queries, oracle = toy()
    res = b63.score_scene(pred, queries, oracle)
    assert res["template_4px"]["accepted_visible"] == 8
    assert res["template_streak2"]["accepted_visible"] == 4
    assert res["template_streak3"]["accepted_visible"] == 0
    assert res["template_4px"]["invisible_false_accept"] == 8
    assert res["template_streak2"]["invisible_false_accept"] == 8
    assert res["template_streak2"]["max_false_accept_episode_frames"] == 2
    assert res["template_streak3"]["invisible_false_accept"] == 8


def test_score_fails_query_population():
    pred, queries, oracle = toy()
    with pytest.raises(RuntimeError, match="QUERY_POPULATION"):
        b63.score_scene(pred, queries[:-1], oracle)


def test_score_fails_oracle_length():
    pred, queries, oracle = toy()
    oracle["tracks"]["p0"]["occluded"].pop()
    with pytest.raises(RuntimeError, match="ORACLE_LENGTH"):
        b63.score_scene(pred, queries, oracle)


def test_score_fails_non_bool_visibility():
    pred, queries, oracle = toy()
    oracle["tracks"]["p0"]["occluded"][2] = 0
    with pytest.raises(RuntimeError, match="BAD_VISIBILITY"):
        b63.score_scene(pred, queries, oracle)


def test_percentile_boundary_and_interpolation():
    assert b63.percentile([2, 8, 4], 0) == 2
    assert b63.percentile([2, 8, 4], 1) == 8
    assert b63.percentile([2, 8, 4], .5) == 4


def test_percentile_empty():
    with pytest.raises(RuntimeError, match="EMPTY"):
        b63.percentile([], 0.5)


def test_bootstrap_deterministic_and_paired():
    scenes = {s: {
        "template_4px": {"visible_coverage": 0.5,
                          "invisible_false_accept_rate": 0.2 if i % 3 else None},
        "template_streak2": {"visible_coverage": 0.4,
                             "invisible_false_accept_rate": 0.1 if i % 3 else None},
    } for i, s in enumerate(b62.SCENES)}
    a = b63.paired_scene_bootstrap(scenes, "template_streak2")
    assert a == b63.paired_scene_bootstrap(scenes, "template_streak2")
    assert a["delta_scene_macro_visible_coverage"]["median"] == pytest.approx(-0.1)
    assert a["delta_scene_macro_invisible_false_accept_rate"]["median"] == pytest.approx(-0.1)


def test_real_frozen_dev_if_available():
    if not (b62.IN / "dev_predictions.json").is_file():
        pytest.skip("DEV data is only mounted on Tang")
    pred = b62.pinned_json(b62.IN / "dev_predictions.json")
    sel = b62.pinned_json(b62.IN / "dev_selection.json")
    raw = (b62.IN / "dev_results.json").read_bytes()
    assert b62.sha(raw) == sel["dev_results_sha256"]
    dev = json.loads(raw)
    queries = json.loads((b62.DATA / "queries-public.json").read_bytes())
    # Read the DEV prefix only. Never access TEST oracle records.
    oracle = b62.load_dev_only_oracle(b62.DATA / "oracle-private.json", b62.SCENES)
    res = b63.audit(pred, sel, dev, queries, oracle)
    assert res["dev_scene_count"] == 36
    assert res["test_oracle_read"] is False and res["test_predictions_read"] is False
    assert res["results"]["baseline"]["scene_budget_violation_count"] == 4
    assert res["results"]["template_streak2"]["micro"]["accepted_visible"] <= (
        res["results"]["template_4px"]["micro"]["accepted_visible"])
