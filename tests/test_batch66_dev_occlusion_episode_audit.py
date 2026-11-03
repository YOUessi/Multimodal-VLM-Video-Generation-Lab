"""Batch66 event-risk audit tests; synthetic traces are not independent TEST data."""
from __future__ import annotations

import copy
import json

import pytest

from experiments import batch66_dev_occlusion_episode_audit as m
from experiments import batch62_dev_consensus_veto as b62


def traces(occ, acc, query_time=0):
    return m.occlusion_episodes(occ, acc, query_time)


def test_simple_two_episodes_and_exact_indices():
    ep = traces(
        [False, True, True, True, False, True, True, False],
        [False, False, True, True, True, True, False, True])
    assert ep == [
        {"start": 1, "end_exclusive": 4, "occluded_frames": 3,
         "false_accepted_frames": 2, "longest_false_accept_run": 2,
         "first_false_accept_offset": 1},
        {"start": 5, "end_exclusive": 7, "occluded_frames": 2,
         "false_accepted_frames": 1, "longest_false_accept_run": 1,
         "first_false_accept_offset": 0},
    ]


def test_query_frame_is_excluded_even_if_accepted_and_occluded():
    ep = traces([True, True, True, False, True],
                [True, True, False, True, True], query_time=1)
    assert len(ep) == 2
    assert ep[0]["start"] == 2
    assert ep[0]["false_accepted_frames"] == 0
    assert ep[1]["start"] == 4
    assert ep[1]["false_accepted_frames"] == 1


def test_query_time_late_no_event():
    assert traces([True, True, False, False],
                  [True, True, False, True], query_time=2) == []


def test_trailing_ongoing_occlusion_is_closed():
    ep = traces([False, False, True, True, True],
                [False, False, True, False, True])
    assert ep[0]["end_exclusive"] == 5
    assert ep[0]["longest_false_accept_run"] == 1
    assert ep[0]["false_accepted_frames"] == 2


def test_accepted_visible_frames_do_not_count_as_false_accept():
    ep = traces([False, False, True, False],
                [False, True, False, True])
    assert len(ep) == 1
    assert ep[0]["false_accepted_frames"] == 0
    assert ep[0]["first_false_accept_offset"] is None


@pytest.mark.parametrize("occluded,accepted,t0", [
    ([True], [True], -1),
    ([True, False], [True], 0),
    ([True], [True], 1),
    ([True, False], [True, False], True),
])
def test_invalid_length_and_query_time_rejected(occluded, accepted, t0):
    with pytest.raises(RuntimeError, match="BATCH66_QUERY_LENGTH_INVALID"):
        traces(occluded, accepted, t0)


@pytest.mark.parametrize("occ,acc", [
    ([False, 1], [False, True]),
    ([False, None], [False, True]),
    ([False, True], [False, 1]),
    ([False, True], [False, None]),
    ([False, True], [False, 0.0]),
])
def test_non_boolean_occlusion_and_acceptance_rejected(occ, acc):
    with pytest.raises(RuntimeError, match="BATCH66_BOOLEAN_TRACE_REQUIRED"):
        traces(occ, acc)


def test_one_long_episode_fragmented_runs_do_not_fake_long_run():
    occ = [False] + [True] * 65
    accepted = [False] + [i % 2 == 0 for i in range(65)]
    ep = traces(occ, accepted)
    assert ep[0]["false_accepted_frames"] == 33
    assert ep[0]["longest_false_accept_run"] == 1
    c = m.episode_totals(ep)
    assert c["events_with_run_ge1"] == 1
    assert c["events_with_run_ge3"] == 0
    assert c["events_with_run_ge30"] == 0
    assert c["false_accept_frames_in_events_with_run_ge30"] == 0


@pytest.mark.parametrize("length", [1, 3, 10, 30, 60, 117])
def test_threshold_definitions_exact(length):
    ep = traces([False] + [True]*length, [False] + [True]*length)
    row = m.episode_totals(ep)
    assert row["false_accepted_frames"] == length
    assert row["worst_false_accept_run"] == length
    for th in m.RUN_THRESHOLDS:
        assert row[f"events_with_run_ge{th}"] == (1 if length >= th else 0)


def test_empty_events_are_unsupported_not_implicitly_safe():
    row = m.episode_totals([])
    assert row["occlusion_events"] == 0
    assert row["false_accepted_frames"] == 0
    assert row["worst_false_accept_run"] == 0
    assert row["affected_events"] == 0


def test_aggregation_sums_events_takes_maximum_run():
    a = m.episode_totals(traces([False, True, True], [False, True, True]))
    b = m.episode_totals(traces([False, True, True, True],
                                [False, True, False, True]))
    c = m.sum_counts([a, b])
    assert c["occlusion_events"] == 2
    assert c["occluded_frames"] == 5
    assert c["false_accepted_frames"] == 4
    assert c["affected_events"] == 2
    assert c["worst_false_accept_run"] == 2


def test_aggregation_impossible_frames_rejected():
    row = m.episode_totals([])
    row["false_accepted_frames"] = 2
    with pytest.raises(RuntimeError, match="BATCH66_IMPOSSIBLE_COUNTS"):
        m.sum_counts([row])


def test_aggregation_impossible_events_rejected():
    row = m.episode_totals([])
    row["affected_events"] = 2
    with pytest.raises(RuntimeError, match="BATCH66_IMPOSSIBLE_EVENTS"):
        m.sum_counts([row])


def synthetic_track(n=6):
    # Two trackers agree only at t1/t3. Base always accepts.
    base_track = [None] + [[100.0, 100.0] for _ in range(n-1)]
    template = [None, [101., 100.], [110., 100.], [102., 100.],
                None, [100., 100.]]
    affine = [None, [100., 101.], None, [101., 101.],
              [101., 101.], None]
    return {b62.BASE: base_track, "template_recover4": template,
            "affine_stabilized_klt": affine}


def test_synthetic_consensus_reduces_event_acceptance():
    n = 6
    occ = [False, True, True, True, True, True]
    by = m.make_query_events(synthetic_track(), occ, 0, n)
    assert m.episode_totals(by["baseline"])["false_accepted_frames"] == 5
    assert m.episode_totals(by["template_4px"])["false_accepted_frames"] == 3
    assert m.episode_totals(by["both_4px_3px"])["false_accepted_frames"] == 2


def test_track_length_fails_closed():
    tr = synthetic_track()
    del tr["template_recover4"]
    with pytest.raises(RuntimeError, match="BATCH66_PREDICTION_TRACK_LENGTH"):
        m.make_query_events(tr, [False]*6, 0, 6)


def test_predictions_before_query_rejected():
    tr = synthetic_track()
    tr[b62.BASE][0] = [100., 100.]
    with pytest.raises(RuntimeError, match="BATCH66_PREQUERY_PREDICTION"):
        m.make_query_events(tr, [False]*6, 1, 6)


def test_fixed_variants_and_thresholds():
    assert m.VARIANTS == tuple(b62.GATES)
    assert m.RUN_THRESHOLDS == (1, 3, 10, 30, 60)
    assert m.BUDGET == 0.1


def test_output_pin_and_no_overwrite(tmp_path):
    path = tmp_path / "report.json"
    digest = m.write_immutable(path, {"kind": "dev", "value": 42})
    assert digest == b62.sha(path.read_bytes())
    assert path.with_suffix(".json.sha256").read_text().strip() == digest
    assert json.loads(path.read_text())["value"] == 42
    with pytest.raises(RuntimeError, match="BATCH66_NO_OVERWRITE"):
        m.write_immutable(path, {"value": 1})


def test_output_existing_pin_also_blocks_overwrite(tmp_path):
    path = tmp_path / "report.json"
    path.with_suffix(".json.sha256").write_text("old")
    with pytest.raises(RuntimeError, match="BATCH66_NO_OVERWRITE"):
        m.write_immutable(path, {})


def test_output_rejects_non_json_nan(tmp_path):
    path = tmp_path / "report.json"
    with pytest.raises(ValueError):
        m.write_immutable(path, {"invalid": float("nan")})
    assert not path.exists()


def test_frozen_dev_integrity_and_private_test_firewall():
    # Only the pinned DEV artifacts are parsed. The oracle streaming reader
    # stops before the first TEST key; any TEST fields remain unopened.
    report = m.load_and_score()
    assert report["split"] == "dev"
    assert report["dev_scene_count"] == 36
    assert report["dev_query_count"] == 144
    assert report["oracle_dev_only_streamed_through"] == "kinetics_35"
    assert report["test_oracle_read"] is False
    assert report["test_predictions_read"] is False
    assert len(report["per_scene"]) == 36
    for key in m.VARIANTS:
        row = report["summary"][key]
        assert row["occluded_frames"] >= row["false_accepted_frames"]
        assert row["scene_supported_occlusion"] == 33
        assert row["scene_budget_violation_count"] <= 33
    assert report["summary"]["baseline"]["scene_budget_violation_count"] == 4
    assert report["summary"]["template_4px"]["scene_budget_violation_count"] == 1
    assert report["summary"]["baseline"]["worst_false_accept_run"] == 117
    assert report["summary"]["template_4px"]["worst_false_accept_run"] == 117
