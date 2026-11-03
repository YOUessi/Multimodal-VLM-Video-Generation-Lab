"""Regression coverage for Batch79's DEV-only, immutable episode audit."""
import hashlib
import json
import math
from pathlib import Path
import pytest

from experiments import batch79_fresh_dev_reacquisition as m


@pytest.mark.parametrize("bad", [True, 1, "x", [], [1], [1, 2, 3],
                                    ["1", 2], [True, 2],
                                    [float("nan"), 0], [float("inf"), 2], {}])
def test_bad_points_rejected(bad):
    with pytest.raises(RuntimeError, match="BAD_POINT"):
        m.xy(bad)


@pytest.mark.parametrize("value", [None, [1, 2], [1.0, -2.3]])
def test_valid_points(value):
    assert m.xy(value) == value


def test_pinned_good_and_corrupt(tmp_path):
    p = tmp_path / "dev_predictions.json"
    raw = b'{"split":"dev"}\n'
    p.write_bytes(raw)
    p.with_suffix(".json.sha256").write_text(m.digest(raw) + "\n")
    assert m.pinned(p)["split"] == "dev"
    p.write_bytes(b"{}")
    with pytest.raises(RuntimeError, match="PIN_DRIFT"):
        m.pinned(p)


def test_missing_pin(tmp_path):
    (tmp_path / "dev_results.json").write_text("{}")
    with pytest.raises(RuntimeError, match="MISSING_PIN"):
        m.pinned(tmp_path / "dev_results.json")


@pytest.mark.parametrize("path", ["test_predictions", "test_results", "TEST_oracle"])
def test_forbidden_test_path(tmp_path, path):
    d = tmp_path / path
    d.mkdir()
    with pytest.raises(RuntimeError, match="TEST_OR_MISSING"):
        m.load(d, tmp_path)


def test_missing_input_path(tmp_path):
    with pytest.raises(RuntimeError, match="TEST_OR_MISSING"):
        m.load(tmp_path / "missing", tmp_path)


def test_write_once_is_immutable(tmp_path):
    p = tmp_path / "nested/result.json"
    sha = m.write_once(p, {"source": "dev"})
    assert sha == hashlib.sha256(p.read_bytes()).hexdigest()
    assert p.with_suffix(".json.sha256").read_text().strip() == sha
    with pytest.raises(RuntimeError, match="ALREADY_EXISTS"):
        m.write_once(p, {"source": "dev"})


def _summary(recovered, occluded, false, events=5):
    return {"reacquisition_evaluable_events": events,
            "reacquired_events": recovered, "occluded_frames": occluded,
            "false_accepted_occluded_frames": false}


def test_scene_summary_weights_occlusion_frames():
    rows = {"scene1": {"klt_fb": _summary(2, 10, 5, events=2)},
            "scene2": {"klt_fb": _summary(1, 90, 0, events=3)}}
    value = m.scene_sum(rows, "klt_fb", ["scene1", "scene2"])
    assert value["recovery_rate"] == pytest.approx(3 / 5)
    assert value["false_accept_rate"] == pytest.approx(5 / 100)
    assert value["worst_scene_rate"] == 0.5
    assert value["violating_scenes"] == 1


def test_zero_support_not_pass():
    rows = {"s": {"klt_fb": _summary(0, 0, 0, events=0)}}
    value = m.scene_sum(rows, "klt_fb", ["s"])
    assert value["recovery_rate"] is None
    assert value["false_accept_rate"] is None
    assert value["worst_scene_rate"] is None


def test_bootstrap_seed_reproduction():
    rows = {f"s{i}": {
        "klt_fb": _summary(i % 2, 10, i % 3),
        "appearance_fb_055": _summary(0, 10, 0)} for i in range(8)}
    a = m.scene_interval(rows, "klt_fb", "appearance_fb_055", n=30)
    b = m.scene_interval(rows, "klt_fb", "appearance_fb_055", n=30)
    assert a == b
    assert a["unit"] == "scene"
    assert len(a["recovery_rate_paired_95_pct"]) == 2


@pytest.mark.parametrize("count", [-1, 0, 1.5, True])
def test_bad_bootstrap_count(count):
    rows = {"s": {m.METHODS[0]: _summary(1, 10, 1),
                  m.METHODS[1]: _summary(0, 10, 0)}}
    with pytest.raises(RuntimeError, match="BOOTSTRAP_COUNT"):
        m.scene_interval(rows, m.METHODS[0], m.METHODS[1], n=count)


def _synthetic():
    public, oracle, records, per_scene = {}, {}, [], {}
    for scene in m.SCENES:
        public[scene] = []
        oracle[scene] = {"tracks": {}}
        rec = {"name": scene, "nframes": 8, "tracks": {}}
        per_scene[scene] = {}
        for qid in ("q0", "q1", "q2", "q3"):
            public[scene].append({"id": qid, "query_time": 0})
            occluded = [False, False, True, True, False, False, False, False]
            oracle[scene]["tracks"][qid] = {
                "occluded": occluded, "xy": [[10, 10]] * 8}
            rec["tracks"][qid] = {
                method: [[10, 10], [10, 10], None, None,
                         [10, 10], None, None, None]
                for method in m.METHODS}
        records.append(rec)
        per_scene[scene] = {method: {"invisible": 8,
                                    "invisible_false_accept": 0}
                            for method in m.METHODS}
    return ({"records": records, "source_commit": "frozen"},
            {"selected_method": "klt_fb"}, {"per_scene": per_scene},
            public, oracle)


def test_full_synthetic_36_scene_episode_replay():
    data = _synthetic()
    x = m.evaluate(*data, bootstraps=15)
    assert x["split"] == "dev"
    assert x["new_blind_test"] is False
    assert x["test_oracle_parsed"] is False
    assert x["scene_summary"]["klt_fb"]["events"] == 144
    assert x["scene_summary"]["klt_fb"]["recovered"] == 144
    assert x["scene_summary"]["klt_fb"]["false_accepted"] == 0
    assert x["overall"]["klt_fb"]["right_censored_events"] == 0
    assert x["paired_intervals"]["appearance_fb_055"]["unit"] == "scene"


def test_synthetic_frozen_replay_mismatch_fails():
    data = list(_synthetic())
    data[2]["per_scene"]["kinetics_07"]["klt_fb"]["invisible"] = 999
    with pytest.raises(RuntimeError, match="REPLAY_MISMATCH"):
        m.evaluate(*data, bootstraps=3)


def test_synthetic_dev_scene_missing_fails():
    data = list(_synthetic())
    data[4].pop("kinetics_35")
    with pytest.raises(RuntimeError, match="SCENE_DRIFT"):
        m.evaluate(*data, bootstraps=3)


def test_synthetic_query_mismatch_fails():
    data = list(_synthetic())
    data[3]["kinetics_00"].pop()
    with pytest.raises(RuntimeError, match="QUERY_DRIFT"):
        m.evaluate(*data, bootstraps=3)


def test_synthetic_invalid_trace_fails():
    data = list(_synthetic())
    data[0]["records"][0]["tracks"]["q0"]["klt_fb"][0] = None
    with pytest.raises(RuntimeError, match="TRACE_LENGTH_OR_QUERY"):
        m.evaluate(*data, bootstraps=3)


def test_synthetic_invalid_occlusion_fails():
    data = list(_synthetic())
    data[4]["kinetics_00"]["tracks"]["q0"]["occluded"][2] = 1
    with pytest.raises(RuntimeError, match="ORACLE_LENGTH"):
        m.evaluate(*data, bootstraps=3)
