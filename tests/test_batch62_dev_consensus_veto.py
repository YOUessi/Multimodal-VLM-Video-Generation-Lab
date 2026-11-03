"""Pure-function and DEV-firewall regressions for Batch62."""
import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("batch62_dev_consensus_veto",
                                              ROOT / "experiments/batch62_dev_consensus_veto.py")
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _trace():
    return {
        "appearance_fb_055": [[0, 0], [3, 4], None, [5, 5]],
        "template_recover4": [[0, 0], [4, 4], [3, 1], [20, 20]],
        "affine_stabilized_klt": [[0, 0], [3, 5], None, [5, 5]],
    }


def _sample(n=4):
    return {
        "name": "kinetics_00", "nframes": n,
        "tracks": {f"p{i}": _trace() for i in range(4)},
    }


def _queries():
    return [{"id": f"p{i}", "query_time": 0} for i in range(4)]


def _oracle():
    return {"tracks": {f"p{i}": {"xy": [[0, 0], [3, 4], [2, 1], [5, 5]],
                                "occluded": [False, False, True, True]}
                       for i in range(4)}}


def _source():
    pred = {
        "split": "dev", "source_commit": "a", "config_sha256": "c",
        "manifest_sha256": "m", "oracle_unread_during_prediction": True,
        "methods": ["appearance_fb_055", "template_recover4", "affine_stabilized_klt"],
        "records": [{"name": s} for s in mod.SCENES],
    }
    sel = {"split": "dev", "source_commit": "a", "config_sha256": "c",
           "selected_method": "appearance_fb_055"}
    dev = {"split": "dev", "source_commit": "a", "config_sha256": "c",
           "manifest_sha256": "m", "per_scene": {s: {} for s in mod.SCENES}}
    queries = {s: [] for s in mod.SCENES}
    return pred, sel, dev, queries


def test_xy_none():
    assert mod.xy(None) is None


def test_xy_numeric():
    assert mod.xy([3, 4]) == (3.0, 4.0)


@pytest.mark.parametrize("bad", [[1], [1, 2, 3], [float("nan"), 0],
                                   [float("inf"), 0], ["2", 3], [True, 1], {"x": 3}])
def test_xy_bad(bad):
    with pytest.raises(RuntimeError):
        mod.xy(bad)


def test_baseline_accept_when_supporters_absent():
    tr = _trace()
    tr["template_recover4"][1] = None
    assert mod.pass_veto(tr, 1, "baseline")
    assert not mod.pass_veto(tr, 1, "template_4px")


def test_template_veto_agrees():
    assert mod.pass_veto(_trace(), 1, "template_4px")


def test_affine_veto_agrees():
    assert mod.pass_veto(_trace(), 1, "affine_3px")


def test_combined_veto_agrees():
    assert mod.pass_veto(_trace(), 1, "both_4px_3px")


def test_template_veto_rejects_divergence():
    assert not mod.pass_veto(_trace(), 3, "template_4px")


def test_combined_veto_rejects_divergence():
    assert not mod.pass_veto(_trace(), 3, "both_4px_3px")


def test_missing_baseline_never_accepted():
    assert not any(mod.pass_veto(_trace(), 2, variant) for variant in mod.GATES)


def test_unknown_gate():
    with pytest.raises(RuntimeError, match="UNKNOWN"):
        mod.pass_veto(_trace(), 0, "magic")


def test_score_sample_counts_and_delta():
    scored = mod.score_one(_sample(), _queries(), _oracle())
    a, b, c = (scored[k] for k in ("baseline", "template_4px", "affine_3px"))
    # Four points, three times after query: visible at t=1, occluded t=2,3.
    assert a["visible"] == 4 and a["invisible"] == 8
    assert a["accepted_visible"] == 4 and a["correct8"] == 4
    assert a["invisible_false_accept"] == 4
    assert b["invisible_false_accept"] == 0
    assert c["invisible_false_accept"] == 4


def test_score_does_not_infer_unknown_visibility():
    gt = _oracle()
    gt["tracks"]["p0"]["occluded"][2] = 1
    with pytest.raises(RuntimeError, match="VISIBILITY"):
        mod.score_one(_sample(), _queries(), gt)


def test_score_rejects_missing_query():
    with pytest.raises(RuntimeError, match="QUERY_ID"):
        mod.score_one(_sample(), _queries()[:-1], _oracle())


def test_score_rejects_trace_length():
    sample = _sample()
    sample["tracks"]["p0"]["template_recover4"].pop()
    with pytest.raises(RuntimeError, match="TRACK_LENGTH"):
        mod.score_one(sample, _queries(), _oracle())


def test_score_rejects_prequery_data():
    sample = _sample()
    queries = _queries()
    queries[0]["query_time"] = 1
    with pytest.raises(RuntimeError, match="PREQUERY"):
        mod.score_one(sample, queries, _oracle())


def test_rate_empty_sentinel():
    val = mod.rates(mod.counts_empty())
    assert val["visible_coverage"] is None
    assert val["invisible_false_accept_rate"] is None
    assert val["jaccard_mean_1_2_4_8_16"] is None


def test_rate_denominator_and_jaccard():
    counts = mod.counts_empty()
    counts.update(visible=5, invisible=10, accepted_visible=3,
                  correct8=2, predicted_visible=4, invisible_false_accept=1)
    for radius in mod.RADII:
        counts[f"tp{radius}"] = 2
    res = mod.rates(counts)
    assert res["visible_coverage"] == 0.6
    assert res["invisible_false_accept_rate"] == 0.1
    assert res["accepted_precision8"] == 2 / 3
    assert res["jaccard_mean_1_2_4_8_16"] == 2 / 7


def test_add_preserves_raw_counts():
    one = mod.counts_empty()
    two = mod.counts_empty()
    one["visible"] = 4
    two["visible"] = 5
    assert mod.add([one, two])["visible"] == 9


def test_validate_baseline():
    mod.validate_sources(*_source())


def test_validate_rejects_test_split():
    a, b, c, d = _source()
    a["split"] = "test"
    with pytest.raises(RuntimeError, match="DEV_ONLY"):
        mod.validate_sources(a, b, c, d)


def test_validate_rejects_not_predicted_blind():
    a, b, c, d = _source()
    a["oracle_unread_during_prediction"] = False
    with pytest.raises(RuntimeError, match="DEV_ONLY"):
        mod.validate_sources(a, b, c, d)


def test_validate_rejects_selection_drift():
    a, b, c, d = _source()
    b["selected_method"] = "klt_fb"
    with pytest.raises(RuntimeError, match="SELECTION"):
        mod.validate_sources(a, b, c, d)


def test_validate_rejects_manifest_drift():
    a, b, c, d = _source()
    a["manifest_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="SOURCE_PIN"):
        mod.validate_sources(a, b, c, d)


def test_validate_rejects_supporter_missing():
    a, b, c, d = _source()
    a["methods"].remove("template_recover4")
    with pytest.raises(RuntimeError, match="SUPPORT_METHOD"):
        mod.validate_sources(a, b, c, d)


def test_pin_good(tmp_path):
    p = tmp_path / "t.json"
    data = b'{"split":"dev"}\n'
    p.write_bytes(data)
    (tmp_path / "t.json.sha256").write_text(mod.sha(data) + "\n")
    assert mod.pinned_json(p) == {"split": "dev"}


def test_pin_tampered(tmp_path):
    p = tmp_path / "t.json"
    p.write_text('{"split":"test"}')
    (tmp_path / "t.json.sha256").write_text(mod.sha(b'{"split":"dev"}'))
    with pytest.raises(RuntimeError, match="PIN_DRIFT"):
        mod.pinned_json(p)


def test_pin_missing(tmp_path):
    p = tmp_path / "t.json"
    p.write_text("{}")
    with pytest.raises(RuntimeError, match="PIN_MISSING"):
        mod.pinned_json(p)


def test_oracle_prefix_parser_does_not_read_test(tmp_path):
    p = tmp_path / "oracle-private.json"
    records = '{\n' + "".join(
        '  "' + scene + '": {\n    "tracks": {}\n  },\n'
        for scene in mod.SCENES
    )
    # Deliberately corrupt TEST block: parser must terminate BEFORE it.
    records += '  "kinetics_36": DO_NOT_PARSE_ANY_TEST_TRUTH\n'
    p.write_text(records)
    result = mod.load_dev_only_oracle(p, mod.SCENES)
    assert list(result) == mod.SCENES


def test_oracle_parser_rejects_bad_order(tmp_path):
    p = tmp_path / "oracle-private.json"
    p.write_text('{\n  "kinetics_01": {\n    "tracks": {}\n  },\n')
    with pytest.raises(RuntimeError, match="ORACLE_ORDER"):
        mod.load_dev_only_oracle(p, mod.SCENES)


def test_oracle_parser_rejects_population_drift(tmp_path):
    with pytest.raises(RuntimeError, match="POPULATION"):
        mod.load_dev_only_oracle(tmp_path / "nonexistent", mod.SCENES[:-1])


def test_baseline_replay_with_frozen_dev_if_available():
    pred_p = mod.IN / "dev_predictions.json"
    if not pred_p.exists():
        pytest.skip("Frozen local DEV files unavailable in CI")
    pred = mod.pinned_json(pred_p)
    sel = mod.pinned_json(mod.IN / "dev_selection.json")
    dev = json.loads((mod.IN / "dev_results.json").read_text())
    queries = json.loads((mod.DATA / "queries-public.json").read_text())
    oracle = mod.load_dev_only_oracle(mod.DATA / "oracle-private.json", mod.SCENES)
    result = mod.evaluate(pred, sel, dev, queries, oracle)
    assert result["results"]["baseline"]["scene_budget_violation_count"] == 4
    assert not result["test_predictions_read"]
    assert not result["test_oracle_read"]


def test_cpu_component_proxy_with_affine_amortization():
    rec = _sample()
    rec["cpu_query_ms"] = {f"p{i}": {
        "appearance_fb_055": 3.0,
        "template_recover4": 6.0,
        "affine_stabilized_klt": 9.0,
    } for i in range(4)}
    pred = {"records": [rec], "scene_aux": {
        "kinetics_00": {"affine_registration_ms_per_video": 8.0}}}
    result = mod.component_cpu_proxy(pred, {"kinetics_00": _queries()})
    assert result["baseline"]["samples"] == 4
    assert result["baseline"]["sum_component_ms_per_target_frame_p50"] == 1.0
    assert result["template_4px"]["sum_component_ms_per_target_frame_p50"] == 3.0
    # Registration is 8ms / (4 frames * 4 queries) = 0.5ms / target frame.
    assert result["affine_3px"]["sum_component_ms_per_target_frame_p50"] == 4.5
    assert result["both_4px_3px"]["sum_component_ms_per_target_frame_p50"] == 6.5


def test_cpu_proxy_rejects_invalid_timing():
    rec = _sample()
    rec["cpu_query_ms"] = {f"p{i}": {
        "appearance_fb_055": 3.0,
        "template_recover4": 6.0,
        "affine_stabilized_klt": float("nan"),
    } for i in range(4)}
    pred = {"records": [rec], "scene_aux": {
        "kinetics_00": {"affine_registration_ms_per_video": 8.0}}}
    with pytest.raises(RuntimeError, match="BAD_CPU"):
        mod.component_cpu_proxy(pred, {"kinetics_00": _queries()})


def test_cpu_proxy_rejects_bad_affine_registration():
    rec = _sample()
    rec["cpu_query_ms"] = {f"p{i}": {
        "appearance_fb_055": 3.0,
        "template_recover4": 6.0,
        "affine_stabilized_klt": 9.0,
    } for i in range(4)}
    pred = {"records": [rec], "scene_aux": {
        "kinetics_00": {"affine_registration_ms_per_video": -1.0}}}
    with pytest.raises(RuntimeError, match="BAD_AFFINE"):
        mod.component_cpu_proxy(pred, {"kinetics_00": _queries()})
