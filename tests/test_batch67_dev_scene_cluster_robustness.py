"""Synthetic regression tests for retrospective Batch67 DEV-only risk statistics."""
from __future__ import annotations

import copy
import json
import math

import pytest

from experiments import batch67_dev_scene_cluster_robustness as m


def synthetic_source() -> dict:
    scenes = {}
    for i, scene in enumerate(m.SCENE_IDS):
        # Deliberately include one unsupported scene: no occlusion is NOT 0% risk.
        o = 0 if i == 0 else 10
        events = 0 if i == 0 else 1
        by_gate = {}
        for gate, fa in (
            ("baseline", 1 if i % 4 == 0 and o else 0),
            ("template_4px", 1 if i % 9 == 0 and o else 0),
            ("affine_3px", 1 if i % 5 == 0 and o else 0),
            ("both_4px_3px", 0),
        ):
            av = 15 if gate == "baseline" else 10
            by_gate[gate] = {
                "visible_frames": 20, "accepted_visible_frames": av,
                "correct_visible_frames_pck8": av // 2,
                "occluded_frames": o, "false_accepted_frames": fa,
                "occlusion_events": events, "affected_events": int(fa > 0),
                "worst_false_accept_run": int(fa > 0),
                "occluded_false_accept_rate": fa / o if o else None,
                **{f"events_with_run_ge{th}":
                   (int(fa > 0) if th == 1 else 0)
                   for th in m.EVENT_THRESHOLDS},
            }
        scenes[scene] = by_gate
    rows = [scenes[s] for s in m.SCENE_IDS]
    summaries = {}
    for gate in m.VARIANTS:
        a = m.aggregate(rows, gate)
        summaries[gate] = {
            "false_accepted_frames": a["false_accepted_frames"],
            "occluded_frames": a["occluded_frames"],
            "occlusion_events": a["occlusion_events"],
            "affected_events": a["affected_events"],
            "worst_false_accept_run": a["worst_false_accept_run"],
            "scene_budget_violation_count": a["scene_budget_violation_count"],
            "scene_macro_occluded_false_accept_rate": a["scene_macro_false_accept_rate"],
            "micro_visible_coverage": a["micro_visible_coverage"],
            "micro_affected_event_rate": a["micro_affected_event_rate"],
        }
    return {
        "kind": "BATCH66_DEV_ONLY_OCCLUSION_EPISODE_SAFETY_AUDIT",
        "split": "dev", "exploratory": True, "deployable": False,
        "test_predictions_read": False, "test_oracle_read": False,
        "frozen_batch59_changed": False, "oracle_dev_only_streamed_through": "kinetics_35",
        "dev_scene_count": 36, "dev_query_count": 144,
        "fixed_variants": list(m.VARIANTS), "scene_budget": m.BUDGET,
        "source_commit": "synthetic-fixture-sha",
        "per_scene": scenes, "summary": summaries,
    }


def test_synthetic_source_validates_without_private_data():
    m.validate_source(synthetic_source())


def test_unsupported_scene_not_zero_risk():
    source = synthetic_source()
    a = m.aggregate([source["per_scene"][s] for s in m.SCENE_IDS], "baseline")
    assert a["supported_occlusion_scenes"] == 35
    assert a["unsupported_occlusion_scenes"] == 1
    assert a["occlusion_events"] == 35
    assert a["occluded_frames"] == 350
    assert math.isclose(a["scene_macro_false_accept_rate"], a["false_accepted_frames"]/350, abs_tol=1e-14)


def test_all_variants_keep_denominators():
    src = synthetic_source()
    rows = [src["per_scene"][s] for s in m.SCENE_IDS]
    vals = [m.aggregate(rows, v) for v in m.VARIANTS]
    assert {a["visible_frames"] for a in vals} == {720}
    assert {a["occluded_frames"] for a in vals} == {350}
    assert {a["occlusion_events"] for a in vals} == {35}


def test_effect_is_template_minus_baseline():
    src = synthetic_source()
    e = m.paired_effect([src["per_scene"][s] for s in m.SCENE_IDS], "template_4px")
    assert e["scene_macro_false_accept_rate"] < 0
    assert e["micro_visible_coverage"] < 0
    assert e["micro_correct_visible_pck8"] < 0


def test_no_scene_is_created_by_resampling():
    src = synthetic_source()
    rows = [src["per_scene"][s] for s in m.SCENE_IDS]
    result = m.bootstrap_scene_pairs(rows, "template_4px", repetitions=100, seed=11)
    assert result["resampling_unit"].startswith("paired complete scene")
    assert result["draws"] == 100
    assert result["seed"] == 11
    for row in result["descriptive_percentiles"].values():
        assert row["p025"] <= row["p500"] <= row["p975"]


def test_bootstrap_reproducible_and_seed_sensitive():
    src = synthetic_source()
    rows = [src["per_scene"][s] for s in m.SCENE_IDS]
    a = m.bootstrap_scene_pairs(rows, "template_4px", repetitions=131, seed=5)
    b = m.bootstrap_scene_pairs(rows, "template_4px", repetitions=131, seed=5)
    c = m.bootstrap_scene_pairs(rows, "template_4px", repetitions=131, seed=6)
    assert a == b
    assert a != c


@pytest.mark.parametrize("reps,seed", [(0, 1), (99, 1), (100.0, 1), (100, None), (True, 1)])
def test_bootstrap_argument_validation(reps, seed):
    src = synthetic_source()
    rows = [src["per_scene"][s] for s in m.SCENE_IDS]
    with pytest.raises(RuntimeError, match="BATCH67_"):
        m.bootstrap_scene_pairs(rows, "template_4px", repetitions=reps, seed=seed)


@pytest.mark.parametrize("field,bad", [
    ("split", "test"), ("kind", "garbage"), ("exploratory", False),
    ("deployable", True), ("test_oracle_read", True),
    ("test_predictions_read", True), ("frozen_batch59_changed", True),
    ("oracle_dev_only_streamed_through", "kinetics_41"),
    ("dev_scene_count", 42), ("dev_query_count", 168),
    ("fixed_variants", ["baseline"]), ("scene_budget", .11),
])
def test_reject_non_dev_or_nonfrozen_provenance(field, bad):
    report = synthetic_source()
    report[field] = bad
    with pytest.raises(RuntimeError, match="BATCH67_DEV_ONLY_PROVENANCE_REQUIRED"):
        m.validate_source(report)


def test_reject_missing_scene():
    report = synthetic_source()
    del report["per_scene"]["kinetics_35"]
    with pytest.raises(RuntimeError, match="SCENE_POPULATION"):
        m.validate_source(report)


def test_reject_extra_variant():
    report = synthetic_source()
    report["per_scene"]["kinetics_00"]["future_gate"] = {}
    with pytest.raises(RuntimeError, match="VARIANT_POPULATION"):
        m.validate_source(report)


@pytest.mark.parametrize("key,bad", [
    ("visible_frames", -1),
    ("visible_frames", True),
    ("accepted_visible_frames", 100),
    ("correct_visible_frames_pck8", 100),
    ("occluded_frames", 0),
    ("false_accepted_frames", 100),
    ("occlusion_events", -1),
    ("affected_events", 5),
    ("false_accepted_frames", 1.0),
])
def test_reject_impossible_counts(key, bad):
    report = synthetic_source()
    report["per_scene"]["kinetics_04"]["baseline"][key] = bad
    with pytest.raises(RuntimeError, match="BATCH67_"):
        m.validate_source(report)


def test_reject_mismatched_gate_denominator():
    report = synthetic_source()
    report["per_scene"]["kinetics_02"]["template_4px"]["visible_frames"] += 1
    with pytest.raises(RuntimeError, match="DENOMINATOR_CHANGED"):
        m.validate_source(report)


@pytest.mark.parametrize("key,bad", [
    ("occluded_false_accept_rate", .9),
    ("occluded_false_accept_rate", float("nan")),
])
def test_reject_rate_mismatch(key, bad):
    report = synthetic_source()
    report["per_scene"]["kinetics_04"]["baseline"][key] = bad
    with pytest.raises(RuntimeError, match="RATE_REPLAY"):
        m.validate_source(report)


def test_reject_invalid_event_run():
    report = synthetic_source()
    report["per_scene"]["kinetics_04"]["baseline"]["worst_false_accept_run"] = 3
    with pytest.raises(RuntimeError, match="INVALID_EVENT_RUN"):
        m.validate_source(report)


def test_reject_event_threshold_impossible():
    report = synthetic_source()
    report["per_scene"]["kinetics_04"]["baseline"]["events_with_run_ge30"] = 1
    with pytest.raises(RuntimeError, match="EVENT_THRESHOLD_INCONSISTENT"):
        m.validate_source(report)


@pytest.mark.parametrize("key", [
    "false_accepted_frames", "worst_false_accept_run", "scene_budget_violation_count",
    "scene_macro_occluded_false_accept_rate", "micro_visible_coverage",
    "micro_affected_event_rate",
])
def test_reject_aggregate_score_drift(key):
    report = synthetic_source()
    report["summary"]["baseline"][key] += 1
    with pytest.raises(RuntimeError, match="AGGREGATE"):
        m.validate_source(report)


def test_bootstrap_delta_not_raw_percent():
    src = synthetic_source()
    result = m.analyze(src, repetitions=100)
    effects = result["paired_effects_vs_baseline"]["template_4px"]
    assert -1 <= effects["scene_macro_false_accept_rate"] < 0
    assert "template_sensitivity" in result
    assert result["test_oracle_read"] is False
    assert result["test_predictions_read"] is False
    assert result["deployable"] is False
    assert "per_scene" not in result


def test_fixed_blocks_are_six_disjoint_groups():
    src = synthetic_source()
    rows = [src["per_scene"][s] for s in m.SCENE_IDS]
    result = m.sensitivity(rows, "template_4px")
    blocks = result["leave_one_fixed_six_scene_block_out"]
    assert len(blocks) == 6
    assert set(sum((x["removed_scene_ids"] for x in blocks), [])) == set(m.SCENE_IDS)
    assert all(len(x["removed_scene_ids"]) == 6 for x in blocks)


@pytest.mark.parametrize("values,q", [
    ([1, 2, 3], .5), ([0, 1], .25), ([10], .95),
])
def test_percentile_boundaries(values, q):
    x = m.percentile(values, q)
    assert min(values) <= x <= max(values)


@pytest.mark.parametrize("values,q", [([], .5), ([1, 2], -1), ([1, 2], 1.1)])
def test_bad_percentile_fails(values, q):
    with pytest.raises(RuntimeError, match="BAD_PERCENTILE"):
        m.percentile(values, q)


def test_exclusive_create_and_sidecar(tmp_path):
    path = tmp_path / "nested" / "audit.json"
    result = {"gate": "example", "value": 1}
    pin = m.save_pinned(path, result)
    assert pin == m.sha(path.read_bytes())
    assert path.with_suffix(".json.sha256").read_text().strip() == pin
    with pytest.raises(RuntimeError, match="NO_OVERWRITE"):
        m.save_pinned(path, result)


def test_missing_or_unpinned_source_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="SOURCE_MISSING"):
        m.load_pinned_dev_summary(tmp_path / "not_found.json")
    p = tmp_path / "untrusted.json"
    p.write_text("{}")
    with pytest.raises(RuntimeError, match="SOURCE_PIN_MISSING"):
        m.load_pinned_dev_summary(p)
    p.with_suffix(".json.sha256").write_text(m.sha(p.read_bytes()))
    with pytest.raises(RuntimeError, match="SOURCE_SHA_DRIFT"):
        m.load_pinned_dev_summary(p)


def test_source_symlink_rejected(tmp_path):
    origin = tmp_path / "real.json"
    origin.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(origin)
    with pytest.raises(RuntimeError, match="SOURCE_MISSING_OR_SYMLINK"):
        m.load_pinned_dev_summary(link)


def test_full_analysis_json_finite_and_no_uncontrolled_payload():
    result = m.analyze(synthetic_source(), repetitions=100)
    payload = json.dumps(result, allow_nan=False)
    assert "private_oracle_read_by_this_script" in payload
    assert '"per_scene"' not in payload
    assert result["dev_scenes"] == 36
    assert result["frozen_batch59_changed"] is False
