"""Synthetic DEV-only regressions; never read local Kinetics labels or TEST."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import dev_cross_cohort_safety_audit as audit


METHODS = [
    "klt_fb", "template_recover4", "affine_stabilized_klt",
    "appearance_fb_055", "appearance_fb_070", "appearance_fb_085",
    "gradient_fb_050", "gradient_fb_065", "gradient_fb_080",
]
RULES = {
    "candidate_methods": METHODS,
    "fallback_method": "appearance_fb_085",
    "min_visible_coverage": 0.30,
    "max_invisible_false_accept_rate": 0.10,
}


def make_record(method, bad=False, supported=True):
    false_count = 4 if bad else 0
    return {
        "visible": 10, "invisible": 10 if supported else 0,
        "invisible_false_accept": false_count if supported else 0,
        "visible_coverage": 0.70,
        "invisible_false_accept_rate": (
            (false_count / 10) if supported else None
        ),
        "jaccard_mean_1_2_4_8_16": (
            0.60 if method == "klt_fb" else 0.40
        ),
    }


def fixture_cohorts():
    out = []
    for cohort in ("48B", "53B", "55"):
        scenes = {
            f"kinetics_{i:02d}": {
                method: make_record(
                    method, bad=(cohort == "53B" and i in (1, 4) and method == "klt_fb")
                )
                for method in METHODS
            } for i in range(6)
        }
        out.append({
            "name": cohort, "scenes": scenes, "rules": dict(RULES),
            "selected_method": "klt_fb", "result_sha256": "a" * 64,
            "config_sha256": "b" * 64, "source_commit": "c" * 40,
        })
    return out


def test_cross_cohort_folds_are_dev_only_and_holdout_not_used_for_selection():
    a = fixture_cohorts()
    r = audit.compute(a)
    assert len(r["folds"]) == 3
    assert r["test_data_or_labels_read"] is False
    assert r["method_or_threshold_changed"] is False
    fold = next(x for x in r["folds"] if x["heldout_dev_cohort"] == "53B")
    assert fold["training_cohorts"] == ["48B", "55"]
    assert fold["training_dev_scene_count"] == 12
    assert fold["cross_cohort_dev_selected_method"] == "klt_fb"
    assert fold["heldout_dev"]["violating_scene_count"] == 2
    assert fold["heldout_dev"]["occlusion_supported_scenes"] == 6
    assert fold["heldout_dev"]["scene_macro_occlusion_false_accept"] == pytest.approx(0.8 / 6)


def test_zero_support_scene_does_not_count_as_success():
    a = fixture_cohorts()
    a[1]["scenes"]["kinetics_05"]["klt_fb"] = make_record("klt_fb", supported=False)
    result = audit.compute(a)
    fold = next(x for x in result["folds"] if x["heldout_dev_cohort"] == "53B")
    assert fold["heldout_dev"]["occlusion_supported_scenes"] == 5
    assert fold["heldout_dev"]["occlusion_unsupported_scenes"] == 1
    assert fold["heldout_dev"]["violating_scene_count"] == 2


def test_cross_cohort_policy_drift_is_rejected():
    a = fixture_cohorts()
    a[1]["rules"]["max_invisible_false_accept_rate"] = 0.12
    with pytest.raises(ValueError, match="CROSS_COHORT_POLICY_DRIFT"):
        audit.compute(a)


def test_cohort_name_or_count_mismatch_is_rejected():
    a = fixture_cohorts()
    with pytest.raises(ValueError, match="THREE_DISTINCT"):
        audit.compute(a[:2])
    a[2]["name"] = "TEST"
    with pytest.raises(ValueError, match="UNEXPECTED_COHORT_POPULATION"):
        audit.compute(a)


def realish_fixture():
    scenes = {
        f"kinetics_{i:02d}": {method: make_record(method) for method in METHODS}
        for i in range(6)
    }
    config = {"dev_selection": dict(RULES)}
    sha = hashlib.sha256(audit.canonical(config)).hexdigest()
    result = {
        "split": "dev", "source_commit": "a" * 40,
        "config_sha256": sha, "methods": list(METHODS), "per_scene": scenes,
    }
    result_sha = hashlib.sha256(audit.canonical(result)).hexdigest()
    selected = {
        "split": "dev", "source_commit": "a" * 40,
        "config_sha256": sha, "dev_results_sha256": result_sha,
        "selection_source": "DEV only", "rules": dict(RULES),
        "selected_method": "klt_fb", "eligible_candidates": list(METHODS),
    }
    return result, selected, config, result_sha, sha


def test_validate_frozen_choice_and_its_hash_binding():
    r, s, c, rh, ch = realish_fixture()
    checked = audit.validate_cohort("48B", r, s, c, rh, ch)
    assert checked["selected_method"] == "klt_fb"


@pytest.mark.parametrize("mutation,reason", [
    (lambda r, s, c: r.update(split="test"), "DEV_ONLY_SPLIT_REQUIRED"),
    (lambda r, s, c: s.update(dev_results_sha256="0" * 64), "DEV_SELECTION_RESULT_UNPINNED"),
    (lambda r, s, c: s.update(source_commit="0" * 40), "SOURCE_COMMIT_MISMATCH"),
    (lambda r, s, c: s.update(config_sha256="0" * 64), "CONFIG_SHA_MISMATCH"),
    (lambda r, s, c: s.update(selected_method="appearance_fb_070"), "RECOMPUTED_FROZEN_CHOICE_MISMATCH"),
    (lambda r, s, c: r["per_scene"].update(kinetics_06={}), "EXPECTED_SIX_DEV_SCENES"),
    (lambda r, s, c: r["per_scene"]["kinetics_00"]["klt_fb"].update(invisible_false_accept=11), "INVALID_FALSE_ACCEPT_COUNT"),
    (lambda r, s, c: r["per_scene"]["kinetics_00"]["klt_fb"].update(invisible_false_accept_rate=float("nan")), "INVALID_RATE_RANGE"),
    (lambda r, s, c: r["per_scene"]["kinetics_00"]["klt_fb"].update(invisible_false_accept_rate=0.2), "SCENE_RATE_COUNT_MISMATCH"),
    (lambda r, s, c: c["dev_selection"].update(max_invisible_false_accept_rate=0.11), "SELECTION_RULES_DRIFT"),
])
def test_corrupt_or_posthoc_dev_inputs_fail_closed(mutation, reason):
    r, s, c, rh, ch = realish_fixture()
    mutation(r, s, c)
    with pytest.raises(ValueError, match=reason):
        audit.validate_cohort("48B", r, s, c, rh, ch)


def test_rate_bounds_do_not_interpret_zero_support_as_observed_zero():
    a = fixture_cohorts()
    for scene in a[0]["scenes"].values():
        scene["klt_fb"] = make_record("klt_fb", supported=False)
    result = audit.summarize(a[0]["scenes"], "klt_fb", 0.10)
    assert result["occlusion_supported_scenes"] == 0
    assert result["nominal_95_upper_scene_violation_probability"] is None
    assert result["worst_supported_scene_false_accept"] is None


def test_canonical_report_is_stable():
    a = audit.compute(fixture_cohorts())
    assert audit.canonical(a) == audit.canonical(json.loads(audit.canonical(a)))
