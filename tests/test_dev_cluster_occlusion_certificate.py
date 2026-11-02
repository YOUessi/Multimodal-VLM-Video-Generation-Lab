"""DEV-only scene-level risk tests; synthetic data, no TEST oracle access."""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import dev_cluster_occlusion_certificate as audit


def fixture_docs():
    cfg = {"dev_selection": {
        "candidate_methods": ["safe", "risky"],
        "max_invisible_false_accept_rate": 0.1,
        "min_visible_coverage": 0.3,
        "fallback_method": "safe",
    }}
    scenes = {}
    for i in range(6):
        scenes[f"kinetics_0{i}"] = {
            "safe": {"invisible": 100 if i != 5 else 0,
                     "invisible_false_accept": 0,
                     "invisible_false_accept_rate": 0.0 if i != 5 else None},
            "risky": {"invisible": 100 if i != 5 else 0,
                      "invisible_false_accept": 20 if i != 5 else 0,
                      "invisible_false_accept_rate": 0.2 if i != 5 else None},
        }
    results = {"split": "dev", "scene_count": 6,
               "methods": ["safe", "risky"], "per_scene": scenes,
               "source_commit": "a" * 40}
    selection = {"split": "dev", "source_commit": "a" * 40,
                 "selection_source": "DEV only",
                 "eligible_candidates": ["safe"],
                 "selected_method": "safe", "rules": cfg["dev_selection"]}
    provenance = {"frozen_source_sha": "a" * 40, "results_sha256": "f" * 64}
    return results, selection, cfg, provenance


def test_zero_bad_scenes_exact_upper_and_29_scene_requirement():
    assert audit.exact_one_sided_binomial_upper(0, 5, 0.05) == pytest.approx(
        1 - 0.05 ** (1 / 5)
    )
    assert audit.exact_one_sided_binomial_upper(0, 0, 0.05) is None
    assert audit.zero_failure_scenes_needed(0.05, 0.1) == 29


def test_exact_upper_monotonic_with_failures_and_worst_case():
    assert audit.exact_one_sided_binomial_upper(1, 5, 0.05) > audit.exact_one_sided_binomial_upper(0, 5, 0.05)
    assert audit.exact_one_sided_binomial_upper(5, 5, 0.05) == 1.0
    assert audit.exact_one_sided_binomial_upper(3, 5, 0.05) > audit.exact_one_sided_binomial_upper(1, 5, 0.05)


@pytest.mark.parametrize("k,n,a", [(-1, 2, .05), (2, 1, .05),
                                    (0, 0, -1), (1, 1, 1),
                                    (True, 2, .05), (1, False, .05)])
def test_exact_upper_fails_closed_on_invalid_arguments(k, n, a):
    with pytest.raises(ValueError):
        audit.exact_one_sided_binomial_upper(k, n, a)


def test_six_dev_scenes_no_safety_certificate_even_with_zero_breaches():
    results, selected, cfg, provenance = fixture_docs()
    report = audit.calculate(results, selected, cfg, provenance)
    winner = report["selected_method_cluster_support"]
    assert report["split"] == "dev"
    assert winner["supported_independent_scene_count"] == 5
    assert winner["scenes_without_occlusion_support"] == 1
    assert winner["observed_scene_risk_exceedances"] == 0
    assert winner["bonferroni_95pct_familywise_scene_violation_probability_upper"] > 0.1
    assert report["selected_diagnostic_verdict"] == "INSUFFICIENT_TO_CERTIFY_CROSS_SCENE_SAFETY"
    assert report["zero_failures_min_supported_scenes_at_nominal_95pct"] == 29


def test_scene_breach_is_detected_even_when_avg_other_scenes_looks_safe():
    results, selected, cfg, provenance = fixture_docs()
    results["per_scene"]["kinetics_02"]["safe"].update(
        invisible_false_accept=11, invisible_false_accept_rate=0.11
    )
    report = audit.calculate(results, selected, cfg, provenance)
    assert report["selected_method_cluster_support"]["observed_scene_risk_exceedances"] == 1
    assert report["selected_diagnostic_verdict"] == "OBSERVED_SCENE_RISK_BREACH"


@pytest.mark.parametrize("alter,error", [
    (lambda r,s,c: r.update(split="test"), "SIX_DEV_SCENES_ONLY"),
    (lambda r,s,c: r["per_scene"].__setitem__("kinetics_06", r["per_scene"].pop("kinetics_00")), "SIX_DEV_SCENES_ONLY"),
    (lambda r,s,c: r["per_scene"]["kinetics_00"]["safe"].update(invisible_false_accept=101), "INVALID_OCCLUSION_SUPPORT"),
    (lambda r,s,c: r["per_scene"]["kinetics_00"]["safe"].update(invisible_false_accept_rate=0.9), "INVALID_OCCLUSION_RATE"),
    (lambda r,s,c: r["per_scene"]["kinetics_05"]["safe"].update(invisible_false_accept_rate=0.0), "ZERO_SUPPORT_RATE_MUST_BE_NULL"),
    (lambda r,s,c: s.update(selected_method="unknown"), "BAD_SELECTED_METHOD"),
    (lambda r,s,c: c["dev_selection"].update(max_invisible_false_accept_rate=-0.1), "INVALID_FROZEN_SAFETY_THRESHOLD"),
])
def test_invalid_or_test_inputs_rejected(alter, error):
    r, s, c, p = fixture_docs()
    alter(r, s, c)
    with pytest.raises(ValueError, match=error):
        audit.calculate(r, s, c, p)


def make_frozen_fixture(root: Path):
    r,s,c,p = fixture_docs()
    d = root / "outputs/kinetics_batch53"
    d.mkdir(parents=True)
    (root / ".automation").mkdir()
    (root / "configs").mkdir()
    raw_r = audit.canonical_json(r)
    raw_c = audit.canonical_json(c)
    s.update(dev_results_sha256=audit.sha(raw_r),
             config_sha256=audit.sha(raw_c))
    r.update(config_sha256=audit.sha(raw_c))
    raw_r = audit.canonical_json(r)
    s["dev_results_sha256"] = audit.sha(raw_r)
    (d / "dev_results.json").write_bytes(raw_r)
    (d / "dev_selection.json").write_bytes(audit.canonical_json(s))
    (root / "configs/kinetics_batch53.json").write_bytes(raw_c)
    (root / ".automation/state.json").write_text(json.dumps({
        "plan_version": "batch53b-test-fixture",
        "frozen_sha": "a"*40,
    }))
    # A broken TEST artifact proves this audit never imports or opens TEST.
    (d / "test_results.json").write_text("{INVALID_PRIVATE_TEST_JSON")


def test_pinned_DEV_only_io_and_immutable_output(tmp_path):
    make_frozen_fixture(tmp_path)
    dest=tmp_path/"outputs/research/cluster.json"
    pin=audit.run(tmp_path,dest)
    assert pin == audit.sha(dest.read_bytes())
    assert dest.with_suffix(".json.sha256").read_text().strip()==pin
    assert json.loads(dest.read_text())["split"] == "dev"
    with pytest.raises(FileExistsError, match="IMMUTABLE_AUDIT_EXISTS"):
        audit.run(tmp_path,dest)


def test_tampered_DEV_result_or_source_fails_before_analysis(tmp_path):
    make_frozen_fixture(tmp_path)
    result=tmp_path/"outputs/kinetics_batch53/dev_results.json"
    doc=json.loads(result.read_text())
    doc["scene_count"]=7
    result.write_bytes(audit.canonical_json(doc))
    with pytest.raises(ValueError,match="PIN_HASH_MISMATCH"):
        audit.read_pinned_dev(tmp_path)
    doc["scene_count"]=6
    doc["source_commit"]="b"*40
    result.write_bytes(audit.canonical_json(doc))
    with pytest.raises(ValueError,match="SOURCE_SHA_MISMATCH"):
        audit.read_pinned_dev(tmp_path)


def test_frozen_unsupported_batch_is_not_guessable(tmp_path):
    make_frozen_fixture(tmp_path)
    state=tmp_path/".automation/state.json"
    state.write_text(json.dumps({"plan_version":"batch_other", "frozen_sha":"a"*40}))
    with pytest.raises(ValueError,match="UNKNOWN_FROZEN_DEV_BATCH"):
        audit.read_pinned_dev(tmp_path)
