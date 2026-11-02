"""Synthetic falsification tests for the TEST-blind Batch58 DEV grouped audit."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import batch58_dev_grouped_transfer as audit


def fixture():
    cfg = json.loads((ROOT / "configs" / "kinetics_batch56.json").read_text())
    methods = cfg["dev_selection"]["candidate_methods"]
    scenes = {}
    for i in range(36):
        rows = {}
        for k, m in enumerate(methods):
            jac = .65 if k == 0 else .2
            rows[m] = {
                "visible": 10, "accepted_visible": 8,
                "invisible": 10, "invisible_false_accept": 0,
                "visible_coverage": .8,
                "invisible_false_accept_rate": 0.0,
                "jaccard_mean_1_2_4_8_16": jac,
            }
        scenes[f"kinetics_{i:02d}"] = rows
    dev = {
        "split": "dev",
        "source_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "scene_count": 36,
        "per_scene": scenes,
        "scene_macro": {},
    }
    selected = {
        "split": "dev",
        "source_commit": dev["source_commit"],
        "config_sha256": dev["config_sha256"],
        "rules": cfg["dev_selection"],
    }
    refresh(dev, selected, cfg)
    return dev, selected, cfg


def refresh(dev, selected, cfg):
    ids = sorted(dev["per_scene"])
    dev["scene_macro"] = {
        m: {key: audit.average(
            [dev["per_scene"][s][m] for s in ids], key
        ) for key in audit.KEYS}
        for m in cfg["dev_selection"]["candidate_methods"]
    }
    chosen, eligible, _ = audit.choose(dev, ids, cfg["dev_selection"])
    selected["selected_method"] = chosen
    selected["eligible_candidates"] = eligible
    selected["dev_results_sha256"] = hashlib.sha256(audit.canonical(dev)).hexdigest()


def test_36_scene_grouped_transfer_and_loso_are_deterministic():
    dev, selected, cfg = fixture()
    report = audit.evaluate(dev, selected, cfg)
    assert report == audit.evaluate(dev, selected, cfg)
    grouped = report["blocked_sixfold"]
    assert len(grouped["folds"]) == 6
    assert [f["training_scene_count"] for f in grouped["folds"]] == [30] * 6
    assert [len(f["heldout_scene_ids"]) for f in grouped["folds"]] == [6] * 6
    assert grouped["summary"]["heldout_supported_occlusion_scenes"] == 36
    assert grouped["summary"]["heldout_scene_budget_violations"] == 0
    assert grouped["summary"]["heldout_scene_macro_visible_coverage"] == pytest.approx(.8)
    assert grouped["summary"]["method_choice_frequency"] == {
        cfg["dev_selection"]["candidate_methods"][0]: 6
    }
    loo = report["leave_one_scene_out"]
    assert len(loo["folds"]) == 36
    assert all(f["training_scene_count"] == 35 for f in loo["folds"])
    assert report["test_oracle_read"] is False
    assert report["test_predictions_read"] is False
    assert report["changes_frozen_experiment"] is False


def test_unsupported_occlusion_scene_never_counts_as_safe_success():
    dev, selected, cfg = fixture()
    first = cfg["dev_selection"]["candidate_methods"][0]
    r = dev["per_scene"]["kinetics_00"][first]
    r.update(invisible=0, invisible_false_accept=0, invisible_false_accept_rate=None)
    refresh(dev, selected, cfg)
    report = audit.evaluate(dev, selected, cfg)
    grouped = report["blocked_sixfold"]["summary"]
    assert grouped["heldout_unsupported_occlusion_scenes"] == 1
    assert grouped["heldout_supported_occlusion_scenes"] == 35


def test_grouped_leave_out_can_reveal_model_selection_instability():
    dev, selected, cfg = fixture()
    methods = cfg["dev_selection"]["candidate_methods"]
    first, alternative = methods[0], methods[1]
    # The first six scene IDs have a severe occlusion domain shift. Other
    # folds include that regime in training and reject the fast model.
    for i in range(6):
        row = dev["per_scene"][f"kinetics_{i:02d}"][first]
        row["invisible_false_accept"] = 8
        row["invisible_false_accept_rate"] = .8
    for i in range(36):
        dev["per_scene"][f"kinetics_{i:02d}"][alternative]["jaccard_mean_1_2_4_8_16"] = .5
    refresh(dev, selected, cfg)
    assert selected["selected_method"] == alternative
    report = audit.evaluate(dev, selected, cfg)
    assert report["blocked_sixfold"]["summary"]["method_choice_frequency"][first] == 1
    assert report["blocked_sixfold"]["summary"]["method_choice_frequency"][alternative] == 5
    assert report["blocked_sixfold"]["folds"][0]["method_chosen_without_heldout"] == first
    assert report["blocked_sixfold"]["folds"][0]["heldout_metrics"]["violating_occlusion_scenes"] == 6


def test_no_eligible_method_uses_frozen_fallback_and_is_explicit():
    dev, selected, cfg = fixture()
    for scene in dev["per_scene"].values():
        for row in scene.values():
            row["accepted_visible"] = 1
            row["visible_coverage"] = .1
    refresh(dev, selected, cfg)
    report = audit.evaluate(dev, selected, cfg)
    assert selected["selected_method"] == cfg["dev_selection"]["fallback_method"]
    assert not selected["eligible_candidates"]
    assert report["blocked_sixfold"]["summary"]["fallback_fold_count"] == 6
    assert report["leave_one_scene_out"]["summary"]["fallback_fold_count"] == 36


@pytest.mark.parametrize("mutation,expected", [
    (lambda d,s,c: d.update(split="test"), "DEV_ONLY_FIREWALL"),
    (lambda d,s,c: s.update(split="test"), "DEV_ONLY_FIREWALL"),
    (lambda d,s,c: d["per_scene"].update(test_oracle={}), "INVALID_DEV_POPULATION"),
    (lambda d,s,c: d.update(scene_count=35), "INVALID_DEV_POPULATION"),
    (lambda d,s,c: d["per_scene"]["kinetics_00"].pop(c["dev_selection"]["candidate_methods"][0]), "MISSING_METHOD"),
    (lambda d,s,c: d["per_scene"]["kinetics_00"][c["dev_selection"]["candidate_methods"][0]].update(invisible_false_accept=11), "INVALID_COUNTS"),
    (lambda d,s,c: d["per_scene"]["kinetics_00"][c["dev_selection"]["candidate_methods"][0]].update(invisible_false_accept_rate=.2), "INCONSISTENT_RATE"),
    (lambda d,s,c: d["per_scene"]["kinetics_00"][c["dev_selection"]["candidate_methods"][0]].update(jaccard_mean_1_2_4_8_16=float("nan")), "INVALID_JACCARD"),
    (lambda d,s,c: s.update(selected_method="unseen-model"), "SELECTION_INCONSISTENT"),
    (lambda d,s,c: s.update(config_sha256="wrong"), "POLICY_OR_CONFIG_DRIFT"),
    (lambda d,s,c: d["scene_macro"][c["dev_selection"]["candidate_methods"][0]].update(visible_coverage=1.0), "DEV_METRIC_INTEGRITY"),
])
def test_corruption_and_test_leakage_fail_closed(mutation, expected):
    dev, selected, cfg = fixture()
    mutation(dev, selected, cfg)
    with pytest.raises(RuntimeError, match=expected):
        audit.evaluate(dev, selected, cfg)


def test_pinned_io_rejects_modified_bytes_and_missing_tag(tmp_path):
    path = tmp_path / "dev_selection.json"
    path.write_text('{"split":"dev"}\n')
    with pytest.raises(RuntimeError, match="PIN_MISSING"):
        audit.load_pinned(path)
    tag = path.with_suffix(".json.sha256")
    tag.write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n")
    assert audit.load_pinned(path) == {"split": "dev"}
    path.write_text('{"split":"test"}\n')
    with pytest.raises(RuntimeError, match="PIN_INVALID"):
        audit.load_pinned(path)
