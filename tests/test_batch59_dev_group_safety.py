"""Frozen independent-cohort Batch59 grouped DEV-safety screen tests."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import batch59_dev_group_safety as risk
import batch58_dev_grouped_transfer as baseline


def fixture():
    cfg = json.loads((ROOT / "configs/kinetics_batch59.json").read_text())
    methods = cfg["dev_selection"]["candidate_methods"]
    scenes = {}
    for i in range(36):
        scenes[f"kinetics_{i:02d}"] = {
            name: {
                "visible": 20, "accepted_visible": 12,
                "invisible": 20, "invisible_false_accept": 0,
                "visible_coverage": .6,
                "invisible_false_accept_rate": 0.0,
                "jaccard_mean_1_2_4_8_16": (.8 if j == 0 else .3),
            } for j, name in enumerate(methods)
        }
    dev = {
        "split": "dev", "source_commit": "a"*40,
        "config_sha256": "b"*64, "scene_count": 36,
        "per_scene": scenes, "scene_macro": {},
    }
    selected = {"split":"dev", "source_commit":dev["source_commit"],
                "config_sha256":dev["config_sha256"],
                "rules":copy.deepcopy(cfg["dev_selection"])}
    refresh(dev,selected,cfg)
    return dev,selected,cfg


def refresh(dev, selected, cfg):
    ids=sorted(dev["per_scene"])
    dev["scene_macro"]={
        method:{key:baseline.metrics(dev, ids, method)[key] for key in baseline.KEYS}
        for method in cfg["dev_selection"]["candidate_methods"]
    }
    chosen, eligible, _=baseline.choose(dev,ids,cfg["dev_selection"])
    selected["selected_method"]=chosen
    selected["eligible_candidates"]=eligible
    selected["dev_results_sha256"]=hashlib.sha256(risk.canonical(dev)).hexdigest()


def test_all_safe_exploratory_screen_still_denies_deployment():
    dev,selected,cfg=fixture()
    out=risk.analyze(dev,selected,cfg)
    assert out["split"]=="dev"
    assert out["selected_method_group_screen_pass"]
    assert out["counterfactual_dev_only_group_candidate"]==selected["selected_method"]
    assert len(out["predeclared_groups"])==6
    assert len({s for g in out["predeclared_groups"] for s in g})==36
    assert not out["deployment_authorized"]
    assert not out["test_oracle_read"]
    assert not out["test_predictions_read"]
    assert not out["changes_frozen_method_selection"]


def test_bad_six_scene_group_blocks_frozen_winner_without_reselection():
    dev,selected,cfg=fixture()
    chosen=cfg["dev_selection"]["candidate_methods"][0]
    # A method wins global Jaccard and has global risk <10%,
    # but fails the predeclared 6-scene group safety screen.
    for i in range(6):
        item=dev["per_scene"][f"kinetics_{i:02d}"][chosen]
        item["invisible_false_accept"]=8
        item["invisible_false_accept_rate"]=.4
    refresh(dev,selected,cfg)
    assert selected["selected_method"]==chosen
    out=risk.analyze(dev,selected,cfg)
    assert out["selected_by_frozen_policy"]==chosen
    assert not out["selected_method_group_screen_pass"]
    assert out["group_screen_status"]=="BLOCK_GROUP_RISK_OR_MISSING_SUPPORT"
    assert out["methods"][chosen]["groups_exceeding_mean_safety_threshold"]==[0]
    assert out["counterfactual_dev_only_group_candidate"] != chosen


def test_missing_occlusion_group_not_counted_as_safe():
    dev,selected,cfg=fixture()
    for i in range(6):
        for item in dev["per_scene"][f"kinetics_{i:02d}"].values():
            item["invisible"]=0
            item["invisible_false_accept"]=0
            item["invisible_false_accept_rate"]=None
    refresh(dev,selected,cfg)
    out=risk.analyze(dev,selected,cfg)
    assert out["group_screen_candidate_methods"]==[]
    for row in out["methods"].values():
        assert row["groups_without_occlusion_evidence"]==[0]
    assert out["counterfactual_dev_only_group_candidate"] is None


def test_fixed_group_boundaries_not_optimized_for_errors():
    dev, selected, cfg = fixture()
    out=risk.analyze(dev,selected,cfg)
    assert out["predeclared_groups"][0]==[f"kinetics_{i:02d}" for i in range(6)]
    assert out["predeclared_groups"][-1]==[f"kinetics_{i:02d}" for i in range(30,36)]
    assert out["method_count"]==9


def test_scene_risk_does_not_confuse_frames_with_independent_scenes():
    dev,selected,cfg=fixture()
    method=selected["selected_method"]
    row=dev["per_scene"]["kinetics_00"][method]
    row["invisible"]=1
    row["invisible_false_accept"]=1
    row["invisible_false_accept_rate"]=1.
    refresh(dev,selected,cfg)
    report=risk.analyze(dev,selected,cfg)
    assert report["methods"][method]["whole_dev"]["violating_occlusion_scenes"]==1
    assert report["methods"][method]["familywise_scene_violation_upper_95"]>.10


@pytest.mark.parametrize("mutation,expected",[
    (lambda d,s,c: d.update(split="test"), "DEV_ONLY_FIREWALL"),
    (lambda d,s,c: s.update(split="test"), "DEV_ONLY_FIREWALL"),
    (lambda d,s,c: d["per_scene"].pop("kinetics_00"), "INVALID_DEV_POPULATION"),
    (lambda d,s,c: s.update(dev_results_sha256="0"*64), "DEV_SELECTION_NOT_BOUND"),
    (lambda d,s,c: c["dev_selection"].update(max_invisible_false_accept_rate=.2), "POLICY_OR_CONFIG_DRIFT"),
])
def test_bad_or_leaked_inputs_fail_closed(mutation,expected):
    dev,selected,cfg=fixture()
    mutation(dev,selected,cfg)
    with pytest.raises(RuntimeError,match=expected):
        risk.analyze(dev,selected,cfg)


def test_group_risk_report_output_is_canonical_and_repeatable():
    dev,selected,cfg=fixture()
    first=risk.canonical(risk.analyze(dev,selected,cfg))
    assert first == risk.canonical(risk.analyze(dev,selected,cfg))
    assert hashlib.sha256(first).hexdigest()==risk.digest(first)
