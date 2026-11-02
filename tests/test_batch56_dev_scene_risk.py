"""Frozen DEV-scene risk auditing: correct cluster unit and TEST isolation."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"experiments"))
import batch56_dev_scene_risk as risk


def fixture():
    cfg=json.loads((ROOT/"configs/kinetics_batch56.json").read_text())
    methods=cfg["dev_selection"]["candidate_methods"]
    scene = {"invisible":10,"invisible_false_accept":0,"invisible_false_accept_rate":0.0}
    dev={"split":"dev","source_commit":"a"*40,
         "per_scene":{f"kinetics_{i:02d}":{m:dict(scene) for m in methods} for i in range(36)}}
    selected={"split":"dev","selected_method":methods[0],
              "eligible_candidates":methods,"dev_results_sha256":hashlib.sha256(risk.canonical(dev)).hexdigest()}
    return dev,selected,cfg


def test_clopper_pearson_exact_boundary():
    assert risk.one_sided_upper(0,5,.05)==pytest.approx(1-.05**(.2),abs=1e-12)
    assert risk.one_sided_upper(5,5,.05)==1
    assert risk.one_sided_upper(0,0,.05) is None
    assert risk.one_sided_upper(0,36,.05)<.10
    assert risk.one_sided_upper(0,28,.05)>.10


@pytest.mark.parametrize("k,n,alpha",[(2,1,.05),(-1,3,.05),(0,1,0),(0,1,1)])
def test_invalid_binomial_parameters_fail_closed(k,n,alpha):
    with pytest.raises(ValueError,match="BINOMIAL_INPUT_INVALID"):
        risk.one_sided_upper(k,n,alpha)


def test_all_36_scenes_counted_as_independent_units_not_frames():
    d,s,c=fixture()
    out=risk.compute(d,s,c)
    row=out["methods"][s["selected_method"]]
    assert row["supported_scenes"]==36
    assert row["supported_scene_occlusion_frames"]==360
    assert row["violating_scenes"]==0
    assert row["nominal_95_upper_scene_violation_probability"]<0.10
    assert row["nine_method_familywise_95_upper"]>row["nominal_95_upper_scene_violation_probability"]
    assert out["test_oracle_read"] is False
    assert out["test_predictions_read"] is False


def test_unsupported_scene_not_misclassified_as_safe():
    d,s,c=fixture()
    m=s["selected_method"]
    d["per_scene"]["kinetics_35"][m]={"invisible":0,"invisible_false_accept":0,"invisible_false_accept_rate":None}
    s["dev_results_sha256"]=hashlib.sha256(risk.canonical(d)).hexdigest()
    row=risk.compute(d,s,c)["methods"][m]
    assert row["supported_scenes"]==35
    assert row["unsupported_scenes"]==1
    assert row["unsupported_scene_ids"]==["kinetics_35"]


def test_one_bad_small_scene_is_one_violation_not_one_frame_vote():
    d,s,c=fixture()
    m=s["selected_method"]
    d["per_scene"]["kinetics_00"][m]={"invisible":1,"invisible_false_accept":1,"invisible_false_accept_rate":1.0}
    s["dev_results_sha256"]=hashlib.sha256(risk.canonical(d)).hexdigest()
    row=risk.compute(d,s,c)["methods"][m]
    assert row["violating_scenes"]==1
    assert row["max_observed_scene_false_accept"]==1.0


@pytest.mark.parametrize("mutation,error",[
    (lambda d,s,c:d.update(split="test"),"DEV_ONLY_FIREWALL"),
    (lambda d,s,c:d["per_scene"].update(kinetics_36={}),"SCENE_POPULATION_INVALID"),
    (lambda d,s,c:d["per_scene"]["kinetics_01"][c["dev_selection"]["candidate_methods"][0]].update(invisible_false_accept=11),"BAD_SCENE_COUNT"),
    (lambda d,s,c:s.update(selected_method="post_hoc_test_method"),"SELECTED_METHOD_INVALID"),
])
def test_malformed_or_leaky_dev_inputs_refused(mutation,error):
    d,s,c=fixture()
    mutation(d,s,c)
    s["dev_results_sha256"]=hashlib.sha256(risk.canonical(d)).hexdigest()
    with pytest.raises(RuntimeError,match=error):
        risk.compute(d,s,c)


def test_dev_scores_are_pinned_to_selection_without_test_lookup():
    d,s,c=fixture()
    s["dev_results_sha256"]="0"*64
    with pytest.raises(RuntimeError,match="DEV_SELECTION_NOT_BOUND"):
        risk.compute(d,s,c)
