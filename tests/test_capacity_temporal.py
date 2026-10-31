"""No-GPU tests for frozen two-model time-evidence ablation."""
import importlib.util, json, random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILE=ROOT/"experiments/mvbench_capacity_temporal.py"
spec=importlib.util.spec_from_file_location("capacity",FILE)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_mcq_robust_deterministic_parse():
    assert m.parsed_answer("ANSWER: A",4)=="A"
    assert m.parsed_answer("C. Because the object moves left",4)=="C"
    assert m.parsed_answer("Option D",4)=="D"
    assert m.parsed_answer("Choice B",4)=="B"
    assert m.parsed_answer("ANSWER: D",3) is None
    assert m.parsed_answer("The answer is C",4)=="C"
    assert m.parsed_answer("ANSWER: A/B/C/D",4) is None
    assert m.parsed_answer("I cannot tell",4) is None
    assert m.parsed_answer("A and B",4) is None

def test_temporal_indices_and_no_video_condition():
    assert m.frame_positions(100,0)==[]
    assert m.frame_positions(100,1)==[50]
    assert len(m.frame_positions(100,3))==3
    assert len(m.frame_positions(100,5))==5
    for n in (3,5):
        result=m.frame_positions(151,n)
        assert result==sorted(result)
        assert len(set(result))==n
        assert result[0]==round(.05*150)
        assert result[-1]==round(.95*150)

def test_message_has_no_answer_leakage():
    row={"question":"Which way is the object moving?",
         "choices":["Left","Right","Up","Down"],"gold":"B"}
    msg=m.build_message(row,[])
    assert msg[0]["role"]=="user"
    assert "ANSWER: followed" in str(msg)
    assert "frames are provided" in str(msg)
    assert "gold" not in str(msg)
    assert "Gold" not in str(msg)

def test_bootstrap_pairing_is_symmetric_zero():
    a={"id_"+str(i):i%2==0 for i in range(200)}
    diff=m.paired_difference(a,a,seed=15)
    assert diff["delta_pp"]==0
    assert diff["bootstrap_95_ci_pp"]==[0,0]
    other={"other":True}
    try:m.paired_difference(a,other,seed=15)
    except RuntimeError:pass
    else:raise AssertionError("Non-paired data accepted")

def test_frozen_config_real_two_tasks_two_models():
    cfg=json.loads((ROOT/"configs/capacity_temporal_v1.json").read_text())
    assert list(cfg["models"])==["500m","2b"]
    assert cfg["comparison_order"]==["500m","2b"]
    assert cfg["frame_counts"]==[0,3,5]
    assert list(cfg["tasks"])==["moving_direction","scene_transition"]
    assert cfg["test_time_tuning"] is False
    assert cfg["models"]["2b"]["weight_files"]["model-00001-of-00002.safetensors"]>4*2**30
