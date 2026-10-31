"""Unit tests of real MVBench selection, answer parsing, and metrics; no GPU required."""
from pathlib import Path
import importlib.util
import json
p=Path(__file__).resolve().parents[1]/"experiments"/"mvbench_real_eval.py"
spec=importlib.util.spec_from_file_location("mvbench_real_eval",p)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
def test_choices():
    assert m.parse_choice("ANSWER: A")=="A"
    assert m.parse_choice("Some reasoning\nANSWER: B")=="B"
    assert m.parse_choice("ANSWER: C.")=="C"
    assert m.parse_choice("I am uncertain") is None
    assert m.parse_choice("A and B") is None
    assert m.parse_choice("B")=="B"
def test_pairwise_ci_same_answers():
    base={"predictions":[{"id":i,"frames":3,"correct":i%2==0} for i in range(20)]}
    tuned={"predictions":[{"id":i,"frames":3,"correct":i%2==0} for i in range(20)]}
    assert m.paired_ci(base,tuned,3)==[0.0,0.0]
def test_nonpaired_rejected():
    base={"predictions":[{"id":0,"frames":1,"correct":True}]}
    tuned={"predictions":[{"id":1,"frames":1,"correct":True}]}
    try:m.paired_ci(base,tuned,1)
    except AssertionError:pass
    else:raise AssertionError("missing non-paired guard")
def test_config_is_frozen():
    config=json.loads((p.parents[1]/"configs"/"mvbench_round1.json").read_text())
    assert config["sample_count"]==80
    assert config["frame_counts"]==[1,3,5]
    assert len(config["archive_sha256"])==64
    assert config["adapter"].endswith("/adapter")
