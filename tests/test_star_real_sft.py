"""Offline integrity tests for real-video SFT and answer parser."""
from pathlib import Path
import importlib.util,json
P=Path(__file__).resolve().parents[1]/"experiments/star_real_sft.py"
spec=importlib.util.spec_from_file_location("real_sft",P)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_parser_all_four_options():
    assert m.parse_answer("ANSWER: D")=="D"
    assert m.parse_answer("C. Opening the door.")=="C"
    assert m.parse_answer("A")=="A"
    assert m.parse_answer("I cannot tell") is None
def test_paired_identical_answers():
    a={"results":[{"id":i,"frames":3,"correct":i%2==0} for i in range(12)]}
    b={"results":[{"id":i,"frames":3,"correct":i%2==0} for i in range(12)]}
    assert m.paired_ci(a,b,3)["delta_pp"]==0
def test_configuration_split_and_one_validation_criterion():
    c=json.loads((P.parents[1]/"configs/star_real_sft_round1.json").read_text())
    assert sum(c["split_sizes"].values())==200
    assert c["split_sizes"]["test"]==40
    assert c["training_frames"]==3
    assert len(set(c["training_seeds"]))==2
