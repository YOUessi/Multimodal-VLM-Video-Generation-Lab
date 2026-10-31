"""No-GPU order rotation invariance and unbiased candidate remapping tests."""
from pathlib import Path
import importlib.util
p=Path(__file__).resolve().parents[1]/"experiments"/"choice_bias_eval.py"
spec=importlib.util.spec_from_file_location("bias",p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_cyclic_rotations_cover_positions():
    choices=["cat","dog","fox"]
    seen=set()
    for k in (0,1,2):
        rotated=m.rotate_choices(choices,k)
        assert len(set(rotated))==3
        assert sorted(rotated)==sorted(choices)
        assert m.backmap("A",k)==k
        for j,candidate in enumerate(rotated):
            assert choices[m.backmap("ABC"[j],k)]==candidate
        seen.add(tuple(rotated))
    assert len(seen)==3
def test_majority_and_tie_break_contract():
    assert m.majority_vote([0,0,2])==0
    assert m.majority_vote([2,0,2])==2
    assert m.majority_vote([2,1,0])==2
    assert m.majority_vote([None,None,0]) is None
def test_unknown_answer_does_not_map_to_gold():
    assert m.backmap(None,1) is None
    assert m.backmap("D",2) is None
