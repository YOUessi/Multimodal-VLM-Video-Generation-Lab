"""No training on evaluated samples: test read-only evidence-use diagnostics."""
import importlib.util
from pathlib import Path
P=Path(__file__).resolve().parents[1]/"experiments/evidence_use_diagnostics.py"
spec=importlib.util.spec_from_file_location("evidence_diag",P)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_paired_rescue_and_regression():
    a={"p":{"gold":"A","pred":"B","correct":False},
       "q":{"gold":"A","pred":"A","correct":True},
       "r":{"gold":"B","pred":"A","correct":False},
       "s":{"gold":"C","pred":"C","correct":True}}
    b={"p":{"gold":"A","pred":"A","correct":True},
       "q":{"gold":"A","pred":"B","correct":False},
       "r":{"gold":"B","pred":"A","correct":False},
       "s":{"gold":"C","pred":"C","correct":True}}
    c=m.paired_transitions(a,b)
    assert c["items"]==4 and c["video_rescues"]==1
    assert c["video_regressions"]==1 and c["choice_changes"]==2
def test_entropy_and_normalization():
    assert m.entropy([100,0,0,0])==0.0
    assert m.entropy([25,25,25,25])==2.0
    assert m.entropy([]) is None
