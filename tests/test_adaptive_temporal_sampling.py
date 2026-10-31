"""Label-blind temporal sampler guards and exact paired metric tests."""
import importlib.util
from pathlib import Path
import sys
P=Path(__file__).resolve().parents[1]/"experiments"/"adaptive_temporal_sampling.py"
sys.path.insert(0,str(P.parent))
spec=importlib.util.spec_from_file_location("sampling",P)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_endpoint_guard_and_uniqueness():
    times=list(range(5,96,5))
    scores=[float(i % 3) for i in range(len(times))]
    for n in (3,5):
        sel=m.indices_for_signals(times,scores,n,10)
        assert len(sel)==n
        assert sel==sorted(set(sel))
        assert sel[0]==5 and sel[-1]==95
def test_diversity_constraint():
    times=list(range(0,100,10))
    signals=[0.0,1.0,0.0,.9,.1,.2,.3,.4,.1,0.0]
    selected=m.indices_for_signals(times,signals,5,10)
    assert selected[0]==0 and selected[-1]==90
def test_sample_plan_has_no_labels_in_selection_signature():
    import inspect
    signature=list(inspect.signature(m.select_video).parameters)
    assert signature==["video","decoded","count","probe_count"]
def test_pairwise_metric_identical():
    uniform={"results":[{"id":i,"frames":3,"correct":i%2==0} for i in range(40)]}
    adaptive={"predictions":[{"id":i,"frames":3,"correct":i%2==0} for i in range(40)]}
    x=m.paired_improvement(uniform,adaptive,3)
    assert x["delta_pp"]==0 and x["bootstrap_95ci_pp"]==[0.0,0.0]
