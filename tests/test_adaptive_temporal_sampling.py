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

def test_candidate_bounds_across_mp4_lengths():
    # Regression: the old floating-point interpolation could exceed the
    # rounded 5% / 95% endpoint by one frame.
    for decoded in range(6,3000):
        for probe_count in (3,7,16,48,75):
            (lo,hi),samples=m.candidate_positions(decoded,probe_count)
            assert lo==round(.05*(decoded-1))
            assert hi==round(.95*(decoded-1))
            assert samples==sorted(set(samples))
            assert samples[0]==lo and samples[-1]==hi
            assert all(lo<=v<=hi for v in samples)
def test_real_decoded_mp4_endpoint_guards(tmp_path):
    import cv2,numpy as np
    path=tmp_path/"clip.mp4"
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),8,(96,64))
    assert writer.isOpened()
    for i in range(55):
        im=np.full((64,96,3),i*3%256,dtype=np.uint8)
        writer.write(im)
    writer.release()
    for n in (3,5):
        selected,_=m.select_video(path,decoded=55,count=n,probe_count=48)
        assert len(selected)==len(set(selected))==n
        assert selected[0]==round(.05*54)
        assert selected[-1]==round(.95*54)
