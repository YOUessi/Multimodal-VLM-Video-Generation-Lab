"""Offline regression coverage for sampling-cost and Diffusion grid utilities."""
from pathlib import Path
import importlib.util, json, subprocess, sys

ROOT=Path(__file__).resolve().parents[1]
def module(name):
    p=ROOT/"experiments"/(name+".py")
    spec=importlib.util.spec_from_file_location("test_"+name,p)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result
def test_percentile_and_no_empty_input():
    m=module("frame_selection_cost")
    assert m.percentile([1,2,3],.5)==2.0
    assert m.percentile([1,2,3,4],.95)==3.85
    try:m.percentile([],.5)
    except ValueError:pass
    else:raise AssertionError("empty percentile accepted")
def test_configs_pin_steps_and_stars():
    c=json.loads((ROOT/"configs/diffusion_steps_grid.json").read_text())
    assert c["additional_steps"]==[8,20] and c["baseline_steps"]==14
    assert c["frames"]==12 and c["width"]==576 and c["height"]==320
    assert len(c["prompts"])==2
    assert c["model_revision"]=="1f35e3dc3350d649f342bb7618fe34ac8fae5c9e"
    assert c["license"]=="cc-by-nc-4.0 research only"
    d=json.loads((ROOT/"configs/frame_selection_cost.json").read_text())
    assert d["frame_counts"]==[3,5] and d["probe_count"]==48
    assert 2<=d["repeats"]<=5
def test_video_grid_metadata_and_motion_without_model(tmp_path):
    import cv2,numpy as np
    m=module("generation_step_grid")
    q=module("generation_diagnostics")
    p=tmp_path/"t.mp4"
    w=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*"mp4v"),6,(96,64))
    assert w.isOpened()
    for i in range(12):
        img=np.full((64,96,3),min(i*20,255),dtype=np.uint8)
        w.write(img)
    w.release()
    assert m.verify_video(p,12,96,64)==12
    stats=q.inspect_video(p)
    assert stats["frames"]==12
    assert stats["width"]==96 and stats["height"]==64
    assert stats["adjacent_rgb_mae"]>=0
    assert stats["near_identical_pairs"]>=0
    assert stats["brightness_delta_std"]>=0
