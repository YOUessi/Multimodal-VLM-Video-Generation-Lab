"""Strict task-label-blind 12-probe motion-salient frame-selection contracts."""
from pathlib import Path
import importlib.util,json
ROOT=Path(__file__).resolve().parents[1]
FILE=ROOT/"experiments/motion_salient_sampling.py"
spec=importlib.util.spec_from_file_location("motion_salient",FILE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_canonical_endpoint_fraction_for_lengths():
    for length in (6,8,16,21,40,61,100,151,597,1200):
        candidate=m.probe_indices(length,12)
        assert candidate[0]==round(.05*(length-1))
        assert candidate[-1]==round(.95*(length-1))
        assert len(candidate)==len(set(candidate))
        assert candidate==sorted(candidate)
        assert all(0<=x<length for x in candidate)

def test_label_independent_3_and_5_selection():
    positions=m.probe_indices(151,12)
    values=[0.1+float(i%4) for i in range(len(positions))]
    for n in (3,5):
        selected=m.choose_positions(positions,values,n,151)
        assert len(selected)==len(set(selected))==n
        assert selected==sorted(selected)
        assert selected[0]==positions[0]
        assert selected[-1]==positions[-1]
        assert set(selected)<=set(positions)

def test_motion_decoder_on_real_tiny_synthetic_video(tmp_path):
    import cv2,numpy as np
    video=tmp_path/"motion.mp4"
    w=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"mp4v"),6,(96,64))
    assert w.isOpened()
    for i in range(44):
        image=np.zeros((64,96,3),dtype=np.uint8)
        cv2.circle(image,(12+i,32),7,(60,200,80),-1)
        w.write(image)
    w.release()
    row={"key":"no_answer_in_input","task":"fixture","video":str(video),
         "decoded_frames":44,"video_sha256":"testsha"}
    cfg={"probe_count":12,"frame_counts":[3,5]}
    record=m.select_frame_data(row,cfg)
    assert record["key"]==row["key"]
    assert record["peak_motion"]>0
    assert len(record["frame_indices"]["3"])==3
    assert len(record["frame_indices"]["5"])==5

def test_prior_research_represented_truthfully():
    cfg=json.loads((ROOT/"configs/motion_sampling_v1.json").read_text())
    assert cfg["frame_counts"]==[3,5]
    assert cfg["probe_count"]==12
    assert cfg["luma_motion_weight"]==0.7
    assert cfg["hue_change_weight"]==0.3
    assert cfg["test_time_tuning"] is False
    assert cfg["test_dataset_is_new_blind"] is False
    assert "400-video" in cfg["research_limit"]
