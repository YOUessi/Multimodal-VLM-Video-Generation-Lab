"""Temporal causality control integrity, no GPU required."""
import importlib.util,json
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
FILE=ROOT/"experiments/temporal_order_controls.py"
spec=importlib.util.spec_from_file_location("tempcontrol",FILE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_repeated_center_matches_visual_tokens(monkeypatch):
    x=[Image.new("RGB",(40,30),(0,0,i*90)) for i in range(3)]
    monkeypatch.setattr(m,"open_frames",lambda row,n:x if n==3 else [Image.new("RGB",(40,30),"red")])
    inputs=m.build_images({"key":"x"},3,"center_repeated")
    assert len(inputs)==3
    assert len({im.tobytes() for im in inputs})==1
    assert inputs[0] is not inputs[1]

def test_reversed_frames_change_input_order_not_contents(monkeypatch):
    examples=[Image.new("RGB",(40,30),(i*95,0,0)) for i in range(3)]
    monkeypatch.setattr(m,"open_frames",lambda row,n:examples)
    rev=m.build_images({"key":"x"},3,"reverse_order")
    assert rev[0].tobytes()==examples[-1].tobytes()
    assert rev[1].tobytes()==examples[1].tobytes()
    assert rev[-1].tobytes()==examples[0].tobytes()

def test_pinned_config_and_labels_disallowed_on_reversal():
    cfg=json.loads((ROOT/"configs/temporal_controls_v1.json").read_text())
    assert cfg["methods"]==["center_repeated","reverse_order"]
    assert cfg["frame_counts"]==[3,5]
    assert cfg["model_name"]=="2b"
    assert cfg["no_reversed_gold_accuracy"]
    assert cfg["no_model_training"]
    assert len(cfg["pinned_source_manifest_sha256"])==64

def test_wrong_control_rejected():
    try:m.build_images({"key":"x"},3,"shuffled_arbitrarily")
    except ValueError:pass
    else:raise AssertionError("Unapproved control accepted")
