"""Untuned cross-domain CV tracker integration safety contracts."""
from pathlib import Path
import importlib.util,sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
spec=importlib.util.spec_from_file_location("xfer",ROOT/"experiments/clevrer_untuned_tracker_transfer.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def test_all_official_option_semantics():
    assert mod.classify_option("The object is stationary.")=="stationary"
    assert mod.classify_option("Up and to the right.")=="up_right"
    assert mod.classify_option("Up and to the left.")=="up_left"
    assert mod.classify_option("Down and to the right.")=="down_right"
    assert mod.classify_option("Down and to the left.")=="down_left"
    assert mod.classify_option("Ambiguous direction") is None

def test_correct_letter_mapping_and_missing_option_abstention():
    options=["Up and to the right.","Down and to the left.","The object is stationary."]
    assert mod.choose_index(options,"down_left")==1
    assert mod.choose_index(options,"up_left") is None

def test_frozen_high_confidence_gate(monkeypatch):
    monkeypatch.setattr(mod,"measured_trace",
      lambda path,question,n:{"missing_points":0,"delta_xy":[35,-38]})
    row={"video":"x.mp4","question":"What direction is the cyan sphere moving?",
         "choices":["The object is stationary.","Down and to the left.","Up and to the right."]}
    x=mod.track_external(row,9,20)
    assert x["accepted"] is True and x["pred_letter"]=="C"
    monkeypatch.setattr(mod,"measured_trace",
      lambda path,question,n:{"missing_points":0,"delta_xy":[12,-30]})
    x=mod.track_external(row,9,20)
    assert x["accepted"] is False and x["reason"]=="GATE_REJECTED"

def test_unsupported_color_fails_closed(monkeypatch):
    def unrecognized(path,question,n):
        raise RuntimeError("Unsupported named color")
    monkeypatch.setattr(mod,"measured_trace",unrecognized)
    row={"video":"y.mp4","question":"Which way does the brown object move?",
        "choices":["The object is stationary.","Down and to the left."]}
    x=mod.track_external(row,9,20)
    assert x["accepted"] is False and x["pred_letter"] is None
    assert x["reason"]=="RuntimeError"

def test_source_config_pins_no_tuning_and_all_200():
    p=ROOT/"configs/clevrer_untuned_tracker_v1.json"
    c=json.loads(p.read_text())
    assert c["new_gpu_inference"] is False
    assert c["no_test_time_tuning"] is True
    assert c["direction_deadzone_px"]==9
    assert c["min_gate_displacement_px"]==20
    assert len(c["pinned_manifest_sha"])==64
    assert "All 200" in c["denominator_policy"]
