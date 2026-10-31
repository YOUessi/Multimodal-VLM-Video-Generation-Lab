"""Generic named HSV tracking contract without domain-specific label calibration."""
import importlib.util,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
import appearance_hsv_tracker as hsv
import appearance_hsv_transfer as transfer
from object_motion_counterfactual_data import make_scene,render_scene,write_mp4,DIRECTIONS,OPPOSITE

def config():
    return json.loads((ROOT/"configs/appearance_hsv_transfer_v1.json").read_text())

def test_named_colors_include_external_gray_brown_without_CLEVRER_tuning():
    assert hsv.query_target("Where is the gray cylinder moving?")=="gray"
    assert hsv.query_target("What direction is the brown cube going?")=="brown"
    assert hsv.query_target("Describe the grey sphere direction")=="gray"
    assert hsv.query_target("What direction is the cyan sphere moving?")=="cyan"
    assert hsv.query_target("Where does the red sphere move?")=="red"
    try:hsv.query_target("What is the direction?")
    except RuntimeError:pass
    else:raise AssertionError("Missing target color must not be guessed")

def test_hue_detection_on_rendered_development_fixture(tmp_path):
    cfg=config()
    import numpy as np
    for i,direction in enumerate(DIRECTIONS):
        scene=make_scene(99118+i,direction,224,224)
        frames=render_scene(scene,10,224,224)
        file=tmp_path/(direction+".mp4")
        write_mp4(file,frames,5)
        query=f"What direction is the {scene['target_color']} {scene['target_shape']} moving in the video?"
        record=hsv.measured_hsv_trace(file,query,cfg,3)
        assert record["target_color"]==scene["target_color"]
        # The heuristic must not invent a motion when it cannot localize.
        if record["missing"]:
            assert record["direction"] is None
        else:
            assert record["delta_xy"] is not None
            assert len(record["points"])==3

def test_component_excludes_background_gray_and_reports_ambiguous():
    import cv2,numpy as np
    cfg=config()
    empty=np.full((320,480,3),(238,238,238),dtype=np.uint8)
    mask=hsv.threshold_mask(empty,"gray",cfg)
    assert mask.shape==(320,480)
    assert hsv.candidates(empty,"gray",cfg)==[]
    nothing,reason=hsv.select_component([],None,cfg)
    assert nothing is None and reason=="NO_COLOR_COMPONENT"

def test_source_has_no_label_dependent_mask_calibration():
    s=(ROOT/"experiments/appearance_hsv_tracker.py").read_text()
    assert "row[\"gold\"]" not in s
    assert "row[\"answer\"]" not in s
    assert "optimizer.step()" not in s
    cfg=config()
    assert cfg["new_gpu_inference"] is False and cfg["no_external_label_tuning"]
    assert cfg["gate_min_dxdy"]==20
    assert cfg["hue_tolerance"]==15
    assert "Full 200" in cfg["evaluation_policy"]

def test_option_meaning_reuse_consistent():
    assert transfer.classify_option("Down and to the right.")=="down_right"
    assert transfer.choose_index(["Up and to the left.","The object is stationary."],"stationary")==1
