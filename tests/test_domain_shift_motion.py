"""New independently frozen synthetic motion domain-shift + camera registration."""
from pathlib import Path
import sys,random,json,collections
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
import domain_shift_motion_data as data
import domain_shift_camera_tracker as track
import domain_shift_motion_eval as exp

def cfg():
    return json.loads((ROOT/"configs/domain_shift_motion_v1.json").read_text())

def test_new_direction_counterfactuals():
    assert len(data.MODES)==4
    assert len(data.DIRECTIONS)==5
    for direction in data.DIRECTIONS:
        assert data.OPPOSITE[data.OPPOSITE[direction]]==direction
    assert all("relative" in c.lower() or "left" in c.lower() or "right" in c.lower()
               for c in data.LABELS.values())

def test_real_source_video_camera_pan_background_registration(tmp_path):
    props=data.scene(3319,"camera_pan","stationary")
    props["camera_velocity"]=[2,-3]
    # Force stationary target in world; screen movement comes only from camera.
    assert props["world_target_velocity"]==(0,0)
    frames=data.render(props,10,256,256)
    video=tmp_path/"camera.mp4"
    data.write_video(video,frames,6)
    assert data.video_frames(video)==10
    estimate=track.estimated_bg_shift(video,min_response=0.01)
    assert estimate["registration_accepted"]
    dx,dy=estimate["background_shift"]
    # Camera velocity (2,-3) for 9 steps means background shifts (-18,+27).
    assert abs(dx+18)<3.0,(dx,dy)
    assert abs(dy-27)<3.0,(dx,dy)

def test_clips_can_be_decoded_and_counterfactual_exact_reverse(tmp_path):
    s=data.scene(923813,"camera_pan","up_right")
    forward=data.render(s,10,256,256)
    a=tmp_path/"a.mp4";b=tmp_path/"b.mp4"
    data.write_video(a,forward,6)
    data.write_video(b,list(reversed(forward)),6)
    assert data.video_frames(a)==data.video_frames(b)==10
    assert data.OPPOSITE[s["direction"]]=="down_left"
    assert data.digest(a)!=data.digest(b)

def test_tracker_input_is_question_and_pixels_not_source_metadata():
    import inspect
    assert list(inspect.signature(track.tracker).parameters)==[
        "path","question","hsv_cfg","stable"]
    source=inspect.getsource(track.tracker)
    assert "gold_direction" not in source
    assert "camera_velocity" not in source
    assert "scene_spec" not in source
    assert "model_output" not in source

def test_declined_motion_never_overrides_model():
    p={"direction":None,"delta_xy":None}
    assert track.gated_label(p,cfg())[0] is None
    q={"direction":"stationary","delta_xy":[0,0]}
    assert track.gated_label(q,cfg())[0] is None
    r={"direction":"up_right","delta_xy":[6,-8]}
    assert track.gated_label(r,cfg())[0] is None
    r={"direction":"up_right","delta_xy":[27,-35]}
    assert track.gated_label(r,cfg())[0]=="up_right"

def test_scene_pair_cluster_bootstrap_zero_and_n():
    originals={f"x{i}":False for i in range(8)}
    improved={k:False for k in originals}
    groups={f"x{i}":f"pair{i//2}" for i in range(8)}
    assert exp.bootstrap_scene(originals,improved,groups,seed=1)["delta_pp"]==0.0
    improved["x0"]=improved["x1"]=True
    delta=exp.bootstrap_scene(originals,improved,groups,seed=1)
    assert delta["delta_pp"]==25.0
    assert len(delta["cluster_bootstrap_95ci_pp"])==2

def test_pinned_dataset_balanced_and_no_answer_tuning():
    config=cfg()
    assert config["dev_pairs"]==40 and config["test_pairs"]==100
    assert config["frame_budgets"]=={"500m":[3],"2b":[3,5]}
    assert config["min_registration_response"]==0.12
    assert config["predeclare_no_test_tuning"] is True
    assert config["not_natural_video"] is True
    assert config["dataset_dir"].startswith("data/")

def test_small_full_manifest_generation(tmp_path,monkeypatch):
    monkeypatch.setattr(data,"ROOT",tmp_path)
    sample=cfg()
    sample.update({"dataset_dir":"dataset","dev_pairs":20,"test_pairs":20})
    created=data.build(sample)
    assert created["total_video_clips"]==80
    assert created["dev_test_content_overlap"]==0
    assert len(created["rows"])==80
    for split in ("dev","test"):
        original=[r for r in created["rows"]
                  if r["split"]==split and not r["reversed"]]
        groups=collections.Counter((r["variant"],r["gold_direction"]) for r in original)
        assert len(groups)==20 and all(n==1 for n in groups.values())
    assert data.build(sample)["source_sha256"]==created["source_sha256"]
    assert (tmp_path/"dataset/manifest.json").is_file()
    assert (tmp_path/"outputs/motion_render_shift_v1/manifest.json").exists() is False

def test_dataset_and_prediction_results_roots_are_distinct():
    import inspect
    cfg_doc=cfg()
    assert cfg_doc["dataset_dir"]!=cfg_doc["output_dir"]
    assert 'ROOT/cfg["dataset_dir"]' in inspect.getsource(data.build)
    assert 'ROOT/cfg["output_dir"]' not in inspect.getsource(data.build)

