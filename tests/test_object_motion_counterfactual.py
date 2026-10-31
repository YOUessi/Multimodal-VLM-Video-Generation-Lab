"""Source-only tests for target-grounded counterfactual visual motion experiments."""
import importlib.util,inspect,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
import object_motion_counterfactual_data as data
import target_motion_tracking as track
import object_motion_counterfactual_eval as experiment

def test_direction_labels_and_contrafactual_involution():
    assert len(data.DIRECTIONS)==len(set(data.DIRECTIONS))==5
    assert len(data.LABELS)==5
    for direction in data.DIRECTIONS:
        assert data.OPPOSITE[data.OPPOSITE[direction]]==direction
        assert data.OPPOSITE[direction]!=direction or direction=="stationary"

def test_deterministic_video_renderer_and_counterfactual_tracker(tmp_path):
    images=[];args=None
    for direction in data.DIRECTIONS:
        scene=data.make_scene(20261101+len(images),direction,224,224)
        frames=data.render_scene(scene,10,224,224)
        path=tmp_path/(direction+".mp4")
        rev=tmp_path/(direction+"_reversed.mp4")
        data.write_mp4(path,frames,5)
        data.write_mp4(rev,list(reversed(frames)),5)
        question="What direction is the "+scene["target_color"]+" "+scene["target_shape"]+" moving in the video?"
        a=track.measured_trace(path,question,3)
        b=track.measured_trace(rev,question,3)
        assert a["missing_points"]==b["missing_points"]==0
        assert track.infer_direction(a,9)==direction
        assert track.infer_direction(b,9)==data.OPPOSITE[direction]
        if direction!="stationary":
            ax,ay=a["delta_xy"];bx,by=b["delta_xy"]
            assert abs(ax+bx)<2 and abs(ay+by)<2
        images.append(direction)

def test_question_only_no_oracle_data_entering_vision_tracker():
    assert list(inspect.signature(track.measured_trace).parameters)==[
        "video_path","question","frame_count"]
    assert "meta" not in inspect.getsource(track.measured_trace)
    assert track.question_color("What direction is the blue sphere moving?")=="blue"
    try:track.question_color("What direction is the object moving?")
    except RuntimeError:pass
    else:raise AssertionError("Ambiguous color accepted")

def test_wrong_order_hint_is_opposite_coordinate_sequence():
    trace={"points":[[40.0,50.0],[50.0,40.0],[60.0,30.0]]}
    natural=track.numeric_evidence(trace,reverse=False)
    contradictory=track.numeric_evidence(trace,reverse=True)
    assert "x=40.0, y=50.0" in natural
    assert "x=60.0, y=30.0" in contradictory
    assert natural!=contradictory

def test_model_prompt_does_not_include_gold_or_scene_spec():
    row={"question":"Which direction is the red cube moving?",
         "options":list(data.LABELS.values()),"gold_letter":"A",
         "gold_direction":"stationary","scene_spec":{"direction":"stationary"}}
    msg=experiment.make_prompt(row,"text_only",[],None)
    txt=json.dumps(msg,default=str)
    assert "scene_spec" not in txt
    assert "gold_direction" not in txt
    assert "gold_letter" not in txt
    assert "ANSWER: followed" in txt

def test_pair_clustered_bootstrap_treats_two_reversals_as_one_unit():
    ids=[f"scene{i}" for i in range(5) for _ in (0,1)]
    pairs={f"vid{i}":group for i,group in enumerate(ids)}
    before={k:False for k in pairs}
    after={k:False for k in pairs}
    assert experiment.group_bootstrap_delta(before,after,pairs,seed=0)["cluster_bootstrap_95ci_pp"]==[0.0,0.0]
    for i in range(2):
        after[f"vid{i}"]=True
    stats=experiment.group_bootstrap_delta(before,after,pairs,seed=0)
    assert stats["delta_pp"]==20.0
    assert stats["resampling_unit"]=="paired original/reversed scene ID"

def test_frozen_config_and_5_approved_prompt_conditions():
    cfg=json.loads((ROOT/"configs/object_motion_v1.json").read_text())
    assert cfg["dev_pairs"]==20 and cfg["test_pairs"]==120
    assert cfg["width"]==cfg["height"]==224
    assert cfg["methods"]==list(experiment.CONDITIONS)
    assert cfg["test_time_tuning"] is False
    assert "Procedural" in cfg["novelty_limit"]
    assert set(cfg["models"])=={"500m","2b"}

def test_complete_video_manifest_generation_and_counterfactual_pairs(tmp_path,monkeypatch):
    # Regression: tuples cannot be shuffled; this exercises actual file
    # creation, frozen manifest and both counterfactual time directions.
    monkeypatch.setattr(data,"ROOT",tmp_path)
    cfg={"version":"unit-counterfactual-motion-v1",
         "dataset_dir":"generated","dev_pairs":5,"test_pairs":5,
         "dev_seed":123,"test_seed":456,
         "width":224,"height":224,"frames":10,"fps":5}
    doc=data.make_dataset(cfg)
    assert doc["video_count"]==20
    assert len(doc["rows"])==20
    from collections import Counter
    for subset in ("dev","test"):
        rows=[r for r in doc["rows"] if r["split"]==subset]
        assert len(rows)==10
        assert Counter(r["gold_direction"] for r in rows)=={d:2 for d in data.DIRECTIONS}
        grouped={}
        for row in rows:
            grouped.setdefault(row["pair_id"],[]).append(row)
            assert (tmp_path/row["video"]).is_file()
            assert data.sha256(tmp_path/row["video"])==row["video_sha256"]
        for pair in grouped.values():
            assert len(pair)==2
            fwd=next(r for r in pair if not r["reversed"])
            rev=next(r for r in pair if r["reversed"])
            assert rev["gold_direction"]==data.OPPOSITE[fwd["gold_direction"]]
            assert fwd["options"]==rev["options"]
    assert (tmp_path/"generated/manifest.json").is_file()
    assert data.make_dataset(cfg)["rows_sha256"]==doc["rows_sha256"]
