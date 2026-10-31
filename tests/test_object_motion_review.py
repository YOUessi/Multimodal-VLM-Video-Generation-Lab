"""Offline local counterfactual review must display both videos without unsafe HTML."""
import importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
def test_review_pairs_and_escaping(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location("review",ROOT/"experiments/object_motion_review.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    monkeypatch.setattr(m,"ROOT",tmp_path)
    cfg={"dataset_dir":"data","output_dir":"outputs","models":{"500m":{},"2b":{}},
         "methods":["text_only"]}
    rows=[]
    for reversed_ in (False,True):
        id_="test_0001_reverse" if reversed_ else "test_0001_forward"
        path=f"data/videos/{id_}.mp4"
        dest=tmp_path/path;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(b"fixture-video-marker")
        rows.append({"id":id_,"pair_id":"test_0001","split":"test",
                     "reversed":reversed_,"video":path,
                     "question":"Where is the <red> square moving?",
                     "options":["Down and to the left.","The object is stationary."],
                     "gold_letter":"A","gold_direction":"down_left"})
    (tmp_path/"data/manifest.json").write_text(json.dumps({"rows_sha256":"sourcepin","rows":rows}))
    (tmp_path/"outputs").mkdir()
    for name in cfg["models"]:
        obj={"dataset_manifest_sha":"sourcepin","predictions":[
              {"id":row["id"],"mode":"text_only",
               "pred_letter":"A","raw":"<bad-model-text>"} for row in rows]}
        (tmp_path/f"outputs/{name}.json").write_text(json.dumps(obj))
    m.build(cfg)
    text=(tmp_path/"outputs/review.html").read_text()
    assert "test_0001_forward" in text and "test_0001_reverse" in text
    assert "&lt;red&gt;" in text and "&lt;bad-model-text&gt;" in text
    assert "<bad-model-text>" not in text
    assert "Not" in text or "not" in text
