"""Public MVBench scene transition transfer is strictly evaluation-only."""
from pathlib import Path
import importlib.util,sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
p=ROOT/"experiments"/"mvbench_scene_external.py"
spec=importlib.util.spec_from_file_location("external_eval",p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_external_scene_config_disjoint():
    doc=json.loads((ROOT/"configs/mvbench_scene_external_v1.json").read_text())
    assert doc["sample_count"]==200
    assert doc["evaluation_only"] and doc["no_train_on_scene_transition"]
    assert doc["frame_counts"]==[3,5]
    assert doc["grouped_star_manifest"]=="outputs/star_grouped_sft_v1/manifest.json"
    assert len(doc["labels_sha256"])==64 and len(doc["archive_sha256"])==64

def test_no_train_method_in_external_task():
    import ast
    root=ast.parse(p.read_text(encoding="utf8"))
    fn_names={n.name for n in ast.walk(root) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    assert "train" not in fn_names and "fit" not in fn_names
    assert {"manifest","evaluate","compare"} <= fn_names

def test_paired_ci_on_identical_scores():
    base={"results":[{"id":i,"frames":3,"correct":i%2==0} for i in range(200)]}
    assert m.paired_ci(base,base,3)["delta_pp"]==0.0
