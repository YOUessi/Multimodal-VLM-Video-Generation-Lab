"""Offline tests for generation review safety and final report logic."""
import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/"experiments"/(name+".py"))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
def test_unreviewed_generation_is_not_scored(tmp_path):
    mod=load("generation_review")
    data={"status":"REAL_LOCAL_GPU_DIFFUSION_INFERENCE","samples":[{"id":"a"}]}
    f=tmp_path/"generation_report.json";f.write_text(json.dumps(data))
    assert mod.source_report(f)["samples"][0]["id"]=="a"
    assert "human_scores.json" not in [x.name for x in tmp_path.iterdir()]
def test_disallow_unknown_clip(tmp_path):
    mod=load("generation_review")
    f=tmp_path/"generation_report.json"
    f.write_text(json.dumps({"status":"REAL_LOCAL_GPU_DIFFUSION_INFERENCE","samples":[{"id":"valid"}]}))
    csvp=tmp_path/"review.csv"
    csvp.write_text(",".join(mod.FIELDS)+"\n"+"invalid,r1,1,2,3,note\n")
    try:mod.aggregate(f,[csvp])
    except ValueError:pass
    else:raise AssertionError("Unknown clip accepted")
def test_two_independent_reviewers_required(tmp_path):
    mod=load("generation_review")
    f=tmp_path/"generation_report.json"
    f.write_text(json.dumps({"status":"REAL_LOCAL_GPU_DIFFUSION_INFERENCE","samples":[{"id":"valid"}]}))
    csvp=tmp_path/"review.csv"
    csvp.write_text(",".join(mod.FIELDS)+"\n"+"valid,r1,1,2,3,note\n")
    try:mod.aggregate(f,[csvp])
    except RuntimeError as e:assert "TWO_INDEPENDENT" in str(e)
    else:raise AssertionError("Premature human score")
