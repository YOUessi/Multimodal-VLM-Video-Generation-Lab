"""Contract tests for the frozen-stage local scheduler."""
import importlib.util
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("runner",ROOT/"automation"/"runner.py")
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
def test_plan_stages_are_distinct():
    doc=json.loads((ROOT/"automation"/"plan.json").read_text())
    stages=doc["stages"]
    assert len(stages)==len({x["id"] for x in stages})
    assert all(s["max_attempts"]==1 for s in stages)
    assert all(s["timeout_seconds"]>=120 for s in stages)
def test_atomic_writes(tmp_path):
    f=tmp_path/"data.json"
    runner.atomic_json(f,{"hello":"world"})
    assert json.loads(f.read_text())=={"hello":"world"}
    assert not (tmp_path/"data.json.tmp").exists()
def test_gpu_gate_fails_without_nvidia(monkeypatch):
    monkeypatch.setattr(runner.shutil,"which",lambda _:None)
    assert runner.gpu_ready()[0] is False
def test_synthetic_temporal_sequence_count():
    import itertools
    cs=list(itertools.product(range(6),repeat=3))
    assert len([c for c in cs if c[0]!=c[2]])==180
