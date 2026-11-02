"""Batch53B must not turn a dataset schema repair into a new TEST selection."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "automation/plans/batch53_fresh_independent_safety_dev_select_2026-11-03.json"
REPAIRED = ROOT / "automation/plans/batch53b_geometry_integrity_resume_2026-11-03.json"


def test_identical_stages_and_preadmitted_vlm_gate():
    before = json.loads(ORIGINAL.read_text())
    after = json.loads(REPAIRED.read_text())
    assert before["version"] != after["version"]
    assert before["stages"] == after["stages"]
    assert len(after["stages"]) == 22
    assert after["test_policy"] == before["test_policy"]
    assert after["gpu_policy"] == before["gpu_policy"]
    assert after["max_stages_per_tick"] == before["max_stages_per_tick"] == 2


def test_schema_repair_explicitly_prohibits_reselecting_videos_or_relabeling():
    after = json.loads(REPAIRED.read_text())
    assert "identical preselected" in after["data_policy"]
    assert "do not clip, relabel or" in after["data_policy"]
    assert "Fail closed if any scene has <4 valid queries" in after["data_policy"]
    assert "original Batch53 failure state remains unchanged" in after["failure_policy"]


def test_runner_points_to_independent_frozen_plan():
    code = (ROOT / "automation/runner_batch53b.py").read_text()
    assert "batch53b_geometry_integrity_resume_2026-11-03.json" in code
    assert "runner.main()" in code
