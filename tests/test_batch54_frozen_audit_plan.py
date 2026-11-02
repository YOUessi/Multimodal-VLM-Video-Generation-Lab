"""Check the frozen audit plan contains no TEST or GPU execution."""
import json
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'automation'))
import verify_batch54_audits as verifier


def test_dev_only_plan_pins_the_correct_prior_dev_roots():
    plan=json.loads((ROOT/'automation/plans/batch54_dev_scene_cluster_safety_2026-11-03.json').read_text())
    assert plan['enabled'] and plan['max_stages_per_tick']==2
    stages=plan['stages']
    assert len(stages)==5 and len({r['id'] for r in stages})==5
    commands=[' '.join(r['argv']) for r in stages]
    assert 'batch48b-tested' in commands[2]
    assert 'batch53b-tested' in commands[3]
    assert all('test_results' not in cmd and 'test_predictions' not in cmd
               and 'oracle-private' not in cmd for cmd in commands)
    assert not any(r.get('gpu') for r in stages)
    assert all(r['max_attempts']==1 for r in stages)


def test_audit_verifier_fails_on_missing_pin(tmp_path):
    audit_dir=tmp_path/'reports'
    audit_dir.mkdir()
    (audit_dir/'batch48b_dev_cluster.json').write_text('{}')
    with pytest.raises(FileNotFoundError):
        verifier.verify(tmp_path,audit_dir)


def test_runner_uses_new_plan():
    text=(ROOT/'automation/runner_batch54.py').read_text()
    assert 'batch54_dev_scene_cluster_safety_2026-11-03.json' in text
