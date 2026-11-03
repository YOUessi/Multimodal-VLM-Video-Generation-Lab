"""No network or media writes. Fail-closed resumption and runner isolation."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    # Script module imports its sibling by name, just as the real Python CLI.
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


@pytest.fixture
def resume_module():
    return _load(ROOT / "experiments/batch77_resumable_download.py", "_test_batch77_resume")


def test_rejected_path_in_error(resume_module):
    for s in ["KINETICS_SHARD_RETRY_EXHAUSTED:../evil.tar.gz:1/2",
              "KINETICS_SHARD_RETRY_EXHAUSTED:/tmp/k700_val_01.tar.gz:1/2",
              "KINETICS_SHARD_RETRY_EXHAUSTED:k700_val_01.zip:1/2"]:
        with pytest.raises(ValueError):
            resume_module._partial_for_exception(s)


def test_only_exact_retry_exhaustion_is_deferred(resume_module, tmp_path, monkeypatch):
    monkeypatch.setattr(resume_module, "DATA", tmp_path)
    d = tmp_path / "shards"
    d.mkdir()
    partial = d / "k700_val_594.tar.tar.gz.partial"
    partial.write_bytes(b"begin")

    def progressing():
        partial.write_bytes(b"begin-more-bytes")
        raise RuntimeError("KINETICS_SHARD_RETRY_EXHAUSTED:k700_val_594.tar.gz:14/20:rc=28")
    monkeypatch.setattr(resume_module, "original_download", progressing)
    assert resume_module.run_download() == 75


def test_zero_progress_remains_fatal(resume_module, tmp_path, monkeypatch):
    monkeypatch.setattr(resume_module, "DATA", tmp_path)
    (tmp_path / "shards").mkdir()
    def stagnant():
        raise RuntimeError("KINETICS_SHARD_RETRY_EXHAUSTED:k700_val_594.tar.gz:0/20:rc=28")
    monkeypatch.setattr(resume_module, "original_download", stagnant)
    with pytest.raises(RuntimeError, match="RETRY_EXHAUSTED"):
        resume_module.run_download()


def test_non_retry_failure_remains_fatal(resume_module, tmp_path, monkeypatch):
    monkeypatch.setattr(resume_module, "DATA", tmp_path)
    (tmp_path / "shards").mkdir()
    def invalid():
        (tmp_path / "shards" / "k700_val_594.tar.tar.gz.partial").write_bytes(b"changed")
        raise RuntimeError("KINETICS_SHARD_REMOTE_IDENTITY_DRIFT:k700_val_594.tar.gz")
    monkeypatch.setattr(resume_module, "original_download", invalid)
    with pytest.raises(RuntimeError, match="IDENTITY_DRIFT"):
        resume_module.run_download()


def test_real_success_returns_zero(resume_module, monkeypatch):
    monkeypatch.setattr(resume_module, "original_download", lambda: None)
    assert resume_module.run_download() == 0


def _runner_case(monkeypatch, ret, stage_status, attempts, maximum):
    mod = _load(ROOT / "automation/runner_batch77.py", "_test_batch77_runner")
    name = "R00_SELECT_AND_DOWNLOAD_42_FRESH_CLASS_SHARDS"
    state = {"stages": {name: {"status": stage_status, "returncode": ret,
                               "attempts": attempts}}, "last_event": "FAILED"}
    plan = {"stages": [{"id": name, "defer_exit_code": 75,
                        "max_attempts": maximum}]}
    saved = []
    monkeypatch.setattr(mod, "_original_one_stage", lambda p, s: False)
    monkeypatch.setattr(mod.runner, "save", lambda x: saved.append(x["last_event"]))
    assert not mod.one_stage_with_progress_deferral(plan, state)
    return state, saved


def test_deferred_progress_only_under_limit(monkeypatch):
    state, saves = _runner_case(monkeypatch, 75, "failed", 2, 96)
    item = state["stages"]["R00_SELECT_AND_DOWNLOAD_42_FRESH_CLASS_SHARDS"]
    assert item["status"] == "pending"
    assert item["progress_deferrals"] == 1
    assert saves == ["DEFERRED_VERIFIED_PROGRESS:R00_SELECT_AND_DOWNLOAD_42_FRESH_CLASS_SHARDS"]


@pytest.mark.parametrize("code,attempts,maximum", [(1, 2, 96), (75, 96, 96), (28, 2, 96)])
def test_runner_does_not_reset_real_failure(monkeypatch, code, attempts, maximum):
    state, saved = _runner_case(monkeypatch, code, "failed", attempts, maximum)
    assert state["stages"]["R00_SELECT_AND_DOWNLOAD_42_FRESH_CLASS_SHARDS"]["status"] == "failed"
    assert not saved


def test_runner_does_not_reset_success(monkeypatch):
    state, saved = _runner_case(monkeypatch, 75, "done", 2, 96)
    assert state["stages"]["R00_SELECT_AND_DOWNLOAD_42_FRESH_CLASS_SHARDS"]["status"] == "done"
    assert not saved
