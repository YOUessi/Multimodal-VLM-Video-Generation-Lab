"""Regression: long, advancing video-shard transfers must not hit old retry cap."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import kinetics_batch45_data as downloader


def _partial(p: Path) -> Path:
    return p.with_suffix(".tar.gz.partial")


def test_steady_progress_succeeds_after_previous_sixteen_attempt_cutoff(tmp_path, monkeypatch):
    shard = tmp_path / "k700_val_635.tar.gz"
    size, attempts = 21, []
    clock = {"now": 0.0}
    monkeypatch.setattr(downloader.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(downloader, "remote_shard_identity",
                        lambda _: (size, '"immutable-etag"'))
    monkeypatch.setattr(downloader, "verify_complete_gzip", lambda _: None)

    def curl(argv, **kwargs):
        attempts.append(argv)
        assert argv[argv.index("--max-time") + 1] == str(
            downloader.SHARD_CURL_SLICE_SECONDS)
        assert kwargs["timeout"] <= downloader.SHARD_CURL_SLICE_SECONDS + 12
        assert argv[argv.index("-o") + 1] == str(_partial(shard))
        if len(attempts) > 1:
            assert argv[argv.index("-C") + 1] == "-"
        with _partial(shard).open("ab") as f:
            f.write(b"x")
        clock["now"] += 360.0
        return SimpleNamespace(
            returncode=0 if len(attempts) == size else 28,
            stderr="temporary timeout, bytes advanced",
        )

    monkeypatch.setattr(downloader.subprocess, "run", curl)
    assert downloader.SHARD_MAX_ATTEMPTS > 16
    assert downloader.SHARD_BUDGET_SECONDS > 7200
    assert downloader.download_one_shard(shard, "https://example.test/pinned") == (
        size, '"immutable-etag"')
    assert len(attempts) == 21
    assert clock["now"] == 21 * 360
    assert shard.read_bytes() == b"x" * size
    assert not _partial(shard).exists()


def test_no_progress_still_fails_after_three_attempts_and_preserves_bytes(
    tmp_path, monkeypatch,
):
    shard = tmp_path / "k700_val_635.tar.gz"
    _partial(shard).write_bytes(b"safe")
    monkeypatch.setattr(downloader, "remote_shard_identity",
                        lambda _: (10, '"immutable-etag"'))
    calls = []
    def no_progress(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=28, stderr="network unavailable")
    monkeypatch.setattr(downloader.subprocess, "run", no_progress)
    with pytest.raises(RuntimeError, match="KINETICS_SHARD_NO_PROGRESS"):
        downloader.download_one_shard(shard, "https://example.test/pinned")
    assert len(calls) == 3
    assert _partial(shard).read_bytes() == b"safe"
    assert not shard.exists()


def test_cannot_publish_truncated_success_under_larger_budget(tmp_path, monkeypatch):
    shard = tmp_path / "k700_val_635.tar.gz"
    monkeypatch.setattr(downloader, "remote_shard_identity",
                        lambda _: (10, '"immutable-etag"'))
    def truncated(argv, **kwargs):
        _partial(shard).write_bytes(b"short")
        return SimpleNamespace(returncode=0, stderr="")
    monkeypatch.setattr(downloader.subprocess, "run", truncated)
    with pytest.raises(RuntimeError, match="KINETICS_SHARD_TRUNCATED_SUCCESS"):
        downloader.download_one_shard(shard, "https://example.test/pinned")
    assert _partial(shard).read_bytes() == b"short"
    assert not shard.exists()


def test_future_frozen_plan_covers_updated_bounded_total_shard_time():
    import json
    root = Path(__file__).resolve().parents[1]
    for rel in (
        "automation/plan.json",
        "automation/plans/batch48_fresh_independent_safety_dev_select_2026-11-02.json",
    ):
        plan = json.loads((root / rel).read_text(encoding="utf-8"))
        stage = next(s for s in plan["stages"]
                     if s["id"] == "R00_SELECT_AND_DOWNLOAD_12_FRESH_CLASS_SHARDS")
        assert stage["timeout_seconds"] >= (12 * downloader.SHARD_BUDGET_SECONDS + 600)
        assert stage["max_attempts"] == 1
