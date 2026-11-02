"""Batch46C: deterministic, offline shard resume and integrity regression tests."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import kinetics_batch45_data as subject


def _partial(shard: Path) -> Path:
    return shard.with_suffix(".tar.gz.partial")


def _mock_head(monkeypatch, identity=(8, '"etag-1"')):
    monkeypatch.setattr(subject, "remote_shard_identity", lambda url: identity)
    monkeypatch.setattr(subject, "verify_complete_gzip", lambda path: None)


def test_resumes_existing_partial_without_restarting(tmp_path, monkeypatch):
    shard = tmp_path / "k700_val_163.tar.gz"
    _partial(shard).write_bytes(b"abc")
    _mock_head(monkeypatch)
    seen = []

    def call(cmd, **kwargs):
        seen.append(cmd)
        assert cmd[cmd.index("-C") + 1] == "-"
        assert cmd[cmd.index("-o") + 1] == str(_partial(shard))
        with _partial(shard).open("ab") as stream:
            stream.write(b"defgh")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(subject.subprocess, "run", call)
    assert subject.download_one_shard(shard, "https://example.test/file") == (
        8, '"etag-1"'
    )
    assert shard.read_bytes() == b"abcdefgh"
    assert not _partial(shard).exists()
    assert len(seen) == 1


def test_timeout_then_resume_preserves_partial_bytes(tmp_path, monkeypatch):
    shard = tmp_path / "k700_val_163.tar.gz"
    _mock_head(monkeypatch)
    calls = []

    def call(cmd, **kwargs):
        calls.append(cmd)
        with _partial(shard).open("ab") as stream:
            stream.write(b"abc" if len(calls) == 1 else b"defgh")
        if len(calls) == 1:
            return SimpleNamespace(returncode=28, stderr="timeout")
        assert cmd[cmd.index("-C") + 1] == "-"
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(subject.subprocess, "run", call)
    subject.download_one_shard(shard, "https://example.test/file")
    assert shard.read_bytes() == b"abcdefgh"
    assert len(calls) == 2


def test_three_no_progress_attempts_fail_and_preserve_partial(tmp_path, monkeypatch):
    shard = tmp_path / "k700_val_163.tar.gz"
    _partial(shard).write_bytes(b"abc")
    _mock_head(monkeypatch)
    calls = []

    def call(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=28, stderr="network stalled")

    monkeypatch.setattr(subject.subprocess, "run", call)
    with pytest.raises(RuntimeError, match="KINETICS_SHARD_NO_PROGRESS"):
        subject.download_one_shard(shard, "https://example.test/file")
    assert len(calls) == 3
    assert _partial(shard).read_bytes() == b"abc"
    assert not shard.exists()


def test_rejects_oversized_partial_without_running_curl(tmp_path, monkeypatch):
    shard = tmp_path / "k700_val_163.tar.gz"
    _partial(shard).write_bytes(b"more than expected")
    _mock_head(monkeypatch)
    monkeypatch.setattr(subject.subprocess, "run", lambda *_a, **_k:
                        pytest.fail("must not download oversized partial"))
    with pytest.raises(RuntimeError, match="KINETICS_SHARD_PARTIAL_OVERSIZE"):
        subject.download_one_shard(shard, "https://example.test/file")
    assert _partial(shard).is_file()


def test_rejects_successful_curl_reporting_truncated_file(tmp_path, monkeypatch):
    shard = tmp_path / "k700_val_163.tar.gz"
    _mock_head(monkeypatch)

    def call(cmd, **kwargs):
        _partial(shard).write_bytes(b"abcd")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(subject.subprocess, "run", call)
    with pytest.raises(RuntimeError, match="KINETICS_SHARD_TRUNCATED_SUCCESS"):
        subject.download_one_shard(shard, "https://example.test/file")
    assert _partial(shard).read_bytes() == b"abcd"
    assert not shard.exists()


def test_remote_etag_drift_blocks_final_publication(tmp_path, monkeypatch):
    shard = tmp_path / "k700_val_163.tar.gz"
    identities = iter([(8, '"etag-1"'), (8, '"etag-2"')])
    monkeypatch.setattr(subject, "remote_shard_identity", lambda _: next(identities))

    def call(cmd, **kwargs):
        _partial(shard).write_bytes(b"abcdefgh")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(subject.subprocess, "run", call)
    with pytest.raises(RuntimeError, match="KINETICS_SHARD_REMOTE_IDENTITY_DRIFT"):
        subject.download_one_shard(shard, "https://example.test/file")
    assert _partial(shard).read_bytes() == b"abcdefgh"
    assert not shard.exists()


def test_existing_shard_wrong_size_fails_closed(tmp_path, monkeypatch):
    shard = tmp_path / "k700_val_163.tar.gz"
    shard.write_bytes(b"short")
    _mock_head(monkeypatch)
    with pytest.raises(RuntimeError, match="KINETICS_SHARD_FINAL_SIZE_MISMATCH"):
        subject.download_one_shard(shard, "https://example.test/file")
    assert shard.read_bytes() == b"short"


def test_curl_flags_enforce_stall_timeout_and_bounded_attempt(tmp_path, monkeypatch):
    shard = tmp_path / "k700_val_163.tar.gz"
    _mock_head(monkeypatch)
    flags = []

    def call(cmd, **kwargs):
        flags.append(cmd)
        assert cmd[cmd.index("--speed-time") + 1] == str(subject.SHARD_STALL_SECONDS)
        assert cmd[cmd.index("--speed-limit") + 1] == str(
            subject.SHARD_MIN_BYTES_PER_SECOND
        )
        assert int(cmd[cmd.index("--max-time") + 1]) <= subject.SHARD_CURL_SLICE_SECONDS
        assert kwargs["timeout"] <= subject.SHARD_CURL_SLICE_SECONDS + 12
        return SimpleNamespace(returncode=22, stderr="404")

    monkeypatch.setattr(subject.subprocess, "run", call)
    with pytest.raises(RuntimeError, match="KINETICS_SHARD_NONRETRYABLE"):
        subject.download_one_shard(shard, "https://example.test/file")
    assert len(flags) == 1


def test_complete_gzip_verification_rejects_corruption(tmp_path):
    shard = tmp_path / "bad.tar.gz.partial"
    shard.write_bytes(b"not a gzip stream")
    with pytest.raises(RuntimeError, match="KINETICS_SHARD_GZIP_INTEGRITY_FAILED"):
        subject.verify_complete_gzip(shard)
    assert shard.exists()


def test_completed_partial_is_verified_and_promoted_without_curl(tmp_path, monkeypatch):
    shard = tmp_path / "k700_val_163.tar.gz"
    _partial(shard).write_bytes(b"abcdefgh")
    _mock_head(monkeypatch)
    monkeypatch.setattr(
        subject.subprocess, "run",
        lambda *_a, **_k: pytest.fail("full partial should need no curl"),
    )
    assert subject.download_one_shard(shard, "https://example.test/file") == (
        8, '"etag-1"'
    )
    assert shard.read_bytes() == b"abcdefgh"


def test_stall_counter_resets_after_new_bytes(tmp_path, monkeypatch):
    shard = tmp_path / "k700_val_163.tar.gz"
    _mock_head(monkeypatch)
    calls = []

    def call(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) in (2, 4):
            with _partial(shard).open("ab") as stream:
                stream.write(b"abc" if len(calls) == 2 else b"de")
        elif len(calls) == 5:
            with _partial(shard).open("ab") as stream:
                stream.write(b"fgh")
        return SimpleNamespace(
            returncode=0 if len(calls) == 5 else 28, stderr="timeout"
        )

    monkeypatch.setattr(subject.subprocess, "run", call)
    subject.download_one_shard(shard, "https://example.test/file")
    assert len(calls) == 5
    assert shard.read_bytes() == b"abcdefgh"
