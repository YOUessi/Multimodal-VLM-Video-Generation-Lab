"""Batch46E: transient remote HEAD retries preserve identity and frozen media."""
from __future__ import annotations

from http.client import RemoteDisconnected
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import kinetics_batch45_data as subject


class Head:
    def __init__(self, *, size="8", etag='"etag-1"', ranges="bytes", status=200):
        self.status = status
        self.headers = {
            "Content-Length": size,
            "ETag": etag,
            "Accept-Ranges": ranges,
        }

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_remote_disconnected_twice_then_success(monkeypatch):
    seen, sleeps = [], []
    responses = iter([RemoteDisconnected("closed before headers"),
                      RemoteDisconnected("closed again"), Head()])

    def fake_open(request, timeout):
        assert request.get_method() == "HEAD"
        assert timeout == 25
        seen.append(1)
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(subject, "urlopen", fake_open)
    monkeypatch.setattr(subject.time, "sleep", sleeps.append)
    assert subject.remote_shard_identity("https://example.test/archive") == (8, '"etag-1"')
    assert len(seen) == 3
    assert sleeps == [1, 2]


def test_retry_exhaustion_preserves_partial_without_curl(tmp_path, monkeypatch):
    shard = tmp_path / "k700_val_292.tar.gz"
    partial = shard.with_suffix(".tar.gz.partial")
    partial.write_bytes(b"existing verified-by-future-gzip-bytes")
    seen, sleeps = [], []
    def fake_open(*_a, **_k):
        seen.append(True)
        raise RemoteDisconnected("connection closed")
    monkeypatch.setattr(subject, "urlopen", fake_open)
    monkeypatch.setattr(subject.time, "sleep", sleeps.append)
    monkeypatch.setattr(subject.subprocess, "run", lambda *_a, **_k:
                        pytest.fail("must never curl if HEAD identity is unavailable"))
    with pytest.raises(RuntimeError, match="KINETICS_SHARD_HEAD_RETRY_EXHAUSTED"):
        subject.download_one_shard(shard, "https://example.test/archive")
    assert len(seen) == subject.SHARD_HEAD_MAX_ATTEMPTS == 5
    assert sleeps == [1, 2, 4, 8]
    assert partial.read_bytes() == b"existing verified-by-future-gzip-bytes"
    assert not shard.exists()


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_http_transient_then_valid_identity(monkeypatch, status):
    seen, responses = [], iter([HTTPError("u", status, "transient", {}, None), Head()])
    def fake_open(*_a, **_k):
        seen.append(1)
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response
    monkeypatch.setattr(subject, "urlopen", fake_open)
    monkeypatch.setattr(subject.time, "sleep", lambda *_: None)
    assert subject.remote_shard_identity("https://example.test/archive") == (8, '"etag-1"')
    assert len(seen) == 2


@pytest.mark.parametrize("status", [400, 403, 404, 410])
def test_http_permanent_fails_immediately(monkeypatch, status):
    seen = []
    def fake_open(*_a, **_k):
        seen.append(1)
        raise HTTPError("u", status, "permanent", {}, None)
    monkeypatch.setattr(subject, "urlopen", fake_open)
    monkeypatch.setattr(subject.time, "sleep", lambda *_:
                        pytest.fail("must not retry permanent errors"))
    with pytest.raises(RuntimeError, match=f"KINETICS_SHARD_REMOTE_HEAD_HTTP_FATAL:{status}"):
        subject.remote_shard_identity("https://example.test/archive")
    assert len(seen) == 1


@pytest.mark.parametrize("headers", [
    {"etag": None},
    {"ranges": "none"},
    {"size": "not-an-int"},
    {"size": "0"},
    {"status": 206},
])
def test_invalid_head_identity_fails_closed_without_retry(monkeypatch, headers):
    calls = []
    def fake_open(*_a, **_k):
        calls.append(1)
        return Head(**headers)
    monkeypatch.setattr(subject, "urlopen", fake_open)
    monkeypatch.setattr(subject.time, "sleep", lambda *_:
                        pytest.fail("must not retry an invalid response"))
    with pytest.raises(RuntimeError, match="KINETICS_SHARD_REMOTE_(IDENTITY_MISSING|SIZE_INVALID)"):
        subject.remote_shard_identity("https://example.test/archive")
    assert len(calls) == 1


def test_etag_drift_still_aborts_publication_after_curl(tmp_path, monkeypatch):
    shard = tmp_path / "k700_val_292.tar.gz"
    partial = shard.with_suffix(".tar.gz.partial")
    heads = iter([Head(), Head(etag='"etag-CHANGED"')])
    monkeypatch.setattr(subject, "urlopen", lambda *_a, **_k: next(heads))
    def curl(_cmd, **_kwargs):
        partial.write_bytes(b"abcdefgh")
        return SimpleNamespace(returncode=0, stderr="")
    monkeypatch.setattr(subject.subprocess, "run", curl)
    with pytest.raises(RuntimeError, match="KINETICS_SHARD_REMOTE_IDENTITY_DRIFT"):
        subject.download_one_shard(shard, "https://example.test/archive")
    assert partial.read_bytes() == b"abcdefgh"
    assert not shard.exists()


def test_url_error_network_retry(monkeypatch):
    responses = iter([URLError("temporary DNS"), Head()])
    def fake_open(*_a, **_k):
        r = next(responses)
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(subject, "urlopen", fake_open)
    monkeypatch.setattr(subject.time, "sleep", lambda *_: None)
    assert subject.remote_shard_identity("https://example.test/archive") == (8, '"etag-1"')
