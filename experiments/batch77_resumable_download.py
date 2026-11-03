#!/usr/bin/env python3
"""Safely defer a *progressing* Batch59 Kinetics shard after a bounded download session.

Never marks the 42-video cohort complete until the original Batch59 downloader
returns success. Previous verified archives and partial bytes remain untouched.
Only the exact retry-budget-exhausted error with NEW persistent bytes may defer.
"""
from __future__ import annotations

import sys
from pathlib import Path

from kinetics_batch59_data import DATA, main as original_download

DEFER_EXIT = 75
RETRY_PREFIX = "KINETICS_SHARD_RETRY_EXHAUSTED:"


def _partial_for_exception(message: str) -> Path:
    if not message.startswith(RETRY_PREFIX):
        raise ValueError("NOT_A_SHARD_RETRY_EXHAUSTION")
    name = message[len(RETRY_PREFIX):].split(":", 1)[0]
    # Never allow an exception string to escape the pinned shard directory.
    if not name.startswith("k700_val_") or not name.endswith(".tar.gz"):
        raise ValueError("INVALID_SHARD_NAME")
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("UNSAFE_SHARD_NAME")
    return (DATA / "shards" / name).with_suffix(".tar.gz.partial")


def _sizes() -> dict[Path, int]:
    directory = DATA / "shards"
    return {path: path.stat().st_size
            for path in directory.glob("k700_val_*.tar.tar.gz.partial")
            if path.is_file()}


def run_download() -> int:
    before = _sizes()
    try:
        # The original downloader enforces class-disjoint preflight, immutable
        # remote HEAD identity, gzip integrity, SHA256 and manifest pinning.
        # Keep the original CLI contract, disk guard, and all other exceptions.
        original_download()
        return 0
    except RuntimeError as exc:
        message = str(exc)
        if not message.startswith(RETRY_PREFIX):
            raise
        partial = _partial_for_exception(message)
        previous = before.get(partial, 0)
        current = partial.stat().st_size if partial.is_file() else 0
        # Zero progress, deleted files and unrecognized errors are fatal.
        if current <= previous:
            raise
        print(
            f"BATCH77_PROGRESS_DEFERRED shard={partial.name} "
            f"bytes_before={previous} bytes_after={current}",
            flush=True,
        )
        return DEFER_EXIT


if __name__ == "__main__":
    if sys.argv[1:] != ["download"]:
        raise SystemExit("USAGE: batch77_resumable_download.py download")
    sys.exit(run_download())
