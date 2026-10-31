#!/usr/bin/env python3
"""Bounded, resumable model-weight recovery for the approved local diffusion stage.

This helper is independent of the frozen experimental source checkout. It never
updates any tracked experiment code, never downloads other files and never runs GPU.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os
import sys
import traceback

from filelock import FileLock, Timeout

BASE = Path("/home/you/projects/Multimodal-VLM-Video-Generation-Lab")
LOCAL_DIR = BASE / "models" / "zeroscope_v2_576w_safe"
TARGET = LOCAL_DIR / "unet" / "diffusion_pytorch_model.safetensors"
CACHE_LOCK_DIR = LOCAL_DIR / ".cache" / "huggingface" / "download" / "unet"
STATE = BASE / "ops" / "download_state.json"

REPO = "cerspense/zeroscope_v2_576w"
COMMIT = "1f35e3dc3350d649f342bb7618fe34ac8fae5c9e"
FILENAME = "unet/diffusion_pytorch_model.safetensors"
EXPECTED_SIZE = 2822650224
MAX_ATTEMPTS = 4


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def persist(state):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = timestamp()
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, STATE)


def download_is_active():
    """Honor Hugging Face's own locks; don't compete with an active manual download."""
    for lock_path in CACHE_LOCK_DIR.glob("*.lock"):
        try:
            lock = FileLock(str(lock_path), timeout=0)
            lock.acquire(timeout=0)
            lock.release()
        except Timeout:
            return True
    return False


def verify():
    if not TARGET.is_file() or TARGET.stat().st_size != EXPECTED_SIZE:
        return False
    from safetensors import safe_open
    with safe_open(str(TARGET), framework="pt", device="cpu") as f:
        return bool(f.keys())


def main():
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        "model": REPO, "revision": COMMIT, "expected_bytes": EXPECTED_SIZE,
        "attempts": 0, "status": "new"
    }
    if state.get("revision") != COMMIT or state.get("expected_bytes") != EXPECTED_SIZE:
        raise RuntimeError("RECOVERY_CONFIG_DRIFT")
    if verify():
        state["status"] = "complete"
        state["file_bytes"] = EXPECTED_SIZE
        persist(state)
        print("WEIGHTS_VERIFIED", TARGET, flush=True)
        return
    if download_is_active():
        state["status"] = "another_download_in_progress"
        persist(state)
        print("SKIPPING: Hugging Face lock is held by another download", flush=True)
        return
    if state["attempts"] >= MAX_ATTEMPTS:
        state["status"] = "blocked_retry_budget"
        persist(state)
        print("BLOCKED: retry budget exhausted; requires a new verified download plan", flush=True)
        return

    state["attempts"] += 1
    state["status"] = "resuming"
    persist(state)
    print(f"RESUMING attempt={state['attempts']}/{MAX_ATTEMPTS}", flush=True)
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "20"
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    from huggingface_hub import hf_hub_download
    try:
        result = hf_hub_download(
            repo_id=REPO,
            filename=FILENAME,
            revision=COMMIT,
            local_dir=str(LOCAL_DIR)
        )
        if not verify():
            raise RuntimeError(f"INCOMPLETE_OR_INVALID: {Path(result).stat().st_size} bytes")
        state["status"] = "complete"
        state["file_bytes"] = EXPECTED_SIZE
        print("WEIGHTS_VERIFIED", TARGET, flush=True)
    except Exception as exc:
        state["status"] = "download_error"
        state["last_error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        print("DOWNLOAD_RECOVERABLE_ERROR", state["last_error"], flush=True)
        persist(state)
        sys.exit(1)
    persist(state)


if __name__ == "__main__":
    main()
