#!/usr/bin/env python3
"""Read-only audit of a frozen long-running experiment; never opens labels or media.

For Batch59: assess runner/cron/GPU exclusivity, byte-level download progress,
source SHA immutability, and *estimated* (not guaranteed) storage headroom.
Nothing in this module executes or advances an experiment stage.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MARKER = "MVVG_PROJECT4_10MIN"
EXPECTED_SHARDS = 42
MIN_RESERVE_GIB = 10
STALE_SECONDS = 3600
READY = re.compile(r"^k700_val_\d{3}\.tar\.gz$")
PROGRESS = re.compile(
    r"KINETICS_SHARD_RESUME_PROGRESS (k700_val_\d{3}\.tar\.gz)"
    r" attempt=(\d+) rc=(\d+) bytes=(\d+)/(\d+)"
)
FINISHED = "ALL_APPROVED_STAGES_DONE"


def call(args: list[str], timeout: int = 12) -> str | None:
    """External commands are read-only and time-limited; None is an unknown."""
    try:
        result = subprocess.run(args, capture_output=True, text=True,
                                timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    return result.stdout


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    seq = sorted(values)
    return seq[min(len(seq) - 1, max(0, math.ceil(p * len(seq)) - 1))]


def parse_download_log(raw: str) -> dict[str, Any]:
    """Never mistake curl rc=28 (recoverable timeout) for a completed shard."""
    last: dict[str, Any] | None = None
    ready: set[str] = set()
    attempts = 0
    for line in raw.splitlines():
        match = PROGRESS.search(line)
        if match:
            name, tries, rc, size, total = match.groups()
            attempts += 1
            last = {"name": name, "attempt": int(tries), "curl_rc": int(rc),
                    "bytes": int(size), "total_bytes": int(total)}
        if "KINETICS_BATCH45_SHARD_READY" in line:
            chunks = line.split()
            if len(chunks) > 1 and chunks[1].isdigit():
                ready.add(f"k700_val_{int(chunks[1]):03}.tar.gz")
    if last:
        last["fraction"] = round(last["bytes"] / last["total_bytes"], 5) if last["total_bytes"] else None
    return {"last_progress": last, "ready_log_count": len(ready),
            "resume_events": attempts}


def inspect_shards(shards: Path, now: float, expected: int) -> dict[str, Any]:
    """Only inspect filenames and stat() metadata; never load video bytes."""
    completed: list[Path] = []
    partial: list[Path] = []
    if shards.is_dir():
        for file in shards.iterdir():
            if not file.is_file():
                continue
            if READY.fullmatch(file.name):
                completed.append(file)
            elif file.name.endswith(".partial"):
                partial.append(file)
    sizes = [file.stat().st_size for file in completed]
    pending = max(0, expected - len(completed))
    latest_partial = max(partial, key=lambda f: f.stat().st_mtime) if partial else None
    return {
        "complete": len(completed), "expected": expected, "remaining": pending,
        "complete_bytes": sum(sizes), "partial_count": len(partial),
        "active_partial": latest_partial.name if latest_partial else None,
        "active_partial_bytes": latest_partial.stat().st_size if latest_partial else None,
        "active_partial_age_seconds": max(0, round(now - latest_partial.stat().st_mtime))
        if latest_partial else None,
        "completed_shard_p50_bytes": round(statistics.median(sizes)) if sizes else 0,
        "completed_shard_p95_bytes": percentile(sizes, .95),
    }


def process_presence(ps: str) -> dict[str, Any]:
    runner: list[int] = []
    downloader: list[int] = []
    for line in ps.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        pid, command = int(fields[0]), fields[1]
        # Require Python as argv[0], excluding shell/grep lines containing snippets.
        executable = command.split(maxsplit=1)[0].rsplit("/", 1)[-1]
        if executable in {"python", "python3", "python3.10", "python3.11"} and "runner_batch59.py tick" in command:
            runner.append(pid)
        if executable in {"python", "python3", "python3.10", "python3.11"} and "kinetics_batch59_data.py download" in command:
            downloader.append(pid)
    return {"runner_pids": sorted(set(runner)),
            "downloader_pids": sorted(set(downloader))}


def parse_cron(cron: str) -> dict[str, Any]:
    matches = [line.strip() for line in cron.splitlines()
               if MARKER in line and not line.lstrip().startswith("#")]
    return {"matching_entries": len(matches),
            "valid_single_frozen_tick": (len(matches) == 1
                and matches[0].startswith("*/10 ")
                and "repo-batch59-tested" in matches[0]
                and "runner_batch59.py tick" in matches[0]
                and "ALL_APPROVED_STAGES_DONE" in matches[0])}


def analyze(state: dict[str, Any], checkout_sha: str | None, cron: str,
            ps: str, gpu: str | None, shards: dict[str, Any], log: str,
            free_bytes: int, now: float, done_marker: bool = False,
            remote_main_sha: str | None = None,
            tracked_checkout_dirty: bool | None = False) -> dict[str, Any]:
    """Pure analyzer: suitable for deterministic fault-injection regression."""
    problems: list[dict[str, str]] = []
    def flag(severity: str, code: str, detail: str) -> None:
        problems.append({"level": severity, "code": code, "detail": detail})

    stages = state.get("stages")
    if not isinstance(stages, dict) or not stages:
        flag("BLOCK", "MALFORMED_STATE", "stage map absent or empty")
        stages = {}
    counts: dict[str, int] = {}
    running: list[str] = []
    for name, row in stages.items():
        status = row.get("status") if isinstance(row, dict) else "invalid"
        counts[str(status)] = counts.get(str(status), 0) + 1
        if status == "running":
            running.append(name)
    if len(running) > 1:
        flag("BLOCK", "CONCURRENT_STAGES", "more than one stage marked running")
    if state.get("frozen_sha") != checkout_sha:
        flag("BLOCK", "FROZEN_SOURCE_DRIFT", "runner checkout SHA differs from frozen state")
    if tracked_checkout_dirty is None:
        flag("WARN", "FROZEN_TREE_STATUS_UNKNOWN", "tracked-file cleanliness could not be verified")
    elif tracked_checkout_dirty:
        flag("BLOCK", "FROZEN_TRACKED_FILE_MUTATION", "tracked source changed without a new frozen SHA")
    if done_marker and counts.get("pending", 0) + counts.get("running", 0) > 0:
        flag("BLOCK", "PREMATURE_DONE_MARKER", "terminal marker conflicts with pending stages")
    if not done_marker and not running and counts.get("pending", 0) == 0:
        flag("WARN", "TERMINAL_MARKER_MISSING", "all stages terminal but marker absent")
    cron_info = parse_cron(cron)
    if not cron_info["valid_single_frozen_tick"]:
        flag("BLOCK", "CRON_CONTRACT", "expected exactly one 10-minute frozen batch59 tick")
    pids = process_presence(ps)
    if len(pids["runner_pids"]) > 1 or len(pids["downloader_pids"]) > 1:
        flag("BLOCK", "DUPLICATE_BATCH59_PROCESS", "multiple runner or downloader processes")
    if running and not pids["runner_pids"]:
        flag("WARN", "STATE_RUNNING_WITHOUT_RUNNER", "state running; no matching runner PID")
    if running and running[0].startswith("R00") and not pids["downloader_pids"]:
        flag("WARN", "R00_NO_DOWNLOADER", "R00 running; downloader process not observed")

    # GPU compute output is a *compute-process list*, never total GPU memory usage.
    gpu_processes = [x.strip() for x in (gpu or "").splitlines() if x.strip()]
    if gpu is None:
        flag("WARN", "GPU_QUERY_UNKNOWN", "nvidia-smi query could not run")
    elif gpu_processes:
        flag("WARN", "GPU_COMPUTE_OCCUPIED", "unexpected compute processes; do not start inference")

    download = parse_download_log(log)
    if shards["complete"] > shards["expected"]:
        flag("BLOCK", "SHARD_COUNT_OVERFLOW", "completed archive count exceeds frozen plan")
    if download["ready_log_count"] > shards["complete"]:
        flag("WARN", "READY_VS_DISK_COUNT", "ready log count exceeds number of completed files")
    age = shards["active_partial_age_seconds"]
    if running and running[0].startswith("R00") and age is not None and age > STALE_SECONDS:
        flag("WARN", "NO_PARTIAL_BYTE_PROGRESS", "partial shard has not changed for over one hour")
    p95 = shards["completed_shard_p95_bytes"]
    estimated_remaining = p95 * shards["remaining"]
    reserve = MIN_RESERVE_GIB * 1024 ** 3
    headroom = free_bytes - estimated_remaining - reserve
    if free_bytes < reserve:
        flag("BLOCK", "DISK_BELOW_GUARD", "fewer than 10 GiB free; do not launch another batch")
    elif headroom < 0:
        flag("WARN", "DISK_FORECAST_TIGHT", "p95-shard forecast would cross 10 GiB reserve")
    if remote_main_sha and remote_main_sha != checkout_sha:
        # Expected: the running batch is frozen and main has advanced.
        # Never treat normal main movement as source drift.
        pass
    return {
        "schema": "batch74-readonly-audit-v1",
        "checked_unix": round(now), "frozen_sha": state.get("frozen_sha"),
        "checkout_sha": checkout_sha, "remote_main_sha": remote_main_sha,
        "tracked_checkout_dirty": tracked_checkout_dirty,
        "stages": {"total": len(stages), "counts": counts,
                   "running": running, "done_marker": done_marker},
        "cron": cron_info, "processes": pids,
        "gpu_compute_process_count": len(gpu_processes) if gpu is not None else None,
        "download": {**shards, **download},
        "storage": {"free_bytes": free_bytes,
                    "remaining_at_observed_p95_bytes": estimated_remaining,
                    "reserve_bytes": reserve, "forecast_headroom_bytes": headroom,
                    "forecast_is_heuristic": True},
        "alerts": problems, "status": "BLOCK" if any(x["level"] == "BLOCK" for x in problems)
        else "WARN" if problems else "OK",
        "contract": "read-only; no changes to frozen runner, GPU, cron, shards, oracle or predictions",
    }


def snapshot(root: Path, now: float | None = None) -> dict[str, Any]:
    root = root.resolve()
    state_path = root / ".automation/state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).timestamp() if now is None else now
    log_path = root / ".automation/logs/R00_SELECT_AND_DOWNLOAD_42_FRESH_CLASS_SHARDS.log"
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    shard_info = inspect_shards(root / "data/tapvid_kinetics_batch59/shards",
                                now, EXPECTED_SHARDS)
    disk = shutil.disk_usage(root)
    checkout_sha = (call(["git", "-C", str(root), "rev-parse", "HEAD"]) or "").strip() or None
    tracked_status = call(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"])
    tracked_dirty = bool(tracked_status.strip()) if tracked_status is not None else None
    remote_line = call(["git", "-C", str(root), "ls-remote", "origin", "refs/heads/main"], timeout=15)
    remote_main_sha = (remote_line.split()[0] if remote_line and remote_line.split() else None)
    cron = call(["crontab", "-l"]) or ""
    ps = call(["ps", "-eo", "pid=,args="]) or ""
    gpu = call(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader"])
    return analyze(state, checkout_sha, cron, ps, gpu, shard_info, log,
                   disk.free, now, (root / ".automation" / FINISHED).exists(),
                   remote_main_sha, tracked_dirty)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="FROZEN checkout path")
    args = parser.parse_args()
    result = snapshot(args.root)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2))
    # Diagnostics should not interrupt a running approved stage; structured status
    # is emitted for orchestration, but this read-only command exits zero.


if __name__ == "__main__":
    main()
