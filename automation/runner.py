#!/usr/bin/env python3
"""Bounded 10-minute local scheduler: frozen source, single writer, GPU admission, fail closed."""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / ".automation"
PLAN = ROOT / "automation" / "plan.json"
GLOBAL_GPU_LOCK = Path("/tmp/tang_shared_gpu_experiments.lock")
UTC = dt.timezone.utc

def stamp():
    return dt.datetime.now(UTC).isoformat(timespec="seconds")

def atomic_json(path: Path, doc: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def gitsha():
    status = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True, text=True, timeout=10,
    )
    if status.returncode:
        raise RuntimeError("GIT_STATUS_NOT_READY: checkout the source repo before arming")
    if status.stdout.strip():
        raise RuntimeError("DIRTY_SOURCE_TREE_CANNOT_FREEZE_AUTOMATION")
    p = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=10,
    )
    if p.returncode:
        raise RuntimeError("GIT_NOT_READY: checkout the source repo before arming")
    return p.stdout.strip()

def prepare():
    PRIVATE.mkdir(parents=True, exist_ok=True)
    (PRIVATE / "logs").mkdir(exist_ok=True)
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    statefile = PRIVATE / "state.json"
    if statefile.exists():
        state = json.loads(statefile.read_text(encoding="utf-8"))
        if state.get("plan_version") != plan["version"]:
            raise RuntimeError("PLAN_VERSION_CHANGED: approve new frozen plan before continuing")
    else:
        state = {"plan_version": plan["version"], "frozen_sha": gitsha(),
                 "created_at": stamp(), "last_check": None, "last_event": "initialized",
                 "stages": {x["id"]: {"status": "pending", "attempts": 0} for x in plan["stages"]}}
        atomic_json(statefile, state)
    return plan, state

def save(state: dict):
    state["last_check"] = stamp()
    atomic_json(PRIVATE / "state.json", state)

def gpu_ready(required_mb=11000):
    if not shutil.which("nvidia-smi"):
        return False, "NO_NVIDIA_SMI"
    p = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=12)
    if p.returncode:
        return False, "GPU_QUERY_FAILED"
    try:
        free = int(p.stdout.strip().splitlines()[0].strip())
    except (ValueError, IndexError):
        return False, "GPU_FREE_PARSE_FAILED"
    if free < required_mb:
        return False, f"GPU_FREE_TOO_LOW:{free}MiB"
    p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=12)
    if p.returncode:
        return False, "GPU_PROCESS_QUERY_FAILED"
    proc = [x.strip() for x in p.stdout.splitlines() if x.strip() and x.strip() != "No running processes found"]
    if proc:
        return False, "GPU_IN_USE_BY_OTHER_PROCESS:" + ",".join(proc[:8])
    return True, f"GPU_OK:{free}MiB"

def one_stage(plan: dict, state: dict):
    if gitsha() != state["frozen_sha"]:
        state["last_event"] = "SOURCE_DRIFT: froze " + state["frozen_sha"] + ", actual " + gitsha()
        save(state)
        return False
    if not plan.get("enabled", False):
        state["last_event"] = "DISABLED_BY_PLAN"
        save(state)
        return False
    stages = plan["stages"]
    pending = next((x for x in stages if state["stages"].get(x["id"], {}).get("status") != "done"), None)
    if pending is None:
        state["last_event"] = "ALL_APPROVED_STAGES_DONE"
        save(state)
        return False
    rec = state["stages"][pending["id"]]
    if rec["status"] in ("blocked", "failed", "paused"):
        state["last_event"] = "STOPPED_AT_" + pending["id"] + ":" + rec["status"]
        save(state)
        return False
    required = pending.get("requires_files", [])
    missing = [x for x in required if not (ROOT / x).is_file()]
    if missing:
        state["last_event"] = "WAITING_FOR_REQUIRED_FILES:" + ",".join(missing[:5])
        save(state)
        return False
    max_attempts = int(pending.get("max_attempts", 1))
    if rec["attempts"] >= max_attempts:
        rec["status"] = "failed"
        state["last_event"] = "ATTEMPT_LIMIT_REACHED:" + pending["id"]
        save(state)
        return False
    gpu_fd = None
    if pending.get("gpu"):
        GLOBAL_GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
        gpu_fd = GLOBAL_GPU_LOCK.open("a+")
        try:
            fcntl.flock(gpu_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            state["last_event"] = "SHARED_GPU_LOCK_BUSY"
            save(state)
            gpu_fd.close()
            return False
        ok, status = gpu_ready(int(pending.get("min_free_mib", 11000)))
        if not ok:
            state["last_event"] = status
            save(state)
            gpu_fd.close()
            return False
    stageid = pending["id"]
    rec.update({"status": "running", "attempts": rec["attempts"] + 1, "started_at": stamp()})
    state["last_event"] = "RUNNING:" + stageid
    save(state)
    command = [z.replace("{python}", sys.executable) for z in pending["argv"]]
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false", "MVVG_WORKER": "1",
                "PYTHONUNBUFFERED": "1"})
    log_path = PRIVATE / "logs" / (stageid + ".log")
    try:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{stamp()}] BEGIN stage={stageid} sha={state['frozen_sha']} command={command}\n")
            log.flush()
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=int(pending["timeout_seconds"]))
            code = result.returncode
            log.write(f"[{stamp()}] END stage={stageid} returncode={code}\n")
        rec.update({"status": "done" if code == 0 else "failed", "returncode": code,
                    "finished_at": stamp(), "log": str(log_path.relative_to(ROOT))})
        state["last_event"] = ("DONE:" if code == 0 else "FAILED:") + stageid
    except subprocess.TimeoutExpired:
        rec.update({"status": "failed", "reason": "TIMEOUT", "finished_at": stamp()})
        state["last_event"] = "TIMEOUT:" + stageid
    except Exception as exc:
        rec.update({"status": "failed", "reason": repr(exc), "finished_at": stamp()})
        state["last_event"] = "EXCEPTION:" + stageid
    finally:
        save(state)
        if gpu_fd:
            fcntl.flock(gpu_fd.fileno(), fcntl.LOCK_UN)
            gpu_fd.close()
    return rec["status"] == "done"

def tick(max_stages=1):
    PRIVATE.mkdir(exist_ok=True)
    lock = (PRIVATE / "runner.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps({"event": "RUNNER_ALREADY_ACTIVE", "time": stamp()}))
        return 0
    try:
        plan, state = prepare()
        for _ in range(min(max_stages, 4)):
            if not one_stage(plan, state):
                break
        if (gitsha() == state["frozen_sha"] and
                all(state["stages"][x["id"]]["status"] == "done" for x in plan["stages"])):
            # Only the current frozen plan can end its cron lifecycle.
            marker = PRIVATE / "ALL_APPROVED_STAGES_DONE"
            if marker.exists() and marker.read_text().strip() != state["frozen_sha"]:
                raise RuntimeError("COMPLETION_MARKER_FOREIGN_SHA")
            marker.write_text(state["frozen_sha"] + "\n", encoding="utf-8")
            state["last_event"] = "ALL_APPROVED_STAGES_DONE"
            save(state)
        print(json.dumps({"event": state["last_event"], "frozen_sha": state["frozen_sha"],
                          "stages": state["stages"]}, ensure_ascii=False))
        return 0 if "FAILED" not in state["last_event"] and "TIMEOUT" not in state["last_event"] else 1
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["tick", "status", "initialize"])
    parser.add_argument("--max-stages", type=int, default=1)
    a = parser.parse_args()
    if a.action == "status":
        f = PRIVATE / "state.json"
        print(f.read_text(encoding="utf-8") if f.exists() else '{"status":"NOT_INITIALIZED"}')
    elif a.action == "initialize":
        prepare()
        print("INITIALIZED", PRIVATE / "state.json")
    else:
        sys.exit(tick(a.max_stages))

if __name__ == "__main__":
    main()
