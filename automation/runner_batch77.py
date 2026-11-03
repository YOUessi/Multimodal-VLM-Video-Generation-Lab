#!/usr/bin/env python3
"""Batch77: independently frozen continuation of failed Batch59 natural-video run.

A dedicated runner is used so the original Batch59 SHA, state and failure logs
remain immutable. Only an exit 75 emitted by the verified-progress download
wrapper may be retried; all other nonzero stage exits are terminal.
"""
from pathlib import Path
import runner

runner.PLAN = (Path(__file__).resolve().parent / "plans" /
               "batch77_resume_batch59_2026-11-04.json")
_original_one_stage = runner.one_stage


def one_stage_with_progress_deferral(plan: dict, state: dict) -> bool:
    pending = next((step for step in plan["stages"]
                    if state["stages"][step["id"]]["status"] != "done"), None)
    result = _original_one_stage(plan, state)
    if result or not pending:
        return result
    if pending["id"] != "R00_SELECT_AND_DOWNLOAD_42_FRESH_CLASS_SHARDS":
        return False
    record = state["stages"][pending["id"]]
    if (pending.get("defer_exit_code") == 75
            and record.get("status") == "failed"
            and record.get("returncode") == 75
            and record.get("attempts", 0) < int(pending["max_attempts"])):
        record["status"] = "pending"
        record["progress_deferrals"] = record.get("progress_deferrals", 0) + 1
        state["last_event"] = "DEFERRED_VERIFIED_PROGRESS:" + pending["id"]
        runner.save(state)
    return False


runner.one_stage = one_stage_with_progress_deferral

if __name__ == "__main__":
    runner.main()
