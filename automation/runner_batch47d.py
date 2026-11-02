#!/usr/bin/env python3
"""Batch47D frozen-plan entrypoint; leave legacy production plan.json unchanged."""
from pathlib import Path

import runner

runner.PLAN = Path(__file__).resolve().parent / "plans" / "batch47d_temporal_bridge_dev_only_2026-11-02.json"

if __name__ == "__main__":
    runner.main()
