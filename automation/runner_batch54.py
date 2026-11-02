#!/usr/bin/env python3
"""Batch54 frozen DEV-only five-stage clustered safety audit."""
from pathlib import Path
import runner
runner.PLAN = (Path(__file__).resolve().parent / "plans" /
               "batch54_dev_scene_cluster_safety_2026-11-03.json")
if __name__ == "__main__":
    runner.main()
