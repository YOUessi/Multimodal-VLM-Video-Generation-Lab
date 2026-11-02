#!/usr/bin/env python3
"""Frozen Batch59 36-DEV/6-TEST independent-video 24-stage experiment."""
from pathlib import Path
import runner
runner.PLAN = (Path(__file__).resolve().parent / "plans" /
               "batch59_kinetics_114_class_disjoint_2026-11-03.json")
if __name__ == "__main__":
    runner.main()
