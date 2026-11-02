#!/usr/bin/env python3
"""Frozen Batch56 independent-video 22-stage experiment."""
from pathlib import Path
import runner
runner.PLAN = (Path(__file__).resolve().parent / "plans" /
               "batch56_kinetics_72_class_disjoint_2026-11-03.json")
if __name__ == "__main__":
    runner.main()
