#!/usr/bin/env python3
"""Batch53B: immutable 22-stage resume of the same videos under schema repair."""
from pathlib import Path
import runner
runner.PLAN = (Path(__file__).resolve().parent / "plans" /
               "batch53b_geometry_integrity_resume_2026-11-03.json")
if __name__ == "__main__":
    runner.main()
