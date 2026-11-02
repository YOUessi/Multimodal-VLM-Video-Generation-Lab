#!/usr/bin/env python3
"""Batch48 frozen plan entrypoint; generic runner's legacy plan remains untouched."""
from pathlib import Path
import runner

runner.PLAN = Path(__file__).resolve().parent / "plans" / "batch48_fresh_independent_safety_dev_select_2026-11-02.json"

if __name__ == "__main__":
    runner.main()
