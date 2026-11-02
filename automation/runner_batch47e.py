#!/usr/bin/env python3
"""Batch47E frozen-plan runner."""
from pathlib import Path
import runner
runner.PLAN = Path(__file__).resolve().parent / 'plans' / 'batch47e_dev_counterfactual_layers_2026-11-02.json'
if __name__ == '__main__':
    runner.main()
