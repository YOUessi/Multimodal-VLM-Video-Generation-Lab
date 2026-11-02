"""Batch48 independent Kinetics CPU tracker evaluation; select on new DEV only.

All frozen CPU methods are unchanged. DEV safety/coverage constraints select a
single candidate *before* sealed TEST predictions and one-time scoring. Unlike
Batch46, no previous TEST-selected method is transferred or promoted.
"""
from __future__ import annotations

from pathlib import Path
import sys
import importlib.util
import kinetics_batch48_data as fresh


def _private_baseline():
    path = Path(__file__).with_name("kinetics_batch45_eval.py")
    spec = importlib.util.spec_from_file_location("_batch48_private_kinetics_batch45_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("BATCH48_PRIVATE_EVAL_MODULE_LOAD_FAILED")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = _private_baseline()

ROOT = Path(__file__).resolve().parents[1]
DATA = fresh.FROZEN
OUT = ROOT / "outputs/kinetics_batch48"

baseline.DATA = DATA
baseline.OUT = OUT
baseline.src = fresh.base


def main() -> None:
    if "--config" not in sys.argv:
        sys.argv += ["--config", "configs/kinetics_batch48.json"]
    baseline.main()


if __name__ == "__main__":
    main()
