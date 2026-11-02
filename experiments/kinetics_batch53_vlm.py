"""Frozen SmolVLM2 visibility reference on fresh Batch53 DEV; TEST only if gate passes."""
from __future__ import annotations
import sys
from pathlib import Path
import importlib.util
import kinetics_batch53_data as fresh
import kinetics_batch53_eval as evaluation


def _private_vlm():
    path = Path(__file__).with_name("kinetics_batch45_vlm.py")
    spec = importlib.util.spec_from_file_location("_batch53_private_kinetics_batch45_vlm", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("BATCH53_PRIVATE_VLM_MODULE_LOAD_FAILED")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = _private_vlm()

reference.DATA = fresh.FROZEN
reference.OUT = evaluation.OUT
reference.src = fresh.base
reference.ev = evaluation.baseline


def main() -> None:
    if "--config" not in sys.argv:
        sys.argv += ["--config", "configs/kinetics_batch53.json"]
    reference.main()


if __name__ == "__main__":
    main()
