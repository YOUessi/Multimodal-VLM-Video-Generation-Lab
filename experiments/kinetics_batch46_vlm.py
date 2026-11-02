"""Batch46 independent natural-video VLM baseline with the unchanged Batch45E gate."""
from __future__ import annotations
import sys
import kinetics_batch46_data as fresh
import kinetics_batch46_eval as experiment
import kinetics_batch45_vlm as previous

previous.DATA = fresh.FROZEN
previous.OUT = experiment.OUT
previous.src = fresh.base
previous.ev = experiment.previous


def main() -> None:
    if "--config" not in sys.argv:
        sys.argv += ["--config", "configs/kinetics_batch46.json"]
    previous.main()


if __name__ == "__main__":
    main()
