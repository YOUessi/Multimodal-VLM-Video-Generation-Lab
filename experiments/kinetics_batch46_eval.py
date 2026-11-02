"""Batch46: unchanged Batch45E CPU methods on an unseen action-class holdout.

The selected method is inherited from the SHA-pinned Batch45E DEV decision.
Neither Batch46 DEV nor TEST metrics can alter that choice.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import kinetics_batch46_data as fresh
import kinetics_batch45_eval as previous

ROOT = Path(__file__).resolve().parents[1]
DATA = fresh.FROZEN
OUT = ROOT / "outputs/kinetics_batch46"
PRIOR_SELECTION = ROOT / "outputs/kinetics_batch45e/dev_selection.json"
PRIOR_SOURCE = "e59aa07c6ab69a40ab2bb35e64cad4645410d10c"
PRIOR_SELECTION_SHA256 = "c7a7804918af64ff5bd8cc817db22ab16745f7dcd910e6e26c1601bf85133dc2"
FROZEN_METHOD = "affine_stabilized_klt"

previous.DATA = DATA
previous.OUT = OUT
previous.src = fresh.base


def verify_prior_selection() -> dict:
    if not PRIOR_SELECTION.is_file():
        raise RuntimeError("BATCH46_PREVIOUS_DEV_SELECTION_MISSING")
    if previous.sha_bytes(PRIOR_SELECTION.read_bytes()) != PRIOR_SELECTION_SHA256:
        raise RuntimeError("BATCH46_PREVIOUS_DEV_SELECTION_SHA_DRIFT")
    doc = json.loads(PRIOR_SELECTION.read_text(encoding="utf-8"))
    if (doc.get("source_commit") != PRIOR_SOURCE
            or doc.get("selected_method") != FROZEN_METHOD
            or doc.get("selection_source") != "DEV only"):
        raise RuntimeError("BATCH46_PREVIOUS_DEV_SELECTION_CONTRACT")
    return doc


def select_pre_registered(cfg: dict) -> dict:
    verify_prior_selection()
    path = OUT / "dev_results.json"
    if not path.is_file():
        raise RuntimeError("BATCH46_DEV_RESULT_MISSING")
    dev = previous.core.doc(path)
    cfg_hash = previous.sha_bytes(previous.json_bytes(cfg))
    if (dev.get("split") != "dev"
            or dev.get("source_commit") != previous.gitsha()
            or dev.get("config_sha256") != cfg_hash):
        raise RuntimeError("BATCH46_DEV_SOURCE_CONTRACT")
    selection = {
        "source_commit": previous.gitsha(),
        "split": "dev",
        "config_sha256": cfg_hash,
        "dev_results_sha256": previous.sha_bytes(path.read_bytes()),
        "selected_method": FROZEN_METHOD,
        "eligible_candidates": [FROZEN_METHOD],
        "selection_source": "PREVIOUS_BATCH45E_DEV_ONLY_FROZEN",
        "previous_selection_source_commit": PRIOR_SOURCE,
        "previous_selection_sha256": PRIOR_SELECTION_SHA256,
        "reason": "pre-registered transfer without reading Batch46 DEV or TEST scores for method choice",
        "rules": cfg["dev_selection"],
    }
    output = OUT / "dev_selection.json"
    previous.core.once(output, selection)
    previous.core.pin(output)
    print("BATCH46_TRANSFER_METHOD_FROZEN", FROZEN_METHOD, flush=True)
    return selection


def main() -> None:
    previous.core.select_dev = select_pre_registered
    if "--config" not in sys.argv:
        sys.argv += ["--config", "configs/kinetics_batch46.json"]
    previous.main()


if __name__ == "__main__":
    main()
