"""Batch46 fresh action-class-disjoint Kinetics replication dataset.

Reuse the audited Batch45 decoder, but never reuse video IDs or action classes from
Batch42 and Batch45E. The manifest/exclusion inputs contain metadata, not test
answers. Failure to read an older manifest fails closed rather than silently
weakening the independence contract.
"""
from __future__ import annotations
from pathlib import Path
import json
import kinetics_batch45_data as base

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/tapvid_kinetics_batch46"
FROZEN = DATA / "batch46_frozen"
OLD_MANIFESTS = (
    ROOT / "data/tapvid_kinetics_batch42/batch42_frozen/manifest.json",
    ROOT / "data/tapvid_kinetics_batch45e/batch45e_frozen/manifest.json",
)
ORIGINAL_SELECT = base.deterministic_class_selection


def historical_classes() -> set[str]:
    labels: set[str] = set()
    for path in OLD_MANIFESTS:
        if not path.is_file():
            raise RuntimeError("BATCH46_REQUIRED_HISTORY_MISSING:" + str(path))
        document = json.loads(path.read_text(encoding="utf-8"))
        scenes = document.get("scenes")
        if not isinstance(scenes, list) or not scenes:
            raise RuntimeError("BATCH46_HISTORY_INVALID:" + str(path))
        for scene in scenes:
            label = scene.get("action_class")
            if not isinstance(label, str) or not label:
                raise RuntimeError("BATCH46_HISTORY_CLASS_MISSING:" + str(path))
            labels.add(label)
    return labels


def new_class_selection(candidates, all_labels, previous_ids, previous_labels,
                        path_lines, source_hashes):
    excluded = historical_classes() | set(previous_labels)
    selected = ORIGINAL_SELECT(
        candidates, all_labels, previous_ids, excluded, path_lines,
        source_hashes,
    )
    chosen = {video["label"] for video in selected["videos"]}
    if chosen & excluded:
        raise RuntimeError("BATCH46_HISTORICAL_ACTION_CLASS_LEAK")
    selected["previous_action_classes_excluded"] = sorted(excluded)
    selected["protocol"] = (
        "TAP-Vid-Kinetics natural videos; SHA256-preselected unused video IDs "
        "AND action classes disjoint from BOTH Batch42 and Batch45E; "
        "6 DEV / 6 TEST with four point queries each; no tuning on TEST"
    )
    return selected


def audit_historical_disjointness() -> None:
    frozen_manifest = FROZEN / "manifest.json"
    if not frozen_manifest.is_file():
        raise RuntimeError("BATCH46_MANIFEST_MISSING")
    selected = json.loads(frozen_manifest.read_text(encoding="utf-8"))
    labels = {scene["action_class"] for scene in selected["scenes"]}
    ids = {scene["video_id"] for scene in selected["scenes"]}
    if len(labels) != 12 or len(ids) != 12:
        raise RuntimeError("BATCH46_UNIQUE_VIDEO_OR_CLASS_COUNT")
    if labels & historical_classes():
        raise RuntimeError("BATCH46_CLASS_REUSE")
    known_ids = set()
    for path in OLD_MANIFESTS:
        old = json.loads(path.read_text(encoding="utf-8"))
        known_ids.update(s["video_id"] for s in old["scenes"])
    if ids & known_ids:
        raise RuntimeError("BATCH46_VIDEO_REUSE")


base.DATA = DATA
base.FROZEN = FROZEN
base.PATH_LIST = ROOT / "data/tapvid_kinetics_batch45e/preflight/k700_2020_val_path.txt"
base.DOWNLOAD_MANIFEST = DATA / "preflight/download_manifest.json"
base.SALT = "MVVG-BATCH46-INDEPENDENT-KINETICS-CLASS-HOLDOUT-20261102"
base.deterministic_class_selection = new_class_selection


def main() -> None:
    import sys
    action = sys.argv[1] if len(sys.argv) > 1 else None
    base.main()
    if action in ("prepare", "audit-public", "audit"):
        audit_historical_disjointness()


if __name__ == "__main__":
    main()
