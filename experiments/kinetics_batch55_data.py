"""Batch55 fresh action-class-disjoint Kinetics replication dataset.

Reuse the audited Batch45 decoder, but never reuse video IDs or action classes from
Batch42, Batch45E, Batch46E, Batch48B and Batch53B. The manifest/exclusion inputs contain metadata, not test
answers. Failure to read an older manifest fails closed rather than silently
weakening the independence contract.
"""
from __future__ import annotations
from pathlib import Path
import importlib.util
import json


def _isolated_batch45_data():
    """Load a private data-module instance: Batch55 must never patch Batch45.

    Batch45's functions reference module globals (DATA, FROZEN, SALT and the
    selector). Import-time monkeypatching the *shared* module silently changes
    Batch45 tests and any simultaneous in-process Batch45 research. Loading
    the source under a different module namespace preserves the original API
    without mutating another experiment's namespace.
    """
    path = Path(__file__).with_name("kinetics_batch45_data.py")
    spec = importlib.util.spec_from_file_location("_batch55_private_kinetics_batch45_data", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("BATCH55_PRIVATE_BASE_MODULE_LOAD_FAILED")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _isolated_batch45_data()
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/tapvid_kinetics_batch55"
FROZEN = DATA / "batch55_frozen"
OLD_MANIFESTS = (
    ROOT / "data/tapvid_kinetics_batch42/batch42_frozen/manifest.json",
    ROOT / "data/tapvid_kinetics_batch45e/batch45e_frozen/manifest.json",
    ROOT / "data/tapvid_kinetics_batch46/batch46_frozen/manifest.json",
    ROOT / "data/tapvid_kinetics_batch48/batch48_frozen/manifest.json",
    ROOT / "data/tapvid_kinetics_batch53/batch53_frozen/manifest.json",
)
ORIGINAL_SELECT = base.deterministic_class_selection


def historical_identifiers() -> tuple[set[str], set[str]]:
    """Fail closed unless each prior audited manifest supplies unique IDs and classes.

    Only metadata of previously opened experiments is used for exclusion;
    historical private tracks and TEST answers are neither opened nor scored.
    """
    labels: set[str] = set()
    ids: set[str] = set()
    for path in OLD_MANIFESTS:
        if not path.is_file():
            raise RuntimeError("BATCH55_REQUIRED_HISTORY_MISSING:" + str(path))
        document = json.loads(path.read_text(encoding="utf-8"))
        scenes = document.get("scenes")
        if not isinstance(scenes, list) or len(scenes) != 12:
            raise RuntimeError("BATCH55_HISTORY_INVALID:" + str(path))
        class_names, video_ids = [], []
        for scene in scenes:
            label, video_id = scene.get("action_class"), scene.get("video_id")
            if not isinstance(label, str) or not label.strip():
                raise RuntimeError("BATCH55_HISTORY_CLASS_MISSING:" + str(path))
            if not isinstance(video_id, str) or not video_id.strip():
                raise RuntimeError("BATCH55_HISTORY_VIDEO_ID_MISSING:" + str(path))
            class_names.append(label)
            video_ids.append(video_id)
        if len(set(class_names)) != len(class_names) or len(set(video_ids)) != len(video_ids):
            raise RuntimeError("BATCH55_HISTORY_DUPLICATE_ID_OR_CLASS:" + str(path))
        labels.update(class_names)
        ids.update(video_ids)
    return labels, ids


def historical_classes() -> set[str]:
    return historical_identifiers()[0]


def audit_preflight_disjointness(selection: dict) -> None:
    """Check cached or newly generated selection BEFORE any archive download."""
    rows = selection.get("videos")
    if not isinstance(rows, list) or len(rows) != 12:
        raise RuntimeError("BATCH55_SELECTION_COUNT_INVALID")
    classes, ids = historical_identifiers()
    video_ids = [r.get("video_id") for r in rows]
    class_names = [r.get("label") for r in rows]
    if (any(not isinstance(v, str) or not v for v in video_ids + class_names)
            or len(set(video_ids)) != 12 or len(set(class_names)) != 12):
        raise RuntimeError("BATCH55_SELECTION_DUPLICATE_OR_INVALID")
    if video_ids and set(video_ids) & ids:
        raise RuntimeError("BATCH55_SELECTION_HISTORICAL_VIDEO_LEAK")
    if set(class_names) & classes:
        raise RuntimeError("BATCH55_SELECTION_HISTORICAL_CLASS_LEAK")
    if [r.get("split") for r in rows] != ["dev"] * 6 + ["test"] * 6:
        raise RuntimeError("BATCH55_SELECTION_SPLIT_INVALID")


def new_class_selection(candidates, all_labels, previous_ids, previous_labels,
                        path_lines, source_hashes):
    historical_labels, historical_ids = historical_identifiers()
    excluded = historical_labels | set(previous_labels)
    previous_ids = set(previous_ids) | historical_ids
    selected = ORIGINAL_SELECT(
        candidates, all_labels, previous_ids, excluded, path_lines,
        source_hashes,
    )
    chosen = {video["label"] for video in selected["videos"]}
    if chosen & excluded:
        raise RuntimeError("BATCH55_HISTORICAL_ACTION_CLASS_LEAK")
    selected["previous_action_classes_excluded"] = sorted(excluded)
    audit_preflight_disjointness(selected)
    selected["protocol"] = (
        "TAP-Vid-Kinetics natural videos; SHA256-preselected unused video IDs "
        "AND action classes disjoint from Batch42, Batch45E, Batch46E, Batch48B and Batch53B; "
        "6 DEV / 6 TEST with four point queries each; no tuning on TEST"
    )
    return selected


def audit_historical_disjointness() -> None:
    manifest_path = FROZEN / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("BATCH55_MANIFEST_MISSING")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenes = document.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != 12:
        raise RuntimeError("BATCH55_MANIFEST_SCENE_COUNT")
    old_classes, old_ids = historical_identifiers()
    labels = [row.get("action_class") for row in scenes]
    ids = [row.get("video_id") for row in scenes]
    if (any(not isinstance(s, str) or not s for s in labels + ids)
            or len(set(labels)) != 12 or len(set(ids)) != 12):
        raise RuntimeError("BATCH55_UNIQUE_VIDEO_OR_CLASS_COUNT")
    if set(labels) & old_classes:
        raise RuntimeError("BATCH55_CLASS_REUSE")
    if set(ids) & old_ids:
        raise RuntimeError("BATCH55_VIDEO_REUSE")
    audit_preflight_disjointness(base.load_selection())


ORIGINAL_BUILD_SELECTION = base.build_selection


def guarded_build_selection(path_lines: list[str]) -> dict:
    selected = ORIGINAL_BUILD_SELECTION(path_lines)
    audit_preflight_disjointness(selected)
    return selected


base.DATA = DATA
base.FROZEN = FROZEN
base.PATH_LIST = ROOT / "data/tapvid_kinetics_batch45e/preflight/k700_2020_val_path.txt"
base.DOWNLOAD_MANIFEST = DATA / "preflight/download_manifest.json"
base.SALT = "MVVG-BATCH55-NEW-CLASS-DISJOINT-REPLICATION-20261103"
base.deterministic_class_selection = new_class_selection
base.build_selection = guarded_build_selection


def main() -> None:
    import sys
    import shutil
    action = sys.argv[1] if len(sys.argv) > 1 else None
    if action == "download":
        free = shutil.disk_usage(DATA.parent).free
        if free < 10 * 1024 ** 3:
            raise RuntimeError(f"BATCH55_DISK_PREFLIGHT_GUARD:{free}")
    base.main()
    if action == "download":
        audit_preflight_disjointness(base.load_selection())
    elif action in ("prepare", "audit-public", "audit"):
        audit_historical_disjointness()


if __name__ == "__main__":
    main()
