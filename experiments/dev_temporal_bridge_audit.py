"""Audit write-once Batch47D DEV-only bridging evidence without TEST or oracle access."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import dev_affine_evidence_audit as common
import dev_temporal_bridge_affine as bridge


def verify(report_path: Path, selection_path: Path) -> dict:
    pin = common.selection_sha256(selection_path)
    if pin != bridge.EXPECTED_SELECTION_SHA256:
        raise RuntimeError("BRIDGE_AUDIT_SELECTION_PIN_CHANGED")
    d = json.loads(report_path.read_text(encoding="utf-8"))
    if d.get("scope") != "DEV_ONLY_NO_TEST_MEDIA_NO_PRIVATE_ORACLE":
        raise RuntimeError("BRIDGE_AUDIT_SCOPE_NOT_DEV_ONLY")
    if d.get("selection_sha256") != pin:
        raise RuntimeError("BRIDGE_AUDIT_SELECTION_MISMATCH")
    manifest_path = selection_path.with_name("download_manifest.json")
    if (not manifest_path.exists()
            or common.file_sha256(manifest_path) != bridge.EXPECTED_DOWNLOAD_MANIFEST_SHA256
            or d.get("download_manifest_sha256") != bridge.EXPECTED_DOWNLOAD_MANIFEST_SHA256):
        raise RuntimeError("BRIDGE_AUDIT_DOWNLOAD_MANIFEST_MISMATCH")
    if d.get("source_sha256") != common.file_sha256(Path(bridge.__file__)):
        raise RuntimeError("BRIDGE_AUDIT_SOURCE_MISMATCH")
    settings = d.get("settings", {})
    if (settings.get("extent_frames") != 30
            or settings.get("anchor_count") != 8
            or settings.get("methods") != bridge.METHODS):
        raise RuntimeError("BRIDGE_AUDIT_PROTOCOL_DRIFT")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    expected = common.dev_rows(selection)
    scenes = d.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != 6:
        raise RuntimeError("BRIDGE_AUDIT_EXPECTED_SIX_DEV_SCENES")
    if [s.get("dev_video_id") for s in scenes] != [r["video_id"] for r in expected]:
        raise RuntimeError("BRIDGE_AUDIT_DEV_VIDEO_ORDER_OR_ID_MISMATCH")
    total_pairs = 0
    for scene in scenes:
        pairs = scene.get("pairs", [])
        if len(pairs) != 8:
            raise RuntimeError("BRIDGE_AUDIT_ANCHOR_COUNT")
        if scene.get("anchors") != [p.get("anchor_frame") for p in pairs]:
            raise RuntimeError("BRIDGE_AUDIT_ANCHOR_ORDER")
        for row in pairs:
            methods = row.get("methods", {})
            if set(methods) != set(bridge.METHODS):
                raise RuntimeError("BRIDGE_AUDIT_METHOD_SET")
            for name, hop in bridge.METHODS.items():
                record = methods[name]
                count = record["detected"]
                survivors = record["survivors"]
                h = record["hop_survivors"]
                if not (isinstance(count, int) and isinstance(survivors, int)
                        and 0 <= survivors <= count
                        and len(h) == 30 // hop
                        and all(0 <= x <= count for x in h)
                        and all(a >= b for a, b in zip([count, *h], h))
                        and h[-1] == survivors):
                    raise RuntimeError("BRIDGE_AUDIT_SURVIVOR_INVARIANT")
                if record["spatial"]["match_count"] != survivors:
                    raise RuntimeError("BRIDGE_AUDIT_MATCH_COUNT_MISMATCH")
            total_pairs += 1
    if total_pairs != 48:
        raise RuntimeError("BRIDGE_AUDIT_EXPECTED_48_PAIRS")
    # No comparison is admissible without reporting the paired support count.
    summary = d.get("overall", {})
    if set(summary) != set(bridge.METHODS):
        raise RuntimeError("BRIDGE_AUDIT_SUMMARY_METHODS")
    for name, metrics in summary.items():
        n = metrics.get("paired_with_direct_valid")
        valid = metrics.get("valid_spatial_crossfit")
        if (metrics.get("pair_count") != 48 or not isinstance(n, int)
                or not isinstance(valid, int) or not 0 <= n <= valid <= 48):
            raise RuntimeError("BRIDGE_AUDIT_PAIRED_SUPPORT_INVALID")
    return {
        "result": "BRIDGE_DEV_EVIDENCE_AUDIT_PASS",
        "selection_sha256": pin,
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "source_sha256": d["source_sha256"],
        "scenes": len(scenes), "pairs": total_pairs,
        "note": "2D DEV diagnostic only; not a new blind TEST or physical camera-motion measurement",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.report, args.selection), sort_keys=True))


if __name__ == "__main__":
    main()
