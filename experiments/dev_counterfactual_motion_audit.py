"""Fail-closed audit of write-once Batch47E synthetic DEV evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import dev_affine_evidence_audit as common
import dev_counterfactual_motion_layers as experiment
import dev_temporal_bridge_affine as bridge


def verify(report_path: Path, selection_path: Path) -> dict:
    if common.selection_sha256(selection_path) != bridge.EXPECTED_SELECTION_SHA256:
        raise RuntimeError("B47E_AUDIT_SELECTION_PIN")
    manifest = selection_path.with_name("download_manifest.json")
    if common.file_sha256(manifest) != bridge.EXPECTED_DOWNLOAD_MANIFEST_SHA256:
        raise RuntimeError("B47E_AUDIT_MANIFEST_PIN")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("scope") != "DEV_ONLY_GENERATED_2D_TRUTH_NO_TEST_NO_PRIVATE_ORACLE":
        raise RuntimeError("B47E_AUDIT_SCOPE")
    if (report.get("selection_sha256") != bridge.EXPECTED_SELECTION_SHA256
            or report.get("manifest_sha256") != bridge.EXPECTED_DOWNLOAD_MANIFEST_SHA256
            or report.get("source_sha256") != common.file_sha256(Path(experiment.__file__))):
        raise RuntimeError("B47E_AUDIT_REPORT_SOURCE_OR_DATA_PIN")
    protocol = report.get("protocol", {})
    if (protocol.get("conditions") != list(experiment.CONDITIONS)
            or protocol.get("anchors_per_clip") != experiment.ANCHORS
            or protocol.get("width_px") != experiment.WIDTH
            or protocol.get("truth_global_dxdy_px") != [experiment.DX, experiment.DY]
            or protocol.get("wrong_error_gt_px") != experiment.MAX_GROUND_TRUTH_ERROR_PX
            or protocol.get("gate") != {
                "min_valid_folds": experiment.MIN_GATE_FOLDS,
                "max_worst_holdout_median_px": experiment.MAX_WORST_HOLDOUT_PX,
                "min_fraction_holdout_within_3px": experiment.MIN_FRACTION_WITHIN_3PX,
            }):
        raise RuntimeError("B47E_AUDIT_PROTOCOL_DRIFT")
    selected = common.dev_rows(json.loads(selection_path.read_text(encoding="utf-8")))
    scenes = report.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != len(selected) or len(scenes) != 6:
        raise RuntimeError("B47E_AUDIT_SCENE_COUNT")
    counts = {c: [] for c in experiment.CONDITIONS}
    for scene, original in zip(scenes, selected):
        if (scene.get("dev_video_id") != original["video_id"]
                or scene.get("action_class") != original["label"]):
            raise RuntimeError("B47E_AUDIT_DEV_ID_ORDER")
        rows = scene.get("rows")
        if not isinstance(rows, list) or len(rows) != experiment.ANCHORS * len(experiment.CONDITIONS):
            raise RuntimeError("B47E_AUDIT_ROWS_PER_SCENE")
        anchors = [rows[i]["anchor_frame"] for i in range(0, len(rows), len(experiment.CONDITIONS))]
        if anchors != sorted(set(anchors)):
            raise RuntimeError("B47E_AUDIT_ANCHOR_DISTINCTNESS")
        for i, row in enumerate(rows):
            if (row["condition"] != experiment.CONDITIONS[i % len(experiment.CONDITIONS)]
                    or row["anchor_frame"] != anchors[i // len(experiment.CONDITIONS)]):
                raise RuntimeError("B47E_AUDIT_CONDITION_ORDER")
            if not (isinstance(row["detected"], int)
                    and isinstance(row["matched"], int)
                    and 0 <= row["matched"] <= row["detected"]
                    and isinstance(row["valid_folds"], int)
                    and 0 <= row["valid_folds"] <= 4
                    and row["synthetic_background_grid_count"] >= 12):
                raise RuntimeError("B47E_AUDIT_GEOMETRY_INVARIANT")
            if row["gate_accept"] and (not row["fit_valid"]
                                        or not experiment.gate_crossfit({
                                            "status": row["crossfit_status"],
                                            "valid_folds": row["valid_folds"],
                                            "crossfit_worst_quadrant_median_px": row["worst_holdout_median_px"],
                                            "mean_heldout_fraction_within_3px": row["fraction_holdout_within_3px"],
                                        })):
                raise RuntimeError("B47E_AUDIT_GATE_INCONSISTENCY")
            if row["fit_valid"]:
                error = row["synthetic_bg_median_error_px"]
                if (not isinstance(error, (float, int)) or not math.isfinite(error)
                        or row["synthetic_bg_wrong"] !=
                        (error > experiment.MAX_GROUND_TRUTH_ERROR_PX)):
                    raise RuntimeError("B47E_AUDIT_ORACLE_SCORE")
            elif (row["gate_accept"] or row["synthetic_bg_median_error_px"] is not None
                  or row["synthetic_bg_wrong"] is not None):
                raise RuntimeError("B47E_AUDIT_INVALID_FIT_HAS_SCORE")
            counts[row["condition"]].append(row)
        if scene.get("condition_summaries") != {
                condition: experiment.summary([r for r in rows if r["condition"] == condition])
                for condition in experiment.CONDITIONS}:
            raise RuntimeError("B47E_AUDIT_SCENE_SUMMARY_MISMATCH")
    recomputed = {condition: experiment.summary(rows) for condition, rows in counts.items()}
    if report.get("overall") != recomputed:
        raise RuntimeError("B47E_AUDIT_OVERALL_SUMMARY_MISMATCH")
    if any(len(rows) != 6 * experiment.ANCHORS for rows in counts.values()):
        raise RuntimeError("B47E_AUDIT_CONDITION_UNBALANCED")
    return {
        "result": "B47E_DEV_SYNTHETIC_AUDIT_PASS",
        "scenes": len(scenes),
        "cases": sum(len(rows) for rows in counts.values()),
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "selection_sha256": bridge.EXPECTED_SELECTION_SHA256,
        "note": "Ground truth is synthetic 2D warping, not camera pose or target tracking TEST",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(verify(a.report, a.selection), sort_keys=True))


if __name__ == "__main__":
    main()
