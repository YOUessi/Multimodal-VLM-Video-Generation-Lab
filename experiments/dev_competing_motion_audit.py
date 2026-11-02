"""DEV-only competing-motion counterexample audit; not a camera pose estimator.

This is a falsification-oriented diagnostic, not a background/foreground oracle.
A coherent second 2D motion layer can refute a *unique* global-affine model.
The absence of a detected competitor DOES NOT certify the dominant layer as
background. In particular, no observable feature can distinguish a foreground
occupying every pixel from a camera-induced translation on two frames alone.
Do not use this module on Batch46E or Batch48 sealed TEST to tune any gate.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import dev_affine_evidence_audit as pins
import dev_counterfactual_motion_layers as control
import dev_temporal_bridge_affine as bridge

# Explicit exploratory hypothesis, fixed before synthetic DEV execution.
PRIMARY_INLIER_TOL_PX = 3.0
SECOND_INLIER_TOL_PX = 2.0
MIN_SECOND_INLIERS = 12
MIN_SECOND_FRACTION = 0.10
MIN_SECOND_QUADRANTS = 2
MIN_MODEL_DISAGREEMENT_PX = 4.0
MAX_RANSAC_ITERS = 2000


def _finite_points(src: np.ndarray, dst: np.ndarray, height: int, width: int):
    a = np.asarray(src, dtype=np.float32)
    b = np.asarray(dst, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape or a.shape[1:] != (2,):
        raise ValueError("COMPETING_MOTION_INVALID_CORRESPONDENCES")
    if height < 32 or width < 32 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("COMPETING_MOTION_NONFINITE_OR_SMALL_FRAME")
    if len(a) and ((a[:, 0] < 0).any() or (a[:, 0] >= width).any()
                   or (a[:, 1] < 0).any() or (a[:, 1] >= height).any()):
        raise ValueError("COMPETING_MOTION_SOURCE_OUT_OF_BOUNDS")
    return a, b


def _reprojection(model: np.ndarray, points: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points @ model[:, :2].T + model[:, 2] - target, axis=1)


def _quadrants(points: np.ndarray, height: int, width: int) -> int:
    q = (points[:, 0] >= width / 2).astype(np.uint8)
    q += 2 * (points[:, 1] >= height / 2).astype(np.uint8)
    return int(np.unique(q).size)


def _agreement_distance(primary: np.ndarray, competitor: np.ndarray,
                        height: int, width: int) -> float:
    gx, gy = np.meshgrid(np.linspace(0.15 * width, 0.85 * width, 3),
                         np.linspace(0.15 * height, 0.85 * height, 3))
    pts = np.stack((gx.ravel(), gy.ravel()), axis=1)
    d = pts @ (primary[:, :2] - competitor[:, :2]).T
    d += primary[:, 2] - competitor[:, 2]
    return float(np.median(np.linalg.norm(d, axis=1)))


def competing_motion_evidence(
    source: np.ndarray, dest: np.ndarray, height: int, width: int,
    primary: np.ndarray | None = None,
) -> dict:
    """Reject uniqueness if a spatially distributed, incompatible second fit exists.

    No truth/labels enter the function. The second hypothesis is fitted only
    on residual points inconsistent with the original RANSAC affine. Missing
    motion evidence is always reported explicitly rather than scored correct.
    """
    src, dst = _finite_points(source, dest, height, width)
    result = dict(matched=len(src), primary_valid=False, alternate_valid=False,
                  alternate_inliers=0, alternate_fraction=0.0,
                  alternate_quadrants=0, median_model_disagreement_px=None,
                  detected_competing_layer=False, status="insufficient_matches")
    if len(src) < MIN_SECOND_INLIERS * 2:
        return result
    if primary is None:
        cv2.setRNGSeed(1129)
        primary, _ = cv2.estimateAffine2D(
            src, dst, method=cv2.RANSAC,
            ransacReprojThreshold=PRIMARY_INLIER_TOL_PX,
            maxIters=MAX_RANSAC_ITERS, confidence=0.99, refineIters=10,
        )
    if (primary is None or np.shape(primary) != (2, 3)
            or not np.isfinite(primary).all()):
        return {**result, "status": "no_valid_primary_fit"}
    result["primary_valid"] = True
    residual = _reprojection(primary, src, dst)
    candidates = residual > PRIMARY_INLIER_TOL_PX
    if int(candidates.sum()) < MIN_SECOND_INLIERS:
        return {**result, "status": "insufficient_nonprimary_support"}
    cv2.setRNGSeed(1129)
    secondary, mask = cv2.estimateAffine2D(
        src[candidates], dst[candidates], method=cv2.RANSAC,
        ransacReprojThreshold=SECOND_INLIER_TOL_PX,
        maxIters=MAX_RANSAC_ITERS, confidence=0.99, refineIters=10,
    )
    if secondary is None or mask is None or not np.isfinite(secondary).all():
        return {**result, "status": "no_coherent_secondary_fit"}
    # Apply residual and support checks on all non-primary correspondences.
    alt_inlier = _reprojection(secondary, src[candidates], dst[candidates]) <= SECOND_INLIER_TOL_PX
    n_alt = int(alt_inlier.sum())
    frac = float(n_alt / len(src))
    quadrants = _quadrants(src[candidates][alt_inlier], height, width) if n_alt else 0
    distance = _agreement_distance(primary, secondary, height, width)
    valid = (n_alt >= MIN_SECOND_INLIERS
             and frac >= MIN_SECOND_FRACTION
             and quadrants >= MIN_SECOND_QUADRANTS
             and distance >= MIN_MODEL_DISAGREEMENT_PX)
    return {**result, "alternate_valid": bool(valid),
            "alternate_inliers": n_alt, "alternate_fraction": frac,
            "alternate_quadrants": quadrants,
            "median_model_disagreement_px": distance,
            "detected_competing_layer": bool(valid),
            "status": "competing_models" if valid else "secondary_below_evidence_threshold"}


def _rate(n: int, d: int):
    return float(n / d) if d else None


def summarize(rows: list[dict]) -> dict:
    baseline = [r for r in rows if r["old_gate_accept"]]
    safer = [r for r in rows if r["ambiguity_gate_accept"]]
    return {
        "pairs": len(rows),
        "fit_count": sum(r["fit_valid"] for r in rows),
        "competitor_detected": sum(r["detected_competing_layer"] for r in rows),
        "old_gate_accepted": len(baseline),
        "new_gate_accepted": len(safer),
        "old_wrong_accepted": sum(r["synthetic_bg_wrong"] is True for r in baseline),
        "new_wrong_accepted": sum(r["synthetic_bg_wrong"] is True for r in safer),
        "old_gate_coverage": _rate(len(baseline), len(rows)),
        "new_gate_coverage": _rate(len(safer), len(rows)),
        "old_wrong_fraction_among_accepted": _rate(
            sum(r["synthetic_bg_wrong"] is True for r in baseline), len(baseline)),
        "new_wrong_fraction_among_accepted": _rate(
            sum(r["synthetic_bg_wrong"] is True for r in safer), len(safer)),
        "median_extra_cpu_ms": (float(np.median([r["extra_cpu_ms"] for r in rows]))
                                if rows else None),
    }


def run(selection: Path, shards: Path, output: Path) -> dict:
    if pins.selection_sha256(selection) != bridge.EXPECTED_SELECTION_SHA256:
        raise RuntimeError("COMPETING_MOTION_SELECTION_PIN_MISMATCH")
    manifest_file = selection.with_name("download_manifest.json")
    if pins.file_sha256(manifest_file) != bridge.EXPECTED_DOWNLOAD_MANIFEST_SHA256:
        raise RuntimeError("COMPETING_MOTION_MANIFEST_PIN_MISMATCH")
    selected = pins.dev_rows(json.loads(selection.read_text(encoding="utf-8")))
    if len(selected) != 6:
        raise RuntimeError("COMPETING_MOTION_EXPECTED_SIX_DEV_CLIPS")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    scenes = []
    for scene_num, row in enumerate(selected):
        entry = manifest["shards"].get(f"{row['shard_index']:03d}")
        if (not isinstance(entry, dict) or entry.get("archive_member") != row["archive_member"]
                or not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64):
            raise RuntimeError("COMPETING_MOTION_BAD_PINNED_MEMBER")
        anchored = control._clip_frames(row, shards, entry["sha256"])
        rows = []
        for anchor, frame in anchored:
            seed = __import__("zlib").crc32((row["video_id"] + ":" + str(anchor)).encode())
            for condition in control.CONDITIONS:
                src, dst, truth, grid = control.synthesize(frame, condition, seed)
                base, model = control.estimate_pair(src, dst)
                t0 = time.perf_counter()
                tracks = bridge.chained_cycle_matches([src, dst])
                evidence = competing_motion_evidence(
                    tracks["src"], tracks["dst"], src.shape[0], src.shape[1], model)
                elapsed = float((time.perf_counter() - t0) * 1000)
                row_out = {
                    "anchor_frame": anchor, "condition": condition,
                    "fit_valid": base["fit_valid"],
                    "old_gate_accept": base["gate_accept"],
                    "ambiguity_gate_accept": bool(
                        base["gate_accept"] and not evidence["detected_competing_layer"]),
                    "extra_cpu_ms": elapsed,
                    **evidence,
                    **control.score_known_affine(model, truth, grid),
                }
                rows.append(row_out)
        scenes.append({
            "dev_video_id": row["video_id"], "action_class": row["label"],
            "rows": rows,
            "by_condition": {
                c: summarize([r for r in rows if r["condition"] == c])
                for c in control.CONDITIONS
            },
        })
        print("DEV_COMPETING_MOTION_SCENE", scene_num + 1, row["label"],
              len(rows), flush=True)
    all_rows = [r for s in scenes for r in s["rows"]]
    report = {
        "scope": "DEV_ONLY_SYNTHETIC_2D_NO_TEST_NO_PRIVATE_ORACLE",
        "selection_sha256": bridge.EXPECTED_SELECTION_SHA256,
        "manifest_sha256": bridge.EXPECTED_DOWNLOAD_MANIFEST_SHA256,
        "source_sha256": pins.file_sha256(Path(__file__)),
        "conditions": list(control.CONDITIONS),
        "hypothesis": {
            "second_reprojection_tolerance_px": SECOND_INLIER_TOL_PX,
            "minimum_second_inliers": MIN_SECOND_INLIERS,
            "minimum_second_fraction": MIN_SECOND_FRACTION,
            "minimum_second_quadrants": MIN_SECOND_QUADRANTS,
            "minimum_model_disagreement_px": MIN_MODEL_DISAGREEMENT_PX,
        },
        "overall": {
            c: summarize([r for r in all_rows if r["condition"] == c])
            for c in control.CONDITIONS
        },
        "scenes": scenes,
        "limitations": [
            "Dominant coherent motion can be foreground even when no secondary layer is detectable.",
            "A second motion layer refutes uniqueness, not which layer represents background.",
            "Uses previously studied DEV frames and artificial shifts, not a fresh TEST.",
            "Does not measure TAP-Vid target tracking, actual occlusion, or 3D camera motion.",
            "Extra CPU time includes a second feature tracking pass for diagnostic parity.",
            "No TEST-driven threshold selection or frozen production tracker update.",
        ],
    }
    if pins.selection_sha256(selection) != bridge.EXPECTED_SELECTION_SHA256:
        raise RuntimeError("COMPETING_MOTION_SELECTION_DRIFT")
    payload = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != payload:
            raise RuntimeError("COMPETING_MOTION_WRITE_ONCE_CONFLICT")
    else:
        with output.open("xb") as f:
            f.write(payload)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--shards", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run(args.selection, args.shards, args.output)
    for condition, summary in report["overall"].items():
        print("DEV_COMPETING_MOTION_RESULT", condition,
              json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
