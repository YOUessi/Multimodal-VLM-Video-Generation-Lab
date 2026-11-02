"""DEV-only out-of-region evidence for dominant 2D affine compensation.

Four spatial leave-one-quadrant-out fits are evaluated on KLT matches held out
from RANSAC fitting. This does NOT infer camera pose, 3D world motion, target
visibility, or tracking quality. It is diagnostic only and never opens TEST.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

import dev_affine_evidence_audit as audit

MIN_TRAIN = 24
MIN_HOLDOUT = 12
PIXEL_TOLERANCE = 3.0


def crossfit_affine(src: np.ndarray, dst: np.ndarray, width: int, height: int) -> dict:
    """Spatial holdout: never score a correspondence used for its fit."""
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    if src.ndim != 2 or src.shape[1:] != (2,) or src.shape != dst.shape:
        raise ValueError("SPATIAL_CV_POINT_SHAPE_INVALID")
    if width <= 0 or height <= 0 or not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("SPATIAL_CV_NONFINITE_OR_BAD_SIZE")
    if len(src) < MIN_TRAIN + MIN_HOLDOUT:
        return {"status": "insufficient_total_correspondences", "match_count": int(len(src)),
                "valid_folds": 0, "folds": []}
    if (src[:, 0] < 0).any() or (src[:, 0] >= width).any() or (
            src[:, 1] < 0).any() or (src[:, 1] >= height).any():
        raise ValueError("SPATIAL_CV_POINTS_OUTSIDE_IMAGE")

    quadrant = (src[:, 0] >= width / 2).astype(np.int32) + (
        2 * (src[:, 1] >= height / 2).astype(np.int32)
    )
    folds: list[dict] = []
    for q in range(4):
        hold = quadrant == q
        train = ~hold
        status = "ok"
        row = {"quadrant": q, "train_matches": int(train.sum()),
               "holdout_matches": int(hold.sum())}
        if train.sum() < MIN_TRAIN or hold.sum() < MIN_HOLDOUT:
            status = "insufficient_spatial_support"
        else:
            # Each held-out quadrant is completely excluded from robust fitting.
            cv2.setRNGSeed(711 + q)
            matrix, inlier_mask = cv2.estimateAffine2D(
                src[train], dst[train], method=cv2.RANSAC,
                ransacReprojThreshold=PIXEL_TOLERANCE,
                maxIters=2000, confidence=0.99, refineIters=10,
            )
            if matrix is None or inlier_mask is None:
                status = "fit_failed"
            elif not np.isfinite(matrix).all():
                status = "fit_nonfinite"
            else:
                estimate = src[hold] @ matrix[:, :2].T + matrix[:, 2]
                errors = np.linalg.norm(dst[hold] - estimate, axis=1)
                row.update(
                    heldout_error_median_px=float(np.median(errors)),
                    heldout_error_p90_px=float(np.percentile(errors, 90)),
                    heldout_fraction_within_3px=float(np.mean(errors <= PIXEL_TOLERANCE)),
                    train_inlier_fraction=float(np.mean(inlier_mask.ravel() != 0)),
                )
        row["status"] = status
        folds.append(row)

    valid = [item for item in folds if item["status"] == "ok"]
    return {
        "status": "ok" if len(valid) >= 2 else "insufficient_valid_spatial_folds",
        "match_count": int(len(src)),
        "valid_folds": len(valid),
        "folds": folds,
        "crossfit_error_median_px": (
            float(np.median([r["heldout_error_median_px"] for r in valid])) if valid else None
        ),
        "crossfit_worst_quadrant_median_px": (
            float(max(r["heldout_error_median_px"] for r in valid)) if valid else None
        ),
        "mean_heldout_fraction_within_3px": (
            float(np.mean([r["heldout_fraction_within_3px"] for r in valid])) if valid else None
        ),
        "interpretation": "unlabeled_image_plane_spatial_transfer_not_camera_pose",
    }


def klt_matches(frame_a: np.ndarray, frame_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if frame_a.ndim != 2 or frame_a.shape != frame_b.shape:
        raise ValueError("SPATIAL_CV_GRAYSCALE_FRAME_MISMATCH")
    points = cv2.goodFeaturesToTrack(
        frame_a, maxCorners=500, qualityLevel=0.02, minDistance=6, blockSize=7
    )
    if points is None:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty.copy()
    nxt, valid, _ = cv2.calcOpticalFlowPyrLK(
        frame_a, frame_b, points, None, winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if nxt is None or valid is None:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty.copy()
    mask = valid.ravel() == 1
    a, b = points.reshape(-1, 2)[mask], nxt.reshape(-1, 2)[mask]
    # Reject flow endpoints that leave the observable image plane.
    height, width = frame_a.shape
    bounded = ((a[:, 0] >= 0) & (a[:, 0] < width)
               & (a[:, 1] >= 0) & (a[:, 1] < height)
               & (b[:, 0] >= 0) & (b[:, 0] < width)
               & (b[:, 1] >= 0) & (b[:, 1] < height))
    return a[bounded], b[bounded]


def analyze_one_dev_clip(row: dict, shards: Path, frame_count: int, width: int) -> dict:
    path = shards / f"k700_val_{row['shard_index']:03d}.tar.gz"
    digest = audit.file_sha256(path)
    with tempfile.TemporaryDirectory(prefix="mvvg-spatial-cv-dev-") as tmp:
        tmp_video = Path(tmp) / "dev.mp4"
        with tarfile.open(path, "r:gz") as archive:
            member = archive.getmember(row["archive_member"])
            if not member.isfile():
                raise RuntimeError("SPATIAL_CV_NONREGULAR_VIDEO_MEMBER")
            with archive.extractfile(member) as source, tmp_video.open("wb") as destination:
                shutil.copyfileobj(source, destination)
        frames = audit.read_sampled_grayscale(tmp_video, frame_count, width)

    pairs = []
    for frame_a, frame_b in zip(frames, frames[1:]):
        started = time.perf_counter()
        src, dst = klt_matches(frame_a, frame_b)
        matched_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        result = crossfit_affine(src, dst, frame_a.shape[1], frame_a.shape[0])
        crossfit_ms = (time.perf_counter() - started) * 1000
        result["klt_cpu_elapsed_ms"] = float(matched_ms)
        result["four_fold_cpu_elapsed_ms"] = float(crossfit_ms)
        pairs.append(result)

    valid = [p for p in pairs if p["status"] == "ok"]
    median = lambda field: float(np.median([p[field] for p in valid])) if valid else None
    return {
        "action_class": row["label"],
        "dev_video_id": row["video_id"],
        "shard_sha256": digest,
        "frame_count": len(frames),
        "total_pairs": len(pairs),
        "valid_crossfit_pairs": len(valid),
        "median_crossfit_error_px": median("crossfit_error_median_px"),
        "median_worst_quadrant_error_px": median("crossfit_worst_quadrant_median_px"),
        "median_heldout_fraction_within_3px": median("mean_heldout_fraction_within_3px"),
        "median_klt_cpu_elapsed_ms": float(np.median([p["klt_cpu_elapsed_ms"] for p in pairs])),
        "median_four_fold_cpu_elapsed_ms": float(np.median([p["four_fold_cpu_elapsed_ms"] for p in pairs])),
        "pairs": pairs,
    }


def run(selection: Path, shards: Path, output: Path, frame_count: int = 11, width: int = 512) -> dict:
    if frame_count < 2 or width < 64:
        raise ValueError("SPATIAL_CV_INVALID_SAMPLING")
    before = audit.selection_sha256(selection)
    rows = audit.dev_rows(json.loads(selection.read_text(encoding="utf-8")))
    absent = [row["shard_index"] for row in rows if not (
        shards / f"k700_val_{row['shard_index']:03d}.tar.gz").is_file()]
    if absent:
        raise FileNotFoundError(f"SPATIAL_CV_MISSING_DEV_SHARDS:{absent}")
    scenes = [analyze_one_dev_clip(row, shards, frame_count, width) for row in rows]
    if audit.selection_sha256(selection) != before:
        raise RuntimeError("SPATIAL_CV_SELECTION_DRIFT")
    report = {
        "scope": "DEV_ONLY_SIX_CLIPS_NO_TEST_MEDIA_OR_PRIVATE_ORACLE",
        "selection_sha256": before,
        "source_sha256": audit.file_sha256(Path(__file__)),
        "sampling": {"frames_per_dev_video": frame_count, "width_px": width},
        "protocol": "quadrant_leave_one_out_KLT_affine_RANSAC",
        "warning": "Not camera pose, global background identification, or target tracking accuracy.",
        "scenes": scenes,
    }
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != payload:
            raise RuntimeError("SPATIAL_CV_REPORT_ALREADY_EXISTS_DIFFERENT")
    else:
        with output.open("xb") as f:
            f.write(payload)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-frames", type=int, default=11)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()
    result = run(args.selection, args.shards, args.output, args.sample_frames, args.width)
    for scene in result["scenes"]:
        print("DEV_SPATIAL_CV", scene["action_class"], "valid",
              scene["valid_crossfit_pairs"], "/", scene["total_pairs"],
              "median_holdout_px", scene["median_crossfit_error_px"],
              "worst_quadrant_px", scene["median_worst_quadrant_error_px"], flush=True)
    print("DEV_SPATIAL_CV_COMPLETE", result["selection_sha256"], flush=True)


if __name__ == "__main__":
    main()
