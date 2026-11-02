"""DEV-only image-plane motion audit for class-disjoint Kinetics clips.

This code deliberately reads only the public preflight selection metadata and
the six DEV video members. It never touches TEST frames or any private oracle.
Nothing here selects thresholds, models, or a physical camera/3D motion label.

All distances are in pixels of the uniformly resized image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import cv2
import numpy as np

GRID_CELLS = 4
DEFAULT_SAMPLE_FRAMES = 11
DEFAULT_WIDTH = 512


def selection_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dev_rows(selection: dict) -> list[dict]:
    videos = selection.get("videos")
    if not isinstance(videos, list):
        raise ValueError("DEV_AUDIT_SELECTION_MISSING_VIDEOS")
    rows = [item for item in videos if item.get("split") == "dev"]
    if len(rows) != 6:
        raise ValueError("DEV_AUDIT_REQUIRES_SIX_DEV_CLIPS")
    ids = [item.get("video_id") for item in rows]
    if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != 6:
        raise ValueError("DEV_AUDIT_DUPLICATE_OR_MISSING_DEV_ID")
    for row in rows:
        if not isinstance(row.get("archive_member"), str) or not row["archive_member"].endswith(".mp4"):
            raise ValueError("DEV_AUDIT_MEMBER_INVALID")
        if not isinstance(row.get("shard_index"), int):
            raise ValueError("DEV_AUDIT_SHARD_INDEX_INVALID")
    return rows


def _median_length(displacements: np.ndarray) -> float:
    return float(np.median(np.linalg.norm(displacements, axis=1)))


def affine_evidence(src: np.ndarray, dst: np.ndarray, width: int, height: int) -> dict:
    """Fit dominant 2D affine correspondence; residual is NOT world motion."""
    src, dst = np.asarray(src, dtype=np.float32), np.asarray(dst, dtype=np.float32)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("DEV_AUDIT_POINT_SHAPE_INVALID")
    if width <= 0 or height <= 0:
        raise ValueError("DEV_AUDIT_IMAGE_SIZE_INVALID")
    if len(src) < 12:
        return {"status": "too_few_correspondences", "match_count": len(src)}
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("DEV_AUDIT_NONFINITE_POINTS")

    cv2.setRNGSeed(0)  # Deterministic RANSAC in this single-process DEV audit.
    matrix, flags = cv2.estimateAffine2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0,
        maxIters=2000, confidence=0.99, refineIters=10,
    )
    if matrix is None or flags is None:
        return {"status": "affine_estimation_failed", "match_count": len(src)}
    inside = flags.ravel().astype(bool)
    if not inside.any():
        return {"status": "affine_zero_inliers", "match_count": len(src)}
    transformed = src @ matrix[:, :2].T + matrix[:, 2]
    screen_displacement = dst - src
    compensated_displacement = dst - transformed

    cell_x = np.clip((src[inside, 0] * GRID_CELLS / width).astype(int), 0, GRID_CELLS - 1)
    cell_y = np.clip((src[inside, 1] * GRID_CELLS / height).astype(int), 0, GRID_CELLS - 1)
    spatial_cells = len(set(zip(cell_x.tolist(), cell_y.tolist())))
    screen_median_xy = np.median(screen_displacement, axis=0)
    # Subtracting a fitted transform describes dominant-image-plane-relative
    # motion ONLY: it does not recover physical pose, parallax, or a 3D frame.
    result = {
        "status": "ok",
        "match_count": int(len(src)),
        "inlier_count": int(inside.sum()),
        "inlier_fraction": float(inside.mean()),
        "spatial_cells_occupied_4x4": spatial_cells,
        "spatial_support_fraction_4x4": float(spatial_cells / (GRID_CELLS ** 2)),
        "median_screen_displacement_px": _median_length(screen_displacement),
        "median_screen_flow_xy_px": [float(x) for x in screen_median_xy],
        "median_affine_residual_all_px": _median_length(compensated_displacement),
        "median_affine_residual_inliers_px": _median_length(compensated_displacement[inside]),
        "median_affine_residual_outliers_px": (
            _median_length(compensated_displacement[~inside]) if (~inside).any() else None
        ),
        "affine_2x3": [[float(x) for x in row] for row in matrix],
        "interpretation": "image_plane_only_not_physical_camera_motion",
    }
    return result


def klt_pair(frame_a: np.ndarray, frame_b: np.ndarray) -> dict:
    if frame_a.shape != frame_b.shape or frame_a.ndim != 2:
        raise ValueError("DEV_AUDIT_GRAY_FRAME_MISMATCH")
    points = cv2.goodFeaturesToTrack(
        frame_a, maxCorners=500, qualityLevel=0.02, minDistance=6, blockSize=7,
    )
    if points is None or len(points) < 12:
        return {"status": "too_few_features", "match_count": 0 if points is None else len(points)}
    nxt, valid, _ = cv2.calcOpticalFlowPyrLK(
        frame_a, frame_b, points, None, winSize=(21, 21),
        maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if nxt is None or valid is None:
        return {"status": "klt_failed", "match_count": 0}
    used = valid.ravel() == 1
    src, dst = points.reshape(-1, 2)[used], nxt.reshape(-1, 2)[used]
    result = affine_evidence(src, dst, frame_a.shape[1], frame_a.shape[0])
    result["mean_brightness_change_8bit"] = float(
        abs(frame_a.mean(dtype=np.float64) - frame_b.mean(dtype=np.float64))
    )
    result["laplacian_variance_a"] = float(cv2.Laplacian(frame_a, cv2.CV_64F).var())
    return result


def read_sampled_grayscale(video_path: Path, sample_frames: int, width: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError("DEV_AUDIT_VIDEO_OPEN_FAILED")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 2:
            raise RuntimeError("DEV_AUDIT_VIDEO_TOO_SHORT")
        n = max(2, min(sample_frames, count))
        indices = sorted(set(np.linspace(0, count - 1, n).round().astype(int).tolist()))
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("DEV_AUDIT_VIDEO_FRAME_READ_FAILED")
            h, w = frame.shape[:2]
            target_h = max(16, round(width * h / w))
            resized = cv2.resize(frame, (width, target_h), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))
        return frames
    finally:
        cap.release()


def analyze_dev_clip(row: dict, shard_dir: Path, sample_frames: int, width: int) -> dict:
    shard = shard_dir / f"k700_val_{row['shard_index']:03d}.tar.gz"
    if not shard.is_file():
        raise FileNotFoundError("DEV_AUDIT_DEV_SHARD_MISSING:" + shard.name)
    source_hash = file_sha256(shard)
    with tempfile.TemporaryDirectory(prefix="mvvg-dev-motion-") as tmp_dir:
        video_path = Path(tmp_dir) / "dev.mp4"
        with tarfile.open(shard, mode="r:gz") as archive:
            member = archive.getmember(row["archive_member"])
            if not member.isfile():
                raise RuntimeError("DEV_AUDIT_NOT_REGULAR_VIDEO_MEMBER")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError("DEV_AUDIT_MEMBER_UNREADABLE")
            with source, video_path.open("wb") as output:
                shutil.copyfileobj(source, output)
        frames = read_sampled_grayscale(video_path, sample_frames, width)
    pairs = [klt_pair(a, b) for a, b in zip(frames, frames[1:])]
    good = [p for p in pairs if p["status"] == "ok"]
    names = (
        "inlier_fraction", "spatial_support_fraction_4x4",
        "median_screen_displacement_px", "median_affine_residual_all_px",
        "median_affine_residual_inliers_px", "mean_brightness_change_8bit",
        "laplacian_variance_a",
    )
    stats = {
        key + "_median": float(np.median([item[key] for item in good])) if good else None
        for key in names
    }
    return {
        "video_id": row["video_id"],
        "action_class": row["label"],
        "source_shard_sha256": source_hash,
        "sampled_frames": len(frames),
        "pair_count": len(pairs),
        "successful_pairs": len(good),
        "summary": stats,
        "pair_evidence": pairs,
    }


def run(selection_path: Path, shard_dir: Path, output_path: Path,
        sample_frames: int = DEFAULT_SAMPLE_FRAMES, width: int = DEFAULT_WIDTH) -> dict:
    if width < 64 or sample_frames < 2:
        raise ValueError("DEV_AUDIT_INVALID_SAMPLE_SETTINGS")
    input_sha = selection_sha256(selection_path)
    doc = json.loads(selection_path.read_text(encoding="utf-8"))
    rows = dev_rows(doc)
    # Verify completeness *before* calculating, no TEST member ever accessed.
    missing = [
        row["shard_index"] for row in rows
        if not (shard_dir / f"k700_val_{row['shard_index']:03d}.tar.gz").is_file()
    ]
    if missing:
        raise FileNotFoundError("DEV_AUDIT_DEV_SHARDS_MISSING:" + ",".join(map(str, missing)))
    scenes = [analyze_dev_clip(row, shard_dir, sample_frames, width) for row in rows]
    if selection_sha256(selection_path) != input_sha:
        raise RuntimeError("DEV_AUDIT_SELECTION_CHANGED_DURING_RUN")
    report = {
        "scope": "DEV_ONLY_SIX_UNSEEN_ACTION_CLASSES_NO_TEST_FRAMES_OR_ORACLE",
        "selection_sha256": input_sha,
        "analysis_source_sha256": file_sha256(Path(__file__)),
        "resized_width_px": width,
        "sample_frames_per_clip": sample_frames,
        "method": "KLT_correspondences_RANSAC_full_affine_noncalibrated_2D",
        "warning": "Dominant affine compensation is not physical 3D camera motion.",
        "scenes": scenes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Do not silently overwrite an earlier DEV diagnostic.
    payload = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if output_path.exists():
        if output_path.read_bytes() != payload:
            raise RuntimeError("DEV_AUDIT_EXISTING_REPORT_DIFFERS")
    else:
        with output_path.open("xb") as output:
            output.write(payload)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-frames", type=int, default=DEFAULT_SAMPLE_FRAMES)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    args = parser.parse_args()
    result = run(args.selection, args.shards, args.output, args.sample_frames, args.width)
    for scene in result["scenes"]:
        print("DEV_AFFINE_EVIDENCE", scene["action_class"], json.dumps(scene["summary"], sort_keys=True))
    print("DEV_AUDIT_DONE", result["selection_sha256"], len(result["scenes"]))


if __name__ == "__main__":
    main()
