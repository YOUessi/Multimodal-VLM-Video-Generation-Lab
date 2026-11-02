"""DEV-only temporal-cadence sensitivity of spatial affine diagnostics.

All reported motion is in the 2D image plane. Four temporal strides use the
same anchor frames per DEV clip, and both plain and forward-backward-validated
KLT points are assessed out-of-region. No TEST media, oracle, or thresholds for
the production tracker are accessed or modified.
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
import dev_spatial_affine_crosscheck as spatial

EXPECTED_SELECTION_SHA256 = "0d49c3096e8d302596c6044eefd313508d072e113d880b50ec74cc237f84384c"
DEFAULT_STRIDES = (1, 3, 10, 30)
DEFAULT_ANCHORS = 8
DEFAULT_WIDTH = 512
FB_MAX_CYCLE_ERROR_PX = 1.5


def _inside(pts: np.ndarray, width: int, height: int) -> np.ndarray:
    return (np.isfinite(pts).all(axis=1) &
            (pts[:, 0] >= 0) & (pts[:, 0] < width) &
            (pts[:, 1] >= 0) & (pts[:, 1] < height))


def matches_with_cycle_filter(frame_a: np.ndarray, frame_b: np.ndarray) -> dict:
    """Use identical source corners for baseline KLT and cycle-checked KLT."""
    if frame_a.ndim != 2 or frame_a.dtype != np.uint8 or frame_a.shape != frame_b.shape:
        raise ValueError("TEMPORAL_AFFINE_EXPECTS_EQUAL_UINT8_GRAY_FRAMES")
    height, width = frame_a.shape
    empty = np.empty((0, 2), dtype=np.float32)
    corners = cv2.goodFeaturesToTrack(
        frame_a, maxCorners=500, qualityLevel=0.02, minDistance=6, blockSize=7
    )
    if corners is None:
        return {"src": empty, "dst": empty, "src_fb": empty, "dst_fb": empty,
                "detected": 0, "forward_valid": 0, "cycle_valid": 0}
    forward, valid_f, _ = cv2.calcOpticalFlowPyrLK(
        frame_a, frame_b, corners, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if forward is None or valid_f is None:
        return {"src": empty, "dst": empty, "src_fb": empty, "dst_fb": empty,
                "detected": len(corners), "forward_valid": 0, "cycle_valid": 0}
    backward, valid_b, _ = cv2.calcOpticalFlowPyrLK(
        frame_b, frame_a, forward, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    a, b = corners.reshape(-1, 2), forward.reshape(-1, 2)
    fmask = valid_f.ravel().astype(bool) & _inside(a, width, height) & _inside(b, width, height)
    if backward is None or valid_b is None:
        fbmask = np.zeros(len(corners), dtype=bool)
    else:
        back = backward.reshape(-1, 2)
        cycle = np.linalg.norm(back - a, axis=1)
        fbmask = (fmask & valid_b.ravel().astype(bool) &
                  _inside(back, width, height) &
                  np.isfinite(cycle) & (cycle <= FB_MAX_CYCLE_ERROR_PX))
    return {
        "src": a[fmask], "dst": b[fmask],
        "src_fb": a[fbmask], "dst_fb": b[fbmask],
        "detected": int(len(corners)),
        "forward_valid": int(fmask.sum()),
        "cycle_valid": int(fbmask.sum()),
    }


def _sample_frames(video: Path, strides: tuple[int, ...], anchor_count: int,
                   width: int) -> tuple[float, list[int], dict[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise RuntimeError("TEMPORAL_AFFINE_VIDEO_OPEN_FAILED")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if count < max(strides) + 2 or not np.isfinite(fps) or fps <= 0:
            raise RuntimeError("TEMPORAL_AFFINE_VIDEO_TOO_SHORT_OR_FPS_INVALID")
        start_limit = count - max(strides) - 1
        anchors = sorted(set(np.linspace(0, start_limit, anchor_count).round().astype(int).tolist()))
        if len(anchors) != anchor_count:
            raise RuntimeError("TEMPORAL_AFFINE_DUPLICATE_ANCHORS")
        needed = sorted(set(anchors) | {a + s for a in anchors for s in strides})
        frames = {}
        for index in needed:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"TEMPORAL_AFFINE_FRAME_READ_FAILED:{index}")
            height = max(16, round(width * frame.shape[0] / frame.shape[1]))
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            frames[index] = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return fps, anchors, frames
    finally:
        cap.release()


def analyze_clip(row: dict, shards: Path, strides: tuple[int, ...],
                 anchors_n: int, width: int) -> dict:
    path = shards / ("k700_val_%03d.tar.gz" % row["shard_index"])
    archive_hash = audit.file_sha256(path)
    with tempfile.TemporaryDirectory(prefix="mvvg-temporal-dev-") as tmp:
        video_path = Path(tmp) / "dev.mp4"
        with tarfile.open(path, "r:gz") as archive:
            member = archive.getmember(row["archive_member"])
            if not member.isfile():
                raise RuntimeError("TEMPORAL_AFFINE_VIDEO_MEMBER_NOT_REGULAR")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError("TEMPORAL_AFFINE_MEMBER_UNREADABLE")
            with source, video_path.open("wb") as target:
                shutil.copyfileobj(source, target)
        fps, anchors, frames = _sample_frames(video_path, strides, anchors_n, width)

    by_stride = []
    for stride in strides:
        pairs = []
        for anchor in anchors:
            t0 = time.perf_counter()
            m = matches_with_cycle_filter(frames[anchor], frames[anchor + stride])
            match_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            raw = spatial.crossfit_affine(m["src"], m["dst"],
                                          frames[anchor].shape[1], frames[anchor].shape[0])
            cycle = spatial.crossfit_affine(m["src_fb"], m["dst_fb"],
                                            frames[anchor].shape[1], frames[anchor].shape[0])
            crossfit_ms = (time.perf_counter() - t0) * 1000
            pairs.append({
                "anchor_frame": anchor,
                "detected": m["detected"],
                "forward_valid": m["forward_valid"],
                "cycle_valid": m["cycle_valid"],
                "raw": raw, "cycle": cycle,
                "klt_forward_backward_cpu_ms": float(match_ms),
                "crossfits_cpu_ms": float(crossfit_ms),
            })
        valid_raw = [p for p in pairs if p["raw"]["status"] == "ok"]
        valid_cycle = [p for p in pairs if p["cycle"]["status"] == "ok"]
        paired = [p for p in pairs if p["raw"]["status"] == p["cycle"]["status"] == "ok"]

        def median(items: list, getter) -> float | None:
            return float(np.median([getter(p) for p in items])) if items else None

        by_stride.append({
            "stride_frames": stride,
            "stride_seconds_at_clip_fps": float(stride / fps),
            "pair_count": len(pairs),
            "raw_valid_folds_at_least_two": len(valid_raw),
            "cycle_valid_folds_at_least_two": len(valid_cycle),
            "joint_valid_pair_count": len(paired),
            "median_detected_corners": median(pairs, lambda p: p["detected"]),
            "median_forward_valid": median(pairs, lambda p: p["forward_valid"]),
            "median_cycle_valid": median(pairs, lambda p: p["cycle_valid"]),
            "median_cycle_retain_fraction": median(
                pairs, lambda p: p["cycle_valid"] / max(1, p["forward_valid"])
            ),
            "raw_median_holdout_error_px_conditional": median(
                valid_raw, lambda p: p["raw"]["crossfit_error_median_px"]
            ),
            "cycle_median_holdout_error_px_conditional": median(
                valid_cycle, lambda p: p["cycle"]["crossfit_error_median_px"]
            ),
            # Compare only identical anchors if both methods support spatial CV.
            "raw_median_holdout_error_px_paired": median(
                paired, lambda p: p["raw"]["crossfit_error_median_px"]
            ),
            "cycle_median_holdout_error_px_paired": median(
                paired, lambda p: p["cycle"]["crossfit_error_median_px"]
            ),
            "median_matching_cpu_ms": median(pairs, lambda p: p["klt_forward_backward_cpu_ms"]),
            "median_two_crossfits_cpu_ms": median(pairs, lambda p: p["crossfits_cpu_ms"]),
            "pairs": pairs,
        })
    return {
        "dev_video_id": row["video_id"],
        "action_class": row["label"],
        "source_shard_sha256": archive_hash,
        "fps": fps,
        "anchor_frames": anchors,
        "strides": by_stride,
    }


def run(selection: Path, shards: Path, output: Path, *,
        expected_selection_sha256: str = EXPECTED_SELECTION_SHA256,
        strides: tuple[int, ...] = DEFAULT_STRIDES, anchor_count: int = DEFAULT_ANCHORS,
        width: int = DEFAULT_WIDTH) -> dict:
    if (not strides or any(not isinstance(s, int) or s < 1 for s in strides)
            or len(set(strides)) != len(strides) or anchor_count < 2 or width < 64):
        raise ValueError("TEMPORAL_AFFINE_INVALID_PROTOCOL")
    pin = audit.selection_sha256(selection)
    if pin != expected_selection_sha256:
        raise RuntimeError("TEMPORAL_AFFINE_SELECTION_PIN_MISMATCH")
    rows = audit.dev_rows(json.loads(selection.read_text(encoding="utf-8")))
    missing = [r["shard_index"] for r in rows if not (
        shards / ("k700_val_%03d.tar.gz" % r["shard_index"])).is_file()]
    if missing:
        raise FileNotFoundError(f"TEMPORAL_AFFINE_DEV_SHARDS_MISSING:{missing}")
    scenes = [analyze_clip(r, shards, strides, anchor_count, width) for r in rows]
    if audit.selection_sha256(selection) != pin:
        raise RuntimeError("TEMPORAL_AFFINE_SELECTION_DRIFT")
    report = {
        "scope": "DEV_ONLY_NO_TEST_VIDEO_NO_PRIVATE_ORACLE",
        "selection_sha256": pin,
        "source_sha256": audit.file_sha256(Path(__file__)),
        "protocol": "matched_temporal_anchors_KLT_raw_vs_forward_backward_spatial_crossfit",
        "sampling": {"strides_frames": list(strides), "anchor_count_per_video": anchor_count,
                     "resize_width_px": width, "fb_cycle_threshold_px": FB_MAX_CYCLE_ERROR_PX},
        "warning": "2D image-plane only: temporal stability, not physical camera pose, target tracking accuracy, or production latency.",
        "scenes": scenes,
    }
    payload = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != payload:
            raise RuntimeError("TEMPORAL_AFFINE_REPORT_EXISTS_DIFFERENT")
    else:
        with output.open("xb") as stream:
            stream.write(payload)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    args = parser.parse_args()
    report = run(args.selection, args.shards, args.output, width=args.width)
    for scene in report["scenes"]:
        for stride in scene["strides"]:
            print("DEV_TEMPORAL", scene["action_class"], "stride", stride["stride_frames"],
                  "raw_cv", stride["raw_valid_folds_at_least_two"],
                  "cycle_cv", stride["cycle_valid_folds_at_least_two"],
                  "joint", stride["joint_valid_pair_count"],
                  "raw_holdout", stride["raw_median_holdout_error_px_conditional"],
                  "cycle_holdout", stride["cycle_median_holdout_error_px_conditional"],
                  flush=True)
    print("DEV_TEMPORAL_COMPLETE", report["selection_sha256"], flush=True)


if __name__ == "__main__":
    main()
