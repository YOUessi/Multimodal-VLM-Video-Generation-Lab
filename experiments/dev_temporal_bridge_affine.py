"""DEV-only controlled temporal-bridging study of 2D affine evidence.

Compare direct 30-frame KLT to 3x10, 10x3 and 30x1 sequential
forward/backward tracks, using identical initial corners and anchor frames.
Spatial leave-one-quadrant-out fits provide *diagnostic* evidence only.
No TEST media or private target labels are opened; production tracker unchanged.
"""
from __future__ import annotations

import argparse
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
EXPECTED_DOWNLOAD_MANIFEST_SHA256 = "f4f116c84092ef8c99d8f9c423226467a3475c868943bc460994a816175658e1"
STRIDE_FRAMES = 30
ANCHOR_COUNT = 8
WIDTH_PX = 512
MAX_CYCLE_ERROR_PX = 1.5
METHODS = {"direct_30": 30, "bridge_3x10": 10, "bridge_10x3": 3, "bridge_30x1": 1}


def _inside(pts: np.ndarray, width: int, height: int) -> np.ndarray:
    return (np.isfinite(pts).all(axis=1) &
            (pts[:, 0] >= 0) & (pts[:, 0] < width) &
            (pts[:, 1] >= 0) & (pts[:, 1] < height))


def chained_cycle_matches(frames: list[np.ndarray]) -> dict:
    """Preserve a single set of initial corners across ALL temporal hops."""
    if len(frames) < 2 or any(
        f.ndim != 2 or f.dtype != np.uint8 or f.shape != frames[0].shape
        for f in frames
    ):
        raise ValueError("TEMPORAL_BRIDGE_REQUIRES_EQUAL_UINT8_GRAY_FRAMES")
    h, w = frames[0].shape
    empty = np.empty((0, 2), dtype=np.float32)
    corners = cv2.goodFeaturesToTrack(
        frames[0], maxCorners=500, qualityLevel=0.02, minDistance=6, blockSize=7
    )
    if corners is None:
        return {"src": empty, "dst": empty, "detected": 0,
                "survivors": 0, "hop_survivors": [0] * (len(frames) - 1)}
    start = corners.reshape(-1, 2)
    src = start.copy()
    cur = start.copy()
    hop_survivors = []
    for a, b in zip(frames, frames[1:]):
        if not len(cur):
            hop_survivors.append(0)
            continue
        forward, status_f, _ = cv2.calcOpticalFlowPyrLK(
            a, b, cur.reshape(-1, 1, 2), None, winSize=(21, 21),
            maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        if forward is None or status_f is None:
            src, cur = empty.copy(), empty.copy()
            hop_survivors.append(0)
            continue
        backward, status_b, _ = cv2.calcOpticalFlowPyrLK(
            b, a, forward, None, winSize=(21, 21),
            maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        nxt = forward.reshape(-1, 2)
        good = (status_f.ravel().astype(bool) & _inside(cur, w, h) &
                _inside(nxt, w, h))
        if backward is None or status_b is None:
            good[:] = False
        else:
            back = backward.reshape(-1, 2)
            cycle = np.linalg.norm(back - cur, axis=1)
            good &= (status_b.ravel().astype(bool) & _inside(back, w, h) &
                     np.isfinite(cycle) & (cycle <= MAX_CYCLE_ERROR_PX))
        src, cur = src[good], nxt[good]
        hop_survivors.append(int(len(src)))
    return {"src": src.astype(np.float32), "dst": cur.astype(np.float32),
            "detected": int(len(start)), "survivors": int(len(src)),
            "hop_survivors": hop_survivors}


def _read_requested_frames(video: Path, *, width: int,
                           anchors: int, extent: int) -> tuple[float, list[int], dict[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise RuntimeError("TEMPORAL_BRIDGE_CANNOT_OPEN_DEV_VIDEO")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if count < extent + anchors + 1 or not np.isfinite(fps) or fps <= 0:
            raise RuntimeError("TEMPORAL_BRIDGE_SHORT_VIDEO_OR_BAD_FPS")
        starts = sorted(set(np.linspace(0, count - extent - 1, anchors).round().astype(int).tolist()))
        if len(starts) != anchors:
            raise RuntimeError("TEMPORAL_BRIDGE_DUPLICATE_ANCHORS")
        needed = {a + j for a in starts for j in range(extent + 1)}
        result = {}
        for i in range(max(needed) + 1):
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f"TEMPORAL_BRIDGE_DECODE_FAILED_FRAME:{i}")
            if i in needed:
                height = max(16, round(width * bgr.shape[0] / bgr.shape[1]))
                gray = cv2.cvtColor(cv2.resize(
                    bgr, (width, height), interpolation=cv2.INTER_AREA
                ), cv2.COLOR_BGR2GRAY)
                result[i] = gray
        if len(result) != len(needed):
            raise RuntimeError("TEMPORAL_BRIDGE_INCOMPLETE_DECODE")
        return fps, starts, result
    finally:
        cap.release()


def _median(rows, field):
    values = [r[field] for r in rows if r.get(field) is not None]
    return float(np.median(values)) if values else None


def evaluate_frame_sequence(frames: dict[int, np.ndarray], anchor: int) -> dict:
    row = {"anchor_frame": anchor, "methods": {}}
    for method, step in METHODS.items():
        indices = list(range(anchor, anchor + STRIDE_FRAMES + 1, step))
        times = [frames[i] for i in indices]
        t0 = time.perf_counter()
        match = chained_cycle_matches(times)
        matching_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        cv = spatial.crossfit_affine(match["src"], match["dst"],
                                     times[0].shape[1], times[0].shape[0])
        fitting_ms = (time.perf_counter() - t0) * 1000
        row["methods"][method] = {
            "detected": match["detected"], "survivors": match["survivors"],
            "hop_survivors": match["hop_survivors"], "spatial": cv,
            "match_cpu_ms": float(matching_ms),
            "crossfit_cpu_ms": float(fitting_ms),
        }
    return row


def _summarize(pairs: list[dict]) -> dict:
    result = {}
    for method in METHODS:
        rows = [p["methods"][method] for p in pairs]
        valid = [r for r in rows if r["spatial"]["status"] == "ok"]
        joint = [p for p in pairs if
                 p["methods"]["direct_30"]["spatial"]["status"] == "ok" and
                 p["methods"][method]["spatial"]["status"] == "ok"]
        result[method] = {
            "pair_count": len(rows),
            "valid_spatial_crossfit": len(valid),
            "median_survivors": _median(rows, "survivors"),
            "median_survival_fraction": float(np.median([
                r["survivors"] / max(1, r["detected"]) for r in rows
            ])) if rows else None,
            "median_holdout_error_px_conditional": _median([
                {"err": r["spatial"]["crossfit_error_median_px"]} for r in valid
            ], "err"),
            "median_matching_cpu_ms": _median(rows, "match_cpu_ms"),
            "paired_with_direct_valid": len(joint),
            "direct_holdout_error_px_paired": _median([
                {"err": p["methods"]["direct_30"]["spatial"]["crossfit_error_median_px"]}
                for p in joint
            ], "err"),
            "method_holdout_error_px_paired": _median([
                {"err": p["methods"][method]["spatial"]["crossfit_error_median_px"]}
                for p in joint
            ], "err"),
        }
    return result


def analyze_dev_clip(row: dict, shards: Path, anchor_count: int, width: int,
                     expected_shard_sha256: str) -> dict:
    shard = shards / f"k700_val_{row['shard_index']:03d}.tar.gz"
    checksum = audit.file_sha256(shard)
    if checksum != expected_shard_sha256:
        raise RuntimeError("TEMPORAL_BRIDGE_SHARD_PIN_MISMATCH")
    with tempfile.TemporaryDirectory(prefix="mvvg-bridge-dev-") as tmp:
        video = Path(tmp) / "dev.mp4"
        with tarfile.open(shard, "r:gz") as archive:
            member = archive.getmember(row["archive_member"])
            if not member.isfile():
                raise RuntimeError("TEMPORAL_BRIDGE_NONREGULAR_MEMBER")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError("TEMPORAL_BRIDGE_MISSING_MEMBER")
            with source, video.open("wb") as dest:
                shutil.copyfileobj(source, dest)
        fps, anchors, frames = _read_requested_frames(
            video, width=width, anchors=anchor_count, extent=STRIDE_FRAMES
        )
    pairs = [evaluate_frame_sequence(frames, anchor) for anchor in anchors]
    return {"dev_video_id": row["video_id"], "action_class": row["label"],
            "source_shard_sha256": checksum, "fps": fps,
            "anchors": anchors, "summary": _summarize(pairs), "pairs": pairs}


def run(selection: Path, shards: Path, output: Path, *,
        expected_selection_sha256: str = EXPECTED_SELECTION_SHA256,
        anchor_count: int = ANCHOR_COUNT, width: int = WIDTH_PX,
        expected_download_manifest_sha256: str = EXPECTED_DOWNLOAD_MANIFEST_SHA256) -> dict:
    if anchor_count < 2 or width < 64:
        raise ValueError("TEMPORAL_BRIDGE_INVALID_PROTOCOL")
    pin = audit.selection_sha256(selection)
    if pin != expected_selection_sha256:
        raise RuntimeError("TEMPORAL_BRIDGE_SELECTION_PIN_MISMATCH")
    rows = audit.dev_rows(json.loads(selection.read_text(encoding="utf-8")))
    if len(rows) != 6:
        raise RuntimeError("TEMPORAL_BRIDGE_DEV_SPLIT_EXPECTED_SIX")
    manifest_path = selection.with_name("download_manifest.json")
    if not manifest_path.is_file() or audit.file_sha256(manifest_path) != expected_download_manifest_sha256:
        raise RuntimeError("TEMPORAL_BRIDGE_DOWNLOAD_MANIFEST_PIN_MISMATCH")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified_shards = {}
    for row in rows:
        key = f"{row['shard_index']:03d}"
        entry = manifest["shards"].get(key)
        if (not isinstance(entry, dict)
                or entry.get("archive_member") != row["archive_member"]
                or not isinstance(entry.get("sha256"), str)
                or len(entry["sha256"]) != 64):
            raise RuntimeError("TEMPORAL_BRIDGE_DEV_SHARD_MANIFEST_MISMATCH")
        verified_shards[key] = entry["sha256"]
    absent = [r["shard_index"] for r in rows if not (
        shards / f"k700_val_{r['shard_index']:03d}.tar.gz").is_file()]
    if absent:
        raise FileNotFoundError(f"TEMPORAL_BRIDGE_DEV_SHARDS_MISSING:{absent}")
    scenes = [analyze_dev_clip(row, shards, anchor_count, width,
                               verified_shards[f"{row['shard_index']:03d}"])
              for row in rows]
    if audit.selection_sha256(selection) != pin:
        raise RuntimeError("TEMPORAL_BRIDGE_SELECTION_DRIFT")
    report = {
        "scope": "DEV_ONLY_NO_TEST_MEDIA_NO_PRIVATE_ORACLE",
        "selection_sha256": pin,
        "download_manifest_sha256": expected_download_manifest_sha256,
        "source_sha256": audit.file_sha256(Path(__file__)),
        "protocol": "same_anchors_and_initial_corners_direct_vs_cycle_checked_multihop_2D_affine_spatial_holdout",
        "settings": {"extent_frames": STRIDE_FRAMES, "anchor_count": anchor_count,
                     "resize_width_px": width, "cycle_threshold_px": MAX_CYCLE_ERROR_PX,
                     "methods": METHODS},
        "limitations": ("Not camera pose or ground-truth target tracking. Conditional spatial errors "
                        "cannot be compared without paired coverage; CPU costs are diagnostics only."),
        "scenes": scenes,
        "overall": _summarize([p for scene in scenes for p in scene["pairs"]]),
    }
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != payload:
            raise RuntimeError("TEMPORAL_BRIDGE_OUTPUT_EXISTS_DIFFERENT")
    else:
        with output.open("xb") as f:
            f.write(payload)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.selection, args.shards, args.output)
    for method, metrics in report["overall"].items():
        print("DEV_BRIDGE", method, json.dumps(metrics, sort_keys=True), flush=True)
    print("DEV_BRIDGE_COMPLETE", report["selection_sha256"], flush=True)


if __name__ == "__main__":
    main()
