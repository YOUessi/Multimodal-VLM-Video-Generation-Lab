"""DEV-only three-frame anchor provenance falsification of background motion.

Three images do NOT prove a feature belongs to static 3D background. The
controlled assumption is that frame-0 is an unobstructed background witness.
Every reported acceptance uses only the visible images; synthetic latent
motion truth is accessed strictly after all decisions. Not a production tracker.
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

import dev_affine_evidence_audit as audit
import dev_counterfactual_motion_layers as control
import dev_peripheral_anchor_motion as peripheral
import dev_temporal_bridge_affine as bridge
from dev_competing_motion_audit import _agreement_distance

MIN_ANCHORS = 16
MIN_INLIERS = 12
MIN_QUADRANTS = 3
MIN_INLIER_FRACTION = 0.60
MAX_CYCLE_PX = 1.5
RANSAC_TOLERANCE = 2.5
MAX_DISAGREEMENT_PX = 3.0
CONDITIONS = (
    "clean_global", "lighting_shift", "center_foreground",
    "peripheral_foreground", "one_corner_background",
    "full_screen_unrelated_foreground",
    "full_screen_observationally_equivalent",
)
BG = np.array([[1., 0., control.DX], [0., 1., control.DY]], dtype=np.float32)
FG = np.array([[1., 0., -11.], [0., 1., 9.]], dtype=np.float32)


def _move(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def synthesize_triplet(frame: np.ndarray, condition: str, seed: int):
    """Known-2D synthetic control; truth is returned but not used by the gate."""
    control._validate_gray(frame)
    if condition not in CONDITIONS:
        raise ValueError("TEMPORAL_PROVENANCE_UNKNOWN_CONDITION")
    h, w = frame.shape
    source = frame.copy()
    mid = _move(frame, BG)
    two_step = BG.copy()
    two_step[:, 2] *= 2
    end = _move(frame, two_step)
    observable = True
    rng = np.random.default_rng(seed)

    if condition in {"center_foreground", "peripheral_foreground",
                     "one_corner_background"}:
        yy, xx = np.indices((h, w))
        if condition == "center_foreground":
            cover = ((np.abs(xx - w * .5) < w * .39)
                     & (np.abs(yy - h * .5) < h * .37))
        elif condition == "peripheral_foreground":
            # The background is visible ONLY in a centered rectangular island.
            # Most border pixels belong to the moving foreground.
            cover = ~((np.abs(xx - w * .5) < w * .20)
                      & (np.abs(yy - h * .5) < h * .24))
        else:
            # Background confined to a single quadrant, should be abstained.
            cover = ~((xx < w * .42) & (yy < h * .42))
        fg = rng.integers(0, 256, frame.shape, dtype=np.uint8)
        fg = cv2.GaussianBlur(fg, (3, 3), 0.7)
        mid[cover] = fg[cover]
        moving = _move(fg, FG)
        end[cover] = moving[cover]
    elif condition == "lighting_shift":
        end = np.clip(end.astype(np.float32) * .72 + 22, 0, 255).astype(np.uint8)
    elif condition == "full_screen_unrelated_foreground":
        end = cv2.GaussianBlur(
            rng.integers(0, 256, frame.shape, dtype=np.uint8), (3, 3), .7)
        observable = False
    elif condition == "full_screen_observationally_equivalent":
        # A moving foreground with the SAME visible texture as the previous
        # background is observationally indistinguishable from camera motion.
        end = _move(mid, FG)
        observable = False
    gx, gy = np.meshgrid(np.linspace(25, w - 25, 13),
                         np.linspace(25, h - 25, 9))
    grid = np.stack((gx.ravel(), gy.ravel()), axis=-1).astype(np.float32)
    return (source, mid, end), BG.copy(), grid, observable


def _valid(pts: np.ndarray, w: int, h: int) -> np.ndarray:
    return (np.isfinite(pts).all(axis=1)
            & (pts[:, 0] >= 2) & (pts[:, 0] < w - 2)
            & (pts[:, 1] >= 2) & (pts[:, 1] < h - 2))


def _hop(a: np.ndarray, b: np.ndarray, pts: np.ndarray):
    """Forward-backward-consistent KLT, with no synthetic mask or label."""
    if len(pts) == 0:
        return pts.copy(), np.zeros(0, dtype=bool)
    kwargs = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                            30, .01))
    future, s1, _ = cv2.calcOpticalFlowPyrLK(
        a, b, pts.reshape(-1, 1, 2), None, **kwargs)
    if future is None or s1 is None:
        return pts.copy(), np.zeros(len(pts), dtype=bool)
    past, s2, _ = cv2.calcOpticalFlowPyrLK(
        b, a, future, None, **kwargs)
    nxt = future.reshape(-1, 2)
    h, w = a.shape
    ok = s1.ravel().astype(bool) & _valid(pts, w, h) & _valid(nxt, w, h)
    if past is None or s2 is None:
        return nxt, np.zeros(len(pts), dtype=bool)
    rewind = past.reshape(-1, 2)
    err = np.linalg.norm(rewind - pts, axis=1)
    ok &= (s2.ravel().astype(bool) & _valid(rewind, w, h)
           & np.isfinite(err) & (err <= MAX_CYCLE_PX))
    return nxt, ok


def _quadrants(points: np.ndarray, height: int, width: int) -> int:
    q = (points[:, 0] >= width / 2).astype(np.int8)
    q += 2 * (points[:, 1] >= height / 2).astype(np.int8)
    return int(len(np.unique(q))) if len(q) else 0


def provenance_evidence(frames: tuple[np.ndarray, ...],
                        primary: np.ndarray | None) -> dict:
    """Track anchors established at t0 across t1/t2; no oracle reads."""
    if (len(frames) != 3 or any(f.ndim != 2 or f.dtype != np.uint8
                                 or f.shape != frames[0].shape for f in frames)
            or min(frames[0].shape) < 90):
        raise ValueError("TEMPORAL_PROVENANCE_BAD_TRIPLET")
    if (primary is not None and (np.shape(primary) != (2, 3)
                                 or not np.isfinite(primary).all())):
        raise ValueError("TEMPORAL_PROVENANCE_INVALID_PRIMARY")
    h, w = frames[0].shape
    report = {
        "detected": 0, "survived_first_hop": 0, "survived_both_hops": 0,
        "inliers": 0, "inlier_fraction": None, "quadrants": 0,
        "disagreement_px": None, "supported": False,
        "agrees_with_primary": False, "background_identified": False,
        "status": "no_initial_anchors",
    }
    if primary is None:
        return {**report, "status": "missing_primary"}
    features = cv2.goodFeaturesToTrack(
        frames[0], maxCorners=500, qualityLevel=.02,
        minDistance=6, blockSize=7)
    if features is None:
        return report
    pts0 = features.reshape(-1, 2)
    report["detected"] = int(len(pts0))
    pts1, ok1 = _hop(frames[0], frames[1], pts0)
    mids = pts1[ok1]
    report["survived_first_hop"] = int(len(mids))
    if len(mids) < MIN_ANCHORS:
        return {**report, "status": "insufficient_first_hop"}
    pts2, ok2 = _hop(frames[1], frames[2], mids)
    mids, ends = mids[ok2], pts2[ok2]
    report["survived_both_hops"] = int(len(mids))
    if len(mids) < MIN_ANCHORS:
        return {**report, "status": "insufficient_two_hop_anchors"}
    if _quadrants(mids, h, w) < MIN_QUADRANTS:
        return {**report, "status": "anchors_not_spatially_distributed"}
    cv2.setRNGSeed(1553)
    affine, mask = cv2.estimateAffine2D(
        mids, ends, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_TOLERANCE,
        maxIters=2000, confidence=.99, refineIters=10)
    if affine is None or mask is None or not np.isfinite(affine).all():
        return {**report, "status": "no_provenance_affine"}
    err = np.linalg.norm(mids @ affine[:, :2].T
                         + affine[:, 2] - ends, axis=1)
    inside = err <= RANSAC_TOLERANCE
    n = int(inside.sum())
    frac = float(n / len(mids))
    q = _quadrants(mids[inside], h, w)
    supported = (n >= MIN_INLIERS and frac >= MIN_INLIER_FRACTION
                 and q >= MIN_QUADRANTS)
    disagreement = _agreement_distance(primary, affine, h, w)
    agrees = bool(supported and disagreement <= MAX_DISAGREEMENT_PX)
    return {**report, "inliers": n, "inlier_fraction": frac,
            "quadrants": q, "supported": bool(supported),
            "disagreement_px": float(disagreement),
            "agrees_with_primary": agrees,
            "status": ("provenance_agrees" if agrees
                       else "provenance_disagrees" if supported
                       else "provenance_low_support")}


def _rate(n: int, d: int):
    return float(n / d) if d else None


def summarize(rows: list[dict]) -> dict:
    def count(method: str):
        selected = [r for r in rows if r[method]]
        wrong = sum(r["synthetic_bg_wrong"] is True for r in selected)
        return {"accepted": len(selected), "wrong_accepted": wrong,
                "coverage": _rate(len(selected), len(rows)),
                "wrong_among_accepted": _rate(wrong, len(selected))}
    return {
        "pairs": len(rows), "latent_background": sum(
            not r["background_observable"] for r in rows),
        "baseline": count("old_accept"), "periphery": count("ring_accept"),
        "provenance": count("provenance_accept"),
        "supported_provenance": sum(r["supported"] for r in rows),
        "median_extra_cpu_ms": (float(np.median([
            r["extra_cpu_ms"] for r in rows])) if rows else None),
        "median_survivors": (float(np.median([
            r["survived_both_hops"] for r in rows])) if rows else None),
    }


def run(selection: Path, shards: Path, output: Path) -> dict:
    # Verify all public DEV pins BEFORE opening any media.
    if audit.selection_sha256(selection) != bridge.EXPECTED_SELECTION_SHA256:
        raise RuntimeError("TEMPORAL_PROVENANCE_SELECTION_PIN_MISMATCH")
    manifest_path = selection.with_name("download_manifest.json")
    if audit.file_sha256(manifest_path) != bridge.EXPECTED_DOWNLOAD_MANIFEST_SHA256:
        raise RuntimeError("TEMPORAL_PROVENANCE_MANIFEST_PIN_MISMATCH")
    selected = audit.dev_rows(json.loads(selection.read_text(encoding="utf-8")))
    if len(selected) != 6:
        raise RuntimeError("TEMPORAL_PROVENANCE_REQUIRES_SIX_DEV")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_rows = {c: [] for c in CONDITIONS}
    scenes = []
    for idx, row in enumerate(selected):
        entry = manifest["shards"].get(f"{row['shard_index']:03d}")
        if (not isinstance(entry, dict)
                or entry.get("archive_member") != row["archive_member"]
                or not isinstance(entry.get("sha256"), str)
                or len(entry["sha256"]) != 64):
            raise RuntimeError("TEMPORAL_PROVENANCE_MEMBER_PIN_MISMATCH")
        frames = control._clip_frames(row, shards, entry["sha256"])
        per_clip = {c: [] for c in CONDITIONS}
        for anchor, frame in frames:
            seed = zlib.crc32((row["video_id"] + ":" + str(anchor)).encode())
            for condition in CONDITIONS:
                triplet, truth, grid, visible = synthesize_triplet(
                    frame, condition, seed)
                old, primary = control.estimate_pair(triplet[1], triplet[2])
                pair = bridge.chained_cycle_matches(list(triplet[1:]))
                edge = peripheral.peripheral_motion_evidence(
                    pair["src"], pair["dst"], frame.shape[0], frame.shape[1],
                    primary)
                start = time.perf_counter()
                provenance = provenance_evidence(triplet, primary)
                elapsed = (time.perf_counter() - start) * 1000
                # Freeze decisions BEFORE opening synthetic truth.
                row_result = {
                    "condition": condition, "anchor_frame": anchor,
                    "background_observable": visible,
                    "old_accept": bool(old["gate_accept"]),
                    "ring_accept": bool(old["gate_accept"]
                                        and edge["agrees_with_primary"]),
                    "provenance_accept": bool(old["gate_accept"]
                                              and provenance["agrees_with_primary"]),
                    "extra_cpu_ms": float(elapsed),
                    **provenance,
                }
                row_result.update(control.score_known_affine(primary, truth, grid))
                per_clip[condition].append(row_result)
                all_rows[condition].append(row_result)
        scenes.append({"dev_video_id": row["video_id"],
                       "action_class": row["label"],
                       "summary": {c: summarize(per_clip[c]) for c in CONDITIONS}})
        print("DEV_PROVENANCE_SCENE", idx + 1, row["label"], flush=True)
    report = {
        "scope": "REUSED_SHA_PINNED_DEV_2D_SYNTHETIC_TRIPLETS_NOT_NEW_TEST",
        "selection_sha256": bridge.EXPECTED_SELECTION_SHA256,
        "manifest_sha256": bridge.EXPECTED_DOWNLOAD_MANIFEST_SHA256,
        "source_sha256": audit.file_sha256(Path(__file__)),
        "conditions": list(CONDITIONS),
        "fixed_preexperiment_hypothesis": {
            "min_anchors": MIN_ANCHORS, "min_inliers": MIN_INLIERS,
            "min_quadrants": MIN_QUADRANTS,
            "min_inlier_fraction": MIN_INLIER_FRACTION,
            "max_cycle_px": MAX_CYCLE_PX,
            "max_disagreement_px": MAX_DISAGREEMENT_PX,
        },
        "overall": {c: summarize(all_rows[c]) for c in CONDITIONS},
        "scenes": scenes,
        "limitations": [
            "t0 is SYNTHETICALLY guaranteed clean in this experiment; real video is not.",
            "A feature tracked from t0 is not semantically proven static background.",
            "Full-screen appearance-equivalent foreground makes latent camera motion unidentifiable.",
            "Generated 2D frame motion is not actual 3D camera pose or tracking Jaccard.",
            "DEV videos were previously used in research; not a fresh independent TEST.",
            "Extra CPU ms excludes decoding and primary estimation; no production deployment.",
            "Synthetic latent error under full occlusion is NOT a visible-world error.",
        ],
    }
    if audit.selection_sha256(selection) != bridge.EXPECTED_SELECTION_SHA256:
        raise RuntimeError("TEMPORAL_PROVENANCE_SELECTION_DRIFT")
    payload = (json.dumps(report, ensure_ascii=False, sort_keys=True,
                          indent=2, allow_nan=False) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != payload:
            raise RuntimeError("TEMPORAL_PROVENANCE_WRITE_ONCE_CONFLICT")
    else:
        with output.open("xb") as handle:
            handle.write(payload)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", required=True, type=Path)
    p.add_argument("--shards", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    report = run(args.selection, args.shards, args.output)
    for k, v in report["overall"].items():
        print("DEV_PROVENANCE_RESULT", k, json.dumps(v, sort_keys=True),
              flush=True)


if __name__ == "__main__":
    main()
