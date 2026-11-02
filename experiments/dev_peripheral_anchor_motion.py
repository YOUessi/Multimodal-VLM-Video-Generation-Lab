"""DEV-only peripheral evidence falsification of global 2D motion.

The frame boundary is an *assumption*, never semantic proof of background.
A disagreement between well-supported boundary motion and dominant motion
can refute the dominant motion as a unique global transform. Agreement does
not establish which layer is the camera/background. This module is NOT a
production tracker and must not be selected using opened TEST labels.
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
import dev_temporal_bridge_affine as bridge
from dev_competing_motion_audit import _finite_points, _agreement_distance

# Exploratory design fixed in source before the first DEV experiment.
PERIPHERAL_FRACTION = 0.10
MIN_RING_MATCHES = 16
MIN_RING_INLIERS = 12
MIN_RING_QUADRANTS = 3
MIN_RING_INLIER_FRACTION = 0.60
RANSAC_RESIDUAL_PX = 2.5
MAX_PRIMARY_DISAGREEMENT_PX = 3.0
FULL_SCREEN_FOREGROUND = "full_screen_foreground_hidden_background"
CONDITIONS = (*control.CONDITIONS, FULL_SCREEN_FOREGROUND)


def _peripheral_mask(points: np.ndarray, height: int, width: int) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    return ((x <= PERIPHERAL_FRACTION * width)
            | (x >= (1.0 - PERIPHERAL_FRACTION) * width)
            | (y <= PERIPHERAL_FRACTION * height)
            | (y >= (1.0 - PERIPHERAL_FRACTION) * height))


def _quadrants(points: np.ndarray, height: int, width: int) -> int:
    q = (points[:, 0] >= width / 2).astype(np.uint8)
    q += 2 * (points[:, 1] >= height / 2).astype(np.uint8)
    return int(np.unique(q).size)


def peripheral_motion_evidence(
    source: np.ndarray, dest: np.ndarray, height: int, width: int,
    primary: np.ndarray | None,
) -> dict:
    """Check independent edge agreement; never claim camera identification."""
    src, dst = _finite_points(source, dest, height, width)
    out = {
        "matched": len(src), "ring_matches": 0, "ring_inliers": 0,
        "ring_inlier_fraction": None, "ring_quadrants": 0,
        "ring_disagreement_px": None, "peripheral_supported": False,
        "agrees_with_primary": False, "background_identified": False,
        "status": "insufficient_correspondences",
    }
    if len(src) < MIN_RING_MATCHES:
        return out
    if (primary is None or np.shape(primary) != (2, 3)
            or not np.isfinite(primary).all()):
        return {**out, "status": "invalid_primary"}
    ring = _peripheral_mask(src, height, width)
    ring_src, ring_dst = src[ring], dst[ring]
    out["ring_matches"] = len(ring_src)
    out["ring_quadrants"] = _quadrants(ring_src, height, width) if len(ring_src) else 0
    if len(ring_src) < MIN_RING_MATCHES:
        return {**out, "status": "insufficient_peripheral_matches"}
    if out["ring_quadrants"] < MIN_RING_QUADRANTS:
        return {**out, "status": "periphery_not_spatially_distributed"}
    cv2.setRNGSeed(1207)
    candidate, mask = cv2.estimateAffine2D(
        ring_src, ring_dst, method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_RESIDUAL_PX,
        maxIters=2000, confidence=0.99, refineIters=10,
    )
    if candidate is None or mask is None or not np.isfinite(candidate).all():
        return {**out, "status": "no_peripheral_affine"}
    errors = np.linalg.norm(ring_src @ candidate[:, :2].T
                            + candidate[:, 2] - ring_dst, axis=1)
    n_inliers = int((errors <= RANSAC_RESIDUAL_PX).sum())
    frac = float(n_inliers / len(ring_src))
    supported = (n_inliers >= MIN_RING_INLIERS
                 and frac >= MIN_RING_INLIER_FRACTION
                 and _quadrants(ring_src[errors <= RANSAC_RESIDUAL_PX],
                                height, width) >= MIN_RING_QUADRANTS)
    disagreement = _agreement_distance(primary, candidate, height, width)
    agrees = bool(supported and disagreement <= MAX_PRIMARY_DISAGREEMENT_PX)
    return {
        **out, "ring_inliers": n_inliers, "ring_inlier_fraction": frac,
        "ring_disagreement_px": disagreement,
        "peripheral_supported": bool(supported),
        "agrees_with_primary": agrees,
        "status": ("periphery_agrees_with_primary" if agrees
                   else "periphery_disagrees_with_primary" if supported
                   else "periphery_fit_low_support"),
    }


def synthesize_hidden_background(frame: np.ndarray, seed: int):
    """Adversarial no-background-visible control: observationally unidentifiable.

    Visible pixels are ALL moving foreground; latent camera/background shift
    is never observed. No image-only algorithm can recover the hidden shift.
    This is a falsification example, not evidence of real-world bg accuracy.
    """
    control._validate_gray(frame)
    rng = np.random.default_rng(seed)
    source = rng.integers(0, 256, frame.shape, dtype=np.uint8)
    source = cv2.GaussianBlur(source, (3, 3), 0.7)
    foreground_shift = np.array([[1., 0., -11.], [0., 1., 9.]],
                                dtype=np.float32)
    target = cv2.warpAffine(source, foreground_shift,
                            (frame.shape[1], frame.shape[0]),
                            borderMode=cv2.BORDER_REFLECT101)
    truth = np.array([[1., 0., control.DX],
                      [0., 1., control.DY]], dtype=np.float32)
    # These points belong to a LATENT occluded plane; no visible bg grid exists.
    gx, gy = np.meshgrid(np.linspace(24, frame.shape[1] - 24, 9),
                         np.linspace(24, frame.shape[0] - 24, 7))
    latent_grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    return source, target, truth, latent_grid


def _fraction(n: int, d: int):
    return float(n / d) if d else None


def summarize(rows: list[dict]) -> dict:
    accepted_old = [r for r in rows if r["old_accept"]]
    accepted_new = [r for r in rows if r["new_accept"]]
    old_wrong = sum(r["synthetic_bg_wrong"] is True for r in accepted_old)
    new_wrong = sum(r["synthetic_bg_wrong"] is True for r in accepted_new)
    return {
        "pairs": len(rows), "background_visible": (rows[0]["background_visible"] if rows else None),
        "periphery_supported": sum(r["peripheral_supported"] for r in rows),
        "periphery_disagreed": sum(r["status"] == "periphery_disagrees_with_primary" for r in rows),
        "old_accepted": len(accepted_old), "new_accepted": len(accepted_new),
        "old_latent_or_visible_wrong_accepted": old_wrong,
        "new_latent_or_visible_wrong_accepted": new_wrong,
        "old_coverage": _fraction(len(accepted_old), len(rows)),
        "new_coverage": _fraction(len(accepted_new), len(rows)),
        "new_wrong_among_accepted": _fraction(new_wrong, len(accepted_new)),
        "median_extra_cpu_ms": (float(np.median([r["extra_cpu_ms"] for r in rows]))
                                if rows else None),
    }


def run(selection: Path, shards: Path, output: Path) -> dict:
    # All pins are checked before reading ANY media. No TEST IDs or oracles.
    if audit.selection_sha256(selection) != bridge.EXPECTED_SELECTION_SHA256:
        raise RuntimeError("PERIPHERAL_SELECTION_PIN_MISMATCH")
    manifest_file = selection.with_name("download_manifest.json")
    if audit.file_sha256(manifest_file) != bridge.EXPECTED_DOWNLOAD_MANIFEST_SHA256:
        raise RuntimeError("PERIPHERAL_MANIFEST_PIN_MISMATCH")
    selected = audit.dev_rows(json.loads(selection.read_text(encoding="utf-8")))
    if len(selected) != 6:
        raise RuntimeError("PERIPHERAL_EXPECTED_SIX_DEV_CLIPS")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    scenes = []
    totals = {c: [] for c in CONDITIONS}
    for index, row in enumerate(selected):
        entry = manifest["shards"].get(f"{row['shard_index']:03d}")
        if (not isinstance(entry, dict)
                or entry.get("archive_member") != row["archive_member"]
                or not isinstance(entry.get("sha256"), str)
                or len(entry["sha256"]) != 64):
            raise RuntimeError("PERIPHERAL_MEMBER_PIN_MISMATCH")
        frames = control._clip_frames(row, shards, entry["sha256"])
        records = []
        for anchor, frame in frames:
            seed = zlib.crc32((row["video_id"] + ":" + str(anchor)).encode())
            for condition in CONDITIONS:
                hidden = condition == FULL_SCREEN_FOREGROUND
                if hidden:
                    src, dst, truth, grid = synthesize_hidden_background(frame, seed)
                else:
                    src, dst, truth, grid = control.synthesize(frame, condition, seed)
                old, model = control.estimate_pair(src, dst)
                start = time.perf_counter()
                tracks = bridge.chained_cycle_matches([src, dst])
                evidence = peripheral_motion_evidence(
                    tracks["src"], tracks["dst"], src.shape[0], src.shape[1], model
                )
                elapsed = float((time.perf_counter() - start) * 1000)
                new_accept = bool(old["gate_accept"] and evidence["agrees_with_primary"])
                record = {
                    "condition": condition, "anchor_frame": anchor,
                    "fit_valid": old["fit_valid"], "old_accept": old["gate_accept"],
                    "new_accept": new_accept,
                    "extra_cpu_ms": elapsed, "background_visible": not hidden,
                    **evidence,
                }
                # Both gates evaluated BEFORE oracle is ever read.
                record.update(control.score_known_affine(model, truth, grid))
                records.append(record)
        for c in CONDITIONS:
            totals[c].extend(r for r in records if r["condition"] == c)
        scenes.append({
            "dev_video_id": row["video_id"], "action_class": row["label"],
            "condition_summaries": {
                c: summarize([r for r in records if r["condition"] == c])
                for c in CONDITIONS
            },
        })
        print("DEV_PERIPHERAL_SCENE", index + 1, row["label"], len(records), flush=True)
    overall = {c: summarize(totals[c]) for c in CONDITIONS}
    report = {
        "scope": "REUSED_SHA_PINNED_SIX_DEV_CLIPS_SYNTHETIC_2D_ONLY",
        "selection_sha256": bridge.EXPECTED_SELECTION_SHA256,
        "manifest_sha256": bridge.EXPECTED_DOWNLOAD_MANIFEST_SHA256,
        "source_sha256": audit.file_sha256(Path(__file__)),
        "conditions": list(CONDITIONS),
        "hypothesis": {
            "peripheral_fraction": PERIPHERAL_FRACTION,
            "min_ring_matches": MIN_RING_MATCHES,
            "min_ring_inliers": MIN_RING_INLIERS,
            "min_ring_quadrants": MIN_RING_QUADRANTS,
            "min_ring_inlier_fraction": MIN_RING_INLIER_FRACTION,
            "max_primary_disagreement_px": MAX_PRIMARY_DISAGREEMENT_PX,
        },
        "scenes": scenes,
        "overall": overall,
        "limitations": [
            "Frame-edge regions are NOT verified static background.",
            "Full-screen foreground hides all background: camera motion is unidentifiable.",
            "Source frames are previously researched DEV; not fresh independent TEST.",
            "2D synthetic translation, not physical camera pose, tracking or occlusion ground truth.",
            "No newly tuned production tracker or inference on sealed TEST.",
        ],
    }
    payload = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != payload:
            raise RuntimeError("PERIPHERAL_WRITE_ONCE_CONFLICT")
    else:
        with output.open("xb") as f:
            f.write(payload)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", required=True, type=Path)
    p.add_argument("--shards", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    report = run(args.selection, args.shards, args.output)
    for c, row in report["overall"].items():
        print("DEV_PERIPHERAL_RESULT", c, json.dumps(row, sort_keys=True), flush=True)
    print("DEV_PERIPHERAL_COMPLETE", report["source_sha256"], flush=True)


if __name__ == "__main__":
    main()
