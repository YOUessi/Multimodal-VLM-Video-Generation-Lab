"""Batch47E: DEV-only counterfactual motion layers with known image-plane truth.

The source is a SHA-pinned DEV frame from six existing Kinetics clips.  The
experiment creates a deterministic synthetic global 2D translation and
(optional) independently moving foreground, illumination and occlusion.
Ground truth is the generated translation, *not* a physical camera pose or
TAP-Vid target annotation.  Never read TEST clips or private target labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

import dev_affine_evidence_audit as audit
import dev_spatial_affine_crosscheck as spatial
import dev_temporal_bridge_affine as bridge

CONDITIONS = (
    "clean_affine", "lighting_shift", "local_foreground",
    "dominant_foreground", "foreground_and_occlusion",
)
DX, DY = 6.0, -4.0
MAX_GROUND_TRUTH_ERROR_PX = 2.0
# Fixed prior to experiment; these are hypotheses, not DEV-tuned values.
MIN_GATE_FOLDS = 3
MAX_WORST_HOLDOUT_PX = 3.0
MIN_FRACTION_WITHIN_3PX = 0.65
ANCHORS = 8
WIDTH = 512


def _validate_gray(frame: np.ndarray) -> None:
    if (frame.ndim != 2 or frame.dtype != np.uint8
            or min(frame.shape) < 90):
        raise ValueError("COUNTERFACTUAL_REQUIRES_UINT8_GRAY_FRAME")


def synthesize(frame: np.ndarray, condition: str, seed: int):
    """Return synthetic pair, true global affine, and untouched-background grid.

    Masks exclude original foreground and its *destination* placement, newly
    occluded pixels, and invalid image borders. Oracle grid is never used to
    fit or gate the KLT correspondences.
    """
    _validate_gray(frame)
    if condition not in CONDITIONS:
        raise ValueError("COUNTERFACTUAL_UNKNOWN_CONDITION")
    h, w = frame.shape
    matrix = np.array([[1., 0., DX], [0., 1., DY]], dtype=np.float32)
    source = frame.copy()
    dest = cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT101)
    src_fg = None
    dst_fg = None
    if "foreground" in condition:
        dominant = condition == "dominant_foreground"
        pw, ph = round(w * (0.76 if dominant else 0.42)), round(h * (0.72 if dominant else 0.5))
        x0, y0 = round(w * (0.5 if dominant else 0.26) - pw / 2), round(h * (0.5 if dominant else 0.3) - ph / 2)
        # Deterministic textured moving layer, with an independent displacement.
        rng = np.random.default_rng(seed)
        texture = rng.integers(0, 256, (ph, pw), dtype=np.uint8)
        texture = cv2.GaussianBlur(texture, (3, 3), 0.7)
        source[y0:y0 + ph, x0:x0 + pw] = texture
        xf, yf = x0 + int(DX) - 17, y0 + int(DY) + 13
        dest[yf:yf + ph, xf:xf + pw] = texture
        src_fg = (x0, y0, x0 + pw, y0 + ph)
        dst_fg = (xf, yf, xf + pw, yf + ph)
    if condition == "lighting_shift":
        dest = np.clip(dest.astype(np.float32) * 0.72 + 22., 0, 255).astype(np.uint8)
    occlusion = None
    if condition == "foreground_and_occlusion":
        occlusion = (round(w * 0.72), round(h * 0.15), round(w * 0.97), round(h * 0.85))
        x0, y0, x1, y1 = occlusion
        dest[y0:y1, x0:x1] = 16

    gx, gy = np.meshgrid(np.linspace(25, w - 25, 13),
                         np.linspace(25, h - 25, 9))
    grid = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    target = grid + np.array([DX, DY])
    valid = (target[:, 0] >= 10) & (target[:, 0] < w - 10)
    valid &= (target[:, 1] >= 10) & (target[:, 1] < h - 10)
    if src_fg is not None:
        x0, y0, x1, y1 = src_fg
        valid &= ~((grid[:, 0] >= x0) & (grid[:, 0] < x1)
                   & (grid[:, 1] >= y0) & (grid[:, 1] < y1))
        x0, y0, x1, y1 = dst_fg
        valid &= ~((target[:, 0] >= x0) & (target[:, 0] < x1)
                   & (target[:, 1] >= y0) & (target[:, 1] < y1))
    if occlusion is not None:
        x0, y0, x1, y1 = occlusion
        valid &= ~((target[:, 0] >= x0) & (target[:, 0] < x1)
                   & (target[:, 1] >= y0) & (target[:, 1] < y1))
    grid = grid[valid].astype(np.float32)
    if len(grid) < 12:
        raise RuntimeError("COUNTERFACTUAL_TOO_FEW_BACKGROUND_GRID_POINTS")
    return source, dest, matrix, grid


def gate_crossfit(crossfit: dict) -> bool:
    """Conservative *hypothetical* gate using only unlabeled match evidence."""
    return bool(
        crossfit.get("status") == "ok"
        and crossfit.get("valid_folds", 0) >= MIN_GATE_FOLDS
        and crossfit.get("crossfit_worst_quadrant_median_px") is not None
        and crossfit["crossfit_worst_quadrant_median_px"] <= MAX_WORST_HOLDOUT_PX
        and crossfit.get("mean_heldout_fraction_within_3px") is not None
        and crossfit["mean_heldout_fraction_within_3px"] >= MIN_FRACTION_WITHIN_3PX
    )


def estimate_pair(source: np.ndarray, dest: np.ndarray) -> dict:
    """Estimate without access to synthetic truth; gate without oracle labels."""
    start = time.perf_counter()
    tracks = bridge.chained_cycle_matches([source, dest])
    points = tracks["src"]
    endpoints = tracks["dst"]
    crossfit = spatial.crossfit_affine(points, endpoints, source.shape[1], source.shape[0])
    matrix = None
    if len(points) >= 12:
        cv2.setRNGSeed(1129)
        matrix, mask = cv2.estimateAffine2D(
            points, endpoints, method=cv2.RANSAC,
            ransacReprojThreshold=spatial.PIXEL_TOLERANCE,
            maxIters=2000, confidence=0.99, refineIters=10
        )
    else:
        mask = None
    usable = (matrix is not None and np.isfinite(matrix).all()
              and mask is not None)
    result = {
        "detected": tracks["detected"], "matched": tracks["survivors"],
        "fit_valid": bool(usable), "gate_accept": bool(usable and gate_crossfit(crossfit)),
        "inlier_fraction": (float(np.mean(mask != 0)) if usable else None),
        "crossfit_status": crossfit["status"],
        "valid_folds": crossfit["valid_folds"],
        "worst_holdout_median_px": crossfit.get("crossfit_worst_quadrant_median_px"),
        "fraction_holdout_within_3px": crossfit.get("mean_heldout_fraction_within_3px"),
        "cpu_ms_match_fit_crossfit": float((time.perf_counter() - start) * 1000),
    }
    return result, (matrix if usable else None)


def score_known_affine(fitted: np.ndarray | None, truth: np.ndarray, grid: np.ndarray) -> dict:
    """Oracle evaluation strictly after fitting and gate decisions are frozen."""
    if fitted is None:
        return {"synthetic_bg_median_error_px": None,
                "synthetic_bg_p90_error_px": None, "synthetic_bg_wrong": None}
    pred = grid @ fitted[:, :2].T + fitted[:, 2]
    expected = grid @ truth[:, :2].T + truth[:, 2]
    errors = np.linalg.norm(pred - expected, axis=1)
    return {
        "synthetic_bg_median_error_px": float(np.median(errors)),
        "synthetic_bg_p90_error_px": float(np.percentile(errors, 90)),
        "synthetic_bg_wrong": bool(float(np.median(errors)) > MAX_GROUND_TRUTH_ERROR_PX),
    }


def _rate(num: int, denom: int) -> float | None:
    return float(num / denom) if denom else None


def summary(rows: list[dict]) -> dict:
    fit = [r for r in rows if r["fit_valid"]]
    accepted = [r for r in rows if r["gate_accept"]]
    wrong = [r for r in fit if r["synthetic_bg_wrong"]]
    false_accept = [r for r in accepted if r["synthetic_bg_wrong"]]
    return {
        "total": len(rows), "fit_count": len(fit), "gate_accept_count": len(accepted),
        "baseline_wrong_count": len(wrong), "gate_wrong_count": len(false_accept),
        "baseline_wrong_fraction_over_all": _rate(len(wrong), len(rows)),
        "gate_wrong_fraction_over_all": _rate(len(false_accept), len(rows)),
        "gate_wrong_fraction_conditional": _rate(len(false_accept), len(accepted)),
        "fit_coverage": _rate(len(fit), len(rows)),
        "gate_coverage": _rate(len(accepted), len(rows)),
        "median_fit_error_px": (float(np.median([r["synthetic_bg_median_error_px"] for r in fit])) if fit else None),
        "median_gate_error_px": (float(np.median([r["synthetic_bg_median_error_px"] for r in accepted])) if accepted else None),
        "median_cpu_ms_match_fit_crossfit": float(np.median([r["cpu_ms_match_fit_crossfit"] for r in rows])) if rows else None,
    }


def _clip_frames(dev_row: dict, shards: Path, expected_shard_sha256: str):
    shard = shards / f"k700_val_{dev_row['shard_index']:03d}.tar.gz"
    if audit.file_sha256(shard) != expected_shard_sha256:
        raise RuntimeError("COUNTERFACTUAL_DEV_SHARD_PIN_MISMATCH")
    with tempfile.TemporaryDirectory(prefix="mvvg-b47e-dev-") as tmp:
        video = Path(tmp) / "dev.mp4"
        with tarfile.open(shard, "r:gz") as archive:
            member = archive.getmember(dev_row["archive_member"])
            if not member.isfile():
                raise RuntimeError("COUNTERFACTUAL_NONREGULAR_MEMBER")
            with archive.extractfile(member) as source, video.open("wb") as sink:
                shutil.copyfileobj(source, sink)
        _, anchors, frames = bridge._read_requested_frames(
            video, width=WIDTH, anchors=ANCHORS, extent=bridge.STRIDE_FRAMES
        )
    return [(anchor, frames[anchor]) for anchor in anchors]


def run(selection: Path, shards: Path, output: Path) -> dict:
    if audit.selection_sha256(selection) != bridge.EXPECTED_SELECTION_SHA256:
        raise RuntimeError("COUNTERFACTUAL_SELECTION_PIN_MISMATCH")
    manifest_path = selection.with_name("download_manifest.json")
    if audit.file_sha256(manifest_path) != bridge.EXPECTED_DOWNLOAD_MANIFEST_SHA256:
        raise RuntimeError("COUNTERFACTUAL_MANIFEST_PIN_MISMATCH")
    dev = audit.dev_rows(json.loads(selection.read_text(encoding="utf-8")))
    if len(dev) != 6:
        raise RuntimeError("COUNTERFACTUAL_EXPECTED_SIX_DEV_CLIPS")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenes = []
    for scene_idx, row in enumerate(dev):
        shard_key = f"{row['shard_index']:03d}"
        entry = manifest["shards"].get(shard_key)
        if (not isinstance(entry, dict)
                or entry.get("archive_member") != row["archive_member"]
                or not isinstance(entry.get("sha256"), str)
                or len(entry["sha256"]) != 64):
            raise RuntimeError("COUNTERFACTUAL_MANIFEST_MEMBER_MISMATCH")
        anchors = _clip_frames(row, shards, entry["sha256"])
        outputs = []
        for anchor, frame in anchors:
            seed = zlib.crc32((row["video_id"] + ":" + str(anchor)).encode())
            for condition in CONDITIONS:
                source, target, truth, grid = synthesize(frame, condition, seed)
                estimated, model = estimate_pair(source, target)
                # The generated oracle is accessed only AFTER deciding acceptance.
                estimated.update(score_known_affine(model, truth, grid))
                estimated.update(anchor_frame=anchor, condition=condition,
                                 synthetic_background_grid_count=len(grid))
                outputs.append(estimated)
        scenes.append({
            "dev_video_id": row["video_id"], "action_class": row["label"],
            "condition_summaries": {
                condition: summary([r for r in outputs if r["condition"] == condition])
                for condition in CONDITIONS
            }, "rows": outputs,
        })
        print("DEV_COUNTERFACTUAL_SCENE", scene_idx + 1, row["label"],
              len(outputs), flush=True)
    report = {
        "scope": "DEV_ONLY_GENERATED_2D_TRUTH_NO_TEST_NO_PRIVATE_ORACLE",
        "selection_sha256": bridge.EXPECTED_SELECTION_SHA256,
        "manifest_sha256": bridge.EXPECTED_DOWNLOAD_MANIFEST_SHA256,
        "source_sha256": audit.file_sha256(Path(__file__)),
        "protocol": {"anchors_per_clip": ANCHORS, "width_px": WIDTH,
                     "conditions": list(CONDITIONS), "truth_global_dxdy_px": [DX, DY],
                     "gate": {"min_valid_folds": MIN_GATE_FOLDS,
                              "max_worst_holdout_median_px": MAX_WORST_HOLDOUT_PX,
                              "min_fraction_holdout_within_3px": MIN_FRACTION_WITHIN_3PX},
                     "wrong_error_gt_px": MAX_GROUND_TRUTH_ERROR_PX},
        "overall": {condition: summary([
            r for s in scenes for r in s["rows"] if r["condition"] == condition
        ]) for condition in CONDITIONS},
        "scenes": scenes,
        "limitations": [
            "Synthetic image-plane known translation, NOT true camera motion or 3D geometry.",
            "DEV frames reused for exploratory controlled ablations; NOT fresh independent TEST.",
            "Synthetic foreground texture and occlusion may not match natural domains.",
            "CPU timings include LK and fitting, not decoding or full tracker.",
            "No proposed threshold promoted on Batch46E opened TEST."
        ],
    }
    if audit.selection_sha256(selection) != bridge.EXPECTED_SELECTION_SHA256:
        raise RuntimeError("COUNTERFACTUAL_SELECTION_DRIFT")
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != payload:
            raise RuntimeError("COUNTERFACTUAL_WRITE_ONCE_REPORT_CONFLICT")
    else:
        with output.open("xb") as dest:
            dest.write(payload)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", required=True, type=Path)
    p.add_argument("--shards", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    report = run(args.selection, args.shards, args.output)
    for condition, row in report["overall"].items():
        print("DEV_SYNTHETIC", condition, json.dumps(row, sort_keys=True), flush=True)
    print("DEV_COUNTERFACTUAL_COMPLETE", report["selection_sha256"], flush=True)


if __name__ == "__main__":
    main()
