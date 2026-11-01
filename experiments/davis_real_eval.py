"""Real DAVIS pixel tracking with a strict initial-mask query/future-label boundary.

Task: screen-coordinate target mask CENTROID displacement in natural camera
video. This does NOT identify physical world motion, depth, or 3-D trajectory.
No part of the inference path reads GT after the first query frame.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

import davis_real_data as data

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("stationary_pixel", "klt_forward", "klt_forward_backward")
SCORE_THRESHOLDS = (4, 8, 16)


def document(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def write_once(path: Path, doc: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(doc, indent=2, sort_keys=True).encode("utf8") + b"\n"
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError("FROZEN_PREDICTION_OR_REPORT_ALREADY_EXISTS")
        return
    data.save_if_frozen(path, raw)


def pin_prediction_bytes(path: Path):
    """Record immutable trajectory bytes before private-label scoring."""
    digest = (data.sha256(path.read_bytes()) + "\n").encode("ascii")
    data.save_if_frozen(path.with_suffix(".sha256"), digest)


def require_pinned_prediction(path: Path):
    marker = path.with_suffix(".sha256")
    if not marker.is_file():
        raise RuntimeError("UNPINNED_PREDICTION_REFUSED")
    expected = (data.sha256(path.read_bytes()) + "\n").encode("ascii")
    if marker.read_bytes() != expected:
        raise RuntimeError("PREDICTION_SHA256_DRIFT")


def point_xy(points: np.ndarray, query_xy: np.ndarray, start: np.ndarray):
    if points.shape[0] < 3:
        return None
    new = query_xy + np.median(points.reshape(-1, 2) - start.reshape(-1, 2), axis=0)
    if not np.isfinite(new).all():
        return None
    return [round(float(new[0]), 3), round(float(new[1]), 3)]


def klt_track(frames: list, mask: np.ndarray, center: list, *, use_fb: bool,
              min_features: int = 5, fb_error_px: float = 1.5) -> list:
    gray = [cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) for bgr in frames]
    if mask.shape != gray[0].shape:
        raise RuntimeError("QUERY_MASK_DIMENSION_MISMATCH")
    pts = cv2.goodFeaturesToTrack(gray[0], maxCorners=90, qualityLevel=0.015,
                                  minDistance=5, mask=mask)
    trace = [[round(float(center[0]), 3), round(float(center[1]), 3)]]
    if pts is None or len(pts) < min_features:
        return trace + [None] * (len(frames) - 1)
    initial = pts.copy()
    active = pts.copy()
    for k in range(1, len(gray)):
        if len(active) < min_features:
            trace.extend([None] * (len(gray) - k))
            break
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            gray[k-1], gray[k], active, None, winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if nxt is None or status is None:
            trace.extend([None] * (len(gray) - k))
            break
        good = status.flatten().astype(bool) & np.isfinite(nxt.reshape(-1, 2)).all(axis=1)
        xys = nxt.reshape(-1, 2)
        h, w = gray[k].shape
        good &= (xys[:, 0] >= 0) & (xys[:, 0] < w) & (xys[:, 1] >= 0) & (xys[:, 1] < h)
        if use_fb:
            back, bst, _ = cv2.calcOpticalFlowPyrLK(
                gray[k], gray[k-1], nxt, None, winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            if back is None or bst is None:
                good[:] = False
            else:
                residual = np.linalg.norm(back.reshape(-1, 2) - active.reshape(-1, 2), axis=1)
                good &= bst.flatten().astype(bool) & np.isfinite(residual) & (residual <= fb_error_px)
        active = nxt[good]
        initial = initial[good]
        trace.append(point_xy(active, np.asarray(center), initial)
                     if len(active) >= min_features else None)
    if len(trace) != len(frames):
        raise RuntimeError("TRACK_LENGTH_INVALID")
    return trace


def affine_background_delta(frames: list, query_mask: np.ndarray, center: list) -> dict:
    """Diagnostic only: 2-D dominant-background relative motion, *not* world motion."""
    first = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    last = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
    # Exclude initial target; last-frame masks are NOT available to inference.
    use = 255 - cv2.dilate(query_mask, np.ones((23, 23), dtype=np.uint8))
    orb = cv2.ORB_create(nfeatures=1000)
    key0, desc0 = orb.detectAndCompute(first, use)
    key1, desc1 = orb.detectAndCompute(last, None)
    if desc0 is None or desc1 is None or len(key0) < 12 or len(key1) < 12:
        return {"accepted": False, "reason": "FEW_BACKGROUND_FEATURES"}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc0, desc1, k=2)
    good = [a for a, b in pairs if a.distance < 0.78 * b.distance]
    if len(good) < 12:
        return {"accepted": False, "reason": "FEW_BACKGROUND_MATCHES"}
    p0 = np.float32([key0[m.queryIdx].pt for m in good])
    p1 = np.float32([key1[m.trainIdx].pt for m in good])
    A, inlier = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC,
                                            ransacReprojThreshold=3.0,
                                            maxIters=1500, confidence=0.995)
    if A is None or inlier is None:
        return {"accepted": False, "reason": "BACKGROUND_AFFINE_FAILED"}
    n = int(inlier.sum())
    if n < 12 or n / len(good) < 0.5:
        return {"accepted": False, "reason": "BACKGROUND_AFFINE_WEAK",
                "matches": len(good), "inliers": n}
    projected = (A @ np.array([center[0], center[1], 1], dtype=np.float64))
    return {"accepted": True, "reason": "DOMINANT_2D_AFFINE_ONLY",
            "matches": len(good), "inliers": n,
            "projected_background_xy": [round(float(v), 3) for v in projected]}


def scene_predict(dataset: Path, row: dict, query: dict) -> dict:
    start = time.perf_counter()
    frames = [cv2.imread(str(dataset / a["jpg"]), cv2.IMREAD_COLOR) for a in row["assets"]]
    if any(frame is None for frame in frames) or len({f.shape for f in frames}) != 1:
        raise RuntimeError("DAVIS_REAL_DECODE_OR_DIMENSION_DRIFT")
    mask = cv2.imread(str(dataset / query["query_mask"]), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError("DAVIS_REAL_INITIAL_QUERY_MISSING")
    center = query["center"]
    trajectories = {}
    timing = {}
    for method in METHODS:
        t = time.perf_counter()
        if method == "stationary_pixel":
            trace = [center[:] for _ in frames]
        else:
            trace = klt_track(frames, mask, center, use_fb=method == "klt_forward_backward")
        trajectories[method] = trace
        timing[method] = round((time.perf_counter() - t)*1000, 3)
    camera = affine_background_delta(frames, mask, center)
    # Only display camera-relativity for the existing KLT endpoint estimate.
    last = trajectories["klt_forward_backward"][-1]
    if camera["accepted"] and last is not None:
        camera["klt_relative_background_delta_xy"] = [
            round(last[i] - camera["projected_background_xy"][i], 3) for i in (0, 1)
        ]
    else:
        camera["klt_relative_background_delta_xy"] = None
    return {"name": row["name"], "nframes": len(frames),
            "trajectories": trajectories, "cpu_total_ms": round((time.perf_counter()-start)*1000,3),
            "cpu_method_ms": timing, "background_2d_diagnostic": camera}


def predict(cfg: dict, split: str):
    dataset = ROOT / cfg["dataset_dir"]
    manifest = data.check(cfg, include_future_masks_and_gold=False)
    manifest_sha = data.sha256((dataset / "manifest.json").read_bytes())
    dest = ROOT / cfg["output_dir"] / (split + "_predictions.json")
    if dest.exists():
        old = document(dest)
        if old.get("manifest_sha256") != manifest_sha or old.get("methods") != list(METHODS):
            raise RuntimeError("PREEXISTING_PREDICTIONS_SOURCE_DRIFT")
        pin_prediction_bytes(dest)
        print("DAVIS_PREDICTION_ALREADY_FROZEN", split, flush=True)
        return
    queries = document(dataset / "initial_queries.json")
    # This function deliberately does not read gold-private.json or future PNG masks.
    rows = [r for r in manifest["rows"] if r["split"] == split]
    predictions = [scene_predict(dataset, r, queries[r["name"]]) for r in rows]
    write_once(dest, {"manifest_sha256": manifest_sha, "split": split,
                      "methods": list(METHODS), "no_future_gold_in_inference": True,
                      "nscenes": len(rows), "records": predictions})
    pin_prediction_bytes(dest)
    print("DAVIS_FROZEN_REAL_PREDICTIONS", split, len(predictions), flush=True)


def summarize(gold: dict, rows: list, records: dict, method: str) -> dict:
    total_visible = 0
    total_invisible = 0
    accepted_visible = 0
    accepted_absent = 0
    error_pixels = []
    right = {t: 0 for t in SCORE_THRESHOLDS}
    for row in rows:
        scene = row["name"]
        truth = gold[scene]["points"]
        pred = records[scene]["trajectories"][method]
        if len(pred) != len(truth):
            raise RuntimeError("FROZEN_TRACK_FRAME_COUNT_DRIFT")
        for item, xy in zip(truth[1:], pred[1:]):
            if item["visible"]:
                total_visible += 1
                if xy is None:
                    continue
                accepted_visible += 1
                err = float(np.linalg.norm(np.subtract(xy, item["center"])))
                error_pixels.append(err)
                for t in SCORE_THRESHOLDS:
                    right[t] += int(err <= t)
            else:
                total_invisible += 1
                accepted_absent += int(xy is not None)
    accepted_all = accepted_visible + accepted_absent
    return {
        "scenes": len(rows), "visible_evaluations": total_visible,
        "invisible_evaluations": total_invisible,
        "visible_accepted": accepted_visible, "visible_abstained": total_visible - accepted_visible,
        "absent_false_accepted": accepted_absent,
        "accepted_overall": accepted_all,
        "accepted_precision_at_8px": round(right[8] / accepted_all, 5) if accepted_all else None,
        "visible_pck4_unconditional": round(right[4] / total_visible, 5) if total_visible else None,
        "visible_pck8_unconditional": round(right[8] / total_visible, 5) if total_visible else None,
        "visible_pck16_unconditional": round(right[16] / total_visible, 5) if total_visible else None,
        "pck8_conditional_on_visible_prediction": round(right[8] / accepted_visible, 5)
        if accepted_visible else None,
        "pixel_error_p50_on_accepted_visible": round(float(np.percentile(error_pixels,50)),2)
        if error_pixels else None,
        "pixel_error_p95_on_accepted_visible": round(float(np.percentile(error_pixels,95)),2)
        if error_pixels else None,
        "cpu_scene_p50_ms": round(float(np.median(
            [records[r["name"]]["cpu_method_ms"][method] for r in rows])), 2),
        "cpu_scene_p95_ms": round(float(np.percentile(
            [records[r["name"]]["cpu_method_ms"][method] for r in rows],95)), 2),
    }


def scene_bootstrap(rows, gold, records, *, seed=20261101, n=600):
    rng = np.random.default_rng(seed)
    # CI clusters by independent natural video (NOT individual frames).
    diffs = []
    for _ in range(n):
        sampled = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        # Repeated scene values must still count as independent cluster draws;
        # summarize loops over rows, not a unique scene dictionary.
        result = {
            m: summarize(gold, sampled, records, m)["visible_pck8_unconditional"]
            for m in ("stationary_pixel", "klt_forward_backward")
        }
        if all(v is not None for v in result.values()):
            diffs.append((result["klt_forward_backward"]-result["stationary_pixel"])*100)
    return [round(float(np.percentile(diffs,p)),2) for p in (2.5,97.5)] if diffs else None


def score(cfg: dict, split: str):
    dataset = ROOT / cfg["dataset_dir"]
    manifest = data.check(cfg)
    rows = [r for r in manifest["rows"] if r["split"] == split]
    pred_path = ROOT / cfg["output_dir"] / (split + "_predictions.json")
    require_pinned_prediction(pred_path)
    pred = document(pred_path)
    if (pred.get("manifest_sha256") != data.sha256((dataset / "manifest.json").read_bytes())
            or pred.get("split") != split or pred.get("methods") != list(METHODS)
            or pred.get("no_future_gold_in_inference") is not True
            or pred.get("nscenes") != len(rows)):
        raise RuntimeError("SCORING_REJECTED_UNFROZEN_PREDICTIONS")
    results = {r["name"]: r for r in pred["records"]}
    if len(pred["records"]) != len(rows) or set(results) != {r["name"] for r in rows}:
        raise RuntimeError("SCORING_REJECTED_INCOMPLETE_REAL_VIDEO_PREDICTIONS")
    gold = document(dataset / "gold-private.json")
    metrics = {method: summarize(gold, rows, results, method) for method in METHODS}
    diagnostics = [results[r["name"]]["background_2d_diagnostic"] for r in rows]
    out = {
        "dataset": "official_davis_2017_real_camera_video_frames",
        "split": split, "source_manifest_sha256": pred["manifest_sha256"],
        "predictions_sha256": data.sha256(pred_path.read_bytes()),
        "real_videos": True, "nscenes": len(rows), "nframes_per_scene": cfg["source_frames"],
        "no_world_relative_motion_ground_truth": True,
        "task": "initial-instance-mask centroid tracking in IMAGE coordinates; independent annotations for future frames",
        "methods": metrics, "background_2d_affine_registration_accepted":
            sum(v["accepted"] for v in diagnostics),
        "cluster_bootstrap_klt_fb_minus_static_pck8_pp_ci95":
            scene_bootstrap(rows, gold, results),
        "notes": ["No VLM baseline: point tracking with an initial mask is not comparable to multiple-choice direction QA.",
                  "DAVIS scene may overlap pretrained model training; no model-novelty claim.",
                  "A dominant global affine is NOT physical camera or 3D motion compensation.",
                  "The test dataset is never used for threshold or model choice."],
    }
    write_once(ROOT / cfg["output_dir"] / (split + "_results.json"), out)
    print("DAVIS_REAL_SCORED",split,json.dumps({
        "scenes":len(rows),"methods":{m: {k:v for k,v in metrics[m].items()
          if k in ("visible_pck8_unconditional","visible_accepted","visible_abstained",
                   "absent_false_accepted","cpu_scene_p50_ms")}
          for m in METHODS},
        "bootstrap_ci":out["cluster_bootstrap_klt_fb_minus_static_pck8_pp_ci95"],
        "background_affine_accepted":out["background_2d_affine_registration_accepted"],
    },sort_keys=True),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("predict-dev", "score-dev", "predict-test", "score-test"))
    p.add_argument("--config", default="configs/davis_real_v1.json")
    args = p.parse_args()
    cfg = document(ROOT / args.config)
    if args.action.startswith("predict-"):
        predict(cfg,args.action.split("-")[1])
    else:
        score(cfg,args.action.split("-")[1])


if __name__ == "__main__":
    main()
