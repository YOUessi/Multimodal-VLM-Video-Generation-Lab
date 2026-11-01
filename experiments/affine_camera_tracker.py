"""Robust global affine registration from VIDEO PIXELS ONLY.

The new estimator compensates 2D camera rotation/zoom/translation under
a dominant stationary textured plane. No scene-generation parameters or
ground-truth labels ever enter the tracker. Rejects weak RANSAC geometry.
"""
from __future__ import annotations
import math
from pathlib import Path
import cv2
import numpy as np
from domain_shift_camera_tracker import grayscale_features, tracker as frozen_tracker, gated_label
from target_motion_tracking import infer_direction


def estimate_background_affine(path, cfg):
    (first, last), indices = grayscale_features(Path(path))
    a = np.clip(first, 0, 255).astype(np.uint8)
    b = np.clip(last, 0, 255).astype(np.uint8)
    orb = cv2.ORB_create(nfeatures=int(cfg["orb_features"]), fastThreshold=8)
    kp0, des0 = orb.detectAndCompute(a, None)
    kp1, des1 = orb.detectAndCompute(b, None)
    record = {"accepted": False, "frame_indices": [indices[0], indices[-1]],
              "method": "ORB ratio matching + RANSAC partial affine from video only",
              "keypoints": [len(kp0), len(kp1)], "matches": 0,
              "inliers": 0, "inlier_fraction": 0.0, "median_error": None,
              "matrix": None, "reason": "INSUFFICIENT_FEATURES"}
    if des0 is None or des1 is None or len(kp0) < 5 or len(kp1) < 5:
        return record
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(des0, des1, k=2)
    good = [m for pair in pairs if len(pair) == 2
            for m, n in [pair] if m.distance < cfg["orb_ratio"] * n.distance]
    record["matches"] = len(good)
    if len(good) < cfg["affine_min_matches"]:
        record["reason"] = "TOO_FEW_MATCHES"
        return record
    points0 = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    points1 = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, mask = cv2.estimateAffinePartial2D(
        points0, points1, method=cv2.RANSAC,
        ransacReprojThreshold=float(cfg["affine_ransac_px"]),
        maxIters=2500, confidence=0.995, refineIters=10,
    )
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        record["reason"] = "RANSAC_INVALID"
        return record
    keep = mask.flatten().astype(bool)
    count = int(keep.sum())
    ratio = count / len(good)
    record["inliers"] = count
    record["inlier_fraction"] = round(ratio, 4)
    predicted = cv2.transform(points0[keep], matrix).reshape(-1, 2)
    actual = points1[keep].reshape(-1, 2)
    residual = float(np.median(np.linalg.norm(predicted - actual, axis=1))) if count else float("inf")
    record["median_error"] = round(residual, 4)
    scale = math.hypot(matrix[0, 0], matrix[1, 0])
    angle = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    record["estimated_scale"] = round(scale, 5)
    record["estimated_angle_deg"] = round(angle, 3)
    record["matrix"] = matrix.round(7).tolist()
    if count < cfg["affine_min_inliers"] or ratio < cfg["affine_min_inlier_fraction"]:
        record["reason"] = "WEAK_RANSAC_INLIERS"
    elif residual > cfg["affine_max_median_error_px"]:
        record["reason"] = "HIGH_REPROJECTION_ERROR"
    elif not (cfg["affine_scale_min"] <= scale <= cfg["affine_scale_max"]):
        record["reason"] = "IMPLAUSIBLE_CAMERA_SCALE"
    elif abs(angle) > cfg["affine_angle_max_deg"]:
        record["reason"] = "IMPLAUSIBLE_CAMERA_ROTATION"
    elif math.hypot(matrix[0, 2], matrix[1, 2]) > cfg["affine_max_translation_px"]:
        record["reason"] = "IMPLAUSIBLE_CAMERA_TRANSLATION"
    else:
        record["accepted"] = True
        record["reason"] = "GEOMETRY_ACCEPTED"
    return record


def corrected_delta(points, matrix):
    """Map final target screen coordinate back to the initial camera plane."""
    if points is None or len(points) < 2 or points[0] is None or points[-1] is None:
        return None
    inv = cv2.invertAffineTransform(np.asarray(matrix, dtype=np.float64))
    last = np.array([points[-1][0], points[-1][1], 1.0])
    aligned = inv @ last
    return [round(float(aligned[i] - points[0][i]), 3) for i in (0, 1)]


def tracker(path, question, cfg, method="plain"):
    if method not in ("plain", "phase", "affine"):
        raise ValueError("Unknown geometric estimator")
    if method != "affine":
        base = frozen_tracker(path, question, cfg, stable=(method == "phase"))
        base["registration_method"] = method
        return base
    base = frozen_tracker(path, question, cfg, stable=False)
    base["registration_method"] = "affine"
    base["camera_compensated"] = True
    if base["delta_xy"] is None:
        return base
    registration = estimate_background_affine(path, cfg)
    base["registration"] = registration
    if not registration["accepted"]:
        base["direction"] = None
        base["delta_xy"] = None
        base["reason"] = "AFFINE_REGISTRATION_ABSTAIN:" + registration["reason"]
        return base
    base["delta_xy"] = corrected_delta(base["target_points"], registration["matrix"])
    base["direction"] = infer_direction(
        {"delta_xy": base["delta_xy"]}, cfg["direction_deadzone_px"]
    )
    base["reason"] = "AFFINE_TRACK_OK" if base["direction"] else "AFFINE_DIRECTION_AMBIGUOUS"
    return base
