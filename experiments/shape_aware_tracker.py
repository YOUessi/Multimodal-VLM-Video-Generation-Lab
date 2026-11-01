"""Question-grounded color AND contour-shape target identity with conservative camera gating.

Only video pixels, query text, and declared generic thresholds are used.
The tracker does not import the video generator or read the manifest.
Ambiguity (two same-shape same-color candidates), partial occlusion,
or weak parallax-consensus must lead to abstention, not a guessed identity.
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from affine_camera_tracker import estimate_background_affine, corrected_delta
from appearance_hsv_tracker import query_target, threshold_mask
from target_motion_tracking import choose_frames, infer_direction

SHAPES = ("circle", "square", "triangle")


def question_shape(question):
    found = [shape for shape in SHAPES
             if re.search(r"\b" + shape + r"\b", question.lower())]
    if len(found) != 1:
        raise ValueError("UNSUPPORTED_OR_AMBIGUOUS_QUERY_SHAPE")
    return found[0]


def contour_shape(contour, epsilon_fraction):
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None
    approx = cv2.approxPolyDP(contour, epsilon_fraction * perimeter, True)
    vertices = len(approx)
    if vertices == 3:
        return "triangle"
    if vertices == 4:
        return "square"
    if vertices >= 6:
        return "circle"
    return None


def semantic_candidates(frame, color, shape, cfg):
    mask = threshold_mask(frame, color, cfg)
    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    lo = max(cfg["min_component_pixels"],
             int(width*height*cfg["min_component_fraction"]))
    hi = min(cfg["max_component_pixels"],
             int(width*height*cfg["max_component_fraction"]))
    result = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not lo <= area <= hi:
            continue
        shape_guess = contour_shape(contour, cfg["identity_polygon_epsilon"])
        if shape_guess != shape:
            continue
        moment = cv2.moments(contour)
        if moment["m00"] <= 0:
            continue
        x = moment["m10"] / moment["m00"]
        y = moment["m01"] / moment["m00"]
        if not (cfg["roi_min_x"] * width <= x <= cfg["roi_max_x"]*width
                and cfg["roi_min_y"]*height <= y <= cfg["roi_max_y"]*height):
            continue
        result.append({"xy": [round(x, 3), round(y, 3)],
                       "area": round(float(area), 3),
                       "shape": shape_guess})
    result.sort(key=lambda z: z["xy"][0])
    return result


def video_shape_trace(video, query, cfg):
    color = query_target(query)
    shape = question_shape(query)
    capture = cv2.VideoCapture(str(Path(video)))
    if not capture.isOpened():
        raise RuntimeError("IDENTITY_VIDEO_UNREADABLE")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 3:
        capture.release()
        raise RuntimeError("IDENTITY_VIDEO_TOO_SHORT")
    indices = choose_frames(total, 3)
    points, reasons = [], []
    previous = None
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError("IDENTITY_FRAME_DECODE_FAILURE")
            options = semantic_candidates(bgr, color, shape, cfg)
            if len(options) == 0:
                points.append(None)
                reasons.append("NO_MATCHING_COLOR_SHAPE")
                continue
            if len(options) > 1:
                # Identical-looking distractors have no unique identity;
                # refusing to guess is more defensible than taking biggest.
                points.append(None)
                reasons.append("SAME_APPEARANCE_TARGET_IDENTITY_AMBIGUOUS")
                continue
            point = options[0]["xy"]
            if previous is not None and np.linalg.norm(
                np.subtract(point, previous)
            ) > cfg["max_step_displacement_px"]:
                points.append(None)
                reasons.append("IMPLAUSIBLE_TARGET_JUMP")
                continue
            points.append(point)
            previous = point
            reasons.append("UNIQUE_COLOR_SHAPE")
    finally:
        capture.release()
    missing = sum(x is None for x in points)
    if missing:
        return {"points": points, "delta_xy": None,
                "missing": missing, "reason": ";".join(reasons),
                "target_color": color, "target_shape": shape}
    delta = [round(points[-1][i] - points[0][i], 3) for i in (0, 1)]
    return {"points": points, "delta_xy": delta,
            "missing": 0, "reason": "SHAPE_IDENTITY_OK",
            "target_color": color, "target_shape": shape}


def tracker(path, query, cfg):
    trace = video_shape_trace(path, query, cfg)
    out = {
        "camera_compensated": True,
        "registration_method": "shape_affine_parallax_abstain",
        "target_color": trace["target_color"],
        "target_shape": trace["target_shape"],
        "target_points": trace["points"],
        "screen_delta_xy": trace["delta_xy"],
        "delta_xy": trace["delta_xy"],
        "missing": trace["missing"],
        "direction": None,
        "reason": trace["reason"],
        "registration": None,
    }
    if trace["delta_xy"] is None:
        return out
    geom = estimate_background_affine(path, cfg)
    out["registration"] = geom
    if not geom["accepted"]:
        out["delta_xy"] = None
        out["reason"] = "GEOMETRY_ABSTAIN:" + geom["reason"]
        return out
    if geom["inlier_fraction"] < cfg["identity_min_background_inlier_fraction"]:
        out["delta_xy"] = None
        out["reason"] = "DEPTH_PARALLAX_OR_MIXED_MOTION_ABSTAIN"
        return out
    delta = corrected_delta(trace["points"], geom["matrix"])
    out["delta_xy"] = delta
    out["direction"] = infer_direction({"delta_xy": delta},
                                       cfg["direction_deadzone_px"])
    out["reason"] = ("SHAPE_GEOMETRY_TRACK_OK" if out["direction"] is not None
                     else "WORLD_DIRECTION_UNCERTAIN")
    return out
