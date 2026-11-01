"""Batch 39 CPU point tracking and image-only camera-motion diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

import tapvidmv_data as src

ROOT = Path(__file__).resolve().parents[1]
DATA = src.FROZEN
OUT = ROOT / "outputs/tapvidmv_droid_batch39c"
METHODS = ("stationary_point", "klt_forward", "klt_fb", "template_recover4")
FB_MAX_ERROR = 1.5
TEMPLATE_SIZE = 21
TEMPLATE_RADIUS = 32
TEMPLATE_NCC = 0.75
TEMPLATE_MAX_MISSES = 4
ORB_FEATURES = 1000
ORB_RATIO = 0.75
ORB_RANSAC_PX = 3.0
STATUS = src.STATUS


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def doc(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def gitsha() -> str:
    status = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True, text=True, timeout=10,
    )
    if status.returncode:
        raise RuntimeError("SOURCE_STATUS_CHECK_FAILED")
    if status.stdout.strip():
        raise RuntimeError("DIRTY_SOURCE_TREE_CANNOT_PIN_PREDICTIONS")
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, timeout=10
    ).strip()


def once(path: Path, payload: dict) -> None:
    src.write_once(path, src.json_bytes(payload))


def pin(path: Path) -> None:
    src.write_once(path.with_suffix(path.suffix + ".sha256"),
                   (sha_bytes(path.read_bytes()) + "\n").encode())


def require_pin(path: Path) -> None:
    pin_path = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not pin_path.is_file():
        raise RuntimeError("MISSING_TAPVIDMV_PREDICTION_PIN")
    if pin_path.read_text().strip() != sha_bytes(path.read_bytes()):
        raise RuntimeError("TAPVIDMV_PREDICTION_CHANGED_AFTER_SEAL")


def rgb_to_gray(frames: np.ndarray) -> list[np.ndarray]:
    if frames.ndim != 4 or frames.dtype != np.uint8 or frames.shape[-1] != 3:
        raise ValueError("INVALID_PUBLIC_RGB_FRAMES")
    return [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]


def stationary_trace(nframes: int, query_xy: list[float]) -> list:
    point = [float(query_xy[0]), float(query_xy[1])]
    return [point.copy() for _ in range(nframes)]


def klt_trace(gray: list[np.ndarray], query_xy: list[float], *, fb: bool) -> list:
    height, width = gray[0].shape
    point = np.asarray(query_xy, dtype=np.float32).reshape(1, 1, 2)
    trace = [None] * len(gray)
    trace[0] = [float(point[0, 0, 0]), float(point[0, 0, 1])]
    for frame_id in range(1, len(gray)):
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            gray[frame_id - 1], gray[frame_id], point, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if nxt is None or status is None or int(status[0, 0]) != 1:
            break
        xy = nxt.reshape(2)
        if (not np.isfinite(xy).all() or xy[0] < 0 or xy[0] >= width
                or xy[1] < 0 or xy[1] >= height):
            break
        if fb:
            back, back_status, _ = cv2.calcOpticalFlowPyrLK(
                gray[frame_id], gray[frame_id - 1], nxt, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            if (back is None or back_status is None or int(back_status[0, 0]) != 1
                    or not np.isfinite(back).all()
                    or float(np.linalg.norm(back - point)) > FB_MAX_ERROR):
                break
        trace[frame_id] = [round(float(x), 3) for x in xy]
        point = nxt
    return trace


def padded_patch(gray: np.ndarray, xy: np.ndarray, half: int) -> np.ndarray:
    pad = half + 3
    image = cv2.copyMakeBorder(
        gray, pad, pad, pad, pad, cv2.BORDER_REFLECT_101
    )
    x = int(round(float(xy[0]))) + pad
    y = int(round(float(xy[1]))) + pad
    return image[y-half:y+half+1, x-half:x+half+1]


def template_trace(gray: list[np.ndarray], query_xy: list[float]) -> list:
    height, width = gray[0].shape
    anchor = np.asarray(query_xy, dtype=np.float64)
    trace = [None] * len(gray)
    trace[0] = anchor.tolist()
    half = TEMPLATE_SIZE // 2
    template = padded_patch(gray[0], anchor, half)
    if template.shape != (TEMPLATE_SIZE, TEMPLATE_SIZE) or np.std(template) < 8:
        return trace
    last = anchor.copy()
    misses = 0
    for frame_id in range(1, len(gray)):
        if misses >= TEMPLATE_MAX_MISSES:
            break
        pad = TEMPLATE_RADIUS + half + 3
        image = cv2.copyMakeBorder(
            gray[frame_id], pad, pad, pad, pad, cv2.BORDER_REFLECT_101
        )
        center_x = int(round(float(last[0]))) + pad
        center_y = int(round(float(last[1]))) + pad
        roi = image[
            center_y-TEMPLATE_RADIUS-half:center_y+TEMPLATE_RADIUS+half+1,
            center_x-TEMPLATE_RADIUS-half:center_x+TEMPLATE_RADIUS+half+1,
        ]
        if roi.shape[0] < TEMPLATE_SIZE or roi.shape[1] < TEMPLATE_SIZE:
            misses += 1
            continue
        scores = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, top_left = cv2.minMaxLoc(scores)
        found = last + np.asarray(
            [top_left[0] - TEMPLATE_RADIUS, top_left[1] - TEMPLATE_RADIUS],
            dtype=np.float64,
        )
        if (not math.isfinite(score) or score < TEMPLATE_NCC
                or not (0 <= found[0] < width and 0 <= found[1] < height)):
            misses += 1
            continue
        last = found
        misses = 0
        trace[frame_id] = [round(float(x), 3) for x in found]
    return trace


def estimate_affine_sequence(gray: list[np.ndarray]) -> tuple[list, list]:
    """Estimate frame-0 to frame-t background affine from RGB only."""
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    key0, desc0 = orb.detectAndCompute(gray[0], None)
    matrices = [np.eye(2, 3, dtype=np.float64)]
    metadata = [{"valid": True, "matches": 0, "inliers": 0, "cpu_ms": 0.0}]
    if desc0 is None or len(key0) < 8:
        return [None] + [None] * (len(gray) - 1), metadata + [
            {"valid": False, "matches": 0, "inliers": 0, "cpu_ms": 0.0}
            for _ in range(len(gray) - 1)
        ]
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    xy_predictions = [[None] for _ in gray]
    matrices = [np.eye(2, 3, dtype=np.float64)]
    metadata = [{"valid": True, "matches": 0, "inliers": 0, "cpu_ms": 0.0}]
    for frame_id in range(1, len(gray)):
        started = time.perf_counter()
        key_t, desc_t = orb.detectAndCompute(gray[frame_id], None)
        matrix = None
        matches_count = 0
        inliers_count = 0
        if desc_t is not None and len(key_t) >= 8:
            pairs = matcher.knnMatch(desc0, desc_t, k=2)
            good = [m for pair in pairs if len(pair) == 2
                    for m, n in [pair] if m.distance < ORB_RATIO * n.distance]
            matches_count = len(good)
            if matches_count >= 8:
                source_points = np.float32([key0[m.queryIdx].pt for m in good])
                target_points = np.float32([key_t[m.trainIdx].pt for m in good])
                matrix, inlier_mask = cv2.estimateAffinePartial2D(
                    source_points, target_points, method=cv2.RANSAC,
                    ransacReprojThreshold=ORB_RANSAC_PX,
                    maxIters=2000, confidence=0.99, refineIters=10,
                )
                if matrix is not None and inlier_mask is not None:
                    inliers_count = int(np.asarray(inlier_mask).sum())
                    matrix = np.asarray(matrix, dtype=np.float64)
        elapsed = round(1000 * (time.perf_counter() - started), 3)
        matrices.append(matrix)
        metadata.append({
            "valid": matrix is not None,
            "matches": matches_count,
            "inliers": inliers_count,
            "cpu_ms": elapsed,
        })
    return matrices, metadata


def apply_affine(matrix, xy: list[float]):
    if matrix is None:
        return None
    point = np.asarray([float(xy[0]), float(xy[1]), 1.0], dtype=np.float64)
    result = np.asarray(matrix, dtype=np.float64) @ point
    if not np.isfinite(result).all():
        return None
    return [round(float(result[0]), 3), round(float(result[1]), 3)]


def predict(split: str) -> None:
    manifest = src.audit_public()
    queries = doc(DATA / "queries-public.json")
    manifest_sha = sha_bytes((DATA / "manifest.json").read_bytes())
    errata_sha = sha_bytes((DATA / "manifest-errata.json").read_bytes())
    public_sha = sha_bytes((DATA / "queries-public.json").read_bytes())
    OUT.mkdir(parents=True, exist_ok=True)
    output_path = OUT / f"{split}_predictions.json"
    if output_path.exists():
        require_pin(output_path)
        saved = doc(output_path)
        if (saved.get("source_commit") != gitsha()
                or saved.get("manifest_sha256") != manifest_sha
                or saved.get("manifest_errata_sha256") != errata_sha
                or saved.get("public_query_sha256") != public_sha):
            raise RuntimeError("TAPVIDMV_STALE_PINNED_PREDICTION")
        print("TAPVIDMV_EXISTING_PIN_OK", split, flush=True)
        return
    records = []
    for scene in manifest["scenes"]:
        if scene["split"] != split:
            continue
        scene_queries = queries[scene["name"]]
        camera_cache = {}
        for camera_id in range(3):
            relative = scene["frame_files"][str(camera_id)]
            frames = np.load(DATA / relative, mmap_mode="r", allow_pickle=False)
            gray = rgb_to_gray(frames)
            matrices, affine_meta = estimate_affine_sequence(gray)
            camera_cache[camera_id] = (gray, matrices, affine_meta)
        for query in scene_queries:
            camera_id = int(query["camera_id"])
            gray, matrices, affine_meta = camera_cache[camera_id]
            xy = query["query_xy"]
            traces = {}
            cpu_query_ms = {}
            for method in METHODS:
                started = time.perf_counter()
                if method == "stationary_point":
                    trace = stationary_trace(len(gray), xy)
                elif method == "klt_forward":
                    trace = klt_trace(gray, xy, fb=False)
                elif method == "klt_fb":
                    trace = klt_trace(gray, xy, fb=True)
                else:
                    trace = template_trace(gray, xy)
                cpu_query_ms[method] = round(1000 * (time.perf_counter() - started), 3)
                if len(trace) != len(gray):
                    raise RuntimeError("TAPVIDMV_TRACE_LENGTH_MISMATCH")
                traces[method] = trace
            affine_xy = [apply_affine(matrix, xy) for matrix in matrices]
            records.append({
                "name": scene["name"],
                "query_id": query["id"],
                "camera_id": camera_id,
                "target_frame": query["target_frame"],
                "traces": traces,
                "cpu_query_ms": cpu_query_ms,
                "camera_affine_xy": affine_xy,
                "camera_affine_meta": affine_meta,
            })
        print("TAPVIDMV_CPU_PREDICTED", split, scene["name"],
              len(scene_queries), "queries", flush=True)
    payload = {
        "dataset": "TAPVid-MV DROID public Rerun viewer subset",
        "split": split,
        "source_commit": gitsha(),
        "manifest_sha256": manifest_sha,
        "manifest_errata_sha256": errata_sha,
        "public_query_sha256": public_sha,
        "oracle_unread_during_prediction": True,
        "methods": list(METHODS),
        "camera_motion_estimator": {
            "method": "ORB + BFMatcher ratio test + RANSAC partial affine",
            "nfeatures": ORB_FEATURES,
            "ratio": ORB_RATIO,
            "ransac_reprojection_px": ORB_RANSAC_PX,
            "frame_pair": "query frame 0 to each target frame independently",
        },
        "records": records,
    }
    once(output_path, payload)
    pin(output_path)
    print("TAPVIDMV_CPU_PREDICTIONS_PINNED", split, len(records),
          sha_bytes(output_path.read_bytes()), flush=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float | None:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if not math.isfinite(na + nb) or na < 1e-4 or nb < 1e-4:
        return None
    return float(np.dot(a, b) / (na * nb))


def summarize(values: list[float]) -> dict:
    clean = np.asarray([x for x in values if math.isfinite(float(x))], dtype=np.float64)
    if not len(clean):
        return {"n": 0, "p50": None, "p90": None, "mean": None}
    return {
        "n": int(len(clean)),
        "p50": round(float(np.percentile(clean, 50)), 4),
        "p90": round(float(np.percentile(clean, 90)), 4),
        "mean": round(float(np.mean(clean)), 4),
    }


def bootstrap_scene_ci(scene_values: dict[str, float], seed: int = 39) -> dict:
    values = np.asarray([v for v in scene_values.values() if v is not None], dtype=np.float64)
    if not len(values):
        return {"n_scenes": 0, "low95": None, "high95": None}
    if len(values) == 1:
        return {"n_scenes": 1, "low95": float(values[0]), "high95": float(values[0])}
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return {
        "n_scenes": int(len(values)),
        "low95": round(float(np.percentile(samples, 2.5)), 5),
        "high95": round(float(np.percentile(samples, 97.5)), 5),
    }


def score(split: str) -> None:
    manifest = src.audit()
    query_doc = doc(DATA / "queries-public.json")
    prediction_path = OUT / f"{split}_predictions.json"
    require_pin(prediction_path)
    saved = doc(prediction_path)
    manifest_sha = sha_bytes((DATA / "manifest.json").read_bytes())
    errata_sha = sha_bytes((DATA / "manifest-errata.json").read_bytes())
    public_sha = sha_bytes((DATA / "queries-public.json").read_bytes())
    if (saved.get("source_commit") != gitsha()
            or saved.get("manifest_sha256") != manifest_sha
            or saved.get("manifest_errata_sha256") != errata_sha
            or saved.get("public_query_sha256") != public_sha
            or saved.get("oracle_unread_during_prediction") is not True
            or saved.get("methods") != list(METHODS)):
        raise RuntimeError("TAPVIDMV_SCORE_SOURCE_ORACLE_FIREWALL_FAILED")
    scenes = [x for x in manifest["scenes"] if x["split"] == split]
    scene_by_name = {x["name"]: x for x in scenes}
    prediction_by_key = {(x["name"], x["query_id"]): x for x in saved["records"]}
    expected_keys = {
        (scene["name"], q["id"])
        for scene in scenes for q in query_doc[scene["name"]]
    }
    if set(prediction_by_key) != expected_keys:
        raise RuntimeError("TAPVIDMV_PREDICTION_QUERY_SET_MISMATCH")
    per_scene = {}
    method_scene_pck = {method: {} for method in METHODS}
    method_totals = {
        method: {
            "all_inframe": 0, "all_inframe_correct8": 0, "all_inframe_accepted": 0,
            "visible": 0, "visible_correct8": 0, "visible_accepted": 0,
            "occluded": 0, "occluded_false_accept": 0, "occluded_abstain": 0,
            "out": 0, "out_false_accept": 0, "unknown": 0, "missing": 0,
        } for method in METHODS
    }
    cpu_per_frame = {method: [] for method in METHODS}
    affine_errors = []
    affine_valid = 0
    affine_total = 0
    screen_world_cosines = []
    affine_world_cosines = []
    oracle_world_cosines = []
    camera_dominance = []
    camera_translation = []
    camera_rotation = []
    per_camera = {str(i): {"queries": 0, "translation_m": [], "rotation_deg": []}
                  for i in range(3)}
    status_names = {value: key for key, value in STATUS.items()}
    for scene in scenes:
        oracle_path = DATA / scene["oracle_file"]
        with np.load(oracle_path, allow_pickle=False) as oracle:
            qids = [str(x) for x in oracle["qids"].tolist()]
            q_index = {qid: i for i, qid in enumerate(qids)}
            scene_rows = [r for r in saved["records"] if r["name"] == scene["name"]]
            scene_metrics = {method: {
                "all_inframe": 0, "all_inframe_correct8": 0, "all_inframe_accepted": 0,
                "visible": 0, "visible_correct8": 0, "visible_accepted": 0,
                "occluded": 0, "occluded_false_accept": 0, "occluded_abstain": 0,
                "out": 0, "out_false_accept": 0, "unknown": 0, "missing": 0,
                "cpu_ms_per_target_frame": [],
            } for method in METHODS}
            for row in scene_rows:
                qid = row["query_id"]
                if qid not in q_index:
                    raise RuntimeError("TAPVIDMV_ORACLE_QUERY_ID_MISSING")
                qi = q_index[qid]
                query = next(q for q in query_doc[scene["name"]] if q["id"] == qid)
                gt = np.asarray(oracle["gt_xy"][qi], dtype=np.float64)
                present = np.asarray(oracle["track_present"][qi], dtype=bool)
                in_frame = np.asarray(oracle["in_frame"][qi], dtype=bool)
                depth = np.asarray(oracle["depth_status"][qi], dtype=np.uint8)
                camera_only = np.asarray(oracle["camera_only_xy"][qi], dtype=np.float64)
                world_only = np.asarray(oracle["world_only_xy"][qi], dtype=np.float64)
                world_xyz = np.asarray(oracle["world_xyz"][qi], dtype=np.float64)
                cam_translation = np.asarray(oracle["camera_translation_m"][qi], dtype=np.float64)
                cam_rotation = np.asarray(oracle["camera_rotation_deg"][qi], dtype=np.float64)
                camera_id = int(query["camera_id"])
                per_camera[str(camera_id)]["queries"] += 1
                camera_translation.append(float(cam_translation[-1]))
                camera_rotation.append(float(cam_rotation[-1]))
                per_camera[str(camera_id)]["translation_m"].append(float(cam_translation[-1]))
                per_camera[str(camera_id)]["rotation_deg"].append(float(cam_rotation[-1]))
                qxy = np.asarray(query["query_xy"], dtype=np.float64)
                nframes = scene["nframes"]
                if (gt.shape != (nframes, 2) or present.shape != (nframes,)
                        or in_frame.shape != (nframes,) or depth.shape != (nframes,)):
                    raise RuntimeError("TAPVIDMV_ORACLE_TRACE_SHAPE_MISMATCH")
                affine_xy = row["camera_affine_xy"]
                affine_meta = row["camera_affine_meta"]
                if len(affine_xy) != nframes or len(affine_meta) != nframes:
                    raise RuntimeError("TAPVIDMV_AFFINE_TRACE_SHAPE_MISMATCH")
                for frame_id in range(1, nframes):
                    if present[frame_id] and in_frame[frame_id] and np.isfinite(gt[frame_id]).all():
                        screen_delta = gt[frame_id] - qxy
                        if np.isfinite(world_only[frame_id]).all():
                            world_delta = world_only[frame_id] - qxy
                            value = cosine_similarity(screen_delta, world_delta)
                            if value is not None:
                                screen_world_cosines.append(value)
                            if np.isfinite(camera_only[frame_id]).all():
                                camera_delta = camera_only[frame_id] - qxy
                                value = cosine_similarity(screen_delta - camera_delta, world_delta)
                                if value is not None:
                                    oracle_world_cosines.append(value)
                                denom = (np.linalg.norm(camera_delta)
                                         + np.linalg.norm(world_delta) + 1e-6)
                                camera_dominance.append(float(np.linalg.norm(camera_delta) / denom))
                            if affine_xy[frame_id] is not None:
                                affine_delta = np.asarray(affine_xy[frame_id], dtype=np.float64) - qxy
                                value = cosine_similarity(screen_delta - affine_delta, world_delta)
                                if value is not None:
                                    affine_world_cosines.append(value)
                    camera_gt = camera_only[frame_id]
                    affine_pred = affine_xy[frame_id]
                    width, height = scene["resolution_by_camera"][str(camera_id)]
                    if (np.isfinite(camera_gt).all()
                            and 0 <= camera_gt[0] < width
                            and 0 <= camera_gt[1] < height):
                        affine_total += 1
                        if affine_pred is not None:
                            affine_valid += 1
                            affine_errors.append(float(np.linalg.norm(
                                np.asarray(affine_pred, dtype=np.float64) - camera_gt
                            )))
                    status = int(depth[frame_id])
                    for method in METHODS:
                        pred = row["traces"][method][frame_id]
                        metric = scene_metrics[method]
                        if status == STATUS["missing"]:
                            metric["missing"] += 1
                        elif status == STATUS["unknown"]:
                            metric["unknown"] += 1
                        elif status == STATUS["out"]:
                            metric["out"] += 1
                            metric["out_false_accept"] += int(pred is not None)
                        elif status == STATUS["occluded"]:
                            metric["occluded"] += 1
                            metric["occluded_false_accept"] += int(pred is not None)
                            metric["occluded_abstain"] += int(pred is None)
                        elif status == STATUS["visible"]:
                            metric["visible"] += 1
                            metric["visible_accepted"] += int(pred is not None)
                            if pred is not None:
                                error = float(np.linalg.norm(
                                    np.asarray(pred, dtype=np.float64) - gt[frame_id]
                                ))
                                metric["visible_correct8"] += int(error <= 8)
                        if present[frame_id] and in_frame[frame_id] and np.isfinite(gt[frame_id]).all():
                            metric["all_inframe"] += 1
                            metric["all_inframe_accepted"] += int(pred is not None)
                            if pred is not None:
                                error = float(np.linalg.norm(
                                    np.asarray(pred, dtype=np.float64) - gt[frame_id]
                                ))
                                metric["all_inframe_correct8"] += int(error <= 8)
                for method in METHODS:
                    duration = float(row["cpu_query_ms"][method])
                    scene_metrics[method]["cpu_ms_per_target_frame"].append(
                        duration / max(1, nframes - 1)
                    )
            for method in METHODS:
                metric = scene_metrics[method]
                for key in ("all_inframe", "all_inframe_correct8", "all_inframe_accepted",
                            "visible", "visible_correct8", "visible_accepted",
                            "occluded", "occluded_false_accept", "occluded_abstain",
                            "out", "out_false_accept", "unknown", "missing"):
                    method_totals[method][key] += metric[key]
                metric["pck8_all_inframe"] = (
                    metric["all_inframe_correct8"] / metric["all_inframe"]
                    if metric["all_inframe"] else None
                )
                metric["pck8_depth_visible"] = (
                    metric["visible_correct8"] / metric["visible"]
                    if metric["visible"] else None
                )
                metric["coverage_depth_visible"] = (
                    metric["visible_accepted"] / metric["visible"]
                    if metric["visible"] else None
                )
                metric["precision8_depth_visible"] = (
                    metric["visible_correct8"] / metric["visible_accepted"]
                    if metric["visible_accepted"] else None
                )
                metric["false_accept_occluded"] = (
                    metric["occluded_false_accept"] / metric["occluded"]
                    if metric["occluded"] else None
                )
                metric["false_accept_out"] = (
                    metric["out_false_accept"] / metric["out"]
                    if metric["out"] else None
                )
                metric["cpu_ms_per_target_frame_p50"] = (
                    float(np.median(metric["cpu_ms_per_target_frame"]))
                    if metric["cpu_ms_per_target_frame"] else None
                )
                method_scene_pck[method][scene["name"]] = metric["pck8_all_inframe"]
            per_scene[scene["name"]] = scene_metrics
    aggregate = {}
    for method in METHODS:
        counts = method_totals[method]
        aggregate[method] = {
            **counts,
            "pck8_all_inframe": (
                counts["all_inframe_correct8"] / counts["all_inframe"]
                if counts["all_inframe"] else None
            ),
            "pck8_depth_visible": (
                counts["visible_correct8"] / counts["visible"]
                if counts["visible"] else None
            ),
            "coverage_depth_visible": (
                counts["visible_accepted"] / counts["visible"]
                if counts["visible"] else None
            ),
            "precision8_depth_visible": (
                counts["visible_correct8"] / counts["visible_accepted"]
                if counts["visible_accepted"] else None
            ),
            "false_accept_occluded": (
                counts["occluded_false_accept"] / counts["occluded"]
                if counts["occluded"] else None
            ),
            "false_accept_out": (
                counts["out_false_accept"] / counts["out"]
                if counts["out"] else None
            ),
            "cpu_ms_per_target_frame_p50": (
                round(float(np.median(cpu_per_frame[method])), 4)
                if cpu_per_frame[method] else None
            ),
            "scene_bootstrap_ci95_pck8_all_inframe": bootstrap_scene_ci(
                method_scene_pck[method]
            ),
        }
    for method in METHODS:
        vals = []
        for scene in per_scene.values():
            for row in scene.values():
                vals.extend(row["cpu_ms_per_target_frame"])
        # Per-method CPU values are collected from the exact same scene/query set.
        cpu_per_frame[method] = [
            x for scene in per_scene.values() for x in scene[method]["cpu_ms_per_target_frame"]
        ]
        aggregate[method]["cpu_ms_per_target_frame_p50"] = (
            round(float(np.median(cpu_per_frame[method])), 4) if cpu_per_frame[method] else None
        )
        aggregate[method]["cpu_ms_per_target_frame_p95"] = (
            round(float(np.percentile(cpu_per_frame[method], 95)), 4) if cpu_per_frame[method] else None
        )
    motion = {
        "screen_vs_world_only_cosine": summarize(screen_world_cosines),
        "image_affine_compensated_vs_world_only_cosine": summarize(affine_world_cosines),
        "oracle_camera_subtracted_vs_world_only_cosine": summarize(oracle_world_cosines),
        "camera_motion_dominance_ratio": summarize(camera_dominance),
        "camera_translation_net_m": summarize(camera_translation),
        "camera_rotation_net_deg": summarize(camera_rotation),
        "orb_affine_static_point_reprojection_error_px": summarize(affine_errors),
        "orb_affine_coverage": round(affine_valid / affine_total, 5) if affine_total else None,
        "orb_affine_valid": affine_valid,
        "orb_affine_evaluable": affine_total,
        "per_camera_pose": {
            camera: {
                "queries": values["queries"],
                "net_translation_m": summarize(values["translation_m"]),
                "net_rotation_deg": summarize(values["rotation_deg"]),
            } for camera, values in per_camera.items()
        },
    }
    result = {
        "dataset": "TAPVid-MV DROID public Rerun viewer subset",
        "source_commit": gitsha(),
        "split": split,
        "scene_count": len(scenes),
        "query_count": len(saved["records"]),
        "manifest_sha256": manifest_sha,
        "manifest_errata_sha256": errata_sha,
        "sealed_prediction_sha256": sha_bytes(prediction_path.read_bytes()),
        "primary_metric": "PCK@8 on projected 3D point tracks; visibility-specific metrics only on exact depth-cloud snapshots",
        "visibility_contract": "depth-cloud-derived visible/occluded labels at exact snapshots; unknown elsewhere",
        "methods": aggregate,
        "camera_motion_decomposition": motion,
        "per_scene": per_scene,
        "caveats": [
            "The Rerun recordings display a subsampled track set; these are not the full metadata-declared tracks.",
            "Per-view visibility is inferred from exact released depth-cloud snapshots, not a separately published binary visibility label.",
            "PCK on all in-frame projections includes positions whose visibility is unknown and is not a visible-only benchmark score.",
            "ORB affine is a background camera-motion diagnostic, not a physical point-identity tracker.",
            "Camera-only and world-only projections are oracle diagnostics and are never inputs to predictors.",
            "The VLM endpoint baseline has a different temporal evidence budget and is reported separately.",
            "Only DEV may be used for method selection; TEST outcomes are descriptive and must not tune thresholds.",
        ],
    }
    output = OUT / f"{split}_results.json"
    once(output, result)
    print("TAPVIDMV_SCORED", split, json.dumps({
        "scenes": len(scenes),
        "queries": len(saved["records"]),
        "methods": {
            method: {
                "pck8_all_inframe": aggregate[method]["pck8_all_inframe"],
                "pck8_depth_visible": aggregate[method]["pck8_depth_visible"],
                "coverage_depth_visible": aggregate[method]["coverage_depth_visible"],
                "false_accept_occluded": aggregate[method]["false_accept_occluded"],
                "false_accept_out": aggregate[method]["false_accept_out"],
                "cpu_ms_p50": aggregate[method]["cpu_ms_per_target_frame_p50"],
            } for method in METHODS
        },
        "motion": {
            "screen_world_cosine": motion["screen_vs_world_only_cosine"],
            "affine_error_px": motion["orb_affine_static_point_reprojection_error_px"],
        },
    }, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("predict-dev", "score-dev", "predict-test", "score-test"))
    args = parser.parse_args()
    operation, split = args.action.split("-")
    if operation == "predict":
        predict(split)
    else:
        score(split)


if __name__ == "__main__":
    main()
