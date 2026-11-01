"""Batch 43 multi-view point tracking, visibility gating and calibrated 3-D diagnostics."""
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

# Tracker timing is a CPU baseline. Disable OpenCL so OpenCV cannot create an
# unguarded CUDA/OpenCL context during a nominally CPU-only stage.
if hasattr(cv2, "ocl"):
    cv2.ocl.setUseOpenCL(False)
CV_OPENCL_DISABLED = not hasattr(cv2, "ocl") or not cv2.ocl.useOpenCL()

import tapvidmv_batch44c_data as src

ROOT = Path(__file__).resolve().parents[1]
DATA = src.FROZEN
OUT = ROOT / "outputs/tapvidmv_batch44c"
BASE_METHODS = (
    "stationary_point", "klt_forward", "klt_fb", "template_recover4",
    "affine_stabilized_klt", "appearance_fb_055", "appearance_fb_070", "appearance_fb_085",
)
XVIEW_BASES = ("klt_fb", "klt_forward")
XVIEW_THRESHOLDS = (4, 8, 12)
XVIEW_METHODS = tuple(f"xview_{base}_{threshold}px" for base in XVIEW_BASES for threshold in XVIEW_THRESHOLDS)
METHODS = BASE_METHODS + XVIEW_METHODS
JACCARD_RADII = (1, 2, 4, 8, 16)
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


def sum_affine_cpu_ms(meta_rows) -> float:
    """Sum measured CPU registration time; pre-query entries are intentionally None."""
    return sum(
        float(item.get("cpu_ms") or 0.0)
        for item in (meta_rows or [])
        if isinstance(item, dict)
    )


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
    """Estimate query-frame to frame-t background affine from RGB only."""
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



def appearance_gate_trace(gray: list[np.ndarray], query_xy: list[float], threshold: float) -> list:
    """KLT-FB proposal with a local normalized-correlation appearance abstention gate."""
    proposal = klt_trace(gray, query_xy, fb=True)
    half = 5
    template = padded_patch(gray[0], np.asarray(query_xy, dtype=np.float64), half)
    if template.shape != (2 * half + 1, 2 * half + 1) or float(np.std(template)) < 5.0:
        return [proposal[0]] + [None] * (len(proposal) - 1)
    output = list(proposal)
    for frame_id in range(1, len(proposal)):
        point = proposal[frame_id]
        if point is None:
            continue
        patch = padded_patch(gray[frame_id], np.asarray(point, dtype=np.float64), half)
        score = float(cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)[0, 0])
        if not math.isfinite(score) or score < threshold:
            output[frame_id] = None
    return output


def affine_stabilized_klt_trace(gray: list[np.ndarray], query_xy: list[float], matrices: list) -> list:
    """Warp each frame to query-frame background coordinates, track there, then map back."""
    height, width = gray[0].shape
    stabilized = [gray[0]]
    valid_matrices = [matrices[0]]
    for frame_id in range(1, len(gray)):
        matrix = matrices[frame_id]
        if matrix is None:
            break
        warped = cv2.warpAffine(
            gray[frame_id], np.asarray(matrix, dtype=np.float64), (width, height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        stabilized.append(warped)
        valid_matrices.append(matrix)
    local = klt_trace(stabilized, query_xy, fb=True)
    output = [None] * len(gray)
    for frame_id, point in enumerate(local):
        if point is None:
            continue
        mapped = apply_affine(valid_matrices[frame_id], point)
        if mapped is None or not (0 <= mapped[0] < width and 0 <= mapped[1] < height):
            continue
        output[frame_id] = mapped
    return output


def scene_quality(frames: np.ndarray, gray: list[np.ndarray]) -> dict:
    sample_ids = list(range(0, len(gray), max(1, len(gray) // 20)))
    luma, contrast, texture, keypoints = [], [], [], []
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    for frame_id in sample_ids:
        image = gray[frame_id]
        luma.append(float(np.mean(image)))
        contrast.append(float(np.std(image)))
        texture.append(float(cv2.Laplacian(image, cv2.CV_64F).var()))
        found, _ = orb.detectAndCompute(image, None)
        keypoints.append(float(len(found or [])))
    def stats(values):
        return {"mean": round(float(np.mean(values)), 4), "p10": round(float(np.percentile(values, 10)), 4),
                "p90": round(float(np.percentile(values, 90)), 4)}
    return {"sampled_frames": len(sample_ids), "mean_luma": stats(luma),
            "contrast_std": stats(contrast), "laplacian_variance": stats(texture),
            "orb_keypoints": stats(keypoints)}


def triangulate_rays(pixels, rotations, translations, intrinsics) -> np.ndarray | None:
    """Least-squares intersection of calibrated world-frame camera rays."""
    if not (len(pixels) >= 2 and len(pixels) == len(rotations) == len(translations) == len(intrinsics)):
        return None
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    for pixel, rotation, translation, K in zip(pixels, rotations, translations, intrinsics):
        if pixel is None:
            return None
        try:
            homogeneous = np.asarray([float(pixel[0]), float(pixel[1]), 1.0], dtype=np.float64)
            camera_ray = homogeneous @ np.linalg.inv(np.asarray(K, dtype=np.float64))
            camera_ray = np.asarray([camera_ray[0], camera_ray[1], 1.0], dtype=np.float64)
            direction = np.asarray(rotation, dtype=np.float64).T @ camera_ray
            direction /= np.linalg.norm(direction)
            center = np.asarray(translation, dtype=np.float64)
            projector = identity - np.outer(direction, direction)
            A += projector
            b += projector @ center
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return None
    if not np.isfinite(A).all() or np.linalg.cond(A) > 1e8:
        return None
    try:
        point = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return point if np.isfinite(point).all() else None

def project_world_point(point: np.ndarray, rotation: np.ndarray,
                        translation: np.ndarray, intrinsics: np.ndarray) -> np.ndarray | None:
    camera = np.asarray(rotation, dtype=np.float64) @ (
        np.asarray(point, dtype=np.float64) - np.asarray(translation, dtype=np.float64)
    )
    if not np.isfinite(camera).all() or camera[2] <= 1e-8:
        return None
    homogeneous = np.asarray([camera[0] / camera[2], camera[1] / camera[2], 1.0]) @ np.asarray(
        intrinsics, dtype=np.float64
    )
    if not np.isfinite(homogeneous).all() or homogeneous[2] <= 1e-8:
        return None
    pixel = homogeneous[:2] / homogeneous[2]
    return pixel if np.isfinite(pixel).all() else None


def cross_view_consensus_group(camera_rows: dict, base_method: str, threshold_px: float,
                               rotations: np.ndarray, translations: np.ndarray,
                               intrinsics: np.ndarray, resolutions: np.ndarray,
                               nframes: int) -> dict[int, list]:
    if set(camera_rows) != {0, 1, 2}:
        raise RuntimeError("XVIEW_REQUIRES_EXACTLY_THREE_CAMERAS")
    query_frames = {int(row.get("query_frame", 0)) for row in camera_rows.values()}
    if len(query_frames) != 1:
        raise RuntimeError("XVIEW_QUERY_FRAME_ALIGNMENT_DRIFT")
    query_frame = query_frames.pop()
    if not 0 <= query_frame < nframes:
        raise RuntimeError("XVIEW_QUERY_FRAME_OUT_OF_RANGE")
    traces = {}
    for camera_id, row in camera_rows.items():
        base_trace = row["traces"][base_method]
        if len(base_trace) != nframes:
            raise RuntimeError("XVIEW_BASE_TRACE_LENGTH_MISMATCH")
        traces[camera_id] = [None] * nframes
        traces[camera_id][query_frame] = base_trace[query_frame]
    for frame_id in range(query_frame + 1, nframes):
        pixels = [camera_rows[c]["traces"][base_method][frame_id] for c in range(3)]
        if any(pixel is None or not np.isfinite(np.asarray(pixel, dtype=np.float64)).all()
               for pixel in pixels):
            continue
        point = triangulate_rays(
            pixels, [rotations[c, frame_id] for c in range(3)],
            [translations[c, frame_id] for c in range(3)],
            [intrinsics[c] for c in range(3)],
        )
        if point is None:
            continue
        projected, residuals, valid = {}, [], True
        for camera_id in range(3):
            pixel = project_world_point(point, rotations[camera_id, frame_id],
                                        translations[camera_id, frame_id], intrinsics[camera_id])
            if pixel is None:
                valid = False
                break
            width, height = [int(x) for x in resolutions[camera_id]]
            if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
                valid = False
                break
            residuals.append(float(np.linalg.norm(pixel - np.asarray(pixels[camera_id], dtype=np.float64))))
            projected[camera_id] = pixel.astype(float).tolist()
        if valid and len(residuals) == 3 and max(residuals) <= float(threshold_px):
            for camera_id in range(3):
                traces[camera_id][frame_id] = projected[camera_id]
    return traces


def predict(split: str) -> None:
    if split not in ("dev", "test"):
        raise ValueError("INVALID_PREDICTION_SPLIT")
    manifest = src.audit_public()
    queries = doc(DATA / "queries-public.json")
    manifest_sha = sha_bytes((DATA / "manifest.json").read_bytes())
    config_sha = sha_bytes(src.json_bytes(src.config()))
    public_sha = sha_bytes((DATA / "queries-public.json").read_bytes())
    selection_sha = None
    if split == "test":
        selection_path = OUT / "dev_selection.json"
        require_pin(selection_path)
        selection = doc(selection_path)
        selection_sha = sha_bytes(selection_path.read_bytes())
        dev_results_path = OUT / "dev_results.json"
        require_pin(dev_results_path)
        if (selection.get("source_commit") != gitsha()
                or selection.get("config_sha256") != config_sha
                or selection.get("selection_source") != "DEV only"
                or selection.get("selected_method") not in METHODS
                or selection.get("dev_results_sha256") != sha_bytes(dev_results_path.read_bytes())):
            raise RuntimeError("TAPVIDMV_TEST_WITHOUT_VALID_DEV_SELECTION")
    OUT.mkdir(parents=True, exist_ok=True)
    output_path = OUT / f"{split}_predictions.json"
    if output_path.exists():
        require_pin(output_path)
        saved = doc(output_path)
        if (saved.get("source_commit") != gitsha()
                or saved.get("config_sha256") != config_sha
                or saved.get("manifest_sha256") != manifest_sha
                or saved.get("public_query_sha256") != public_sha
                or saved.get("split") != split
                or saved.get("dev_selection_sha256") != selection_sha):
            raise RuntimeError("TAPVIDMV_STALE_PINNED_PREDICTION")
        print("TAPVIDMV_EXISTING_PIN_OK", split, flush=True)
        return
    records = []
    scene_quality_all = {}
    for scene in manifest["scenes"]:
        if scene["split"] != split:
            continue
        scene_queries = queries[scene["name"]]
        query_frames = {int(q["query_frame"]) for q in scene_queries}
        if len(query_frames) != 1:
            raise RuntimeError("SCENE_QUERY_FRAME_ALIGNMENT_DRIFT")
        scene_query_frame = query_frames.pop()
        scene_record_start = len(records)
        camera_cache = {}
        scene_quality_by_camera = {}
        for camera_id in range(3):
            relative = scene["frame_files"][str(camera_id)]
            frames = np.load(DATA / relative, mmap_mode="r", allow_pickle=False)
            gray_full = rgb_to_gray(frames)
            gray = gray_full[scene_query_frame:]
            matrices, affine_meta = estimate_affine_sequence(gray)
            camera_cache[camera_id] = (gray, matrices, affine_meta, scene_query_frame, len(frames))
            scene_quality_by_camera[str(camera_id)] = scene_quality(frames, gray_full)
        scene_quality_all[scene["name"]] = scene_quality_by_camera
        for query in scene_queries:
            camera_id = int(query["camera_id"])
            gray, matrices, affine_meta, query_frame, nframes = camera_cache[camera_id]
            xy = query["query_xy"]
            traces = {}
            cpu_query_ms = {}
            for method in BASE_METHODS:
                started = time.perf_counter()
                if method == "stationary_point":
                    trace = stationary_trace(len(gray), xy)
                elif method == "klt_forward":
                    trace = klt_trace(gray, xy, fb=False)
                elif method == "klt_fb":
                    trace = klt_trace(gray, xy, fb=True)
                elif method == "template_recover4":
                    trace = template_trace(gray, xy)
                elif method == "affine_stabilized_klt":
                    trace = affine_stabilized_klt_trace(gray, xy, matrices)
                elif method == "appearance_fb_055":
                    trace = appearance_gate_trace(gray, xy, 0.55)
                elif method == "appearance_fb_070":
                    trace = appearance_gate_trace(gray, xy, 0.70)
                elif method == "appearance_fb_085":
                    trace = appearance_gate_trace(gray, xy, 0.85)
                else:
                    raise RuntimeError("UNKNOWN_TAPVIDMV_METHOD:" + method)
                cpu_query_ms[method] = round(1000 * (time.perf_counter() - started), 3)
                if len(trace) != len(gray):
                    raise RuntimeError("TAPVIDMV_TRACE_LENGTH_MISMATCH")
                traces[method] = [None] * query_frame + trace
            affine_xy = [None] * query_frame + [apply_affine(matrix, xy) for matrix in matrices]
            affine_meta_full = [None] * query_frame + affine_meta
            records.append({
                "name": scene["name"],
                "query_id": query["id"],
                "camera_id": camera_id,
                "query_frame": query_frame,
                "cross_view_group": query["cross_view_group"],
                "track_slot": query["track_slot"],
                "target_frame": query["target_frame"],
                "traces": traces,
                "cpu_query_ms": cpu_query_ms,
                "camera_affine_xy": affine_xy,
                "camera_affine_meta": affine_meta_full,
            })
        scene_records = records[scene_record_start:]
        with np.load(DATA / scene["calibration_file"], allow_pickle=False) as calibration:
            rotations = np.asarray(calibration["rotations"], dtype=np.float64)
            translations = np.asarray(calibration["translations"], dtype=np.float64)
            intrinsics = np.asarray(calibration["intrinsics"], dtype=np.float64)
            resolutions = np.asarray(calibration["resolution"], dtype=np.int32)
        grouped_rows = {}
        for row in scene_records:
            grouped_rows.setdefault(row["cross_view_group"], []).append(row)
        for group_rows in grouped_rows.values():
            camera_rows = {int(row["camera_id"]): row for row in group_rows}
            if len(camera_rows) != 3 or set(camera_rows) != {0, 1, 2}:
                raise RuntimeError("XVIEW_QUERY_GROUP_CAMERA_SET_DRIFT")
            for base_method in XVIEW_BASES:
                for threshold in XVIEW_THRESHOLDS:
                    started = time.perf_counter()
                    joint = cross_view_consensus_group(
                        camera_rows, base_method, threshold, rotations, translations,
                        intrinsics, resolutions, int(scene["nframes"])
                    )
                    elapsed_ms = 1000.0 * (time.perf_counter() - started)
                    method = f"xview_{base_method}_{threshold}px"
                    for camera_id, row in camera_rows.items():
                        row["traces"][method] = joint[camera_id]
                        row["cpu_query_ms"][method] = round(elapsed_ms / 3.0, 3)
        print("TAPVIDMV_CPU_PREDICTED", split, scene["name"],
              len(scene_queries), "queries", flush=True)
    payload = {
        "dataset": "TAPVid-MV EgoExo4D + PACE public Rerun viewer subsets",
        "split": split,
        "source_commit": gitsha(),
        "config_sha256": config_sha,
        "manifest_sha256": manifest_sha,
        "public_query_sha256": public_sha,
        "dev_selection_sha256": selection_sha,
        "oracle_unread_during_prediction": True,
        "methods": list(METHODS),
        "scene_quality_by_camera": scene_quality_all,
        "camera_motion_estimator": {
            "method": "ORB + BFMatcher ratio test + RANSAC partial affine; affine-stabilized KLT warps each frame into query-frame coordinates",
            "nfeatures": ORB_FEATURES,
            "ratio": ORB_RATIO,
            "ransac_reprojection_px": ORB_RANSAC_PX,
            "frame_pair": "query query frame to each target frame independently",
        },
        "cross_view_consensus": {
            "method": "triangulate three RGB-only per-camera KLT predictions with public camera calibration, reproject into all three synchronized views, accept only if all three reprojection residuals are within the frozen threshold",
            "base_methods": list(XVIEW_BASES),
            "thresholds_px": list(XVIEW_THRESHOLDS),
            "calibration_files_are_public_pose_only": True,
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
    if split not in ("dev", "test"):
        raise ValueError("INVALID_SCORE_SPLIT")
    manifest = src.audit_private(split)
    query_doc = doc(DATA / "queries-public.json")
    prediction_path = OUT / f"{split}_predictions.json"
    require_pin(prediction_path)
    saved = doc(prediction_path)
    manifest_sha = sha_bytes((DATA / "manifest.json").read_bytes())
    config_sha = sha_bytes(src.json_bytes(src.config()))
    public_sha = sha_bytes((DATA / "queries-public.json").read_bytes())
    if (saved.get("source_commit") != gitsha()
            or saved.get("config_sha256") != config_sha
            or saved.get("manifest_sha256") != manifest_sha
            or saved.get("public_query_sha256") != public_sha
            or saved.get("oracle_unread_during_prediction") is not True
            or saved.get("methods") != list(METHODS)
            or saved.get("split") != split):
        raise RuntimeError("TAPVIDMV_SCORE_SOURCE_ORACLE_FIREWALL_FAILED")
    if split == "test":
        selection_path = OUT / "dev_selection.json"
        require_pin(selection_path)
        selection = doc(selection_path)
        dev_results_path = OUT / "dev_results.json"
        require_pin(dev_results_path)
        if (selection.get("source_commit") != gitsha()
                or selection.get("config_sha256") != config_sha
                or selection.get("selection_source") != "DEV only"
                or selection.get("selected_method") not in METHODS
                or selection.get("dev_results_sha256") != sha_bytes(dev_results_path.read_bytes())
                or saved.get("dev_selection_sha256") != sha_bytes(selection_path.read_bytes())):
            raise RuntimeError("TAPVIDMV_TEST_SCORE_WITHOUT_VALID_DEV_SELECTION")
    scenes = [x for x in manifest["scenes"] if x["split"] == split]
    records = saved.get("records", [])
    keys = [(x.get("name"), x.get("query_id")) for x in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError("TAPVIDMV_DUPLICATE_PREDICTION_RECORD")
    prediction_by_key = {(x["name"], x["query_id"]): x for x in records}
    expected_keys = {
        (scene["name"], q["id"])
        for scene in scenes for q in query_doc[scene["name"]]
    }
    if set(prediction_by_key) != expected_keys or len(records) != len(expected_keys):
        raise RuntimeError("TAPVIDMV_PREDICTION_QUERY_SET_MISMATCH")
    for row in records:
        if set(row.get("traces", {})) != set(METHODS):
            raise RuntimeError("TAPVIDMV_PREDICTION_METHOD_SET_MISMATCH")
        if any(len(row["traces"][method]) != next(
            scene["nframes"] for scene in scenes if scene["name"] == row["name"]
        ) for method in METHODS):
            raise RuntimeError("TAPVIDMV_PREDICTION_TRACE_LENGTH_MISMATCH")
    per_scene = {}
    method_scene_pck = {method: {} for method in METHODS}
    method_scene_jaccard = {method: {} for method in METHODS}
    method_totals = {
        method: {
            "all_inframe": 0, "all_inframe_correct8": 0, "all_inframe_accepted": 0,
            "visible": 0, "visible_correct8": 0, "visible_accepted": 0,
            "occluded": 0, "occluded_false_accept": 0, "occluded_abstain": 0,
            "out": 0, "out_false_accept": 0, "unknown": 0, "missing": 0,
            "jaccard_tp": {str(r): 0 for r in JACCARD_RADII},
            "jaccard_fp": {str(r): 0 for r in JACCARD_RADII},
            "jaccard_fn": {str(r): 0 for r in JACCARD_RADII},
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
    for scene in scenes:
        oracle_path = DATA / scene["oracle_file"]
        with np.load(oracle_path, allow_pickle=False) as oracle:
            qids = [str(x) for x in oracle["qids"].tolist()]
            q_index = {qid: i for i, qid in enumerate(qids)}
            scene_rows = [prediction_by_key[(scene["name"], q["id"])]
                          for q in query_doc[scene["name"]]]
            scene_metrics = {method: {
                "all_inframe": 0, "all_inframe_correct8": 0, "all_inframe_accepted": 0,
                "visible": 0, "visible_correct8": 0, "visible_accepted": 0,
                "occluded": 0, "occluded_false_accept": 0, "occluded_abstain": 0,
                "out": 0, "out_false_accept": 0, "unknown": 0, "missing": 0,
                "jaccard_tp": {str(r): 0 for r in JACCARD_RADII},
                "jaccard_fp": {str(r): 0 for r in JACCARD_RADII},
                "jaccard_fn": {str(r): 0 for r in JACCARD_RADII},
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
                query_frame = int(query["query_frame"])
                if int(row.get("query_frame", -1)) != query_frame:
                    raise RuntimeError("PREDICTION_QUERY_FRAME_DRIFT")
                for frame_id in range(query_frame + 1, nframes):
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
                        pred_error = None
                        if pred is not None and np.isfinite(gt[frame_id]).all():
                            pred_error = float(np.linalg.norm(
                                np.asarray(pred, dtype=np.float64) - gt[frame_id]
                            ))
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
                            if pred is not None and pred_error is not None:
                                metric["visible_correct8"] += int(pred_error <= 8)
                        if status in (STATUS["visible"], STATUS["occluded"], STATUS["out"]):
                            for radius in JACCARD_RADII:
                                key = str(radius)
                                if status == STATUS["visible"]:
                                    if pred is None:
                                        metric["jaccard_fn"][key] += 1
                                    elif pred_error is not None and pred_error <= radius:
                                        metric["jaccard_tp"][key] += 1
                                    else:
                                        metric["jaccard_fp"][key] += 1
                                        metric["jaccard_fn"][key] += 1
                                elif pred is not None:
                                    metric["jaccard_fp"][key] += 1
                        if present[frame_id] and in_frame[frame_id] and np.isfinite(gt[frame_id]).all():
                            metric["all_inframe"] += 1
                            metric["all_inframe_accepted"] += int(pred is not None)
                            if pred is not None and pred_error is not None:
                                metric["all_inframe_correct8"] += int(pred_error <= 8)
                for method in METHODS:
                    duration = float(row["cpu_query_ms"][method])
                    scene_metrics[method]["cpu_ms_per_target_frame"].append(
                        duration / max(1, nframes - int(row["query_frame"]) - 1)
                    )
            for method in METHODS:
                metric = scene_metrics[method]
                for key in ("all_inframe", "all_inframe_correct8", "all_inframe_accepted",
                            "visible", "visible_correct8", "visible_accepted",
                            "occluded", "occluded_false_accept", "occluded_abstain",
                            "out", "out_false_accept", "unknown", "missing"):
                    method_totals[method][key] += metric[key]
                for radius in JACCARD_RADII:
                    key = str(radius)
                    for count_name in ("jaccard_tp", "jaccard_fp", "jaccard_fn"):
                        method_totals[method][count_name][key] += metric[count_name][key]
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
                metric["jaccard_by_radius"] = {
                    str(radius): (
                        metric["jaccard_tp"][str(radius)] /
                        max(1, metric["jaccard_tp"][str(radius)] + metric["jaccard_fp"][str(radius)] + metric["jaccard_fn"][str(radius)])
                    ) for radius in JACCARD_RADII
                }
                metric["jaccard_mean_1_2_4_8_16"] = float(np.mean(list(metric["jaccard_by_radius"].values())))
                method_scene_pck[method][scene["name"]] = metric["pck8_all_inframe"]
                method_scene_jaccard[method][scene["name"]] = metric["jaccard_mean_1_2_4_8_16"]
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
            "jaccard_by_radius_micro": {
                str(radius): (
                    counts["jaccard_tp"][str(radius)] /
                    max(1, counts["jaccard_tp"][str(radius)] + counts["jaccard_fp"][str(radius)] + counts["jaccard_fn"][str(radius)])
                ) for radius in JACCARD_RADII
            },
            "jaccard_mean_micro": float(np.mean([
                counts["jaccard_tp"][str(radius)] /
                max(1, counts["jaccard_tp"][str(radius)] + counts["jaccard_fp"][str(radius)] + counts["jaccard_fn"][str(radius)])
                for radius in JACCARD_RADII
            ])),
            "jaccard_mean_scene_macro": round(float(np.mean(list(method_scene_jaccard[method].values()))), 5),
            "scene_bootstrap_ci95_jaccard": bootstrap_scene_ci(method_scene_jaccard[method], seed=43),
        }
    for method in METHODS:
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
    per_subset = {}
    for subset in sorted({scene["subset"] for scene in scenes}):
        subset_names = [scene["name"] for scene in scenes if scene["subset"] == subset]
        per_subset[subset] = {}
        for method in METHODS:
            rows = [per_scene[name][method] for name in subset_names]
            per_subset[subset][method] = {
                "scene_count": len(rows),
                "pck8_depth_visible_scene_macro": (
                    round(float(np.mean(values)), 5)
                    if (values := [r["pck8_depth_visible"] for r in rows if r["pck8_depth_visible"] is not None]) else None
                ),
                "coverage_depth_visible_scene_macro": (
                    round(float(np.mean(values)), 5)
                    if (values := [r["coverage_depth_visible"] for r in rows if r["coverage_depth_visible"] is not None]) else None
                ),
                "false_accept_occluded_scene_macro": (
                    round(float(np.mean(values)), 5)
                    if (values := [r["false_accept_occluded"] for r in rows if r["false_accept_occluded"] is not None]) else None
                ),
                "jaccard_mean_scene_macro": round(float(np.mean([r["jaccard_mean_1_2_4_8_16"] for r in rows])), 5),
            }

    cross_view_3d = {}
    record_by_key = {(row["name"], row["query_id"]): row for row in saved["records"]}
    for method in METHODS:
        errors, oracle_errors = [], []
        all_visible_groups = 0
        triangulated_groups = 0
        dropped_groups = 0
        for scene in scenes:
            scene_queries = query_doc[scene["name"]]
            grouped = {}
            for query in scene_queries:
                grouped.setdefault(query["cross_view_group"], []).append(query)
            with np.load(DATA / scene["oracle_file"], allow_pickle=False) as oracle:
                qids = [str(x) for x in oracle["qids"].tolist()]
                q_index = {qid: i for i, qid in enumerate(qids)}
                for group_name, group_queries in grouped.items():
                    group_queries = sorted(group_queries, key=lambda q: int(q["camera_id"]))
                    if len(group_queries) != 3:
                        raise RuntimeError("CROSS_VIEW_GROUP_CARDINALITY_DRIFT")
                    qi_values = [q_index[q["id"]] for q in group_queries]
                    target_frames = {int(q["target_frame"]) for q in group_queries}
                    if len(target_frames) != 1:
                        raise RuntimeError("CROSS_VIEW_TARGET_FRAME_DRIFT")
                    target = next(iter(target_frames))
                    statuses = [int(oracle["depth_status"][qi, target]) for qi in qi_values]
                    if any(status != STATUS["visible"] for status in statuses):
                        continue
                    all_visible_groups += 1
                    predictions = [record_by_key[(scene["name"], q["id"])]["traces"][method][target]
                                   for q in group_queries]
                    if any(point is None for point in predictions):
                        dropped_groups += 1
                        continue
                    rotations = [oracle["camera_rotation_matrix"][qi, target] for qi in qi_values]
                    translations = [oracle["camera_translation_xyz"][qi, target] for qi in qi_values]
                    intrinsics = [oracle["camera_intrinsics"][qi] for qi in qi_values]
                    world_points = np.asarray([oracle["world_xyz"][qi, target] for qi in qi_values], dtype=np.float64)
                    if not np.isfinite(world_points).all() or np.max(np.linalg.norm(world_points - world_points[0], axis=1)) > 1e-3:
                        raise RuntimeError("CROSS_VIEW_WORLD_POINT_IDENTITY_DRIFT")
                    estimate = triangulate_rays(predictions, rotations, translations, intrinsics)
                    oracle_pixels = [oracle["gt_xy"][qi, target].tolist() for qi in qi_values]
                    oracle_estimate = triangulate_rays(oracle_pixels, rotations, translations, intrinsics)
                    if oracle_estimate is not None:
                        oracle_errors.append(float(np.linalg.norm(oracle_estimate - world_points[0])))
                    if estimate is None:
                        dropped_groups += 1
                        continue
                    triangulated_groups += 1
                    errors.append(float(np.linalg.norm(estimate - world_points[0])))
        cross_view_3d[method] = {
            "all_three_views_ground_truth_visible_groups": all_visible_groups,
            "triangulated_groups": triangulated_groups,
            "dropped_groups": dropped_groups,
            "coverage_given_all_three_visible": round(triangulated_groups / all_visible_groups, 5) if all_visible_groups else None,
            "world_error_m": summarize(errors),
            "within_5cm": round(float(np.mean(np.asarray(errors) <= 0.05)), 5) if errors else None,
            "within_10cm": round(float(np.mean(np.asarray(errors) <= 0.10)), 5) if errors else None,
            "oracle_gt_pixel_triangulation_error_m": summarize(oracle_errors),
        }

    affine_cost_by_camera_video = {}
    for row in saved["records"]:
        key = (row["name"], int(row["camera_id"]))
        affine_cost_by_camera_video.setdefault(
            key, sum_affine_cpu_ms(row["camera_affine_meta"])
        )
    motion = {
        "screen_vs_world_only_cosine": summarize(screen_world_cosines),
        "image_affine_compensated_vs_world_only_cosine": summarize(affine_world_cosines),
        "oracle_camera_subtracted_vs_world_only_cosine": summarize(oracle_world_cosines),
        "camera_only_screen_displacement_share_proxy": summarize(camera_dominance),
        "camera_translation_net_m": summarize(camera_translation),
        "camera_rotation_net_deg": summarize(camera_rotation),
        "orb_affine_static_point_reprojection_error_px": summarize(affine_errors),
        "orb_affine_coverage": round(affine_valid / affine_total, 5) if affine_total else None,
        "orb_affine_valid": affine_valid,
        "orb_affine_evaluable": affine_total,
        "affine_registration_ms_per_camera_video": summarize(
            list(affine_cost_by_camera_video.values())
        ),
        "per_camera_pose": {
            camera: {
                "queries": values["queries"],
                "net_translation_m": summarize(values["translation_m"]),
                "net_rotation_deg": summarize(values["rotation_deg"]),
            } for camera, values in per_camera.items()
        },
    }
    result = {
        "dataset": "TAPVid-MV EgoExo4D + PACE public Rerun viewer subsets",
        "source_commit": gitsha(),
        "split": split,
        "scene_count": len(scenes),
        "query_count": len(saved["records"]),
        "manifest_sha256": manifest_sha,
        "config_sha256": config_sha,
        "sealed_prediction_sha256": sha_bytes(prediction_path.read_bytes()),
        "primary_metric": "PCK@8 on projected 3D point tracks; visibility-specific metrics only on exact depth-cloud snapshots",
        "visibility_contract": "depth-cloud-derived visible/occluded labels at exact snapshots; unknown elsewhere",
        "methods": aggregate,
        "per_subset": per_subset,
        "cross_view_3d": cross_view_3d,
        "scene_quality_by_camera": saved.get("scene_quality_by_camera", {}),
        "camera_motion_decomposition": motion,
        "per_scene": per_scene,
        "caveats": [
            "The Rerun recordings display a subsampled track set; these are not the full metadata-declared tracks.",
            "The same query-frame 3-D point IDs are queried in all three views; calibrated triangulation is scored only when all three depth-derived labels are visible.",
            "The 3-D triangulation uses the released camera calibration and poses; it is an evaluation diagnostic, not a predictor input.",
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
    pin(output)
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



def select_dev() -> dict:
    config = src.config()
    config_sha = sha_bytes(src.json_bytes(config))
    results_path = OUT / "dev_results.json"
    if not results_path.is_file():
        raise RuntimeError("TAPVIDMV_DEV_RESULTS_MISSING")
    require_pin(results_path)
    results = doc(results_path)
    if (results.get("source_commit") != gitsha()
            or results.get("split") != "dev"
            or results.get("config_sha256") != config_sha):
        raise RuntimeError("TAPVIDMV_DEV_SELECTION_SOURCE_MISMATCH")
    rules = config["selection"]
    candidates = []
    for method in rules["candidate_methods"]:
        metric = results["methods"][method]
        if (metric["coverage_depth_visible"] is not None
                and metric["false_accept_occluded"] is not None
                and metric["false_accept_out"] is not None
                and metric["coverage_depth_visible"] >= rules["min_visible_coverage"]
                and metric["false_accept_occluded"] <= rules["max_invisible_false_accept_rate"]
                and metric["false_accept_out"] <= rules["max_invisible_false_accept_rate"]):
            candidates.append(method)
    eligible = sorted(
        candidates,
        key=lambda method: (
            results["methods"][method]["jaccard_mean_scene_macro"],
            results["methods"][method]["coverage_depth_visible"],
        ),
        reverse=True,
    )
    selected = eligible[0] if eligible else rules["fallback_method"]
    payload = {
        "dataset": "TAPVid-MV EgoExo4D + PACE",
        "split": "dev",
        "status": "PASS" if eligible else "BLOCK_NO_ELIGIBLE_METHOD",
        "source_commit": gitsha(),
        "config_sha256": config_sha,
        "dev_results_sha256": sha_bytes(results_path.read_bytes()),
        "rules": rules,
        "eligible_candidates": eligible,
        "selected_method": selected,
        "selection_source": "DEV only",
        "reason": "maximize scene-macro Jaccard mean among candidates satisfying predeclared coverage and occlusion/out-of-frame false-accept limits; otherwise use frozen fallback",
    }
    path = OUT / "dev_selection.json"
    once(path, payload)
    pin(path)
    print("TAPVIDMV_DEV_SELECTION", json.dumps(payload, sort_keys=True), flush=True)
    return payload


def final_audit() -> None:
    src.audit()
    config_sha = sha_bytes(src.json_bytes(src.config()))
    for split in ("dev", "test"):
        pred = OUT / f"{split}_predictions.json"
        result = OUT / f"{split}_results.json"
        require_pin(pred)
        if not result.is_file():
            raise RuntimeError("TAPVIDMV_FINAL_RESULT_MISSING:" + split)
        require_pin(result)
        prediction = doc(pred)
        scored = doc(result)
        if (prediction.get("source_commit") != gitsha()
                or prediction.get("config_sha256") != config_sha
                or scored.get("source_commit") != gitsha()
                or scored.get("config_sha256") != config_sha
                or scored.get("sealed_prediction_sha256") != sha_bytes(pred.read_bytes())):
            raise RuntimeError("TAPVIDMV_FINAL_SOURCE_OR_PIN_AUDIT_FAILED:" + split)
    selection = OUT / "dev_selection.json"
    require_pin(selection)
    chosen = doc(selection)
    if (chosen.get("source_commit") != gitsha()
            or chosen.get("config_sha256") != config_sha
            or chosen.get("selection_source") != "DEV only"
            or chosen.get("selected_method") not in METHODS):
        raise RuntimeError("TAPVIDMV_FINAL_DEV_SELECTION_AUDIT_FAILED")
    test_prediction = doc(OUT / "test_predictions.json")
    if test_prediction.get("dev_selection_sha256") != sha_bytes(selection.read_bytes()):
        raise RuntimeError("TAPVIDMV_FINAL_TEST_SELECTION_PIN_MISMATCH")
    print("TAPVIDMV_FINAL_AUDIT_PASS", gitsha(), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=(
        "predict-dev", "score-dev", "select-dev",
        "predict-test", "score-test", "final-audit",
    ))
    args = parser.parse_args()
    if args.action.startswith("predict-"):
        predict(args.action.split("-", 1)[1])
    elif args.action.startswith("score-"):
        score(args.action.split("-", 1)[1])
    elif args.action == "select-dev":
        select_dev()
    else:
        final_audit()


if __name__ == "__main__":
    main()
