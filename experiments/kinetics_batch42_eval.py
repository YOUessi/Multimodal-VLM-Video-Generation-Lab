"""Batch42 CPU tracking, visibility gating and 2-D background-affine diagnostics."""
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

import kinetics_batch42_data as src

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/tapvid_kinetics_batch42/batch42_frozen"
OUT = ROOT / "outputs/kinetics_batch42"
METHODS = (
    "stationary_point", "klt_forward", "klt_fb", "template_recover4",
    "affine_stabilized_klt", "appearance_fb_055", "appearance_fb_070",
    "appearance_fb_085",
)
RADII = (1, 2, 4, 8, 16)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def doc(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()


def once(path: Path, payload: dict) -> None:
    src.atomic_once(path, json_bytes(payload))


def pin(path: Path) -> None:
    src.atomic_once(path.with_suffix(path.suffix + ".sha256"),
                    (sha_bytes(path.read_bytes()) + "\n").encode())


def require_pin(path: Path) -> None:
    tag = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not tag.is_file():
        raise RuntimeError("MISSING_ORACLE_FIREWALL_PREDICTION_PIN:" + path.name)
    if tag.read_text(encoding="utf-8").strip() != sha_bytes(path.read_bytes()):
        raise RuntimeError("PREDICTION_CHANGED_AFTER_SEAL:" + path.name)


def gitsha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def load_config(path: str | Path | None = None) -> dict:
    return doc(Path(path) if path else ROOT / "configs/kinetics_batch42.json")


def check_video(frames: np.ndarray, queries: list[dict]) -> None:
    if (frames.ndim != 4 or frames.dtype != np.uint8
            or frames.shape[1:] != (256, 256, 3)):
        raise ValueError("KINETICS_VIDEO_SHAPE_OR_DTYPE_INVALID")
    for q in queries:
        xy = np.asarray(q["query_xy"], dtype=np.float64)
        if (not 0 <= int(q["query_time"]) < len(frames) - 1
                or xy.shape != (2,) or not np.isfinite(xy).all()
                or np.any(xy < -0.5) or np.any(xy >= [256, 256])):
            raise ValueError("KINETICS_BAD_PUBLIC_QUERY")


def internal_point(xy: list | np.ndarray) -> np.ndarray:
    point = np.asarray(xy, dtype=np.float32)
    point = np.clip(point, np.float32(0.0), np.float32(255.0))
    return np.ascontiguousarray(point, dtype=np.float32).reshape(1, 1, 2)


def klt_point(gray: list[np.ndarray], query: dict, *, fb: bool,
              fb_error_px: float) -> list:
    t = int(query["query_time"])
    original = [float(v) for v in query["query_xy"]]
    point = internal_point(original)
    trace = [None] * len(gray)
    trace[t] = original
    h, w = gray[0].shape
    for i in range(t + 1, len(gray)):
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            gray[i - 1], gray[i], point, None, winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if nxt is None or status is None or int(status.reshape(-1)[0]) != 1:
            break
        xy = nxt.reshape(2)
        if not np.isfinite(xy).all() or not (0 <= xy[0] < w and 0 <= xy[1] < h):
            break
        if fb:
            back, back_status, _ = cv2.calcOpticalFlowPyrLK(
                gray[i], gray[i - 1], nxt, None, winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            if (back is None or back_status is None
                    or int(back_status.reshape(-1)[0]) != 1
                    or not np.isfinite(back).all()
                    or float(np.linalg.norm(back.reshape(2) - point.reshape(2))) > fb_error_px):
                break
        trace[i] = [round(float(v), 4) for v in xy]
        point = nxt
    return trace


def patch_at(gray: np.ndarray, xy: list | np.ndarray, size: int) -> np.ndarray | None:
    point = np.clip(np.asarray(xy, dtype=np.float32), [0.0, 0.0],
                    [gray.shape[1] - 1.0, gray.shape[0] - 1.0])
    patch = cv2.getRectSubPix(gray, (size, size), (float(point[0]), float(point[1])))
    if patch is None or patch.shape != (size, size):
        return None
    return patch


def normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float32) - float(np.mean(a))
    y = np.asarray(b, dtype=np.float32) - float(np.mean(b))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom < 1e-6:
        return 0.0
    return float(np.sum(x * y) / denom)


def appearance_gate(gray: list[np.ndarray], query: dict, base_trace: list,
                    threshold: float, patch_size: int) -> list:
    t = int(query["query_time"])
    anchor = patch_at(gray[t], query["query_xy"], patch_size)
    out = list(base_trace)
    if anchor is None or float(np.std(anchor)) < 5.0:
        return [out[i] if i <= t else None for i in range(len(out))]
    for i in range(t + 1, len(gray)):
        if out[i] is None:
            continue
        candidate = patch_at(gray[i], out[i], patch_size)
        if candidate is None or normalized_correlation(anchor, candidate) < threshold:
            out[i] = None
    return out


def template_recover(gray: list[np.ndarray], query: dict, cfg: dict) -> list:
    t = int(query["query_time"])
    size = int(cfg["patch_size"])
    radius = int(cfg["template_search_radius_px"])
    threshold = float(cfg["template_ncc_min"])
    max_misses = int(cfg["template_max_consecutive_misses"])
    anchor = np.asarray(query["query_xy"], dtype=np.float64)
    out = [None] * len(gray)
    out[t] = [float(v) for v in anchor]
    template = patch_at(gray[t], anchor, size)
    if template is None or float(np.std(template)) < 5.0:
        return out
    half = size // 2
    last = np.clip(anchor, [0, 0], [gray[0].shape[1] - 1, gray[0].shape[0] - 1])
    misses = 0
    h, w = gray[0].shape
    for i in range(t + 1, len(gray)):
        if misses >= max_misses:
            break
        pad = radius + half + 3
        padded = cv2.copyMakeBorder(
            gray[i], pad, pad, pad, pad, cv2.BORDER_REFLECT_101
        )
        cx = int(round(float(last[0]))) + pad
        cy = int(round(float(last[1]))) + pad
        roi = padded[cy-radius-half:cy+radius+half+1,
                     cx-radius-half:cx+radius+half+1]
        if roi.shape[0] < size or roi.shape[1] < size:
            misses += 1
            continue
        response = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, top_left = cv2.minMaxLoc(response)
        found = last + np.asarray([top_left[0] - radius, top_left[1] - radius])
        if (not np.isfinite(score) or score < threshold
                or not (0 <= found[0] < w and 0 <= found[1] < h)):
            misses += 1
            continue
        last = found
        misses = 0
        out[i] = [round(float(v), 4) for v in found]
    return out


def estimate_affine_current_to_previous(previous: np.ndarray, current: np.ndarray,
                                         cfg: dict) -> tuple[np.ndarray | None, dict]:
    orb = cv2.ORB_create(nfeatures=int(cfg["affine_orb_features"]), scaleFactor=1.2,
                         nlevels=8, fastThreshold=8)
    kp_prev, des_prev = orb.detectAndCompute(previous, None)
    kp_curr, des_curr = orb.detectAndCompute(current, None)
    base = {"accepted": False, "matches": 0, "inliers": 0,
            "inlier_ratio": 0.0, "reprojection_p50_px": None,
            "translation_px": None, "rotation_deg": None, "scale": None}
    if des_prev is None or des_curr is None:
        base["reason"] = "NO_DESCRIPTORS"
        return None, base
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_curr, des_prev, k=2)
    good = [a for pair in pairs if len(pair) == 2
            for a, b in [pair] if a.distance < float(cfg["affine_ratio"]) * b.distance]
    base["matches"] = len(good)
    if len(good) < int(cfg["affine_min_matches"]):
        base["reason"] = "TOO_FEW_MATCHES"
        return None, base
    source = np.float32([kp_curr[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    target = np.float32([kp_prev[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, mask = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC,
        ransacReprojThreshold=float(cfg["affine_ransac_reprojection_px"]),
        maxIters=2000, confidence=0.99, refineIters=10,
    )
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        base["reason"] = "RANSAC_INVALID"
        return None, base
    keep = mask.reshape(-1).astype(bool)
    count = int(keep.sum())
    ratio = count / max(1, len(good))
    predicted = cv2.transform(source[keep], matrix).reshape(-1, 2)
    actual = target[keep].reshape(-1, 2)
    residual = (np.linalg.norm(predicted - actual, axis=1)
                if count else np.asarray([], dtype=np.float64))
    p50 = float(np.median(residual)) if len(residual) else None
    scale = math.hypot(float(matrix[0, 0]), float(matrix[1, 0]))
    angle = math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
    translation = math.hypot(float(matrix[0, 2]), float(matrix[1, 2]))
    accepted = (
        count >= int(cfg["affine_min_inliers"])
        and ratio >= float(cfg["affine_min_inlier_ratio"])
        and p50 is not None and p50 <= float(cfg["affine_max_reprojection_p50_px"])
        and float(cfg["affine_scale_min"]) <= scale <= float(cfg["affine_scale_max"])
        and abs(angle) <= float(cfg["affine_max_rotation_deg"])
    )
    base.update({
        "accepted": bool(accepted), "inliers": count,
        "inlier_ratio": round(float(ratio), 5),
        "reprojection_p50_px": round(p50, 4) if p50 is not None else None,
        "translation_px": round(translation, 4),
        "rotation_deg": round(angle, 4),
        "scale": round(scale, 5),
        "reason": "ACCEPTED" if accepted else "GEOMETRY_REJECTED",
    })
    return (matrix.astype(np.float64) if accepted else None), base


def estimate_affines(gray: list[np.ndarray], cfg: dict) -> tuple[list, list, float]:
    matrices = [None] * len(gray)
    diagnostics = [{"accepted": True, "matches": 0, "inliers": 0,
                    "inlier_ratio": 1.0, "reprojection_p50_px": 0.0,
                    "reason": "REFERENCE_FRAME"}]
    started = time.perf_counter()
    for i in range(1, len(gray)):
        matrix, diag = estimate_affine_current_to_previous(gray[i - 1], gray[i], cfg)
        matrices[i] = matrix
        diagnostics.append(diag)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return matrices, diagnostics, round(elapsed_ms, 3)


def homogeneous(matrix: np.ndarray) -> np.ndarray:
    out = np.eye(3, dtype=np.float64)
    out[:2, :] = np.asarray(matrix, dtype=np.float64)
    return out


def relative_transforms(matrices: list, query_time: int) -> list:
    transforms = [None] * len(matrices)
    transforms[query_time] = np.eye(3, dtype=np.float64)
    for i in range(query_time + 1, len(matrices)):
        previous = transforms[i - 1]
        matrix = matrices[i]
        if previous is None or matrix is None:
            transforms[i] = None
        else:
            transforms[i] = previous @ homogeneous(matrix)
    return transforms


def affine_stabilized_klt(gray: list[np.ndarray], query: dict, matrices: list,
                          cfg: dict) -> list:
    t = int(query["query_time"])
    h, w = gray[0].shape
    transforms = relative_transforms(matrices, t)
    stabilized = list(gray)
    for i in range(t + 1, len(gray)):
        if transforms[i] is None:
            stabilized[i] = np.zeros_like(gray[i])
        else:
            stabilized[i] = cv2.warpAffine(
                gray[i], transforms[i][:2, :], (w, h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            )
    ref_trace = klt_point(stabilized, query, fb=True,
                          fb_error_px=float(cfg["fb_error_px"]))
    out = [None] * len(gray)
    out[t] = [float(v) for v in query["query_xy"]]
    for i in range(t + 1, len(gray)):
        if transforms[i] is None or ref_trace[i] is None:
            continue
        inverse = np.linalg.inv(transforms[i])
        mapped = inverse @ np.asarray([ref_trace[i][0], ref_trace[i][1], 1.0])
        xy = mapped[:2] / mapped[2]
        if np.isfinite(xy).all() and 0 <= xy[0] < w and 0 <= xy[1] < h:
            out[i] = [round(float(v), 4) for v in xy]
    return out


def predict_query(gray: list[np.ndarray], query: dict, matrices: list,
                  cfg: dict) -> tuple[dict, dict]:
    predictions: dict[str, list] = {}
    timings: dict[str, float] = {}
    t = int(query["query_time"])

    started = time.perf_counter()
    predictions["stationary_point"] = (
        [None if i < t else [float(v) for v in query["query_xy"]]
         for i in range(len(gray))]
    )
    timings["stationary_point"] = round((time.perf_counter() - started) * 1000, 3)

    started = time.perf_counter()
    predictions["klt_forward"] = klt_point(
        gray, query, fb=False, fb_error_px=float(cfg["fb_error_px"])
    )
    timings["klt_forward"] = round((time.perf_counter() - started) * 1000, 3)

    started = time.perf_counter()
    base_fb = klt_point(gray, query, fb=True,
                        fb_error_px=float(cfg["fb_error_px"]))
    fb_ms = (time.perf_counter() - started) * 1000
    predictions["klt_fb"] = base_fb
    timings["klt_fb"] = round(fb_ms, 3)

    started = time.perf_counter()
    predictions["template_recover4"] = template_recover(gray, query, cfg)
    timings["template_recover4"] = round((time.perf_counter() - started) * 1000, 3)

    started = time.perf_counter()
    predictions["affine_stabilized_klt"] = affine_stabilized_klt(
        gray, query, matrices, cfg
    )
    timings["affine_stabilized_klt"] = round((time.perf_counter() - started) * 1000, 3)

    for threshold, suffix in zip(cfg["appearance_thresholds"], ("055", "070", "085")):
        started = time.perf_counter()
        predictions["appearance_fb_" + suffix] = appearance_gate(
            gray, query, base_fb, float(threshold), int(cfg["patch_size"])
        )
        gate_ms = (time.perf_counter() - started) * 1000
        timings["appearance_fb_" + suffix] = round(fb_ms + gate_ms, 3)

    for method in METHODS:
        trace = predictions[method]
        if len(trace) != len(gray):
            raise RuntimeError("KINETICS_TRACE_LENGTH_MISMATCH:" + method)
        if trace[:t] != [None] * t or trace[t] is None:
            raise RuntimeError("KINETICS_QUERY_TIME_CONTRACT:" + method)
        for point in trace:
            if point is not None and (
                len(point) != 2 or not np.isfinite(np.asarray(point, dtype=np.float64)).all()
            ):
                raise RuntimeError("KINETICS_BAD_PREDICTION_POINT:" + method)
    return predictions, timings


def scene_pixel_diagnostics(gray: list[np.ndarray], affine_diag: list[dict]) -> dict:
    sample = list(range(0, len(gray), max(1, len(gray) // 20)))
    luma = np.asarray([gray[i] for i in sample], dtype=np.uint8)
    lap = [float(cv2.Laplacian(gray[i], cv2.CV_64F).var()) for i in sample]
    orb = cv2.ORB_create(nfeatures=500, fastThreshold=8)
    keypoints = [len(orb.detect(gray[i], None)) for i in sample]
    diffs = [
        float(np.mean(cv2.absdiff(gray[i], gray[i - 1])))
        for i in range(1, len(gray), max(1, len(gray) // 20))
    ]
    accepted = [x for x in affine_diag[1:] if x.get("accepted")]
    reproj = [x["reprojection_p50_px"] for x in accepted
              if x.get("reprojection_p50_px") is not None]
    trans = [x["translation_px"] for x in accepted if x.get("translation_px") is not None]
    rotations = [x["rotation_deg"] for x in accepted if x.get("rotation_deg") is not None]
    scales = [x["scale"] for x in accepted if x.get("scale") is not None]
    return {
        "mean_luma": round(float(np.mean(luma)), 3),
        "luma_std": round(float(np.std(luma)), 3),
        "luma_p10": round(float(np.percentile(luma, 10)), 3),
        "luma_p90": round(float(np.percentile(luma, 90)), 3),
        "laplacian_variance_mean": round(float(np.mean(lap)), 3),
        "orb_keypoints_median": round(float(np.median(keypoints)), 2),
        "mean_adjacent_frame_absdiff": round(float(np.mean(diffs)), 3) if diffs else None,
        "affine_pair_acceptance": round(len(accepted) / max(1, len(affine_diag) - 1), 5),
        "affine_reprojection_p50_median_px": round(float(np.median(reproj)), 4) if reproj else None,
        "affine_translation_p50_px": round(float(np.median(trans)), 4) if trans else None,
        "affine_rotation_p50_deg": round(float(np.median(rotations)), 4) if rotations else None,
        "affine_scale_p50": round(float(np.median(scales)), 5) if scales else None,
    }


def predict(split: str, cfg: dict) -> None:
    manifest = src.audit()
    queries = doc(DATA / "queries-public.json")
    manifest_sha = sha_bytes((DATA / "manifest.json").read_bytes())
    public_sha = sha_bytes((DATA / "queries-public.json").read_bytes())
    config_sha = sha_bytes(json_bytes(cfg))
    dev_selection_sha = None
    if split == "test":
        selection_path = OUT / "dev_selection.json"
        require_pin(selection_path)
        selection = doc(selection_path)
        if (selection.get("split") != "dev" or selection.get("source_commit") != gitsha()
                or selection.get("config_sha256") != config_sha):
            raise RuntimeError("KINETICS_TEST_PREDICTION_BEFORE_DEV_SELECTION")
        dev_selection_sha = sha_bytes(selection_path.read_bytes())

    OUT.mkdir(parents=True, exist_ok=True)
    outfile = OUT / f"{split}_predictions.json"
    if outfile.exists():
        require_pin(outfile)
        saved = doc(outfile)
        if (saved.get("source_commit") != gitsha()
                or saved.get("manifest_sha256") != manifest_sha
                or saved.get("public_query_sha256") != public_sha
                or saved.get("config_sha256") != config_sha):
            raise RuntimeError("KINETICS_STALE_PREDICTION_SOURCE")
        print("KINETICS_EXISTING_PREDICTIONS_PINNED", split, flush=True)
        return

    records = []
    aux = {}
    for scene in manifest["scenes"]:
        if scene["split"] != split:
            continue
        frames = np.load(ROOT / scene["frame_file"], mmap_mode="r", allow_pickle=False)
        qrows = queries[scene["name"]]
        check_video(frames, qrows)
        gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
        matrices, affine_diag, affine_ms = estimate_affines(gray, cfg["tracker"])
        pixel_diag = scene_pixel_diagnostics(gray, affine_diag)
        scene_tracks = {}
        scene_times = {}
        for query in qrows:
            predictions, timings = predict_query(gray, query, matrices, cfg["tracker"])
            scene_tracks[query["id"]] = predictions
            scene_times[query["id"]] = timings
            print("KINETICS_CPU_QUERY_DONE", split, scene["name"], query["id"], flush=True)
        records.append({
            "name": scene["name"],
            "nframes": int(len(frames)),
            "tracks": scene_tracks,
            "cpu_query_ms": scene_times,
        })
        aux[scene["name"]] = {
            "pixel_diagnostics": pixel_diag,
            "affine_registration_ms_per_video": affine_ms,
            "affine_diagnostics": affine_diag,
            "affine_matrices_current_to_previous": [
                None if m is None else np.asarray(m).round(8).tolist() for m in matrices
            ],
        }
        print("KINETICS_SCENE_CPU_DONE", split, scene["name"],
              "affine_accept", pixel_diag["affine_pair_acceptance"],
              "registration_ms", affine_ms, flush=True)

    payload = {
        "dataset": "TAP-Vid-Kinetics natural video",
        "split": split,
        "source_commit": gitsha(),
        "manifest_sha256": manifest_sha,
        "public_query_sha256": public_sha,
        "config_sha256": config_sha,
        "dev_selection_sha256": dev_selection_sha,
        "oracle_unread_during_prediction": True,
        "methods": list(METHODS),
        "records": records,
        "scene_aux": aux,
    }
    once(outfile, payload)
    pin(outfile)
    print("KINETICS_PREDICTIONS_PINNED", split, len(records),
          sha_bytes(outfile.read_bytes()), flush=True)


def method_counts() -> dict:
    return {
        "visible": 0, "invisible": 0, "correct8": 0, "accepted_visible": 0,
        "wrong_visible_accept": 0, "visible_abstained": 0,
        "invisible_false_accept": 0, "correct_invisible_abstain": 0,
        "visibility_correct": 0, **{"tp" + str(r): 0 for r in RADII},
        "predicted_visible": 0,
    }


def metric_rates(counts: dict, latency: list[float]) -> dict:
    v, inv = counts["visible"], counts["invisible"]
    accepted = counts["accepted_visible"]
    predicted_visible = counts["predicted_visible"]
    result = dict(counts)
    result.update({
        "pck8_visible_unconditional": counts["correct8"] / v if v else None,
        "visible_coverage": accepted / v if v else None,
        "accepted_precision8": counts["correct8"] / accepted if accepted else None,
        "overall_accept_precision8": counts["correct8"] / predicted_visible if predicted_visible else None,
        "visible_false_abstain_rate": counts["visible_abstained"] / v if v else None,
        "invisible_false_accept_rate": counts["invisible_false_accept"] / inv if inv else None,
        "visibility_accuracy": counts["visibility_correct"] / (v + inv) if v + inv else None,
        "jaccard_by_radius": {
            str(r): counts["tp" + str(r)] / (v + predicted_visible - counts["tp" + str(r)])
            if (v + predicted_visible - counts["tp" + str(r)]) else None
            for r in RADII
        },
        "jaccard_mean_1_2_4_8_16": float(np.mean([
            counts["tp" + str(r)] / (v + predicted_visible - counts["tp" + str(r)])
            if (v + predicted_visible - counts["tp" + str(r)]) else 0.0
            for r in RADII
        ])) if v else None,
        "cpu_ms_per_target_frame_p50": float(np.median(latency)) if latency else None,
        "cpu_ms_per_target_frame_p95": float(np.percentile(latency, 95)) if latency else None,
    })
    return result


def background_relative_diagnostic(prediction_aux: dict, queries: list[dict],
                                   oracle_scene: dict) -> dict:
    matrices_raw = prediction_aux["affine_matrices_current_to_previous"]
    matrices = [None if x is None else np.asarray(x, dtype=np.float64) for x in matrices_raw]
    screen, residual, explained = [], [], []
    valid_points = 0
    for query in queries:
        gold = oracle_scene["tracks"][query["id"]]
        t0 = int(query["query_time"])
        transforms = relative_transforms(matrices, t0)
        qxy = np.asarray(query["query_xy"], dtype=np.float64)
        for i in range(t0 + 1, len(gold["occluded"])):
            if gold["occluded"][i] or transforms[i] is None:
                continue
            gt = np.asarray(gold["xy"][i], dtype=np.float64)
            delta = float(np.linalg.norm(gt - qxy))
            aligned_h = transforms[i] @ np.asarray([gt[0], gt[1], 1.0])
            aligned = aligned_h[:2] / aligned_h[2]
            relative = float(np.linalg.norm(aligned - qxy))
            if np.isfinite([delta, relative]).all():
                screen.append(delta)
                residual.append(relative)
                if delta > 1.0:
                    explained.append(1.0 - relative / delta)
                valid_points += 1
    diag = prediction_aux["pixel_diagnostics"]
    return {
        "affine_pair_acceptance": diag["affine_pair_acceptance"],
        "affine_reprojection_p50_median_px": diag["affine_reprojection_p50_median_px"],
        "visible_gt_points_with_valid_affine": valid_points,
        "screen_displacement_mean_px": float(np.mean(screen)) if screen else None,
        "background_relative_residual_mean_px": float(np.mean(residual)) if residual else None,
        "camera_explained_fraction_mean_unclipped": float(np.mean(explained)) if explained else None,
        "interpretation": "2-D dominant-background affine proxy only; not calibrated camera pose, depth-aware parallax or 3-D world motion",
    }


def bootstrap_scene_differences(per_scene: dict, method_a: str, method_b: str,
                                reps: int, seed: int) -> dict:
    names = sorted(per_scene)
    if len(names) < 2:
        return {"mean_difference": None, "ci95": None, "scene_count": len(names)}
    values = np.asarray([
        per_scene[n][method_a]["jaccard_mean_1_2_4_8_16"]
        - per_scene[n][method_b]["jaccard_mean_1_2_4_8_16"]
        for n in names
    ], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        samples[i] = float(np.mean(values[rng.integers(0, len(values), len(values))]))
    return {
        "mean_difference": float(np.mean(values)),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "scene_count": len(names),
        "descriptive_only": True,
    }


def score(split: str, cfg: dict) -> dict:
    manifest = src.audit()
    if (manifest.get("oracle_sha256") != sha_bytes((DATA / "oracle-private.json").read_bytes())
            or manifest.get("public_queries_sha256") != sha_bytes((DATA / "queries-public.json").read_bytes())):
        raise RuntimeError("KINETICS_FROZEN_ORACLE_OR_QUERY_DIGEST_DRIFT")
    prediction_path = OUT / f"{split}_predictions.json"
    require_pin(prediction_path)
    saved = doc(prediction_path)
    expected_manifest = sha_bytes((DATA / "manifest.json").read_bytes())
    expected_public = sha_bytes((DATA / "queries-public.json").read_bytes())
    if (saved.get("source_commit") != gitsha()
            or saved.get("config_sha256") != sha_bytes(json_bytes(cfg))
            or saved.get("manifest_sha256") != expected_manifest
            or saved.get("public_query_sha256") != expected_public
            or saved.get("oracle_unread_during_prediction") is not True
            or saved.get("methods") != list(METHODS)
            or saved.get("split") != split
            or (split == "test" and saved.get("dev_selection_sha256") != sha_bytes((OUT / "dev_selection.json").read_bytes()))):
        raise RuntimeError("KINETICS_ORACLE_SCORE_FIREWALL_FAILED")

    oracle = doc(DATA / "oracle-private.json")
    queries = doc(DATA / "queries-public.json")
    expected_names = {s["name"] for s in manifest["scenes"] if s["split"] == split}
    prediction_map = {r["name"]: r for r in saved["records"]}
    if set(prediction_map) != expected_names or len(saved["records"]) != len(expected_names):
        raise RuntimeError("KINETICS_PREDICTION_SCENE_SET_MISMATCH")

    per_scene = {}
    for name in sorted(expected_names):
        qmap = {q["id"]: q for q in queries[name]}
        gold = oracle[name]["tracks"]
        pred_scene = prediction_map[name]
        if set(qmap) != set(gold) or set(pred_scene["tracks"]) != set(gold):
            raise RuntimeError("KINETICS_QUERY_TRACK_ID_MISMATCH")
        scene_metrics = {}
        for method in METHODS:
            counts = method_counts()
            latency = []
            for qid, query in qmap.items():
                trace = pred_scene["tracks"][qid][method]
                truth = gold[qid]
                if len(trace) != len(truth["xy"]) or len(trace) != len(truth["occluded"]):
                    raise RuntimeError("KINETICS_TRACE_GOLD_LENGTH_MISMATCH")
                n_target = max(1, len(trace) - int(query["query_time"]) - 1)
                latency.append(float(pred_scene["cpu_query_ms"][qid][method]) / n_target)
                for i in range(int(query["query_time"]) + 1, len(trace)):
                    point = trace[i]
                    visible = not bool(truth["occluded"][i])
                    if point is not None:
                        xy = np.asarray(point, dtype=np.float64)
                        if xy.shape != (2,) or not np.isfinite(xy).all():
                            raise RuntimeError("KINETICS_NONFINITE_PREDICTION")
                        counts["predicted_visible"] += 1
                    if visible:
                        counts["visible"] += 1
                        if point is None:
                            counts["visible_abstained"] += 1
                        else:
                            counts["accepted_visible"] += 1
                            error = float(np.linalg.norm(
                                np.asarray(point, dtype=np.float64)
                                - np.asarray(truth["xy"][i], dtype=np.float64)
                            ))
                            counts["correct8"] += int(error <= 8.0)
                            counts["wrong_visible_accept"] += int(error > 8.0)
                            for radius in RADII:
                                counts["tp" + str(radius)] += int(error <= radius)
                    else:
                        counts["invisible"] += 1
                        if point is None:
                            counts["correct_invisible_abstain"] += 1
                        else:
                            counts["invisible_false_accept"] += 1
                    counts["visibility_correct"] += int(visible == (point is not None))
            scene_metrics[method] = metric_rates(counts, latency)
        per_scene[name] = scene_metrics

    micro = {}
    macro = {}
    for method in METHODS:
        totals = method_counts()
        for name in per_scene:
            for key in totals:
                totals[key] += per_scene[name][method][key]
        micro[method] = metric_rates(totals, [])
        rate_keys = (
            "pck8_visible_unconditional", "visible_coverage", "accepted_precision8",
            "overall_accept_precision8", "visible_false_abstain_rate", "invisible_false_accept_rate",
            "visibility_accuracy", "jaccard_mean_1_2_4_8_16",
            "cpu_ms_per_target_frame_p50", "cpu_ms_per_target_frame_p95",
        )
        macro[method] = {
            key: float(np.mean([
                per_scene[name][method][key] for name in per_scene
                if per_scene[name][method][key] is not None
            ])) if any(per_scene[name][method][key] is not None for name in per_scene)
            else None
            for key in rate_keys
        }

    aux = saved["scene_aux"]
    background = {
        name: background_relative_diagnostic(
            aux[name], queries[name], oracle[name]
        )
        for name in sorted(expected_names)
    }
    cfg_eval = cfg["evaluation"]
    contrasts = {
        "klt_fb_minus_klt_forward": bootstrap_scene_differences(
            per_scene, "klt_fb", "klt_forward",
            int(cfg_eval["bootstrap_replicates"]), int(cfg_eval["bootstrap_seed"])
        ),
        "affine_minus_klt_fb": bootstrap_scene_differences(
            per_scene, "affine_stabilized_klt", "klt_fb",
            int(cfg_eval["bootstrap_replicates"]), int(cfg_eval["bootstrap_seed"]) + 1
        ),
        "appearance_070_minus_klt_fb": bootstrap_scene_differences(
            per_scene, "appearance_fb_070", "klt_fb",
            int(cfg_eval["bootstrap_replicates"]), int(cfg_eval["bootstrap_seed"]) + 2
        ),
    }
    result = {
        "dataset": "TAP-Vid-Kinetics natural YouTube/Kinetics video",
        "source_commit": gitsha(),
        "split": split,
        "config_sha256": sha_bytes(json_bytes(cfg)),
        "manifest_sha256": expected_manifest,
        "prediction_sha256": sha_bytes(prediction_path.read_bytes()),
        "scene_count": len(expected_names),
        "query_count": sum(len(queries[n]) for n in expected_names),
        "point_threshold_px": int(cfg_eval["point_threshold_px"]),
        "primary_metrics": "scene-macro and micro PCK@8, visible coverage, accepted precision, invisible false accepts, TAP-Vid Jaccard",
        "methods": list(METHODS),
        "micro": micro,
        "scene_macro": macro,
        "per_scene": per_scene,
        "scene_pixel_diagnostics": {
            name: aux[name]["pixel_diagnostics"] for name in sorted(expected_names)
        },
        "background_relative_diagnostics": background,
        "paired_scene_bootstrap": contrasts,
        "affine_registration_ms_per_video": {
            name: aux[name]["affine_registration_ms_per_video"] for name in sorted(expected_names)
        },
        "limitations": [
            "Kinetics videos are natural YouTube clips, but this 12-video stratified subset is small and not a population estimate.",
            "TAP-Vid Kinetics has point/occlusion labels but no calibrated camera pose or depth; background-relative values are 2-D affine proxies.",
            "CPU methods see all intermediate 25-fps frames; the VLM baseline sees only two fixed RGB images per query.",
            "Only DEV is used for tracker selection and VLM gate decisions; TEST predictions are sealed before TEST labels are scored once.",
        ],
    }
    result_path = OUT / f"{split}_results.json"
    once(result_path, result)
    print("KINETICS_SCORED", split, json.dumps({
        "scenes": len(expected_names),
        "macro": {m: {k: macro[m][k] for k in (
            "pck8_visible_unconditional", "visible_coverage",
            "accepted_precision8", "invisible_false_accept_rate",
            "jaccard_mean_1_2_4_8_16", "cpu_ms_per_target_frame_p50",
        )} for m in METHODS},
    }, sort_keys=True), flush=True)
    return result


def select_dev(cfg: dict) -> dict:
    path = OUT / "dev_results.json"
    if not path.is_file():
        raise RuntimeError("KINETICS_DEV_RESULTS_REQUIRED")
    results = doc(path)
    if (results.get("split") != "dev" or results.get("source_commit") != gitsha()
            or results.get("config_sha256") != sha_bytes(json_bytes(cfg))):
        raise RuntimeError("KINETICS_DEV_SELECTION_SOURCE_MISMATCH")
    rules = cfg["dev_selection"]
    eligible = []
    for method in rules["candidate_methods"]:
        metric = results["scene_macro"][method]
        coverage = metric["visible_coverage"]
        false_accept = metric["invisible_false_accept_rate"]
        if (coverage is not None and false_accept is not None
                and coverage >= float(rules["min_visible_coverage"])
                and false_accept <= float(rules["max_invisible_false_accept_rate"])):
            eligible.append(method)
    if eligible:
        selected = max(eligible, key=lambda m: (
            results["scene_macro"][m]["jaccard_mean_1_2_4_8_16"],
            results["scene_macro"][m]["visible_coverage"],
        ))
        reason = "best DEV scene-macro Jaccard among candidates meeting both predeclared constraints"
    else:
        selected = rules["fallback_method"]
        reason = "no candidate met both DEV constraints; predeclared fallback used"
    payload = {
        "source_commit": gitsha(),
        "split": "dev",
        "config_sha256": sha_bytes(json_bytes(cfg)),
        "dev_results_sha256": sha_bytes(path.read_bytes()),
        "eligible_candidates": eligible,
        "selected_method": selected,
        "reason": reason,
        "rules": rules,
        "selection_source": "DEV only",
    }
    out = OUT / "dev_selection.json"
    once(out, payload)
    pin(out)
    print("KINETICS_DEV_SELECTED", selected, reason, flush=True)
    return payload


def build_report() -> Path:
    dev = doc(OUT / "dev_results.json")
    test = doc(OUT / "test_results.json")
    selection = doc(OUT / "dev_selection.json")
    rows = []
    for method in METHODS:
        m = test["scene_macro"][method]
        def fmt(value):
            return "n/a" if value is None else f"{value:.3f}"
        rows.append(
            f"| {method} | {fmt(m['pck8_visible_unconditional'])} | "
            f"{fmt(m['visible_coverage'])} | {fmt(m['accepted_precision8'])} | "
            f"{fmt(m['invisible_false_accept_rate'])} | "
            f"{fmt(m['jaccard_mean_1_2_4_8_16'])} | "
            f"{fmt(m['cpu_ms_per_target_frame_p50'])} |"
        )
    vlm_dev = doc(OUT / "dev_vlm_results.json")
    vlm_gate = doc(OUT / "dev_vlm_gate.json")
    vlm_test = doc(OUT / "test_vlm_results.json")
    scene_rows = []
    for name in sorted(test["scene_pixel_diagnostics"]):
        d = test["scene_pixel_diagnostics"][name]
        b = test["background_relative_diagnostics"][name]
        def fmt(value):
            return "n/a" if value is None else f"{value:.3f}"
        scene_rows.append(
            f"| {name} | {fmt(d['mean_luma'])} | {fmt(d['laplacian_variance_mean'])} | "
            f"{fmt(d['orb_keypoints_median'])} | {fmt(d['affine_pair_acceptance'])} | "
            f"{fmt(b['screen_displacement_mean_px'])} | "
            f"{fmt(b['background_relative_residual_mean_px'])} |"
        )
    vd = vlm_dev["vlm"]
    vlm_lines = [
        f"- DEV VLM gate: {vlm_gate['status']}; checks: {vlm_gate['checks']}.",
        f"- DEV VLM balanced accuracy / coverage / invalid rate: {vd['balanced_accuracy']} / {vd['coverage']} / {vd['invalid_rate']}.",
    ]
    if vlm_test.get("status") == "SCORED":
        vt = vlm_test["vlm"]
        vlm_lines.append(
            f"- TEST VLM balanced accuracy / coverage / invalid rate: {vt['balanced_accuracy']} / {vt['coverage']} / {vt['invalid_rate']}."
        )
    else:
        vlm_lines.append("- TEST VLM inference was skipped because the predeclared DEV gate blocked it; TEST VLM labels were not opened.")
    text = [
        "# Batch42 — TAP-Vid-Kinetics natural-video tracking and visibility",
        "",
        f"- Frozen source commit: {gitsha()}",
        f"- DEV / TEST videos: {dev['scene_count']} / {test['scene_count']}",
        f"- DEV / TEST point queries: {dev['query_count']} / {test['query_count']}",
        f"- DEV-selected method: {selection['selected_method']} ({selection['reason']}).",
        "- Source clips are local only; no video, frame array, annotation oracle or model weight is published.",
        "",
        "## TEST CPU results (scene macro)",
        "",
        "| Method | PCK@8 visible | Visible coverage | Accepted precision | Invisible false accept | Jaccard | CPU P50 ms/frame |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Light, texture and camera-motion strata",
        "",
        "| Scene | Mean luma | Laplacian variance | ORB keypoints | Affine pair acceptance | Screen motion px | Background-relative residual px |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *scene_rows,
        "",
        "Background-relative motion is a 2-D dominant-background affine proxy, not calibrated camera pose, depth-aware parallax or 3-D world motion.",
        "",
        "## SmolVLM2-500M visibility baseline",
        "",
        *vlm_lines,
        "",
        "## Interpretation limits",
        "",
        "- Twelve videos across twelve action classes are an exploratory independent draw, not a broad-domain generalization claim.",
        "- TEST results are descriptive only; thresholds and method choice were fixed from DEV before TEST predictions.",
        "- CPU methods use all intermediate 25-fps frames; the VLM receives only a query image and one later frame.",
        "- Annotation CC BY 4.0 does not replace the individual terms of the source YouTube videos.",
        "",
    ]
    path = OUT / "BATCH42_KINETICS_REPORT.md"
    path.write_text("\n".join(text), encoding="utf-8")
    print("KINETICS_REPORT_WRITTEN", path, flush=True)
    return path


def final_audit(cfg: dict) -> None:
    manifest = src.audit()
    for split in ("dev", "test"):
        p = OUT / f"{split}_predictions.json"
        require_pin(p)
        saved = doc(p)
        if (saved.get("source_commit") != gitsha()
                or saved.get("manifest_sha256") != sha_bytes((DATA / "manifest.json").read_bytes())
                or saved.get("config_sha256") != sha_bytes(json_bytes(cfg))
                or saved.get("oracle_unread_during_prediction") is not True):
            raise RuntimeError("KINETICS_FINAL_PREDICTION_AUDIT_FAILED:" + split)
        result = doc(OUT / f"{split}_results.json")
        if (result.get("prediction_sha256") != sha_bytes(p.read_bytes())
                or result.get("source_commit") != gitsha()
                or result.get("config_sha256") != sha_bytes(json_bytes(cfg))):
            raise RuntimeError("KINETICS_FINAL_SCORE_PIN_MISMATCH:" + split)
    require_pin(OUT / "dev_selection.json")
    test_saved = doc(OUT / "test_predictions.json")
    if test_saved.get("dev_selection_sha256") != sha_bytes((OUT / "dev_selection.json").read_bytes()):
        raise RuntimeError("KINETICS_TEST_DEV_SELECTION_SHA_MISMATCH")
    if not (OUT / "BATCH42_KINETICS_REPORT.md").is_file():
        build_report()
    if len(manifest["scenes"]) != 12:
        raise RuntimeError("KINETICS_FINAL_SCENE_COUNT_MISMATCH")
    print("KINETICS_BATCH42_FINAL_AUDIT_PASS", gitsha(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=(
        "predict-dev", "score-dev", "select-dev", "predict-test", "score-test",
        "report", "final-audit",
    ))
    parser.add_argument("--config", default="configs/kinetics_batch42.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    action = args.action
    if action.startswith("predict-"):
        predict(action.split("-")[1], cfg)
    elif action.startswith("score-"):
        score(action.split("-")[1], cfg)
    elif action == "select-dev":
        select_dev(cfg)
    elif action == "report":
        build_report()
    else:
        final_audit(cfg)


if __name__ == "__main__":
    main()
