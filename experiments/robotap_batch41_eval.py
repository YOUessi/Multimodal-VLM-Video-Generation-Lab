"""Batch41 CPU point tracking, explicit occlusion scoring and 2-D camera diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

import robotap_batch41_data as src

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/robotap_batch41.json"
METHODS = (
    "stationary_point", "klt_forward", "klt_fb",
    "template_recover4", "affine_stabilized_klt",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha_file(path: Path) -> str:
    return src.sha_file(path)


def pin(path: Path) -> None:
    tag = path.with_suffix(path.suffix + ".sha256")
    src.atomic_once(tag, (sha_file(path) + "\n").encode())


def require_pin(path: Path) -> None:
    tag = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not tag.is_file():
        raise RuntimeError("ROBOTAP_MISSING_PREDICTION_PIN")
    if tag.read_text().strip() != sha_file(path):
        raise RuntimeError("ROBOTAP_PREDICTION_CHANGED_AFTER_SEAL")


def write_once(path: Path, payload: dict) -> None:
    src.atomic_once(path, src.json_bytes(payload))


def current_sha() -> str:
    status = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True, text=True, timeout=10,
    )
    if status.returncode or status.stdout.strip():
        raise RuntimeError("ROBOTAP_SOURCE_TREE_NOT_CLEAN")
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True, timeout=10,
    ).strip()


def patch_ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32) - float(a.mean())
    b = b.astype(np.float32) - float(b.mean())
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not np.isfinite(denominator) or denominator < 1e-6:
        return -1.0
    score = float(np.sum(a * b) / denominator)
    return score if np.isfinite(score) else -1.0


def crop_patch(gray: np.ndarray, xy: np.ndarray, size: int) -> np.ndarray:
    half = size // 2
    padded = cv2.copyMakeBorder(
        gray, half + 2, half + 2, half + 2, half + 2, cv2.BORDER_REFLECT_101
    )
    x = int(round(float(xy[0]))) + half + 2
    y = int(round(float(xy[1]))) + half + 2
    return padded[y-half:y+half+1, x-half:x+half+1]


def klt_point(gray: list[np.ndarray], query: dict, *, use_fb: bool,
              fb_error_px: float) -> tuple[list, list]:
    query_time = int(query["query_time"])
    point = np.asarray(query["query_xy"], dtype=np.float32).reshape(1, 1, 2)
    trace = [None] * len(gray)
    visible = [False] * len(gray)
    trace[query_time] = [float(v) for v in point.reshape(2)]
    visible[query_time] = True
    height, width = gray[0].shape
    for frame_index in range(query_time + 1, len(gray)):
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            gray[frame_index - 1], gray[frame_index], point, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if nxt is None or status is None or int(status.reshape(-1)[0]) != 1:
            break
        xy = nxt.reshape(2)
        if (not np.isfinite(xy).all() or xy[0] < 0 or xy[0] >= width
                or xy[1] < 0 or xy[1] >= height):
            break
        if use_fb:
            back, back_status, _ = cv2.calcOpticalFlowPyrLK(
                gray[frame_index], gray[frame_index - 1], nxt, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            if (back is None or back_status is None
                    or int(back_status.reshape(-1)[0]) != 1
                    or not np.isfinite(back).all()
                    or float(np.linalg.norm(back.reshape(2) - point.reshape(2))) > fb_error_px):
                break
        trace[frame_index] = [round(float(v), 3) for v in xy]
        visible[frame_index] = True
        point = nxt
    return trace, visible


def template_recover(gray: list[np.ndarray], query: dict, cfg: dict) -> tuple[list, list]:
    query_time = int(query["query_time"])
    anchor = np.asarray(query["query_xy"], dtype=np.float64)
    size = int(cfg["patch_size"])
    radius = int(cfg["template_search_radius_px"])
    threshold = float(cfg["template_ncc_min"])
    max_misses = int(cfg["template_max_consecutive_misses"])
    trace = [None] * len(gray)
    visible = [False] * len(gray)
    trace[query_time] = anchor.tolist()
    visible[query_time] = True
    template = crop_patch(gray[query_time], anchor, size)
    if template.shape != (size, size) or float(template.std()) < 5.0:
        return trace, visible
    half = size // 2
    last = anchor.copy()
    misses = 0
    height, width = gray[0].shape
    for frame_index in range(query_time + 1, len(gray)):
        padded = cv2.copyMakeBorder(
            gray[frame_index], radius + half + 2, radius + half + 2,
            radius + half + 2, radius + half + 2, cv2.BORDER_REFLECT_101,
        )
        cx = int(round(float(last[0]))) + radius + half + 2
        cy = int(round(float(last[1]))) + radius + half + 2
        roi = padded[cy-radius-half:cy+radius+half+1,
                     cx-radius-half:cx+radius+half+1]
        if roi.shape[0] < size or roi.shape[1] < size:
            misses += 1
            if misses > max_misses:
                break
            continue
        response = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, top_left = cv2.minMaxLoc(response)
        found = last + np.asarray(
            [top_left[0] - radius, top_left[1] - radius], dtype=np.float64
        )
        if (not np.isfinite(score) or score < threshold
                or found[0] < 0 or found[0] >= width
                or found[1] < 0 or found[1] >= height):
            misses += 1
            if misses > max_misses:
                break
            continue
        last = found
        misses = 0
        trace[frame_index] = [round(float(v), 3) for v in found]
        visible[frame_index] = True
    return trace, visible


def estimate_affine_current_to_previous(previous: np.ndarray, current: np.ndarray,
                                         cfg: dict) -> tuple[np.ndarray | None, dict]:
    orb = cv2.ORB_create(nfeatures=700, scaleFactor=1.2, nlevels=8)
    kp_prev, desc_prev = orb.detectAndCompute(previous, None)
    kp_curr, desc_curr = orb.detectAndCompute(current, None)
    if desc_prev is None or desc_curr is None:
        return None, {"accepted": False, "matches": 0, "inliers": 0,
                      "inlier_ratio": 0.0, "reprojection_p50_px": None}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_curr, desc_prev, k=2)
    good = [a for a, b in pairs if a.distance < 0.75 * b.distance]
    if len(good) < int(cfg["affine_min_matches"]):
        return None, {"accepted": False, "matches": len(good), "inliers": 0,
                      "inlier_ratio": 0.0, "reprojection_p50_px": None}
    source = np.float32([kp_curr[m.queryIdx].pt for m in good])
    target = np.float32([kp_prev[m.trainIdx].pt for m in good])
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC,
        ransacReprojThreshold=float(cfg["affine_ransac_reprojection_px"]),
        maxIters=2000, confidence=0.99, refineIters=10,
    )
    if matrix is None or inlier_mask is None:
        return None, {"accepted": False, "matches": len(good), "inliers": 0,
                      "inlier_ratio": 0.0, "reprojection_p50_px": None}
    inliers = inlier_mask.reshape(-1).astype(bool)
    count = int(np.count_nonzero(inliers))
    ratio = count / max(1, len(good))
    predicted = cv2.transform(source.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    residual = np.linalg.norm(predicted[inliers] - target[inliers], axis=1) if count else np.array([])
    p50 = float(np.median(residual)) if len(residual) else None
    accepted = (
        count >= int(cfg["affine_min_inliers"])
        and ratio >= float(cfg["affine_min_inlier_ratio"])
        and p50 is not None and np.isfinite(p50)
    )
    diag = {
        "accepted": bool(accepted), "matches": len(good), "inliers": count,
        "inlier_ratio": round(float(ratio), 5),
        "reprojection_p50_px": round(float(p50), 4) if p50 is not None else None,
    }
    return (matrix.astype(np.float64) if accepted else None), diag


def estimate_affines(gray: list[np.ndarray], cfg: dict) -> tuple[list, list]:
    matrices = [np.eye(3, dtype=np.float64)]
    diagnostics = [{"accepted": True, "matches": 0, "inliers": 0,
                    "inlier_ratio": 1.0, "reprojection_p50_px": 0.0}]
    for index in range(1, len(gray)):
        matrix, diag = estimate_affine_current_to_previous(
            gray[index - 1], gray[index], cfg
        )
        if matrix is None:
            matrices.append(None)
        else:
            homogeneous = np.eye(3, dtype=np.float64)
            homogeneous[:2, :] = matrix
            matrices.append(homogeneous)
        diagnostics.append(diag)
    return matrices, diagnostics


def affine_stabilized_klt(gray: list[np.ndarray], query: dict, matrices: list,
                          cfg: dict) -> tuple[list, list]:
    query_time = int(query["query_time"])
    trace = [None] * len(gray)
    visible = [False] * len(gray)
    trace[query_time] = [float(v) for v in query["query_xy"]]
    visible[query_time] = True
    height, width = gray[0].shape
    transforms = [np.eye(3, dtype=np.float64)]
    valid_chain = [True]
    for index in range(1, len(gray)):
        previous_valid = valid_chain[-1]
        matrix = matrices[index]
        if matrix is None or not previous_valid:
            transforms.append(transforms[-1].copy())
            valid_chain.append(False)
        else:
            transforms.append(transforms[-1] @ matrix)
            valid_chain.append(True)
    if not valid_chain[query_time]:
        return trace, visible
    stabilized = []
    for index, frame in enumerate(gray):
        if valid_chain[index]:
            stabilized.append(cv2.warpPerspective(
                frame, transforms[index], (width, height),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            ))
        else:
            stabilized.append(np.zeros_like(frame))
    q = np.asarray(query["query_xy"], dtype=np.float64)
    q_ref_h = transforms[query_time] @ np.asarray([q[0], q[1], 1.0])
    q_ref = q_ref_h[:2] / q_ref_h[2]
    ref_query = {"query_time": query_time, "query_xy": q_ref.tolist()}
    ref_trace, ref_visible = klt_point(
        stabilized, ref_query, use_fb=True,
        fb_error_px=float(cfg["fb_error_px"]),
    )
    for index in range(query_time + 1, len(gray)):
        if not valid_chain[index] or not ref_visible[index] or ref_trace[index] is None:
            continue
        inverse = np.linalg.inv(transforms[index])
        p = np.asarray(ref_trace[index] + [1.0], dtype=np.float64)
        original_h = inverse @ p
        xy = original_h[:2] / original_h[2]
        if (np.isfinite(xy).all() and 0 <= xy[0] < width and 0 <= xy[1] < height):
            trace[index] = [round(float(v), 3) for v in xy]
            visible[index] = True
    return trace, visible


def predict_query(gray: list[np.ndarray], query: dict, matrices: list,
                  cfg: dict) -> tuple[dict, dict]:
    predictions = {}
    timings = {}
    for method in METHODS:
        started = time.perf_counter()
        if method == "stationary_point":
            xy = [None] * len(gray)
            visible = [False] * len(gray)
            for index in range(int(query["query_time"]), len(gray)):
                xy[index] = [float(v) for v in query["query_xy"]]
                visible[index] = True
        elif method == "klt_forward":
            xy, visible = klt_point(
                gray, query, use_fb=False, fb_error_px=float(cfg["fb_error_px"])
            )
        elif method == "klt_fb":
            xy, visible = klt_point(
                gray, query, use_fb=True, fb_error_px=float(cfg["fb_error_px"])
            )
        elif method == "template_recover4":
            xy, visible = template_recover(gray, query, cfg)
        else:
            xy, visible = affine_stabilized_klt(gray, query, matrices, cfg)
        timings[method] = round(1000 * (time.perf_counter() - started), 3)
        if len(xy) != len(gray) or len(visible) != len(gray):
            raise RuntimeError("ROBOTAP_TRACE_LENGTH_DRIFT")
        if any(bool(v) != (p is not None) for v, p in zip(visible, xy)):
            raise RuntimeError("ROBOTAP_VISIBLE_COORDINATE_CONTRACT_DRIFT")
        predictions[method] = {"xy": xy, "visible": visible}
    return predictions, timings


def predict(split: str, cfg: dict) -> None:
    manifest = src.audit(cfg)
    queries = load_json(ROOT / cfg["dataset_dir"] / "batch41_frozen/queries-public.json")
    frozen = ROOT / cfg["dataset_dir"] / "batch41_frozen"
    output_dir = ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "dev_selection.json"
    selection = None
    if split == "test":
        if not selection_path.is_file():
            raise RuntimeError("ROBOTAP_DEV_SELECTION_MISSING")
        selection = load_json(selection_path)
        if (selection.get("source_commit") != current_sha()
                or selection.get("status") != "selected_from_dev_only"):
            raise RuntimeError("ROBOTAP_DEV_SELECTION_SOURCE_DRIFT")
    output = output_dir / f"{split}_predictions.json"
    if output.exists():
        require_pin(output)
        saved = load_json(output)
        expected_selection_sha = sha_file(selection_path) if selection else None
        if (saved.get("source_commit") != current_sha()
                or saved.get("split") != split
                or saved.get("manifest_sha256") != sha_file(frozen / "manifest.json")
                or saved.get("dev_selection_sha256") != expected_selection_sha):
            raise RuntimeError("ROBOTAP_EXISTING_PREDICTION_STALE")
        return
    rows = [scene for scene in manifest["scenes"] if scene["split"] == split]
    records = []
    for scene in rows:
        frames = np.load(frozen / scene["frames"], mmap_mode="r", allow_pickle=False)
        rgb = [np.asarray(frame) for frame in frames]
        gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in rgb]
        matrices, affine_diag = estimate_affines(gray, cfg["tracker"])
        query_records = []
        for query in queries[scene["name"]]:
            predictions, timings = predict_query(gray, query, matrices, cfg["tracker"])
            query_records.append({
                "query_id": query["id"], "query_time": int(query["query_time"]),
                "predictions": predictions, "cpu_ms": timings,
            })
            print("ROBOTAP_CPU_QUERY_DONE", split, scene["name"], query["id"], flush=True)
        records.append({
            "name": scene["name"], "nframes": len(gray),
            "queries": query_records,
            "affine_matrices_current_to_previous": [
                matrix.tolist() if matrix is not None else None for matrix in matrices
            ],
            "affine_diagnostics": affine_diag,
        })
    payload = {
        "dataset": "RoboTAP synthetic robotics",
        "split": split, "source_commit": current_sha(),
        "manifest_sha256": sha_file(frozen / "manifest.json"),
        "public_query_sha256": sha_file(frozen / "queries-public.json"),
        "dev_selection_sha256": sha_file(selection_path) if selection else None,
        "dev_selected_method": selection.get("selected_method") if selection else None,
        "oracle_unread_during_prediction": True,
        "methods": list(METHODS), "records": records,
    }
    write_once(output, payload)
    pin(output)
    print("ROBOTAP_CPU_PREDICTIONS_PINNED", split, len(records), sha_file(output), flush=True)


def _empty_counts() -> dict:
    return {
        "eval_total": 0, "visible_total": 0, "occluded_total": 0,
        "visible_accepted": 0, "correct8": 0, "wrong_visible_accept": 0,
        "visible_abstained": 0, "occluded_false_accept": 0,
        "occluded_correct_abstain": 0, "occlusion_correct": 0,
    }


def _metrics(counts: dict, cpu_ms: list[float]) -> dict:
    visible = counts["visible_total"]
    occluded = counts["occluded_total"]
    accepted = counts["visible_accepted"] + counts["occluded_false_accept"]
    union = visible + counts["wrong_visible_accept"] + counts["occluded_false_accept"]
    return {
        **counts,
        "pck8_visible_unconditional": counts["correct8"] / visible if visible else None,
        "jaccard8": counts["correct8"] / union if union else None,
        "visible_coverage": counts["visible_accepted"] / visible if visible else None,
        "accepted_precision8": counts["correct8"] / accepted if accepted else None,
        "occluded_false_accept_rate": (
            counts["occluded_false_accept"] / occluded if occluded else None
        ),
        "visible_false_abstain_rate": (
            counts["visible_abstained"] / visible if visible else None
        ),
        "occlusion_accuracy": (
            counts["occlusion_correct"] / counts["eval_total"]
            if counts["eval_total"] else None
        ),
        "cpu_p50_ms": float(np.percentile(cpu_ms, 50)) if cpu_ms else None,
        "cpu_p95_ms": float(np.percentile(cpu_ms, 95)) if cpu_ms else None,
    }


def _score_one_scene(scene: dict, prediction: dict, gold_scene: dict,
                     threshold: float) -> tuple[dict, dict]:
    counts = {method: _empty_counts() for method in METHODS}
    timings = {method: [] for method in METHODS}
    tracks = gold_scene["tracks"]
    for query_record in prediction["queries"]:
        query_id = query_record["query_id"]
        if query_id not in tracks:
            raise RuntimeError("ROBOTAP_SCORE_QUERY_NOT_IN_ORACLE")
        truth = tracks[query_id]
        gt_xy = truth["xy"]
        gt_occ = truth["occluded"]
        query_time = int(query_record["query_time"])
        for method in METHODS:
            proposal = query_record["predictions"][method]
            pred_xy = proposal["xy"]
            pred_visible = proposal["visible"]
            if len(pred_xy) != len(gt_xy) or len(pred_visible) != len(gt_occ):
                raise RuntimeError("ROBOTAP_SCORE_TRACE_LENGTH_MISMATCH")
            c = counts[method]
            timings[method].append(float(query_record["cpu_ms"][method]))
            for index in range(query_time + 1, len(gt_occ)):
                gt_visible = not bool(gt_occ[index])
                accepted = bool(pred_visible[index])
                c["eval_total"] += 1
                c["visible_total"] += int(gt_visible)
                c["occluded_total"] += int(not gt_visible)
                c["occlusion_correct"] += int(accepted == gt_visible)
                if gt_visible:
                    if accepted:
                        c["visible_accepted"] += 1
                        xy = np.asarray(pred_xy[index], dtype=np.float64)
                        target = np.asarray(gt_xy[index], dtype=np.float64)
                        error = float(np.linalg.norm(xy - target))
                        if error <= threshold:
                            c["correct8"] += 1
                        else:
                            c["wrong_visible_accept"] += 1
                    else:
                        c["visible_abstained"] += 1
                elif accepted:
                    c["occluded_false_accept"] += 1
                else:
                    c["occluded_correct_abstain"] += 1
    metrics = {
        method: _metrics(counts[method], timings[method]) for method in METHODS
    }
    return metrics, counts


def _background_relative_diagnostic(prediction: dict, gold_scene: dict) -> dict:
    matrices = prediction["affine_matrices_current_to_previous"]
    screen_displacements = []
    residual_displacements = []
    used = 0
    for query_record in prediction["queries"]:
        truth = gold_scene["tracks"][query_record["query_id"]]
        points = truth["xy"]
        occluded = truth["occluded"]
        for index in range(1, len(points)):
            if occluded[index - 1] or occluded[index] or matrices[index] is None:
                continue
            current_to_previous = np.asarray(matrices[index], dtype=np.float64)
            try:
                previous_to_current = np.linalg.inv(current_to_previous)
            except np.linalg.LinAlgError:
                continue
            previous_xy = np.asarray(points[index - 1], dtype=np.float64)
            current_xy = np.asarray(points[index], dtype=np.float64)
            projected = previous_to_current @ np.asarray(
                [previous_xy[0], previous_xy[1], 1.0], dtype=np.float64
            )
            projected = projected[:2] / projected[2]
            screen_displacements.append(float(np.linalg.norm(current_xy - previous_xy)))
            residual_displacements.append(float(np.linalg.norm(current_xy - projected)))
            used += 1
    return {
        "valid_visible_transitions": used,
        "screen_displacement_mean_px": (
            float(np.mean(screen_displacements)) if screen_displacements else None
        ),
        "background_affine_residual_mean_px": (
            float(np.mean(residual_displacements)) if residual_displacements else None
        ),
        "background_affine_residual_p50_px": (
            float(np.percentile(residual_displacements, 50)) if residual_displacements else None
        ),
        "interpretation": "2-D background-relative proxy only; not calibrated world motion",
    }


def score(split: str, cfg: dict) -> dict:
    manifest = src.audit(cfg)
    frozen = ROOT / cfg["dataset_dir"] / "batch41_frozen"
    output_dir = ROOT / cfg["output_dir"]
    prediction_path = output_dir / f"{split}_predictions.json"
    require_pin(prediction_path)
    saved = load_json(prediction_path)
    selection_path = output_dir / "dev_selection.json"
    selection = load_json(selection_path) if split == "test" else None
    if split == "test" and (
            selection is None
            or selection.get("source_commit") != current_sha()
            or selection.get("status") != "selected_from_dev_only"):
        raise RuntimeError("ROBOTAP_TEST_SELECTION_MISSING_OR_STALE")
    expected_selection_sha = sha_file(selection_path) if selection else None
    if (saved.get("source_commit") != current_sha()
            or saved.get("manifest_sha256") != sha_file(frozen / "manifest.json")
            or saved.get("public_query_sha256") != sha_file(frozen / "queries-public.json")
            or saved.get("dev_selection_sha256") != expected_selection_sha
            or saved.get("oracle_unread_during_prediction") is not True
            or saved.get("methods") != list(METHODS)
            or saved.get("split") != split):
        raise RuntimeError("ROBOTAP_SCORE_SOURCE_OR_ORACLE_FIREWALL_DRIFT")
    gold = src.load_gold(cfg)
    scene_map = {scene["name"]: scene for scene in manifest["scenes"] if scene["split"] == split}
    prediction_map = {record["name"]: record for record in saved["records"]}
    if set(scene_map) != set(prediction_map):
        raise RuntimeError("ROBOTAP_SCORE_SCENE_COVERAGE_DRIFT")
    aggregate = {method: _empty_counts() for method in METHODS}
    timing = {method: [] for method in METHODS}
    per_scene = {}
    affine_valid = []
    affine_residual = []
    for name, scene in scene_map.items():
        metrics, counts = _score_one_scene(
            scene, prediction_map[name], gold[name],
            float(cfg["evaluation"]["point_threshold_px"]),
        )
        per_scene[name] = metrics
        for method in METHODS:
            for key, value in counts[method].items():
                aggregate[method][key] += value
            timing[method].extend(
                float(q["cpu_ms"][method]) for q in prediction_map[name]["queries"]
            )
        diag = prediction_map[name]["affine_diagnostics"]
        if len(diag) > 1:
            affine_valid.extend(float(x["accepted"]) for x in diag[1:])
        relative = _background_relative_diagnostic(prediction_map[name], gold[name])
        if relative["background_affine_residual_mean_px"] is not None:
            affine_residual.append(relative["background_affine_residual_mean_px"])
        per_scene[name]["background_relative"] = relative
    summary = {
        method: _metrics(aggregate[method], timing[method]) for method in METHODS
    }
    result = {
        "dataset": "RoboTAP synthetic robotics",
        "split": split, "source_commit": current_sha(),
        "manifest_sha256": sha_file(frozen / "manifest.json"),
        "prediction_sha256": sha_file(prediction_path),
        "dev_selection_sha256": expected_selection_sha,
        "nscenes": len(scene_map),
        "nqueries": sum(len(r["queries"]) for r in saved["records"]),
        "metrics": summary, "per_scene": per_scene,
        "background_affine": {
            "pair_acceptance_rate": float(np.mean(affine_valid)) if affine_valid else None,
            "mean_scene_residual_px": float(np.mean(affine_residual)) if affine_residual else None,
            "interpretation": "2-D background-relative proxy only; no camera pose or world-motion claim",
        },
        "selection": selection,
    }
    result_path = output_dir / f"{split}_results.json"
    write_once(result_path, result)
    print("ROBOTAP_SCORED", split, json.dumps(summary, sort_keys=True), flush=True)
    return result


def select_dev(cfg: dict) -> dict:
    output_dir = ROOT / cfg["output_dir"]
    path = output_dir / "dev_results.json"
    if not path.is_file():
        raise RuntimeError("ROBOTAP_DEV_RESULTS_MISSING")
    result = load_json(path)
    if result.get("source_commit") != current_sha() or result.get("split") != "dev":
        raise RuntimeError("ROBOTAP_DEV_RESULTS_SOURCE_DRIFT")
    rule = cfg["dev_selection"]
    eligible = []
    for method in rule["candidate_methods"]:
        metrics = result["metrics"][method]
        coverage = metrics["visible_coverage"]
        false_accept = metrics["occluded_false_accept_rate"]
        if (coverage is not None and coverage >= float(rule["min_visible_coverage"])
                and false_accept is not None
                and false_accept <= float(rule["max_occluded_false_accept_rate"])):
            eligible.append(method)
    if eligible:
        selected = max(
            eligible,
            key=lambda method: (
                result["metrics"][method]["jaccard8"],
                result["metrics"][method]["visible_coverage"],
            ),
        )
        reason = "best DEV Jaccard@8 among candidates satisfying coverage and occlusion-risk constraints"
    else:
        selected = rule["fallback_method"]
        reason = "no candidate met both predeclared DEV constraints; use frozen fallback"
    doc = {
        "status": "selected_from_dev_only",
        "source_commit": current_sha(),
        "dev_results_sha256": sha_file(path),
        "selected_method": selected,
        "eligible_candidates": eligible,
        "rule": rule,
        "reason": reason,
        "test_labels_or_predictions_read": False,
    }
    write_once(output_dir / "dev_selection.json", doc)
    print("ROBOTAP_DEV_SELECTED", selected, reason, flush=True)
    return doc


def bootstrap_ci(result: dict, method_a: str, method_b: str,
                 replicates: int, seed: int) -> list | None:
    scenes = list(result["per_scene"].values())
    differences = []
    if len(scenes) < 2:
        return None
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        sample = [scenes[i] for i in rng.integers(0, len(scenes), len(scenes))]
        a = [x[method_a]["jaccard8"] for x in sample if x[method_a]["jaccard8"] is not None]
        b = [x[method_b]["jaccard8"] for x in sample if x[method_b]["jaccard8"] is not None]
        if a and b:
            differences.append((float(np.mean(a)) - float(np.mean(b))) * 100)
    if not differences:
        return None
    return [
        round(float(np.percentile(differences, 2.5)), 2),
        round(float(np.percentile(differences, 97.5)), 2),
    ]


def report(cfg: dict) -> Path:
    output_dir = ROOT / cfg["output_dir"]
    dev = load_json(output_dir / "dev_results.json")
    test = load_json(output_dir / "test_results.json")
    selection = load_json(output_dir / "dev_selection.json")
    vlm_dev_path = output_dir / "vlm_visibility_dev_results.json"
    vlm_test_path = output_dir / "vlm_visibility_test_results.json"
    vlm_dev = load_json(vlm_dev_path) if vlm_dev_path.exists() else {"status": "not_run"}
    vlm_test = load_json(vlm_test_path) if vlm_test_path.exists() else {"status": "not_run"}
    ci = bootstrap_ci(
        test, selection["selected_method"], "klt_fb",
        int(cfg["evaluation"]["bootstrap_replicates"]),
        int(cfg["evaluation"]["bootstrap_seed"]),
    )
    lines = [
        "# Batch 41 — RoboTAP point visibility, occlusion and affine compensation",
        "",
        f"- Frozen source commit: {current_sha()}",
        f"- DEV / TEST videos: {dev['nscenes']} / {test['nscenes']}",
        f"- DEV / TEST point queries: {dev['nqueries']} / {test['nqueries']}",
        f"- Selected from DEV only: {selection['selected_method']}",
        f"- DEV selection reason: {selection['reason']}",
        f"- Dataset manifest SHA-256: {dev['manifest_sha256']}",
        "",
        "## CPU point tracking and abstention",
        "",
        "| Split | Method | PCK@8 visible | Jaccard@8 | Visible coverage | Accepted precision@8 | Occluded false accept | Occlusion accuracy | CPU P50/P95 ms/query |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name, result in (("DEV", dev), ("TEST", test)):
        for method in METHODS:
            m = result["metrics"][method]
            def pct(value):
                return "n/a" if value is None else f"{100*value:.2f}%"
            def ms(key):
                value = m[key]
                return "n/a" if value is None else f"{value:.3f}"
            lines.append(
                f"| {split_name} | {method} | {pct(m['pck8_visible_unconditional'])} "
                f"| {pct(m['jaccard8'])} | {pct(m['visible_coverage'])} "
                f"| {pct(m['accepted_precision8'])} | {pct(m['occluded_false_accept_rate'])} "
                f"| {pct(m['occlusion_accuracy'])} | {ms('cpu_p50_ms')}/{ms('cpu_p95_ms')} |"
            )
    lines += [
        "",
        "## TEST paired scene bootstrap",
        "",
        f"- Selected method versus klt_fb Jaccard@8 difference, 95% scene-bootstrap interval: {ci}.",
        "- Small scene count makes intervals descriptive; they are not a guarantee of broad-domain performance.",
        "",
        "## Background-relative image-plane diagnostic",
        "",
        f"- TEST adjacent-pair affine acceptance: {test['background_affine']['pair_acceptance_rate']}.",
        f"- TEST mean per-scene background-relative residual: {test['background_affine']['mean_scene_residual_px']} px.",
        "- This is a 2-D affine background proxy, not calibrated camera pose or world-relative 3-D motion.",
        "",
        "## SmolVLM2-500M visibility baseline",
        "",
        f"- DEV: {json.dumps(vlm_dev, sort_keys=True)}",
        f"- TEST: {json.dumps(vlm_test, sort_keys=True)}",
        "- VLM visibility classification is separate from point-coordinate tracking; ABSENT conflates full occlusion and out-of-frame.",
        "",
        "## Interpretation limits",
        "",
        "- RoboTAP is synthetic robotics video, not natural camera footage.",
        "- Only two of five official archive shards are used; the split is project-local and scene-disjoint, not an official leaderboard split.",
        "- First-visible point queries follow the TAP-Vid protocol; future point coordinates and visibility are score-only.",
        "- Background-affine residuals are not world motion and may fail under parallax, zoom, rolling shutter or dominant foreground motion.",
        "- VLM and CPU trackers have different temporal evidence budgets and are not directly matched architectures.",
        "- DEV is the only method-selection source. TEST was predicted and pinned before private-oracle scoring.",
        "",
        "## Integrity",
        "",
        "- No videos, pickle shards, extracted frame arrays, private labels, weights or secrets are published.",
        "- Full source-member, frame-array, query, oracle, prediction and code hashes are stored locally on Tang.",
        "",
    ]
    path = output_dir / "BATCH41_RESULTS.md"
    src.atomic_once(path, "\n".join(lines).encode("utf-8"))
    print("ROBOTAP_REPORT_WRITTEN", path, sha_file(path), flush=True)
    return path


def final_audit(cfg: dict) -> None:
    manifest = src.audit(cfg, verify_sources=True)
    output_dir = ROOT / cfg["output_dir"]
    selection = load_json(output_dir / "dev_selection.json")
    if selection.get("source_commit") != current_sha():
        raise RuntimeError("ROBOTAP_FINAL_SELECTION_SHA_DRIFT")
    for split in ("dev", "test"):
        prediction_path = output_dir / f"{split}_predictions.json"
        result_path = output_dir / f"{split}_results.json"
        require_pin(prediction_path)
        prediction = load_json(prediction_path)
        result = load_json(result_path)
        if (prediction.get("source_commit") != current_sha()
                or result.get("source_commit") != current_sha()
                or prediction.get("manifest_sha256") != sha_file(
                    ROOT / cfg["dataset_dir"] / "batch41_frozen/manifest.json"
                )):
            raise RuntimeError("ROBOTAP_FINAL_PREDICTION_OR_RESULT_SHA_DRIFT")
    vlm_prediction = output_dir / "vlm_visibility_dev_predictions.json"
    if vlm_prediction.exists():
        require_pin(vlm_prediction)
        for split in ("dev", "test"):
            result_path = output_dir / f"vlm_visibility_{split}_results.json"
            if not result_path.exists():
                raise RuntimeError("ROBOTAP_FINAL_VLM_RESULT_MISSING:" + split)
    if not (output_dir / "BATCH41_RESULTS.md").is_file():
        raise RuntimeError("ROBOTAP_FINAL_REPORT_MISSING")
    print("ROBOTAP_FINAL_AUDIT_PASS", current_sha(),
          "scenes", len(manifest["scenes"]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=(
        "predict-dev", "score-dev", "select-dev", "predict-test", "score-test",
        "report", "final-audit",
    ))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    cfg = src.load_config(Path(args.config))
    if args.action.startswith("predict-"):
        predict(args.action.split("-")[1], cfg)
    elif args.action.startswith("score-"):
        score(args.action.split("-")[1], cfg)
    elif args.action == "select-dev":
        select_dev(cfg)
    elif args.action == "report":
        report(cfg)
    else:
        final_audit(cfg)


if __name__ == "__main__":
    main()
