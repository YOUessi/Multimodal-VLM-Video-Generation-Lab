"""Batch 37: fresh scene-disjoint DAVIS tracking, uncertainty and disappearance.

All threshold choices are predeclared in configs/davis_occlusion_v2.json.
The predictor may access *only* first-frame query mask, RGB frames and
source digests; future instance labels are score-only. The object centroid
is an image-plane proxy (not a 3D material point or true camera motion).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

import davis_real_data as raw
import davis_real_eval as base

ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "stationary_pixel", "klt_forward_backward",
    "klt_fb_consensus", "klt_fb_guarded",
    "klt_fb_appearance_045", "klt_fb_appearance_055",
    "klt_fb_appearance_065",
)
THRESHOLDS = (4, 8, 16)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def current_sha() -> str:
    status = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True, text=True, timeout=10,
    )
    if status.returncode:
        raise RuntimeError("DAVIS_SOURCE_STATUS_CHECK_FAILED")
    if status.stdout.strip():
        raise RuntimeError("DAVIS_DIRTY_SOURCE_TREE_CANNOT_PIN_PREDICTIONS")
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, timeout=10
    ).strip()


def verify_split(cfg: dict, rows: list) -> None:
    names = [r["name"] for r in rows]
    if len(names) != len(set(names)):
        raise RuntimeError("BATCH37_SCENE_REUSE_WITHIN_NEW_SPLIT")
    if set(names) & set(cfg["exclude_scenes"]):
        raise RuntimeError("BATCH37_PREVIOUS_B36_SCENE_REUSED")
    if [r["split"] for r in rows] != (
            ["dev"] * cfg["dev_scenes"] + ["test"] * cfg["test_scenes"]):
        raise RuntimeError("BATCH37_SCENE_SPLIT_NOT_FROZEN")


def patch_ncc(
    previous_gray: np.ndarray, current_gray: np.ndarray,
    previous_xy: np.ndarray, current_xy: np.ndarray, size: int = 11,
) -> float:
    """Mean-normalized local patch correlation; illumination offset is removed."""
    if size < 5 or size % 2 == 0:
        raise ValueError("PATCH_SIZE_MUST_BE_ODD_AND_AT_LEAST_5")
    half = size // 2
    h, w = previous_gray.shape
    x0, y0 = map(float, previous_xy)
    x1, y1 = map(float, current_xy)
    if min(x0, y0, x1, y1) < half + 1 or x0 >= w-half-1 or x1 >= w-half-1 or y0 >= h-half-1 or y1 >= h-half-1:
        return -1.0
    a = cv2.getRectSubPix(previous_gray, (size, size), (x0, y0)).astype(np.float32)
    b = cv2.getRectSubPix(current_gray, (size, size), (x1, y1)).astype(np.float32)
    a -= float(a.mean())
    b -= float(b.mean())
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not np.isfinite(denominator) or denominator < 1e-6:
        return -1.0
    score = float(np.sum(a * b) / denominator)
    return score if np.isfinite(score) else -1.0


def consensus_track(
    frames: list[np.ndarray],
    initial_mask: np.ndarray,
    center: list[float],
    *,
    min_features: int,
    fb_error_px: float,
    max_step_px: float,
    inlier_residual_px: float,
    min_consensus_ratio: float,
    patch_ncc_min: float | None = None,
    patch_size: int = 11,
) -> list[list[float] | None]:
    """Initial-target sparse LK with per-step FB and robust motion consensus.

    No re-identification on loss: predictions abstain permanently, rather than
    silently switching identities after complete occlusion.
    """
    if not frames:
        raise ValueError("EMPTY_VIDEO")
    gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in frames]
    if any(g.shape != gray[0].shape for g in gray):
        raise RuntimeError("IMAGE_DIMENSION_DRIFT")
    if initial_mask.shape != gray[0].shape:
        raise RuntimeError("MASK_DIMENSION_DRIFT")
    trace = [[round(float(center[0]), 3), round(float(center[1]), 3)]]
    pts = cv2.goodFeaturesToTrack(
        gray[0], maxCorners=90, qualityLevel=.015, minDistance=5,
        mask=initial_mask)
    if pts is None or len(pts) < min_features:
        return trace + [None] * (len(frames) - 1)

    active = pts.astype(np.float32)
    origin = active.copy()
    h, w = gray[0].shape
    for k in range(1, len(gray)):
        if len(active) < min_features:
            trace.extend([None] * (len(gray) - k))
            break
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(
            gray[k - 1], gray[k], active, None, winSize=(21, 21),
            maxLevel=3, criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        if nxt is None or st is None:
            trace.extend([None] * (len(gray) - k))
            break
        back, bst, _ = cv2.calcOpticalFlowPyrLK(
            gray[k], gray[k - 1], nxt, None, winSize=(21, 21),
            maxLevel=3, criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        if back is None or bst is None:
            trace.extend([None] * (len(gray) - k))
            break

        x = nxt.reshape(-1, 2)
        previous = active.reshape(-1, 2)
        backward = back.reshape(-1, 2)
        backward_error = np.linalg.norm(backward - previous, axis=1)
        valid = (
            st.reshape(-1).astype(bool)
            & bst.reshape(-1).astype(bool)
            & np.isfinite(x).all(axis=1)
            & np.isfinite(backward).all(axis=1)
            & np.isfinite(backward_error)
            & (backward_error <= fb_error_px)
            & (x[:, 0] >= 0) & (x[:, 0] < w)
            & (x[:, 1] >= 0) & (x[:, 1] < h)
        )
        if patch_ncc_min is not None:
            ncc = np.asarray([
                patch_ncc(gray[k - 1], gray[k], p0, p1, patch_size)
                for p0, p1 in zip(previous, x)
            ], dtype=np.float32)
            valid &= np.isfinite(ncc) & (ncc >= patch_ncc_min)
        if np.count_nonzero(valid) < min_features:
            trace.extend([None] * (len(gray) - k))
            break
        motion = x[valid] - previous[valid]
        median_motion = np.median(motion, axis=0)
        residual = np.linalg.norm(motion - median_motion, axis=1)
        consensus = residual <= inlier_residual_px
        keep = np.where(valid)[0][consensus]
        if (len(keep) < min_features
                or float(np.mean(consensus)) < min_consensus_ratio
                or not np.isfinite(median_motion).all()
                or np.linalg.norm(median_motion) > max_step_px):
            trace.extend([None] * (len(gray) - k))
            break
        active = nxt[keep]
        origin = origin[keep]
        estimate = np.asarray(center, dtype=np.float64) + np.median(
            active.reshape(-1, 2) - origin.reshape(-1, 2), axis=0)
        if (not np.isfinite(estimate).all()
                or not 0 <= estimate[0] < w
                or not 0 <= estimate[1] < h):
            trace.extend([None] * (len(gray) - k))
            break
        trace.append([round(float(v), 3) for v in estimate])
    if len(trace) != len(frames):
        raise RuntimeError("BATCH37_TRACE_LENGTH_DRIFT")
    return trace


def scene_predict(dataset: Path, row: dict, query: dict, cfg: dict) -> dict:
    frames = [cv2.imread(str(dataset / x["jpg"]), cv2.IMREAD_COLOR)
              for x in row["assets"]]
    if any(x is None for x in frames):
        raise RuntimeError("BATCH37_SOURCE_FRAME_DECODE_FAILED")
    mask = cv2.imread(str(dataset / query["query_mask"]),
                      cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError("BATCH37_INITIAL_QUERY_MASK_NOT_FOUND")
    timings: dict[str, float] = {}
    tracks: dict[str, list] = {}
    for method in METHODS:
        start = time.perf_counter()
        if method == "stationary_pixel":
            tracks[method] = [query["center"][:] for _ in frames]
        elif method == "klt_forward_backward":
            tracks[method] = base.klt_track(
                frames, mask, query["center"], use_fb=True)
        else:
            tracks[method] = consensus_track(
                frames, mask, query["center"],
                **cfg["predeclared_models"][method])
        timings[method] = round(1000 * (time.perf_counter() - start), 3)
    camera = base.affine_background_delta(frames, mask, query["center"])
    if camera["accepted"]:
        projection = camera["projected_background_xy"]
        for method in METHODS:
            endpoint = tracks[method][-1]
            camera["relative_" + method] = (
                [round(endpoint[i] - projection[i], 3) for i in (0, 1)]
                if endpoint is not None else None)
    return {
        "name": row["name"], "nframes": len(frames),
        "trajectories": tracks, "cpu_method_ms": timings,
        "background_2d_diagnostic": camera,
    }


def predict(cfg: dict, split: str) -> None:
    dataset = ROOT / cfg["dataset_dir"]
    manifest = raw.check(cfg, include_future_masks_and_gold=False)
    verify_split(cfg, manifest["rows"])
    mh = raw.sha256((dataset / "manifest.json").read_bytes())
    selection_path = ROOT / cfg["output_dir"] / "dev_selection.json"
    selection = None
    if split == "test":
        if not selection_path.is_file():
            raise RuntimeError("BATCH40_DEV_SELECTION_MISSING")
        selection = load(selection_path)
        if (selection.get("source_commit") != current_sha()
                or selection.get("selected_method") not in METHODS):
            raise RuntimeError("BATCH40_DEV_SELECTION_SOURCE_OR_METHOD_DRIFT")
    output = ROOT / cfg["output_dir"] / (split + "_predictions.json")
    if output.is_file():
        doc = load(output)
        expected_selection_sha = raw.sha256(selection_path.read_bytes()) if selection else None
        expected_selected_method = selection.get("selected_method") if selection else None
        if (doc.get("manifest_sha256") != mh
                or doc.get("methods") != list(METHODS)
                or doc.get("source_commit") != current_sha()
                or doc.get("dev_selection_sha256") != expected_selection_sha
                or doc.get("dev_selected_method") != expected_selected_method
                or doc.get("split") != split):
            raise RuntimeError("BATCH37_EXISTING_PREDICTION_CODE_OR_DATA_DRIFT")
        base.pin_prediction_bytes(output)
        print("BATCH37_EXISTING_PREDICTION_FROZEN", split, flush=True)
        return
    queries = load(dataset / "initial_queries.json")
    rows = [r for r in manifest["rows"] if r["split"] == split]
    result = [scene_predict(dataset, row, queries[row["name"]], cfg)
              for row in rows]
    base.write_once(output, {
        "methods": list(METHODS), "split": split,
        "source_commit": current_sha(), "manifest_sha256": mh,
        "dev_selection_sha256": raw.sha256(selection_path.read_bytes()) if selection else None,
        "dev_selected_method": selection.get("selected_method") if selection else None,
        "no_future_gold_in_inference": True, "nscenes": len(rows),
        "records": result,
    })
    base.pin_prediction_bytes(output)
    print("BATCH37_SEALED_PREDICTIONS", split, len(result), flush=True)


def occlusion_proxies(gold: dict, rows: list, records: dict,
                      method: str, threshold: float) -> dict:
    """Mask-area shrink is a proxy, not a claim about annotated occlusion."""
    absent_total = absent_accepted = 0
    shrunk_total = shrunk_accepted = shrunk_wrong = 0
    correct_visible = wrong_visible = 0
    for row in rows:
        key = row["name"]
        points = gold[key]["points"]
        preds = records[key]["trajectories"][method]
        first_area = points[0]["pixels"]
        for gt, xy in zip(points[1:], preds[1:]):
            if not gt["visible"]:
                absent_total += 1
                absent_accepted += xy is not None
                continue
            if xy is not None:
                error = np.linalg.norm(np.subtract(xy, gt["center"]))
                correct_visible += int(error <= 8)
                wrong_visible += int(error > 8)
            if gt["pixels"] < threshold * first_area:
                shrunk_total += 1
                shrunk_accepted += xy is not None
                shrunk_wrong += int(xy is not None and np.linalg.norm(
                    np.subtract(xy, gt["center"])) > 8)
    return {
        "absent_positions": absent_total,
        "absent_false_acceptances": absent_accepted,
        "mask_area_shrink_positions_not_verified_occlusions": shrunk_total,
        "mask_area_shrink_accepted": shrunk_accepted,
        "mask_area_shrink_wrong_accepted": shrunk_wrong,
        "visible_correct_accepted": correct_visible,
        "visible_wrong_accepted": wrong_visible,
    }


def bootstrap_ci(gold: dict, rows: list, records: dict, method: str) -> list | None:
    rng = np.random.default_rng(20261102)
    diff = []
    for _ in range(800):
        subsample = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        a = base.summarize(gold, subsample, records, method)["visible_pck8_unconditional"]
        b = base.summarize(gold, subsample, records, "klt_forward_backward")["visible_pck8_unconditional"]
        if a is not None and b is not None:
            diff.append((a - b) * 100)
    return ([round(float(np.percentile(diff, p)), 2) for p in (2.5, 97.5)]
            if diff else None)


def score(cfg: dict, split: str) -> None:
    dataset = ROOT / cfg["dataset_dir"]
    manifest = raw.check(cfg, include_future_masks_and_gold=True)
    verify_split(cfg, manifest["rows"])
    rows = [r for r in manifest["rows"] if r["split"] == split]
    path = ROOT / cfg["output_dir"] / (split + "_predictions.json")
    base.require_pinned_prediction(path)
    sealed = load(path)
    selection_path = ROOT / cfg["output_dir"] / "dev_selection.json"
    selection = load(selection_path) if split == "test" and selection_path.is_file() else None
    if split == "test" and (
            selection is None
            or selection.get("source_commit") != current_sha()
            or selection.get("status") != "selected_from_dev_only"):
        raise RuntimeError("BATCH40_TEST_DEV_SELECTION_MISSING_OR_STALE")
    expected_selection_sha = raw.sha256(selection_path.read_bytes()) if selection else None
    expected_selected_method = selection.get("selected_method") if selection else None
    if (sealed.get("manifest_sha256") != raw.sha256((dataset / "manifest.json").read_bytes())
            or sealed.get("methods") != list(METHODS)
            or sealed.get("source_commit") != current_sha()
            or sealed.get("dev_selection_sha256") != expected_selection_sha
            or sealed.get("dev_selected_method") != expected_selected_method
            or sealed.get("no_future_gold_in_inference") is not True
            or sealed.get("split") != split
            or sealed.get("nscenes") != len(rows)):
        raise RuntimeError("BATCH37_UNSEALED_OR_CROSS_VERSION_PREDICTIONS")
    records = {r["name"]: r for r in sealed["records"]}
    if len(sealed["records"]) != len(rows) or set(records) != {r["name"] for r in rows}:
        raise RuntimeError("BATCH37_INCOMPLETE_OR_DUPLICATED_SCENE_PREDICTIONS")
    gold = load(dataset / "gold-private.json")
    aggregate = {
        method: {
            **base.summarize(gold, rows, records, method),
            **occlusion_proxies(
                gold, rows, records, method, cfg["shrink_proxy_threshold"]),
        } for method in METHODS
    }
    camera_count = sum(
        bool(records[r["name"]]["background_2d_diagnostic"]["accepted"])
        for r in rows)
    out = {
        "dataset": "DAVIS 2017 official real camera 480p",
        "split": split, "source_commit": current_sha(),
        "source_manifest_sha256": sealed["manifest_sha256"],
        "sealed_predictions_sha256": raw.sha256(path.read_bytes()),
        "scenes": len(rows), "frames_per_scene": cfg["source_frames"],
        "all_baselines_predeclared": True,
        "previous_batch36_scene_disjoint": True,
        "world_relative_motion_not_labeled": True,
        "future_labels_never_used_for_inference": True,
        "methods": aggregate, "background_2d_registration_accepted": camera_count,
        "dev_selected_method": sealed.get("dev_selected_method"),
        "dev_selection_sha256": sealed.get("dev_selection_sha256"),
        "scene_bootstrap_pck8_vs_klt_fb_pp_ci95": {
            method: bootstrap_ci(gold, rows, records, method)
            for method in METHODS if method != "klt_forward_backward"
        },
        "caveats": [
            "Mask-center is not a material point; changing masks shift the proxy.",
            "Mask-area shrink is not a validated occlusion annotation.",
            "Dominant background affine is not camera pose, depth or world motion.",
            "No directly comparable SmolVLM2 point tracking baseline.",
            "Same DAVIS corpus, new scene-disjoint draw; no pretraining novelty claim.",
            "All test variants and thresholds were fixed before viewing test labels.",
        ],
    }
    base.write_once(ROOT / cfg["output_dir"] / (split + "_results.json"), out)
    print("BATCH37_SCORED", split, json.dumps({
        "scenes": len(rows), "background_registration": camera_count,
        "metrics": {
            m: {k: aggregate[m][k] for k in (
                "visible_evaluations", "invisible_evaluations",
                "visible_pck8_unconditional", "visible_accepted",
                "absent_false_accepted", "visible_wrong_accepted",
                "cpu_scene_p50_ms", "mask_area_shrink_positions_not_verified_occlusions",
            )}
            for m in METHODS
        }
    }, sort_keys=True), flush=True)


def select_dev(cfg: dict) -> dict:
    output_dir = ROOT / cfg["output_dir"]
    results_path = output_dir / "dev_results.json"
    if not results_path.is_file():
        raise RuntimeError("BATCH40_DEV_RESULTS_MISSING")
    results = load(results_path)
    if results.get("source_commit") != current_sha() or results.get("split") != "dev":
        raise RuntimeError("BATCH40_DEV_RESULTS_SOURCE_DRIFT")
    rule = cfg["dev_selection"]
    eligible = []
    diagnostics = {}
    for method in rule["candidate_methods"]:
        if method not in results["methods"]:
            raise RuntimeError("BATCH40_DEV_CANDIDATE_METHOD_MISSING:" + method)
        m = results["methods"][method]
        absent_n = int(m["invisible_evaluations"])
        visible_n = int(m["visible_evaluations"])
        absent_rate = (m["absent_false_accepted"] / absent_n) if absent_n else None
        coverage = (m["visible_accepted"] / visible_n) if visible_n else 0.0
        diagnostics[method] = {
            "visible_pck8_unconditional": m["visible_pck8_unconditional"],
            "visible_coverage": round(coverage, 6),
            "absent_false_accept_rate": None if absent_rate is None else round(absent_rate, 6),
            "visible_evaluations": visible_n,
            "invisible_evaluations": absent_n,
        }
        if (absent_n >= int(rule["minimum_absent_evaluations"])
                and absent_rate is not None
                and absent_rate <= float(rule["max_absent_false_accept_rate"])
                and coverage >= float(rule["min_visible_coverage"])):
            eligible.append(method)
    selected = (
        max(eligible, key=lambda name: (
            results["methods"][name]["visible_pck8_unconditional"],
            diagnostics[name]["visible_coverage"], name,
        )) if eligible else rule["fallback_method"]
    )
    doc = {
        "status": "selected_from_dev_only",
        "source_commit": current_sha(),
        "dev_results_sha256": raw.sha256(results_path.read_bytes()),
        "candidate_diagnostics": diagnostics,
        "eligible_methods": eligible,
        "selection_rule": rule,
        "selected_method": selected,
        "fallback_used": not bool(eligible),
        "test_labels_read": False,
    }
    base.write_once(output_dir / "dev_selection.json", doc)
    print("BATCH40_DEV_SELECTED_METHOD", selected, json.dumps(doc, sort_keys=True), flush=True)
    return doc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=(
        "predict-dev", "score-dev", "select-dev", "predict-test", "score-test"))
    parser.add_argument("--config", default="configs/davis_occlusion_v3.json")
    args = parser.parse_args()
    cfg = load(ROOT / args.config)
    if args.action == "select-dev":
        select_dev(cfg)
    else:
        action, split = args.action.split("-")
        if action == "predict":
            predict(cfg, split)
        else:
            score(cfg, split)


if __name__ == "__main__":
    main()
