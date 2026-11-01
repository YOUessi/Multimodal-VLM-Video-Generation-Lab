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
)
THRESHOLDS = (4, 8, 16)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def current_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def verify_split(cfg: dict, rows: list) -> None:
    names = [r["name"] for r in rows]
    if len(names) != len(set(names)):
        raise RuntimeError("BATCH37_SCENE_REUSE_WITHIN_NEW_SPLIT")
    if set(names) & set(cfg["exclude_scenes"]):
        raise RuntimeError("BATCH37_PREVIOUS_B36_SCENE_REUSED")
    if [r["split"] for r in rows] != (
            ["dev"] * cfg["dev_scenes"] + ["test"] * cfg["test_scenes"]):
        raise RuntimeError("BATCH37_SCENE_SPLIT_NOT_FROZEN")


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
    output = ROOT / cfg["output_dir"] / (split + "_predictions.json")
    if output.is_file():
        doc = load(output)
        if (doc.get("manifest_sha256") != mh
                or doc.get("methods") != list(METHODS)
                or doc.get("source_commit") != current_sha()
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
                correct_visible += error <= 8
                wrong_visible += error > 8
            if gt["pixels"] < threshold * first_area:
                shrunk_total += 1
                shrunk_accepted += xy is not None
                shrunk_wrong += xy is not None and np.linalg.norm(
                    np.subtract(xy, gt["center"])) > 8
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
    if (sealed.get("manifest_sha256") != raw.sha256((dataset / "manifest.json").read_bytes())
            or sealed.get("methods") != list(METHODS)
            or sealed.get("source_commit") != current_sha()
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
        "scene_bootstrap_consensus_minus_klt_fb_pp_ci95": bootstrap_ci(
            gold, rows, records, "klt_fb_consensus"),
        "scene_bootstrap_guarded_minus_klt_fb_pp_ci95": bootstrap_ci(
            gold, rows, records, "klt_fb_guarded"),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=(
        "predict-dev", "score-dev", "predict-test", "score-test"))
    parser.add_argument("--config", default="configs/davis_occlusion_v2.json")
    args = parser.parse_args()
    cfg = load(ROOT / args.config)
    action, split = args.action.split("-")
    if action == "predict":
        predict(cfg, split)
    else:
        score(cfg, split)


if __name__ == "__main__":
    main()
