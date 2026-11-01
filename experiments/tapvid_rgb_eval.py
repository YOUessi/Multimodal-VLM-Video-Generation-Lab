"""Batch 38 frozen synthetic robotics point tracking with explicit point occlusion.

Four CPU baselines receive exactly the same first-visible query (x,y,t).
TEST receives no occlusion labels or future trajectories until after SHA seal.
Metric is *physical surface point* tracking, not mask centroid or 3D motion.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import tapvid_rgb_data as src

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/tapvid_rgb_stacking_v1/batch38_frozen"
OUT = ROOT / "outputs/tapvid_rgb_stacking_v1"
METHODS = ("stationary_point", "klt_forward", "klt_fb", "template_recover4")
EVAL_RADIUS = (1, 2, 4, 8, 16)
TEMPLATE_SIZE = 21
TEMPLATE_RADIUS = 32
TEMPLATE_NCC = 0.75
TEMPLATE_MAX_MISSES = 4
FB_MAX_ERROR = 1.5


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def doc(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def once(path: Path, payload: dict):
    src.atomic_once(path, src.as_json(payload))


def pin(path: Path):
    tag = path.with_suffix(".sha256")
    src.atomic_once(tag, (sha_bytes(path.read_bytes()) + "\n").encode())


def require_pin(path: Path):
    if not path.exists() or not path.with_suffix(".sha256").exists():
        raise RuntimeError("MISSING_ORACLE_FIREWALL_PREDICTION_PIN")
    if path.with_suffix(".sha256").read_text().strip() != sha_bytes(path.read_bytes()):
        raise RuntimeError("PREDICTIONS_CHANGED_AFTER_SEAL")


def gitsha():
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def check_video(frames: np.ndarray, queries: list):
    if frames.ndim != 4 or frames.dtype != np.uint8 or frames.shape[-1] != 3:
        raise ValueError("VIDEO_SHAPE_OR_DTYPE_INVALID")
    h, w = frames.shape[1:3]
    for q in queries:
        t, xy = q["query_time"], np.asarray(q["query_xy"], dtype=np.float64)
        if t < 0 or t >= len(frames) or xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("BAD_PUBLIC_QUERY")
        if not (0 <= xy[0] < w and 0 <= xy[1] < h):
            raise ValueError("QUERY_OUTSIDE_VIDEO")


def klt_point(gray: list[np.ndarray], query: dict, *, fb: bool) -> list:
    t = int(query["query_time"])
    p = np.asarray(query["query_xy"], dtype=np.float32).reshape(1, 1, 2)
    out = [None] * len(gray)
    out[t] = [float(v) for v in p.reshape(2)]
    h, w = gray[0].shape
    for k in range(t + 1, len(gray)):
        if p is None:
            break
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(
            gray[k-1], gray[k], p, None, winSize=(21,21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if nxt is None or st is None or int(st[0, 0]) != 1:
            break
        v = nxt.reshape(2)
        if not np.isfinite(v).all() or not (0 <= v[0] < w and 0 <= v[1] < h):
            break
        if fb:
            back, bst, _ = cv2.calcOpticalFlowPyrLK(
                gray[k], gray[k-1], nxt, None, winSize=(21,21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            if (back is None or bst is None or int(bst[0, 0]) != 1
                    or not np.isfinite(back).all()
                    or float(np.linalg.norm(back - p)) > FB_MAX_ERROR):
                break
        out[k] = [round(float(x), 3) for x in v]
        p = nxt
    return out


def patch(gray: np.ndarray, xy: np.ndarray, half: int) -> np.ndarray:
    pad = half + 3
    g = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    x = int(round(float(xy[0]))) + pad
    y = int(round(float(xy[1]))) + pad
    return g[y-half:y+half+1, x-half:x+half+1]


def template_trace(gray: list[np.ndarray], query: dict) -> list:
    t = int(query["query_time"])
    anchor = np.asarray(query["query_xy"], dtype=np.float64)
    out = [None] * len(gray)
    out[t] = anchor.tolist()
    h, w = gray[0].shape
    half = TEMPLATE_SIZE // 2
    templ = patch(gray[t], anchor, half)
    if templ.shape != (TEMPLATE_SIZE, TEMPLATE_SIZE) or np.std(templ) < 8:
        return out
    last = anchor.copy()
    misses = 0
    for k in range(t + 1, len(gray)):
        if misses >= TEMPLATE_MAX_MISSES:
            break
        pad = TEMPLATE_RADIUS + half + 3
        padded = cv2.copyMakeBorder(gray[k], pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        cx, cy = int(round(float(last[0]))) + pad, int(round(float(last[1]))) + pad
        radius = TEMPLATE_RADIUS
        roi = padded[
            cy-radius-half:cy+radius+half+1,
            cx-radius-half:cx+radius+half+1,
        ]
        if roi.shape[0] < TEMPLATE_SIZE or roi.shape[1] < TEMPLATE_SIZE:
            misses += 1
            continue
        result = cv2.matchTemplate(roi, templ, cv2.TM_CCOEFF_NORMED)
        _, score, _, top = cv2.minMaxLoc(result)
        found = last + np.array([top[0] - radius, top[1] - radius], dtype=np.float64)
        if not np.isfinite(score) or score < TEMPLATE_NCC or not (
                0 <= found[0] < w and 0 <= found[1] < h):
            misses += 1
            continue
        last = found
        misses = 0
        out[k] = [round(float(x), 3) for x in found]
    return out


def predict_query(gray: list[np.ndarray], q: dict) -> tuple[dict, dict]:
    out, ms = {}, {}
    t = int(q["query_time"])
    for method in METHODS:
        start = time.perf_counter()
        if method == "stationary_point":
            trace = [None if k < t else [float(x) for x in q["query_xy"]]
                     for k in range(len(gray))]
        elif method == "klt_forward":
            trace = klt_point(gray, q, fb=False)
        elif method == "klt_fb":
            trace = klt_point(gray, q, fb=True)
        else:
            trace = template_trace(gray, q)
        ms[method] = round(1000 * (time.perf_counter() - start), 3)
        if len(trace) != len(gray):
            raise RuntimeError("TRACE_LENGTH_MISMATCH")
        out[method] = trace
    return out, ms


def predict(split: str):
    manifest = src.audit()
    queries = doc(DATA / "queries-public.json")
    original_manifest_sha = sha_bytes((DATA / "manifest.json").read_bytes())
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{split}_predictions.json"
    if p.exists():
        require_pin(p)
        old = doc(p)
        if old.get("source_commit") != gitsha() or old.get("manifest_sha256") != original_manifest_sha:
            raise RuntimeError("STALE_PREDICTION_SOURCE")
        print("TAPVID_RGB_EXISTING_PINNED", split, flush=True)
        return
    records = []
    for scene in manifest["scenes"]:
        if scene["split"] != split:
            continue
        frames = np.load(DATA / scene["frames"], mmap_mode="r", allow_pickle=False)
        qs = queries[scene["name"]]
        check_video(frames, qs)
        gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
        tracks, timings = {}, {}
        for q in qs:
            tracks[q["id"]], timings[q["id"]] = predict_query(gray, q)
        records.append({"name": scene["name"], "tracks": tracks,
                        "cpu_query_ms": timings, "nframes": int(len(frames))})
        print("PREDICTED_UNLABELED_SCENE", split, scene["name"], len(qs), flush=True)
    once(p, {
        "dataset": "TAP-Vid RGB-Stacking synthetic robotics",
        "split": split, "source_commit": gitsha(),
        "manifest_sha256": original_manifest_sha,
        "public_query_sha256": sha_bytes((DATA / "queries-public.json").read_bytes()),
        "oracle_unread_during_prediction": True, "models": list(METHODS),
        "records": records,
    })
    pin(p)
    print("TAPVID_RGB_BLIND_PREDICTIONS_PINNED", split, sha_bytes(p.read_bytes()), flush=True)


def score(split: str):
    manifest = src.audit()
    oracle = doc(DATA / "oracle-private.json")
    queries = doc(DATA / "queries-public.json")
    p = OUT / f"{split}_predictions.json"
    require_pin(p)
    saved = doc(p)
    names = [x["name"] for x in manifest["scenes"] if x["split"] == split]
    records = {r["name"]: r for r in saved["records"]}
    if (saved.get("source_commit") != gitsha()
            or saved.get("manifest_sha256") != sha_bytes((DATA / "manifest.json").read_bytes())
            or saved.get("public_query_sha256") != sha_bytes((DATA / "queries-public.json").read_bytes())
            or saved.get("oracle_unread_during_prediction") is not True
            or saved.get("models") != list(METHODS)
            or saved.get("split") != split
            or len(saved["records"]) != len(names)
            or set(records) != set(names)):
        raise RuntimeError("FROZEN_ORACLE_SCORE_GUARD_FAILED")

    per_scene = {}
    for name in names:
        query_rows = {q["id"]: q for q in queries[name]}
        gold = oracle[name]["tracks"]
        if set(query_rows) != set(gold) or set(records[name]["tracks"]) != set(gold):
            raise RuntimeError("QUERY_IDENTITY_MISMATCH")
        metrics = {}
        for method in METHODS:
            c = {"visible":0, "occluded":0, "correct8":0, "accepted_visible":0,
                 "wrong_accepted":0, "visible_abstained":0, "absent_false_accepted":0,
                 "correct_absent":0, "visibility_correct":0, "tp1":0, "tp2":0,
                 "tp4":0, "tp8":0, "tp16":0}
            lat = []
            for qid, q in query_rows.items():
                tr = records[name]["tracks"][qid][method]
                truth = gold[qid]
                if len(tr) != len(truth["xy"]) or len(tr) != len(truth["occluded"]):
                    raise RuntimeError("MISMATCHED_TRACE_LENGTH")
                lat.append(records[name]["cpu_query_ms"][qid][method] / max(1, len(tr)-q["query_time"]-1))
                for i in range(q["query_time"]+1, len(tr)):
                    pred = tr[i]
                    vis = not truth["occluded"][i]
                    if pred is not None:
                        v = np.asarray(pred, dtype=np.float64)
                        if not np.isfinite(v).all() or v.shape != (2,):
                            raise RuntimeError("BAD_PREDICTION_POINT")
                    if vis:
                        c["visible"] += 1
                        if pred is None:
                            c["visible_abstained"] += 1
                        else:
                            c["accepted_visible"] += 1
                            error = float(np.linalg.norm(np.subtract(pred,truth["xy"][i])))
                            c["correct8"] += int(error <= 8)
                            c["wrong_accepted"] += int(error > 8)
                            for r in EVAL_RADIUS:
                                c["tp"+str(r)] += int(error <= r)
                    else:
                        c["occluded"] += 1
                        c["correct_absent"] += int(pred is None)
                        c["absent_false_accepted"] += int(pred is not None)
                    c["visibility_correct"] += int(vis == (pred is not None))
            # AJ variants are computed over the same fixed query-target positions.
            v = c["visible"]
            a = c["absent_false_accepted"]
            metrics[method] = {
                **c,
                "pck8_unconditional": round(c["correct8"] / v, 5) if v else None,
                "accepted_precision8": round(c["correct8"] / c["accepted_visible"], 5) if c["accepted_visible"] else None,
                "accepted_coverage_visible": round(c["accepted_visible"] / v, 5) if v else None,
                "absent_false_accept_rate": round(a / c["occluded"], 5) if c["occluded"] else None,
                "occlusion_accuracy": round(c["visibility_correct"] / (v+c["occluded"]), 5) if v+c["occluded"] else None,
                "jaccard_mean_1_2_4_8_16": round(float(np.mean([
                    c["tp"+str(r)] / (v + a + (c["accepted_visible"] - c["tp"+str(r)]))
                    for r in EVAL_RADIUS])), 5) if v else None,
                "cpu_ms_per_target_frame_p50": round(float(np.median(lat)), 3) if lat else None,
                "cpu_ms_per_target_frame_p95": round(float(np.percentile(lat,95)), 3) if lat else None,
            }
        per_scene[name] = metrics

    aggregated = {}
    for m in METHODS:
        sums = {k: sum(per_scene[s][m][k] for s in names)
                for k in ("visible", "occluded", "correct8", "accepted_visible",
                          "wrong_accepted", "visible_abstained", "absent_false_accepted",
                          "correct_absent", "visibility_correct")}
        v = sums["visible"]
        inv = sums["occluded"]
        aggregated[m] = {
            **sums,
            "pck8_unconditional": round(sums["correct8"]/v,5) if v else None,
            "accepted_precision8": round(sums["correct8"]/sums["accepted_visible"],5)
                if sums["accepted_visible"] else None,
            "absence_false_accept_rate": round(sums["absent_false_accepted"]/inv,5) if inv else None,
            "occlusion_accuracy": round(sums["visibility_correct"]/(v+inv),5) if v+inv else None,
            "cpu_frame_p50_ms_by_scene": round(float(np.median([
                per_scene[s][m]["cpu_ms_per_target_frame_p50"] for s in names])),3),
        }
    out = {
        "dataset": "TAP-Vid RGB-Stacking (synthetic robotic manipulations)",
        "source_commit": gitsha(), "split": split,
        "dataset_sha256": manifest["dataset_origin_sha256"],
        "manifest_sha256": saved["manifest_sha256"],
        "sealed_prediction_sha256": sha_bytes(p.read_bytes()),
        "scene_count": len(names), "methods_predeclared": list(METHODS),
        "primary": "PCK@8 on annotated visible point positions, including abstentions as wrong",
        "occlusion_gold": "official point-level occluded labels; no mask area proxy",
        "methods": aggregated, "scene_details": per_scene,
        "caveats": [
            "RGB-Stacking is simulated robotics footage, not independent real-camera evidence.",
            "Point identity is material surface point, not object mask centroid.",
            "Only tracked positions after each first-visible query are scored.",
            "No VLM model or ground-truth camera pose/3D world motion is evaluated here.",
            "Scene counts small; no population generalization or post-TEST threshold choice.",
        ],
    }
    once(OUT / f"{split}_results.json", out)
    print("TAPVID_RGB_SCORED", split, json.dumps({
        "scenes":len(names), "metrics": {m:{
            k:aggregated[m][k] for k in (
                "visible", "occluded", "correct8", "accepted_visible",
                "absent_false_accepted", "pck8_unconditional", "absence_false_accept_rate")}
            for m in METHODS}
    }), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("predict-dev","score-dev","predict-test","score-test"))
    a=p.parse_args()
    operation,split=a.action.split("-")
    if operation == "predict":
        predict(split)
    else:
        score(split)


if __name__ == "__main__":
    main()
