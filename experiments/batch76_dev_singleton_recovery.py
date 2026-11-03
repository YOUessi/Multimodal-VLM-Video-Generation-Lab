"""Batch76: unlabeled causal DEV redetection challenge and DEV-only adjudication.

Exploratory on previously studied Batch56 natural DEV, NOT a new independent TEST.
No oracle, hidden TEST video, or TEST prediction is read by predict(). The
evaluator opens only 36 explicitly named DEV oracle records after an immutable
prediction pin exists. Batch59's frozen runner is never imported or modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from experiments import batch62_dev_consensus_veto as b62
from experiments import batch75_dev_reacquisition_audit as b75

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data/tapvid_kinetics_batch56/batch56_frozen"
# Historical public predictions on the 36 natural DEV scenes only.
METHODS = ("baseline", "singleton_raw", "singleton_3confirm")
PATCH = 17
MIN_STD = 4.0
MIN_SCORE = 0.86
MIN_MARGIN = 0.07
MIN_MISSING = 12
STRIDE = 3
CONFIRMATIONS = 3
MAX_STEP_PIXELS = 15.0  # between consecutive sampled frames
RADIUS = 8.0
PROTOCOL = "batch76-dev-causal-singleton-stress-v1"


def digest(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def immutable_json(path: Path, value: dict) -> str:
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    pin = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or pin.exists():
        raise RuntimeError("BATCH76_IMMUTABLE_OUTPUT_EXISTS")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as out:
        out.write(data)
    with pin.open("x", encoding="ascii") as out:
        out.write(digest(data) + "\n")
    return digest(data)


def pinned(path: Path) -> dict:
    return b62.pinned_json(path)


def strict_manifest(manifest: dict, predictions: dict, queries: dict) -> list[dict]:
    if not manifest.get("dataset", "").startswith("TAP-Vid-Kinetics"):
        raise RuntimeError("BATCH76_INVALID_MANIFEST_DATASET")
    if predictions.get("split") != "dev" or predictions.get("oracle_unread_during_prediction") is not True:
        raise RuntimeError("BATCH76_DEV_ONLY_FIREWALL")
    if digest((DATA / "manifest.json").read_bytes()) != predictions["manifest_sha256"]:
        raise RuntimeError("BATCH76_MANIFEST_PIN_DRIFT")
    scenes = manifest["scenes"]
    dev = [v for v in scenes if v.get("split") == "dev"]
    if len(dev) != 36 or [v["name"] for v in dev] != b62.SCENES:
        raise RuntimeError("BATCH76_DEV_SPLIT_DRIFT")
    if {v["name"] for v in predictions["records"]} != set(b62.SCENES):
        raise RuntimeError("BATCH76_FROZEN_PREDICTION_DRIFT")
    if not set(b62.SCENES).issubset(queries):
        raise RuntimeError("BATCH76_PUBLIC_QUERY_DRIFT")
    return dev


def find_singleton(frame: np.ndarray, template: np.ndarray) -> list[float] | None:
    """Find a *visually unique* match; NOT evidence of physical identity."""
    if frame.ndim != 2 or template.shape != (PATCH, PATCH):
        raise RuntimeError("BATCH76_INVALID_IMAGE_SHAPE")
    if min(frame.shape) < PATCH or not np.isfinite(frame).all():
        raise RuntimeError("BATCH76_INVALID_IMAGE")
    if float(np.std(template)) < MIN_STD:
        return None
    sim = cv2.matchTemplate(np.ascontiguousarray(frame), template, cv2.TM_CCOEFF_NORMED)
    if not np.isfinite(sim).all():
        return None
    _, score, _, loc = cv2.minMaxLoc(sim)
    if score < MIN_SCORE:
        return None
    x, y = loc
    remainder = sim.copy()
    # Suppress overlapping sliding windows of the *same* object.
    remainder[max(0, y-PATCH):min(sim.shape[0], y+PATCH+1),
              max(0, x-PATCH):min(sim.shape[1], x+PATCH+1)] = -1.0
    if score - float(np.max(remainder)) < MIN_MARGIN:
        return None
    p = PATCH // 2
    return [float(x+p), float(y+p)]


def redetect(frames: np.ndarray, query_xy: list[float], query_t: int,
             baseline: list) -> dict:
    """No oracle access. One-sided augmentation: never changes baseline accepts.

    Stateless template is pinned at query frame; missingness is inferred ONLY
    from causal baseline abstentions. Every candidate confirmation uses past or
    current frames, never future frames or oracle visibility. Output on sampled
    frames only; not a dense/interpolated track.
    """
    n = len(frames)
    if not (0 <= query_t < n - 1 and len(baseline) == n):
        raise RuntimeError("BATCH76_QUERY_LENGTH_DRIFT")
    if frames.ndim not in (3, 4):
        raise RuntimeError("BATCH76_INVALID_FRAMES")
    if frames.ndim == 4 and frames.shape[-1] != 3:
        raise RuntimeError("BATCH76_INVALID_RGB")
    qx, qy = map(float, query_xy)
    if not all(math.isfinite(z) for z in (qx, qy)):
        raise RuntimeError("BATCH76_INVALID_QUERY")
    x, y = int(round(qx)), int(round(qy))
    p = PATCH // 2
    h, w = frames.shape[1:3]
    raw = [None for _ in range(n)]
    confirmed = [None for _ in range(n)]
    if x-p < 0 or y-p < 0 or x+p >= w or y+p >= h:
        return {"raw": raw, "confirmed": confirmed, "skipped_query_border": True,
                "search_calls": 0, "search_ms": 0.0}
    ref = frames[query_t]
    ref = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY) if ref.ndim == 3 else ref
    template = np.ascontiguousarray(ref[y-p:y+p+1, x-p:x+p+1])
    if float(np.std(template)) < MIN_STD:
        return {"raw": raw, "confirmed": confirmed, "skipped_low_texture": True,
                "search_calls": 0, "search_ms": 0.0}
    missing, confirmations, previous = 0, 0, None
    search_ms = 0.0
    calls = 0
    for t in range(query_t + 1, n):
        if baseline[t] is not None:
            missing, confirmations, previous = 0, 0, None
            continue
        missing += 1
        if missing < MIN_MISSING or missing % STRIDE != 0:
            continue
        frame = frames[t]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        start = time.perf_counter()
        candidate = find_singleton(gray, template)
        search_ms += (time.perf_counter() - start) * 1000
        calls += 1
        if candidate is None:
            confirmations, previous = 0, None
            continue
        raw[t] = candidate
        if previous is not None and math.dist(previous, candidate) <= MAX_STEP_PIXELS:
            confirmations += 1
        else:
            confirmations = 1
        previous = candidate
        if confirmations >= CONFIRMATIONS:
            confirmed[t] = candidate
    return {"raw": raw, "confirmed": confirmed, "search_calls": calls,
            "search_ms": search_ms}


def predict() -> dict:
    # No read, import or reference to private DEV/TEST oracle here.
    source = pinned(b62.IN / "dev_predictions.json")
    queries_bytes = (DATA / "queries-public.json").read_bytes()
    if digest(queries_bytes) != source["public_query_sha256"]:
        raise RuntimeError("BATCH76_PUBLIC_QUERY_HASH_DRIFT")
    queries = json.loads(queries_bytes)
    manifest = json.loads((DATA / "manifest.json").read_bytes())
    scenes = strict_manifest(manifest, source, queries)
    records = {rec["name"]: rec for rec in source["records"]}
    result = []
    for info in scenes:
        name = info["name"]
        frame_path = ROOT.parent / info["frame_file"]
        if digest(frame_path.read_bytes()) != info["frame_sha256"]:
            raise RuntimeError("BATCH76_FRAME_HASH_DRIFT:" + name)
        frames = np.load(frame_path, mmap_mode="r", allow_pickle=False)
        rec = records[name]
        if len(frames) != rec["nframes"]:
            raise RuntimeError("BATCH76_FRAMES_LENGTH_DRIFT")
        qmap = {q["id"]: q for q in queries[name]}
        if len(qmap) != 4 or set(qmap) != set(rec["tracks"]):
            raise RuntimeError("BATCH76_QUERY_DRIFT")
        tracks = {}
        for qid, q in qmap.items():
            out = redetect(frames, q["query_xy"], q["query_time"],
                           rec["tracks"][qid][b62.BASE])
            tracks[qid] = out
        result.append({"name": name, "tracks": tracks, "nframes": len(frames)})
        print("BATCH76_PUBLIC_PREDICTED", name, flush=True)
    return {"kind": "BATCH76_PUBLIC_ONLY_CAUSAL_REDETECTION",
            "protocol": PROTOCOL, "split": "dev", "exploratory": True,
            "deployable": False, "source_commit": source["source_commit"],
            "source_predictions_sha256": digest((b62.IN / "dev_predictions.json").read_bytes()),
            "manifest_sha256": digest((DATA / "manifest.json").read_bytes()),
            "queries_sha256": digest(queries_bytes),
            "oracle_unread_during_prediction": True,
            "test_predictions_read": False, "test_oracle_read": False,
            "methods": list(METHODS), "records": result,
            "parameters": {"patch": PATCH, "min_score": MIN_SCORE,
                           "min_margin": MIN_MARGIN, "min_missing": MIN_MISSING,
                           "stride": STRIDE, "confirmations": CONFIRMATIONS,
                           "max_step_pixels": MAX_STEP_PIXELS}}


def merge_track(baseline: list, supplement: list) -> list:
    if len(baseline) != len(supplement):
        raise RuntimeError("BATCH76_MERGE_LENGTH_DRIFT")
    return [v if v is not None else s for v, s in zip(baseline, supplement)]


def evaluate(predict_path: Path) -> dict:
    # This function MUST only be called after public predictions have a SHA pin.
    result = pinned(predict_path)
    if (result.get("kind") != "BATCH76_PUBLIC_ONLY_CAUSAL_REDETECTION"
            or result.get("split") != "dev"
            or result.get("oracle_unread_during_prediction") is not True
            or result.get("protocol") != PROTOCOL):
        raise RuntimeError("BATCH76_PREDICTION_FIREWALL")
    source = pinned(b62.IN / "dev_predictions.json")
    if result["source_predictions_sha256"] != digest(
            (b62.IN / "dev_predictions.json").read_bytes()):
        raise RuntimeError("BATCH76_SOURCE_PREDICTIONS_DRIFT")
    if result["source_commit"] != source["source_commit"]:
        raise RuntimeError("BATCH76_SOURCE_COMMIT_DRIFT")
    if result["manifest_sha256"] != digest((DATA / "manifest.json").read_bytes()):
        raise RuntimeError("BATCH76_MANIFEST_DRIFT")
    query_bytes = (DATA / "queries-public.json").read_bytes()
    if result["queries_sha256"] != digest(query_bytes):
        raise RuntimeError("BATCH76_QUERY_PIN_DRIFT")
    public_queries = json.loads(query_bytes)
    oracle = b62.load_dev_only_oracle(DATA / "oracle-private.json", b62.SCENES)
    originals = {rec["name"]: rec for rec in source["records"]}
    attempted = {rec["name"]: rec for rec in result["records"]}
    if set(attempted) != set(b62.SCENES):
        raise RuntimeError("BATCH76_DEV_SCENES_DRIFT")
    scene_metrics = {}
    all_events = {method: [] for method in METHODS}
    totals = {m: {"visible": 0, "accepted_visible": 0, "correct8": 0,
                  "invisible": 0, "false_accepts": 0} for m in METHODS}
    cpu_search = []
    for name in b62.SCENES:
        old, proposed = originals[name], attempted[name]
        qmap = {q["id"]: q for q in public_queries[name]}
        if set(proposed["tracks"]) != set(qmap):
            raise RuntimeError("BATCH76_TRACK_IDS_DRIFT")
        scene_counts = {m: {"visible": 0, "accepted_visible": 0, "correct8": 0,
                            "invisible": 0, "false_accepts": 0} for m in METHODS}
        for qid, query in qmap.items():
            original = old["tracks"][qid][b62.BASE]
            new = proposed["tracks"][qid]
            gt = oracle[name]["tracks"][qid]
            n = old["nframes"]
            if (len(original) != n or len(gt["xy"]) != n or
                    len(gt["occluded"]) != n or
                    len(new["raw"]) != n or len(new["confirmed"]) != n):
                raise RuntimeError("BATCH76_TRACK_LENGTH_DRIFT")
            cpu_search.append({"scene": name, "query": qid,
                               "search_calls": new["search_calls"],
                               "search_ms": new["search_ms"]})
            options = {"baseline": original,
                       "singleton_raw": merge_track(original, new["raw"]),
                       "singleton_3confirm": merge_track(original, new["confirmed"])}
            t0 = query["query_time"]
            for method, track in options.items():
                accepted = [p is not None and t > t0 for t, p in enumerate(track)]
                all_events[method].extend(b75.post_occlusion_events(
                    gt["occluded"], gt["xy"], track, accepted, t0))
                for t in range(t0 + 1, n):
                    is_visible = not gt["occluded"][t]
                    c = scene_counts[method]
                    c["visible" if is_visible else "invisible"] += 1
                    if accepted[t]:
                        if is_visible:
                            c["accepted_visible"] += 1
                            pred_xy = b62.xy(track[t])
                            truth_xy = b62.xy(gt["xy"][t])
                            if pred_xy is None or truth_xy is None:
                                raise RuntimeError("BATCH76_VISIBLE_COORDINATE_MISSING")
                            c["correct8"] += int(math.dist(pred_xy, truth_xy) <= RADIUS)
                        else:
                            c["false_accepts"] += 1
        scene_metrics[name] = scene_counts
        for method in METHODS:
            for key in totals[method]:
                totals[method][key] += scene_counts[method][key]
    if totals["baseline"]["false_accepts"] != 317:
        raise RuntimeError("BATCH76_BASELINE_REPLAY_FAILURE")
    if any(totals[m]["correct8"] < totals["baseline"]["correct8"]
           or totals[m]["false_accepts"] < totals["baseline"]["false_accepts"]
           for m in METHODS[1:]):
        raise RuntimeError("BATCH76_AUGMENTATION_MONOTONICITY_DRIFT")
    overall = {}
    for method in METHODS:
        c = totals[method]
        overall[method] = {
            **c, "visible_coverage": c["accepted_visible"]/c["visible"],
            "visible_pck8_unconditional": c["correct8"]/c["visible"],
            "occlusion_false_accept_rate": c["false_accepts"]/c["invisible"],
            "event": b75.summarize(b75.aggregate(all_events[method]))}
    return {"kind": "BATCH76_RETROSPECTIVE_NATURAL_DEV_REDETECTION_STRESS",
            "split": "dev", "exploratory": True, "deployable": False,
            "source_commit": source["source_commit"],
            "public_predictions_sha256": digest(predict_path.read_bytes()),
            "oracle_scope": "36 named DEV scenes only",
            "test_oracle_read": False, "test_scoring_calls": 0,
            "methods": list(METHODS), "overall": overall, "per_scene": scene_metrics,
            "search_calls": sum(x["search_calls"] for x in cpu_search),
            "search_cpu_wall_ms": sum(x["search_ms"] for x in cpu_search),
            "note": ("Singleton margin is appearance uniqueness, NOT physical identity. "
                     "PCK8 reacquisition is spatial, NOT true object re-identification. "
                     "Previously studied DEV only; never tune on TEST." )}


def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--predict", action="store_true")
    group.add_argument("--score", action="store_true")
    p.add_argument("--predictions", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.predict:
        data = predict()
    else:
        if a.predictions is None:
            p.error("--score requires --predictions")
        data = evaluate(a.predictions)
    sha = immutable_json(a.output, data)
    print("BATCH76_PINNED", a.output, sha, flush=True)
    if a.score:
        for method, v in data["overall"].items():
            print("BATCH76_DEV_RESULT", method, json.dumps(v, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()