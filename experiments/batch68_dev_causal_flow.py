"""Batch68: label-blind causal one-step optical-flow verification on frozen DEV.

The last natural TEST is NOT reopened or adjusted. This retrospective experiment
may reject (never move) original tracker predictions. A one-step flow cycle is
not independent evidence of target identity or visibility: correlated flow may
continue tracking an occluder. Image-plane phase shifts are only diagnostic 2-D
proxies and must not be described as 3-D camera motion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from experiments import batch62_dev_consensus_veto as b62
from experiments import batch63_dev_temporal_consensus as b63
from experiments import batch64_dev_rgb_identity as b64

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "research"
FEATURES = OUTPUT / "batch68_unlabeled_dev_flow.json"
REPORT = OUTPUT / "batch68_dev_flow_consistency.json"

# Predeclared exploration grid. Never choose using TEST or re-fit on these
# already studied DEV outcomes. Ordered as (max tracker-vs-LK, max LK FB) px.
POLICIES = {
    "lk_2px_fb1px": (False, 2.0, 1.0),
    "lk_4px_fb2px": (False, 4.0, 2.0),
    "template_lk_2px_fb1px": (True, 2.0, 1.0),
    "template_lk_4px_fb2px": (True, 4.0, 2.0),
}
VARIANTS = ("baseline", "template_4px") + tuple(POLICIES)
LK_WINDOW = (17, 17)
LK_LEVEL = 2
PHASE_MIN_RESPONSE = 0.05
PHASE_DOWNSAMPLE = (128, 128)


def gray(frame: np.ndarray) -> np.ndarray:
    if frame.shape != (256, 256, 3) or frame.dtype != np.uint8:
        raise RuntimeError("BATCH68_RGB_SHAPE")
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def valid_point(point: object, border: float = 3.0) -> tuple[float, float] | None:
    v = b62.xy(point)
    if v is None:
        return None
    if not (border <= v[0] < 256-border and border <= v[1] < 256-border):
        return None
    return v


def phase_translation(g0: np.ndarray, g1: np.ndarray) -> dict | None:
    """Whole-image 2-D translation proxy (possibly foreground-dominated)."""
    if g0.shape != (256, 256) or g1.shape != g0.shape:
        raise RuntimeError("BATCH68_PHASE_SHAPE")
    im0 = cv2.resize(g0, PHASE_DOWNSAMPLE).astype(np.float32)
    im1 = cv2.resize(g1, PHASE_DOWNSAMPLE).astype(np.float32)
    # A constant scene has no identifiable shift. Hanning alone can yield
    # spuriously high phase-correlation response on a constant image.
    if float(im0.std()) < 4.0 or float(im1.std()) < 4.0:
        return None
    hann = cv2.createHanningWindow(PHASE_DOWNSAMPLE, cv2.CV_32F)
    (x, y), response = cv2.phaseCorrelate(im0, im1, hann)
    if not all(math.isfinite(v) for v in (x, y, response)):
        return None
    if response < PHASE_MIN_RESPONSE:
        return None
    return {"dx": 2*x, "dy": 2*y, "response": response}


def lk_evidence(g0: np.ndarray, g1: np.ndarray, prev_point: object,
                predicted_point: object, phase: dict | None) -> dict | None:
    """Use adjacent observed RGB frames, *not* GT, future frames or labels."""
    if g0.shape != (256, 256) or g1.shape != g0.shape:
        raise RuntimeError("BATCH68_LK_SHAPE")
    p = valid_point(prev_point)
    nxt = valid_point(predicted_point)
    if p is None or nxt is None:
        return None
    source = np.asarray([[[p[0], p[1]]]], dtype=np.float32)
    args = {"winSize": LK_WINDOW, "maxLevel": LK_LEVEL,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)}
    dst, ok, _ = cv2.calcOpticalFlowPyrLK(g0, g1, source, None, **args)
    if dst is None or ok is None or int(ok[0, 0]) != 1:
        return None
    back, b_ok, _ = cv2.calcOpticalFlowPyrLK(g1, g0, dst, None, **args)
    if back is None or b_ok is None or int(b_ok[0, 0]) != 1:
        return None
    q = tuple(float(x) for x in dst[0, 0])
    recovered = tuple(float(x) for x in back[0, 0])
    if not all(math.isfinite(v) for v in (*q, *recovered)):
        return None
    if not (3 <= q[0] < 253 and 3 <= q[1] < 253):
        return None
    out = {
        "tracker_flow_residual_px": math.dist(nxt, q),
        "flow_cycle_error_px": math.dist(p, recovered),
    }
    if phase is not None:
        # Candidate displacement minus dominant image translation, NOT actual
        # camera pose, object velocity, parallax or calibrated background flow.
        out["candidate_minus_global_2d_px"] = math.hypot(
            nxt[0] - p[0] - phase["dx"], nxt[1] - p[1] - phase["dy"])
    if not all(math.isfinite(v) and v >= 0 for v in out.values()):
        raise RuntimeError("BATCH68_FLOW_NONFINITE")
    return out


def per_scene_features(frames: np.ndarray, queries: list[dict],
                       pred_scene: dict) -> dict:
    n = pred_scene["nframes"]
    if frames.shape != (n, 256, 256, 3) or frames.dtype != np.uint8:
        raise RuntimeError("BATCH68_FRAMES_INVALID")
    qmap = {q["id"]: q for q in queries}
    if len(qmap) != 4 or len(queries) != 4 or set(qmap) != set(pred_scene["tracks"]):
        raise RuntimeError("BATCH68_PUBLIC_QUERY_POPULATION")
    grays = [gray(frames[t]) for t in range(n)]
    phase: list[dict | None] = [None]
    for t in range(1, n):
        phase.append(phase_translation(grays[t-1], grays[t]))
    out = {}
    for qid, query in qmap.items():
        t0 = query["query_time"]
        base = pred_scene["tracks"][qid][b62.BASE]
        if (type(t0) is not int or not 0 <= t0 < n-1 or
                len(base) != n or any(p is not None for p in base[:t0])):
            raise RuntimeError("BATCH68_TRACK_TIME_INVALID")
        samples: list[dict | None] = [None] * n
        for t in range(t0+1, n):
            p0 = query["query_xy"] if t == t0+1 else base[t-1]
            samples[t] = lk_evidence(grays[t-1], grays[t], p0, base[t], phase[t])
        out[qid] = samples
    return {"phase": phase, "query": out}


def extract(pred: dict, queries: dict, manifest: dict) -> dict:
    # The ONLY permitted phase-1 files are pinned unscored DEV predictions,
    # public queries, public manifest and public DEV frames.
    b64.validate_public(pred, queries, manifest)
    records = {r["name"]: r for r in pred["records"]}
    expected = {r["name"]: r for r in manifest["scenes"] if r["split"] == "dev"}
    features, hashes, timings = {}, {}, {}
    for scene in b62.SCENES:
        path = b62.DATA / "frames" / (scene + ".npy")
        digest = b64.sha_file(path)
        if digest != expected[scene]["frame_sha256"]:
            raise RuntimeError("BATCH68_PUBLIC_FRAME_PIN_DRIFT:" + scene)
        hashes[scene] = digest
        t0 = time.perf_counter()
        features[scene] = per_scene_features(
            np.load(path, mmap_mode="r"), queries[scene], records[scene])
        timings[scene] = (time.perf_counter()-t0)*1000
    return {
        "kind": "BATCH68_PINNED_LABEL_BLIND_CAUSAL_FLOW", "split": "dev",
        "source_commit": pred["source_commit"], "exploratory": True,
        "prediction_sha256": b64.sha_file(b62.IN / "dev_predictions.json"),
        "public_query_sha256": b64.sha_file(b62.DATA / "queries-public.json"),
        "manifest_sha256": b64.sha_file(b62.DATA / "manifest.json"),
        "frame_sha256_by_scene": hashes,
        "policies": POLICIES,
        "lk_window": LK_WINDOW, "lk_level": LK_LEVEL,
        "phase_min_response": PHASE_MIN_RESPONSE,
        "evidence_by_scene": features,
        "feature_cpu_ms_per_scene": timings,
        "dev_oracle_read_during_extraction": False,
        "test_predictions_read": False, "test_oracle_read": False,
    }


def accepted(track: dict, evidence: dict | None, t: int, variant: str) -> bool:
    if variant in ("baseline", "template_4px"):
        return b62.pass_veto(track, t, variant)
    if variant not in POLICIES:
        raise RuntimeError("BATCH68_UNKNOWN_POLICY")
    template, max_residual, max_cycle = POLICIES[variant]
    if not b62.pass_veto(track, t, "template_4px" if template else "baseline"):
        return False
    return bool(evidence is not None
                and evidence["tracker_flow_residual_px"] <= max_residual
                and evidence["flow_cycle_error_px"] <= max_cycle)


def score_scene(pred_scene: dict, queries: list[dict], oracle_scene: dict,
                evidence_scene: dict) -> dict:
    n = pred_scene["nframes"]
    qmap = {q["id"]: q for q in queries}
    signals = evidence_scene["query"]
    if (len(qmap) != 4 or len(queries) != 4
            or set(qmap) != set(pred_scene["tracks"])
            or set(qmap) != set(oracle_scene["tracks"])
            or set(qmap) != set(signals)):
        raise RuntimeError("BATCH68_SCORER_QUERY_DRIFT")
    counters = {name: b62.counts_empty() for name in VARIANTS}
    runs = {name: [] for name in VARIANTS}
    for qid, query in qmap.items():
        t0 = query["query_time"]
        tr = pred_scene["tracks"][qid]
        gt = oracle_scene["tracks"][qid]
        vals = signals[qid]
        if len(vals) != n or len(gt["xy"]) != n or len(gt["occluded"]) != n:
            raise RuntimeError("BATCH68_SCORER_SHAPE")
        if any(v is not None for v in vals[:t0+1]):
            raise RuntimeError("BATCH68_PREQUERY_EVIDENCE")
        for variant in VARIANTS:
            flags = [False] * n
            c = counters[variant]
            for t in range(t0+1, n):
                occ = gt["occluded"][t]
                if type(occ) is not bool:
                    raise RuntimeError("BATCH68_SCORER_OCCLUSION_TYPE")
                c["invisible" if occ else "visible"] += 1
                flags[t] = accepted(tr, vals[t], t, variant)
                if not flags[t]:
                    continue
                c["predicted_visible"] += 1
                if occ:
                    c["invisible_false_accept"] += 1
                else:
                    c["accepted_visible"] += 1
                    err = math.dist(b62.xy(tr[b62.BASE][t]), b62.xy(gt["xy"][t]))
                    c["correct8"] += int(err <= 8)
                    for rad in b62.RADII:
                        c[f"tp{rad}"] += int(err <= rad)
            runs[variant].extend(b63.false_accept_runs(flags, gt["occluded"], t0))
    return {
        k: {**b62.rates(counters[k]),
            "longest_false_accept_run": max(runs[k], default=0),
            "false_accept_runs_ge10": sum(x >= 10 for x in runs[k])}
        for k in VARIANTS
    }


def evaluate(pred: dict, sel: dict, dev: dict, queries: dict,
             features: dict, oracle: dict) -> dict:
    b62.validate_sources(pred, sel, dev, queries)
    if (features.get("kind") != "BATCH68_PINNED_LABEL_BLIND_CAUSAL_FLOW"
            or features.get("split") != "dev"
            or features.get("source_commit") != pred["source_commit"]
            or features.get("prediction_sha256") != b64.sha_file(b62.IN / "dev_predictions.json")
            or features.get("public_query_sha256") != b64.sha_file(b62.DATA / "queries-public.json")
            or features.get("manifest_sha256") != b64.sha_file(b62.DATA / "manifest.json")
            or features.get("policies") != json.loads(json.dumps(POLICIES))
            or features.get("dev_oracle_read_during_extraction") is not False
            or features.get("test_oracle_read") is not False
            or features.get("test_predictions_read") is not False
            or set(features.get("evidence_by_scene", {})) != set(b62.SCENES)
            or set(oracle) != set(b62.SCENES)):
        raise RuntimeError("BATCH68_FEATURE_PIN_OR_SPLIT_DRIFT")
    records = {row["name"]: row for row in pred["records"]}
    scenes = {}
    for scene in b62.SCENES:
        row = score_scene(records[scene], queries[scene], oracle[scene],
                          features["evidence_by_scene"][scene])
        for field in b62.counts_empty():
            if row["baseline"][field] != dev["per_scene"][scene][b62.BASE][field]:
                raise RuntimeError("BATCH68_BASELINE_COUNT_DRIFT:" + scene + ":" + field)
        for variant in VARIANTS:
            for field in ("predicted_visible", "accepted_visible",
                          "invisible_false_accept", "correct8"):
                if row[variant][field] > row["baseline"][field]:
                    raise RuntimeError("BATCH68_NON_VETO_METHOD")
        for variant, (template, _, _) in POLICIES.items():
            if template:
                for field in ("predicted_visible", "accepted_visible",
                              "invisible_false_accept", "correct8"):
                    if row[variant][field] > row["template_4px"][field]:
                        raise RuntimeError("BATCH68_NOT_TEMPLATE_SUBSET")
        scenes[scene] = row
    summary = {}
    for variant in VARIANTS:
        rows = [scenes[s][variant] for s in b62.SCENES]
        supported = [r for r in rows if r["invisible"] > 0]
        summary[variant] = {
            "scene_macro_occluded_false_accept_rate": b62.average(rows, "invisible_false_accept_rate"),
            "scene_macro_visible_coverage": b62.average(rows, "visible_coverage"),
            "scene_macro_jaccard": b62.average(rows, "jaccard_mean_1_2_4_8_16"),
            "occlusion_supported_scenes": len(supported),
            "violating_scenes": sum(r["invisible_false_accept_rate"] > b62.BUDGET for r in supported),
            "worst_scene_false_accept_rate": max(r["invisible_false_accept_rate"] for r in supported),
            "longest_false_accept_run": max(r["longest_false_accept_run"] for r in rows),
            "false_accept_frames_vetoed_vs_template": sum(
                scenes[s]["template_4px"]["invisible_false_accept"]
                - scenes[s][variant]["invisible_false_accept"] for s in b62.SCENES),
            "correct_visible_frames_lost_vs_template": sum(
                scenes[s]["template_4px"]["correct8"] - scenes[s][variant]["correct8"]
                for s in b62.SCENES),
        }
    return {
        "kind": "BATCH68_DEV_CAUSAL_LK_CONSISTENCY_FALSIFICATION",
        "split": "dev", "exploratory": True, "deployable": False,
        "test_predictions_read": False, "test_oracle_read": False,
        "features_pinned_before_dev_oracle_read": True,
        "selected_method_unchanged": True, "frozen_batch59_unchanged": True,
        "scene_count": len(scenes), "query_count": 144,
        "cpu_ms_feature_extraction_sum": sum(features["feature_cpu_ms_per_scene"].values()),
        "cpu_ms_feature_extraction_median_per_scene": statistics.median(
            features["feature_cpu_ms_per_scene"].values()),
        "timing_limit": "Offline optical-flow feature phase only; excludes tracker, video decode, hashing, orchestration",
        "motion_limit": "Global phase shift is only 2-D dominant image motion; NOT camera pose or true background motion",
        "summary": summary,
        "kinetics_17": scenes["kinetics_17"],
        "per_scene": scenes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("features", "score"))
    args = parser.parse_args()
    pred = b62.pinned_json(b62.IN / "dev_predictions.json")
    queries = json.loads((b62.DATA / "queries-public.json").read_bytes())
    if args.phase == "features":
        manifest = json.loads((b62.DATA / "manifest.json").read_bytes())
        features = extract(pred, queries, manifest)
        digest = b64.pin_write(FEATURES, features)
        print(json.dumps({"phase": "features", "sha256": digest,
                          "scenes": len(features["evidence_by_scene"])}))
    else:
        features = b62.pinned_json(FEATURES)
        sel = b62.pinned_json(b62.IN / "dev_selection.json")
        raw = (b62.IN / "dev_results.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != sel["dev_results_sha256"]:
            raise RuntimeError("BATCH68_DEV_SCORE_PIN_DRIFT")
        dev = json.loads(raw)
        oracle = b62.load_dev_only_oracle(b62.DATA / "oracle-private.json", b62.SCENES)
        report = evaluate(pred, sel, dev, queries, features, oracle)
        digest = b64.pin_write(REPORT, report)
        print(json.dumps({"phase": "score", "sha256": digest,
                          "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
