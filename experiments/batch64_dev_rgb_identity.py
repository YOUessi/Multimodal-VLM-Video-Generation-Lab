"""Batch64: DEV-only causal RGB appearance identity veto.

Exploratory after Batch62/63 negative findings. Unlabeled RGB evidence is pinned
BEFORE reading any DEV oracle. TEST videos/predictions/labels are never opened.
No tuning or changes to frozen Batch59. Appearance is not proof of identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np

from experiments import batch62_dev_consensus_veto as b62
from experiments import batch63_dev_temporal_consensus as b63

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/research"
FEATURES = OUT / "batch64_unlabeled_dev_rgb_identity.json"
REPORT = OUT / "batch64_dev_rgb_identity_report.json"
BINS = 8
RADIUS = 7
# A priori descriptive grid, NOT fitted after looking at labels.
THRESHOLDS = (0.55, 0.70, 0.85)
VARIANTS = ("baseline", "template_4px") + tuple(
    f"{prefix}_{threshold:.2f}" for prefix in ("rgb", "template_rgb")
    for threshold in THRESHOLDS
)


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pin_write(path: Path, obj: object) -> str:
    pin = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or pin.exists():
        raise RuntimeError("BATCH64_NO_OVERWRITE")
    data = canonical(obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(data)
    digest = hashlib.sha256(data).hexdigest()
    with pin.open("x") as f:
        f.write(digest + "\n")
    return digest


def pinned(path: Path) -> dict:
    return b62.pinned_json(path)


def histogram(frame: np.ndarray, point: object) -> np.ndarray | None:
    """8-bin/channel local RGB histogram, normalized; no labels, no learning."""
    coords = b62.xy(point)
    if coords is None:
        return None
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise RuntimeError("BATCH64_BAD_FRAME")
    x, y = coords
    if x < 0 or y < 0 or x >= frame.shape[1] or y >= frame.shape[0]:
        return None
    cx, cy = int(round(x)), int(round(y))
    x0, x1 = max(0, cx-RADIUS), min(frame.shape[1], cx+RADIUS+1)
    y0, y1 = max(0, cy-RADIUS), min(frame.shape[0], cy+RADIUS+1)
    patch = frame[y0:y1, x0:x1]
    if patch.shape[0] < RADIUS or patch.shape[1] < RADIUS:
        return None
    bins = (patch.reshape(-1, 3).astype(np.int32) * BINS) // 256
    h = np.concatenate([
        np.bincount(bins[:, c], minlength=BINS).astype(np.float64)
        for c in range(3)
    ])
    return h / (3 * patch.shape[0] * patch.shape[1])


def overlap(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    if a.shape != (24,) or b.shape != (24,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError("BATCH64_BAD_HISTOGRAM")
    val = float(np.minimum(a, b).sum())
    if not 0 <= val <= 1.0000001:
        raise RuntimeError("BATCH64_BAD_OVERLAP")
    return min(1.0, val)


def frame_scores(frames: np.ndarray, queries: list[dict], pred_scene: dict) -> dict:
    """Only query RGB and current frame RGB at original frozen tracker coords."""
    n = pred_scene["nframes"]
    if frames.ndim != 4 or frames.shape != (n, 256, 256, 3) or frames.dtype != np.uint8:
        raise RuntimeError("BATCH64_FRAME_SHAPE")
    qmap = {q["id"]: q for q in queries}
    if len(qmap) != 4 or set(qmap) != set(pred_scene["tracks"]):
        raise RuntimeError("BATCH64_QUERY_POPULATION")
    results = {}
    for qid, q in qmap.items():
        t0 = q["query_time"]
        if type(t0) is not int or not 0 <= t0 < n-1:
            raise RuntimeError("BATCH64_BAD_QUERY_TIME")
        track = pred_scene["tracks"][qid][b62.BASE]
        if len(track) != n or any(p is not None for p in track[:t0]):
            raise RuntimeError("BATCH64_BAD_TRACK")
        source = histogram(frames[t0], q["query_xy"])
        if source is None:
            raise RuntimeError("BATCH64_BAD_REFERENCE")
        scores: list[float | None] = [None] * n
        for t in range(t0 + 1, n):
            scores[t] = overlap(source, histogram(frames[t], track[t]))
        results[qid] = scores
    return results


def validate_public(pred: dict, sel: dict, dev: dict, queries: dict, manifest: dict) -> None:
    b62.validate_sources(pred, sel, dev, queries)
    records = manifest["scenes"]
    items = {r["name"]: r for r in records if r["split"] == "dev"}
    if set(items) != set(b62.SCENES) or any(
            items[s]["frame_file"] != f"data/tapvid_kinetics_batch56/batch56_frozen/frames/{s}.npy"
            for s in b62.SCENES):
        raise RuntimeError("BATCH64_DEV_MANIFEST_DRIFT")
    if any(int(items[s]["query_count"]) != 4 for s in items):
        raise RuntimeError("BATCH64_QUERY_COUNT_DRIFT")


def extract(pred: dict, sel: dict, dev: dict, queries: dict, manifest: dict) -> dict:
    validate_public(pred, sel, dev, queries, manifest)
    by_scene = {}
    hashes = {}
    records = {r["name"]: r for r in pred["records"]}
    items = {r["name"]: r for r in manifest["scenes"] if r["split"] == "dev"}
    elapsed = []
    for scene in b62.SCENES:
        f = b62.DATA / "frames" / (scene + ".npy")
        digest = sha_file(f)
        if digest != items[scene]["frame_sha256"]:
            raise RuntimeError("BATCH64_RGB_FRAME_PIN_DRIFT:" + scene)
        hashes[scene] = digest
        start = time.perf_counter()
        by_scene[scene] = frame_scores(np.load(f, mmap_mode="r"), queries[scene], records[scene])
        elapsed.append((time.perf_counter()-start) * 1000)
    return {
        "kind": "BATCH64_UNLABELED_DEV_RGB_IDENTITY",
        "split": "dev", "source_commit": pred["source_commit"],
        "prediction_sha256": sha_file(b62.IN / "dev_predictions.json"),
        "public_query_sha256": sha_file(b62.DATA / "queries-public.json"),
        "manifest_sha256": sha_file(b62.DATA / "manifest.json"),
        "fixed_bins": BINS, "fixed_radius": RADIUS,
        "thresholds_predeclared": list(THRESHOLDS),
        "frame_sha256_by_scene": hashes,
        "rgb_extraction_ms_per_scene": dict(zip(b62.SCENES, elapsed)),
        "score_by_scene": by_scene,
        "test_predictions_read": False, "test_oracle_read": False,
        "dev_oracle_read_during_extraction": False,
    }


def accepted(track: dict, rgb: float | None, t: int, variant: str) -> bool:
    if variant == "baseline":
        return b62.pass_veto(track, t, "baseline")
    if variant == "template_4px":
        return b62.pass_veto(track, t, "template_4px")
    prefix, threshold_text = variant.rsplit("_", 1)
    if prefix not in ("rgb", "template_rgb") or float(threshold_text) not in THRESHOLDS:
        raise RuntimeError("BATCH64_BAD_VARIANT")
    return (b62.pass_veto(track, t, "template_4px" if prefix == "template_rgb" else "baseline")
            and rgb is not None and rgb >= float(threshold_text))


def score_scene(pred_scene: dict, queries: list[dict], oracle_scene: dict, feature_scene: dict) -> dict:
    n = pred_scene["nframes"]
    qmap = {q["id"]: q for q in queries}
    if len(qmap) != 4 or set(qmap) != set(oracle_scene["tracks"]) or set(qmap) != set(feature_scene):
        raise RuntimeError("BATCH64_SCORING_POPULATION")
    counts = {v: b62.counts_empty() for v in VARIANTS}
    runs = {v: [] for v in VARIANTS}
    for qid, q in qmap.items():
        t0 = q["query_time"]
        tr = pred_scene["tracks"][qid]
        oracle = oracle_scene["tracks"][qid]
        scores = feature_scene[qid]
        if len(scores) != n or len(oracle["xy"]) != n or len(oracle["occluded"]) != n or any(
                v is not None for v in scores[:t0+1]):
            raise RuntimeError("BATCH64_SCORING_SHAPE")
        for v in VARIANTS:
            flags = [False] * n
            c = counts[v]
            for t in range(t0 + 1, n):
                occ = oracle["occluded"][t]
                if type(occ) is not bool:
                    raise RuntimeError("BATCH64_VISIBILITY_TYPE")
                c["invisible" if occ else "visible"] += 1
                flags[t] = accepted(tr, scores[t], t, v)
                if flags[t]:
                    c["predicted_visible"] += 1
                    if occ:
                        c["invisible_false_accept"] += 1
                    else:
                        c["accepted_visible"] += 1
                        e = math.dist(b62.xy(tr[b62.BASE][t]), b62.xy(oracle["xy"][t]))
                        c["correct8"] += int(e <= 8)
                        for radius in b62.RADII:
                            c[f"tp{radius}"] += int(e <= radius)
            runs[v].extend(b63.false_accept_runs(flags, oracle["occluded"], t0))
    return {v: {
        **b62.rates(counts[v]),
        "max_false_accept_episode_frames": max(runs[v], default=0),
        "false_accept_episodes_ge_16": sum(x >= 16 for x in runs[v]),
    } for v in VARIANTS}


def evaluate(pred: dict, sel: dict, dev: dict, queries: dict, features: dict, oracle: dict) -> dict:
    b62.validate_sources(pred, sel, dev, queries)
    if (features.get("kind") != "BATCH64_UNLABELED_DEV_RGB_IDENTITY" or
        features["source_commit"] != pred["source_commit"] or
        features["prediction_sha256"] != sha_file(b62.IN / "dev_predictions.json") or
        features["public_query_sha256"] != sha_file(b62.DATA / "queries-public.json") or
        features["manifest_sha256"] != sha_file(b62.DATA / "manifest.json") or
        set(features["score_by_scene"]) != set(b62.SCENES) or
        set(oracle) != set(b62.SCENES) or
        features["dev_oracle_read_during_extraction"] is not False):
        raise RuntimeError("BATCH64_FEATURE_OR_SPLIT_DRIFT")
    records = {r["name"]: r for r in pred["records"]}
    per_scene = {}
    for scene in b62.SCENES:
        per_scene[scene] = score_scene(records[scene], queries[scene], oracle[scene],
                                       features["score_by_scene"][scene])
        baseline = per_scene[scene]["baseline"]
        old = dev["per_scene"][scene][b62.BASE]
        for field in b62.counts_empty():
            if baseline[field] != old[field]:
                raise RuntimeError("BATCH64_BASELINE_REPLAY_DRIFT:" + scene + ":" + field)
        # RGB can only veto: it cannot create new predictions or increase correct matches.
        for v in VARIANTS:
            for k in ("accepted_visible", "invisible_false_accept", "correct8"):
                if per_scene[scene][v][k] > baseline[k]:
                    raise RuntimeError("BATCH64_NONVETO_FAILURE")
        for threshold in THRESHOLDS:
            v = f"template_rgb_{threshold:.2f}"
            for k in ("accepted_visible", "invisible_false_accept", "correct8"):
                if per_scene[scene][v][k] > per_scene[scene]["template_4px"][k]:
                    raise RuntimeError("BATCH64_TEMPLATE_SUBSET_FAILURE")
    summary = {}
    for v in VARIANTS:
        rows = [per_scene[s][v] for s in b62.SCENES]
        supported = [s for s in b62.SCENES if per_scene[s][v]["invisible"]]
        bad = [s for s in supported if per_scene[s][v]["invisible_false_accept_rate"] > b62.BUDGET]
        summary[v] = {
            "micro": b62.rates(b62.add(rows)),
            "scene_macro_visible_coverage": b62.average(rows, "visible_coverage"),
            "scene_macro_occluded_false_accept_rate": b62.average(rows, "invisible_false_accept_rate"),
            "scene_macro_jaccard": b62.average(rows, "jaccard_mean_1_2_4_8_16"),
            "occlusion_supported_scenes": len(supported),
            "scene_budget_violation_count": len(bad),
            "scene_budget_violation_ids": bad,
            "worst_scene_false_accept": max((per_scene[s][v]["invisible_false_accept_rate"] for s in supported), default=None),
            "max_false_accept_episode_frames": max(per_scene[s][v]["max_false_accept_episode_frames"] for s in b62.SCENES),
            "paired_vetoed_occluded_vs_template": sum(
                per_scene[s]["template_4px"]["invisible_false_accept"] - per_scene[s][v]["invisible_false_accept"]
                for s in b62.SCENES),
            "paired_lost_correct_vs_template": sum(
                per_scene[s]["template_4px"]["correct8"] - per_scene[s][v]["correct8"]
                for s in b62.SCENES),
        }
    return {
        "kind": "BATCH64_DEV_RGB_IDENTITY_EXPLORATORY_FALSIFICATION",
        "split": "dev", "exploratory": True, "deployable": False,
        "test_oracle_read": False, "test_predictions_read": False,
        "features_pinned_before_dev_oracle_read": True,
        "selected_method_unchanged": True, "frozen_batch59_unchanged": True,
        "data_limitations": ("Already-studied DEV; appearance histograms cannot prove "
                             "identity or distinguish similar foreground/background. "
                             "Framewise RGB CPU timings exclude decode and full tracker costs."),
        "summary": summary,
        "kinetics_17": per_scene["kinetics_17"],
        "rgb_extraction_ms_per_scene": features["rgb_extraction_ms_per_scene"],
        "per_scene": per_scene,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("features", "score"))
    args = p.parse_args()
    pred = pinned(b62.IN / "dev_predictions.json")
    sel = pinned(b62.IN / "dev_selection.json")
    raw = (b62.IN / "dev_results.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != sel["dev_results_sha256"]:
        raise RuntimeError("BATCH64_DEV_RESULTS_PIN_DRIFT")
    dev = json.loads(raw)
    queries = json.loads((b62.DATA / "queries-public.json").read_bytes())
    if args.phase == "features":
        manifest = json.loads((b62.DATA / "manifest.json").read_bytes())
        payload = extract(pred, sel, dev, queries, manifest)
        print(json.dumps({"phase": "features", "sha256": pin_write(FEATURES, payload),
                          "scenes": len(payload["score_by_scene"])}))
    else:
        features = pinned(FEATURES)
        oracle = b62.load_dev_only_oracle(b62.DATA / "oracle-private.json", b62.SCENES)
        report = evaluate(pred, sel, dev, queries, features, oracle)
        digest = pin_write(REPORT, report)
        print(json.dumps({"phase": "score", "sha256": digest,
                          "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
