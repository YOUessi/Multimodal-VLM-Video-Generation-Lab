"""Prospectively frozen Batch34: 2.5D parallax and target identity.

DEV is the only diagnostic/selection split. TEST tracker predictions and
both model predictions are persisted before any answer scoring. The scoring
function cannot run without complete frozen output from every model.
"""
from __future__ import annotations

import argparse
import collections
import html
import json
import statistics
import time
from pathlib import Path

import numpy as np

import domain_shift_motion_eval as base
import parallax_identity_data as data
from affine_camera_tracker import tracker as affine_tracker
from domain_shift_camera_tracker import gated_label
from shape_aware_tracker import tracker as shape_tracker

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("plain", "affine", "shape_affine")


def estimate(video, query, cfg, method):
    start = time.perf_counter()
    try:
        if method == "shape_affine":
            raw = shape_tracker(ROOT / video, query, cfg)
        else:
            raw = affine_tracker(ROOT / video, query, cfg, method=method)
        choice, why = gated_label(raw, cfg)
        record = {
            "direction": raw["direction"],
            "raw_delta_xy": raw["screen_delta_xy"],
            "world_delta_xy": raw["delta_xy"],
            "missing": raw["missing"], "reason": raw["reason"],
            "registration": raw["registration"],
            "gated_direction": choice, "gate_reason": why,
            "accepted": choice is not None,
        }
    except Exception as error:
        record = {
            "direction": None, "raw_delta_xy": None, "world_delta_xy": None,
            "missing": None, "reason": type(error).__name__,
            "error_detail": str(error)[:160],
            "registration": None, "gated_direction": None,
            "gate_reason": "TRACKER_EXCEPTION", "accepted": False,
        }
    record["cpu_seconds"] = round(time.perf_counter() - start, 6)
    return record


def audit(cfg):
    manifest = base.manifest(cfg, verify_bytes=True)
    pairs = manifest["pairs"]
    if len(pairs) != cfg["test_pairs"] + cfg["dev_pairs"]:
        raise RuntimeError("INDEPENDENT_SCENE_PAIR_COUNT_DRIFT")
    if len(set(p["seed"] for p in pairs)) != len(pairs):
        raise RuntimeError("CROSS_SPLIT_SCENE_SEED_COLLISION")
    if manifest["config_sha256"] != data.configuration_hash(cfg):
        raise RuntimeError("FROZEN_GENERATOR_CONFIG_DRIFT")
    if manifest["synthetic_only"] is not True:
        raise RuntimeError("PROVENANCE_MUST_BE_SYNTHETIC")
    print("PARALLAX_MEDIA_AUDIT_PASS", manifest["source_sha256"],
          len(manifest["rows"]), "clips", flush=True)


def metrics(rows, all_predictions, method):
    records = [all_predictions[row["id"]]["predictions"][method] for row in rows]
    correct = sum(p["direction"] == row["gold_direction"]
                  for p, row in zip(records, rows))
    produced = sum(p["direction"] is not None for p in records)
    accepted = sum(p["accepted"] for p in records)
    gate_wrong = sum(p["accepted"] and p["gated_direction"] != row["gold_direction"]
                     for p, row in zip(records, rows))
    cost = [1000 * p["cpu_seconds"] for p in records]
    return {
        "total": len(rows), "correct": correct,
        "produced": produced, "abstained": len(rows) - produced,
        "gate_accepted": accepted, "gate_correct": accepted - gate_wrong,
        "gate_wrong": gate_wrong,
        "gate_precision": round((accepted - gate_wrong) / accepted, 5)
                         if accepted else None,
        "cpu_p50_ms": round(float(np.percentile(cost, 50)), 2),
        "cpu_p95_ms": round(float(np.percentile(cost, 95)), 2),
    }


def dev_only(cfg):
    source = base.manifest(cfg)
    rows = [row for row in source["rows"] if row["split"] == "dev"]
    results = {}
    for method in METHODS:
        raw = {}
        for row in rows:
            raw[row["id"]] = {"predictions": {method: estimate(
                row["video"], row["query"], cfg, method)}}
        results[method] = {
            "all": metrics(rows, raw, method),
            "per_mode": {mode: metrics(
                [r for r in rows if r["variant"] == mode], raw, method
            ) for mode in data.MODES},
        }
    base.put(ROOT / cfg["output_dir"] / "DEV_ONLY.json",
             {"dev_only": True, "test_score_used": False, "metrics": results})
    print("PARALLAX_DEV_ONLY", json.dumps(results, sort_keys=True), flush=True)


def predict_test(cfg):
    source = base.manifest(cfg)
    rows = [row for row in source["rows"] if row["split"] == "test"]
    dest = ROOT / cfg["output_dir"] / "cv_predictions.json"
    if dest.exists():
        doc = json.loads(dest.read_text())
        if doc["source_sha256"] != source["source_sha256"]:
            raise RuntimeError("FROZEN_TEST_PREDICTIONS_SOURCE_DRIFT")
        if set(r["id"] for r in doc["records"]) != set(r["id"] for r in rows):
            raise RuntimeError("FROZEN_TEST_PREDICTIONS_COVERAGE_DRIFT")
        if doc["methods"] != list(METHODS):
            raise RuntimeError("FROZEN_METHODS_DRIFT")
        print("PARALLAX_FROZEN_TEST_PREDICTIONS_REUSED", flush=True)
        return
    results = []
    for row in rows:
        # Only use ID, video bytes and query. Gold and options are not
        # accessed in this stage; the scorer receives them much later.
        predictions = {method: estimate(row["video"], row["query"], cfg, method)
                       for method in METHODS}
        results.append({"id": row["id"], "pair_id": row["pair_id"],
                        "predictions": predictions})
        if len(results) % 40 == 0:
            print("PARALLAX_LABEL_BLIND_TEST_VIDEO", len(results),
                  len(rows), flush=True)
    base.put(dest, {
        "source_sha256": source["source_sha256"],
        "methods": list(METHODS), "prediction_count": len(results),
        "no_gold_labels_in_prediction_records": True,
        "records": results,
    })
    print("PARALLAX_CV_PREDICTIONS_FROZEN", len(results), flush=True)


def reverse_consistency(rows, predicted, method):
    pairs = collections.defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)
    both, consistent = 0, 0
    for pair in pairs.values():
        a = next(r for r in pair if not r["reversed"])
        b = next(r for r in pair if r["reversed"])
        left = predicted[a["id"]]["predictions"][method]["direction"]
        right = predicted[b["id"]]["predictions"][method]["direction"]
        if left is not None and right is not None:
            both += 1
            consistent += int(data.OPPOSITE[left] == right)
    return {"source_scene_pairs": len(pairs),
            "two_predictions_available": both,
            "opposite_consistent": consistent}


def report(cfg):
    source = base.manifest(cfg, verify_bytes=True)
    rows = [row for row in source["rows"] if row["split"] == "test"]
    pred_doc = base.read(Path(cfg["output_dir"]) / "cv_predictions.json")
    if pred_doc["source_sha256"] != source["source_sha256"] or (
            pred_doc["methods"] != list(METHODS)):
        raise RuntimeError("UNFROZEN_TEST_TRACKING")
    predicted = {row["id"]: row for row in pred_doc["records"]}
    if len(predicted) != len(rows) or set(predicted) != {r["id"] for r in rows}:
        raise RuntimeError("INCOMPLETE_TEST_TRACKING")
    models = {}
    for model_name, budgets in cfg["frame_budgets"].items():
        doc = base.read(Path(cfg["output_dir"]) / (model_name + ".json"))
        if doc["source_sha256"] != source["source_sha256"]:
            raise RuntimeError("VLM_SOURCE_SHA_MISMATCH")
        lookup = {(r["id"], r["frames"]): r for r in doc["records"]}
        if len(lookup) != len(rows) * len(budgets) or (
                set(lookup) != {(r["id"], b) for r in rows for b in budgets}):
            raise RuntimeError("VLM_FROZEN_OUTPUT_INCOMPLETE")
        models[model_name] = {
            "lookup": lookup,
            "gpu_peak_allocated_gib": doc["peak_allocated_GPU_gib"],
        }
    truth = {r["id"]: r["gold_direction"] for r in rows}
    pair_ids = {r["id"]: r["pair_id"] for r in rows}
    scored = {method: {
        row["id"]: predicted[row["id"]]["predictions"][method]["direction"]
                   == row["gold_direction"] for row in rows
    } for method in METHODS}
    result = {
        "source_sha256": source["source_sha256"],
        "synthetic_only": True, "real_world_transfer_unverified": True,
        "test_clips": len(rows), "independent_scene_pairs": cfg["test_pairs"],
        "methods": {}, "paired": {}, "models": {},
    }
    lines = [
        "# Frozen independent 2.5D parallax and identity study",
        "",
        "All videos are procedurally rendered, NOT natural captured camera footage.",
        f"Source manifest: {source['source_sha256']}. "
        f"Test: {len(rows)} videos / {cfg['test_pairs']} source pairs; "
        f"DEV: {2*cfg['dev_pairs']} videos.",
        "World-coordinate target direction is NOT apparent screen motion. "
        "Reverse clips belong to the same source pair.",
        "Every TEST prediction and 500M/2.2B output was saved before scoring.",
        "",
        "## Direct motion, confidence gate and CPU risk",
        "",
        "| Method | Mode | Correct | Emits | Abstains | Gate accepts | "
        "Gate correct | Gate wrong | CPU P50/P95 ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        overall = metrics(rows, predicted, method)
        slices = {mode: metrics([r for r in rows if r["variant"] == mode],
                                 predicted, method) for mode in data.MODES}
        result["methods"][method] = {
            "all": overall, "per_mode": slices,
            "reverse": reverse_consistency(rows, predicted, method),
        }
        for mode, s in [*slices.items(), ("ALL", overall)]:
            lines.append("| " + " | ".join([
                method, mode, f"{s['correct']}/{s['total']}",
                str(s["produced"]), str(s["abstained"]),
                str(s["gate_accepted"]), str(s["gate_correct"]),
                str(s["gate_wrong"]),
                f"{s['cpu_p50_ms']}/{s['cpu_p95_ms']}",
            ]) + " |")
    lines += [
        "", "## Scene-pair-cluster bootstrap of direct correctness",
        "",
        "| Comparison | Delta pp | 95% paired-scene CI pp |",
        "|---|---:|---|",
    ]
    for old in ("plain", "affine"):
        name = old + "_to_shape_affine"
        stat = base.bootstrap_scene(scored[old], scored["shape_affine"],
                                     pair_ids,
                                     seed=cfg["bootstrap_seed"] + len(old))
        result["paired"][name] = stat
        lines.append(f"| {name} | {stat['delta_pp']:+.2f} | "
                     f"{stat['cluster_bootstrap_95ci_pp']} |")
    lines += [
        "", "## Frozen pretrained model-only vs conservative tool fallback",
        "",
        "| Model | Frames | VLM only | Plain gate+VLM | "
        "Affine gate+VLM | Shape-affine gate+VLM |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_name, budgets in cfg["frame_budgets"].items():
        lookup = models[model_name]["lookup"]
        result["models"][model_name] = {
            "peak_allocated_gpu_gib": models[model_name]["gpu_peak_allocated_gib"]}
        for budget in budgets:
            vlm = {row["id"]: lookup[(row["id"], budget)]["prediction"]
                   == row["gold_letter"] for row in rows}
            item = {"base_correct": sum(vlm.values()), "hybrid": {}}
            cells = [model_name, str(budget), f"{sum(vlm.values())}/{len(rows)}"]
            for method in METHODS:
                hybrid = {}
                for row in rows:
                    record = predicted[row["id"]]["predictions"][method]
                    letter = (base.choice_for_direction(
                        row, record["gated_direction"]) if record["accepted"]
                        else None)
                    choice = letter if letter else lookup[(row["id"], budget)]["prediction"]
                    hybrid[row["id"]] = choice == row["gold_letter"]
                paired = base.bootstrap_scene(vlm, hybrid, pair_ids,
                                              seed=cfg["bootstrap_seed"]
                                              + budget + len(method))
                item["hybrid"][method] = {"correct": sum(hybrid.values()), **paired}
                cells.append(f"{sum(hybrid.values())}/{len(rows)}")
            result["models"][model_name][str(budget)] = item
            lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "", "## Boundaries and negative outcomes",
        "- The global ORB/RANSAC partial-affine mapping is *not* a 3D depth estimator. "
        "Multiple textured depth planes can have incompatible camera flows.",
        "- Shape matching cannot identify two identical-looking targets; deliberate "
        "abstention is preferable to confident identity switches.",
        "- Tool abstention counts as incorrect in direct full-denominator accuracy. "
        "Wrong high-confidence overrides are counted, not erased.",
        "- This synthetic test cannot establish licensed natural-video robustness "
        "or independent real-camera motion labels.",
        "- All thresholds were declared before TEST predictions and scoring. "
        "No retuning on this now-scored heldout is valid.",
    ]
    base.put(ROOT / cfg["output_dir"] / "results.json", result)
    out = ROOT / cfg["output_dir"] / "RESULTS.md"
    out.write_text("\n".join(lines) + "\n")
    print("PARALLAX_FIRST_HELDOUT_SCORE_COMPLETE", out, flush=True)


def viewer(cfg):
    source = base.manifest(cfg)
    test = [r for r in source["rows"]
            if r["split"] == "test" and not r["reversed"]]
    selected, groups = collections.Counter(), []
    for row in test:
        mode = row["variant"]
        if selected[mode] >= 3:
            continue
        selected[mode] += 1
        partner = next(x for x in source["rows"]
                       if x["pair_id"] == row["pair_id"] and x["reversed"])
        previews = "".join(
            "<video width='310' controls preload='metadata' src='"
            + html.escape((ROOT / x["video"]).resolve().as_uri(), quote=True)
            + "'></video>" for x in (row, partner))
        groups.append("<section><h2>" + html.escape(mode)
                      + "</h2>" + previews + "</section>")
    page = "<!doctype html><meta charset='utf-8'><title>Local parallax review</title>" \
           "<style>body{font:16px sans-serif;max-width:950px;margin:auto}" \
           "video{margin:10px}section{border-bottom:1px solid #aaa}</style>" \
           "<h1>Paired world-relative target motion: local review</h1>" \
           + "".join(groups)
    dest = ROOT / cfg["output_dir"] / "review.html"
    dest.write_text(page)
    print("LOCAL_ONLY_PARALLAX_VIEWER", dest, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=[
        "audit", "dev", "tracker", "model", "report", "viewer"])
    parser.add_argument("--config", default="configs/parallax_identity_v1.json")
    parser.add_argument("--model", choices=["500m", "2b"])
    args = parser.parse_args()
    cfg = base.cfg_load(args.config)
    if args.stage == "audit":
        audit(cfg)
    elif args.stage == "dev":
        dev_only(cfg)
    elif args.stage == "tracker":
        predict_test(cfg)
    elif args.stage == "model":
        if args.model is None:
            raise RuntimeError("MISSING_MODEL_NAME")
        base.evaluate(cfg, args.model)
    elif args.stage == "report":
        report(cfg)
    else:
        viewer(cfg)


if __name__ == "__main__":
    main()
