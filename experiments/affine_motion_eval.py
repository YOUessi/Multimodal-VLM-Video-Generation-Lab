"""Frozen 4-mode affine-camera test: labels seen only in dev/report stages.

Stages: audit -> dev -> tracker -> model (500m,2b) -> report -> viewer.
Tracker stage serializes immutable pixel-only predictions BEFORE test scoring.
"""
from __future__ import annotations
import argparse
import collections
import json
import statistics
import time
from pathlib import Path
import numpy as np
import cv2

import affine_domain_data as data
import domain_shift_motion_eval as base
from affine_camera_tracker import tracker as method_tracker
from domain_shift_camera_tracker import gated_label

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("plain", "phase", "affine")


def prediction(video, question, cfg, method):
    start = time.perf_counter()
    try:
        item = method_tracker(ROOT / video, question, cfg, method=method)
        label, reason = gated_label(item, cfg)
        result = {
            "direction": item["direction"],
            "raw_delta_xy": item["screen_delta_xy"],
            "world_delta_xy": item["delta_xy"],
            "missing": item["missing"],
            "reason": item["reason"],
            "gate_reason": reason, "gated_direction": label,
            "accepted": label is not None, "registration": item["registration"],
        }
    except Exception as exc:
        result = {
            "direction": None, "raw_delta_xy": None, "world_delta_xy": None,
            "missing": None, "reason": type(exc).__name__,
            "detail": str(exc)[:200], "gate_reason": "TRACKER_EXCEPTION",
            "gated_direction": None, "accepted": False, "registration": None,
        }
    result["cpu_seconds"] = round(time.perf_counter() - start, 6)
    return result


def development(cfg):
    source = base.manifest(cfg)
    rows = [r for r in source["rows"] if r["split"] == "dev"]
    results = {}
    for method in METHODS:
        total = collections.Counter()
        by_mode = {m: collections.Counter() for m in data.MODES}
        seconds = []
        for r in rows:
            x = prediction(r["video"], r["query"], cfg, method)
            stats = by_mode[r["variant"]]
            total["total"] += 1
            stats["total"] += 1
            for c in (total, stats):
                c["correct"] += int(x["direction"] == r["gold_direction"])
                c["produced"] += int(x["direction"] is not None)
                c["accepted"] += int(x["accepted"])
                c["accepted_wrong"] += int(x["accepted"] and x["gated_direction"] != r["gold_direction"])
            seconds.append(x["cpu_seconds"])
        results[method] = {"aggregate": dict(total),
                           "per_mode": {k: dict(v) for k, v in by_mode.items()},
                           "cpu_median_ms": round(1000 * statistics.median(seconds), 2)}
    print("AFFINE_DEV_ONLY_NO_TEST_LABELS", json.dumps(results, sort_keys=True), flush=True)
    return results


def predict_test(cfg):
    source = base.manifest(cfg)
    rows = [r for r in source["rows"] if r["split"] == "test"]
    dest = ROOT / cfg["output_dir"] / "cv_predictions.json"
    if dest.exists():
        saved = json.loads(dest.read_text())
        if saved["source_sha256"] != source["source_sha256"] or len(saved["records"]) != len(rows):
            raise RuntimeError("FROZEN_AFFINE_PREDICTIONS_DRIFT")
        return saved
    records = []
    for r in rows:
        records.append({"id": r["id"], "pair_id": r["pair_id"],
                        "predictions": {m: prediction(r["video"], r["query"], cfg, m)
                                        for m in METHODS}})
        if len(records) % 40 == 0:
            print("AFFINE_TEST_LABEL_BLIND_TRACKING", len(records), len(rows), flush=True)
    doc = {"source_sha256": source["source_sha256"], "sample_count": len(rows),
           "gold_labels_used_by_trackers": False, "methods": list(METHODS),
           "records": records}
    base.put(dest, doc)
    return doc


def vlm_table_row(model_name, budget, result, total):
    """A six-column GitHub Markdown row; never concatenate cells without a pipe."""
    cells = [model_name, str(budget), f"{result['base']}/{total}"]
    cells.extend(f"{result['hybrids'][method]['correct']}/{total}" for method in METHODS)
    return "| " + " | ".join(cells) + " |"


def method_scores(method, items, preds, cfg):
    correct = {r["id"]: preds[r["id"]]["predictions"][method]["direction"] == r["gold_direction"]
               for r in items}
    accepted = [preds[r["id"]]["predictions"][method] for r in items
                if preds[r["id"]]["predictions"][method]["accepted"]]
    labels = {r["id"]: r["gold_direction"] for r in items}
    wrong = sum(
        preds[rid]["predictions"][method]["gated_direction"] != correct_label
        for rid, correct_label in labels.items()
        if preds[rid]["predictions"][method]["accepted"]
    )
    cpu = [preds[r["id"]]["predictions"][method]["cpu_seconds"] * 1000 for r in items]
    produced = sum(preds[r["id"]]["predictions"][method]["direction"] is not None for r in items)
    return {
        "total": len(items), "correct": sum(correct.values()), "produced": produced,
        "abstained": len(items) - produced, "gate_accepted": len(accepted),
        "gate_correct": len(accepted) - wrong, "gate_wrong": wrong,
        "gate_precision": (len(accepted) - wrong) / len(accepted) if accepted else None,
        "cpu_p50_ms": round(float(np.percentile(cpu, 50)), 2),
        "cpu_p95_ms": round(float(np.percentile(cpu, 95)), 2),
    }


def reverse_consistency(rows, pred, method):
    grouped = collections.defaultdict(list)
    for r in rows:
        grouped[r["pair_id"]].append(r)
    both, consistent, one_missing = 0, 0, 0
    for group in grouped.values():
        forward = next(x for x in group if not x["reversed"])
        reverse = next(x for x in group if x["reversed"])
        a = pred[forward["id"]]["predictions"][method]["direction"]
        b = pred[reverse["id"]]["predictions"][method]["direction"]
        if a is None or b is None:
            one_missing += 1
        else:
            both += 1
            consistent += int(data.OPPOSITE[a] == b)
    return {"independent_scene_pairs": len(grouped), "both_directions_produced": both,
            "reverse_consistent": consistent, "missing_at_least_one_direction": one_missing}



def registration_residuals(source, rows, predictions, cfg):
    """Post-hoc ground-truth CAMERA geometry only, not available at inference.

    Report median mapped-point pixel error in the declared stationary-world
    camera plane. This measures camera-estimator fidelity, not target ID.
    """
    pair_info = {r["pair_id"]: r for r in source["source_pairs"]}
    sample_points = np.array([
        [64.0, 64.0, 1.0], [192.0, 64.0, 1.0],
        [64.0, 192.0, 1.0], [192.0, 192.0, 1.0],
    ], dtype=np.float64).T
    out = {}
    for method in METHODS:
        per_mode = {}
        for mode in data.MODES:
            errors = []
            for row in rows:
                if row["variant"] != mode:
                    continue
                params = pair_info[row["pair_id"]]["camera_params"]
                true_forward = data.affine_for_frame(params, cfg["frames"] - 1,
                                                       cfg["height"], cfg["width"])
                true_map = cv2.invertAffineTransform(true_forward) if row["reversed"] else true_forward
                item = predictions[row["id"]]["predictions"][method]
                if method == "plain":
                    measured = np.array([[1., 0., 0.], [0., 1., 0.]])
                elif method == "phase":
                    reg = item.get("registration")
                    if not reg or not reg.get("registration_accepted"):
                        continue
                    dx, dy = reg["background_shift"]
                    measured = np.array([[1., 0., dx], [0., 1., dy]])
                else:
                    reg = item.get("registration")
                    if not reg or not reg.get("accepted"):
                        continue
                    measured = np.asarray(reg["matrix"])
                expected = true_map @ sample_points
                actual = measured @ sample_points
                errors.append(float(np.median(np.linalg.norm(expected - actual, axis=0))))
            per_mode[mode] = {
                "registered_clips": len(errors),
                "all_clips": sum(r["variant"] == mode for r in rows),
                "median_mapping_error_px": round(float(np.median(errors)), 3) if errors else None,
                "p95_mapping_error_px": round(float(np.percentile(errors, 95)), 3) if errors else None,
            }
        out[method] = per_mode
    return out


def score(cfg):
    source = base.manifest(cfg, verify_bytes=True)
    test = [r for r in source["rows"] if r["split"] == "test"]
    pred_doc = base.read(Path(cfg["output_dir"]) / "cv_predictions.json")
    if pred_doc["source_sha256"] != source["source_sha256"]:
        raise RuntimeError("FROZEN_CV_PREDICTION_SOURCE_DRIFT")
    preds = {r["id"]: r for r in pred_doc["records"]}
    if set(preds) != {r["id"] for r in test} or pred_doc["methods"] != list(METHODS):
        raise RuntimeError("FROZEN_CV_PREDICTION_COVERAGE_DRIFT")
    summary = {
        "source_sha256": source["source_sha256"], "synthetic_only": True,
        "natural_camera_generalization": False, "test_clips": len(test),
        "scene_pairs": cfg["test_pairs"], "methods": {}, "paired": {}, "models": {},
    }
    pair_ids = {r["id"]: r["pair_id"] for r in test}
    summary["camera_geometry_posthoc_oracle_only"] = registration_residuals(
        source, test, preds, cfg
    )
    truth = {r["id"]: r["gold_direction"] for r in test}
    scored = {
        m: {r["id"]: preds[r["id"]]["predictions"][m]["direction"] == truth[r["id"]]
            for r in test} for m in METHODS
    }
    lines = [
        "# Independent affine/ambiguity heldout synthetic-camera benchmark",
        "",
        f"Frozen source SHA256: `{source['source_sha256']}`.",
        f"Total: {len(test)} heldout MP4 clips / {cfg['test_pairs']} independent scene pairs, plus "
        f"{cfg['dev_pairs']*2} independent development MP4 clips.",
        "Five balanced world-relative motion classes; forward/reversed pairs are dependent.",
        "All camera estimators consume decoded source frames and query only; answer labels enter only this report.",
        "**All videos are generated 2D scenes, NOT captured real-world footage.**",
        "",
        "## Pixel motion versus stationary-world-relative motion",
        "",
        "| Method | Mode | Correct / all | Direction coverage | Abstain | Gate accepted | Gate right | Gate wrong | CPU P50/P95 (ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in METHODS:
        summary["methods"][m] = {
            "total": method_scores(m, test, preds, cfg),
            "per_mode": {}, "reverse": reverse_consistency(test, preds, m),
        }
        for mode in data.MODES:
            subset = [r for r in test if r["variant"] == mode]
            metric = method_scores(m, subset, preds, cfg)
            summary["methods"][m]["per_mode"][mode] = metric
            lines.append(
                f"| {m} | {mode} | {metric['correct']}/{metric['total']} "
                f"| {metric['produced']} | {metric['abstained']} "
                f"| {metric['gate_accepted']} | {metric['gate_correct']} "
                f"| {metric['gate_wrong']} | {metric['cpu_p50_ms']}/{metric['cpu_p95_ms']} |"
            )
        metric = summary["methods"][m]["total"]
        lines.append(f"| **{m}** | **ALL** | **{metric['correct']}/{metric['total']}** "
                     f"| {metric['produced']} | {metric['abstained']} "
                     f"| {metric['gate_accepted']} | {metric['gate_correct']} "
                     f"| {metric['gate_wrong']} | {metric['cpu_p50_ms']}/{metric['cpu_p95_ms']} |")
    lines += [
        "", "## Background camera mapping residual (post-hoc oracle, NOT seen by trackers)",
        "",
        "| Estimator | Nuisance | Registered / all | Median mapping error px | P95 mapping error px |",
        "|---|---|---:|---:|---:|",
    ]
    for method in METHODS:
        for mode, stat in summary["camera_geometry_posthoc_oracle_only"][method].items():
            lines.append(
                f"| {method} | {mode} | {stat['registered_clips']}/{stat['all_clips']} "
                f"| {stat['median_mapping_error_px']} | {stat['p95_mapping_error_px']} |"
            )
    lines += ["", "## Paired scene-cluster comparison: affine vs plain, affine vs translation-phase", "",
              "| Comparison | Slice | delta correct percentage points | 95% scene-pair bootstrap CI |",
              "|---|---|---:|---|"]
    for old in ("plain", "phase"):
        comparison = old + "_to_affine"
        summary["paired"][comparison] = {}
        for variant in ("ALL", *data.MODES):
            subset = test if variant == "ALL" else [r for r in test if r["variant"] == variant]
            ids = {r["id"]: pair_ids[r["id"]] for r in subset}
            stat = base.bootstrap_scene(
                {r["id"]: scored[old][r["id"]] for r in subset},
                {r["id"]: scored["affine"][r["id"]] for r in subset},
                ids, seed=cfg["bootstrap_seed"] + len(variant) + len(old),
            )
            summary["paired"][comparison][variant] = stat
            lines.append(
                f"| {comparison} | {variant} | {stat['delta_pp']:+.2f} "
                f"| {stat['cluster_bootstrap_95ci_pp']} |"
            )
    lines += ["", "## Frozen VLM-only and conservative geometry-tool fallback", "",
              "| Model | Frames | VLM-only correct | Plain gate+VLM | Phase gate+VLM | Affine gate+VLM |",
              "|---|---:|---:|---:|---:|---:|"]
    for model_name, budgets in cfg["frame_budgets"].items():
        doc = base.read(Path(cfg["output_dir"]) / (model_name + ".json"))
        if doc["source_sha256"] != source["source_sha256"]:
            raise RuntimeError("VLM_AFFINE_SOURCE_DRIFT")
        models = {(r["id"], r["frames"]): r for r in doc["records"]}
        if len(models) != len(test) * len(budgets):
            raise RuntimeError("VLM_AFFINE_INFERENCE_INCOMPLETE")
        model_times = [float(r["generate_gpu_s"]) for r in doc["records"]]
        summary["models"][model_name] = {
            "peak_allocated_GPU_gib": doc["peak_allocated_GPU_gib"],
            "model_generate_p50_s": round(float(np.percentile(model_times, 50)), 3),
            "model_generate_p95_s": round(float(np.percentile(model_times, 95)), 3),
        }
        for budget in budgets:
            vlm = {r["id"]: models[(r["id"], budget)]["prediction"] == r["gold_letter"]
                   for r in test}
            result = {"base": sum(vlm.values()), "hybrids": {}}
            for m in METHODS:
                gated = {}
                for r in test:
                    p = preds[r["id"]]["predictions"][m]
                    predicted_option = (base.choice_for_direction(r, p["gated_direction"])
                                        if p["accepted"] else None)
                    output = predicted_option if predicted_option is not None else models[(r["id"], budget)]["prediction"]
                    gated[r["id"]] = output == r["gold_letter"]
                gain = base.bootstrap_scene(vlm, gated, pair_ids,
                                            seed=cfg["bootstrap_seed"] + budget + len(m))
                result["hybrids"][m] = {"correct": sum(gated.values()), **gain}
            summary["models"][model_name][str(budget)] = result
            lines.append(vlm_table_row(model_name, budget, result, len(test)))
    lines += [
        "", "## Interpretation and invalid claims",
        "- All methods use the same older HSV target segmenter; same-color distractor can create identity switches.",
        "- RANSAC fits a single dominant affine background. Rotation/scale works only under planar-dominance and robust features; multi-depth parallax is NOT solved.",
        "- Gated wrong overrides MUST be counted. More tool coverage does not imply better precision.",
        "- Model scores are for synthetic-shape questions only. No natural-video or architecture-improvement claim.",
        "- The test set was not used to select thresholds; successful or negative results remain frozen.",
        "- Confidence intervals cluster forward/reversed clips by their independent source scene.",
        "- CPU values include per-method frame decode, registration, tracking; do not include GPU VLM inference.",
    ]
    out = ROOT / cfg["output_dir"]
    base.put(out / "results.json", summary)
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("AFFINE_HELDOUT_REPORT_COMPLETE", out / "RESULTS.md", flush=True)


def viewer(cfg):
    import html
    source = base.manifest(cfg)
    test = [r for r in source["rows"] if r["split"] == "test"]
    pairs = collections.defaultdict(list)
    for row in test:
        pairs[row["pair_id"]].append(row)
    selected = collections.defaultdict(int)
    blocks = []
    for group in pairs.values():
        row = next(r for r in group if not r["reversed"])
        mode = row["variant"]
        if selected[mode] >= 3:
            continue
        selected[mode] += 1
        a, b = sorted(group, key=lambda x: x["reversed"])
        def clip_html(item):
            uri = (ROOT / item["video"]).resolve().as_uri()
            return ("<video width='300' controls preload='metadata' src='"
                    + html.escape(uri, quote=True) + "'></video>")
        blocks.append("<section><h2>" + html.escape(mode) + " / "
                      + html.escape(row["pair_id"]) + "</h2>"
                      + clip_html(a) + clip_html(b) + "</section>")
    page = ("<!doctype html><meta charset='utf-8'><title>Affine motion review</title>"
            "<style>body{font:16px sans-serif;max-width:1000px;margin:35px auto}"
            "section{padding:15px;border-bottom:1px solid #ddd}video{margin-right:8px}</style>"
            "<h1>Local synthetic motion forward / reverse inspection</h1>"
            + "".join(blocks))
    out = ROOT / cfg["output_dir"] / "review.html"
    out.write_text(page)
    print("AFFINE_LOCAL_REVIEW", out, "SCENE_PAIRS", sum(selected.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["audit", "dev", "tracker", "model", "report", "viewer"])
    ap.add_argument("--config", default="configs/affine_camera_motion_v1.json")
    ap.add_argument("--model", choices=["500m", "2b"])
    args = ap.parse_args()
    cfg = base.cfg_load(args.config)
    if args.stage == "audit":
        manifest = base.manifest(cfg, verify_bytes=True)
        print("AFFINE_SOURCE_AUDITED", manifest["source_sha256"])
    elif args.stage == "dev":
        development(cfg)
    elif args.stage == "tracker":
        predict_test(cfg)
    elif args.stage == "model":
        if args.model is None:
            raise RuntimeError("MODEL_STAGE_MISSING_MODEL")
        base.evaluate(cfg, args.model)
    elif args.stage == "report":
        score(cfg)
    else:
        viewer(cfg)


if __name__ == "__main__":
    main()
