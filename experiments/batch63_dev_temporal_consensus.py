"""Batch63: causal persistence audit of per-frame consensus on frozen DEV only.

Exploratory falsification on the *already studied* Batch56 DEV data. This module
does not tune thresholds, alter Batch59, or access TEST labels/predictions.
The private mixed-split oracle is parsed only through the DEV-prefix guard
in Batch62. Frame count is not treated as the number of independent samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from experiments import batch62_dev_consensus_veto as b62

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/research/batch63_dev_temporal_consensus.json"
BUDGET = 0.10
# Fixed before re-reading oracle; all variants are causal (current/past only).
# "baseline" and single-frame "template_4px" replay the Batch62 audit.
VARIANTS = {
    "baseline": ("baseline", 1),
    "template_4px": ("template_4px", 1),
    "template_streak2": ("template_4px", 2),
    "template_streak3": ("template_4px", 3),
    "affine_streak2": ("affine_3px", 2),
    "dual_streak2": ("both_4px_3px", 2),
}
BOOTSTRAP_SEED = 6303
BOOTSTRAP_RESAMPLES = 2000


def causal_flags(track: dict, query_time: int, nframes: int,
                 evidence: str, streak: int) -> list[bool]:
    """No future evidence and no query-frame credit; gaps reset confidence."""
    if type(query_time) is not int or type(nframes) is not int or not 0 <= query_time < nframes - 1:
        raise RuntimeError("BATCH63_INVALID_QUERY")
    if evidence not in b62.GATES or type(streak) is not int or streak < 1:
        raise RuntimeError("BATCH63_INVALID_GATE")
    for method in (b62.BASE,) + tuple(key for key, _ in b62.GATES[evidence]):
        if method not in track or len(track[method]) != nframes:
            raise RuntimeError("BATCH63_INVALID_TRACE")
        if any(item is not None for item in track[method][:query_time]):
            raise RuntimeError("BATCH63_PREQUERY_LEAK")
    result = [False] * nframes
    consecutive = 0
    for t in range(query_time + 1, nframes):
        consecutive = consecutive + 1 if b62.pass_veto(track, t, evidence) else 0
        result[t] = consecutive >= streak
    return result


def false_accept_runs(accepted: list[bool], occluded: list[bool],
                      query_time: int) -> list[int]:
    """Consecutive erroneous acceptance within annotated occluded episodes."""
    if len(accepted) != len(occluded):
        raise RuntimeError("BATCH63_EPISODE_SHAPE")
    runs: list[int] = []
    length = 0
    for t in range(query_time + 1, len(accepted)):
        if type(occluded[t]) is not bool:
            raise RuntimeError("BATCH63_BAD_VISIBILITY")
        if occluded[t] and accepted[t]:
            length += 1
        elif length:
            runs.append(length)
            length = 0
    if length:
        runs.append(length)
    return runs


def score_scene(pred_scene: dict, queries: list[dict], oracle_scene: dict) -> dict:
    n = pred_scene["nframes"]
    if type(n) is not int or n < 2:
        raise RuntimeError("BATCH63_INVALID_FRAMES")
    qmap = {q["id"]: q for q in queries}
    if len(qmap) != 4 or set(qmap) != set(pred_scene["tracks"]) or set(qmap) != set(oracle_scene["tracks"]):
        raise RuntimeError("BATCH63_QUERY_POPULATION")
    counters = {name: b62.counts_empty() for name in VARIANTS}
    episodes = {name: [] for name in VARIANTS}
    for qid, q in qmap.items():
        t0 = q["query_time"]
        tr = pred_scene["tracks"][qid]
        gt = oracle_scene["tracks"][qid]
        if len(gt["occluded"]) != n or len(gt["xy"]) != n:
            raise RuntimeError("BATCH63_ORACLE_LENGTH")
        for name, (evidence, streak) in VARIANTS.items():
            flags = causal_flags(tr, t0, n, evidence, streak)
            episodes[name].extend(false_accept_runs(flags, gt["occluded"], t0))
            c = counters[name]
            for t in range(t0 + 1, n):
                occ = gt["occluded"][t]
                if type(occ) is not bool:
                    raise RuntimeError("BATCH63_BAD_VISIBILITY")
                visible = not occ
                c["visible" if visible else "invisible"] += 1
                if flags[t]:
                    c["predicted_visible"] += 1
                    if visible:
                        c["accepted_visible"] += 1
                        err = math.dist(b62.xy(tr[b62.BASE][t]), b62.xy(gt["xy"][t]))
                        c["correct8"] += int(err <= 8)
                        for radius in b62.RADII:
                            c[f"tp{radius}"] += int(err <= radius)
                    else:
                        c["invisible_false_accept"] += 1
    return {
        name: {
            **b62.rates(counters[name]),
            "false_accept_episode_count": len(episodes[name]),
            "max_false_accept_episode_frames": max(episodes[name], default=0),
            "false_accept_episodes_ge_8": sum(x >= 8 for x in episodes[name]),
            "false_accept_episodes_ge_16": sum(x >= 16 for x in episodes[name]),
        }
        for name in VARIANTS
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise RuntimeError("BATCH63_EMPTY_PERCENTILE")
    a = sorted(values)
    j = p * (len(a) - 1)
    k = int(j)
    return a[k] * (1 - (j - k)) + a[min(k + 1, len(a) - 1)] * (j - k)


def paired_scene_bootstrap(scenes: dict, candidate: str) -> dict:
    """Descriptive cluster resampling, *not* a deployment confidence bound."""
    rng = random.Random(BOOTSTRAP_SEED)
    names = list(b62.SCENES)
    out = {"delta_scene_macro_visible_coverage": [],
           "delta_scene_macro_invisible_false_accept_rate": []}
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [names[rng.randrange(len(names))] for _ in names]
        for key, name in (("visible_coverage", "delta_scene_macro_visible_coverage"),
                          ("invisible_false_accept_rate", "delta_scene_macro_invisible_false_accept_rate")):
            paired = [
                scenes[s][candidate][key] - scenes[s]["template_4px"][key]
                for s in sample
                if scenes[s][candidate][key] is not None and
                scenes[s]["template_4px"][key] is not None
            ]
            out[name].append(statistics.mean(paired) if paired else 0.0)
    return {
        k: {"percentile_2_5": percentile(v, 0.025),
            "median": percentile(v, 0.5),
            "percentile_97_5": percentile(v, 0.975)}
        for k, v in out.items()
    }


def audit(pred: dict, sel: dict, dev: dict, queries: dict, oracle: dict) -> dict:
    b62.validate_sources(pred, sel, dev, queries)
    if set(oracle) != set(b62.SCENES):
        raise RuntimeError("BATCH63_DEV_ORACLE_POPULATION")
    recs = {r["name"]: r for r in pred["records"]}
    if set(recs) != set(b62.SCENES):
        raise RuntimeError("BATCH63_DEV_PREDICTION_POPULATION")
    by_scene = {}
    for scene in b62.SCENES:
        scored = score_scene(recs[scene], queries[scene], oracle[scene])
        official = dev["per_scene"][scene][b62.BASE]
        for field in ("visible", "invisible", "accepted_visible", "correct8",
                      "invisible_false_accept", "predicted_visible"):
            if scored["baseline"][field] != official[field]:
                raise RuntimeError("BATCH63_BASELINE_REPLAY_MISMATCH:" + scene + ":" + field)
        old = b62.score_one(recs[scene], queries[scene], oracle[scene])
        for variant in ("baseline", "template_4px"):
            for field in ("visible", "invisible", "correct8", "predicted_visible"):
                if scored[variant][field] != old[variant][field]:
                    raise RuntimeError("BATCH63_BATCH62_REPLAY_MISMATCH")
        for name in VARIANTS:
            if scored[name]["accepted_visible"] > scored["baseline"]["accepted_visible"]:
                raise RuntimeError("BATCH63_NONVETO_COVERAGE")
            if scored[name]["invisible_false_accept"] > scored["baseline"]["invisible_false_accept"]:
                raise RuntimeError("BATCH63_NONVETO_OCCLUSION")
            if name.startswith("template_streak"):
                for field in ("accepted_visible", "invisible_false_accept", "correct8"):
                    if scored[name][field] > scored["template_4px"][field]:
                        raise RuntimeError("BATCH63_NOT_TEMPLATE_SUBSET")
        by_scene[scene] = scored
    summaries = {}
    for name in VARIANTS:
        rows = [by_scene[s][name] for s in b62.SCENES]
        supported = [s for s in b62.SCENES if by_scene[s][name]["invisible"]]
        violations = [s for s in supported if
                      by_scene[s][name]["invisible_false_accept_rate"] > BUDGET]
        summaries[name] = {
            "micro": b62.rates(b62.add(rows)),
            "scene_macro_visible_coverage": b62.average(rows, "visible_coverage"),
            "scene_macro_invisible_false_accept_rate": b62.average(rows, "invisible_false_accept_rate"),
            "scene_macro_jaccard": b62.average(rows, "jaccard_mean_1_2_4_8_16"),
            "occlusion_supported_scenes": len(supported),
            "scene_budget_violation_ids": violations,
            "scene_budget_violation_count": len(violations),
            "worst_supported_scene_false_accept_rate":
                max((by_scene[s][name]["invisible_false_accept_rate"] for s in supported), default=None),
            "persistent_false_accept_episodes_ge_8":
                sum(by_scene[s][name]["false_accept_episodes_ge_8"] for s in b62.SCENES),
            "persistent_false_accept_episodes_ge_16":
                sum(by_scene[s][name]["false_accept_episodes_ge_16"] for s in b62.SCENES),
            "max_false_accept_episode_frames":
                max(by_scene[s][name]["max_false_accept_episode_frames"] for s in b62.SCENES),
            "paired_vetoed_occluded_frames_vs_template":
                sum(by_scene[s]["template_4px"]["invisible_false_accept"] -
                    by_scene[s][name]["invisible_false_accept"] for s in b62.SCENES),
            "paired_lost_correct_visible_frames_vs_template":
                sum(by_scene[s]["template_4px"]["correct8"] -
                    by_scene[s][name]["correct8"] for s in b62.SCENES),
            "paired_lost_incorrect_visible_frames_vs_template":
                sum((by_scene[s]["template_4px"]["accepted_visible"] - by_scene[s]["template_4px"]["correct8"]) -
                    (by_scene[s][name]["accepted_visible"] - by_scene[s][name]["correct8"])
                    for s in b62.SCENES),
        }
        for key in ("scene_macro_visible_coverage", "scene_macro_invisible_false_accept_rate",
                    "scene_macro_jaccard"):
            summaries[name]["delta_" + key + "_vs_template"] = (
                summaries[name][key] - summaries["template_4px"][key]
                if "template_4px" in summaries else None
            )
        if name == "template_4px":
            for previous in summaries.values():
                for key in ("scene_macro_visible_coverage", "scene_macro_invisible_false_accept_rate",
                            "scene_macro_jaccard"):
                    previous["delta_" + key + "_vs_template"] = previous[key] - summaries[name][key]
    bootstrap = {
        name: paired_scene_bootstrap(by_scene, name)
        for name in ("template_streak2", "template_streak3", "affine_streak2", "dual_streak2")
    }
    return {
        "kind": "BATCH63_FROZEN_DEV_CAUSAL_TEMPORAL_CONSENSUS_FALSIFICATION",
        "split": "dev", "exploratory": True, "deployable": False,
        "test_oracle_read": False, "test_predictions_read": False,
        "oracle_read_only_until_dev_35": True, "source_commit": pred["source_commit"],
        "dev_prediction_sha256": dev["prediction_sha256"],
        "dev_results_sha256": sel["dev_results_sha256"],
        "dev_scene_count": len(b62.SCENES), "dev_query_count": 144,
        "fixed_variants": VARIANTS, "budget": BUDGET,
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_samples": BOOTSTRAP_RESAMPLES,
        "bootstrap_caveat": "Descriptive scene-cluster sensitivity on reused DEV, not independent certification.",
        "risk_caveat": ("Agreement can persist while foreground drifts. "
                       "Occlusion safety cannot be inferred from correlated trackers."),
        "temporal_causality": "Current and past frames only, never future frames; no query-frame credit.",
        "frozen_method_selection_changed": False, "running_batch59_changed": False,
        "results": summaries, "paired_scene_cluster_bootstrap": bootstrap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    pred = b62.pinned_json(b62.IN / "dev_predictions.json")
    sel = b62.pinned_json(b62.IN / "dev_selection.json")
    raw = (b62.IN / "dev_results.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != sel["dev_results_sha256"]:
        raise RuntimeError("BATCH63_DEV_RESULTS_PIN_DRIFT")
    dev = json.loads(raw)
    queries = json.loads((b62.DATA / "queries-public.json").read_bytes())
    b62.validate_sources(pred, sel, dev, queries)  # inspect metadata before private DEV oracle
    oracle = b62.load_dev_only_oracle(b62.DATA / "oracle-private.json", b62.SCENES)
    report = audit(pred, sel, dev, queries, oracle)
    target = args.output
    pin = target.with_suffix(target.suffix + ".sha256")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or pin.exists():
        raise RuntimeError("BATCH63_NO_OVERWRITE")
    payload = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    with target.open("xb") as f:
        f.write(payload)
    with pin.open("x") as f:
        f.write(digest + "\n")
    print(json.dumps({"output": str(target), "sha256": digest,
                      "results": {name: {k: summary[k] for k in
                       ("scene_macro_visible_coverage", "scene_macro_invisible_false_accept_rate",
                        "scene_budget_violation_count", "persistent_false_accept_episodes_ge_8",
                        "paired_vetoed_occluded_frames_vs_template",
                        "paired_lost_correct_visible_frames_vs_template")}
                       for name, summary in report["results"].items()}}, indent=2))


if __name__ == "__main__":
    main()
