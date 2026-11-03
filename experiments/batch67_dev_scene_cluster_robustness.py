"""Batch67: paired scene-cluster robustness audit of previously scored DEV events.

This is NOT an algorithm, a runtime visibility gate, a new blind evaluation, or
a certification of deployment safety. Source is exclusively the pinned,
aggregate-only Batch66 DEV report; no frames, TEST predictions, TEST oracle,
or private per-frame tracks are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from experiments import batch66_dev_occlusion_episode_audit as b66

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT.parent / "repo-batch66-dev-clean/outputs/research/batch66_dev_episode_safety.json"
DEFAULT_OUTPUT = ROOT / "outputs/research/batch67_dev_scene_cluster_robustness.json"
SOURCE_SHA256 = "6a631534f8adb2dce11e89764994d38dcfefaa299be5ec31f8c78d97e7467e37"
SEED = 20261103
BOOTSTRAP_REPLICATES = 4096
SCENE_IDS = tuple(f"kinetics_{i:02d}" for i in range(36))
VARIANTS = tuple(b66.VARIANTS)
COUNTS = ("visible_frames", "accepted_visible_frames", "correct_visible_frames_pck8",
          "occluded_frames", "false_accepted_frames", "occlusion_events",
          "affected_events")
EVENT_THRESHOLDS = tuple(b66.RUN_THRESHOLDS)
BUDGET = b66.BUDGET


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_pinned_dev_summary(path: Path) -> dict:
    """Accept exactly the historical Batch66 aggregate artifact by hash."""
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("BATCH67_SOURCE_MISSING_OR_SYMLINK")
    pin = path.with_suffix(path.suffix + ".sha256")
    if not pin.is_file() or pin.is_symlink():
        raise RuntimeError("BATCH67_SOURCE_PIN_MISSING")
    data = path.read_bytes()
    if sha(data) != SOURCE_SHA256 or pin.read_text().strip() != SOURCE_SHA256:
        raise RuntimeError("BATCH67_SOURCE_SHA_DRIFT")
    report = json.loads(data)
    validate_source(report)
    return report


def check_int(row: dict, key: str) -> int:
    v = row.get(key)
    if type(v) is not int or v < 0:
        raise RuntimeError("BATCH67_INVALID_COUNTER:" + key)
    return v


def validate_source(report: dict) -> None:
    if (report.get("kind") != "BATCH66_DEV_ONLY_OCCLUSION_EPISODE_SAFETY_AUDIT"
            or report.get("split") != "dev"
            or report.get("exploratory") is not True
            or report.get("deployable") is not False
            or report.get("test_predictions_read") is not False
            or report.get("test_oracle_read") is not False
            or report.get("frozen_batch59_changed") is not False
            or report.get("dev_scene_count") != 36
            or report.get("dev_query_count") != 144
            or report.get("oracle_dev_only_streamed_through") != "kinetics_35"
            or tuple(report.get("fixed_variants", [])) != VARIANTS
            or abs(report.get("scene_budget", -1) - BUDGET) > 1e-15):
        raise RuntimeError("BATCH67_DEV_ONLY_PROVENANCE_REQUIRED")
    by_scene = report.get("per_scene")
    if not isinstance(by_scene, dict) or set(by_scene) != set(SCENE_IDS):
        raise RuntimeError("BATCH67_SCENE_POPULATION_DRIFT")
    for scene, gates in by_scene.items():
        if set(gates) != set(VARIANTS):
            raise RuntimeError("BATCH67_VARIANT_POPULATION_DRIFT:" + scene)
        denominators = None
        for gate, row in gates.items():
            if not isinstance(row, dict):
                raise RuntimeError("BATCH67_BAD_ROW")
            values = [check_int(row, k) for k in COUNTS]
            v, av, cv, o, fa, events, affected = values
            if av > v or cv > av or fa > o or affected > events:
                raise RuntimeError("BATCH67_IMPOSSIBLE_COUNT:" + scene + ":" + gate)
            worst = check_int(row, "worst_false_accept_run")
            if worst > fa or (affected == 0 and worst != 0):
                raise RuntimeError("BATCH67_INVALID_EVENT_RUN")
            for threshold in EVENT_THRESHOLDS:
                k = f"events_with_run_ge{threshold}"
                n = check_int(row, k)
                if n > affected or (worst < threshold and n != 0):
                    raise RuntimeError("BATCH67_EVENT_THRESHOLD_INCONSISTENT")
            if denominators is None:
                denominators = (v, o, events)
            elif (v, o, events) != denominators:
                raise RuntimeError("BATCH67_DENOMINATOR_CHANGED_BY_GATE")
            expected = fa / o if o else None
            actual = row.get("occluded_false_accept_rate")
            if (expected is None and actual is not None or
                    expected is not None and
                    (type(actual) not in (float, int) or
                     not math.isfinite(actual) or
                     abs(actual - expected) > 1e-12)):
                raise RuntimeError("BATCH67_RATE_REPLAY_MISMATCH")
    if not isinstance(report.get("summary"), dict) or set(report["summary"]) != set(VARIANTS):
        raise RuntimeError("BATCH67_SUMMARY_POPULATION_DRIFT")
    rows = [by_scene[s] for s in SCENE_IDS]
    for gate in VARIANTS:
        actual = aggregate(rows, gate)
        reference = report["summary"][gate]
        for key, ref in (
            ("false_accepted_frames", "false_accepted_frames"),
            ("occluded_frames", "occluded_frames"),
            ("affected_events", "affected_events"),
            ("occlusion_events", "occlusion_events"),
            ("worst_false_accept_run", "worst_false_accept_run"),
            ("scene_budget_violation_count", "scene_budget_violation_count"),
        ):
            if actual[key] != reference[ref]:
                raise RuntimeError("BATCH67_AGGREGATE_REPLAY_MISMATCH:" + gate + ":" + key)
        for key, ref in (
            ("scene_macro_false_accept_rate", "scene_macro_occluded_false_accept_rate"),
            ("micro_visible_coverage", "micro_visible_coverage"),
            ("micro_affected_event_rate", "micro_affected_event_rate"),
        ):
            if not math.isclose(actual[key], reference[ref], abs_tol=1e-12, rel_tol=0):
                raise RuntimeError("BATCH67_AGGREGATE_RATE_DRIFT:" + gate + ":" + key)


def aggregate(rows: list[dict], gate: str) -> dict:
    if not rows:
        raise RuntimeError("BATCH67_NO_SCENES")
    values = [r[gate] for r in rows]
    sums = {key: sum(r[key] for r in values) for key in COUNTS}
    visible = sums["visible_frames"]
    occluded = sums["occluded_frames"]
    events = sums["occlusion_events"]
    if min(visible, occluded, events) <= 0:
        raise RuntimeError("BATCH67_UNSUPPORTED_POPULATION")
    supported = [r for r in values if r["occluded_frames"] > 0]
    affected_supported = [r for r in values if r["occlusion_events"] > 0]
    return {
        **sums,
        "supported_occlusion_scenes": len(supported),
        "unsupported_occlusion_scenes": len(values) - len(supported),
        "scene_macro_false_accept_rate": sum(
            r["false_accepted_frames"] / r["occluded_frames"] for r in supported
        ) / len(supported),
        "scene_macro_affected_event_rate": sum(
            r["affected_events"] / r["occlusion_events"] for r in affected_supported
        ) / len(affected_supported),
        "micro_false_accept_rate": sums["false_accepted_frames"] / occluded,
        "micro_affected_event_rate": sums["affected_events"] / events,
        "micro_visible_coverage": sums["accepted_visible_frames"] / visible,
        "micro_correct_visible_pck8": sums["correct_visible_frames_pck8"] / visible,
        "scene_budget_violation_count": sum(
            r["false_accepted_frames"] / r["occluded_frames"] > BUDGET
            for r in supported),
        "worst_false_accept_run": max(r["worst_false_accept_run"] for r in values),
    }


METRICS = ("scene_macro_false_accept_rate", "micro_false_accept_rate",
           "micro_affected_event_rate", "micro_visible_coverage",
           "micro_correct_visible_pck8")


def paired_effect(rows: list[dict], gate: str) -> dict:
    ref = aggregate(rows, "baseline")
    cur = aggregate(rows, gate)
    return {k: cur[k] - ref[k] for k in METRICS}


def percentile(values: list[float], fraction: float) -> float:
    if not values or not 0 <= fraction <= 1:
        raise RuntimeError("BATCH67_BAD_PERCENTILE")
    sorted_values = sorted(values)
    point = (len(sorted_values) - 1) * fraction
    lo = int(point)
    weight = point - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[
        min(lo + 1, len(sorted_values) - 1)] * weight


def bootstrap_scene_pairs(rows: list[dict], gate: str,
                          repetitions: int = BOOTSTRAP_REPLICATES,
                          seed: int = SEED) -> dict:
    """Paired resampling at SCENE level; never at frame or episode level."""
    if type(repetitions) is not int or repetitions < 100:
        raise RuntimeError("BATCH67_BOOTSTRAP_REPETITIONS")
    if type(seed) is not int:
        raise RuntimeError("BATCH67_SEED")
    rng = random.Random(seed)
    effects = {metric: [] for metric in METRICS}
    safe_and_useful = 0
    for _ in range(repetitions):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        effect = paired_effect(sample, gate)
        for metric in METRICS:
            effects[metric].append(effect[metric])
        # Illustrative policy check only; NOT a new fitted gate or approval rule:
        # require a safety decrease without >5pp absolute coverage loss.
        safe_and_useful += (
            effect["scene_macro_false_accept_rate"] < 0 and
            effect["micro_visible_coverage"] >= -0.05)
    return {
        "resampling_unit": "paired complete scene (36 draws with replacement)",
        "draws": repetitions,
        "seed": seed,
        "descriptive_percentiles": {
            metric: {
                "p025": percentile(values, .025),
                "p500": percentile(values, .5),
                "p975": percentile(values, .975),
                "fraction_delta_less_than_zero": sum(x < 0 for x in values) / len(values),
            }
            for metric, values in effects.items()
        },
        "fraction_with_safety_improvement_and_at_most_5pp_coverage_loss":
            safe_and_useful / repetitions,
        "note": (
            "These are descriptive empirical scene-resampling intervals, not a "
            "distribution-free assurance for unseen scenes. The 5pp check is "
            "illustrative, not a pre-registered product acceptance criterion."
        ),
    }


def sensitivity(rows: list[dict], gate: str) -> dict:
    full = paired_effect(rows, gate)
    influences = []
    for idx, name in enumerate(SCENE_IDS):
        partial = [r for j, r in enumerate(rows) if j != idx]
        effect = paired_effect(partial, gate)
        influences.append({
            "scene": name,
            "risk_delta_without_scene": effect["scene_macro_false_accept_rate"],
            "risk_delta_shift": effect["scene_macro_false_accept_rate"] -
                                full["scene_macro_false_accept_rate"],
            "coverage_delta_shift": effect["micro_visible_coverage"] -
                                    full["micro_visible_coverage"],
            "max_bad_run_without_scene": aggregate(partial, gate)["worst_false_accept_run"],
        })
    blocks = []
    for i in range(6):
        partial = [r for j, r in enumerate(rows) if j // 6 != i]
        effect = paired_effect(partial, gate)
        blocks.append({
            "block": i,
            "removed_scene_ids": list(SCENE_IDS[6*i:6*i+6]),
            "risk_delta": effect["scene_macro_false_accept_rate"],
            "coverage_delta": effect["micro_visible_coverage"],
            "affected_event_delta": effect["micro_affected_event_rate"],
            "max_bad_run": aggregate(partial, gate)["worst_false_accept_run"],
        })
    return {
        "highest_risk_delta_influence": sorted(
            influences, key=lambda x: (-abs(x["risk_delta_shift"]), x["scene"]))[:5],
        "leave_one_fixed_six_scene_block_out": blocks,
        "risk_delta_sign_stable_for_all_six_blocks": all(
            x["risk_delta"] < 0 for x in blocks),
    }


def analyze(source: dict, repetitions: int = BOOTSTRAP_REPLICATES) -> dict:
    validate_source(source)
    rows = [source["per_scene"][s] for s in SCENE_IDS]
    full = {gate: aggregate(rows, gate) for gate in VARIANTS}
    effects = {gate: paired_effect(rows, gate) for gate in VARIANTS if gate != "baseline"}
    concentrated = sorted(
        ((s, source["per_scene"][s]["template_4px"]["false_accepted_frames"])
         for s in SCENE_IDS), key=lambda x: (-x[1], x[0]))
    n_false = full["template_4px"]["false_accepted_frames"]
    return {
        "kind": "BATCH67_DEV_ONLY_SCENE_CLUSTER_UNCERTAINTY_AUDIT",
        "source_report_sha256": SOURCE_SHA256,
        "source_commit": source["source_commit"],
        "split": "dev",
        "exploratory": True,
        "deployable": False,
        "test_oracle_read": False,
        "test_predictions_read": False,
        "private_oracle_read_by_this_script": False,
        "frozen_batch59_changed": False,
        "method_selection_changed": False,
        "dev_scenes": 36,
        "dev_queries": 144,
        "fixed_variants": VARIANTS,
        "fixed_scene_blocks": 6,
        "metrics": full,
        "paired_effects_vs_baseline": effects,
        "template_scene_concentration": {
            "highest_false_accept_scene": concentrated[0][0],
            "highest_false_accept_frames": concentrated[0][1],
            "share_of_template_errors": concentrated[0][1] / n_false,
            "top_two_share": sum(n for _, n in concentrated[:2]) / n_false,
        },
        "template_bootstrap": bootstrap_scene_pairs(
            rows, "template_4px", repetitions=repetitions),
        "template_sensitivity": sensitivity(rows, "template_4px"),
        "limitations": [
            "Previously used DEV only; these are not new independent results.",
            "36 natural scenes, four correlated queries per scene; do not bootstrap frames.",
            "Bootstrap does not model arbitrary dataset shift or guarantee future safety.",
            "Observed maximum wrong-accept run is not an upper bound on unseen videos.",
            "Visibility and occlusion annotations are offline scoring only, not inference inputs.",
            "Image-plane affine displacement is not 3D camera motion.",
        ],
    }


def save_pinned(path: Path, result: dict) -> str:
    payload = (json.dumps(result, ensure_ascii=False, indent=2,
                          sort_keys=True, allow_nan=False) + "\n").encode()
    pin = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or pin.exists() or path.is_symlink() or pin.is_symlink():
        raise RuntimeError("BATCH67_OUTPUT_NO_OVERWRITE")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
    with pin.open("x", encoding="utf-8") as stream:
        stream.write(sha(payload) + "\n")
    return sha(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    result = analyze(load_pinned_dev_summary(args.source), repetitions=args.bootstrap)
    digest = save_pinned(args.output, result)
    summary = {
        "output": str(args.output),
        "sha256": digest,
        "template_effect": result["paired_effects_vs_baseline"]["template_4px"],
        "template_concentration": result["template_scene_concentration"],
        "template_ci": result["template_bootstrap"]["descriptive_percentiles"][
            "scene_macro_false_accept_rate"],
        "block_sign_stability": result["template_sensitivity"][
            "risk_delta_sign_stable_for_all_six_blocks"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
