"""Batch62: falsify *observable* per-frame tracker-consensus vetoes on pinned DEV.

Exploratory only: the 36 Batch56 DEV scenes have already been studied. This
analysis neither updates a frozen tracker nor reads/scans any TEST predictions,
labels, or video. It accepts predictions computed before labels were opened.
The private DEV oracle is streamed scene-by-scene and stops at scene 35.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT.parent / "repo-batch56b-tested"
DATA = ROOT.parent / "data/tapvid_kinetics_batch56/batch56_frozen"
IN = FROZEN / "outputs/kinetics_batch56"
OUTPUT = ROOT / "outputs/research/batch62_dev_consensus_veto.json"
SCENES = [f"kinetics_{i:02d}" for i in range(36)]
BASE = "appearance_fb_055"
# Fixed a priori for this exploratory comparison, never fit to GT.
GATES = {
    "baseline": (),
    "template_4px": (("template_recover4", 4.0),),
    "affine_3px": (("affine_stabilized_klt", 3.0),),
    "both_4px_3px": (("template_recover4", 4.0),
                      ("affine_stabilized_klt", 3.0)),
}
RADII = (1, 2, 4, 8, 16)
BUDGET = 0.10


def sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def pinned_json(path: Path) -> dict:
    pin = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not pin.is_file():
        raise RuntimeError("BATCH62_PIN_MISSING:" + path.name)
    content = path.read_bytes()
    if sha(content) != pin.read_text().strip():
        raise RuntimeError("BATCH62_PIN_DRIFT:" + path.name)
    return json.loads(content)


def load_dev_only_oracle(path: Path, scene_ids: list[str]) -> dict:
    """Consume exactly 36 DEV objects from pretty-printed immutable oracle.

    The oracle file also holds TEST labels: intentionally do NOT json.load it.
    Reject reordering, missing DEV entries and malformed top-level framing.
    Stop at end of kinetics_35, before reading the next (TEST) key.
    """
    if scene_ids != SCENES:
        raise RuntimeError("BATCH62_DEV_POPULATION_DRIFT")
    result = {}
    with path.open("r", encoding="utf-8") as f:
        if f.readline().strip() != "{":
            raise RuntimeError("BATCH62_INVALID_ORACLE_START")
        for expected in scene_ids:
            line = f.readline()
            if line != f'  "{expected}": {{\n':
                raise RuntimeError("BATCH62_ORACLE_ORDER_OR_FORMAT_DRIFT")
            block = ["{"]
            for _ in range(100000):
                line = f.readline()
                if not line:
                    raise RuntimeError("BATCH62_TRUNCATED_DEV_ORACLE")
                if line == "  },\n":
                    break
                block.append(line)
            else:
                raise RuntimeError("BATCH62_ORACLE_RECORD_TOO_LARGE")
            # The closing brace of this object appears at two-space indent.
            entry = json.loads("\n".join(['{'] + block[1:]) + "  }")
            if set(entry) != {"tracks"} or not isinstance(entry["tracks"], dict):
                raise RuntimeError("BATCH62_INVALID_DEV_ORACLE_RECORD")
            result[expected] = entry
    return result


def validate_sources(pred: dict, sel: dict, dev: dict, queries: dict) -> None:
    if (pred.get("split") != "dev" or sel.get("split") != "dev" or
        dev.get("split") != "dev" or
        pred.get("oracle_unread_during_prediction") is not True):
        raise RuntimeError("BATCH62_DEV_ONLY_FIREWALL")
    if (pred.get("source_commit") != dev.get("source_commit") or
        pred["source_commit"] != sel.get("source_commit") or
        sel.get("selected_method") != BASE):
        raise RuntimeError("BATCH62_SOURCE_OR_SELECTION_DRIFT")
    for key in ("config_sha256", "manifest_sha256"):
        if pred.get(key) != dev.get(key):
            raise RuntimeError("BATCH62_SOURCE_PIN_DRIFT:" + key)
    # The PUBLIC query manifest also contains six unused TEST query entries.
    # Require only the DEV subset and never inspect TEST oracle records.
    if not set(SCENES).issubset(queries):
        raise RuntimeError("BATCH62_PUBLIC_QUERY_SPLIT_DRIFT")
    if (len(pred.get("records", [])) != 36 or
        {r.get("name") for r in pred["records"]} != set(SCENES) or
        set(dev.get("per_scene", {})) != set(SCENES)):
        raise RuntimeError("BATCH62_PREDICTION_POPULATION_DRIFT")
    if sel.get("config_sha256") != dev["config_sha256"]:
        raise RuntimeError("BATCH62_SELECTION_CONFIG_DRIFT")
    # Reject accidental private metadata in the public-only prediction object.
    if set(GATES["both_4px_3px"][i][0] for i in range(2)) - set(pred["methods"]):
        raise RuntimeError("BATCH62_SUPPORT_METHOD_MISSING")
    if BASE not in pred["methods"]:
        raise RuntimeError("BATCH62_SELECTED_METHOD_MISSING")


def xy(point: object) -> tuple[float, float] | None:
    if point is None:
        return None
    if not isinstance(point, list) or len(point) != 2:
        raise RuntimeError("BATCH62_INVALID_XY")
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) for v in point):
        raise RuntimeError("BATCH62_NONFINITE_XY")
    return float(point[0]), float(point[1])


def pass_veto(track: dict, frame: int, variant: str) -> bool:
    if variant not in GATES:
        raise RuntimeError("BATCH62_UNKNOWN_VETO")
    target = xy(track[BASE][frame])
    if target is None:
        return False
    for key, radius in GATES[variant]:
        supporter = xy(track[key][frame])
        if supporter is None or math.dist(target, supporter) > radius:
            return False
    return True


def counts_empty() -> dict:
    return {"visible": 0, "invisible": 0, "accepted_visible": 0,
            "correct8": 0, "invisible_false_accept": 0,
            "predicted_visible": 0,
            **{f"tp{r}": 0 for r in RADII}}


def score_one(pred_scene: dict, queries: list[dict], oracle_scene: dict) -> dict:
    out = {g: counts_empty() for g in GATES}
    qmap = {q["id"]: q for q in queries}
    if len(qmap) != 4 or set(qmap) != set(oracle_scene["tracks"]) \
            or set(qmap) != set(pred_scene["tracks"]):
        raise RuntimeError("BATCH62_QUERY_ID_MISMATCH")
    for qid, q in qmap.items():
        track = pred_scene["tracks"][qid]
        gt = oracle_scene["tracks"][qid]
        t0 = q["query_time"]
        n = pred_scene["nframes"]
        if type(t0) is not int or not 0 <= t0 < n - 1:
            raise RuntimeError("BATCH62_QUERY_TIME_INVALID")
        for key in (BASE, "template_recover4", "affine_stabilized_klt"):
            if len(track[key]) != n or any(x is not None for x in track[key][:t0]):
                raise RuntimeError("BATCH62_TRACK_LENGTH_OR_PREQUERY")
        if len(gt["occluded"]) != n or len(gt["xy"]) != n:
            raise RuntimeError("BATCH62_ORACLE_TRACE_LENGTH")
        for i in range(t0 + 1, n):
            occ = gt["occluded"][i]
            if type(occ) is not bool:
                raise RuntimeError("BATCH62_INVALID_VISIBILITY")
            for variant in GATES:
                c = out[variant]
                accepted = pass_veto(track, i, variant)
                visible = not occ
                c["visible" if visible else "invisible"] += 1
                if accepted:
                    c["predicted_visible"] += 1
                    if visible:
                        c["accepted_visible"] += 1
                        err = math.dist(xy(track[BASE][i]), xy(gt["xy"][i]))
                        c["correct8"] += int(err <= 8)
                        for rad in RADII:
                            c[f"tp{rad}"] += int(err <= rad)
                    else:
                        c["invisible_false_accept"] += 1
    return out


def rates(c: dict) -> dict:
    v, inv = c["visible"], c["invisible"]
    covered, accepted = c["accepted_visible"], c["predicted_visible"]
    return {
        **c,
        "visible_coverage": covered / v if v else None,
        "invisible_false_accept_rate": c["invisible_false_accept"] / inv if inv else None,
        "accepted_precision8": c["correct8"] / covered if covered else None,
        "pck8_visible_unconditional": c["correct8"] / v if v else None,
        "jaccard_mean_1_2_4_8_16": (
            statistics.mean(c[f"tp{r}"] /
                            (v + accepted - c[f"tp{r}"])
                            for r in RADII) if v else None),
    }


def add(rows: list[dict]) -> dict:
    total = counts_empty()
    for r in rows:
        for k in total:
            total[k] += r[k]
    return total


def average(rows: list[dict], key: str) -> float | None:
    values = [r[key] for r in rows if r[key] is not None]
    return statistics.mean(values) if values else None



def component_cpu_proxy(pred: dict, queries: dict) -> dict:
    """Reconstruct additive CPU-cost *proxies* from frozen per-query timings.

    These are NOT new online pipeline measurements. Shared KLT work may be
    double-counted, decoder and Python orchestration are excluded, and affine
    registration is only amortized per scene-target frame for comparison.
    """
    time_by_gate = {g: [] for g in GATES}
    affine_registration_by_gate = {g: [] for g in GATES}
    for record in pred["records"]:
        name = record["name"]
        n = record["nframes"]
        qmap = {q["id"]: q for q in queries[name]}
        source = pred["scene_aux"][name]
        registration = source["affine_registration_ms_per_video"]
        if isinstance(registration, bool) or not isinstance(registration, (float, int)) \
                or not math.isfinite(registration) or registration < 0:
            raise RuntimeError("BATCH62_BAD_AFFINE_REGISTRATION")
        # Registration is computed once per video rather than once per query.
        amortized = registration / (n * len(qmap))
        for qid, query in qmap.items():
            denom = n - int(query["query_time"]) - 1
            if denom <= 0:
                raise RuntimeError("BATCH62_INVALID_TIMING_DENOMINATOR")
            times = record["cpu_query_ms"][qid]
            for variant, supporters in GATES.items():
                required = (BASE,) + tuple(k for k, _ in supporters)
                vals = [times[k] for k in required]
                if any(isinstance(x, bool) or not isinstance(x, (float, int))
                       or not math.isfinite(x) or x < 0 for x in vals):
                    raise RuntimeError("BATCH62_BAD_CPU_TIMING")
                time_by_gate[variant].append(sum(vals) / denom)
                affine_registration_by_gate[variant].append(
                    amortized if "affine_stabilized_klt" in required else 0.0
                )

    def percentile(values: list[float], q: float) -> float:
        if not values:
            raise RuntimeError("BATCH62_MISSING_TIMINGS")
        vals = sorted(values)
        pos = q * (len(vals) - 1)
        lo = int(pos)
        return vals[lo] * (1 - (pos - lo)) + vals[min(len(vals) - 1, lo + 1)] * (pos - lo)

    out = {}
    for name in GATES:
        per = [x + y for x, y in zip(time_by_gate[name],
                                      affine_registration_by_gate[name])]
        out[name] = {
            "samples": len(per),
            "sum_component_ms_per_target_frame_p50": percentile(per, 0.5),
            "sum_component_ms_per_target_frame_p95": percentile(per, 0.95),
            "affine_registration_amortized_ms_per_target_frame_p50": percentile(
                affine_registration_by_gate[name], 0.5),
        }
    return out

def evaluate(pred: dict, sel: dict, dev: dict, queries: dict, oracle: dict) -> dict:
    validate_sources(pred, sel, dev, queries)
    per_scene = {}
    for rec in pred["records"]:
        name = rec["name"]
        per_scene[name] = {
            variant: rates(c)
            for variant, c in score_one(rec, queries[name], oracle[name]).items()
        }
        actual = per_scene[name]["baseline"]
        prior = dev["per_scene"][name][BASE]
        for metric in ("visible", "invisible", "accepted_visible", "correct8",
                       "invisible_false_accept", "predicted_visible"):
            if actual[metric] != prior[metric]:
                raise RuntimeError("BATCH62_BASELINE_REPLAY_MISMATCH:" + name + ":" + metric)
    report = {}
    for variant in GATES:
        rows = [per_scene[s][variant] for s in SCENES]
        agg = rates(add(rows))
        supported = [s for s in SCENES if per_scene[s][variant]["invisible"]]
        violations = [s for s in supported if
                      per_scene[s][variant]["invisible_false_accept_rate"] > BUDGET]
        groups = []
        for i in range(6):
            block = SCENES[i * 6:(i + 1) * 6]
            subset = [per_scene[s][variant] for s in block]
            groups.append({"block": i, "scene_ids": block,
                           "scene_macro_visible_coverage": average(subset, "visible_coverage"),
                           "scene_macro_occluded_false_accept": average(
                               subset, "invisible_false_accept_rate"),
                           "violating_occlusion_scenes": sum(
                               per_scene[s][variant]["invisible"] > 0 and
                               per_scene[s][variant]["invisible_false_accept_rate"] > BUDGET
                               for s in block)})
        report[variant] = {
            "micro": agg,
            "scene_macro_visible_coverage": average(rows, "visible_coverage"),
            "scene_macro_invisible_false_accept_rate": average(rows, "invisible_false_accept_rate"),
            "scene_macro_jaccard": average(rows, "jaccard_mean_1_2_4_8_16"),
            "occlusion_supported_scenes": len(supported),
            "occlusion_unsupported_scenes": 36 - len(supported),
            "scene_budget_violation_ids": violations,
            "scene_budget_violation_count": len(violations),
            "worst_supported_scene_false_accept_rate": max(
                (per_scene[s][variant]["invisible_false_accept_rate"] for s in supported),
                default=None),
            "fixed_index_blocks": groups,
        }
    baseline = report["baseline"]
    for name, row in report.items():
        row["delta_scene_macro_coverage"] = (row["scene_macro_visible_coverage"] -
                                               baseline["scene_macro_visible_coverage"])
        row["delta_scene_macro_occluded_fa"] = (
            row["scene_macro_invisible_false_accept_rate"] -
            baseline["scene_macro_invisible_false_accept_rate"])
    return {
        "kind": "BATCH62_DEV_ONLY_PREDICTION_CONSENSUS_VETO_FALSIFICATION",
        "split": "dev", "exploratory": True, "deployable": False,
        "source_commit": pred["source_commit"],
        "baseline": BASE, "budget": BUDGET, "fixed_gates": GATES,
        "dev_scene_count": 36, "dev_query_count": 144,
        "prediction_sha256": dev["prediction_sha256"],
        "dev_results_sha256": sel["dev_results_sha256"],
        "oracle_read_only_until_dev_35": True,
        "test_predictions_read": False, "test_oracle_read": False,
        "method_selection_changed": False, "frozen_run_changed": False,
        "cpu_limitations": (
            "Consensus requires running an additional template and/or affine tracker; "
            "this offline veto does not measure actual new end-to-end CPU overhead."
        ),
        "causality_limitations": (
            "Support trajectories are correlated and can agree on foreground drift; "
            "consensus is not proof of visible target identity, static background or 3D camera motion."
        ),
        "offline_component_cpu_proxies": component_cpu_proxy(pred, queries),
        "results": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    pred = pinned_json(IN / "dev_predictions.json")
    sel = pinned_json(IN / "dev_selection.json")
    dev_path = IN / "dev_results.json"
    dev_bytes = dev_path.read_bytes()
    if sha(dev_bytes) != sel["dev_results_sha256"]:
        raise RuntimeError("BATCH62_DEV_RESULTS_PIN_DRIFT")
    dev = json.loads(dev_bytes)
    queries = json.loads((DATA / "queries-public.json").read_bytes())
    validate_sources(pred, sel, dev, queries)
    # Only after all public input/prediction integrity checks; stops BEFORE TEST keys.
    oracle = load_dev_only_oracle(DATA / "oracle-private.json", SCENES)
    outcome = evaluate(pred, sel, dev, queries, oracle)
    payload = (json.dumps(outcome, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    target = args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.with_suffix(target.suffix + ".sha256").exists():
        raise RuntimeError("BATCH62_NO_OVERWRITE")
    with target.open("xb") as f:
        f.write(payload)
    with target.with_suffix(target.suffix + ".sha256").open("x") as f:
        f.write(sha(payload) + "\n")
    print(json.dumps({"output": str(target), "sha256": sha(payload),
                      "summary": {k: {x: v[x] for x in (
                          "scene_macro_visible_coverage",
                          "scene_macro_invisible_false_accept_rate",
                          "scene_budget_violation_count", "scene_macro_jaccard")}
                          for k, v in outcome["results"].items()}},
                     indent=2))


if __name__ == "__main__":
    main()
