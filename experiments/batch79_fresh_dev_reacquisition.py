"""Batch79: independent Batch59 natural DEV occlusion/reacquisition audit.
No TEST oracle, predictions, or scores are read. No frozen method/runner edits.
DEV labels are used for retrospective scoring only, never for inference.
PCK8 agreement does not establish physical identity continuity.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from experiments.batch62_dev_consensus_veto import load_dev_only_oracle
from experiments.batch75_dev_reacquisition_audit import (
    post_occlusion_events, aggregate, summarize)

SCENES = tuple(f"kinetics_{i:02d}" for i in range(36))
METHODS = ("klt_fb", "appearance_fb_055", "appearance_fb_070",
           "appearance_fb_085", "gradient_fb_050", "gradient_fb_065",
           "template_recover4", "affine_stabilized_klt")
PROTOCOL = "batch79-fresh-dev-reacquisition-v1"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def pinned(path: Path) -> dict:
    pin = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not pin.is_file():
        raise RuntimeError("BATCH79_MISSING_PIN")
    raw = path.read_bytes()
    if digest(raw) != pin.read_text().strip():
        raise RuntimeError("BATCH79_PIN_DRIFT")
    return json.loads(raw)


def xy(value):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2 or any(
        type(v) not in (float, int) or not math.isfinite(v) for v in value
    ):
        raise RuntimeError("BATCH79_BAD_POINT")
    return value


def load(inp: Path, data: Path) -> tuple:
    for path in (inp, data):
        if "test_" in str(path.resolve()).lower() or not path.is_dir():
            raise RuntimeError("BATCH79_TEST_OR_MISSING_PATH")
    pred = pinned(inp / "dev_predictions.json")
    sel = pinned(inp / "dev_selection.json")
    raw = (inp / "dev_results.json").read_bytes()
    if digest(raw) != sel.get("dev_results_sha256"):
        raise RuntimeError("BATCH79_DEV_SCORE_PIN_DRIFT")
    score = json.loads(raw)
    public_raw = (data / "queries-public.json").read_bytes()
    public = json.loads(public_raw)
    manifest_raw = (data / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    if not (pred.get("split") == sel.get("split") == score.get("split") == "dev"
            and pred.get("oracle_unread_during_prediction") is True):
        raise RuntimeError("BATCH79_DEV_FIREWALL")
    if any(x.get("source_commit") != pred.get("source_commit")
           for x in (sel, score, manifest)):
        raise RuntimeError("BATCH79_SOURCE_DRIFT")
    if (pred.get("config_sha256") != sel.get("config_sha256")
            or pred.get("config_sha256") != score.get("config_sha256")
            or pred.get("manifest_sha256") != digest(manifest_raw)
            or pred.get("public_query_sha256") != digest(public_raw)
            or score.get("prediction_sha256") !=
            digest((inp / "dev_predictions.json").read_bytes())):
        raise RuntimeError("BATCH79_SOURCE_HASH_DRIFT")
    if (manifest.get("dev_videos") != 36 or manifest.get("test_videos") != 6
            or not set(SCENES).issubset(public)
            or len(pred.get("records", [])) != 36
            or set(score.get("per_scene", {})) != set(SCENES)
            or not set(METHODS).issubset(pred.get("methods", []))
            or sel.get("selected_method") not in METHODS):
        raise RuntimeError("BATCH79_SPLIT_OR_METHOD_DRIFT")
    # Stream exactly kinetics_00 through kinetics_35; never parse the 6 TEST entries.
    oracle = load_dev_only_oracle(data / "oracle-private.json", list(SCENES))
    return pred, sel, score, public, oracle


def scene_sum(rows: dict, method: str, scene_names: list[str]) -> dict:
    values = [rows[name][method] for name in scene_names]
    eligible = sum(r["reacquisition_evaluable_events"] for r in values)
    recovered = sum(r["reacquired_events"] for r in values)
    occluded = sum(r["occluded_frames"] for r in values)
    false = sum(r["false_accepted_occluded_frames"] for r in values)
    supporting = [r for r in values if r["occluded_frames"] > 0]
    return {"events": eligible, "recovered": recovered,
            "recovery_rate": recovered / eligible if eligible else None,
            "occluded_frames": occluded, "false_accepted": false,
            "false_accept_rate": false / occluded if occluded else None,
            "support_scenes": len(supporting),
            "violating_scenes": sum(r["false_accepted_occluded_frames"] > 0
                                    for r in supporting),
            "worst_scene_rate": max(
                (r["false_accepted_occluded_frames"] / r["occluded_frames"]
                 for r in supporting), default=None)}


def scene_interval(rows: dict, chosen: str, control: str, n: int = 2000,
                   seed: int = 20261104) -> dict:
    if n <= 0 or type(n) is not int:
        raise RuntimeError("BATCH79_BAD_BOOTSTRAP_COUNT")
    rand = random.Random(seed)
    scenes = sorted(rows)
    deltas = {"recovery_rate": [], "false_accept_rate": []}
    for _ in range(n):
        sample = rand.choices(scenes, k=len(scenes))
        a, b = scene_sum(rows, chosen, sample), scene_sum(rows, control, sample)
        for key in deltas:
            if a[key] is not None and b[key] is not None:
                deltas[key].append(a[key] - b[key])
    result = {}
    for key, values in deltas.items():
        values.sort()
        result[key + "_paired_95_pct"] = (
            [values[int((len(values) - 1) * 0.025)],
             values[int((len(values) - 1) * 0.975)]] if values else None)
    return {"control": control, "chosen": chosen, "seed": seed,
            "resamples": n, "unit": "scene",
            "exploratory_not_test_confidence": True, **result}


def evaluate(pred: dict, selection: dict, score: dict,
             public: dict, oracle: dict, bootstraps: int = 2000) -> dict:
    records = {r["name"]: r for r in pred["records"]}
    if len(records) != 36 or set(records) != set(SCENES) or set(oracle) != set(SCENES):
        raise RuntimeError("BATCH79_SCENE_DRIFT")
    per_scene, all_events = {}, {m: [] for m in METHODS}
    for scene in SCENES:
        rec, queries, truths = records[scene], public[scene], oracle[scene]["tracks"]
        qmap = {q["id"]: q for q in queries}
        if len(qmap) != 4 or len(queries) != 4 or (
                set(qmap) != set(rec["tracks"]) or set(qmap) != set(truths)):
            raise RuntimeError("BATCH79_QUERY_DRIFT")
        grouped = {m: [] for m in METHODS}
        for query_id, query in qmap.items():
            n, start = rec["nframes"], query["query_time"]
            gt, predictions = truths[query_id], rec["tracks"][query_id]
            if (type(n) is not int or type(start) is not int or
                    not 0 <= start < n - 1 or
                    len(gt["occluded"]) != n or len(gt["xy"]) != n or
                    any(type(z) is not bool for z in gt["occluded"])):
                raise RuntimeError("BATCH79_ORACLE_LENGTH")
            for method in METHODS:
                trace = predictions[method]
                if (len(trace) != n or any(x is not None for x in trace[:start])
                        or xy(trace[start]) is None):
                    raise RuntimeError("BATCH79_TRACE_LENGTH_OR_QUERY")
                for pos in trace:
                    xy(pos)
                grouped[method].extend(post_occlusion_events(
                    gt["occluded"], gt["xy"], trace,
                    [x is not None for x in trace], start, 8.0))
        per_scene[scene] = {}
        for method in METHODS:
            counts = aggregate(grouped[method])
            previous = score["per_scene"][scene][method]
            if (counts["occluded_frames"] != previous["invisible"] or
                    counts["false_accepted_occluded_frames"] !=
                    previous["invisible_false_accept"]):
                raise RuntimeError("BATCH79_REPLAY_MISMATCH")
            per_scene[scene][method] = summarize(counts)
            all_events[method].extend(grouped[method])
    chosen = selection["selected_method"]
    return {
        "protocol": PROTOCOL, "split": "dev", "scenes": 36, "queries": 144,
        "source_commit": pred["source_commit"],
        "frozen_selection": chosen, "methods": list(METHODS),
        "new_blind_test": False, "test_oracle_parsed": False,
        "test_predictions_read": False, "test_scores_read": False,
        "algorithm_changed": False, "deployable": False,
        "meaning": "any PCK8-accepted position after an occlusion; not identity proof",
        "censor": "occlusion ending at video end excluded from recovery denominator",
        "limitations": [
            "DEV-only exploratory analysis, not a new blind TEST",
            "scene bootstrap assumes exchangeable video scenes",
            "no physical identity verification or calibrated camera poses",
            "nested frames are not independent study units",
        ],
        "overall": {m: summarize(aggregate(all_events[m])) for m in METHODS},
        "scene_summary": {m: scene_sum(per_scene, m, list(SCENES)) for m in METHODS},
        "paired_intervals": {
            m: scene_interval(per_scene, chosen, m, bootstraps)
            for m in ("appearance_fb_055", "appearance_fb_085") if m != chosen
        },
        "per_scene": per_scene,
    }


def write_once(path: Path, result: dict) -> str:
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    pin = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or pin.exists():
        raise RuntimeError("BATCH79_OUTPUT_ALREADY_EXISTS")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(raw)
    with pin.open("x", encoding="ascii") as f:
        f.write(digest(raw) + "\n")
    return digest(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prediction, selection, score, public, oracle = load(args.input_dir, args.data_dir)
    result = evaluate(prediction, selection, score, public, oracle)
    sha = write_once(args.output, result)
    print(json.dumps({"event": "BATCH79_NEW_NATURAL_DEV_AUDIT_PASS",
                      "sha256": sha, "selected": result["frozen_selection"],
                      "scene_summary": result["scene_summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
