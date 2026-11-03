"""Batch75: historical DEV-only evaluation of *post-occlusion reacquisition*.

Read-only retrospective analysis. All target locations and visibility labels are
used ONLY by the offline DEV scorer. Never use oracle-derived event features
during tracker inference. There are no parameters to tune on held-out TEST.
Batch59's frozen runner/plan/inputs are not modified or imported.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from experiments import batch62_dev_consensus_veto as base
from experiments import batch66_dev_occlusion_episode_audit as b66

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/research/batch75_dev_reacquisition_safety.json"
VARIANTS = b66.VARIANTS
RADIUS = 8.0


def post_occlusion_events(
    occluded: list[bool], ground_truth: list, predictions: list,
    accepted: list[bool], query_time: int, radius: float = RADIUS,
) -> list[dict]:
    """Disjoint post-occlusion visible windows, up to next occlusion/video end.

    Events ending at the last frame are right-censored: *not* recorded as
    failures to reacquire. Wrong spatial predictions must NOT be interpreted
    as proven physical identity switches.
    """
    n = len(occluded)
    if (type(query_time) is not int or query_time < 0 or query_time >= n or
            not (len(ground_truth) == len(predictions) == len(accepted) == n)):
        raise RuntimeError("BATCH75_INVALID_TRACE_LENGTH")
    if (not isinstance(radius, (float, int)) or isinstance(radius, bool) or
            not math.isfinite(radius) or radius <= 0):
        raise RuntimeError("BATCH75_INVALID_RADIUS")
    if any(type(x) is not bool for x in occluded + accepted):
        raise RuntimeError("BATCH75_BOOLEAN_TRACE_REQUIRED")
    if any(accepted[t] for t in range(query_time + 1, n)
           if predictions[t] is None):
        raise RuntimeError("BATCH75_ACCEPTED_NULL_PREDICTION")
    events = []
    t = query_time + 1
    while t < n:
        if not occluded[t]:
            t += 1
            continue
        start = t
        false_accepts = 0
        while t < n and occluded[t]:
            false_accepts += int(accepted[t])
            t += 1
        end_occ = t
        start_visible = t
        first_accepted = None
        first_correct = None
        accepted_visible = correct_visible = wrong_visible = 0
        longest_wrong = wrong_streak = 0
        while t < n and not occluded[t]:
            if accepted[t]:
                pred = base.xy(predictions[t])
                truth = base.xy(ground_truth[t])
                if pred is None or truth is None:
                    raise RuntimeError("BATCH75_VISIBLE_ACCEPT_WITHOUT_COORDINATES")
                correct = math.dist(pred, truth) <= radius
                accepted_visible += 1
                correct_visible += int(correct)
                wrong_visible += int(not correct)
                wrong_streak = 0 if correct else wrong_streak + 1
                longest_wrong = max(longest_wrong, wrong_streak)
                if first_accepted is None:
                    first_accepted = "correct" if correct else "wrong"
                if correct and first_correct is None:
                    first_correct = t - start_visible
            else:
                wrong_streak = 0
            t += 1
        visible_frames = t - start_visible
        events.append({
            "start": start,
            "end_occlusion_exclusive": end_occ,
            "occluded_frames": end_occ - start,
            "occluded_false_accepted": false_accepts,
            "visible_window_frames": visible_frames,
            "right_censored": visible_frames == 0,
            "first_accepted_status": first_accepted,
            "first_correct_lag_frames": first_correct,
            "accepted_post_visible_frames": accepted_visible,
            "correct_post_visible_frames": correct_visible,
            "wrong_post_visible_frames": wrong_visible,
            "longest_wrong_visible_run": longest_wrong,
        })
    return events


def aggregate(events: list[dict]) -> dict:
    evaluated = [e for e in events if not e["right_censored"]]
    recovered = [e for e in evaluated if e["first_correct_lag_frames"] is not None]
    lags = [e["first_correct_lag_frames"] for e in recovered]
    total_post_visible = sum(e["visible_window_frames"] for e in evaluated)
    return {
        "occlusion_events": len(events),
        "right_censored_events": len(events) - len(evaluated),
        "reacquisition_evaluable_events": len(evaluated),
        "reacquired_events": len(recovered),
        "first_accept_wrong_events": sum(
            e["first_accepted_status"] == "wrong" for e in evaluated),
        "never_accepted_events": sum(
            e["first_accepted_status"] is None for e in evaluated),
        "occluded_frames": sum(e["occluded_frames"] for e in events),
        "false_accepted_occluded_frames": sum(e["occluded_false_accepted"] for e in events),
        "post_visible_frames": total_post_visible,
        "accepted_post_visible_frames": sum(
            e["accepted_post_visible_frames"] for e in evaluated),
        "correct_post_visible_frames": sum(
            e["correct_post_visible_frames"] for e in evaluated),
        "wrong_post_visible_frames": sum(
            e["wrong_post_visible_frames"] for e in evaluated),
        "worst_wrong_visible_run": max(
            (e["longest_wrong_visible_run"] for e in evaluated), default=0),
        "recovered_lags_frames": lags,
    }


def summarize(counts: dict) -> dict:
    # Rates with no supported events must remain null (not magically safe).
    e = counts["reacquisition_evaluable_events"]
    visible = counts["post_visible_frames"]
    accepted = counts["accepted_post_visible_frames"]
    lags = counts["recovered_lags_frames"]
    return {
        **{k: v for k, v in counts.items() if k != "recovered_lags_frames"},
        "event_any_correct_recovery_rate": counts["reacquired_events"] / e if e else None,
        "first_accept_wrong_event_rate": counts["first_accept_wrong_events"] / e if e else None,
        "post_visible_coverage": accepted / visible if visible else None,
        "post_visible_correct_pck8": counts["correct_post_visible_frames"] / visible
            if visible else None,
        "post_visible_wrong_accept_fraction": counts["wrong_post_visible_frames"] / accepted
            if accepted else None,
        "recovered_event_median_lag_frames": statistics.median(lags) if lags else None,
    }


def evaluate(pred: dict, selected: dict, dev: dict,
             public_queries: dict, oracle: dict) -> dict:
    base.validate_sources(pred, selected, dev, public_queries)
    if set(oracle) != set(base.SCENES):
        raise RuntimeError("BATCH75_DEV_ORACLE_POPULATION_DRIFT")
    if tuple(base.GATES) != VARIANTS:
        raise RuntimeError("BATCH75_METHOD_DRIFT")
    records = {r["name"]: r for r in pred["records"]}
    if len(records) != len(base.SCENES) or set(records) != set(base.SCENES):
        raise RuntimeError("BATCH75_DEV_RECORD_SPLIT_DRIFT")

    scenes: dict[str, dict] = {}
    all_events = {v: [] for v in VARIANTS}
    for scene in base.SCENES:
        rec = records[scene]
        qrows = public_queries[scene]
        qmap = {q["id"]: q for q in qrows}
        if (len(qmap) != 4 or set(qmap) != set(rec["tracks"]) or
                set(qmap) != set(oracle[scene]["tracks"])):
            raise RuntimeError("BATCH75_QUERY_ID_DRIFT")
        by_variant = {v: [] for v in VARIANTS}
        for qid, query in qmap.items():
            t0 = query["query_time"]
            n = rec["nframes"]
            gt = oracle[scene]["tracks"][qid]
            tracks = rec["tracks"][qid]
            if (type(t0) is not int or not 0 <= t0 < n - 1 or
                    len(gt["occluded"]) != n or len(gt["xy"]) != n or
                    any(len(tracks[k]) != n for k in (
                        base.BASE, "template_recover4", "affine_stabilized_klt"))):
                raise RuntimeError("BATCH75_QUERY_LENGTH_DRIFT")
            for v in VARIANTS:
                accepted = [False] * n
                for t in range(t0 + 1, n):
                    accepted[t] = base.pass_veto(tracks, t, v)
                by_variant[v].extend(post_occlusion_events(
                    gt["occluded"], gt["xy"], tracks[base.BASE], accepted, t0))
        # Exact replay of existing DEV frame scoring. No method selection.
        replay = base.score_one(rec, qrows, oracle[scene])
        scenes[scene] = {}
        for v in VARIANTS:
            count = aggregate(by_variant[v])
            if (count["occluded_frames"] != replay[v]["invisible"] or
                    count["false_accepted_occluded_frames"] !=
                    replay[v]["invisible_false_accept"]):
                raise RuntimeError("BATCH75_REPLAY_OCCLUSION_MISMATCH")
            scenes[scene][v] = summarize(count)
            all_events[v].extend(by_variant[v])

    overall = {v: summarize(aggregate(all_events[v])) for v in VARIANTS}
    return {
        "kind": "BATCH75_RETROSPECTIVE_DEV_POST_OCCLUSION_RECOVERY_AUDIT",
        "split": "dev",
        "exploratory": True,
        "deployable": False,
        "source_commit": pred["source_commit"],
        "scenes": len(base.SCENES),
        "queries": len(base.SCENES) * 4,
        "variants": list(VARIANTS),
        "pck_radius_pixels": RADIUS,
        "hidden_test_oracle_read": False,
        "test_predictions_read": False,
        "test_score_executions": 0,
        "frozen_batch59_changed": False,
        "interpretation": (
            "Reacquired means >=1 accepted PCK8-correct post-occlusion visible frame; "
            "not continuous identity confirmation. Wrong visible positions are not "
            "proof of physical ID swaps. A right-censored event contributes to "
            "occluded false accepts but not reacquisition success/failure. "
            "Lag is measured only among recovered events and is selection-biased. "
            "Nested frames and queries are not independent test units."
        ),
        "overall": overall,
        "per_scene": scenes,
    }


def load_and_score() -> dict:
    pred = base.pinned_json(base.IN / "dev_predictions.json")
    selected = base.pinned_json(base.IN / "dev_selection.json")
    payload = (base.IN / "dev_results.json").read_bytes()
    if base.sha(payload) != selected["dev_results_sha256"]:
        raise RuntimeError("BATCH75_DEV_RESULTS_PIN_DRIFT")
    dev = json.loads(payload)
    public = json.loads((base.DATA / "queries-public.json").read_bytes())
    base.validate_sources(pred, selected, dev, public)
    oracle = base.load_dev_only_oracle(base.DATA / "oracle-private.json", base.SCENES)
    return evaluate(pred, selected, dev, public, oracle)


def write_immutable(path: Path, value: dict) -> str:
    body = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    pin = path.with_suffix(path.suffix + ".sha256")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or pin.exists():
        raise RuntimeError("BATCH75_OUTPUT_ALREADY_EXISTS")
    with path.open("xb") as f:
        f.write(body)
    digest = base.sha(body)
    with pin.open("x", encoding="ascii") as f:
        f.write(digest + "\n")
    return digest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUT)
    args = p.parse_args()
    result = load_and_score()
    digest = write_immutable(args.output, result)
    print(json.dumps({"status": "BATCH75_DEV_REACQUISITION_AUDIT_PASS",
                      "sha256": digest, "output": str(args.output),
                      "overall": result["overall"]}, sort_keys=True))


if __name__ == "__main__":
    main()
