"""Batch66: evaluate *occlusion episodes*, not only average accepted frames.

Exploratory on previously studied Batch56 DEV only. Uses a private DEV oracle
only after pinned public predictions and DEV selection are authenticated.
TEST keys are never parsed. This is retrospective scoring, NOT a deployable
visibility gate and NOT a fresh blind test. No tracker parameter is changed.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from experiments import batch62_dev_consensus_veto as base

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs/research/batch66_dev_episode_safety.json"
VARIANTS = ("baseline", "template_4px", "affine_3px", "both_4px_3px")
RUN_THRESHOLDS = (1, 3, 10, 30, 60)
SCENE_IDS = tuple(base.SCENES)
BUDGET = base.BUDGET


def occlusion_episodes(occluded: list[bool], accepted: list[bool],
                       query_time: int) -> list[dict]:
    """Maximal contiguous occlusion blocks strictly AFTER query_time.

    A false acceptance at frame t means (occluded[t] and accepted[t]).
    No location labels or future state enter the acceptance decision.
    The full sequence is allowed ONLY in this offline DEV scorer.
    """
    if (type(query_time) is not int or query_time < 0 or
            len(occluded) != len(accepted) or query_time >= len(occluded)):
        raise RuntimeError("BATCH66_QUERY_LENGTH_INVALID")
    if any(type(v) is not bool for v in occluded + accepted):
        raise RuntimeError("BATCH66_BOOLEAN_TRACE_REQUIRED")
    episodes = []
    t = query_time + 1
    while t < len(occluded):
        if not occluded[t]:
            t += 1
            continue
        start = t
        longest = current_run = accepts = 0
        first_accept_offset = None
        while t < len(occluded) and occluded[t]:
            if accepted[t]:
                accepts += 1
                current_run += 1
                longest = max(longest, current_run)
                if first_accept_offset is None:
                    first_accept_offset = t - start
            else:
                current_run = 0
            t += 1
        episodes.append({
            "start": start,
            "end_exclusive": t,
            "occluded_frames": t - start,
            "false_accepted_frames": accepts,
            "longest_false_accept_run": longest,
            "first_false_accept_offset": first_accept_offset,
        })
    return episodes


def episode_totals(episodes: list[dict]) -> dict:
    total = {
        "occlusion_events": len(episodes),
        "occluded_frames": sum(e["occluded_frames"] for e in episodes),
        "false_accepted_frames": sum(e["false_accepted_frames"] for e in episodes),
        "affected_events": sum(e["false_accepted_frames"] > 0 for e in episodes),
        "worst_false_accept_run": max(
            (e["longest_false_accept_run"] for e in episodes), default=0),
        "false_accept_frames_in_events_with_run_ge30": sum(
            e["false_accepted_frames"] for e in episodes
            if e["longest_false_accept_run"] >= 30),
    }
    for n in RUN_THRESHOLDS:
        total[f"events_with_run_ge{n}"] = sum(
            e["longest_false_accept_run"] >= n for e in episodes)
    return total


def make_query_events(pred_track: dict, occ: list[bool],
                      query_time: int, n: int) -> dict[str, list[dict]]:
    if len(occ) != n:
        raise RuntimeError("BATCH66_ORACLE_TRACK_LENGTH")
    for method in (base.BASE, "template_recover4", "affine_stabilized_klt"):
        if method not in pred_track or len(pred_track[method]) != n:
            raise RuntimeError("BATCH66_PREDICTION_TRACK_LENGTH")
        if any(p is not None for p in pred_track[method][:query_time]):
            raise RuntimeError("BATCH66_PREQUERY_PREDICTION")
    result = {}
    for variant in VARIANTS:
        accepted = [False] * n
        for t in range(query_time + 1, n):
            accepted[t] = base.pass_veto(pred_track, t, variant)
        result[variant] = occlusion_episodes(occ, accepted, query_time)
    return result


def sum_counts(rows: list[dict]) -> dict:
    out = {k: 0 for k in episode_totals([])}
    for row in rows:
        for k in out:
            if k == "worst_false_accept_run":
                out[k] = max(out[k], row[k])
            else:
                out[k] += row[k]
    if out["false_accepted_frames"] > out["occluded_frames"]:
        raise RuntimeError("BATCH66_IMPOSSIBLE_COUNTS")
    if out["affected_events"] > out["occlusion_events"]:
        raise RuntimeError("BATCH66_IMPOSSIBLE_EVENTS")
    return out


def evaluate(pred: dict, sel: dict, dev: dict,
             public_queries: dict, oracle: dict) -> dict:
    base.validate_sources(pred, sel, dev, public_queries)
    if set(oracle) != set(SCENE_IDS) or len(pred["records"]) != len(SCENE_IDS):
        raise RuntimeError("BATCH66_DEV_ONLY_POPULATION_REQUIRED")
    records = {r["name"]: r for r in pred["records"]}
    if set(records) != set(SCENE_IDS):
        raise RuntimeError("BATCH66_PREDICTION_SPLIT_INVALID")
    if set(VARIANTS) != set(base.GATES):
        raise RuntimeError("BATCH66_VARIANTS_DRIFT")

    scene_rows = {}
    for scene in SCENE_IDS:
        rec = records[scene]
        queries = public_queries[scene]
        qmap = {q["id"]: q for q in queries}
        oracle_tracks = oracle[scene]["tracks"]
        if len(qmap) != 4 or set(qmap) != set(rec["tracks"]) or set(qmap) != set(oracle_tracks):
            raise RuntimeError("BATCH66_QUERY_ID_MISMATCH")
        by_variant = {k: [] for k in VARIANTS}
        for qid, q in qmap.items():
            gt = oracle_tracks[qid]
            t0, n = q["query_time"], rec["nframes"]
            if (type(t0) is not int or t0 < 0 or t0 >= n - 1 or
                    len(gt["occluded"]) != n or len(gt["xy"]) != n):
                raise RuntimeError("BATCH66_QUERY_ORACLE_LENGTH_INVALID")
            events = make_query_events(rec["tracks"][qid], gt["occluded"], t0, n)
            for variant in VARIANTS:
                by_variant[variant].extend(events[variant])

        # Independently replay the existing frozen frame-level metrics and
        # compare exact counts with the trusted DEV result before reporting.
        frame_counts = base.score_one(rec, queries, oracle[scene])
        scene_rows[scene] = {}
        for variant in VARIANTS:
            rows = episode_totals(by_variant[variant])
            c = frame_counts[variant]
            if (rows["occluded_frames"] != c["invisible"] or
                    rows["false_accepted_frames"] != c["invisible_false_accept"]):
                raise RuntimeError("BATCH66_EPISODE_FRAME_REPLAY_MISMATCH")
            rows["visible_frames"] = c["visible"]
            rows["accepted_visible_frames"] = c["accepted_visible"]
            rows["correct_visible_frames_pck8"] = c["correct8"]
            rows["occluded_false_accept_rate"] = (
                rows["false_accepted_frames"] / rows["occluded_frames"]
                if rows["occluded_frames"] else None)
            rows["event_affected_rate"] = (
                rows["affected_events"] / rows["occlusion_events"]
                if rows["occlusion_events"] else None)
            scene_rows[scene][variant] = rows
        reference = dev["per_scene"][scene][base.BASE]
        for key in ("visible", "invisible", "accepted_visible", "correct8",
                    "invisible_false_accept", "predicted_visible"):
            if frame_counts["baseline"][key] != reference[key]:
                raise RuntimeError("BATCH66_PRIOR_DEV_METRIC_MISMATCH:" + scene + ":" + key)

    summary = {}
    for variant in VARIANTS:
        rows = [scene_rows[s][variant] for s in SCENE_IDS]
        totals = sum_counts(rows)
        support = [r for r in rows if r["occluded_frames"] > 0]
        e_support = [r for r in rows if r["occlusion_events"] > 0]
        violations = [s for s in SCENE_IDS if scene_rows[s][variant][
            "occluded_false_accept_rate"] is not None and
            scene_rows[s][variant]["occluded_false_accept_rate"] > BUDGET]
        summary[variant] = {
            **totals,
            "scene_supported_occlusion": len(support),
            "scene_unsupported_occlusion": len(rows) - len(support),
            "scene_budget_violations": violations,
            "scene_budget_violation_count": len(violations),
            "scene_macro_occluded_false_accept_rate": statistics.mean(
                r["occluded_false_accept_rate"] for r in support) if support else None,
            "scene_macro_affected_occlusion_events": statistics.mean(
                r["event_affected_rate"] for r in e_support) if e_support else None,
            "micro_occluded_false_accept_rate": (
                totals["false_accepted_frames"] / totals["occluded_frames"]
                if totals["occluded_frames"] else None),
            "micro_affected_event_rate": (
                totals["affected_events"] / totals["occlusion_events"]
                if totals["occlusion_events"] else None),
            "micro_visible_coverage": (
                sum(r["accepted_visible_frames"] for r in rows) /
                sum(r["visible_frames"] for r in rows)),
            "micro_correct_visible_pck8": (
                sum(r["correct_visible_frames_pck8"] for r in rows) /
                sum(r["visible_frames"] for r in rows)),
            "worst_supported_scene_false_accept_rate": max(
                (r["occluded_false_accept_rate"] for r in support), default=None),
        }
    return {
        "kind": "BATCH66_DEV_ONLY_OCCLUSION_EPISODE_SAFETY_AUDIT",
        "split": "dev",
        "exploratory": True,
        "deployable": False,
        "source_commit": pred["source_commit"],
        "dev_scene_count": len(SCENE_IDS),
        "dev_query_count": len(SCENE_IDS) * 4,
        "prediction_sha256": dev["prediction_sha256"],
        "dev_results_sha256": sel["dev_results_sha256"],
        "fixed_variants": VARIANTS,
        "fixed_run_thresholds": RUN_THRESHOLDS,
        "scene_budget": BUDGET,
        "oracle_dev_only_streamed_through": SCENE_IDS[-1],
        "test_oracle_read": False,
        "test_predictions_read": False,
        "selected_method_changed": False,
        "frozen_batch59_changed": False,
        "interpretation": (
            "False acceptance during an oracle-labeled occlusion does not prove "
            "which object or background was followed. Event metrics require "
            "private DEV occlusion labels and cannot serve as inference gates. "
            "Nested temporal samples are not independent validation units."
        ),
        "summary": summary,
        "per_scene": scene_rows,
    }


def load_and_score() -> dict:
    pred = base.pinned_json(base.IN / "dev_predictions.json")
    sel = base.pinned_json(base.IN / "dev_selection.json")
    payload = (base.IN / "dev_results.json").read_bytes()
    if base.sha(payload) != sel["dev_results_sha256"]:
        raise RuntimeError("BATCH66_DEV_RESULTS_PIN_DRIFT")
    dev = json.loads(payload)
    public_queries = json.loads((base.DATA / "queries-public.json").read_bytes())
    base.validate_sources(pred, sel, dev, public_queries)
    oracle = base.load_dev_only_oracle(base.DATA / "oracle-private.json", list(SCENE_IDS))
    return evaluate(pred, sel, dev, public_queries, oracle)


def write_immutable(path: Path, report: dict) -> str:
    data = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    hash_path = path.with_suffix(path.suffix + ".sha256")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or hash_path.exists():
        raise RuntimeError("BATCH66_NO_OVERWRITE")
    with path.open("xb") as stream:
        stream.write(data)
    digest = base.sha(data)
    with hash_path.open("x", encoding="ascii") as stream:
        stream.write(digest + "\n")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args()
    report = load_and_score()
    digest = write_immutable(args.output, report)
    print(json.dumps({"output": str(args.output), "sha256": digest,
                      "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
