"""Batch58: immutable, TEST-blind grouped transfer diagnostic on 36 new Batch56 DEV scenes.

This is exploratory analysis, not a model-selection replacement or a safety
certificate. Never reads any TEST filename, private oracle, weights, or media.
The 36 frozen Batch56 DEV scenes have previously been scored; transfer folds
are *within* DEV, and folds share training scenes.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT.parent / "repo-batch56b-tested"
IN_DIR = FROZEN / "outputs" / "kinetics_batch56"
OUTPUT = ROOT / "outputs" / "research" / "batch58_dev_grouped_transfer.json"
KEYS = ("visible_coverage", "invisible_false_accept_rate",
        "jaccard_mean_1_2_4_8_16")


def canonical(obj: Any) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_pinned(path: Path) -> dict:
    tag = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not tag.is_file():
        raise RuntimeError("BATCH58_PIN_MISSING:" + path.name)
    content = path.read_bytes()
    if tag.read_text().strip() != digest(content):
        raise RuntimeError("BATCH58_PIN_INVALID:" + path.name)
    return json.loads(content)


def average(records: list[dict], key: str) -> float | None:
    values = [r[key] for r in records if r[key] is not None]
    return statistics.mean(values) if values else None


def metrics(dev: dict, scenes: list[str], method: str) -> dict:
    records = [dev["per_scene"][s][method] for s in scenes]
    supported = [s for s, r in zip(scenes, records) if r["invisible"] > 0]
    threshold = 0.1
    violating = [s for s in supported
                 if dev["per_scene"][s][method]["invisible_false_accept_rate"] > threshold]
    return {
        "visible_coverage": average(records, KEYS[0]),
        "invisible_false_accept_rate": average(records, KEYS[1]),
        "jaccard_mean_1_2_4_8_16": average(records, KEYS[2]),
        "supported_occlusion_scenes": len(supported),
        "unsupported_occlusion_scenes": len(scenes) - len(supported),
        "violating_occlusion_scenes": len(violating),
        "violating_scene_ids": violating,
        "worst_supported_scene_false_accept_rate": (
            max(dev["per_scene"][s][method]["invisible_false_accept_rate"]
                for s in supported) if supported else None
        ),
        "occlusion_frames": sum(r["invisible"] for r in records),
        "false_accept_frames": sum(r["invisible_false_accept"] for r in records),
    }


def choose(dev: dict, ids: list[str], rules: dict) -> tuple[str, list[str], bool]:
    eligible = []
    for method in rules["candidate_methods"]:
        summary = metrics(dev, ids, method)
        cov, err = summary["visible_coverage"], summary["invisible_false_accept_rate"]
        if (cov is not None and err is not None and
            cov >= rules["min_visible_coverage"] and
            err <= rules["max_invisible_false_accept_rate"]):
            eligible.append(method)
    if not eligible:
        return rules["fallback_method"], [], False
    chosen = max(eligible, key=lambda m: (
        metrics(dev, ids, m)["jaccard_mean_1_2_4_8_16"],
        metrics(dev, ids, m)["visible_coverage"],
    ))
    return chosen, eligible, True


def _check_rate(row: dict, scene: str, method: str) -> None:
    for count in ("visible", "accepted_visible", "invisible", "invisible_false_accept"):
        val = row.get(count)
        if isinstance(val, bool) or not isinstance(val, int) or val < 0:
            raise RuntimeError("BATCH58_INVALID_COUNTS:" + scene + ":" + method)
    if row["accepted_visible"] > row["visible"] or row["invisible_false_accept"] > row["invisible"]:
        raise RuntimeError("BATCH58_INVALID_COUNTS:" + scene + ":" + method)
    for name, numerator, denominator in (
        ("visible_coverage", "accepted_visible", "visible"),
        ("invisible_false_accept_rate", "invisible_false_accept", "invisible"),
    ):
        value = row.get(name)
        if row[denominator] == 0:
            if value is not None:
                raise RuntimeError("BATCH58_UNSUPPORTED_RATE:" + scene + ":" + name)
        elif (isinstance(value, bool) or not isinstance(value, (int, float))
              or not math.isfinite(value) or
              abs(value - row[numerator] / row[denominator]) > 1e-8):
            raise RuntimeError("BATCH58_INCONSISTENT_RATE:" + scene + ":" + name)
    jac = row.get("jaccard_mean_1_2_4_8_16")
    if jac is not None and (
        isinstance(jac, bool) or not isinstance(jac, (int, float))
        or not math.isfinite(jac) or not 0 <= jac <= 1
    ):
        raise RuntimeError("BATCH58_INVALID_JACCARD:" + scene + ":" + method)


def validate(dev: dict, selected: dict, cfg: dict) -> list[str]:
    rules = cfg["dev_selection"]
    methods = rules["candidate_methods"]
    ids = [f"kinetics_{i:02d}" for i in range(36)]
    if (dev.get("split") != "dev" or selected.get("split") != "dev" or
        dev.get("source_commit") != selected.get("source_commit")):
        raise RuntimeError("BATCH58_DEV_ONLY_FIREWALL")
    if (len(methods) != 9 or len(set(methods)) != 9 or
        rules["fallback_method"] not in methods or
        not (0 < rules["max_invisible_false_accept_rate"] < 1) or
        not (0 < rules["min_visible_coverage"] < 1)):
        raise RuntimeError("BATCH58_INVALID_FROZEN_POLICY")
    if (selected.get("rules") != rules or
        selected.get("config_sha256") != dev.get("config_sha256")):
        raise RuntimeError("BATCH58_POLICY_OR_CONFIG_DRIFT")
    if (dev.get("scene_count") != 36 or
        not isinstance(dev.get("per_scene"), dict) or
        set(dev["per_scene"]) != set(ids)):
        raise RuntimeError("BATCH58_INVALID_DEV_POPULATION")
    for scene in ids:
        for method in methods:
            row = dev["per_scene"][scene].get(method)
            if not isinstance(row, dict):
                raise RuntimeError("BATCH58_MISSING_METHOD:" + scene + ":" + method)
            _check_rate(row, scene, method)
    for method in methods:
        summary = metrics(dev, ids, method)
        old = dev["scene_macro"][method]
        for key in KEYS:
            actual, claimed = summary[key], old[key]
            if (actual is None) != (claimed is None) or (
                actual is not None and abs(actual - claimed) > 1e-8
            ):
                raise RuntimeError("BATCH58_DEV_METRIC_INTEGRITY:" + method + ":" + key)
    chosen, eligible, _ = choose(dev, ids, rules)
    if (chosen != selected.get("selected_method") or
        eligible != selected.get("eligible_candidates")):
        raise RuntimeError("BATCH58_SELECTION_INCONSISTENT")
    return ids


def one_fold(dev: dict, heldout: list[str], ids: list[str], rules: dict) -> dict:
    train = [s for s in ids if s not in set(heldout)]
    chosen, eligible, admitted = choose(dev, train, rules)
    return {
        "training_scene_count": len(train),
        "heldout_scene_ids": heldout,
        "method_chosen_without_heldout": chosen,
        "training_method_eligible": admitted,
        "training_eligible_methods": eligible,
        "training_metrics": metrics(dev, train, chosen),
        "heldout_metrics": metrics(dev, heldout, chosen),
    }


def evaluate(dev: dict, selected: dict, cfg: dict) -> dict:
    ids = validate(dev, selected, cfg)
    rules = cfg["dev_selection"]
    # Fixed scene-index blocks, before looking at outcomes; not tuned to errors.
    folds = [one_fold(dev, ids[i * 6:(i + 1) * 6], ids, rules) for i in range(6)]
    loo = [one_fold(dev, [s], ids, rules) for s in ids]
    def summarize(items: list[dict]) -> dict:
        freq: dict[str, int] = {}
        fallback = 0
        scene_rows = []
        for f in items:
            m = f["method_chosen_without_heldout"]
            freq[m] = freq.get(m, 0) + 1
            fallback += not f["training_method_eligible"]
            for s in f["heldout_scene_ids"]:
                row = dev["per_scene"][s][m]
                scene_rows.append((s, m, row))
        supported = [(s, m, r) for s, m, r in scene_rows if r["invisible"] > 0]
        violations = [s for s, m, r in supported
                      if r["invisible_false_accept_rate"] > rules["max_invisible_false_accept_rate"]]
        return {
            "method_choice_frequency": dict(sorted(freq.items())),
            "fallback_fold_count": fallback,
            "heldout_scene_count": len(scene_rows),
            "heldout_supported_occlusion_scenes": len(supported),
            "heldout_unsupported_occlusion_scenes": len(scene_rows) - len(supported),
            "heldout_scene_budget_violations": len(violations),
            "violating_heldout_scene_ids": violations,
            "heldout_scene_macro_visible_coverage": average([r for _, _, r in scene_rows], KEYS[0]),
            "heldout_scene_macro_invisible_false_accept_rate": average([r for _, _, r in scene_rows], KEYS[1]),
            "heldout_scene_macro_jaccard": average([r for _, _, r in scene_rows], KEYS[2]),
            "max_heldout_occlusion_false_accept_rate": (
                max(r["invisible_false_accept_rate"] for _, _, r in supported)
                if supported else None
            ),
            "heldout_total_occlusion_frames": sum(r["invisible"] for _, _, r in scene_rows),
            "heldout_total_false_accept_frames": sum(
                r["invisible_false_accept"] for _, _, r in scene_rows
            ),
        }
    return {
        "kind": "EXPLORATORY_DEV_ONLY_BATCH56_GROUPED_TRANSFER",
        "split": "dev",
        "source_frozen_commit": dev["source_commit"],
        "dev_config_sha256": dev["config_sha256"],
        "dev_results_sha256": selected["dev_results_sha256"],
        "original_selected_method": selected["selected_method"],
        "original_selected_method_all_dev": metrics(dev, ids, selected["selected_method"]),
        "blocked_sixfold": {"folds": folds, "summary": summarize(folds)},
        "leave_one_scene_out": {"folds": loo, "summary": summarize(loo)},
        "interpretation": [
            "DEV-only grouped holdouts; no TEST scores, oracles, videos, predictions or labels loaded.",
            "Blocked folds contain six consecutive scene IDs; not randomized or optimized for outcomes.",
            "Group folds and leave-one-out training sets overlap; results are descriptive, not independent replications.",
            "Methods and thresholds are frozen from Batch56; no changes to the active Batch56 runner.",
            "Scene-level error-budget violations are not a safety certificate, causal mechanism attribution or deployment guarantee.",
            "Some scenes have zero annotated invisible frames and are excluded from occlusion-risk denominators.",
        ],
        "test_oracle_read": False,
        "test_predictions_read": False,
        "retrained_models": False,
        "changes_frozen_experiment": False,
    }


def run() -> dict:
    path = IN_DIR / "dev_results.json"
    selected = load_pinned(IN_DIR / "dev_selection.json")
    cfg = json.loads((FROZEN / "configs" / "kinetics_batch56.json").read_text())
    dev_raw = path.read_bytes()
    if digest(dev_raw) != selected.get("dev_results_sha256"):
        raise RuntimeError("BATCH58_DEV_RESULTS_PIN_INVALID")
    if digest(canonical(cfg)) != selected.get("config_sha256"):
        raise RuntimeError("BATCH58_CONFIG_PIN_INVALID")
    frozen_head = subprocess.check_output(
        ["git", "-C", str(FROZEN), "rev-parse", "HEAD"], text=True
    ).strip()
    if frozen_head != selected.get("source_commit"):
        raise RuntimeError("BATCH58_FROZEN_SOURCE_DRIFT")
    result = evaluate(json.loads(dev_raw), selected, cfg)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(result)
    tag = OUTPUT.with_suffix(".json.sha256")
    if OUTPUT.exists() and OUTPUT.read_bytes() != data:
        raise RuntimeError("BATCH58_IMMUTABLE_REPORT_CONFLICT")
    if tag.exists() and tag.read_text().strip() != digest(data):
        raise RuntimeError("BATCH58_OUTPUT_PIN_INVALID")
    if not OUTPUT.exists():
        OUTPUT.write_bytes(data)
    if not tag.exists():
        tag.write_text(digest(data) + "\n")
    print("BATCH58_DEV_GROUPED_TRANSFER_PINNED", digest(data))
    for name in ("blocked_sixfold", "leave_one_scene_out"):
        print(name, json.dumps(result[name]["summary"], sort_keys=True))
    return result


if __name__ == "__main__":
    run()
