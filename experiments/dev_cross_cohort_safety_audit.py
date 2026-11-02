"""DEV-only cross-cohort occlusion safety audit for already-frozen natural-video batches.

The three input cohorts are Batch48B DEV, Batch53B DEV and Batch55 DEV.
Never access TEST videos, labels, scores, or predictions. This exploratory
diagnostic cannot change the Batch56 frozen method or any TEST policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from batch56_dev_scene_risk import one_sided_upper
from dev_scene_heldout_safety_audit import choose, macro

COHORTS = (("48B", "48b", "48"), ("53B", "53b", "53"), ("55", "55", "55"))
RATE_KEYS = ("visible_coverage", "invisible_false_accept_rate",
             "jaccard_mean_1_2_4_8_16")
ALPHA = 0.05


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def validate_cohort(name: str, result: dict, selected: dict, config: dict,
                    result_sha: str, config_sha: str) -> dict:
    """Reject wrong split, unbound selection, corrupt rates, and missing support."""
    if result.get("split") != "dev" or selected.get("split") != "dev":
        raise ValueError("DEV_ONLY_SPLIT_REQUIRED:" + name)
    if result.get("source_commit") != selected.get("source_commit"):
        raise ValueError("SOURCE_COMMIT_MISMATCH:" + name)
    if result.get("config_sha256") != config_sha or selected.get("config_sha256") != config_sha:
        raise ValueError("CONFIG_SHA_MISMATCH:" + name)
    if selected.get("dev_results_sha256") != result_sha:
        raise ValueError("DEV_SELECTION_RESULT_UNPINNED:" + name)
    if selected.get("selection_source") != "DEV only":
        raise ValueError("NOT_DEV_ONLY_SELECTION:" + name)
    rules = config["dev_selection"]
    if selected.get("rules") != rules:
        raise ValueError("SELECTION_RULES_DRIFT:" + name)
    candidates = rules["candidate_methods"]
    if not isinstance(candidates, list) or len(candidates) != 9 or len(set(candidates)) != len(candidates):
        raise ValueError("CANDIDATE_SET_INVALID:" + name)
    if rules["fallback_method"] not in candidates:
        raise ValueError("FALLBACK_INVALID:" + name)
    if (rules["max_invisible_false_accept_rate"] != 0.10
            or rules["min_visible_coverage"] != 0.30):
        raise ValueError("FROZEN_POLICY_MISMATCH:" + name)
    scenes = result.get("per_scene")
    expected_names = {f"kinetics_{idx:02d}" for idx in range(6)}
    if not isinstance(scenes, dict) or set(scenes) != expected_names:
        raise ValueError("EXPECTED_SIX_DEV_SCENES:" + name)
    if not set(candidates).issubset(result.get("methods", [])):
        raise ValueError("MISSING_CANDIDATE_METHODS:" + name)
    for scene_name, methods in scenes.items():
        for method in candidates:
            record = methods.get(method)
            if not isinstance(record, dict):
                raise ValueError("SCENE_METHOD_MISSING:" + scene_name + ":" + method)
            for key in ("visible", "invisible", "invisible_false_accept"):
                n = record.get(key)
                if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                    raise ValueError("INVALID_COUNT:" + name + ":" + key)
            if record["invisible_false_accept"] > record["invisible"]:
                raise ValueError("INVALID_FALSE_ACCEPT_COUNT:" + name)
            for key in RATE_KEYS:
                rate = record.get(key)
                if rate is None:
                    if key != "invisible_false_accept_rate" or record["invisible"] != 0:
                        raise ValueError("INVALID_NULL_RATE:" + key)
                    continue
                if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                    raise ValueError("INVALID_RATE_TYPE:" + key)
                if not math.isfinite(rate) or not (0 <= rate <= 1):
                    raise ValueError("INVALID_RATE_RANGE:" + key)
            n = record["invisible"]
            if n and abs(record["invisible_false_accept_rate"] - record["invisible_false_accept"] / n) > 1e-8:
                raise ValueError("SCENE_RATE_COUNT_MISMATCH:" + name)
    expected_method, eligible = choose(scenes, sorted(scenes), rules)
    if selected["selected_method"] != expected_method or selected["eligible_candidates"] != eligible:
        raise ValueError("RECOMPUTED_FROZEN_CHOICE_MISMATCH:" + name)
    return {
        "name": name, "scenes": scenes, "rules": rules,
        "selected_method": expected_method, "eligible": eligible,
        "result_sha256": result_sha, "config_sha256": config_sha,
        "source_commit": result["source_commit"],
    }


def load_frozen_dev(root: Path) -> list[dict]:
    cohorts = []
    for name, suffix, version in COHORTS:
        repo = root / f"repo-batch{suffix}-tested"
        folder = repo / f"outputs/kinetics_batch{version}"
        paths = (folder / "dev_results.json", folder / "dev_selection.json",
                 repo / f"configs/kinetics_batch{version}.json")
        if not all(p.is_file() for p in paths):
            raise FileNotFoundError("MISSING_FROZEN_DEV_SOURCE:" + name)
        for path in paths:
            if path.name.startswith("test_") or "/test_" in str(path).lower():
                raise ValueError("TEST_FILE_PROHIBITED")
        result, selected, config = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        if sha(paths[1]) != (paths[1].with_suffix(".json.sha256")).read_text().strip():
            raise ValueError("FROZEN_SELECTION_PIN_MISMATCH:" + name)
        cohorts.append(validate_cohort(
            name, result, selected, config, sha(paths[0]),
            hashlib.sha256(canonical(config)).hexdigest(),
        ))
    return cohorts


def summarize(scenes: dict, method: str, threshold: float) -> dict:
    supported = [
        (name, methods[method]) for name, methods in sorted(scenes.items())
        if methods[method]["invisible"] > 0
    ]
    violations = [
        name for name, row in supported
        if row["invisible_false_accept_rate"] > threshold
    ]
    n = len(supported)
    return {
        "method": method,
        "scene_macro_visible_coverage": macro(scenes, sorted(scenes), method, "visible_coverage"),
        "scene_macro_occlusion_false_accept": macro(scenes, sorted(scenes), method, "invisible_false_accept_rate"),
        "scene_macro_jaccard": macro(scenes, sorted(scenes), method, "jaccard_mean_1_2_4_8_16"),
        "occlusion_supported_scenes": n,
        "occlusion_unsupported_scenes": len(scenes) - n,
        "violating_scene_count": len(violations),
        "violating_scene_ids": violations,
        "worst_supported_scene_false_accept": max(
            (row["invisible_false_accept_rate"] for _, row in supported), default=None
        ),
        "nominal_95_upper_scene_violation_probability": one_sided_upper(len(violations), n, ALPHA),
    }


def compute(cohorts: list[dict]) -> dict:
    if len(cohorts) != 3 or len({c["name"] for c in cohorts}) != 3:
        raise ValueError("THREE_DISTINCT_DEV_COHORTS_REQUIRED")
    names = {c["name"] for c in cohorts}
    if names != {"48B", "53B", "55"}:
        raise ValueError("UNEXPECTED_COHORT_POPULATION")
    rules = cohorts[0]["rules"]
    policy_keys = ("candidate_methods", "fallback_method", "min_visible_coverage",
                   "max_invisible_false_accept_rate")
    for cohort in cohorts:
        if any(cohort["rules"][k] != rules[k] for k in policy_keys):
            raise ValueError("CROSS_COHORT_POLICY_DRIFT")
    folds = []
    for heldout in cohorts:
        train = {
            cohort["name"] + ":" + name: methods
            for cohort in cohorts if cohort["name"] != heldout["name"]
            for name, methods in cohort["scenes"].items()
        }
        selected, eligible = choose(train, sorted(train), rules)
        fold = {
            "heldout_dev_cohort": heldout["name"],
            "training_dev_scene_count": len(train),
            "training_cohorts": sorted(c["name"] for c in cohorts if c["name"] != heldout["name"]),
            "cross_cohort_dev_selected_method": selected,
            "training_eligible_candidates": eligible,
            "train_dev": summarize(train, selected, rules["max_invisible_false_accept_rate"]),
            "heldout_dev": summarize(
                heldout["scenes"], selected, rules["max_invisible_false_accept_rate"]
            ),
            "heldout_cohort_original_in_sample_choice": heldout["selected_method"],
            "heldout_cohort_original_in_sample_summary": summarize(
                heldout["scenes"], heldout["selected_method"],
                rules["max_invisible_false_accept_rate"]
            ),
        }
        folds.append(fold)
    return {
        "kind": "EXPLORATORY_DEV_ONLY_CROSS_COHORT_SCENE_SAFETY_AUDIT",
        "input_cohorts": [
            {k: c[k] for k in ("name", "result_sha256", "config_sha256",
                               "source_commit", "selected_method")}
            for c in cohorts
        ],
        "frozen_policy": {k: rules[k] for k in policy_keys},
        "folds": folds,
        "test_data_or_labels_read": False,
        "method_or_threshold_changed": False,
        "interpretation": (
            "All three cohorts are previously studied DEV splits, not a prospectively blind "
            "TEST. Each fold selects with two DEV cohorts and audits on the third DEV cohort. "
            "Six scene clusters per heldout fold have limited support. Clopper-Pearson bounds "
            "assume independent/exchangeable scenes, unproven for Kinetics. Do not use this "
            "exploratory analysis to tune existing frozen methods or claim deployment safety."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = compute(load_frozen_dev(args.root.resolve()))
    data = canonical(audit)
    if args.output:
        dest = args.output.resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        pin = dest.with_suffix(dest.suffix + ".sha256")
        new_sha = hashlib.sha256(data).hexdigest()
        if dest.exists() and dest.read_bytes() != data:
            raise RuntimeError("EXPLORATORY_DEV_AUDIT_WRITE_ONCE_CONFLICT")
        if pin.exists() and pin.read_text().strip() != new_sha:
            raise RuntimeError("EXPLORATORY_DEV_AUDIT_SHA_CONFLICT")
        if not dest.exists():
            dest.write_bytes(data)
        if not pin.exists():
            pin.write_text(new_sha + "\n")
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
