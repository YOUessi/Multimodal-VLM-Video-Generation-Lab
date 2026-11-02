"""Prospectively frozen Batch59 DEV-only group-risk *diagnostic*.

Uses exactly the sealed 36 DEV scenes, six predeclared contiguous groups,
and nine unchanged CPU methods. Computes no TEST statistics and changes
neither method selection nor the TEST prediction/score sequence.

A group-safe candidate is an exploratory comparison, NOT a deployment
certificate: groups overlap in their natural-domain sources, selection and
diagnosis share the same DEV set, and scene exchangeability is unproved.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import batch58_dev_grouped_transfer as previous
import batch59_dev_scene_risk as scene_risk

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "kinetics_batch59"
CONFIG = ROOT / "configs" / "kinetics_batch59.json"
GROUP_SIZE = 6
ALPHA = .05


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def canonical(data: object) -> bytes:
    return (json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def analyze(dev: dict, selected: dict, cfg: dict) -> dict:
    ids = previous.validate(dev, selected, cfg)
    if len(ids) != 36 or GROUP_SIZE != 6:
        raise RuntimeError("BATCH59_FROZEN_GROUP_DEFINITION_INVALID")
    policy = cfg["dev_selection"]
    methods = policy["candidate_methods"]
    safety = float(policy["max_invisible_false_accept_rate"])
    cover = float(policy["min_visible_coverage"])
    scene_certificate = scene_risk.compute(dev, selected, cfg)
    groups = [ids[i:i + GROUP_SIZE] for i in range(0, len(ids), GROUP_SIZE)]
    if len(groups) != 6 or any(len(x) != GROUP_SIZE for x in groups):
        raise RuntimeError("BATCH59_INCOMPLETE_PREDECLARED_FOLDS")
    records: dict[str, dict] = {}
    for method in methods:
        total = previous.metrics(dev, ids, method)
        per_group = [previous.metrics(dev, group, method) for group in groups]
        unsupported = [i for i, row in enumerate(per_group)
                       if not row["supported_occlusion_scenes"]]
        exceeds = [i for i, row in enumerate(per_group)
                   if row["invisible_false_accept_rate"] is not None
                   and row["invisible_false_accept_rate"] > safety]
        conservative_group_pass = (
            not unsupported and not exceeds
            and total["visible_coverage"] is not None
            and total["visible_coverage"] >= cover
        )
        clustered = scene_certificate["methods"][method]
        bound = clustered["nine_method_familywise_95_upper"]
        statistical_support = (
            bound is not None and bound <= safety
        )
        records[method] = {
            "whole_dev": total,
            "groups": [
                {"group_index": i, "scene_ids": group, "metrics": row}
                for i, (group, row) in enumerate(zip(groups, per_group))
            ],
            "groups_without_occlusion_evidence": unsupported,
            "groups_exceeding_mean_safety_threshold": exceeds,
            "passes_exploratory_group_mean_screen": conservative_group_pass,
            "familywise_scene_violation_upper_95": bound,
            "passes_familywise_scene_violation_screen": statistical_support,
            "deployment_certificate": False,
        }
    grouped_candidates = [
        method for method in methods
        if records[method]["passes_exploratory_group_mean_screen"]
    ]
    proposed = (
        max(grouped_candidates, key=lambda method: (
            records[method]["whole_dev"]["jaccard_mean_1_2_4_8_16"],
            records[method]["whole_dev"]["visible_coverage"]
        ))
        if grouped_candidates else None
    )
    old = selected["selected_method"]
    return {
        "kind": "BATCH59_PREDECLARED_DEV_ONLY_GROUP_RISK_SCREEN",
        "split": "dev",
        "source_commit": dev["source_commit"],
        "config_sha256": dev["config_sha256"],
        "dev_results_sha256": selected["dev_results_sha256"],
        "selected_by_frozen_policy": old,
        "selected_method_group_screen_pass": records[old]["passes_exploratory_group_mean_screen"],
        "group_screen_status": ("EXPLORATORY_GROUP_PASS"
                                if records[old]["passes_exploratory_group_mean_screen"]
                                else "BLOCK_GROUP_RISK_OR_MISSING_SUPPORT"),
        "group_screen_candidate_methods": grouped_candidates,
        "counterfactual_dev_only_group_candidate": proposed,
        "risk_limit": safety,
        "min_visible_coverage": cover,
        "predeclared_groups": groups,
        "method_count": len(methods),
        "methods": records,
        "deployment_authorized": False,
        "interpretation": [
            "All groups and thresholds fixed before Batch59 DEV outcomes are observed.",
            "This is a secondary diagnostic; it cannot change the primary frozen method or its TEST score.",
            "Selected and counterfactual methods share DEV labels, so no independent performance claim.",
            "Contiguous six-scene groups are not necessarily distinct environmental domains.",
            "Unsupported occlusion groups are not treated as passing safety.",
            "Familywise Clopper-Pearson requires unverified scene exchangeability, and measures scene risk-budget violation, not frame risk.",
            "Passing this audit never authorizes production deployment.",
        ],
        "test_oracle_read": False,
        "test_predictions_read": False,
        "historical_test_scores_used": False,
        "changes_frozen_method_selection": False,
    }


def run() -> dict:
    dev_raw = (OUT / "dev_results.json").read_bytes()
    selection_path = OUT / "dev_selection.json"
    selection_raw = selection_path.read_bytes()
    tag = selection_path.with_suffix(".json.sha256")
    if not tag.is_file() or tag.read_text().strip() != digest(selection_raw):
        raise RuntimeError("BATCH59_SELECTION_PIN_INVALID")
    dev = json.loads(dev_raw)
    selected = json.loads(selection_raw)
    if selected.get("dev_results_sha256") != digest(dev_raw):
        raise RuntimeError("BATCH59_DEV_RESULT_PIN_INVALID")
    cfg = json.loads(CONFIG.read_bytes())
    if digest(canonical(cfg)) != selected.get("config_sha256"):
        raise RuntimeError("BATCH59_CONFIG_PIN_INVALID")
    frozen_sha = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
    ).strip()
    if frozen_sha != selected.get("source_commit"):
        raise RuntimeError("BATCH59_SOURCE_DRIFT")
    result = analyze(dev, selected, cfg)
    output = OUT / "dev_group_safety.json"
    content = canonical(result)
    pin = output.with_suffix(".json.sha256")
    if output.exists() and output.read_bytes() != content:
        raise RuntimeError("BATCH59_IMMUTABLE_GROUP_REPORT_CONFLICT")
    if pin.exists() and pin.read_text().strip() != digest(content):
        raise RuntimeError("BATCH59_GROUP_REPORT_PIN_INVALID")
    if not output.exists():
        output.write_bytes(content)
    if not pin.exists():
        pin.write_text(digest(content) + "\n")
    print("BATCH59_GROUP_SCREEN", result["group_screen_status"],
          "selected", result["selected_by_frozen_policy"],
          "counterfactual", result["counterfactual_dev_only_group_candidate"],
          "sha256", digest(content), flush=True)
    return result


if __name__ == "__main__":
    run()
