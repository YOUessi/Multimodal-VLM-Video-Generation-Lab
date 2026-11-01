"""Read-only, post-hoc failure attribution on an already scored frozen heldout.

This is NOT threshold optimization, model selection, or an additional blind test.
Reports rescue/harm counts, full-denominator and selective risk with source-pair
cluster bootstrap. Every input byte is pinned to an expected SHA-256.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
from pathlib import Path

import domain_shift_motion_eval as base

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("plain", "affine", "shape_affine")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen_source(root: Path, config: dict) -> None:
    if not (root / ".git").exists():
        raise RuntimeError("SOURCE_ROOT_NOT_GIT")
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != config["source_commit"]:
        raise RuntimeError("FROZEN_SOURCE_COMMIT_DRIFT")
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("FROZEN_SOURCE_TRACKED_DIRTY")
    for rel, expected in config["artifact_sha256"].items():
        target = root / rel
        if not target.is_file():
            raise FileNotFoundError(f"FROZEN_ARTIFACT_MISSING:{rel}")
        if sha256(target) != expected:
            raise RuntimeError(f"FROZEN_ARTIFACT_SHA_DRIFT:{rel}")


def complete_records(doc: dict, expected: set, key, name: str) -> dict:
    records = doc["records"]
    result = {key(r): r for r in records}
    if len(result) != len(records) or set(result) != expected:
        raise RuntimeError(f"FROZEN_{name}_COVERAGE_OR_DUPLICATE")
    return result


def risk_counts(rows: list, cv: dict, model: dict, method: str, budget: int) -> dict:
    """Pure accounting; never changes a prediction or estimates a threshold."""
    count = collections.Counter()
    for row in rows:
        pred = cv[row["id"]]["predictions"][method]
        vlm = model[(row["id"], budget)]["prediction"]
        truth = row["gold_letter"]
        vlm_ok = vlm == truth
        accepted = pred["accepted"]
        # The accepted direction selects an existing multiple-choice answer;
        # missing/invalid accepted directions fail closed instead of falling
        # back silently and hiding incorrect tool overrides.
        tool_letter = base.choice_for_direction(row, pred["gated_direction"]) if accepted else None
        if accepted and (not tool_letter or not pred["gated_direction"]):
            raise RuntimeError("ACCEPTED_GATE_WITHOUT_VALID_DIRECTION")
        tool_ok = tool_letter == truth
        hybrid_ok = tool_ok if accepted else vlm_ok
        count["n"] += 1
        count["vlm_correct"] += int(vlm_ok)
        count["hybrid_correct"] += int(hybrid_ok)
        count["accepted"] += int(accepted)
        count["gate_correct"] += int(accepted and tool_ok)
        count["gate_wrong"] += int(accepted and not tool_ok)
        count["rescue"] += int(accepted and tool_ok and not vlm_ok)
        count["harm"] += int(accepted and not tool_ok and vlm_ok)
        count["wrong_both"] += int(accepted and not tool_ok and not vlm_ok)
        count["correct_both"] += int(accepted and tool_ok and vlm_ok)
        count["abstained"] += int(not accepted)
    if count["gate_correct"] + count["gate_wrong"] != count["accepted"]:
        raise RuntimeError("GATE_ACCOUNTING_INCONSISTENT")
    if count["hybrid_correct"] - count["vlm_correct"] != count["rescue"] - count["harm"]:
        raise RuntimeError("RESCUE_HARM_ACCOUNTING_INCONSISTENT")
    return {
        **dict(count),
        "gate_precision": round(count["gate_correct"] / count["accepted"], 5) if count["accepted"] else None,
        "gate_coverage": round(count["accepted"] / count["n"], 5) if count["n"] else None,
        "hybrid_accuracy": round(count["hybrid_correct"] / count["n"], 5) if count["n"] else None,
        "vlm_accuracy": round(count["vlm_correct"] / count["n"], 5) if count["n"] else None,
        "net_change_correct": count["rescue"] - count["harm"],
    }


def audit(config: dict) -> dict:
    src = Path(config["source_root"])
    verify_frozen_source(src, config)
    study_cfg = base.read(config["study_config"], root=src)
    manifest = base.manifest(study_cfg, verify_bytes=True, root=src)
    if manifest["source_sha256"] != config["source_manifest_rows_sha256"]:
        raise RuntimeError("SOURCE_MANIFEST_ROWS_SHA_MISMATCH")
    test = [r for r in manifest["rows"] if r["split"] == "test"]
    expected_ids = {r["id"] for r in test}
    if len(test) != 2 * study_cfg["test_pairs"] or len(expected_ids) != len(test):
        raise RuntimeError("INDEPENDENT_TEST_COVERAGE_DRIFT")
    cv_doc = base.read(Path(study_cfg["output_dir"]) / "cv_predictions.json", root=src)
    if (cv_doc["source_sha256"] != manifest["source_sha256"]
        or cv_doc["methods"] != list(METHODS)
        or cv_doc["prediction_count"] != len(test)
        or cv_doc.get("no_gold_labels_in_prediction_records") is not True):
        raise RuntimeError("INVALID_FROZEN_CV_HEADER")
    cv = complete_records(cv_doc, expected_ids, lambda r: r["id"], "CV")
    reference = base.read(Path(study_cfg["output_dir"]) / "results.json", root=src)
    if (reference["source_sha256"] != manifest["source_sha256"]
            or reference["test_clips"] != len(test)
            or reference["independent_scene_pairs"] != study_cfg["test_pairs"]):
        raise RuntimeError("ORIGINAL_PUBLISHED_RESULT_HEADER_DRIFT")
    pair_ids = {r["id"]: r["pair_id"] for r in test}
    result = {
        "posthoc_read_only": True,
        "already_scored_test_not_for_model_selection": True,
        "synthetic_only": True,
        "source_commit": config["source_commit"],
        "source_manifest_rows_sha256": manifest["source_sha256"],
        "test_clips": len(test), "independent_scene_pairs": study_cfg["test_pairs"],
        "artifact_sha256": config["artifact_sha256"],
        "model_risks": {},
    }
    for model_name, budgets in study_cfg["frame_budgets"].items():
        doc = base.read(Path(study_cfg["output_dir"]) / (model_name + ".json"), root=src)
        if (doc["source_sha256"] != manifest["source_sha256"]
            or doc["model_name"] != model_name
            or doc["model_revision"] != study_cfg["models"][model_name]["revision"]):
            raise RuntimeError("INVALID_FROZEN_MODEL_HEADER")
        expected = {(r["id"], b) for r in test for b in budgets}
        models = complete_records(doc, expected, lambda r: (r["id"], r["frames"]), model_name)
        result["model_risks"][model_name] = {}
        for budget in budgets:
            key = str(budget)
            result["model_risks"][model_name][key] = {}
            for method in METHODS:
                all_scores = risk_counts(test, cv, models, method, budget)
                original = reference["methods"][method]["all"]
                published = reference["models"][model_name][str(budget)]
                if (all_scores["accepted"] != original["gate_accepted"]
                        or all_scores["gate_wrong"] != original["gate_wrong"]
                        or all_scores["vlm_correct"] != published["base_correct"]
                        or all_scores["hybrid_correct"] != published["hybrid"][method]["correct"]):
                    raise RuntimeError("POSTHOC_ACCOUNTING_DISAGREES_WITH_PUBLISHED_BATCH34")
                scored_vlm = {r["id"]: models[(r["id"], budget)]["prediction"] == r["gold_letter"] for r in test}
                scored_hybrid = {}
                for r in test:
                    pr = cv[r["id"]]["predictions"][method]
                    letter = base.choice_for_direction(r, pr["gated_direction"]) if pr["accepted"] else None
                    scored_hybrid[r["id"]] = (letter if pr["accepted"] else models[(r["id"], budget)]["prediction"]) == r["gold_letter"]
                ci = base.bootstrap_scene(scored_vlm, scored_hybrid, pair_ids,
                                          seed=study_cfg["bootstrap_seed"] + budget + len(method))
                per_mode = {
                    mode: risk_counts([r for r in test if r["variant"] == mode],
                                      cv, models, method, budget)
                    for mode in sorted({r["variant"] for r in test})
                }
                result["model_risks"][model_name][key][method] = {
                    "all": all_scores, "per_mode": per_mode,
                    "paired_scene_delta": ci,
                }
    return result


def save_report(result: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    base.put(out / "aggregate_risk.json", result)
    lines = [
        "# Batch 35: frozen Batch 34 post-hoc gate risk attribution",
        "",
        "**This is a read-only analysis of an already-scored synthetic heldout.**",
        "It must NOT be used to choose/tune an algorithm, threshold, model or prompt.",
        f"Frozen source commit: `{result['source_commit']}`.",
        f"Independent source pairs: {result['independent_scene_pairs']}; videos: {result['test_clips']}.",
        "",
        "| VLM | Frames | Gate | VLM correct | Hybrid correct | Gate accepted | Gate wrong | Rescue | Harm | Net | 95% pair-cluster delta pp |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, budgets in result["model_risks"].items():
        for frames, methods in budgets.items():
            for method, metrics in methods.items():
                r = metrics["all"]
                ci = metrics["paired_scene_delta"]
                lines.append(f"| {name} | {frames} | {method} | {r['vlm_correct']}/{r['n']} | "
                             f"{r['hybrid_correct']}/{r['n']} | {r['accepted']} | {r['gate_wrong']} | "
                             f"{r['rescue']} | {r['harm']} | {r['net_change_correct']:+} | "
                             f"{ci['cluster_bootstrap_95ci_pp']} |")
    lines += [
        "", "A **rescue** is an accepted correct tool answer replacing a wrong VLM answer.",
        "A **harm** is an accepted wrong tool answer replacing a correct VLM answer.",
        "A wrong tool override of an already-wrong VLM is still counted as a gate error.",
        "Tool gate abstention keeps the VLM answer; no thresholds or decisions were changed.",
        "Per-nuisance full-denominator and selective-risk counts are in aggregate_risk.json.",
        "Synthetic camera parallax and two identical targets do not establish natural-video transfer.",
    ]
    (out / "RISK_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/parallax_gate_audit_v1.json")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    result = audit(cfg)
    target = ROOT / cfg["output_dir"]
    save_report(result, target)
    print("POSTHOC_GATE_RISK_AUDIT_COMPLETE", target / "RISK_AUDIT.md", flush=True)


if __name__ == "__main__":
    main()
