"""Batch40 result report and final frozen-source audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import davis_real_data as raw
import davis_occlusion_eval as tracker
import davis_real_eval as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/davis_occlusion_v3"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value):
    return "n/a" if value is None else f"{100 * float(value):.2f}%"


def render_report(dev: dict, test: dict, dev_vlm: dict, test_vlm: dict,
                  selection: dict, gate: dict) -> str:
    lines = [
        "# Batch 40 — DAVIS scene-disjoint appearance consensus and VLM visibility",
        "",
        f"- Frozen source commit: {test['source_commit']}",
        f"- DEV / TEST scenes: {dev['scenes']} / {test['scenes']}",
        f"- Frames per scene: {test['frames_per_scene']}",
        f"- Dataset manifest SHA-256: {test['source_manifest_sha256']}",
        f"- DEV-selected method: {selection['selected_method']} (fallback={selection['fallback_used']})",
        f"- VLM DEV interface gate: {gate['status']} ({gate['query_count']} queries; invalid={gate['invalid_rate']:.3f}; coverage={gate['coverage']:.3f})",
        f"- VLM model weights SHA-256: {test_vlm['model_weights_sha256']}",
        "",
        "## CPU tracking and abstention",
        "",
        "| Split | Method | Visible PCK@8 | Visible coverage | Accepted precision@8 | Invisible false accepts | Invisible evaluations | Mask-shrink false accepts (proxy only) | CPU P50/P95 ms/scene |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, result in (("DEV", dev), ("TEST", test)):
        for method, m in result["methods"].items():
            lines.append(
                f"| {split} | {method} | {pct(m['visible_pck8_unconditional'])} | "
                f"{pct(m['visible_accepted'] / m['visible_evaluations'] if m['visible_evaluations'] else None)} | "
                f"{pct(m['accepted_precision_at_8px'])} | {m['absent_false_accepted']} | "
                f"{m['invisible_evaluations']} | {m['mask_area_shrink_wrong_accepted']}/"
                f"{m['mask_area_shrink_positions_not_verified_occlusions']} | "
                f"{m['cpu_scene_p50_ms']}/{m['cpu_scene_p95_ms']} |"
            )
    lines += [
        "",
        "## DEV-only appearance threshold selection",
        "",
        "| Candidate | Visible PCK@8 | Visible coverage | Invisible false-accept rate | Eligible |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, m in selection["candidate_diagnostics"].items():
        eligible = method in selection["eligible_methods"]
        lines.append(
            f"| {method} | {pct(m['visible_pck8_unconditional'])} | "
            f"{pct(m['visible_coverage'])} | {pct(m['absent_false_accept_rate'])} | "
            f"{'yes' if eligible else 'no'} |"
        )
    lines += [
        "",
        "## SmolVLM2-500M two-image visibility classification",
        "",
        "| Split | Visible recall | Absent recall | Balanced accuracy (abstentions count wrong) | Coverage | Selective accuracy | Absent false-accept rate | Uncertain / invalid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, result in (("DEV", dev_vlm), ("TEST", test_vlm)):
        lines.append(
            f"| {split} | {pct(result['visible_recall'])} | {pct(result['absent_recall'])} | "
            f"{pct(result['balanced_accuracy_unconditional'])} | {pct(result['coverage'])} | "
            f"{pct(result['selective_accuracy'])} | {pct(result['absent_false_accept_rate'])} | "
            f"{result['uncertain']} / {result['invalid']} |"
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- DAVIS mask centroid is an image-plane proxy, not a material point or world-relative trajectory.",
        "- Empty future masks indicate the annotated instance is absent from the frame; this does not distinguish full occlusion from leaving the field of view.",
        "- Mask-area shrink is reported only as a proxy, not as verified occlusion.",
        "- The background affine is a 2-D diagnostic, not calibrated 6-DoF camera motion or depth-aware parallax compensation.",
        "- CPU trackers use all intermediate RGB frames; VLM sees only the initial highlighted mask image and four fixed later frames.",
        "- VLM visibility is a separate classifier, not a point-coordinate predictor or a tracker.",
        "- DEV selection is fixed by the predeclared rule. TEST labels are used once for final scoring only.",
        "- This is a new scene-disjoint draw from the DAVIS 2017 corpus, not a new dataset or a pretraining-novelty claim.",
        "",
    ]
    return "\n".join(lines)


def write_report() -> Path:
    dev = load(OUT / "dev_results.json")
    test = load(OUT / "test_results.json")
    dev_vlm = load(OUT / "vlm_visibility_dev_results.json")
    test_vlm = load(OUT / "vlm_visibility_test_results.json")
    selection = load(OUT / "dev_selection.json")
    gate = load(OUT / "vlm_visibility_dev_gate.json")
    path = OUT / "BATCH40_RESULTS.md"
    raw.save_if_frozen(path, render_report(dev, test, dev_vlm, test_vlm, selection, gate).encode())
    print("BATCH40_REPORT_WRITTEN", path, raw.sha256(path.read_bytes()), flush=True)
    return path


def final_audit(cfg: dict) -> None:
    dataset = ROOT / cfg["dataset_dir"]
    raw.check(cfg, include_future_masks_and_gold=True)
    expected_sha = tracker.current_sha()
    for split in ("dev", "test"):
        for name in (f"{split}_predictions.json", f"{split}_results.json",
                     f"vlm_visibility_{split}_predictions.json",
                     f"vlm_visibility_{split}_results.json"):
            path = OUT / name
            if not path.is_file():
                raise RuntimeError("BATCH40_FINAL_FILE_MISSING:" + name)
        for name in (f"{split}_predictions.json", f"vlm_visibility_{split}_predictions.json"):
            base.require_pinned_prediction(OUT / name)
        for name in (f"{split}_results.json", f"vlm_visibility_{split}_results.json"):
            if load(OUT / name).get("source_commit") != expected_sha:
                raise RuntimeError("BATCH40_RESULT_SOURCE_SHA_DRIFT:" + name)
    if load(OUT / "dev_selection.json").get("source_commit") != expected_sha:
        raise RuntimeError("BATCH40_DEV_SELECTION_SOURCE_SHA_DRIFT")
    if load(OUT / "vlm_visibility_dev_gate.json").get("source_commit") != expected_sha:
        raise RuntimeError("BATCH40_VLM_GATE_SOURCE_SHA_DRIFT")
    if not (OUT / "BATCH40_RESULTS.md").is_file():
        raise RuntimeError("BATCH40_REPORT_MISSING")
    print("BATCH40_FINAL_AUDIT_PASS", expected_sha, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("report", "final-audit"))
    parser.add_argument("--config", default="configs/davis_occlusion_v3.json")
    args = parser.parse_args()
    cfg = load(ROOT / args.config)
    if args.action == "report":
        write_report()
    else:
        final_audit(cfg)


if __name__ == "__main__":
    main()
