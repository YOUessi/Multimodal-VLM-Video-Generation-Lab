"""Aggregate, pin and verify Batch46 independent transfer replication.

This module executes after both sealed predictions exist and TEST scoring is done.
It never selects model parameters from TEST.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import kinetics_batch46_data as fresh
import kinetics_batch46_eval as evaluation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/kinetics_batch46"
REPORT = OUT / "BATCH46_INDEPENDENT_TRANSFER_REPORT.md"
METHOD = evaluation.FROZEN_METHOD
COLUMNS = (
    "pck8_visible_unconditional", "visible_coverage", "accepted_precision8",
    "invisible_false_accept_rate", "jaccard_mean_1_2_4_8_16",
    "cpu_ms_per_target_frame_p50",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def require_pinned(name: str) -> str:
    path = OUT / name
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file() or sidecar.read_text().strip() != digest(path):
        raise RuntimeError("BATCH46_OUTPUT_PIN_DRIFT:" + name)
    return digest(path)


def validate_inputs() -> dict:
    manifest = fresh.base.audit_full()
    fresh.audit_historical_disjointness()
    if len(manifest.get("scenes", [])) != 12:
        raise RuntimeError("BATCH46_SCENE_COUNT")
    selected = read("dev_selection.json")
    if (selected.get("selected_method") != METHOD
            or selected.get("selection_source") != "PREVIOUS_BATCH45E_DEV_ONLY_FROZEN"
            or selected.get("previous_selection_sha256") != evaluation.PRIOR_SELECTION_SHA256):
        raise RuntimeError("BATCH46_SELECTION_DRIFT")
    evaluation.verify_prior_selection()
    pins = {name: require_pinned(name) for name in (
        "dev_predictions.json", "test_predictions.json", "dev_selection.json",
        "dev_vlm_predictions.json", "test_vlm_predictions.json",
        "dev_vlm_gate.json",
    )}
    dev, test = read("dev_results.json"), read("test_results.json")
    dev_vlm, test_vlm, gate = (
        read("dev_vlm_results.json"),
        read("test_vlm_results.json"),
        read("dev_vlm_gate.json"),
    )
    current = evaluation.previous.gitsha()
    for key, item in (("DEV", dev), ("TEST", test),
                      ("VLM_DEV", dev_vlm), ("VLM_TEST", test_vlm),
                      ("GATE", gate), ("SELECTION", selected)):
        if item.get("source_commit") != current:
            raise RuntimeError("BATCH46_WRONG_SOURCE:" + key)
    for split, result in (("dev", dev), ("test", test)):
        if result.get("split") != split or result.get("prediction_sha256") != pins[split + "_predictions.json"]:
            raise RuntimeError("BATCH46_SCORE_SOURCE_OR_PIN_DRIFT:" + split)
    if gate["status"] != "PASS":
        if (test_vlm.get("status") != "SKIPPED_DEV_GATE_BLOCKED"
                or test_vlm.get("test_oracle_read") is not False):
            raise RuntimeError("BATCH46_VLM_BLOCKED_TEST_LEAK")
    return dict(manifest=manifest, selected=selected, pins=pins, dev=dev, test=test,
                gate=gate, dev_vlm=dev_vlm, test_vlm=test_vlm, sha=current)


def fmt(key: str, value):
    if value is None:
        return "—"
    if key.startswith(("pck", "visible_", "accepted_", "invisible_")):
        return f"{100 * value:.1f}%"
    if "jaccard" in key:
        return f"{value:.3f}"
    return f"{value:.3f}"


def build() -> None:
    data = validate_inputs()
    lines = [
        "# Batch46 — independent natural-video replication of frozen Batch45E tracker",
        "",
        f"- Frozen source SHA: `{data['sha']}`",
        "- Source: official TAP-Vid-Kinetics / Kinetics-700 validation footage.",
        "- 12 newly selected video IDs and 12 action classes, disjoint from Batch42 AND Batch45E.",
        "- 6 DEV + 6 sealed TEST, 4 official first-visible point queries per video.",
        f"- Prior DEV-selected method, held fixed before this batch: **{METHOD}**.",
        f"- Prior selection SHA256: `{evaluation.PRIOR_SELECTION_SHA256}`.",
        f"- New manifest SHA256: `{digest(fresh.FROZEN / 'manifest.json')}`.",
        f"- New TEST prediction SHA256: `{data['pins']['test_predictions.json']}`.",
        "",
        "## DEV diagnostic (not used to change method or thresholds)",
        "",
        "| Method | PCK@8 | Coverage | Accept precision | Occluded false accept | Jaccard | CPU P50 ms / target frame |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for section in ("dev", "test"):
        result = data[section]
        if section == "test":
            lines += [
                "",
                "## One-time frozen TEST (no parameter selection from these labels)",
                "",
                "| Method | PCK@8 | Coverage | Accept precision | Occluded false accept | Jaccard | CPU P50 ms / target frame |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        for method in result["methods"]:
            m = result["scene_macro"][method]
            lines.append("| " + ("**" + method + "**" if method == METHOD else method) +
                         " | " + " | ".join(fmt(key, m[key]) for key in COLUMNS) + " |")
    primary = data["test"]["scene_macro"][METHOD]
    lines += [
        "",
        "## Pre-registered transfer result",
        "",
        f"- Scene-macro Jaccard: {fmt('jaccard_mean_1_2_4_8_16', primary['jaccard_mean_1_2_4_8_16'])}.",
        f"- PCK@8 visible unconditional: {fmt('pck8_visible_unconditional', primary['pck8_visible_unconditional'])}.",
        f"- Visible coverage: {fmt('visible_coverage', primary['visible_coverage'])}.",
        f"- Invisible false accept: {fmt('invisible_false_accept_rate', primary['invisible_false_accept_rate'])}.",
        f"- CPU P50 ms/target-frame (excludes shared affine registration): {fmt('cpu_ms_per_target_frame_p50', primary['cpu_ms_per_target_frame_p50'])}.",
        "",
        "## VLM admission",
        "",
        f"- DEV gate: **{data['gate']['status']}**.",
        f"- DEV checks: `{json.dumps(data['gate'].get('checks', {}), sort_keys=True)}`.",
        f"- TEST status: **{data['test_vlm'].get('status')}**.",
        "- This 3-image VLM has a different temporal evidence budget than dense-frame CPU trackers.",
        "",
        "## Interpretation",
        "",
        "- These are natural YouTube video frames, not controlled photo capture or calibrated 3D scenes.",
        "- All class and video exclusion uses historical metadata, never historical TEST scores.",
        "- The selected method and all CPU/VLM hyperparameters were frozen before the new TEST labels were opened.",
        "- Screen-space point motion and the dominant-background 2D affine residual are separate; affine residual is NOT a physical camera-pose/3D-motion estimate.",
        "- False-accept and coverage must be considered jointly. CPU time is not end-to-end latency including decoding and affine precomputation.",
        "- Raw videos, frames, weights, predictions and private oracle stay local. Only aggregated results are publishable.",
        "",
    ]
    text = "\n".join(lines)
    if REPORT.exists():
        if REPORT.read_text(encoding="utf-8") != text:
            raise RuntimeError("BATCH46_REPORT_ALREADY_EXISTS_WITH_DIFFERENT_BYTES")
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        with REPORT.open("x", encoding="utf-8") as f:
            f.write(text)
    sidecar = REPORT.with_suffix(REPORT.suffix + ".sha256")
    expected = digest(REPORT) + "\n"
    if sidecar.exists():
        if sidecar.read_text(encoding="utf-8") != expected:
            raise RuntimeError("BATCH46_REPORT_PIN_DRIFT")
    else:
        with sidecar.open("x", encoding="utf-8") as f:
            f.write(expected)
    print("BATCH46_REPORT_BUILT", len(text), digest(REPORT), flush=True)


def verify() -> None:
    data = validate_inputs()
    if not REPORT.is_file() or REPORT.with_suffix(REPORT.suffix + ".sha256").read_text().strip() != digest(REPORT):
        raise RuntimeError("BATCH46_REPORT_NOT_PINNED")
    if data["sha"] not in REPORT.read_text() or METHOD not in REPORT.read_text():
        raise RuntimeError("BATCH46_REPORT_CONTENT")
    print("BATCH46_REPORT_VERIFIED", digest(REPORT), flush=True)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("build", "verify"))
    args = p.parse_args()
    if args.action == "build":
        build()
    else:
        verify()


if __name__ == "__main__":
    main()
