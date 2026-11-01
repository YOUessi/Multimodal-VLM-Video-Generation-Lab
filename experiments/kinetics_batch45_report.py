"""Batch45 report builder. Aggregates sealed results; never opens private oracle files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/tapvid_kinetics_batch45c/batch45c_frozen"
OUT = ROOT / "outputs/kinetics_batch45c"
REPORT = OUT / "BATCH45C_KINETICS_FRESH_HOLDOUT_REPORT.md"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_pin(path: Path) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError("BATCH45_REPORT_MISSING_PIN:" + path.name)
    if sidecar.read_text(encoding="utf-8").strip() != sha_file(path):
        raise RuntimeError("BATCH45_REPORT_PIN_DRIFT:" + path.name)


def pct(value):
    return "—" if value is None else f"{100.0 * float(value):.1f}%"


def num(value, digits=3):
    return "—" if value is None else f"{float(value):.{digits}f}"


def verify_inputs() -> dict:
    manifest = read(DATA / "manifest.json")
    selection = read(ROOT / "data/tapvid_kinetics_batch45c/preflight/selection.json")
    dev = read(OUT / "dev_results.json")
    test = read(OUT / "test_results.json")
    selected = read(OUT / "dev_selection.json")
    gate = read(OUT / "dev_vlm_gate.json")
    dev_vlm = read(OUT / "dev_vlm_results.json")
    test_vlm = read(OUT / "test_vlm_results.json")
    for path in (
        OUT / "dev_predictions.json",
        OUT / "test_predictions.json",
        OUT / "dev_vlm_predictions.json",
        OUT / "test_vlm_predictions.json",
        OUT / "dev_selection.json",
        OUT / "dev_vlm_gate.json",
    ):
        require_pin(path)
    source_sha = dev.get("source_commit")
    if not source_sha or any(x.get("source_commit") != source_sha for x in (
        test, selected, gate, dev_vlm, test_vlm
    )):
        raise RuntimeError("BATCH45_REPORT_SOURCE_SHA_MISMATCH")
    if (dev.get("split") != "dev" or test.get("split") != "test"
            or selected.get("split") != "dev" or gate.get("split") != "dev"):
        raise RuntimeError("BATCH45_REPORT_SPLIT_MISMATCH")
    if (dev.get("manifest_sha256") != test.get("manifest_sha256")
            or manifest.get("selection_sha256") != sha_file(ROOT / "data/tapvid_kinetics_batch45/preflight/selection.json")):
        raise RuntimeError("BATCH45_REPORT_MANIFEST_MISMATCH")
    if len(selection["videos"]) != 12 or len({x["video_id"] for x in selection["videos"]}) != 12:
        raise RuntimeError("BATCH45_REPORT_SELECTION_COUNT")
    if len({x["label"] for x in selection["videos"]}) != 12:
        raise RuntimeError("BATCH45_REPORT_CLASS_OVERLAP")
    return {
        "manifest": manifest,
        "selection": selection,
        "dev": dev,
        "test": test,
        "selected": selected,
        "gate": gate,
        "dev_vlm": dev_vlm,
        "test_vlm": test_vlm,
        "source_sha": source_sha,
    }


def build() -> str:
    data = verify_inputs()
    manifest = data["manifest"]
    selection = data["selection"]
    dev = data["dev"]
    test = data["test"]
    selected = data["selected"]
    gate = data["gate"]
    dev_vlm = data["dev_vlm"]
    test_vlm = data["test_vlm"]
    methods = list(dev["methods"])
    selected_method = selected["selected_method"]
    lines = [
        "# Batch45C — fresh Kinetics action-class-disjoint tracking and VLM visibility",
        "",
        f"- Frozen source SHA: {data['source_sha']}",
        f"- Dataset: TAP-Vid-Kinetics / Kinetics-700-2020 validation; {len(selection['videos'])} new YouTube video IDs from {len({x['label'] for x in selection['videos']})} action classes, all disjoint from Batch42.",
        f"- Split: {manifest['dev_videos']} DEV / {manifest['test_videos']} TEST; {manifest['queries_per_video']} first-visible point queries per scene.",
        f"- Frozen manifest SHA-256: {sha_file(DATA / 'manifest.json')}",
        f"- Public query SHA-256: {sha_file(DATA / 'queries-public.json')}",
        f"- TEST CPU prediction SHA-256: {sha_file(OUT / 'test_predictions.json')}",
        f"- VLM DEV gate: **{gate['status']}**",
        "",
        "## DEV-only tracker selection",
        "",
        f"Selected method: **{selected_method}**. Eligibility required visible coverage ≥ {selected['rules']['min_visible_coverage']:.2f} and invisible false-accept rate ≤ {selected['rules']['max_invisible_false_accept_rate']:.2f}; the objective was scene-macro Jaccard averaged over 1/2/4/8/16 px. TEST was not used for selection.",
        "",
        "| Method | PCK@8 visible | Coverage | Accepted precision@8 | Invisible false accept | Jaccard mean | CPU P50 ms/frame | CPU P95 ms/frame |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        m = dev["scene_macro"][method]
        lines.append(
            f"| {method} | {pct(m['pck8_visible_unconditional'])} | {pct(m['visible_coverage'])} | "
            f"{pct(m['accepted_precision8'])} | {pct(m['invisible_false_accept_rate'])} | "
            f"{num(m['jaccard_mean_1_2_4_8_16'])} | {num(m['cpu_ms_per_target_frame_p50'],4)} | "
            f"{num(m['cpu_ms_per_target_frame_p95'],4)} |"
        )
    lines += [
        "",
        "## Independent TEST — one-time sealed score",
        "",
        "| Method | PCK@8 visible | Coverage | Accepted precision@8 | Invisible false accept | Jaccard mean | CPU P50 ms/frame | CPU P95 ms/frame |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        m = test["scene_macro"][method]
        lines.append(
            f"| {method} | {pct(m['pck8_visible_unconditional'])} | {pct(m['visible_coverage'])} | "
            f"{pct(m['accepted_precision8'])} | {pct(m['invisible_false_accept_rate'])} | "
            f"{num(m['jaccard_mean_1_2_4_8_16'])} | {num(m['cpu_ms_per_target_frame_p50'],4)} | "
            f"{num(m['cpu_ms_per_target_frame_p95'],4)} |"
        )
    chosen = test["scene_macro"][selected_method]
    lines += [
        "",
        f"Frozen DEV-selected method '{selected_method}' on TEST: PCK@8 {pct(chosen['pck8_visible_unconditional'])}, visible coverage {pct(chosen['visible_coverage'])}, accepted precision {pct(chosen['accepted_precision8'])}, invisible false accept {pct(chosen['invisible_false_accept_rate'])}, scene-macro Jaccard {num(chosen['jaccard_mean_1_2_4_8_16'])}.",
        "",
        "## Image-plane versus background-relative diagnostics",
        "",
        "The affine transform is estimated from RGB background features only. The residual below is a 2-D dominant-background affine proxy, not calibrated camera pose, depth-aware parallax or 3-D world motion.",
        "",
        "| TEST scene | Mean luma | Luma std | Laplacian variance | ORB keypoints | Affine acceptance | Screen displacement mean px | Background-relative residual mean px |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scene in sorted(test["scene_pixel_diagnostics"]):
        q = test["scene_pixel_diagnostics"][scene]
        b = test["background_relative_diagnostics"][scene]
        lines.append(
            f"| {scene} | {num(q['mean_luma'],1)} | {num(q['luma_std'],1)} | "
            f"{num(q['laplacian_variance_mean'],1)} | {num(q['orb_keypoints_median'],1)} | "
            f"{pct(q['affine_pair_acceptance'])} | {num(b['screen_displacement_mean_px'],2)} | "
            f"{num(b['background_relative_residual_mean_px'],2)} |"
        )
    lines += [
        "",
        "## SmolVLM2-2.2B visibility baseline",
        "",
        f"DEV gate: **{gate['status']}**; checks: {json.dumps(gate['checks'], sort_keys=True)}.",
    ]
    if dev_vlm.get("status") == "SCORED":
        m = dev_vlm["vlm"]
        lines.append(
            f"DEV VLM: {m['correct']}/{m['count']} correct, balanced accuracy {pct(m['balanced_accuracy'])}, "
            f"coverage {pct(m['coverage'])}, non-visible false accept {pct(m['not_visible_false_accept_rate'])}, "
            f"invalid format {pct(m['invalid_rate'])}."
        )
    else:
        lines.append(f"DEV VLM result status: {dev_vlm.get('status')}.")
    if test_vlm.get("status") == "SCORED":
        m = test_vlm["vlm"]
        lines.append(
            f"TEST VLM: {m['correct']}/{m['count']} correct, balanced accuracy {pct(m['balanced_accuracy'])}, "
            f"coverage {pct(m['coverage'])}, non-visible false accept {pct(m['not_visible_false_accept_rate'])}."
        )
    else:
        lines.append(
            f"TEST VLM inference/scoring status: **{test_vlm.get('status')}**. "
            f"TEST oracle read: {test_vlm.get('test_oracle_read', 'performed only if scored')}."
        )
    lines += [
        "",
        "## Interpretation and limitations",
        "",
        "- The 12 clips are fresh video IDs and action classes relative to Batch42, but they remain within the TAP-Vid-Kinetics/Kinetics-700-2020 source family. This is an independent within-family holdout, not an external dataset-family generalization claim.",
        "- Kinetics clips are natural YouTube footage; each source video's creator/platform terms remain applicable. Raw clips, extracted frames, private point tracks, predictions and model weights remain local on Tang.",
        "- CPU trackers process every intermediate 25-fps frame. The VLM receives a marked query frame, a local reference crop and one later frame; the temporal evidence budgets differ.",
        "- The affine compensation is only a 2-D dominant-background proxy. No calibrated camera pose, depth or 3-D world-motion claim is made.",
        "- Coverage and abstention remain first-class metrics. A high PCK obtained by accepting almost nothing, or a low false-accept rate obtained by abstaining on nearly everything, is not a complete success.",
        "- Only DEV may select tracker thresholds/methods or admit the VLM. TEST predictions are SHA-pinned before the one-time TEST scoring pass; no TEST result may be used to revise this batch.",
        "",
        "## Reproducibility",
        "",
        f"- DEV CPU prediction SHA-256: {sha_file(OUT / 'dev_predictions.json')}",
        f"- TEST CPU prediction SHA-256: {sha_file(OUT / 'test_predictions.json')}",
        f"- DEV VLM prediction SHA-256: {sha_file(OUT / 'dev_vlm_predictions.json')}",
        f"- TEST VLM prediction SHA-256: {sha_file(OUT / 'test_vlm_predictions.json')}",
        "- All private data and media are local; only code, tests, configuration and this aggregated report are published.",
        "",
    ]
    text = "\n".join(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")
    REPORT.with_suffix(REPORT.suffix + ".sha256").write_text(sha_file(REPORT) + "\n", encoding="utf-8")
    print("BATCH45_REPORT_BUILT", len(text), sha_file(REPORT), flush=True)
    return text


def verify() -> None:
    data = verify_inputs()
    sidecar = REPORT.with_suffix(REPORT.suffix + ".sha256")
    if not REPORT.is_file() or not sidecar.is_file() or sidecar.read_text().strip() != sha_file(REPORT):
        raise RuntimeError("BATCH45_REPORT_PIN_INVALID")
    text = REPORT.read_text(encoding="utf-8")
    if data["source_sha"] not in text or data["selected"]["selected_method"] not in text:
        raise RuntimeError("BATCH45_REPORT_CONTENT_MISMATCH")
    print("BATCH45_REPORT_VERIFY_PASS", sha_file(REPORT), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "verify"))
    args = parser.parse_args()
    if args.action == "build":
        build()
    else:
        verify()


if __name__ == "__main__":
    main()
