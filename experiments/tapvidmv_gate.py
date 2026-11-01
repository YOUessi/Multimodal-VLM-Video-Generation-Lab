"""Batch 39C predeclared DEV interface gate and sealed result report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import tapvidmv_data as src
import tapvidmv_eval as ev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/tapvidmv_droid_batch39c"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dev_gate(result: dict, *, expected_queries: int = 12,
             max_invalid_rate: float = 0.20) -> dict:
    """Gate only output-interface validity; never select on TEST accuracy."""
    if result.get("split") != "dev" or int(result.get("query_count", -1)) != expected_queries:
        raise RuntimeError("TAPVIDMV_DEV_GATE_QUERY_SET_MISMATCH")
    metrics = result["metrics"]["vlm_500m_endpoint"]
    invalid = int(metrics["format_invalid"])
    invalid_rate = invalid / expected_queries
    valid = expected_queries - invalid
    if invalid_rate > max_invalid_rate or valid < expected_queries - int(expected_queries * max_invalid_rate):
        raise RuntimeError(
            f"TAPVIDMV_DEV_GATE_FORMAT_FAILURE:invalid={invalid}/{expected_queries}"
        )
    return {
        "status": "passed",
        "split": "dev",
        "query_count": expected_queries,
        "format_invalid": invalid,
        "format_valid": valid,
        "format_invalid_rate": round(invalid_rate, 6),
        "max_invalid_rate": max_invalid_rate,
        "selection_basis": "predeclared output-format/interface threshold only; no accuracy-based TEST tuning",
    }


def pct(value):
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def ms(value):
    return "n/a" if value is None else f"{float(value):.3f}"


def render_report(dev_cpu: dict, test_cpu: dict, dev_vlm: dict,
                  test_vlm: dict, gate: dict) -> str:
    lines = [
        "# Batch 39C — TAPVid-MV DROID multi-view motion decomposition",
        "",
        f"- Frozen source commit: `{test_cpu['source_commit']}`",
        f"- DEV scenes / CPU queries: {dev_cpu['scene_count']} / {dev_cpu['query_count']}",
        f"- TEST scenes / CPU queries: {test_cpu['scene_count']} / {test_cpu['query_count']}",
        f"- DEV VLM queries: {dev_vlm['query_count']}; TEST VLM queries: {test_vlm['query_count']}",
        f"- DEV format gate: {gate['status']} ({gate['format_valid']}/{gate['query_count']} valid)",
        f"- Manifest SHA-256: `{test_cpu['manifest_sha256']}`",
        f"- Manifest errata SHA-256: `{test_cpu['manifest_errata_sha256']}`",
        f"- SmolVLM2-500M weights SHA-256: `{test_vlm['model_weights_sha256']}`",
        "",
        "## CPU point-tracking results",
        "",
        "| Split | Method | PCK@8 all in-frame | PCK@8 depth-visible | Visible coverage | Occluded false accept | Out-of-frame false accept | CPU P50/P95 ms per target frame |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, result in (("DEV", dev_cpu), ("TEST", test_cpu)):
        for method, m in result["methods"].items():
            lines.append(
                f"| {split} | {method} | {pct(m['pck8_all_inframe'])} | "
                f"{pct(m['pck8_depth_visible'])} | {pct(m['coverage_depth_visible'])} | "
                f"{pct(m['false_accept_occluded'])} | {pct(m['false_accept_out'])} | "
                f"{ms(m['cpu_ms_per_target_frame_p50'])}/{ms(m['cpu_ms_per_target_frame_p95'])} |"
            )
    lines += [
        "",
        "## SmolVLM2-500M endpoint results",
        "",
        "| Split | PCK@8 depth-visible | Visible coverage | Visible precision@8 | Occluded false accept | Out false accept | Format invalid |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, result in (("DEV", dev_vlm), ("TEST", test_vlm)):
        m = result["metrics"]["vlm_500m_endpoint"]
        lines.append(
            f"| {split} | {pct(m['pck8_depth_visible'])} | {pct(m['coverage_depth_visible'])} | "
            f"{pct(m['precision8_depth_visible'])} | {pct(m['false_accept_occluded'])} | "
            f"{pct(m['false_accept_out'])} | {m['format_invalid']}/{result['query_count']} |"
        )
    lines += [
        "",
        "## Camera-motion decomposition",
        "",
        "| Split | Screen vs world-only cosine | Affine-compensated vs world-only cosine | Oracle camera-subtracted vs world-only cosine | Camera-motion dominance | ORB affine reprojection error P50/P90 px | Affine coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, result in (("DEV", dev_cpu), ("TEST", test_cpu)):
        m = result["camera_motion_decomposition"]
        lines.append(
            f"| {split} | {m['screen_vs_world_only_cosine']['mean']} | "
            f"{m['image_affine_compensated_vs_world_only_cosine']['mean']} | "
            f"{m['oracle_camera_subtracted_vs_world_only_cosine']['mean']} | "
            f"{m['camera_motion_dominance_ratio']['mean']} | "
            f"{m['orb_affine_static_point_reprojection_error_px']['p50']}/"
            f"{m['orb_affine_static_point_reprojection_error_px']['p90']} | "
            f"{pct(m['orb_affine_coverage'])} |"
        )
    caveats = list(dict.fromkeys(
        dev_cpu.get("caveats", []) + test_cpu.get("caveats", [])
        + dev_vlm.get("limitations", []) + test_vlm.get("limitations", [])
    ))
    lines += ["", "## Interpretation limits", ""]
    lines.extend(f"- {item}" for item in caveats)
    lines += [
        "- This is a study of the public Rerun viewer subset, not the full TAPVid-MV track arrays.",
        "- Visibility is inferred from exact released depth point-cloud snapshots; it is not an official per-frame per-view binary occlusion label.",
        "- The DEV gate evaluates only parser/interface validity. No TEST metric is used for threshold, prompt, or model selection.",
        "- ORB partial-affine is an image-background diagnostic; it is not a calibrated physical camera-motion estimator and is not a point-identity tracker.",
        "",
    ]
    return "\n".join(lines)


def write_report() -> Path:
    dev_cpu = read_json(OUT / "dev_results.json")
    test_cpu = read_json(OUT / "test_results.json")
    dev_vlm = read_json(OUT / "dev_vlm_results.json")
    test_vlm = read_json(OUT / "test_vlm_results.json")
    gate = read_json(OUT / "DEV_GATE_PASSED.json")
    report = render_report(dev_cpu, test_cpu, dev_vlm, test_vlm, gate)
    path = OUT / "BATCH39C_RESULTS.md"
    src.write_once(path, report.encode("utf-8"))
    print("TAPVIDMV_REPORT_WRITTEN", path, ev.sha_bytes(path.read_bytes()), flush=True)
    return path


def final_audit() -> dict:
    src.audit()
    for split in ("dev", "test"):
        for name in (f"{split}_predictions.json", f"{split}_vlm_predictions.json"):
            ev.require_pin(OUT / name)
        for name in (f"{split}_results.json", f"{split}_vlm_results.json"):
            if not (OUT / name).is_file():
                raise RuntimeError("TAPVIDMV_FINAL_RESULT_MISSING:" + name)
    if not (OUT / "DEV_GATE_PASSED.json").is_file() or not (OUT / "BATCH39C_RESULTS.md").is_file():
        raise RuntimeError("TAPVIDMV_FINAL_REPORT_OR_DEV_GATE_MISSING")
    expected_sha = ev.gitsha()
    for split in ("dev", "test"):
        for name in (f"{split}_predictions.json", f"{split}_vlm_predictions.json",
                     f"{split}_results.json", f"{split}_vlm_results.json"):
            doc = read_json(OUT / name)
            if doc.get("source_commit") != expected_sha:
                raise RuntimeError("TAPVIDMV_FINAL_SOURCE_SHA_MISMATCH:" + name)
    print("TAPVIDMV_BATCH39C_FINAL_AUDIT_PASS", expected_sha, flush=True)
    return {"status": "passed", "source_commit": expected_sha}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("dev-gate", "report", "final-audit"))
    args = parser.parse_args()
    if args.action == "dev-gate":
        result = read_json(OUT / "dev_vlm_results.json")
        gate = dev_gate(result)
        src.write_once(OUT / "DEV_GATE_PASSED.json",
                       (json.dumps(gate, indent=2) + "\n").encode())
        print("TAPVIDMV_DEV_GATE_PASS", json.dumps(gate, sort_keys=True), flush=True)
    elif args.action == "report":
        write_report()
    else:
        final_audit()


if __name__ == "__main__":
    main()
