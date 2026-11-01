"""Build a local-only Markdown report from sealed Batch44D DEV/TEST result artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import tapvidmv_batch44d_data as src

REPAIR_CONFIG = json.loads(
    (ROOT / "configs/batch44e_report_repair.json").read_text(encoding="utf-8")
)
OUT = ROOT / REPAIR_CONFIG["output_dir"]
REPORT = ROOT / REPAIR_CONFIG["report_path"]
EXPECTED_BATCH44D_SOURCE_SHA = REPAIR_CONFIG["batch44d_source_sha"]
PINNED_RESULT_JSONS = (
    "dev_results.json", "test_results.json", "dev_selection.json",
    "dev_vlm_results.json", "dev_vlm_gate.json", "test_vlm_results.json",
)
PINNED_PREDICTION_JSONS = (
    "dev_predictions.json", "test_predictions.json",
    "dev_vlm_predictions.json", "test_vlm_predictions.json",
)


def require_pin(path: Path) -> str:
    pin_path = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not pin_path.is_file():
        raise RuntimeError("BATCH44D_REPORT_INPUT_PIN_MISSING:" + path.name)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if pin_path.read_text(encoding="utf-8").strip() != digest:
        raise RuntimeError("BATCH44D_REPORT_INPUT_PIN_DRIFT:" + path.name)
    return digest


def load(name: str) -> dict:
    path = OUT / name
    require_pin(path)
    return json.loads(path.read_text(encoding="utf-8"))


def require_source_commit(payload: dict, artifact: str) -> None:
    if payload.get("source_commit") != EXPECTED_BATCH44D_SOURCE_SHA:
        raise RuntimeError("BATCH44D_REPORT_SOURCE_COMMIT_MISMATCH:" + artifact)


def verify_inputs() -> dict:
    """Verify existing Batch44D artifacts only; never open or rescore private TEST labels."""
    manifest = src.audit_public()
    payloads = {name: load(name) for name in PINNED_RESULT_JSONS}
    predictions = {name: load(name) for name in PINNED_PREDICTION_JSONS}
    for name, payload in {**payloads, **predictions}.items():
        require_source_commit(payload, name)
    dev = payloads["dev_results.json"]
    test = payloads["test_results.json"]
    selection = payloads["dev_selection.json"]
    gate = payloads["dev_vlm_gate.json"]
    test_vlm = payloads["test_vlm_results.json"]
    if dev.get("split") != "dev" or test.get("split") != "test":
        raise RuntimeError("BATCH44D_REPORT_SPLIT_MISMATCH")
    if selection.get("selection_source") != "DEV only" or selection.get("split") != "dev":
        raise RuntimeError("BATCH44D_REPORT_SELECTION_NOT_DEV_ONLY")
    if gate.get("status") != "BLOCK" or test_vlm.get("status") != "SKIPPED_DEV_GATE_BLOCKED":
        raise RuntimeError("BATCH44D_REPORT_VLM_GATE_CONTRACT_DRIFT")
    manifest_sha = src.sha256_file(src.FROZEN / "manifest.json")
    config_sha = src.sha256_bytes(src.json_bytes(src.config()))
    for name, result in (("dev_results.json", dev), ("test_results.json", test)):
        if result.get("manifest_sha256") != manifest_sha or result.get("config_sha256") != config_sha:
            raise RuntimeError("BATCH44D_REPORT_DATA_CONFIG_PIN_MISMATCH:" + name)
    for name, result in (("dev_predictions.json", dev), ("test_predictions.json", test)):
        actual = hashlib.sha256((OUT / name).read_bytes()).hexdigest()
        if result.get("sealed_prediction_sha256") != actual:
            raise RuntimeError("BATCH44D_REPORT_PREDICTION_HASH_MISMATCH:" + name)
    for name, split in (("dev_predictions.json", "dev"), ("test_predictions.json", "test"),
                        ("dev_vlm_predictions.json", "dev"), ("test_vlm_predictions.json", "test")):
        if predictions[name].get("split") != split:
            raise RuntimeError("BATCH44D_REPORT_PREDICTION_SPLIT_MISMATCH:" + name)
    return {
        "manifest": manifest,
        "dev": dev,
        "test": test,
        "selection": selection,
        "dev_vlm": payloads["dev_vlm_results.json"],
        "gate": gate,
        "test_vlm": test_vlm,
        "manifest_sha256": manifest_sha,
    }


def pct(value):
    return "—" if value is None else f"{100 * float(value):.1f}%"


def number(value, digits=3):
    return "—" if value is None else f"{float(value):.{digits}f}"


def count(value):
    return f"{int(value):,}"


def metric_table(results: dict) -> str:
    rows = [
        "| Method | PCK@8 all in-frame | PCK@8 depth-visible | Visible coverage | Precision@8 accepted | Occluded false accept | Out-of-frame false accept | Jaccard scene-macro | CPU P50/P95 ms/frame |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, m in results["methods"].items():
        rows.append(
            f"| {method} | {pct(m['pck8_all_inframe'])} | {pct(m['pck8_depth_visible'])} | "
            f"{pct(m['coverage_depth_visible'])} | {pct(m['precision8_depth_visible'])} | "
            f"{pct(m['false_accept_occluded'])} | {pct(m['false_accept_out'])} | "
            f"{pct(m['jaccard_mean_scene_macro'])} | "
            f"{number(m['cpu_ms_per_target_frame_p50'], 4)} / {number(m.get('cpu_ms_per_target_frame_p95'), 4)} |"
        )
    return "\n".join(rows)


def build() -> str:
    inputs = verify_inputs()
    manifest = inputs["manifest"]
    dev = inputs["dev"]
    test = inputs["test"]
    selection = inputs["selection"]
    dev_vlm = inputs["dev_vlm"]
    gate = inputs["gate"]
    test_vlm = inputs["test_vlm"]
    dev_scenes = [x for x in manifest["scenes"] if x["split"] == "dev"]
    test_scenes = [x for x in manifest["scenes"] if x["split"] == "test"]
    total_queries = sum(x["query_count"] for x in manifest["scenes"])
    source_sha = manifest["source_index_sha256"]
    manifest_sha = inputs["manifest_sha256"]

    lines = [
        "# Batch44D — TAPVid-MV collection-group holdout with calibrated cross-view consensus",
        "",
        f"- Frozen source commit: {dev['source_commit']}",
        f"- Public source index SHA-256: {source_sha}",
        f"- Frozen manifest SHA-256: {manifest_sha}",
        f"- Scenes: {len(manifest['scenes'])} total ({len(dev_scenes)} DEV / {len(test_scenes)} TEST), sequence/collection-group disjoint",
        f"- Query instances: {total_queries} (3 shared 3-D point identities × 3 camera views per scene)",
        "- Source data: public TAPVid-MV Rerun viewer recordings for EgoExo4D and PACE; source recordings and extracted frames remain local",
        "",
        "## Protocol and label semantics",
        "",
        "- Each selected 3-D point is visible in all three camera views at the deterministic query frame: the earliest exact depth snapshot with at least three shared visible point IDs and at least 30 future frames. The same point identity is queried independently in each view.",
        "- CPU predictors see all intermediate RGB frames. The VLM baseline sees only the selected query frame with a red-cross marker and the exact target-frame image.",
        "- VISIBLE, OCCLUDED, OUT, UNKNOWN, and MISSING are kept separate. Visibility is inferred only at exact released depth-cloud snapshots; no interpolation of visibility labels is performed.",
        "- PCK@8 all in-frame measures 2-D point localization for all in-frame projections, including frames whose visibility is unknown. PCK@8 depth-visible is restricted to depth-visible snapshots.",
        "- TAP-Vid Jaccard uses visible/occluded/out labels at exact depth snapshots; unknown and missing labels are excluded.",
        "- The joint tracker reads a separately hashed public calibration file containing only camera poses, intrinsics and resolutions; 3-D point tracks, track IDs and depth clouds remain private oracle data.",
        "",
        "## DEV-only method selection",
        "",
        f"- Selection status: {selection['status']}",
        f"- Selected method: {selection['selected_method']}",
        f"- Eligible candidates: {', '.join(selection['eligible_candidates']) if selection['eligible_candidates'] else 'none'}",
        f"- Rule: {selection['reason']}",
        "",
        metric_table(dev),
        "",
        "## TEST CPU results",
        "",
        "TEST is descriptive only. All candidate methods and thresholds were frozen before TEST scoring; no TEST result was used for selection.",
        "",
        metric_table(test),
        "",
        "## TEST subset diagnostics",
        "",
        "| Source subset | Method | Visible PCK@8 scene-macro | Visible coverage scene-macro | Occluded false accept scene-macro | Jaccard scene-macro |",
        "|---|---|---:|---:|---:|---:|",
    ]
    selected = selection["selected_method"]
    for subset, methods in test["per_subset"].items():
        m = methods[selected]
        lines.append(
            f"| {subset} | {selected} | {pct(m['pck8_depth_visible_scene_macro'])} | "
            f"{pct(m['coverage_depth_visible_scene_macro'])} | "
            f"{pct(m['false_accept_occluded_scene_macro'])} | {pct(m['jaccard_mean_scene_macro'])} |"
        )
    lines += [
        "",
        "## TEST image-quality and camera-motion strata",
        "",
        "| Scene | Camera | Mean luma P10/P90 | Contrast P10/P90 | Laplacian variance P10/P90 | ORB keypoints P10/P90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    quality = test.get("scene_quality_by_camera", {})
    for scene in test_scenes:
        for camera, q in quality.get(scene["name"], {}).items():
            lines.append(
                f"| {scene['name']} | {camera} | {number(q['mean_luma']['p10'], 1)} / {number(q['mean_luma']['p90'], 1)} | "
                f"{number(q['contrast_std']['p10'], 1)} / {number(q['contrast_std']['p90'], 1)} | "
                f"{number(q['laplacian_variance']['p10'], 1)} / {number(q['laplacian_variance']['p90'], 1)} | "
                f"{number(q['orb_keypoints']['p10'], 1)} / {number(q['orb_keypoints']['p90'], 1)} |"
            )
    lines += [
        "",
        "## Cross-view 3-D reconstruction",
        "",
        "| Method | 3-view GT-visible groups | Triangulated | Coverage | World error P50 / P90 (m) | Within 5 cm | Within 10 cm | Oracle pixel-triangulation error P50 (m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, m in test["cross_view_3d"].items():
        world = m["world_error_m"]
        oracle = m["oracle_gt_pixel_triangulation_error_m"]
        lines.append(
            f"| {method} | {count(m['all_three_views_ground_truth_visible_groups'])} | "
            f"{count(m['triangulated_groups'])} | {pct(m['coverage_given_all_three_visible'])} | "
            f"{number(world['p50'])} / {number(world['p90'])} | {pct(m['within_5cm'])} | "
            f"{pct(m['within_10cm'])} | {number(oracle['p50'])} |"
        )
    motion = test["camera_motion_decomposition"]
    lines += [
        "",
        "## Camera motion versus world-relative motion",
        "",
        f"- Screen displacement vs world-only projection cosine: mean {number(motion['screen_vs_world_only_cosine']['mean'], 4)} (n={count(motion['screen_vs_world_only_cosine']['n'])})",
        f"- ORB affine-compensated residual vs world-only projection cosine: mean {number(motion['image_affine_compensated_vs_world_only_cosine']['mean'], 4)} (n={count(motion['image_affine_compensated_vs_world_only_cosine']['n'])})",
        f"- Oracle camera-subtracted residual vs world-only projection cosine: mean {number(motion['oracle_camera_subtracted_vs_world_only_cosine']['mean'], 4)} (n={count(motion['oracle_camera_subtracted_vs_world_only_cosine']['n'])})",
        f"- Camera-only screen displacement share proxy: mean {number(motion['camera_only_screen_displacement_share_proxy']['mean'], 4)}",
        f"- Net camera translation: median {number(motion['camera_translation_net_m']['p50'])} m; net rotation: median {number(motion['camera_rotation_net_deg']['p50'])}°",
        f"- ORB affine static-point reprojection error: P50 {number(motion['orb_affine_static_point_reprojection_error_px']['p50'])} px; registration coverage {pct(motion['orb_affine_coverage'])}",
        "",
        "The camera-only share is a 2-D counterfactual displacement proxy, not a physical causal fraction. Affine compensation is a screen-space diagnostic, not calibrated camera pose recovery.",
        "",
        "## SmolVLM2-500M visibility baseline",
        "",
        f"- DEV gate: {gate['status']}; checks: {json.dumps(gate['checks'], sort_keys=True)}",
        f"- DEV balanced accuracy / coverage / invalid rate: {pct(dev_vlm.get('vlm', {}).get('balanced_accuracy'))} / {pct(dev_vlm.get('vlm', {}).get('coverage'))} / {pct(dev_vlm.get('vlm', {}).get('invalid_rate'))}",
        f"- DEV false-accept rate on non-visible points: {pct(dev_vlm.get('vlm', {}).get('false_accept_nonvisible_rate'))}",
    ]
    if test_vlm.get("status") == "SKIPPED_DEV_GATE_BLOCKED":
        lines.append("- TEST VLM inference and scoring: SKIPPED because the predeclared DEV gate did not pass; TEST VLM labels were not opened.")
    else:
        metrics = test_vlm["vlm"]
        lines.append(
            f"- TEST VLM: balanced accuracy {pct(metrics['balanced_accuracy'])}, coverage {pct(metrics['coverage'])}, "
            f"non-visible false accepts {pct(metrics['false_accept_nonvisible_rate'])}."
        )
    lines += [
        "",
        "## Dataset and evaluation limitations",
        "",
        "- This is a 21-sequence exploratory subset, not a broad benchmark-wide generalization claim.",
        "- These 21 sequences were unused in Batch43 but were used for DEV-only pipeline repair in Batch44C; Batch44C did not score TEST. This is a held-out TEST within the same source groups, not a new dataset-family generalization experiment.",
        "- The Rerun viewer recordings display a subsampled point-track set; metadata-declared track counts are not the number of rendered track identities.",
        "- Per-view visibility is inferred from exact released depth point-cloud snapshots rather than a separate official binary occlusion annotation. Frames without snapshots remain UNKNOWN.",
        "- The 3-D track IDs and depth snapshots remain evaluation-only. The joint method uses public camera poses/intrinsics but no 3-D target coordinates or depth labels.",
        "- Three selected shared point identities per scene remain a small cross-view sample. Cross-view consensus coverage and 3-D reconstruction statistics have wide uncertainty and should not be interpreted as deployment accuracy.",
        "- Source video rights remain governed by the individual EgoExo4D/PACE terms. No source RRD, video, frame array, private oracle, or model weights are published in GitHub.",
        "",
        "## Reproducibility",
        "",
        f"- Source index SHA-256: {source_sha}",
        f"- Manifest SHA-256: {manifest_sha}",
        f"- DEV prediction SHA-256: {dev['sealed_prediction_sha256']}",
        f"- TEST prediction SHA-256: {test['sealed_prediction_sha256']}",
        "- DEV-only selection and VLM admission-gate artifacts are hash-pinned in the local ignored outputs directory.",
        "- All TEST scoring was run once after prediction hashes and source/config pins were verified.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    import argparse
    import hashlib

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "verify", "verify-inputs"))
    action = parser.parse_args().action
    pin_path = REPORT.with_suffix(REPORT.suffix + ".sha256")
    if action == "verify-inputs":
        inputs = verify_inputs()
        print("TAPVIDMV_BATCH44D_INPUT_AUDIT_PASS",
              inputs["dev"]["source_commit"], inputs["manifest_sha256"])
    elif action == "build":
        OUT.mkdir(parents=True, exist_ok=True)
        report = build()
        encoded = report.encode("utf-8")
        src.write_once(REPORT, encoded)
        digest = hashlib.sha256(encoded).hexdigest()
        src.write_once(pin_path, (digest + "\n").encode())
        print("TAPVIDMV_REPORT_BUILT", REPORT, len(encoded), "bytes", digest)
    else:
        if not REPORT.is_file() or not pin_path.is_file():
            raise RuntimeError("TAPVIDMV_REPORT_PIN_MISSING")
        digest = hashlib.sha256(REPORT.read_bytes()).hexdigest()
        if pin_path.read_text(encoding="utf-8").strip() != digest:
            raise RuntimeError("TAPVIDMV_REPORT_PIN_DRIFT")
        print("TAPVIDMV_REPORT_PIN_AUDIT_PASS", digest)


if __name__ == "__main__":
    main()
