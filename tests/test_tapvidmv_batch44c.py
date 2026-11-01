from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import tapvidmv_batch44c_data as data
import tapvidmv_batch44c_eval as evaluation
import tapvidmv_batch44c_vlm as vlm


def synthetic_records(cfg):
    records = {}
    for item in cfg["sequences"]:
        records[item["sequence"]] = {
            "sequence": item["sequence"],
            "subset": item["subset"],
            "group": item["group"],
            "views": 3,
            "frames": item["frames"],
            "tracks_metadata": item["tracks_metadata"],
            "url": "https://example.invalid/" + item["sequence"] + ".rrd",
        }
    return records


def test_project_and_three_camera_triangulation_recover_world_point():
    intrinsics = np.array([
        [120.0, 0.0, 0.0],
        [0.0, 120.0, 0.0],
        [160.0, 120.0, 1.0],
    ])
    rotations = [np.eye(3), np.eye(3), np.eye(3)]
    centers = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    point = np.array([0.3, 0.2, 5.0])
    pixels = [
        data.project_point(point, rotation, center, intrinsics)[0]
        for rotation, center in zip(rotations, centers)
    ]
    recovered = evaluation.triangulate_rays(
        pixels, rotations, centers, [intrinsics] * 3
    )
    assert recovered is not None
    assert np.linalg.norm(recovered - point) < 1e-6


def test_triangulation_abstains_for_degenerate_camera_baseline():
    intrinsics = np.array([
        [100.0, 0.0, 0.0],
        [0.0, 100.0, 0.0],
        [64.0, 64.0, 1.0],
    ])
    point = np.array([0.2, 0.1, 4.0])
    rotations = [np.eye(3)] * 3
    centers = [np.zeros(3)] * 3
    pixels = [
        data.project_point(point, rotation, center, intrinsics)[0]
        for rotation, center in zip(rotations, centers)
    ]
    assert evaluation.triangulate_rays(
        pixels, rotations, centers, [intrinsics] * 3
    ) is None


def test_project_point_marks_behind_camera_as_invalid():
    intrinsics = np.eye(3)
    pixel, depth = data.project_point(
        np.array([0.0, 0.0, -2.0]),
        np.eye(3),
        np.zeros(3),
        intrinsics,
    )
    assert depth < 0
    assert not np.isfinite(pixel).all()


def test_depth_visibility_states_are_conservative():
    depth = np.full((21, 21), np.inf, dtype=np.float32)
    depth[10, 10] = 5.0
    radius = 0
    margin = 0.05
    assert data.classify_depth(np.array([10.0, 10.0]), 5.02, depth, radius, margin) == data.STATUS["visible"]
    assert data.classify_depth(np.array([10.0, 10.0]), 6.0, depth, radius, margin) == data.STATUS["occluded"]
    assert data.classify_depth(np.array([10.0, 10.0]), 4.0, depth, radius, margin) == data.STATUS["unknown"]
    assert data.classify_depth(np.array([-1.0, 10.0]), 5.0, depth, radius, margin) == data.STATUS["out"]
    assert data.classify_depth(np.array([10.0, 10.0]), 5.0, np.full_like(depth, np.inf), radius, margin) == data.STATUS["unknown"]


def test_depth_buffer_keeps_nearest_projected_surface():
    cloud = np.array([
        [0.0, 0.0, 5.0],
        [0.0, 0.0, 3.0],
    ])
    intrinsics = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [10.0, 10.0, 1.0],
    ])
    buffer = data.make_depth_buffer(
        cloud, (np.eye(3), np.zeros(3)), intrinsics, 21, 21
    )
    assert buffer[10, 10] == pytest.approx(3.0)


def test_deterministic_selection_is_group_disjoint_and_subset_balanced():
    cfg = data.config()
    selected = data.deterministic_selection(synthetic_records(cfg), cfg)
    assert len(selected) == 21
    group_splits = {}
    for item in selected:
        group_splits.setdefault(item["group"], set()).add(item["split"])
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert [item["split"] for item in selected] == ["dev"] * 11 + ["test"] * 10
    counts = {}
    for item in selected:
        counts.setdefault(item["subset"], {"dev": 0, "test": 0})[item["split"]] += 1
    assert counts == cfg["subset_split_counts"]


def test_deterministic_selection_rejects_collection_group_overlap():
    cfg = copy.deepcopy(data.config())
    cfg["sequences"][11]["group"] = cfg["sequences"][0]["group"]
    records = synthetic_records(cfg)
    with pytest.raises(RuntimeError, match="CROSS_SPLIT_COLLECTION_GROUP_LEAKAGE"):
        data.deterministic_selection(records, cfg)


def test_public_calibration_schema_contains_only_pose_and_intrinsics():
    arrays = {
        "rotations": np.repeat(np.eye(3, dtype=np.float32)[None, None, :, :], 3, axis=0).repeat(5, axis=1),
        "translations": np.zeros((3, 5, 3), dtype=np.float32),
        "intrinsics": np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 3, axis=0),
        "resolution": np.asarray([[64, 64]] * 3, dtype=np.int32),
    }
    assert data.validate_public_calibration(arrays, 5)
    leaked = dict(arrays)
    leaked["world_xyz"] = np.zeros((3, 5, 3), dtype=np.float32)
    with pytest.raises(RuntimeError, match="PUBLIC_CALIBRATION_SCHEMA_LEAK_OR_DRIFT"):
        data.validate_public_calibration(leaked, 5)


def _synthetic_calibration():
    intrinsics = np.asarray([[[120.0,0.0,0.0],[0.0,120.0,0.0],[160.0,120.0,1.0]]]*3,dtype=np.float64)
    rotations = np.repeat(np.eye(3,dtype=np.float64)[None,None,:,:],3,axis=0).repeat(2,axis=1)
    centers = np.asarray([[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]])
    translations = np.repeat(centers[:,None,:],2,axis=1)
    resolutions = np.asarray([[320,240]]*3,dtype=np.int32)
    point = np.asarray([0.3,0.2,5.0],dtype=np.float64)
    pixels = [data.project_point(point,rotations[c,1],translations[c,1],intrinsics[c])[0] for c in range(3)]
    rows = {c:{"traces":{"klt_fb":[[160.0,120.0],np.asarray(pixels[c]).tolist()]}} for c in range(3)}
    return rows,rotations,translations,intrinsics,resolutions,pixels


def test_cross_view_consensus_accepts_consistent_calibrated_predictions():
    rows,rotations,translations,intrinsics,resolutions,pixels = _synthetic_calibration()
    result = evaluation.cross_view_consensus_group(rows,"klt_fb",2.0,rotations,translations,intrinsics,resolutions,2)
    for camera_id in range(3):
        assert result[camera_id][1] is not None
        assert np.linalg.norm(np.asarray(result[camera_id][1])-pixels[camera_id]) < 1e-4


def test_cross_view_consensus_abstains_when_one_view_is_an_outlier():
    rows,rotations,translations,intrinsics,resolutions,pixels = _synthetic_calibration()
    rows[2]["traces"]["klt_fb"][1] = (np.asarray(pixels[2])+np.asarray([35.0,-20.0])).tolist()
    result = evaluation.cross_view_consensus_group(rows,"klt_fb",4.0,rotations,translations,intrinsics,resolutions,2)
    assert all(result[c][1] is None for c in range(3))


def test_cross_view_consensus_abstains_if_any_view_has_no_prediction():
    rows,rotations,translations,intrinsics,resolutions,_ = _synthetic_calibration()
    rows[1]["traces"]["klt_fb"][1] = None
    result = evaluation.cross_view_consensus_group(rows,"klt_fb",12.0,rotations,translations,intrinsics,resolutions,2)
    assert all(result[c][1] is None for c in range(3))


def test_choose_query_frame_uses_earliest_common_visible_snapshot():
    counts = {0: 0, 6: 2, 12: 3, 18: 7}
    assert data.choose_query_frame(counts, nframes=80, min_future_frames=30, min_common=3) == 12


def test_choose_query_frame_respects_future_horizon():
    counts = {55: 3, 60: 8}
    with pytest.raises(RuntimeError, match="NO_COMMON_THREE_VIEW_VISIBLE_QUERY_FRAME"):
        data.choose_query_frame(counts, nframes=80, min_future_frames=30, min_common=3)


def test_cross_view_consensus_starts_at_nonzero_query_frame():
    rows, rotations, translations, intrinsics, resolutions, pixels = _synthetic_calibration()
    rows = {c: {"query_frame": 1, "traces": {"klt_fb": [None, [160.0,120.0], np.asarray(pixels[c]).tolist()]}} for c in range(3)}
    rotations3 = np.repeat(rotations[:, :1], 3, axis=1)
    translations3 = np.repeat(translations[:, :1], 3, axis=1)
    result = evaluation.cross_view_consensus_group(rows, "klt_fb", 2.0, rotations3, translations3, intrinsics, resolutions, 3)
    for camera_id in range(3):
        assert result[camera_id][0] is None
        assert result[camera_id][1] == [160.0, 120.0]
        assert result[camera_id][2] is not None


def test_write_once_is_idempotent_and_refuses_conflict(tmp_path):
    path = tmp_path / "frozen.bin"
    data.write_once(path, b"first")
    data.write_once(path, b"first")
    assert path.read_bytes() == b"first"
    with pytest.raises(RuntimeError, match="FROZEN_FILE_CONFLICT"):
        data.write_once(path, b"second")


def test_save_array_once_checks_exact_shape_dtype_and_values(tmp_path):
    path = tmp_path / "frames.npy"
    first = np.arange(12, dtype=np.uint8).reshape(3, 4)
    data.save_array_once(path, first)
    data.save_array_once(path, first.copy())
    with pytest.raises(RuntimeError, match="FROZEN_ARRAY_CONFLICT"):
        data.save_array_once(path, first + 1)


def test_save_npz_once_checks_all_private_oracle_arrays(tmp_path):
    path = tmp_path / "oracle.npz"
    data.save_npz_once(path, x=np.array([1, 2]), y=np.array([True, False]))
    data.save_npz_once(path, x=np.array([1, 2]), y=np.array([True, False]))
    with pytest.raises(RuntimeError, match="FROZEN_ORACLE_CONFLICT"):
        data.save_npz_once(path, x=np.array([1, 3]), y=np.array([True, False]))


def test_cpu_eval_disables_opencv_opencl_backend():
    assert evaluation.CV_OPENCL_DISABLED
    if hasattr(evaluation.cv2, "ocl"):
        assert not evaluation.cv2.ocl.useOpenCL()


def test_affine_cpu_latency_ignores_prequery_none_entries():
    assert evaluation.sum_affine_cpu_ms(
        [None, {"cpu_ms": 1.25}, None, {"cpu_ms": 2.75}, {"cpu_ms": None}]
    ) == pytest.approx(4.0)
    assert evaluation.sum_affine_cpu_ms(None) == 0.0


def test_klt_forward_tracks_a_synthetic_translation():
    rng = np.random.default_rng(43)
    first = rng.integers(0, 256, size=(96, 96), dtype=np.uint8)
    matrix = np.float32([[1, 0, 2], [0, 1, 1]])
    second = cv2.warpAffine(first, matrix, (96, 96), flags=cv2.INTER_NEAREST)
    trace = evaluation.klt_trace([first, second], [45.0, 45.0], fb=False)
    assert len(trace) == 2
    assert trace[1] is not None
    assert np.linalg.norm(np.asarray(trace[1]) - np.array([47.0, 46.0])) < 1.5


def test_vlm_label_parser_is_strict_and_accepts_normalized_not_visible():
    assert vlm.parse_label("VISIBLE") == ("VISIBLE", "valid")
    assert vlm.parse_label("NOT VISIBLE") == ("NOT_VISIBLE", "valid")
    assert vlm.parse_label("NOT_VISIBLE.") == ("NOT_VISIBLE", "valid")
    assert vlm.parse_label("UNCERTAIN") == ("UNCERTAIN", "valid")
    assert vlm.parse_label("The point is visible.") == ("INVALID", "invalid_format")


def test_vlm_prediction_pin_detects_post_seal_mutation(tmp_path, monkeypatch):
    path = tmp_path / "predictions.json"
    path.write_text('{"status":"PREDICTED"}', encoding="utf-8")
    vlm.pin(path)
    vlm.require_pin(path)
    path.write_text('{"status":"CHANGED"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="PIN_DRIFT"):
        vlm.require_pin(path)


def test_dev_method_selection_uses_only_pinned_dev_results(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "OUT", tmp_path)
    monkeypatch.setattr(evaluation, "gitsha", lambda: "frozen-source-sha")
    cfg = data.config()
    config_sha = evaluation.sha_bytes(data.json_bytes(cfg))
    methods = {}
    for index, method in enumerate(cfg["selection"]["candidate_methods"]):
        methods[method] = {
            "coverage_depth_visible": 0.60,
            "false_accept_occluded": 0.10,
            "false_accept_out": 0.10,
            "jaccard_mean_scene_macro": 0.50 + index * 0.01,
        }
    results = {
        "source_commit": "frozen-source-sha",
        "split": "dev",
        "config_sha256": config_sha,
        "methods": methods,
    }
    result_path = tmp_path / "dev_results.json"
    evaluation.once(result_path, results)
    evaluation.pin(result_path)
    selected = evaluation.select_dev()
    assert selected["status"] == "PASS"
    assert selected["selection_source"] == "DEV only"
    assert selected["selected_method"] == cfg["selection"]["candidate_methods"][-1]
    evaluation.require_pin(tmp_path / "dev_selection.json")


def test_dev_method_selection_falls_back_when_no_candidate_is_safe(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "OUT", tmp_path)
    monkeypatch.setattr(evaluation, "gitsha", lambda: "frozen-source-sha")
    cfg = data.config()
    config_sha = evaluation.sha_bytes(data.json_bytes(cfg))
    methods = {
        method: {
            "coverage_depth_visible": 0.20,
            "false_accept_occluded": 0.80,
            "false_accept_out": 0.70,
            "jaccard_mean_scene_macro": 0.99,
        }
        for method in cfg["selection"]["candidate_methods"]
    }
    result_path = tmp_path / "dev_results.json"
    evaluation.once(result_path, {
        "source_commit": "frozen-source-sha",
        "split": "dev",
        "config_sha256": config_sha,
        "methods": methods,
    })
    evaluation.pin(result_path)
    selected = evaluation.select_dev()
    assert selected["status"] == "BLOCK_NO_ELIGIBLE_METHOD"
    assert selected["selected_method"] == cfg["selection"]["fallback_method"]


def test_split_specific_oracle_audit_does_not_hash_other_split(tmp_path, monkeypatch):
    frozen = tmp_path / "frozen"
    (frozen / "oracle-private").mkdir(parents=True)
    dev_path = frozen / "oracle-private/dev.npz"
    test_path = frozen / "oracle-private/test.npz"
    for path in (dev_path, test_path):
        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                qids=np.asarray([], dtype="U16"),
                depth_status=np.empty((0, 1), dtype=np.uint8),
                track_present=np.empty((0, 1), dtype=np.bool_),
                in_frame=np.empty((0, 1), dtype=np.bool_),
            )
    (frozen / "queries-public.json").write_text(
        json.dumps({"dev_scene": [], "test_scene": []}), encoding="utf-8"
    )
    scenes = [
        {"name": "dev_scene", "split": "dev", "oracle_file": "oracle-private/dev.npz",
         "oracle_sha256": data.sha256_file(dev_path)},
        {"name": "test_scene", "split": "test", "oracle_file": "oracle-private/test.npz",
         "oracle_sha256": data.sha256_file(test_path)},
    ]
    monkeypatch.setattr(data, "FROZEN", frozen)
    monkeypatch.setattr(data, "audit_public", lambda: {"scenes": scenes})
    real_hash = data.sha256_file
    seen = []

    def tracked_hash(path):
        seen.append(Path(path).name)
        return real_hash(path)

    monkeypatch.setattr(data, "sha256_file", tracked_hash)
    data.audit_private("dev")
    assert "dev.npz" in seen
    assert "test.npz" not in seen


def test_vlm_dev_gate_blocks_if_nonvisible_false_accept_is_high(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm, "OUT", tmp_path)
    monkeypatch.setattr(evaluation, "gitsha", lambda: "frozen-source-sha")
    cfg = data.config()
    metrics = {
        "balanced_accuracy": 0.80,
        "visible_support": 20,
        "nonvisible_support": 12,
        "coverage": 0.90,
        "invalid_rate": 0.0,
        "false_accept_nonvisible_rate": 0.50,
    }
    result_path = tmp_path / "dev_vlm_results.json"
    evaluation.once(result_path, {
        "source_commit": "frozen-source-sha",
        "split": "dev",
        "status": "SCORED",
        "config_sha256": evaluation.sha_bytes(data.json_bytes(cfg)),
        "vlm": metrics,
    })
    evaluation.pin(result_path)
    gate = vlm.make_dev_gate()
    assert gate["status"] == "BLOCK"
    assert gate["checks"]["balanced_accuracy"] is True
    assert gate["checks"]["false_accept_nonvisible_rate"] is False
    vlm.require_pin(tmp_path / "dev_vlm_gate.json")


def test_vlm_dev_gate_passes_only_when_all_frozen_checks_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm, "OUT", tmp_path)
    monkeypatch.setattr(evaluation, "gitsha", lambda: "frozen-source-sha")
    cfg = data.config()
    metrics = {
        "balanced_accuracy": 0.70,
        "visible_support": 20,
        "nonvisible_support": 12,
        "coverage": 0.85,
        "invalid_rate": 0.05,
        "false_accept_nonvisible_rate": 0.10,
    }
    result_path = tmp_path / "dev_vlm_results.json"
    evaluation.once(result_path, {
        "source_commit": "frozen-source-sha",
        "split": "dev",
        "status": "SCORED",
        "config_sha256": evaluation.sha_bytes(data.json_bytes(cfg)),
        "vlm": metrics,
    })
    evaluation.pin(result_path)
    gate = vlm.make_dev_gate()
    assert gate["status"] == "PASS"
    assert all(gate["checks"].values())
