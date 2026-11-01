from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import kinetics_batch42_data as data
import kinetics_batch42_eval as ev
import kinetics_batch42_vlm as vlm


def textured_frames(n=6, dx=1, dy=0):
    rng = np.random.default_rng(2026)
    base = rng.integers(20, 235, (256, 256), dtype=np.uint8)
    return [
        cv2.warpAffine(
            base, np.float32([[1, 0, i * dx], [0, 1, i * dy]]),
            (256, 256), flags=cv2.INTER_LINEAR,
        )
        for i in range(n)
    ]


def affine_cfg():
    return {
        "affine_orb_features": 700,
        "affine_ratio": 0.75,
        "affine_min_matches": 12,
        "affine_min_inliers": 8,
        "affine_min_inlier_ratio": 0.25,
        "affine_ransac_reprojection_px": 3.0,
        "affine_max_reprojection_p50_px": 2.5,
        "affine_scale_min": 0.75,
        "affine_scale_max": 1.33,
        "affine_max_rotation_deg": 25.0,
    }


def test_tapvid_csv_row_parser_and_occlusion_semantics():
    triples = []
    for i in range(250):
        triples.extend([str(0.2 + i / 5000), "0.4", "1" if i in (0, 4) else "0"])
    row = ["videoid", "10", "20"] + triples
    xy, occ = data.parse_track_row(row)
    assert xy.shape == (250, 2)
    assert occ.shape == (250,)
    assert bool(occ[0]) and bool(occ[4])
    assert not bool(occ[1])


def test_tapvid_coordinate_conversion_matches_half_pixel_rule():
    points = np.asarray([[0.0, 0.0], [1.0, 1.0], [0.5, 0.25]])
    actual = data.normalized_to_resized_pixels(points, 1280, 720)
    np.testing.assert_allclose(
        actual,
        [[-0.1, -0.1777777778], [255.9, 255.8222222222],
         [127.9, 63.8222222222]],
        atol=1e-8,
    )


def test_first_visible_query_uses_first_visible_frame_only():
    xy = np.asarray([[0.2, 0.3], [0.25, 0.35], [0.4, 0.5]])
    occ = np.asarray([True, False, False])
    query = data.first_visible_query(7, xy, occ, 1280, 720)
    assert query["query_time"] == 1
    assert query["id"] == "p0007"
    assert query["query_protocol"] == "official_first_visible_point"


def test_first_visible_query_accepts_official_half_pixel_edge():
    xy = np.asarray([[0.0, 0.4]])
    query = data.first_visible_query(1, xy, np.asarray([False]), 1280, 720)
    assert query["query_xy"][0] == -0.1


def test_affine_chain_composes_current_to_previous_matrices():
    matrices = [
        None,
        np.asarray([[1, 0, -2], [0, 1, 0]], dtype=np.float64),
        np.asarray([[1, 0, -3], [0, 1, 1]], dtype=np.float64),
    ]
    chain = ev.relative_transforms(matrices, 0)
    point = chain[2] @ np.asarray([10.0, 10.0, 1.0])
    np.testing.assert_allclose(point[:2], [5.0, 11.0])


def test_orb_ransac_estimates_background_translation():
    frames = textured_frames(n=2, dx=3, dy=2)
    matrix, diag = ev.estimate_affine_current_to_previous(
        frames[0], frames[1], affine_cfg()
    )
    assert matrix is not None
    assert diag["accepted"]
    assert abs(float(matrix[0, 2]) + 3) < 1.5
    assert abs(float(matrix[1, 2]) + 2) < 1.5


def test_klt_forward_backward_tracks_small_translation():
    frames = textured_frames(n=7, dx=1, dy=1)
    query = {"query_time": 0, "query_xy": [120.0, 120.0]}
    trace = ev.klt_point(frames, query, fb=True, fb_error_px=1.5)
    assert trace[0] == [120.0, 120.0]
    assert trace[-1] is not None
    assert abs(trace[-1][0] - 126) < 2.0
    assert abs(trace[-1][1] - 126) < 2.0


def test_klt_query_late_never_predicts_before_query():
    frames = textured_frames(n=6)
    query = {"query_time": 3, "query_xy": [120.0, 120.0]}
    trace = ev.klt_point(frames, query, fb=False, fb_error_px=1.5)
    assert trace[:3] == [None, None, None]
    assert trace[3] == [120.0, 120.0]


def test_klt_abstains_on_textureless_frames():
    frames = [np.zeros((256, 256), dtype=np.uint8) for _ in range(5)]
    query = {"query_time": 0, "query_xy": [128.0, 128.0]}
    trace = ev.klt_point(frames, query, fb=True, fb_error_px=1.5)
    assert trace[0] == [128.0, 128.0]
    assert trace[1:] == [None] * 4


def test_appearance_gate_rejects_photometric_identity_break():
    first = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(first, (100, 100), (120, 120), 220, -1)
    second = 255 - first
    query = {"query_time": 0, "query_xy": [110.0, 110.0]}
    base = [[110.0, 110.0], [110.0, 110.0]]
    result = ev.appearance_gate([first, second], query, base, 0.70, 21)
    assert result[0] == base[0]
    assert result[1] is None


def test_appearance_gate_abstains_on_low_texture_anchor():
    gray = [np.full((256, 256), 128, dtype=np.uint8) for _ in range(4)]
    query = {"query_time": 1, "query_xy": [128.0, 128.0]}
    base = [None, [128.0, 128.0], [129.0, 128.0], [130.0, 128.0]]
    result = ev.appearance_gate(gray, query, base, 0.55, 21)
    assert result == [None, [128.0, 128.0], None, None]


def test_tapvid_jaccard_counts_wrong_visible_and_occluded_false_accepts():
    counts = ev.method_counts()
    counts.update({
        "visible": 2, "invisible": 1, "correct8": 1, "accepted_visible": 2,
        "wrong_visible_accept": 1, "visible_abstained": 0,
        "invisible_false_accept": 1, "correct_invisible_abstain": 0,
        "visibility_correct": 2, "predicted_visible": 3,
        "tp1": 1, "tp2": 1, "tp4": 1, "tp8": 1, "tp16": 1,
    })
    metric = ev.metric_rates(counts, [5.0, 7.0])
    assert metric["pck8_visible_unconditional"] == 0.5
    assert metric["visible_coverage"] == 1.0
    assert metric["invisible_false_accept_rate"] == 1.0
    assert metric["jaccard_mean_1_2_4_8_16"] == pytest.approx(1 / 4)
    assert metric["cpu_ms_per_target_frame_p50"] == 6.0


def test_parse_vlm_label_is_strict_and_case_normalized():
    assert vlm.parse_label("VISIBLE") == ("VISIBLE", "valid")
    assert vlm.parse_label("not_visible.") == ("NOT_VISIBLE", "valid")
    assert vlm.parse_label("UNCERTAIN") == ("UNCERTAIN", "valid")
    assert vlm.parse_label("The point is visible") == ("UNCERTAIN", "invalid_format")


def test_vlm_gate_metrics_count_uncertain_as_abstention():
    metrics = vlm.summarize_binary(
        ["VISIBLE", "NOT_VISIBLE", "UNCERTAIN", "VISIBLE"],
        [True, False, True, False],
        ["valid", "valid", "valid", "valid"],
    )
    assert metrics["visible_support"] == 2
    assert metrics["not_visible_support"] == 2
    assert metrics["coverage"] == 0.75
    assert metrics["invalid_rate"] == 0.0
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["not_visible_false_accept_rate"] == 0.5


def test_prediction_uses_public_queries_without_private_oracle(tmp_path, monkeypatch):
    frames = np.stack([
        np.stack([frame, frame, frame], axis=-1) for frame in textured_frames(n=5)
    ])
    (tmp_path / "data/frames").mkdir(parents=True)
    np.save(tmp_path / "data/frames/clip.npy", frames, allow_pickle=False)
    data_dir = tmp_path / "data/frozen"
    data_dir.mkdir(parents=True)
    (data_dir / "manifest.json").write_text("{}")
    queries = {"clip_00": [{"id": "p0001", "query_time": 0,
                            "query_xy": [120.0, 120.0]}]}
    (data_dir / "queries-public.json").write_text(json.dumps(queries))
    monkeypatch.setattr(ev, "ROOT", tmp_path)
    monkeypatch.setattr(ev, "DATA", data_dir)
    monkeypatch.setattr(ev, "OUT", tmp_path / "out")
    monkeypatch.setattr(ev, "gitsha", lambda: "testsha")
    manifest = {"scenes": [{
        "name": "clip_00", "split": "dev", "frame_file": "data/frames/clip.npy"
    }]}
    monkeypatch.setattr(ev.src, "audit", lambda: manifest)
    cfg = {
        "tracker": {
            "fb_error_px": 1.5, "patch_size": 21,
            "template_search_radius_px": 16, "template_ncc_min": 0.7,
            "template_max_consecutive_misses": 3,
            "affine_orb_features": 300, "affine_ratio": 0.75,
            "affine_min_matches": 8, "affine_min_inliers": 5,
            "affine_min_inlier_ratio": 0.2, "affine_ransac_reprojection_px": 3.0,
            "affine_max_reprojection_p50_px": 3.0, "affine_scale_min": 0.7,
            "affine_scale_max": 1.4, "affine_max_rotation_deg": 35,
            "appearance_thresholds": [0.55, 0.70, 0.85],
        },
        "evaluation": {"bootstrap_seed": 1, "bootstrap_replicates": 10},
        "dev_selection": {"candidate_methods": ["klt_fb"],
                          "min_visible_coverage": 0.4,
                          "max_invisible_false_accept_rate": 0.25,
                          "fallback_method": "klt_fb"},
    }
    assert not (data_dir / "oracle-private.json").exists()
    ev.predict("dev", cfg)
    prediction = json.loads((tmp_path / "out/dev_predictions.json").read_text())
    assert prediction["oracle_unread_during_prediction"] is True
    assert set(prediction["records"][0]["tracks"]["p0001"]) == set(ev.METHODS)
    ev.require_pin(tmp_path / "out/dev_predictions.json")


def test_pin_detects_prediction_mutation(tmp_path):
    path = tmp_path / "predictions.json"
    path.write_text('{"x":1}')
    ev.pin(path)
    ev.require_pin(path)
    path.write_text('{"x":2}')
    with pytest.raises(RuntimeError, match="CHANGED"):
        ev.require_pin(path)


def test_frozen_write_once_rejects_changed_payload(tmp_path):
    path = tmp_path / "oracle.json"
    data.atomic_once(path, b"first")
    data.atomic_once(path, b"first")
    with pytest.raises(RuntimeError, match="CONFLICT"):
        data.atomic_once(path, b"second")


def test_vlm_test_gate_block_skips_without_opening_oracle_or_loading_model(tmp_path, monkeypatch):
    data_dir = tmp_path / "frozen"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text("{}")
    (data_dir / "queries-public.json").write_text("{}")
    out = tmp_path / "out"
    out.mkdir()
    cfg = {"vlm": {"target_fractions": [0.5, 0.9], "max_new_tokens": 8,
                   "model_id": "local", "model_path": "../missing",
                   "dev_gate": {}}}
    config_sha = vlm.sha_bytes(ev.json_bytes(cfg))
    gate = {"status": "BLOCK", "config_sha256": config_sha}
    gate_path = out / "dev_vlm_gate.json"
    gate_path.write_text(json.dumps(gate))
    vlm.pin(gate_path)
    monkeypatch.setattr(vlm, "DATA", data_dir)
    monkeypatch.setattr(vlm, "OUT", out)
    monkeypatch.setattr(vlm.ev, "gitsha", lambda: "testsha")
    monkeypatch.setattr(vlm.src, "audit", lambda: {"scenes": []})
    assert not (data_dir / "oracle-private.json").exists()
    vlm.predict("test", cfg)
    result = json.loads((out / "test_vlm_predictions.json").read_text())
    assert result["status"] == "SKIPPED_DEV_GATE_BLOCKED"
    assert result["oracle_never_read"] is True
    assert not (data_dir / "oracle-private.json").exists()
    vlm.require_pin(out / "test_vlm_predictions.json")