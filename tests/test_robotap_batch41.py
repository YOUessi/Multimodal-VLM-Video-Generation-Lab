"""Batch41 regression tests: immutable RoboTAP data, point visibility and oracle firewall."""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import robotap_batch41_data as data
import robotap_batch41_eval as ev
import robotap_batch41_vlm as vlm


def textured_video(n=7, dx=1, dy=1, seed=21):
    rng = np.random.default_rng(seed)
    base = np.zeros((96, 96), dtype=np.uint8)
    cv2.circle(base, (40, 40), 8, 225, -1)
    cv2.rectangle(base, (26, 52), (57, 79), 140, 2)
    for _ in range(24):
        x = int(rng.integers(5, 88))
        y = int(rng.integers(5, 88))
        radius = int(rng.integers(3, 8))
        value = int(rng.integers(80, 255))
        cv2.circle(base, (x, y), radius, value, -1)
    for _ in range(16):
        x = int(rng.integers(2, 80))
        y = int(rng.integers(2, 80))
        width = int(rng.integers(5, 15))
        height = int(rng.integers(5, 15))
        value = int(rng.integers(80, 255))
        cv2.rectangle(base, (x, y), (x + width, y + height), value, 1)
    return [
        cv2.warpAffine(
            base, np.float32([[1, 0, k * dx], [0, 1, k * dy]]),
            (96, 96), flags=cv2.INTER_LINEAR,
        )
        for k in range(n)
    ]

@pytest.mark.parametrize("seed", ["a", "b", "c", "d"])
def test_scene_selection_is_hash_stable_and_order_independent(seed):
    records = [(seed + str(i), {}) for i in range(40)]
    a = data.selected_video_keys(records, 20)
    b = data.selected_video_keys(records[::-1], 20)
    assert a == b
    assert len(a) == len(set(a)) == 20


def test_scene_selection_fails_closed_when_too_few_videos():
    with pytest.raises(RuntimeError, match="NOT_ENOUGH"):
        data.selected_video_keys([("only", {})], 20)


def test_normalized_points_convert_using_width_and_height():
    points = np.asarray([[[0.5, 0.25], [0.2, 0.9]]], dtype=np.float64)
    occ = np.asarray([[False, True]], dtype=bool)
    pixels = data.normalized_to_pixels(points, occ, 200, 400)
    assert pixels[0, 0].tolist() == [200.0, 50.0]
    assert pixels.shape == (1, 2, 2)


def test_visible_normalized_coordinate_outside_unit_square_fails():
    points = np.asarray([[[1.2, 0.5]]], dtype=np.float64)
    with pytest.raises(RuntimeError, match="NOT_NORMALIZED"):
        data.normalized_to_pixels(points, np.asarray([[False]]), 100, 100)


def test_occluded_coordinate_can_be_outside_without_scoring_it():
    points = np.asarray([[[3.0, -2.0]]], dtype=np.float64)
    pixels = data.normalized_to_pixels(points, np.asarray([[True]]), 100, 100)
    assert pixels.shape == (1, 1, 2)


def test_first_visible_query_uses_first_visible_frame():
    points = np.asarray([[[-5.0, 0.0], [20.0, 30.0], [25.0, 30.0]]])
    occ = np.asarray([[True, False, False]])
    q = data.first_visible_query(3, points[0], occ[0], 100, 100)
    assert q["query_time"] == 1
    assert q["query_xy"] == [20.0, 30.0]
    assert q["id"] == "p0003"


def test_all_occluded_track_has_no_query():
    points = np.zeros((3, 2), dtype=np.float64)
    occ = np.ones(3, dtype=bool)
    assert data.first_visible_query(0, points, occ, 100, 100) is None


def test_first_visible_query_out_of_bounds_fails():
    points = np.asarray([[[-1.0, 10.0], [10.0, 10.0]]])
    occ = np.asarray([False, True])
    with pytest.raises(RuntimeError, match="OUT_OF_BOUNDS"):
        data.first_visible_query(0, points[0], occ, 100, 100)


class FakeResponse:
    def __init__(self, body=b"", status=200, headers=None):
        self.body = body
        self.status = status
        self.headers = headers or {}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self, n=-1):
        return self.body if n is None or n < 0 else self.body[:n]


def test_http_range_reader_uses_verified_byte_ranges(monkeypatch):
    raw = b"0123456789abcdef"
    calls = []
    def fake_urlopen(request, timeout=0):
        if request.get_method() == "HEAD":
            return FakeResponse(status=200, headers={
                "Content-Length": str(len(raw)),
                "ETag": '"etag-1"', "Accept-Ranges": "bytes",
            })
        start, end = map(int, request.headers["Range"].split("=")[1].split("-"))
        calls.append((start, end))
        return FakeResponse(
            raw[start:end + 1], status=206,
            headers={"Content-Range": f"bytes {start}-{end}/{len(raw)}"},
        )
    monkeypatch.setattr(data.urllib.request, "urlopen", fake_urlopen)
    reader = data.HttpRangeFile(
        "https://example.test/archive.zip", len(raw), '"etag-1"',
        block_bytes=4, cache_blocks=2,
    )
    assert reader.read(6) == raw[:6]
    reader.seek(2)
    assert reader.read(5) == raw[2:7]
    assert reader.seek(-3, 2) == 13
    assert reader.read() == raw[13:]
    assert calls


def test_http_range_reader_rejects_archive_etag_drift(monkeypatch):
    def fake_urlopen(request, timeout=0):
        return FakeResponse(status=200, headers={
            "Content-Length": "16", "ETag": '"different"',
            "Accept-Ranges": "bytes",
        })
    monkeypatch.setattr(data.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="ETAG"):
        data.HttpRangeFile("https://example.test/a.zip", 16, '"expected"')


@pytest.mark.parametrize("use_fb", [False, True])
def test_klt_tracks_known_image_plane_translation(use_fb):
    frames = textured_video(n=7, dx=1, dy=1)
    gray = frames
    trace, visible = ev.klt_point(
        gray, {"query_time": 0, "query_xy": [40.0, 40.0]},
        use_fb=use_fb, fb_error_px=1.5,
    )
    assert len(trace) == 7
    assert visible[0] and visible[-1]
    assert abs(trace[-1][0] - 46.0) < 2.0
    assert abs(trace[-1][1] - 46.0) < 2.0


def test_klt_late_query_never_predicts_before_query():
    frames = textured_video(n=5)
    trace, visible = ev.klt_point(
        frames, {"query_time": 3, "query_xy": [40.0, 40.0]},
        use_fb=True, fb_error_px=1.5,
    )
    assert trace[:3] == [None, None, None]
    assert visible[:3] == [False, False, False]
    assert visible[3]


def test_klt_abstains_on_textureless_frames():
    frames = [np.zeros((64, 64), dtype=np.uint8) for _ in range(5)]
    trace, visible = ev.klt_point(
        frames, {"query_time": 0, "query_xy": [32.0, 32.0]},
        use_fb=True, fb_error_px=1.5,
    )
    assert trace[0] == [32.0, 32.0]
    assert trace[1:] == [None] * 4
    assert visible[1:] == [False] * 4


def test_template_recovery_tracks_small_translation():
    trace, visible = ev.template_recover(
        textured_video(n=6, dx=1, dy=0),
        {"query_time": 0, "query_xy": [40.0, 40.0]},
        {"patch_size": 21, "template_search_radius_px": 32,
         "template_ncc_min": 0.75, "template_max_consecutive_misses": 4},
    )
    assert visible[-1]
    assert abs(trace[-1][0] - 45.0) < 3.0


def test_template_recovery_abstains_on_flat_patch():
    frames = [np.zeros((64, 64), dtype=np.uint8) for _ in range(5)]
    trace, visible = ev.template_recover(
        frames, {"query_time": 0, "query_xy": [32.0, 32.0]},
        {"patch_size": 21, "template_search_radius_px": 32,
         "template_ncc_min": 0.75, "template_max_consecutive_misses": 4},
    )
    assert trace[1:] == [None] * 4
    assert visible[1:] == [False] * 4


def test_orb_affine_estimator_recovers_background_translation():
    frames = textured_video(n=2, dx=2, dy=1, seed=44)
    matrix, diag = ev.estimate_affine_current_to_previous(
        frames[0], frames[1],
        {"affine_min_matches": 12, "affine_min_inliers": 8,
         "affine_min_inlier_ratio": 0.25, "affine_ransac_reprojection_px": 3.0},
    )
    assert matrix is not None
    assert diag["accepted"] is True
    assert abs(float(matrix[0, 2]) + 2.0) < 2.0
    assert abs(float(matrix[1, 2]) + 1.0) < 2.0


def test_affine_stabilized_klt_maps_back_to_original_image_plane():
    frames = textured_video(n=6, dx=1, dy=1, seed=51)
    cfg = {
        "affine_min_matches": 12, "affine_min_inliers": 8,
        "affine_min_inlier_ratio": 0.25, "affine_ransac_reprojection_px": 3.0,
        "fb_error_px": 1.5,
    }
    matrices, _ = ev.estimate_affines(frames, cfg)
    trace, visible = ev.affine_stabilized_klt(
        frames, {"query_time": 0, "query_xy": [40.0, 40.0]}, matrices, cfg
    )
    if all(matrix is not None for matrix in matrices[1:]):
        assert visible[-1]
        assert abs(trace[-1][0] - 45.0) < 3.0
        assert abs(trace[-1][1] - 45.0) < 3.0


def test_affine_failure_is_not_silently_treated_as_camera_pose():
    blank = [np.zeros((64, 64), dtype=np.uint8) for _ in range(4)]
    cfg = {
        "affine_min_matches": 12, "affine_min_inliers": 8,
        "affine_min_inlier_ratio": 0.25, "affine_ransac_reprojection_px": 3.0,
        "fb_error_px": 1.5,
    }
    matrices, diagnostics = ev.estimate_affines(blank, cfg)
    assert matrices[1:] == [None, None, None]
    assert all(not d["accepted"] for d in diagnostics[1:])


def test_metrics_keep_abstentions_and_occluded_false_accepts_in_denominator():
    counts = {
        "eval_total": 3, "visible_total": 2, "occluded_total": 1,
        "visible_accepted": 2, "correct8": 2, "wrong_visible_accept": 0,
        "visible_abstained": 0, "occluded_false_accept": 1,
        "occluded_correct_abstain": 0, "occlusion_correct": 2,
    }
    m = ev._metrics(counts, [1.0, 2.0, 3.0])
    assert m["pck8_visible_unconditional"] == 1.0
    assert m["jaccard8"] == pytest.approx(2 / 3)
    assert m["visible_coverage"] == 1.0
    assert m["accepted_precision8"] == pytest.approx(2 / 3)
    assert m["occluded_false_accept_rate"] == 1.0
    assert m["occlusion_accuracy"] == pytest.approx(2 / 3)


def test_vlm_visibility_parser_is_strict():
    assert vlm.parse_visibility("VISIBLE") == ("VISIBLE", "valid_visible")
    assert vlm.parse_visibility("ABSENT") == ("ABSENT", "valid_absent")
    assert vlm.parse_visibility("UNCERTAIN") == ("UNCERTAIN", "valid_uncertain")
    assert vlm.parse_visibility("The point is visible.") == (None, "invalid_format")
    assert vlm.parse_visibility("") == (None, "invalid_format")


def test_prediction_pin_detects_tampering(tmp_path):
    path = tmp_path / "predictions.json"
    path.write_text('{"a": 1}')
    ev.pin(path)
    ev.require_pin(path)
    path.write_text('{"a": 2}')
    with pytest.raises(RuntimeError, match="CHANGED"):
        ev.require_pin(path)


def test_cpu_predictor_does_not_require_oracle_file(tmp_path, monkeypatch):
    frozen = tmp_path / "data" / "batch41_frozen"
    (frozen / "frames").mkdir(parents=True)
    frames = np.stack([
        np.stack([x, x, x], axis=-1) for x in textured_video(n=5, dx=1, dy=0)
    ]).astype(np.uint8)
    np.save(frozen / "frames/scene_000.npy", frames, allow_pickle=False)
    manifest = {
        "scenes": [{
            "name": "scene_000", "split": "dev", "frames": "frames/scene_000.npy",
            "nframes": 5, "nqueries": 1,
        }]
    }
    (frozen / "manifest.json").write_text(json.dumps(manifest))
    (frozen / "queries-public.json").write_text(json.dumps({
        "scene_000": [{"id": "p0000", "query_time": 0, "query_xy": [40.0, 40.0]}]
    }))
    monkeypatch.setattr(ev, "ROOT", tmp_path)
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(ev, "current_sha", lambda: "testsha")
    monkeypatch.setattr(data, "audit", lambda cfg: manifest)
    cfg = {
        "dataset_dir": "data", "output_dir": "out",
        "tracker": {
            "fb_error_px": 1.5, "patch_size": 21,
            "template_search_radius_px": 32, "template_ncc_min": 0.75,
            "template_max_consecutive_misses": 4, "affine_min_matches": 12,
            "affine_min_inliers": 8, "affine_min_inlier_ratio": 0.25,
            "affine_ransac_reprojection_px": 3.0,
        },
    }
    assert not (frozen / "oracle-private.json").exists()
    ev.predict("dev", cfg)
    prediction = json.loads((tmp_path / "out/dev_predictions.json").read_text())
    assert prediction["oracle_unread_during_prediction"] is True
    assert not (frozen / "oracle-private.json").exists()
    ev.require_pin(tmp_path / "out/dev_predictions.json")


def test_score_uses_private_occlusion_only_after_prediction_is_pinned(tmp_path, monkeypatch):
    frozen = tmp_path / "data" / "batch41_frozen"
    out = tmp_path / "out"
    frozen.mkdir(parents=True)
    out.mkdir()
    manifest_doc = {"scenes": [{"name": "scene_test", "split": "test"}]}
    (frozen / "manifest.json").write_text(json.dumps(manifest_doc))
    (frozen / "queries-public.json").write_text(json.dumps({
        "scene_test": [{"id": "p0000", "query_time": 0, "query_xy": [10.0, 10.0]}]
    }))
    monkeypatch.setattr(ev, "ROOT", tmp_path)
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(ev, "current_sha", lambda: "testsha")
    monkeypatch.setattr(data, "audit", lambda cfg: manifest_doc)
    gold = {"scene_test": {"tracks": {"p0000": {
        "xy": [[10, 10], [11, 10], [12, 10], [13, 10]],
        "occluded": [False, False, True, False],
    }}}}
    monkeypatch.setattr(data, "load_gold", lambda cfg: gold)
    selection = {
        "status": "selected_from_dev_only", "source_commit": "testsha",
        "selected_method": "klt_fb",
    }
    (out / "dev_selection.json").write_text(json.dumps(selection))
    ev.pin(out / "dev_selection.json")
    methods = list(ev.METHODS)
    predictions = {}
    for method in methods:
        if method == "klt_fb":
            xy = [[10, 10], [11, 10], None, [13, 10]]
            visible = [True, True, False, True]
        else:
            xy = [[10, 10], [10, 10], [10, 10], [10, 10]]
            visible = [True, True, True, True]
        predictions[method] = {"xy": xy, "visible": visible}
    record = {
        "name": "scene_test", "nframes": 4,
        "affine_matrices_current_to_previous": [
            np.eye(3).tolist(), np.eye(3).tolist(),
            np.eye(3).tolist(), np.eye(3).tolist(),
        ],
        "affine_diagnostics": [{"accepted": True}] * 4,
        "queries": [{
            "query_id": "p0000", "query_time": 0,
            "predictions": predictions,
            "cpu_ms": {method: 1.0 for method in methods},
        }],
    }
    selection_sha = ev.sha_file(out / "dev_selection.json")
    payload = {
        "source_commit": "testsha", "split": "test",
        "manifest_sha256": ev.sha_file(frozen / "manifest.json"),
        "public_query_sha256": ev.sha_file(frozen / "queries-public.json"),
        "dev_selection_sha256": selection_sha,
        "dev_selected_method": "klt_fb",
        "oracle_unread_during_prediction": True,
        "methods": methods, "records": [record],
    }
    prediction_path = out / "test_predictions.json"
    prediction_path.write_text(json.dumps(payload))
    ev.pin(prediction_path)
    result = ev.score("test", {
        "dataset_dir": "data", "output_dir": "out",
        "evaluation": {"point_threshold_px": 8},
    })
    assert result["metrics"]["klt_fb"]["occluded_false_accept_rate"] == 0.0
    assert result["metrics"]["klt_fb"]["jaccard8"] == 1.0
    assert result["metrics"]["stationary_point"]["occluded_false_accept_rate"] == 1.0
    ev.require_pin(out / "test_results.json")


def test_vlm_dev_gate_uses_scored_query_count_and_only_interface_metrics(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    result = {
        "status": "scored", "source_commit": "testsha", "query_count": 10,
        "invalid_rate": 0.1, "coverage": 0.8,
    }
    path = out / "vlm_visibility_dev_results.json"
    path.write_text(json.dumps(result))
    ev.pin(path)
    monkeypatch.setattr(vlm, "ROOT", tmp_path)
    monkeypatch.setattr(ev, "current_sha", lambda: "testsha")
    cfg = {
        "output_dir": "out",
        "vlm_dev_gate": {"max_invalid_rate": 0.2, "min_coverage": 0.6},
    }
    gate = vlm.dev_gate(cfg)
    assert gate["status"] == "passed"
    assert gate["query_count"] == 10
    ev.require_pin(out / "vlm_visibility_dev_gate.json")


def test_vlm_test_is_skipped_when_dev_gate_is_blocked(tmp_path, monkeypatch):
    frozen = tmp_path / "data" / "batch41_frozen"
    out = tmp_path / "out"
    frozen.mkdir(parents=True)
    out.mkdir()
    (frozen / "manifest.json").write_text("{}")
    (frozen / "queries-public.json").write_text(json.dumps({}))
    monkeypatch.setattr(vlm, "ROOT", tmp_path)
    monkeypatch.setattr(ev, "ROOT", tmp_path)
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(ev, "current_sha", lambda: "testsha")
    monkeypatch.setattr(data, "audit", lambda cfg: {"scenes": []})
    gate_path = out / "vlm_visibility_dev_gate.json"
    gate_path.write_text(json.dumps({"status": "blocked", "source_commit": "testsha"}))
    ev.pin(gate_path)
    cfg = {
        "dataset_dir": "data", "output_dir": "out",
        "vlm_model_id": "local-test-model",
    }
    vlm.predict("test", cfg)
    result = json.loads((out / "vlm_visibility_test_predictions.json").read_text())
    assert result["status"] == "skipped_dev_gate"
    assert result["records"] == []
    ev.require_pin(out / "vlm_visibility_test_predictions.json")
