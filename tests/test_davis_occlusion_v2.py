"""Batch37 prospective regression: identity, point tracking, label boundary, integrity."""
from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import davis_occlusion_eval as m
import davis_real_data as raw
import davis_vlm_visibility as vis


def cfg():
    return json.loads((ROOT / "configs/davis_occlusion_v2.json").read_text())


def synthetic():
    rng = np.random.default_rng(37)
    h, w = 128, 176
    image = rng.integers(20, 210, (h, w, 3), dtype=np.uint8)
    cv2.rectangle(image, (40, 38), (89, 86), (220, 220, 220), -1)
    for y in range(41, 84, 5):
        for x in range(43, 87, 5):
            cv2.circle(image, (x, y), 1, (5, 5, 5), -1)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[38:87, 40:90] = 255
    frames = [cv2.warpAffine(
        image, np.float32([[1, 0, 2*k], [0, 1, k]]),
        (w, h), borderMode=cv2.BORDER_CONSTANT) for k in range(9)]
    return frames, mask, [64., 62.]


def test_config_is_prospective_and_disjoint():
    c = cfg()
    assert (c["dev_scenes"], c["test_scenes"]) == (4, 8)
    assert (c["source_frames"], c["source_stride"]) == (16, 2)
    assert len(set(c["exclude_scenes"])) == 12
    assert set(c["exclude_scenes"]) == {
        "paragliding-launch", "rallye", "loading", "surf", "dogs-jump",
        "dancing", "lady-running", "paragliding", "soccerball", "india",
        "hike", "breakdance-flare"}
    assert c["prohibit_test_selection"] is True
    assert m.METHODS == (
        "stationary_pixel", "klt_forward_backward",
        "klt_fb_consensus", "klt_fb_guarded",
        "klt_fb_appearance_045", "klt_fb_appearance_055",
        "klt_fb_appearance_065")


def test_selection_excludes_prior_scenes_and_short_clips(monkeypatch):
    scene = {f"scene-{i}": {
        "frames": [f"{k:05d}" for k in range(35 if i != 15 else 30)],
        "jpg": {f"{k:05d}": f"im/{i}/{k:05d}.jpg" for k in range(35)},
        "png": {f"{k:05d}": f"ms/{i}/{k:05d}.png" for k in range(35)},
    } for i in range(27)}
    monkeypatch.setattr(raw, "usable_sequences", lambda _: scene)
    config = cfg()
    config["exclude_scenes"] = ["scene-0", "scene-1"]
    result = raw.selection({"entries": {}}, config)
    assert len(result) == 12
    assert set(r["name"] for r in result).isdisjoint(
        {"scene-0", "scene-1", "scene-15"})
    assert all(len(r["members"]) == 16 for r in result)


def test_verify_split_rejects_old_scenes_and_overlap():
    config = cfg()
    rows = [{"name": f"new-{i}", "split": "dev" if i < 4 else "test"}
            for i in range(12)]
    m.verify_split(config, rows)
    with pytest.raises(RuntimeError, match="SCENE_REUSE"):
        m.verify_split(config, rows[:11] + [rows[0]])
    rows[2]["name"] = config["exclude_scenes"][0]
    with pytest.raises(RuntimeError, match="PREVIOUS_B36"):
        m.verify_split(config, rows)


def test_forward_backward_consensus_tracks_image_translation():
    frames, mask, center = synthetic()
    for method in ("klt_fb_consensus", "klt_fb_guarded"):
        path = m.consensus_track(
            frames, mask, center, **cfg()["predeclared_models"][method])
        assert len(path) == len(frames)
        assert path[-1] is not None
        assert np.linalg.norm(
            np.subtract(path[-1], [center[0] + 16, center[1] + 8])) < 4


def test_no_target_texture_causes_abstention():
    frames = [np.full((80, 100, 3), 80, np.uint8) for _ in range(8)]
    mask = np.full((80, 100), 255, np.uint8)
    points = m.consensus_track(
        frames, mask, [50, 40], **cfg()["predeclared_models"]["klt_fb_consensus"])
    assert points[0] == [50, 40]
    assert all(v is None for v in points[1:])


def test_inference_rejects_bad_mask_shape_and_empty_video():
    frames, mask, center = synthetic()
    with pytest.raises(RuntimeError, match="MASK_DIMENSION"):
        m.consensus_track(
            frames, mask[:50], center,
            **cfg()["predeclared_models"]["klt_fb_consensus"])
    with pytest.raises(ValueError, match="EMPTY_VIDEO"):
        m.consensus_track(
            [], mask, center, **cfg()["predeclared_models"]["klt_fb_consensus"])


def test_consensus_gate_rejects_incoherent_feature_motion(monkeypatch):
    frames, mask, center = synthetic()
    init = np.float32([[[40+i, 41+i]] for i in range(10)])
    monkeypatch.setattr(m.cv2, "goodFeaturesToTrack", lambda *a, **kw: init)
    calls = [0]

    def flow(src, dst, prev, *args, **kwargs):
        calls[0] += 1
        points = prev.copy()
        if calls[0] % 2:
            points[:5, 0, 0] += 15
            points[5:, 0, 0] -= 15
        else:
            # back to original -> per-feature FB consistent but no consensus
            points[:5, 0, 0] -= 15
            points[5:, 0, 0] += 15
        return points, np.ones((len(points), 1), np.uint8), None

    monkeypatch.setattr(m.cv2, "calcOpticalFlowPyrLK", flow)
    result = m.consensus_track(
        frames, mask, center, **cfg()["predeclared_models"]["klt_fb_consensus"])
    assert result[1:] == [None] * (len(frames) - 1)


def test_absence_and_mask_shrink_scored_as_distinct_conditions():
    gold = {"v": {"points": [
        {"visible": True, "center": [0, 0], "pixels": 100},
        {"visible": True, "center": [5, 0], "pixels": 25},
        {"visible": False, "center": None, "pixels": 0},
        {"visible": True, "center": [15, 0], "pixels": 70},
    ]}}
    records = {"v": {"trajectories": {"guard": [
        [0, 0], [30, 0], [10, 0], None]}}}
    result = m.occlusion_proxies(
        gold, [{"name": "v"}], records, "guard", .4)
    assert result["absent_positions"] == 1
    assert result["absent_false_acceptances"] == 1
    assert result["mask_area_shrink_positions_not_verified_occlusions"] == 1
    assert result["mask_area_shrink_wrong_accepted"] == 1
    assert result["visible_correct_accepted"] == 0
    assert result["visible_wrong_accepted"] == 1
    # Regression for Batch37 V5: NumPy booleans in += used to create int64,
    # silently passing numeric assertions but crashing JSON report export.
    assert json.loads(json.dumps(result)) == result
    assert all(type(value) is int for value in result.values())


def test_prediction_path_cannot_read_any_future_oracle(monkeypatch, tmp_path):
    config = cfg()
    config["dataset_dir"] = "data/sim"
    config["output_dir"] = "outputs/sim"
    root = tmp_path
    data_path = root / "data" / "sim"
    data_path.mkdir(parents=True)
    (data_path / "manifest.json").write_text('{"synthetic":"no future oracle"}')
    rows = [{"name": f"new-{i}", "split": "dev" if i < 4 else "test"}
            for i in range(12)]
    queries = {r["name"]: {"center": [0, 0], "query_mask": "initial.png"}
               for r in rows}
    (data_path / "initial_queries.json").write_text(json.dumps(queries))
    (root / config["output_dir"]).mkdir(parents=True)
    (root / config["output_dir"] / "dev_selection.json").write_text(json.dumps({
        "source_commit": "frozen-code", "selected_method": "klt_fb_guarded"
    }))
    monkeypatch.setattr(m, "ROOT", root)
    monkeypatch.setattr(m, "current_sha", lambda: "frozen-code")
    monkeypatch.setattr(m.raw, "check", lambda cfg, include_future_masks_and_gold: (
        {"rows": rows} if include_future_masks_and_gold is False else
        (_ for _ in ()).throw(RuntimeError("FUTURE_ORACLE_READ"))))
    monkeypatch.setattr(m, "scene_predict", lambda *args: {
        "name": args[1]["name"], "trajectories": {}, "cpu_method_ms": {}})
    m.predict(config, "test")
    output = root / config["output_dir"] / "test_predictions.json"
    assert output.is_file()
    m.base.require_pinned_prediction(output)
    assert len(json.loads(output.read_text())["records"]) == 8
    m.predict(config, "test")  # idempotent if exact frozen version
    output.write_text(output.read_text() + " ")
    with pytest.raises(RuntimeError, match="PREDICTION_OR_REPORT|PREDICTION_SHA"):
        m.base.require_pinned_prediction(output)


def test_wrong_source_hash_or_method_refuses_existing(monkeypatch, tmp_path):
    config = cfg()
    config["dataset_dir"] = "data/sim"
    config["output_dir"] = "outputs/sim"
    data_path = tmp_path / "data" / "sim"
    out_path = tmp_path / "outputs" / "sim"
    data_path.mkdir(parents=True)
    out_path.mkdir(parents=True)
    (data_path / "manifest.json").write_text("{}")
    (out_path / "test_predictions.json").write_text('{"methods": ["old"]}')
    (out_path / "dev_selection.json").write_text(json.dumps({
        "source_commit": "frozen-source", "selected_method": "klt_fb_guarded"
    }))
    rows = [{"name": f"new-{i}", "split": "dev" if i < 4 else "test"}
            for i in range(12)]
    monkeypatch.setattr(m, "current_sha", lambda: "frozen-source")
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m.raw, "check", lambda *args, **kwargs: {"rows": rows})
    with pytest.raises(RuntimeError, match="CODE_OR_DATA_DRIFT"):
        m.predict(config, "test")


def test_cpu_only_and_initial_query_boundary_on_main_predictor(monkeypatch):
    assert "cuda" not in m.consensus_track.__code__.co_names
    assert "gold-private.json" not in m.scene_predict.__code__.co_consts
    assert "gold-private.json" in m.score.__code__.co_consts


def test_new_config_is_valid_json_and_frozen_signature():
    data = cfg()
    assert data["shrink_proxy_threshold"] == 0.4
    assert data["initial_mask_only"] and not data["world_relative_gold"]
    assert data["predeclared_models"]["klt_fb_guarded"]["min_features"] == 9
    assert data["predeclared_models"]["klt_fb_consensus"]["min_features"] == 7



def cfg_v3():
    return json.loads((ROOT / "configs/davis_occlusion_v3.json").read_text())


def test_patch_ncc_is_robust_to_brightness_offset_but_rejects_replacement():
    rng = np.random.default_rng(401)
    a = np.zeros((80, 100), np.uint8)
    b = np.zeros_like(a)
    c = np.zeros_like(a)
    patch = rng.integers(20, 220, (21, 21), dtype=np.uint8)
    a[29:50, 39:60] = patch
    b[29:50, 39:60] = np.clip(patch.astype(np.float32) * 0.65 + 42, 0, 255).astype(np.uint8)
    c[29:50, 39:60] = rng.integers(20, 220, (21, 21), dtype=np.uint8)
    same = m.patch_ncc(a, b, np.array([49., 39.]), np.array([49., 39.]), 11)
    different = m.patch_ncc(a, c, np.array([49., 39.]), np.array([49., 39.]), 11)
    assert same > 0.98
    assert different < 0.5


def test_appearance_consensus_abstains_after_full_target_patch_replacement():
    frames, mask, center = synthetic()
    rng = np.random.default_rng(402)
    for k in range(4, len(frames)):
        x0, y0 = 40 + 2*k, 38 + k
        replacement = rng.integers(0, 255, (49, 50, 3), dtype=np.uint8)
        frames[k][y0:y0+49, x0:x0+50] = replacement
    result = m.consensus_track(
        frames, mask, center,
        min_features=7, fb_error_px=1.5, max_step_px=45.0,
        inlier_residual_px=4.0, min_consensus_ratio=0.55,
        patch_ncc_min=0.65, patch_size=11,
    )
    assert result[1] is not None
    assert result[4] is None
    assert all(point is None for point in result[4:])


def test_dirty_source_tree_cannot_be_pinned(monkeypatch):
    class Status:
        returncode = 0
        stdout = " M experiments/davis_occlusion_eval.py\\n"
    monkeypatch.setattr(m.subprocess, "run", lambda *args, **kwargs: Status())
    with pytest.raises(RuntimeError, match="DIRTY_SOURCE_TREE_CANNOT_PIN"):
        m.current_sha()


def test_dev_selection_uses_only_predeclared_dev_candidates(monkeypatch, tmp_path):
    config = cfg_v3()
    config["output_dir"] = "outputs/sim"
    out = tmp_path / "outputs" / "sim"
    out.mkdir(parents=True)
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "current_sha", lambda: "frozen-source")
    result = {
        "split": "dev", "source_commit": "frozen-source",
        "methods": {
            "klt_fb_appearance_045": {
                "visible_pck8_unconditional": 0.70, "visible_accepted": 8,
                "visible_evaluations": 10, "invisible_evaluations": 4,
                "absent_false_accepted": 2,
            },
            "klt_fb_appearance_055": {
                "visible_pck8_unconditional": 0.65, "visible_accepted": 8,
                "visible_evaluations": 10, "invisible_evaluations": 4,
                "absent_false_accepted": 1,
            },
            "klt_fb_appearance_065": {
                "visible_pck8_unconditional": 0.60, "visible_accepted": 9,
                "visible_evaluations": 10, "invisible_evaluations": 4,
                "absent_false_accepted": 0,
            },
        },
    }
    (out / "dev_results.json").write_text(json.dumps(result))
    selected = m.select_dev(config)
    assert selected["selected_method"] == "klt_fb_appearance_055"
    assert selected["eligible_methods"] == [
        "klt_fb_appearance_055", "klt_fb_appearance_065"
    ]
    assert selected["test_labels_read"] is False


def test_vlm_visibility_parser_and_prompt_are_binary_with_abstention():
    assert vis.parse_visibility("VISIBLE") == ("VISIBLE", "valid_visible")
    assert vis.parse_visibility("The object is not visible.") == ("ABSENT", "valid_absent")
    assert vis.parse_visibility("ABSENT") == ("ABSENT", "valid_absent")
    assert vis.parse_visibility("UNCERTAIN") == ("UNCERTAIN", "valid_uncertain")
    assert vis.parse_visibility("visible or absent") == ("ABSENT", "valid_absent")
    assert vis.parse_visibility("maybe it is there") == (None, "invalid_format")
    assert "VISIBLE, ABSENT, or UNCERTAIN" in vis.PROMPT
    assert "pixel coordinates" not in vis.PROMPT.lower()


def test_vlm_reference_overlay_uses_only_the_initial_mask(tmp_path):
    frame = np.zeros((48, 64, 3), np.uint8)
    mask = np.zeros((48, 64), np.uint8)
    mask[10:30, 15:35] = 255
    path = tmp_path / "mask.png"
    assert cv2.imwrite(str(path), mask)
    result = np.asarray(vis.highlighted_reference(frame, path))
    assert result[20, 25, 0] > result[20, 25, 1]
    assert np.array_equal(result[2, 2], np.array([0, 0, 0], dtype=np.uint8))


def test_vlm_visibility_dev_gate_checks_format_and_coverage(monkeypatch, tmp_path):
    monkeypatch.setattr(vis, "OUT", tmp_path)
    monkeypatch.setattr(vis, "source_sha", lambda: "frozen-source")
    result = {
        "split": "dev", "source_commit": "frozen-source",
        "query_count": 16, "invalid": 1, "coverage": 0.75,
    }
    (tmp_path / "vlm_visibility_dev_results.json").write_text(json.dumps(result))
    gate = vis.dev_gate(cfg_v3())
    assert gate["status"] == "passed"
    assert gate["selection_basis"].startswith("format and decision coverage")
    (tmp_path / "vlm_visibility_dev_results.json").write_text(json.dumps({
        "split": "dev", "source_commit": "frozen-source",
        "query_count": 16, "invalid": 5, "coverage": 0.75,
    }))
    with pytest.raises(RuntimeError, match="INTERFACE_GATE_FAILED"):
        vis.dev_gate(cfg_v3())


def test_vlm_visibility_predictor_does_not_reference_future_private_masks():
    assert "gold-private.json" not in vis.predict.__code__.co_consts
    assert "gold-private.json" in vis.score.__code__.co_consts



def test_v3_protocol_predeclares_appearance_sweep_and_vlm_visibility_endpoints():
    config = cfg_v3()
    assert config["dataset_dir"] == "data/davis_occlusion_v3"
    assert (config["dev_scenes"], config["test_scenes"]) == (4, 8)
    assert config["vlm_target_frame_indices"] == [3, 7, 11, 15]
    models = config["predeclared_models"]
    assert [models[name]["patch_ncc_min"] for name in (
        "klt_fb_appearance_045", "klt_fb_appearance_055",
        "klt_fb_appearance_065")] == [0.45, 0.55, 0.65]
    assert config["dev_selection"]["max_absent_false_accept_rate"] == 0.35
    assert config["prohibit_test_selection"] is True
