"""Batch39 regressions: RRD projection, depth visibility, group split and oracle firewall."""
from pathlib import Path
import hashlib
import inspect
import json
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import tapvidmv_data as data
import tapvidmv_eval as ev
import tapvidmv_vlm as vlm
import tapvidmv_gate as gate


def intrinsics():
    # Rerun row-vector pinhole convention: [x/z,y/z,1] @ K.
    return np.array([
        [100.0, 0.0, 0.0],
        [0.0, 100.0, 0.0],
        [50.0, 40.0, 1.0],
    ], dtype=np.float64)


def test_world_to_camera_projection_known_pixel():
    xy, depth = data.project_point(
        np.array([1.0, 2.0, 10.0]),
        np.eye(3),
        np.zeros(3),
        intrinsics(),
    )
    np.testing.assert_allclose(xy, [60.0, 60.0], atol=1e-6)
    assert depth == pytest.approx(10.0)


def test_camera_translation_changes_image_projection():
    xy, depth = data.project_point(
        np.array([1.0, 2.0, 10.0]),
        np.eye(3),
        np.array([1.0, 0.0, 0.0]),
        intrinsics(),
    )
    np.testing.assert_allclose(xy, [50.0, 60.0], atol=1e-6)
    assert depth == pytest.approx(10.0)


def test_project_points_matches_single_point_projection():
    points = np.array([[1.0, 2.0, 10.0], [0.0, 0.0, 5.0]])
    pixels, depths = data.project_points(points, np.eye(3), np.zeros(3), intrinsics())
    one, one_depth = data.project_point(points[0], np.eye(3), np.zeros(3), intrinsics())
    np.testing.assert_allclose(pixels[0], one)
    assert depths[0] == pytest.approx(one_depth)


def test_points_behind_camera_are_nonfinite():
    xy, depth = data.project_point(
        np.array([0.0, 0.0, -1.0]), np.eye(3), np.zeros(3), intrinsics()
    )
    assert not np.isfinite(xy).all()
    assert depth == pytest.approx(-1.0)


def test_depth_buffer_keeps_nearest_surface():
    K = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    cloud = np.array([[0.,0.,1.],[0.,0.,2.],[3.,0.,3.]])
    db = data.make_depth_buffer(cloud, (np.eye(3), np.zeros(3)), K, 8, 8)
    assert db[0,0] == pytest.approx(1.0)
    assert db[0,1] == pytest.approx(3.0)


def test_depth_visibility_distinguishes_visible_occluded_out_unknown():
    db = np.full((8,8), np.inf, dtype=np.float32)
    db[2,2] = 1.0
    assert data.classify_depth(np.array([2.,2.]), 1.02, db, 0, .05) == data.STATUS["visible"]
    assert data.classify_depth(np.array([2.,2.]), 1.20, db, 0, .05) == data.STATUS["occluded"]
    assert data.classify_depth(np.array([9.,2.]), 1.0, db, 0, .05) == data.STATUS["out"]
    assert data.classify_depth(np.array([5.,5.]), 1.0, db, 0, .05) == data.STATUS["unknown"]


def test_depth_visibility_uses_exact_neighborhood_and_margin():
    db = np.full((12,12), np.inf, dtype=np.float32)
    db[4,4] = 1.0
    # 3px neighborhood can see the depth sample; 4px cannot.
    assert data.classify_depth(np.array([7.,4.]), 1.02, db, 3, .05) == data.STATUS["visible"]
    assert data.classify_depth(np.array([8.,4.]), 1.02, db, 3, .05) == data.STATUS["unknown"]


def fake_index_records(cfg):
    records = {}
    for item in cfg["sequences"]:
        records[item["sequence"]] = {
            "sequence": item["sequence"], "group": item["group"],
            "views": 3, "frames": item["frames"],
            "tracks_metadata": item["tracks_metadata"], "url": item["url"],
        }
    preflight = cfg["excluded_preflight_sequence"]
    records[preflight] = {
        "sequence": preflight, "group": "AUTOLab", "views": 3,
        "frames": 135, "tracks_metadata": 600, "url": "https://example.invalid/preflight.rrd",
    }
    excluded_group = "tri+7dfa2da3+2023-10-12-13h-13m-53s"
    records[excluded_group] = {
        "sequence": excluded_group, "group": "tri", "views": 3,
        "frames": 163, "tracks_metadata": 599, "url": "https://example.invalid/tri.rrd",
    }
    return records


def test_hash_group_split_is_deterministic_and_disjoint():
    cfg = data.config()
    records = fake_index_records(cfg)
    selected = data.deterministic_selection(records, cfg)
    assert [(x["split"], x["sequence"]) for x in selected] == [
        (x["split"], x["sequence"]) for x in cfg["sequences"]
    ]
    assert len({x["group"] for x in selected}) == 12
    assert [x["split"] for x in selected] == ["dev"] * 4 + ["test"] * 8


def test_preflight_sequence_is_never_selected():
    cfg = data.config()
    selected = data.deterministic_selection(fake_index_records(cfg), cfg)
    assert cfg["excluded_preflight_sequence"] not in {x["sequence"] for x in selected}


def test_group_prefix_never_crosses_dev_test():
    cfg = data.config()
    selected = data.deterministic_selection(fake_index_records(cfg), cfg)
    dev = {x["group"] for x in selected if x["split"] == "dev"}
    test = {x["group"] for x in selected if x["split"] == "test"}
    assert dev.isdisjoint(test)


def test_parse_index_reads_only_droid_table():
    rows = ["<tr><th>sequence</th><th>views</th><th>frames</th><th>tracks</th><th>size</th><th></th></tr>"]
    for i in range(50):
        name = f"LAB{i}+id{i}+2023-01-01"
        rows.append(
            f"<tr><td>{name}</td><td>3</td><td>100</td><td>400</td><td>20 MB</td>"
            f"<td><a href='https://viewer.invalid/{i}'>viewer</a> &middot; "
            f"<a href='https://storage.invalid/{i}.rrd'>download .rrd</a></td></tr>"
        )
    html = "<h2 id='droid'>DROID</h2><table>" + "".join(rows) + "</table><h2 id='egoexo4d'>"
    parsed = data.parse_index(html)
    assert len(parsed) == 50
    assert parsed["LAB7+id7+2023-01-01"]["views"] == 3
    assert parsed["LAB7+id7+2023-01-01"]["url"].endswith("7.rrd")


def test_track_identity_uses_rgb_bits_not_alpha():
    colors = np.array([0x150E06FF, 0x150E06BE, 0x150E06AA], dtype=np.uint32)
    ids = colors >> 8
    assert len(set(ids.tolist())) == 1


def test_public_audit_function_does_not_open_private_oracle():
    source = inspect.getsource(data.audit_public)
    assert "oracle_file" not in source
    assert "oracle-private" not in source
    assert "np.load" in source  # public frames only


def test_projection_and_oracle_are_not_predictor_inputs():
    source = inspect.getsource(ev.predict)
    assert "oracle-private" not in source
    assert "camera_only_xy" not in source
    assert "world_only_xy" not in source
    assert "audit_public" in source


def test_vlm_coordinate_parser_accepts_space_and_comma_forms():
    assert vlm.parse_reply("120 120", 448, 252) == ([120, 120], "valid_xy")
    assert vlm.parse_reply("120,120", 448, 252) == ([120, 120], "valid_xy")
    assert vlm.parse_reply("(x=120, y=120)", 448, 252) == ([120, 120], "valid_xy")
    assert vlm.parse_reply("[120, 120]", 448, 252) == ([120, 120], "valid_xy")


def test_vlm_coordinate_parser_handles_absent_and_rejects_prose():
    assert vlm.parse_reply("ABSENT", 448, 252) == (None, "explicit_absent")
    assert vlm.parse_reply("The point is at 120 120", 448, 252) == (None, "invalid_format")
    assert vlm.parse_reply("500 120", 448, 252) == (None, "out_of_bounds")


def test_vlm_parser_does_not_accept_three_numbers():
    assert vlm.parse_reply("120 120 1", 448, 252) == (None, "invalid_format")


def test_apply_affine_maps_query_point():
    matrix = np.array([[1., 0., 5.], [0., 1., -3.]])
    assert ev.apply_affine(matrix, [10., 20.]) == [15.0, 17.0]
    assert ev.apply_affine(None, [10., 20.]) is None


def test_bootstrap_ci_is_deterministic_and_cluster_counted():
    values = {"a": 0.2, "b": 0.4, "c": 0.6, "d": 0.8}
    one = ev.bootstrap_scene_ci(values, seed=39)
    two = ev.bootstrap_scene_ci(values, seed=39)
    assert one == two
    assert one["n_scenes"] == 4
    assert one["low95"] <= 0.5 <= one["high95"]


def test_save_npz_once_compares_nan_and_string_arrays(tmp_path):
    path = tmp_path / "oracle.npz"
    arrays = {
        "qids": np.array(["q0", "q1"], dtype="U8"),
        "xy": np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32),
    }
    data.save_npz_once(path, **arrays)
    data.save_npz_once(path, **arrays)
    changed = dict(arrays)
    changed["xy"] = np.array([[1.0, np.nan], [2.0, 4.0]], dtype=np.float32)
    with pytest.raises(RuntimeError, match="CONFLICT"):
        data.save_npz_once(path, **changed)


def test_prediction_metrics_keep_unknown_visibility_separate():
    # The scorer must distinguish in-frame projection PCK from depth-visible PCK.
    source = inspect.getsource(ev.score)
    assert 'STATUS["unknown"]' in source
    assert 'STATUS["visible"]' in source
    assert 'STATUS["occluded"]' in source
    assert "all_inframe" in source


def test_vlm_endpoint_uses_only_public_query_and_rgb_files():
    source = inspect.getsource(vlm.predict)
    assert "audit_public" in source
    assert "oracle-private" not in source
    assert "oracle" not in source.lower().replace("oracle_unread_during_prediction", "")


def test_batch39b_uses_balanced_two_queries_per_camera():
    cfg = data.config()
    assert cfg["version"].startswith("2026-11-02-batch39b")
    assert cfg["queries_per_camera"] == 2
    assert len(cfg["sequences"]) == 12
    assert len(cfg["sequences"]) * 3 * cfg["queries_per_camera"] == 72


def test_public_audit_uses_per_camera_resolution_contract():
    source = inspect.getsource(data.audit_public)
    assert 'scene["resolution_by_camera"]' in source
    assert 'scene["resolution"]' not in source


def test_vlm_prompt_respects_nonstandard_camera_height():
    prompt = vlm.prompt_for(448, 251)
    assert "0 to 447" in prompt
    assert "0 to 250" in prompt


def test_vlm_parser_respects_per_camera_height():
    assert vlm.parse_reply("100 250", 448, 251) == ([100, 250], "valid_xy")
    assert vlm.parse_reply("100 251", 448, 251) == (None, "out_of_bounds")


def test_depth_classifier_clamps_rounded_edge_pixel_safely():
    db = np.full((8, 8), np.inf, dtype=np.float32)
    db[7, 7] = 1.0
    assert data.classify_depth(np.array([7.8, 7.8]), 1.01, db, 0, .05) == data.STATUS["visible"]



def test_manifest_errata_corrects_stale_human_readable_query_contract():
    manifest = {
        "query_contract": "hash-ranked frame-0 track IDs; 3 depth-visible query points per camera",
        "scenes": [{"name": "scene_00", "query_count": 6}],
    }
    queries = {"scene_00": [
        {"camera_id": 0}, {"camera_id": 0},
        {"camera_id": 1}, {"camera_id": 1},
        {"camera_id": 2}, {"camera_id": 2},
    ]}
    errata = data.build_manifest_errata(
        manifest, queries, {"queries_per_camera": 2}, "a" * 64, "b" * 64
    )
    assert errata["original_query_contract"].endswith("3 depth-visible query points per camera")
    assert errata["corrected_queries_per_camera"] == 2
    assert errata["corrected_query_contract"].endswith("2 depth-visible query points per camera")
    assert "Original frames, queries and private oracle are unchanged" in errata["reason"]


def test_manifest_errata_rejects_unbalanced_camera_queries():
    manifest = {"scenes": [{"name": "scene_00", "query_count": 6}]}
    queries = {"scene_00": [
        {"camera_id": 0}, {"camera_id": 0}, {"camera_id": 0},
        {"camera_id": 1}, {"camera_id": 1}, {"camera_id": 2},
    ]}
    with pytest.raises(RuntimeError, match="QUERY_COUNT_MISMATCH"):
        data.build_manifest_errata(
            manifest, queries, {"queries_per_camera": 2}, "a" * 64, "b" * 64
        )


def test_prediction_source_sha_refuses_dirty_worktree(monkeypatch):
    class Status:
        returncode = 0
        stdout = " M experiments/tapvidmv_eval.py\\n"

    monkeypatch.setattr(ev.subprocess, "run", lambda *args, **kwargs: Status())
    with pytest.raises(RuntimeError, match="DIRTY_SOURCE_TREE_CANNOT_PIN_PREDICTIONS"):
        ev.gitsha()


def test_batch39c_outputs_are_isolated_from_dirty_batch39b_exploration():
    assert ev.OUT.name == "tapvidmv_droid_batch39c"
    assert vlm.OUT == ev.OUT



def test_dev_gate_uses_only_predeclared_format_validity():
    result = {
        "split": "dev",
        "query_count": 12,
        "metrics": {"vlm_500m_endpoint": {"format_invalid": 1}},
    }
    decision = gate.dev_gate(result)
    assert decision["status"] == "passed"
    assert decision["format_valid"] == 11
    assert decision["selection_basis"].startswith("predeclared output-format")


def test_dev_gate_stops_before_test_when_parser_invalid_rate_is_high():
    result = {
        "split": "dev",
        "query_count": 12,
        "metrics": {"vlm_500m_endpoint": {"format_invalid": 3}},
    }
    with pytest.raises(RuntimeError, match="DEV_GATE_FORMAT_FAILURE"):
        gate.dev_gate(result)


def test_dev_gate_rejects_wrong_split_or_query_count():
    result = {
        "split": "test",
        "query_count": 24,
        "metrics": {"vlm_500m_endpoint": {"format_invalid": 0}},
    }
    with pytest.raises(RuntimeError, match="DEV_GATE_QUERY_SET_MISMATCH"):
        gate.dev_gate(result)
