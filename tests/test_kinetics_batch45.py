from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import kinetics_batch45_data as data
import kinetics_batch45_eval as ev
import kinetics_batch45_vlm as vlm


def test_deterministic_selection_is_video_and_class_disjoint():
    labels = [f"class{i:03d}" for i in range(700)]
    candidates = {
        label: [{
            "video_id": f"video-{i:03d}-{j}",
            "label": label,
            "time_start": "1",
            "time_end": "11",
            "track_rows": 30,
            "query_candidates": 30,
        } for j in range(2)]
        for i, label in enumerate(labels[:30])
    }
    path_lines = [
        f"https://s3.amazonaws.com/kinetics/700_2020/val/k700_val_{i:03d}.tar.gz"
        for i in range(1, 701)
    ]
    selected = data.deterministic_class_selection(
        candidates,
        labels,
        {"video-000-0"},
        {"class000"},
        path_lines,
        {
            "source_csv_sha256": "a" * 64,
            "kinetics_val_metadata_sha256": "b" * 64,
            "val_ids_sha256": "c" * 64,
            "path_list_sha256": "d" * 64,
        },
    )
    rows = selected["videos"]
    assert len(rows) == 12
    assert [x["split"] for x in rows] == ["dev"] * 6 + ["test"] * 6
    assert len({x["video_id"] for x in rows}) == 12
    assert len({x["label"] for x in rows}) == 12
    assert "class000" not in {x["label"] for x in rows}
    assert "video-000-0" not in {x["video_id"] for x in rows}
    assert all(x["shard_index"] == labels.index(x["label"]) + 1 for x in rows)


def test_normalized_coordinate_transform_uses_half_pixel_correction():
    point = np.asarray([[0.5, 0.5]], dtype=np.float64)
    result = data.normalized_to_resized_pixels(point, 640, 480)
    assert np.allclose(result[0], [127.8, 127.7333333333], atol=1e-6)


def _pattern():
    image = np.zeros((64, 64), dtype=np.uint8)
    cv2.rectangle(image, (12, 10), (38, 42), 90, -1)
    cv2.line(image, (14, 40), (52, 16), 180, 3)
    cv2.circle(image, (45, 48), 7, 130, -1)
    return image


def test_gradient_descriptor_is_invariant_to_brightness_and_positive_gain():
    a = _pattern()
    b = np.clip(a.astype(np.float32) * 0.7 + 35.0, 0, 255).astype(np.uint8)
    similarity = ev.gradient_similarity(a, b)
    assert similarity is not None
    assert similarity > 0.995


def test_gradient_gate_abstains_on_textureless_candidate():
    first = _pattern()
    second = np.zeros_like(first)
    query = {"query_time": 0, "query_xy": [32.0, 32.0]}
    trace, scores = ev.gradient_gate(
        [first, second], query, [[32.0, 32.0], [32.0, 32.0]], 0.5, 21
    )
    assert trace[0] == [32.0, 32.0]
    assert trace[1] is None
    assert scores[1] is None


def test_gradient_gate_accepts_same_point_under_brightness_shift():
    first = _pattern()
    second = np.clip(first.astype(np.int16) + 28, 0, 255).astype(np.uint8)
    query = {"query_time": 0, "query_xy": [32.0, 32.0]}
    trace, scores = ev.gradient_gate(
        [first, second], query, [[32.0, 32.0], [32.0, 32.0]], 0.5, 21
    )
    assert trace[1] == [32.0, 32.0]
    assert scores[1] is not None and scores[1] > 0.95


def test_vlm_parser_accepts_only_one_unambiguous_label():
    assert vlm.parse_label("VISIBLE.") == ("VISIBLE", "valid")
    assert vlm.parse_label("NOT_VISIBLE") == ("NOT_VISIBLE", "valid")
    assert vlm.parse_label("not visible") == ("NOT_VISIBLE", "valid")
    assert vlm.parse_label("UNCERTAIN\n") == ("UNCERTAIN", "valid")
    assert vlm.parse_label("VISIBLE and NOT_VISIBLE") == ("UNCERTAIN", "invalid_format")
    assert vlm.parse_label("probably visible") == ("UNCERTAIN", "invalid_format")
    assert vlm.parse_label("I think VISIBLE") == ("UNCERTAIN", "invalid_format")
    assert vlm.parse_label("VISIBLE, maybe") == ("UNCERTAIN", "invalid_format")


def test_vlm_reference_crop_is_square_and_edge_safe():
    frame = np.zeros((40, 50, 3), dtype=np.uint8)
    crop = vlm.reference_crop(frame, [1.0, 1.0], 64)
    assert crop.size == (64, 64)


def test_vlm_visibility_gate_metrics_expose_all_visible_collapse():
    result = vlm.summarize_binary(
        ["VISIBLE"] * 12,
        [True] * 8 + [False] * 4,
        ["valid"] * 12,
    )
    assert result["balanced_accuracy"] == 0.5
    assert result["not_visible_false_accept_rate"] == 1.0
    assert result["coverage"] == 1.0


def test_archive_member_path_traversal_is_rejected_before_archive_access(tmp_path):
    with pytest.raises(RuntimeError, match="KINETICS_ARCHIVE_MEMBER_PATH_TRAVERSAL"):
        data.extract_member(
            {"archive_member": "../outside.mp4", "shard_index": 999},
            tmp_path / "outside.mp4",
        )


def test_eval_method_set_contains_photometric_and_gradient_gates():
    assert "klt_fb" in ev.METHODS
    assert "appearance_fb_070" in ev.METHODS
    assert "gradient_fb_050" in ev.METHODS
    assert "gradient_fb_065" in ev.METHODS
    assert "gradient_fb_080" in ev.METHODS


def test_manifest_provenance_pins_all_upstream_hashes(tmp_path, monkeypatch):
    path_list = tmp_path / "path-list.txt"
    download_manifest = tmp_path / "download-manifest.json"
    path_list.write_text("official-shard-list\\n", encoding="utf-8")
    download_manifest.write_text('{"shards":{}}\\n', encoding="utf-8")
    monkeypatch.setattr(data, "PATH_LIST", path_list)
    monkeypatch.setattr(data, "DOWNLOAD_MANIFEST", download_manifest)
    selection = {"previous_video_ids_sha256": "e" * 64}
    provenance = data.manifest_provenance(selection, "frozen-source-sha")
    assert provenance["path_list_sha256"] == data.sha_file(path_list)
    assert provenance["download_manifest_sha256"] == data.sha_file(download_manifest)
    assert provenance["previous_video_ids_sha256"] == "e" * 64
    assert provenance["source_commit"] == "frozen-source-sha"


def test_split_disjointness_rejects_prior_video_and_action_class():
    scenes = [
        {"video_id": "fresh-a", "action_class": "fresh-class-a"},
        {"video_id": "fresh-b", "action_class": "fresh-class-b"},
    ]
    assert data.validate_split_disjointness(scenes, {"old-video"}, {"old-class"})
    with pytest.raises(RuntimeError, match="KINETICS_PREVIOUS_VIDEO_ID_OVERLAP"):
        data.validate_split_disjointness(scenes, {"fresh-a"}, set())
    with pytest.raises(RuntimeError, match="KINETICS_BATCH42_ACTION_CLASS_OVERLAP"):
        data.validate_split_disjointness(scenes, set(), {"fresh-class-b"})
