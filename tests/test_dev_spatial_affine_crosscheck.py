"""Synthetic controls and no-TEST guard for spatially cross-fitted affine diagnostics."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import dev_spatial_affine_crosscheck as cvaudit


def four_quadrant_points(seed=9, per_cell=60):
    rng = np.random.default_rng(seed)
    p = []
    for y0 in [5, 185]:
        for x0 in [5, 325]:
            p.append(np.stack([
                rng.uniform(x0, x0 + 290, per_cell),
                rng.uniform(y0, y0 + 165, per_cell)
            ], axis=1))
    return np.concatenate(p).astype(np.float32)


def test_true_dominant_2d_translation_transfers_to_all_quadrants():
    src = four_quadrant_points()
    dst = src + [7, -3]
    result = cvaudit.crossfit_affine(src, dst, 640, 360)
    assert result["status"] == "ok"
    assert result["valid_folds"] == 4
    assert result["crossfit_error_median_px"] < 0.05
    assert result["crossfit_worst_quadrant_median_px"] < 0.05
    assert result["mean_heldout_fraction_within_3px"] == 1


def test_independent_foreground_in_one_quadrant_fails_out_of_region_transfer():
    src = four_quadrant_points()
    dst = src + [7, -3]
    # One coherent foreground layer replaces a quadrant-wide portion of matches.
    dst[:60] += [22, 8]
    result = cvaudit.crossfit_affine(src, dst, 640, 360)
    assert result["status"] == "ok"
    assert result["valid_folds"] == 4
    assert result["crossfit_worst_quadrant_median_px"] > 15
    assert result["mean_heldout_fraction_within_3px"] < 0.85


def test_locally_concentrated_motion_does_not_claim_global_validation():
    rng = np.random.default_rng(17)
    src = rng.uniform([5, 5], [140, 150], size=(80, 2)).astype(np.float32)
    dst = src + [7, 1]
    result = cvaudit.crossfit_affine(src, dst, 640, 360)
    assert result["status"] == "insufficient_valid_spatial_folds"
    assert result["valid_folds"] == 0
    assert all(row["status"] == "insufficient_spatial_support" for row in result["folds"])


def test_insufficient_matches_fail_closed():
    src = np.zeros((11, 2), dtype=np.float32)
    result = cvaudit.crossfit_affine(src, src, 640, 360)
    assert result["status"] == "insufficient_total_correspondences"


@pytest.mark.parametrize("src,dst", [
    (np.zeros((60, 3)), np.zeros((60, 3))),
    (np.zeros((60, 2)), np.zeros((59, 2))),
    (np.zeros((2, 2, 2)), np.zeros((2, 2, 2))),
])
def test_invalid_point_shapes_fail_closed(src, dst):
    with pytest.raises(ValueError, match="POINT_SHAPE_INVALID"):
        cvaudit.crossfit_affine(src, dst, 640, 360)


def test_nonfinite_or_out_of_frame_points_rejected():
    src = four_quadrant_points()
    dst = src.copy()
    src[1, 1] = np.nan
    with pytest.raises(ValueError, match="NONFINITE"):
        cvaudit.crossfit_affine(src, dst, 640, 360)
    src = four_quadrant_points()
    src[1, 1] = 360
    with pytest.raises(ValueError, match="OUTSIDE_IMAGE"):
        cvaudit.crossfit_affine(src, dst, 640, 360)


def test_klt_translation_uses_only_image_plane_correspondences():
    rng = np.random.default_rng(44)
    frame = rng.integers(0, 256, size=(240, 320), dtype=np.uint8)
    shifted = cv2.warpAffine(frame, np.float32([[1, 0, 5], [0, 1, -2]]), (320, 240))
    p, q = cvaudit.klt_matches(frame, shifted)
    assert len(p) > 70
    result = cvaudit.crossfit_affine(p, q, 320, 240)
    assert result["status"] == "ok"
    assert result["crossfit_worst_quadrant_median_px"] < 1.5
    assert result["mean_heldout_fraction_within_3px"] > 0.9


def test_run_never_opens_test_shards_or_private_oracle(tmp_path, monkeypatch):
    selection = tmp_path / "selection.json"
    videos = [
        dict(split="dev", video_id=f"dev{i}", label=f"action{i}",
             archive_member=f"a/{i}.mp4", shard_index=i)
        for i in range(6)
    ] + [
        dict(split="test", video_id="test_secret", label="test_secret",
             archive_member="secret.mp4", shard_index=999)
    ]
    selection.write_text(json.dumps({"videos": videos}))
    shards = tmp_path / "shards"
    shards.mkdir()
    for i in range(6):
        (shards / f"k700_val_{i:03d}.tar.gz").write_bytes(b"mock-only")

    observed = []
    def mock_dev(row, *_):
        observed.append(row["video_id"])
        return {"action_class": row["label"], "valid_crossfit_pairs": 0}

    monkeypatch.setattr(cvaudit, "analyze_one_dev_clip", mock_dev)
    report = cvaudit.run(selection, shards, tmp_path / "result.json")
    assert observed == [f"dev{i}" for i in range(6)]
    assert "test_secret" not in json.dumps(report)
    assert not (shards / "k700_val_999.tar.gz").exists()
    assert report["scope"].startswith("DEV_ONLY")
    # Immutable report output: same content permitted, changed content refused.
    cvaudit.run(selection, shards, tmp_path / "result.json")
    with pytest.raises(RuntimeError, match="ALREADY_EXISTS_DIFFERENT"):
        cvaudit.run(selection, shards, tmp_path / "result.json", width=640)


def test_missing_dev_shard_rejected_without_starting_media_processing(tmp_path, monkeypatch):
    rows = [dict(split="dev", video_id=f"d{i}", label=f"a{i}",
                 archive_member=f"{i}.mp4", shard_index=i) for i in range(6)]
    s = tmp_path / "selection.json"
    s.write_text(json.dumps({"videos": rows}))
    p = tmp_path / "shards"
    p.mkdir()
    monkeypatch.setattr(cvaudit, "analyze_one_dev_clip", lambda *args: pytest.fail(
        "Must reject before decoding any DEV media."
    ))
    with pytest.raises(FileNotFoundError, match="MISSING_DEV_SHARDS"):
        cvaudit.run(s, p, tmp_path / "result.json")
