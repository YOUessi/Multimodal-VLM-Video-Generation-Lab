"""DEV-only and image-plane-vs-camera-motion scientific boundary checks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import dev_affine_evidence_audit as audit


def _rows():
    return [
        dict(split="dev", video_id=f"dev_{i}", label=f"action_{i}",
             archive_member=f"action_{i}/sample_{i}.mp4", shard_index=i)
        for i in range(6)
    ]


def _camera_and_foreground():
    rng = np.random.default_rng(711)
    background = np.column_stack([
        rng.uniform(10, 630, 160),
        rng.uniform(10, 350, 160),
    ]).astype(np.float32)
    foreground = np.column_stack([
        rng.uniform(260, 330, 30),
        rng.uniform(140, 190, 30),
    ]).astype(np.float32)
    src = np.concatenate([background, foreground])
    dst = src + [7, -3]
    dst[len(background):] += [15, 6]  # independent foreground displacement
    return src, dst


def test_dominant_affine_recovers_background_relative_not_3d_motion():
    src, dst = _camera_and_foreground()
    result = audit.affine_evidence(src, dst, 640, 360)
    assert result["status"] == "ok"
    assert result["inlier_fraction"] > 0.75
    assert result["spatial_support_fraction_4x4"] >= 0.75
    assert abs(result["affine_2x3"][0][2] - 7) < 0.1
    assert abs(result["affine_2x3"][1][2] + 3) < 0.1
    assert result["median_affine_residual_inliers_px"] < 0.2
    assert result["median_affine_residual_outliers_px"] > 10
    assert result["interpretation"] == "image_plane_only_not_physical_camera_motion"


def test_localized_coherent_foreground_has_weak_spatial_support():
    rng = np.random.default_rng(101)
    src = np.column_stack([
        rng.uniform(5, 70, 35), rng.uniform(5, 70, 35)
    ]).astype(np.float32)
    dst = src + [13, 5]
    result = audit.affine_evidence(src, dst, 640, 360)
    assert result["status"] == "ok"
    assert result["inlier_fraction"] > 0.95
    assert result["spatial_support_fraction_4x4"] <= 2 / 16  # false confidence risk


def test_empty_or_insufficient_correspondences_are_not_motion_estimates():
    result = audit.affine_evidence(np.zeros((6, 2)), np.zeros((6, 2)), 640, 360)
    assert result == {"status": "too_few_correspondences", "match_count": 6}


@pytest.mark.parametrize("bad", [
    np.zeros((13, 3), dtype=np.float32),
    np.zeros((4, 2), dtype=np.float32).reshape(2, 2, 2),
])
def test_malformed_correspondences_fail_closed(bad):
    with pytest.raises(ValueError, match="POINT_SHAPE_INVALID"):
        audit.affine_evidence(bad, bad, 640, 360)


def test_nan_motion_fails_closed():
    src, dst = _camera_and_foreground()
    dst[0, 1] = float("nan")
    with pytest.raises(ValueError, match="NONFINITE_POINTS"):
        audit.affine_evidence(src, dst, 640, 360)


def test_dev_rows_do_not_include_test_video_ids():
    doc = {"videos": _rows() + [
        dict(split="test", video_id="secret_test", label="unused",
             shard_index=999, archive_member="secret_test.mp4")
    ]}
    assert [row["video_id"] for row in audit.dev_rows(doc)] == [
        f"dev_{i}" for i in range(6)
    ]


def test_six_dev_rows_required_without_test_fallback():
    with pytest.raises(ValueError, match="REQUIRES_SIX_DEV_CLIPS"):
        audit.dev_rows({"videos": _rows()[:5] + [
            dict(split="test", video_id="t0", shard_index=23, archive_member="t.mp4")
        ]})


def test_dev_only_run_never_reads_test_shard_or_private_oracle(tmp_path, monkeypatch):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"videos": _rows() + [
        dict(split="test", video_id="sealed_test", archive_member="sealed.mp4",
             label="sealed", shard_index=999),
    ]}))
    shards = tmp_path / "shards"
    shards.mkdir()
    for i in range(6):
        (shards / f"k700_val_{i:03d}.tar.gz").write_bytes(b"dummy_for_mock")
    called = []

    def fake_analyze(row, shard_dir, sample_frames, width):
        called.append(row["video_id"])
        return {"video_id": row["video_id"], "summary": {}}

    monkeypatch.setattr(audit, "analyze_dev_clip", fake_analyze)
    output = tmp_path / "report.json"
    result = audit.run(selection, shards, output, sample_frames=11)
    assert len(called) == 6
    assert all(name.startswith("dev_") for name in called)
    assert result["scope"].startswith("DEV_ONLY")
    assert result["selection_sha256"] == audit.selection_sha256(selection)
    assert "sealed_test" not in output.read_text()
    assert not (shards / "k700_val_999.tar.gz").exists()
    # Pin-by-content without overwriting any prior diagnostic.
    audit.run(selection, shards, output, sample_frames=11)
    with pytest.raises(RuntimeError, match="EXISTING_REPORT_DIFFERS"):
        audit.run(selection, shards, output, sample_frames=5)


def test_dev_missing_shard_rejected_before_any_processing(tmp_path, monkeypatch):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"videos": _rows()}))
    shards = tmp_path / "shards"
    shards.mkdir()
    monkeypatch.setattr(audit, "analyze_dev_clip", lambda *args: pytest.fail(
        "Must not begin decoding until all six DEV shards exist."
    ))
    with pytest.raises(FileNotFoundError, match="DEV_SHARDS_MISSING"):
        audit.run(selection, shards, tmp_path / "out.json")


def test_synthetic_moving_camera_image_klt_registration():
    rng = np.random.default_rng(44)
    frame = rng.integers(0, 256, size=(240, 320), dtype=np.uint8)
    matrix = np.float32([[1, 0, 5], [0, 1, -2]])
    shifted = cv2.warpAffine(frame, matrix, (320, 240))
    report = audit.klt_pair(frame, shifted)
    assert report["status"] == "ok"
    assert report["match_count"] > 50
    assert abs(report["affine_2x3"][0][2] - 5) < 0.8
    assert abs(report["affine_2x3"][1][2] + 2) < 0.8
    assert report["median_affine_residual_inliers_px"] < 0.5
