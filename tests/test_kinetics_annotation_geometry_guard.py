"""Generic annotation-schema repair tested on synthetic rows, never TEST labels."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import kinetics_batch45_data as data


def row(*, bad_visible=None, bad_occluded=None, bad_flag=None):
    triples = []
    for index in range(250):
        x, y, occluded = 0.4, 0.55, 0.0
        if bad_visible is not None and index == 2:
            x = bad_visible
        if bad_occluded is not None and index == 7:
            x, occluded = bad_occluded, 1.0
        if bad_flag is not None and index == 11:
            occluded = bad_flag
        triples.extend([str(x), str(y), str(occluded)])
    return ["synthetic", "0", "10", *triples]


def test_normalized_annotation_keeps_unmodified_visible_points():
    result = data.usable_track_for_freeze(row(), 250, 1280, 720)
    assert result is not None
    xy, occ, xy_norm = result
    assert xy.shape == (250, 2)
    assert occ.shape == (250,)
    assert xy_norm.shape == (250, 2)
    assert not occ.any()
    assert xy_norm[0, 0] == pytest.approx(0.4)
    assert xy[0, 0] == pytest.approx(0.4 * 256 - 0.1)


@pytest.mark.parametrize("invalid_x", [-0.001, 1.001, -100, 100])
def test_outside_visible_image_is_excluded_not_clipped(invalid_x):
    original = row(bad_visible=invalid_x)
    assert data.usable_track_for_freeze(original, 250, 1280, 720) is None
    assert float(original[3 + 3 * 2]) == invalid_x


@pytest.mark.parametrize("invisible_x", [-1000, -0.01, 1.001, 2000])
def test_offscreen_occluded_coordinates_are_not_falsely_rejected(invisible_x):
    result = data.usable_track_for_freeze(row(bad_occluded=invisible_x), 250, 1280, 720)
    assert result is not None
    xy, occ, norm = result
    assert bool(occ[7])
    assert norm[7, 0] == invisible_x


@pytest.mark.parametrize("bad_flag", [0.2, -1, 2])
def test_bad_occlusion_values_remain_fatal(bad_flag):
    with pytest.raises(RuntimeError, match="BAD_OCCLUSION"):
        data.usable_track_for_freeze(row(bad_flag=bad_flag), 250, 1280, 720)


def test_nonfinite_annotation_remains_fatal():
    with pytest.raises(RuntimeError, match="NONFINITE"):
        data.usable_track_for_freeze(row(bad_visible=float("nan")), 250, 1280, 720)


@pytest.mark.parametrize("frame_count", [0, -1, 251])
def test_invalid_decode_frame_count_fails_closed(frame_count):
    with pytest.raises(ValueError, match="INVALID_DECODED_FRAME_COUNT"):
        data.usable_track_for_freeze(row(), frame_count, 1280, 720)


def test_resized_frame_edge_geometry_is_checked_independently():
    # Normalized (0,0) is outside the allowed -0.5 pixel tolerance
    # if the original raster was smaller than 256.
    original = row()
    original[3] = "0.0"
    assert data.usable_track_for_freeze(original, 250, 120, 120) is None
    assert data.usable_track_for_freeze(original, 250, 1280, 720) is not None


def test_wrong_row_width_is_not_silently_filtered():
    with pytest.raises(ValueError):
        data.usable_track_for_freeze(row()[:-3], 250, 1280, 720)


def test_query_protocol_keeps_deterministic_original_track_id():
    result = data.usable_track_for_freeze(row(), 250, 1280, 720)
    assert result is not None
    _, occ, norm = result
    query = data.first_visible_query(17, norm, occ, 1280, 720)
    assert query["id"] == "p0017"
    assert query["query_protocol"] == "official_first_visible_point"
