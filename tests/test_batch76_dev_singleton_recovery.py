"""Batch76 unit/contract tests: causal public-only predictions, identity ambiguity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from experiments import batch76_dev_singleton_recovery as b


def scene(mode="identity", n=28):
    rng = np.random.default_rng(76011)
    patch = rng.integers(25, 230, (b.PATCH, b.PATCH), dtype=np.uint8)
    rgb = np.zeros((n, 100, 100, 3), dtype=np.uint8)
    for t in range(n):
        rgb[t, ...] = 33
        if t == 0:
            rgb[t, 15:32, 15:32, :] = patch[:, :, None]
        elif mode in ("identity", "impostor") and t >= 4:
            rgb[t, 65:82, 65:82, :] = patch[:, :, None]
        elif mode == "twins" and t >= 4:
            rgb[t, 65:82, 65:82, :] = patch[:, :, None]
            rgb[t, 15:32, 15:32, :] = patch[:, :, None]
    return rgb


def test_search_finds_unique_patch():
    arr = scene()
    ref = cv2.cvtColor(arr[0], cv2.COLOR_RGB2GRAY)[15:32, 15:32]
    point = b.find_singleton(cv2.cvtColor(arr[19], cv2.COLOR_RGB2GRAY), ref)
    assert point == [73.0, 73.0]


def test_search_does_not_accept_twins():
    arr = scene("twins")
    ref = cv2.cvtColor(arr[0], cv2.COLOR_RGB2GRAY)[15:32, 15:32]
    assert b.find_singleton(cv2.cvtColor(arr[19], cv2.COLOR_RGB2GRAY), ref) is None


def test_low_texture_abstains():
    assert b.find_singleton(np.zeros((100, 100), np.uint8),
                            np.zeros((17, 17), np.uint8)) is None


@pytest.mark.parametrize("mode", ["identity", "impostor"])
def test_observationally_identical_singleton_can_be_impostor(mode):
    """A visual confidence gate CANNOT certify physical identity."""
    out = b.redetect(scene(mode), [23, 23], 0, [None]*28)
    assert out["confirmed"][18] == [73.0, 73.0]
    assert out["raw"][12] == [73.0, 73.0]
    assert out["confirmed"][12] is None


def test_no_lookahead_confirmation():
    out = b.redetect(scene(), [23, 23], 0, [None]*28)
    assert all(v is None for v in out["confirmed"][:18])
    assert out["confirmed"][18] == [73.0, 73.0]


def test_no_retroactive_dense_imputation():
    out = b.redetect(scene(), [23, 23], 0, [None]*28)
    assert out["confirmed"][19] is None and out["confirmed"][20] is None
    assert out["confirmed"][21] == [73.0, 73.0]


def test_query_frame_is_not_output():
    out = b.redetect(scene(), [23, 23], 0, [None]*28)
    assert out["raw"][0] is None and out["confirmed"][0] is None


def test_no_appearance_no_recovery():
    out = b.redetect(scene("blank"), [23, 23], 0, [None]*28)
    assert all(x is None for x in out["raw"])
    assert all(x is None for x in out["confirmed"])


def test_twins_no_recovery():
    out = b.redetect(scene("twins"), [23, 23], 0, [None]*28)
    assert all(x is None for x in out["confirmed"])


def test_baseline_acceptance_resets_history():
    original = [None]*28
    original[15] = [1.0, 1.0]
    out = b.redetect(scene(), [23, 23], 0, original)
    assert out["confirmed"][18] is None
    assert out["confirmed"][27] is None


def test_no_rescue_before_missing_window():
    frames = scene(n=13)
    out = b.redetect(frames, [23, 23], 0, [None]*13)
    assert out["raw"][11] is None


def test_border_query_abstains():
    out = b.redetect(scene(), [2, 2], 0, [None]*28)
    assert out["skipped_query_border"] is True


def test_low_texture_query_abstains():
    arr = np.zeros((28, 100, 100, 3), dtype=np.uint8)
    out = b.redetect(arr, [23, 23], 0, [None]*28)
    assert out["skipped_low_texture"] is True


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_bad_query_rejected(bad):
    with pytest.raises(RuntimeError, match="INVALID_QUERY"):
        b.redetect(scene(), [bad, 23], 0, [None]*28)


@pytest.mark.parametrize("t", [-1, 27])
def test_query_time_invalid(t):
    with pytest.raises(RuntimeError, match="QUERY_LENGTH"):
        b.redetect(scene(), [23, 23], t, [None]*28)


def test_missing_baseline_length_invalid():
    with pytest.raises(RuntimeError, match="QUERY_LENGTH"):
        b.redetect(scene(), [23, 23], 0, [None]*26)


def test_wrong_shape_search_rejected():
    with pytest.raises(RuntimeError, match="INVALID_IMAGE_SHAPE"):
        b.find_singleton(np.ones((50, 50, 3), np.uint8),
                         np.ones((17, 17), np.uint8))


def test_merge_never_overwrites_baseline():
    old = [None, [1, 2], None]
    new = [[4, 4], [8, 8], None]
    assert b.merge_track(old, new) == [[4, 4], [1, 2], None]


def test_merge_length_drift_rejected():
    with pytest.raises(RuntimeError, match="MERGE_LENGTH"):
        b.merge_track([None, None], [None])


def test_immutable_pin_and_no_overwrite(tmp_path):
    path = tmp_path/"prediction.json"
    obj = {"split": "dev", "test_oracle_read": False}
    sha = b.immutable_json(path, obj)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    assert b.pinned(path) == obj
    with pytest.raises(RuntimeError, match="IMMUTABLE"):
        b.immutable_json(path, obj)


def test_tampered_pin_is_rejected(tmp_path):
    path = tmp_path/"prediction.json"
    b.immutable_json(path, {"split": "dev"})
    path.write_text(json.dumps({"split": "test"}))
    with pytest.raises(RuntimeError, match="PIN_DRIFT"):
        b.pinned(path)


def test_unpinned_prediction_cannot_be_scored(tmp_path):
    p = tmp_path/"prediction.json"
    p.write_text(json.dumps({"kind": "BATCH76_PUBLIC_ONLY_CAUSAL_REDETECTION"}))
    with pytest.raises(RuntimeError, match="PIN_MISSING"):
        b.evaluate(p)


def test_wrong_prediction_kind_blocks_oracle(tmp_path):
    path = tmp_path/"prediction.json"
    b.immutable_json(path, {"kind": "IMPOSSIBLE", "split": "dev"})
    with pytest.raises(RuntimeError, match="PREDICTION_FIREWALL"):
        b.evaluate(path)


def test_test_split_blocks_oracle(tmp_path):
    path = tmp_path/"prediction.json"
    b.immutable_json(path, {"kind": "BATCH76_PUBLIC_ONLY_CAUSAL_REDETECTION",
                            "protocol": b.PROTOCOL, "split": "test",
                            "oracle_unread_during_prediction": True})
    with pytest.raises(RuntimeError, match="PREDICTION_FIREWALL"):
        b.evaluate(path)


def test_readonly_source_gate_blocks_test():
    with pytest.raises(RuntimeError, match="DEV_ONLY"):
        b.strict_manifest({"dataset": "TAP-Vid-Kinetics", "scenes": []},
                          {"split": "test"}, {})


def test_predictions_cannot_have_wrong_protocol(tmp_path):
    p = tmp_path/"prediction.json"
    b.immutable_json(p, {"kind": "BATCH76_PUBLIC_ONLY_CAUSAL_REDETECTION",
                         "split": "dev", "oracle_unread_during_prediction": True,
                         "protocol": "unknown"})
    with pytest.raises(RuntimeError, match="PREDICTION_FIREWALL"):
        b.evaluate(p)


def test_not_3d_camera_claim():
    assert "3d" not in b.PROTOCOL.lower()
    assert b.MIN_MARGIN == 0.07 and b.CONFIRMATIONS == 3