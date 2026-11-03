"""Batch64 pinned DEV RGB appearance experiment tests; no TEST oracle access."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import batch62_dev_consensus_veto as b62
from experiments import batch64_dev_rgb_identity as b64


def img(color=(64, 96, 128)):
    out = np.empty((256, 256, 3), dtype=np.uint8)
    out[:] = color
    return out


def trace(n=4, support=True):
    return {
        b62.BASE: [[100, 100] for _ in range(n)],
        "template_recover4": [[100, 100] if support else [150, 150] for _ in range(n)],
        "affine_stabilized_klt": [[100, 100] for _ in range(n)],
    }


def toy(n=4):
    frames = np.stack([img()] * n)
    pred = {"nframes": n, "tracks": {f"p{i}": trace(n) for i in range(4)}}
    qs = [{"id": f"p{i}", "query_time": 0, "query_xy": [100, 100]}
          for i in range(4)]
    oracle = {"tracks": {f"p{i}": {
        "xy": [[100, 100]] * n,
        "occluded": [False, False, True, False],
    } for i in range(4)}}
    return frames, pred, qs, oracle


def test_histogram_normalization():
    h = b64.histogram(img(), [100, 100])
    assert h.shape == (24,)
    assert h.sum() == pytest.approx(1)
    assert int((h > 0).sum()) == 3


def test_overlap_equal():
    h = b64.histogram(img(), [100, 100])
    assert b64.overlap(h, h) == pytest.approx(1)


def test_overlap_different():
    a = b64.histogram(img((0, 0, 0)), [100, 100])
    b = b64.histogram(img((255, 255, 255)), [100, 100])
    assert b64.overlap(a, b) == pytest.approx(0)


def test_overlap_symmetric():
    a = b64.histogram(img((100, 100, 100)), [100, 100])
    b = b64.histogram(img((100, 200, 100)), [100, 100])
    assert b64.overlap(a, b) == b64.overlap(b, a)


@pytest.mark.parametrize("xy", [None, [-1, 5], [256, 5], [0, 300]])
def test_bad_position_returns_none(xy):
    assert b64.histogram(img(), xy) is None


def test_border_ok_when_enough_pixels():
    assert b64.histogram(img(), [0, 0]) is not None


def test_reject_bad_image_dtype():
    with pytest.raises(RuntimeError, match="BAD_FRAME"):
        b64.histogram(img().astype(np.float32), [10, 10])


def test_reject_bad_image_channels():
    with pytest.raises(RuntimeError, match="BAD_FRAME"):
        b64.histogram(np.zeros((256, 256), np.uint8), [10, 10])


def test_reject_nonfinite_histogram():
    with pytest.raises(RuntimeError, match="BAD_HISTOGRAM"):
        b64.overlap(np.ones(24), np.full(24, np.nan))


def test_reject_wrong_histogram_length():
    with pytest.raises(RuntimeError, match="BAD_HISTOGRAM"):
        b64.overlap(np.ones(23), np.ones(24))


def test_frame_scores_query_no_credit():
    frames, pred, qs, _ = toy()
    got = b64.frame_scores(frames, qs, pred)
    assert all(v[0] is None for v in got.values())
    assert all(v[1:] == pytest.approx([1, 1, 1]) for v in got.values())


def test_frame_scores_color_change():
    frames, pred, qs, _ = toy()
    frames[2] = img((200, 210, 220))
    got = b64.frame_scores(frames, qs, pred)
    assert got["p0"][1] == pytest.approx(1)
    assert got["p0"][2] == pytest.approx(0)
    assert got["p0"][3] == pytest.approx(1)


def test_no_future_lookahead():
    frames, pred, qs, _ = toy()
    before = b64.frame_scores(frames, qs, pred)
    frames[3] = img((200, 210, 220))
    after = b64.frame_scores(frames, qs, pred)
    assert all(after[k][:3] == before[k][:3] for k in before)


def test_missing_tracker_veto():
    frames, pred, qs, _ = toy()
    pred["tracks"]["p0"][b62.BASE][2] = None
    got = b64.frame_scores(frames, qs, pred)
    assert got["p0"][2] is None


def test_prequery_leak_refused():
    frames, pred, qs, _ = toy()
    qs[0]["query_time"] = 2
    with pytest.raises(RuntimeError, match="BAD_TRACK"):
        b64.frame_scores(frames, qs, pred)


def test_invalid_frame_shape():
    frames, pred, qs, _ = toy()
    with pytest.raises(RuntimeError, match="FRAME_SHAPE"):
        b64.frame_scores(frames[:, :, :100, :], qs, pred)


def test_invalid_query_population():
    frames, pred, qs, _ = toy()
    with pytest.raises(RuntimeError, match="QUERY_POPULATION"):
        b64.frame_scores(frames, qs[:-1], pred)


@pytest.mark.parametrize("variant,rgb,expected", [
    ("baseline", None, True),
    ("template_4px", None, True),
    ("rgb_0.55", 0.2, False),
    ("rgb_0.55", 0.6, True),
    ("rgb_0.70", 0.7, True),
    ("rgb_0.85", 0.849, False),
    ("template_rgb_0.85", 0.99, True),
])
def test_gate_fixed_variants(variant, rgb, expected):
    assert b64.accepted(trace(), rgb, 2, variant) is expected


def test_template_rgb_cannot_accept_unsupported():
    assert b64.accepted(trace(support=False), 1, 2, "rgb_0.55")
    assert not b64.accepted(trace(support=False), 1, 2, "template_rgb_0.55")


def test_no_base_cannot_accept():
    tr = trace()
    tr[b62.BASE][2] = None
    assert not b64.accepted(tr, 1, 2, "rgb_0.55")


def test_unknown_gate_refused():
    with pytest.raises(RuntimeError, match="BAD_VARIANT"):
        b64.accepted(trace(), 1, 2, "rgb_0.99")


def test_score_scene_oracle():
    frames, pred, qs, oracle = toy()
    rgb = b64.frame_scores(frames, qs, pred)
    out = b64.score_scene(pred, qs, oracle, rgb)
    for v in b64.VARIANTS:
        assert out[v]["accepted_visible"] == 8
        assert out[v]["invisible_false_accept"] == 4
        assert out[v]["correct8"] == 8
        assert out[v]["max_false_accept_episode_frames"] == 1


def test_score_scene_rgb_veto():
    frames, pred, qs, oracle = toy()
    frames[2] = img((200, 210, 220))
    rgb = b64.frame_scores(frames, qs, pred)
    out = b64.score_scene(pred, qs, oracle, rgb)
    assert out["baseline"]["invisible_false_accept"] == 4
    assert out["rgb_0.55"]["invisible_false_accept"] == 0
    assert out["rgb_0.55"]["accepted_visible"] == 8


def test_score_scene_reject_bad_visibility():
    frames, pred, qs, oracle = toy()
    oracle["tracks"]["p0"]["occluded"][2] = 1
    with pytest.raises(RuntimeError, match="VISIBILITY_TYPE"):
        b64.score_scene(pred, qs, oracle, b64.frame_scores(frames, qs, pred))


def test_score_scene_reject_unmatching_scores():
    frames, pred, qs, oracle = toy()
    rgb = b64.frame_scores(frames, qs, pred)
    rgb["p0"].pop()
    with pytest.raises(RuntimeError, match="SCORING_SHAPE"):
        b64.score_scene(pred, qs, oracle, rgb)


def test_canonical_determinism():
    assert b64.canonical({"b": 1, "a": 2}) == b64.canonical({"a": 2, "b": 1})


def test_sha_file_matches_hash(tmp_path: Path):
    import hashlib
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    assert b64.sha_file(p) == hashlib.sha256(b"abc").hexdigest()


def test_pin_roundtrip(tmp_path: Path):
    p = tmp_path / "features.json"
    digest = b64.pin_write(p, {"a": [1]})
    assert digest == b64.sha_file(p)
    assert b64.pinned(p) == {"a": [1]}


def test_pin_no_overwrite(tmp_path: Path):
    p = tmp_path / "features.json"
    b64.pin_write(p, {"x": 0})
    with pytest.raises(RuntimeError, match="NO_OVERWRITE"):
        b64.pin_write(p, {"x": 1})


def test_pin_drift_rejected(tmp_path: Path):
    p = tmp_path / "features.json"
    b64.pin_write(p, {"x": 0})
    p.write_text('{"x": 1}\n')
    with pytest.raises(RuntimeError, match="PIN_DRIFT"):
        b64.pinned(p)


def test_real_dev_only_pinned_artifacts_when_present():
    if not b64.FEATURES.exists() or not b64.REPORT.exists():
        pytest.skip("real DEV phase not yet run")
    feat, result = b64.pinned(b64.FEATURES), b64.pinned(b64.REPORT)
    assert set(feat["score_by_scene"]) == set(b62.SCENES)
    assert result["test_oracle_read"] is False
    assert result["features_pinned_before_dev_oracle_read"] is True
    assert result["summary"]["baseline"]["scene_budget_violation_count"] == 4
    assert result["summary"]["template_4px"]["scene_budget_violation_count"] == 1


def test_public_phase_rejects_nondev_split(monkeypatch):
    names = b62.SCENES
    pred = {
        "split": "test", "oracle_unread_during_prediction": True,
        "manifest_sha256": "x", "public_query_sha256": "x",
        "methods": [b62.BASE, "template_recover4"],
        "records": [{"name": s} for s in names],
    }
    query = {s: [] for s in names}
    manifest = {"scenes": [{
        "name": s, "split": "dev", "query_count": 4,
        "frame_file": f"data/tapvid_kinetics_batch56/batch56_frozen/frames/{s}.npy"
    } for s in names]}
    monkeypatch.setattr(b64, "sha_file", lambda path: "x")
    with pytest.raises(RuntimeError, match="PUBLIC_PHASE_FIREWALL"):
        b64.validate_public(pred, query, manifest)
    pred["split"] = "dev"
    b64.validate_public(pred, query, manifest)
    pred["oracle_unread_during_prediction"] = False
    with pytest.raises(RuntimeError, match="PUBLIC_PHASE_FIREWALL"):
        b64.validate_public(pred, query, manifest)


def test_public_phase_rejects_manifest_file_mismatch(monkeypatch):
    names = b62.SCENES
    pred = {"split": "dev", "oracle_unread_during_prediction": True,
            "manifest_sha256": "x", "public_query_sha256": "x",
            "methods": [b62.BASE, "template_recover4"],
            "records": [{"name": s} for s in names]}
    query = {s: [] for s in names}
    manifest = {"scenes": [{
        "name": s, "split": "dev", "query_count": 4,
        "frame_file": f"data/tapvid_kinetics_batch56/batch56_frozen/frames/{s}.npy"
    } for s in names]}
    monkeypatch.setattr(b64, "sha_file", lambda path: "x")
    manifest["scenes"][0]["frame_file"] = "test/secret.npy"
    with pytest.raises(RuntimeError, match="DEV_MANIFEST_DRIFT"):
        b64.validate_public(pred, query, manifest)
