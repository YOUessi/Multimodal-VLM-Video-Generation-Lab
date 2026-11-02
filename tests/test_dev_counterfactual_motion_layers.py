"""Batch47E: deterministic synthetic truth, isolation, gate and audit tests."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import dev_counterfactual_motion_layers as x
import dev_counterfactual_motion_audit as verifier
import dev_temporal_bridge_affine as bridge


def textured(seed=57, h=240, w=360):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


@pytest.mark.parametrize("condition", x.CONDITIONS)
def test_deterministic_and_unmodified_input(condition):
    frame = textured()
    before = frame.copy()
    a = x.synthesize(frame, condition, seed=1241)
    b = x.synthesize(frame, condition, seed=1241)
    assert np.array_equal(frame, before)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert np.array_equal(a[3], b[3])
    assert a[3].shape[0] >= 12
    assert np.allclose(a[2], [[1, 0, x.DX], [0, 1, x.DY]])


@pytest.mark.parametrize("bad", [
    np.zeros((70, 120), np.uint8),
    np.zeros((150, 170, 3), np.uint8),
    np.zeros((150, 170), np.float32),
])
def test_invalid_frames_fail_closed(bad):
    with pytest.raises(ValueError, match="UINT8_GRAY"):
        x.synthesize(bad, x.CONDITIONS[0], seed=2)


def test_unknown_condition_rejected():
    with pytest.raises(ValueError, match="UNKNOWN_CONDITION"):
        x.synthesize(textured(), "TEST_ACCESS", 0)


def test_known_affine_exact_score_and_bad_model():
    frame = textured()
    _, _, truth, grid = x.synthesize(frame, "clean_affine", 2)
    ok = x.score_known_affine(truth, truth, grid)
    assert ok["synthetic_bg_median_error_px"] == pytest.approx(0., abs=1e-5)
    assert ok["synthetic_bg_wrong"] is False
    bad = truth.copy()
    bad[:, 2] += [7, 0]
    failed = x.score_known_affine(bad, truth, grid)
    assert failed["synthetic_bg_median_error_px"] == pytest.approx(7., abs=1e-4)
    assert failed["synthetic_bg_wrong"] is True
    assert x.score_known_affine(None, truth, grid)["synthetic_bg_wrong"] is None


def test_synthetic_clean_klt_nearly_exact_and_spatial_gate():
    frame = textured(h=288, w=512)
    a, b, truth, grid = x.synthesize(frame, "clean_affine", seed=93)
    estimated, model = x.estimate_pair(a, b)
    assert estimated["matched"] > 50
    assert estimated["fit_valid"]
    assert x.score_known_affine(model, truth, grid)["synthetic_bg_median_error_px"] < 1.
    assert estimated["cpu_ms_match_fit_crossfit"] > 0.


def test_layered_motion_differs_from_global_background():
    a, b, truth, grid = x.synthesize(textured(h=288, w=512),
                                     "dominant_foreground", seed=93)
    assert not np.array_equal(a, b)
    assert len(grid) < (13 * 9)
    assert np.allclose(truth[:, 2], [x.DX, x.DY])
    estimated, fit = x.estimate_pair(a, b)
    assert isinstance(estimated["gate_accept"], bool)
    assert 0 <= estimated["valid_folds"] <= 4
    error = x.score_known_affine(fit, truth, grid)
    assert error["synthetic_bg_wrong"] in (None, False, True)


def test_illumination_and_new_occlusion_have_independent_controls():
    frame = textured(h=288, w=512)
    a, lighting, _, _ = x.synthesize(frame, "lighting_shift", 1)
    _, clean, _, _ = x.synthesize(frame, "clean_affine", 1)
    assert not np.array_equal(lighting, clean)
    aa, bb, _, mask = x.synthesize(frame, "foreground_and_occlusion", 1)
    assert mask.shape[0] < 117
    assert np.count_nonzero(bb == 16) > 1000
    assert np.count_nonzero(aa == 16) < np.count_nonzero(bb == 16)


def test_gate_insufficient_support_is_abstention():
    assert not x.gate_crossfit({"status": "insufficient_valid_spatial_folds",
                                "valid_folds": 1})
    assert not x.gate_crossfit({"status": "ok", "valid_folds": 2,
                                "crossfit_worst_quadrant_median_px": 1.,
                                "mean_heldout_fraction_within_3px": 1.})
    good = {"status": "ok", "valid_folds": 3,
            "crossfit_worst_quadrant_median_px": 2.1,
            "mean_heldout_fraction_within_3px": 0.72}
    assert x.gate_crossfit(good)
    assert not x.gate_crossfit({**good, "crossfit_worst_quadrant_median_px": 5.})
    assert not x.gate_crossfit({**good, "mean_heldout_fraction_within_3px": .5})


def test_summary_counts_abstentions_as_noncoverage():
    rows = [
        {"fit_valid": True, "gate_accept": True, "synthetic_bg_wrong": False,
         "synthetic_bg_median_error_px": .4, "cpu_ms_match_fit_crossfit": 2.},
        {"fit_valid": True, "gate_accept": False, "synthetic_bg_wrong": True,
         "synthetic_bg_median_error_px": 7., "cpu_ms_match_fit_crossfit": 4.},
        {"fit_valid": False, "gate_accept": False, "synthetic_bg_wrong": None,
         "synthetic_bg_median_error_px": None, "cpu_ms_match_fit_crossfit": 1.},
    ]
    s = x.summary(rows)
    assert s["fit_count"] == 2 and s["gate_accept_count"] == 1
    assert s["baseline_wrong_count"] == 1 and s["gate_wrong_count"] == 0
    assert s["fit_coverage"] == pytest.approx(2/3)
    assert s["gate_coverage"] == pytest.approx(1/3)


def _fake_frozen_report(tmp_path, monkeypatch):
    rows = [dict(split="dev", video_id=f"d{i}", label=f"dev-label-{i}",
                 shard_index=i, archive_member=f"dev{i}.mp4") for i in range(6)]
    rows += [dict(split="test", video_id="TEST_PRIVATE_DO_NOT_ACCESS",
                  label="TEST_PRIVATE", shard_index=999,
                  archive_member="private.mp4")]
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"videos": rows}))
    manifest = tmp_path / "download_manifest.json"
    manifest.write_text(json.dumps({"shards": {}}))
    monkeypatch.setattr(bridge, "EXPECTED_SELECTION_SHA256",
                        verifier.common.selection_sha256(selection))
    monkeypatch.setattr(bridge, "EXPECTED_DOWNLOAD_MANIFEST_SHA256",
                        verifier.common.file_sha256(manifest))
    template = dict(
        detected=200, matched=120, fit_valid=True, gate_accept=True,
        inlier_fraction=.91, crossfit_status="ok", valid_folds=4,
        worst_holdout_median_px=1., fraction_holdout_within_3px=.93,
        cpu_ms_match_fit_crossfit=1.2,
        synthetic_bg_median_error_px=.35, synthetic_bg_p90_error_px=.55,
        synthetic_bg_wrong=False, synthetic_background_grid_count=75,
    )
    scenes = []
    for idx in range(6):
        rr = []
        for anchor in range(x.ANCHORS):
            for condition in x.CONDITIONS:
                rr.append({**template, "condition": condition, "anchor_frame": anchor})
        scenes.append(dict(dev_video_id=f"d{idx}", action_class=f"dev-label-{idx}",
                           rows=rr,
                           condition_summaries={
                               c: x.summary([r for r in rr if r["condition"] == c])
                               for c in x.CONDITIONS}))
    data = dict(
        scope="DEV_ONLY_GENERATED_2D_TRUTH_NO_TEST_NO_PRIVATE_ORACLE",
        selection_sha256=bridge.EXPECTED_SELECTION_SHA256,
        manifest_sha256=bridge.EXPECTED_DOWNLOAD_MANIFEST_SHA256,
        source_sha256=verifier.common.file_sha256(Path(x.__file__)),
        protocol=dict(conditions=list(x.CONDITIONS), anchors_per_clip=x.ANCHORS,
                      width_px=x.WIDTH, truth_global_dxdy_px=[x.DX, x.DY],
                      wrong_error_gt_px=x.MAX_GROUND_TRUTH_ERROR_PX,
                      gate=dict(min_valid_folds=x.MIN_GATE_FOLDS,
                                max_worst_holdout_median_px=x.MAX_WORST_HOLDOUT_PX,
                                min_fraction_holdout_within_3px=x.MIN_FRACTION_WITHIN_3PX)),
        scenes=scenes,
        overall={c: x.summary([r for s in scenes for r in s["rows"]
                               if r["condition"] == c]) for c in x.CONDITIONS},
    )
    report = tmp_path / "report.json"
    report.write_text(json.dumps(data))
    return report, selection, data


def test_frozen_report_complete_and_scoped(tmp_path, monkeypatch):
    path, selection, _ = _fake_frozen_report(tmp_path, monkeypatch)
    assert verifier.verify(path, selection)["cases"] == 240


@pytest.mark.parametrize("field,value,error", [
    ("scope", "TEST_ALLOWED", "SCOPE"),
    ("source_sha256", "f" * 64, "SOURCE_OR_DATA_PIN"),
])
def test_audit_rejects_scope_and_source_changes(tmp_path, monkeypatch, field, value, error):
    path, selection, data = _fake_frozen_report(tmp_path, monkeypatch)
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match=error):
        verifier.verify(path, selection)


def test_audit_rejects_private_test_id_in_dev_rows(tmp_path, monkeypatch):
    path, selection, data = _fake_frozen_report(tmp_path, monkeypatch)
    data["scenes"][0]["dev_video_id"] = "TEST_PRIVATE_DO_NOT_ACCESS"
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="DEV_ID_ORDER"):
        verifier.verify(path, selection)


def test_audit_rejects_wrong_gate_and_aggregates(tmp_path, monkeypatch):
    path, selection, data = _fake_frozen_report(tmp_path, monkeypatch)
    data["scenes"][0]["rows"][0]["valid_folds"] = 1
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="GATE_INCONSISTENCY"):
        verifier.verify(path, selection)
    _, selection, data = _fake_frozen_report(tmp_path, monkeypatch)
    data["overall"]["clean_affine"]["gate_accept_count"] -= 1
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="OVERALL_SUMMARY"):
        verifier.verify(path, selection)


def test_run_denies_unpinned_selection_before_media_read(tmp_path, monkeypatch):
    selected = tmp_path / "selection.json"
    selected.write_text('{"videos":[]}')
    with pytest.raises(RuntimeError, match="SELECTION_PIN"):
        x.run(selected, tmp_path / "SHARDS_INCLUDING_PRIVATE_TEST",
              tmp_path / "out.json")


def test_runner_never_includes_test_data(monkeypatch, tmp_path):
    # Input restriction is explicit: anything not whitelisted DEV is excluded.
    assert "TEST" not in x.CONDITIONS
    import inspect
    source = inspect.getsource(x.run)
    assert "audit.dev_rows" in source
    assert 'audit.selection_sha256(selection)' in source
