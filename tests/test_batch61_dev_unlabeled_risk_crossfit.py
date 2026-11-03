"""Adversarial regression suite for DEV-only Batch61 cross-fit risk-proxy audit."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import batch61_dev_unlabeled_risk_crossfit as risk
from test_batch60_dev_covariate_safety import fixture


def inputs():
    dev, selected, cfg, ids = fixture()
    return dev, selected, cfg, ids


def test_image_only_feature_contract_is_explicit():
    assert risk.IMAGE_FEATURES == (
        "mean_luma", "laplacian_variance_mean",
        "mean_adjacent_frame_absdiff", "affine_translation_p50_px"
    )
    assert "background_relative_residual_mean_px" not in risk.IMAGE_FEATURES


def test_image_only_extraction_does_not_access_privileged_section():
    dev, selected, cfg, ids = inputs()
    before = risk.extract_features(dev, ids)
    dev.pop("background_relative_diagnostics")
    assert np.array_equal(before, risk.extract_features(dev, ids))


def test_privileged_mode_takes_only_oracle_diagnostic():
    dev, selected, cfg, ids = inputs()
    before = risk.extract_features(dev, ids, privileged=True)
    dev.pop("scene_pixel_diagnostics")
    assert np.array_equal(before, risk.extract_features(dev, ids, privileged=True))


def test_synthetic_crossfit_end_to_end_deterministic():
    dev, selected, cfg, ids = inputs()
    result = risk.analyze(dev, selected, cfg)
    again = risk.analyze(dev, selected, cfg)
    assert result == again
    assert result["split"] == "dev"
    assert not result["test_data_read"]
    assert not result["model_selection_changed"]
    assert not result["deployment_authorized"]
    assert result["image_only"]["scene_count"] == 36
    assert result["privileged_negative_control"]["privileged_oracle_dependent_feature"]
    assert not result["image_only"]["privileged_oracle_dependent_feature"]
    assert not result["privileged_negative_control"]["deployable"]
    assert not result["image_only"]["deployable"]


def test_no_occlusion_scene_is_unknown_not_safe():
    dev, selected, cfg, ids = inputs()
    method = selected["selected_method"]
    first = dev["per_scene"][ids[0]][method]
    first.update(invisible=0, invisible_false_accept=0, invisible_false_accept_rate=None)
    feature = risk.extract_features(dev, ids)
    out = risk.crossfit(dev, ids, method, feature, risk.IMAGE_FEATURES)
    assert out["unsupported_unknown_scene_count"] == 1
    assert out["supported_scene_count"] == 35
    assert out["rows"][0]["label"] is None


def test_always_six_alerts_fixed_not_outcome_optimized():
    dev, selected, cfg, ids = inputs()
    features = risk.extract_features(dev, ids)
    out = risk.crossfit(dev, ids, selected["selected_method"], features, risk.IMAGE_FEATURES)
    assert out["alerts"] == 6
    assert len({r["fold"] for r in out["rows"] if r["alert"]}) == 6
    assert all(f["train_only_preprocessing"] for f in
               (fold["train_details"] for fold in out["folds"]))


def test_risk_label_is_derived_from_counts_consistently():
    dev, selected, cfg, ids = inputs()
    method = selected["selected_method"]
    labels = risk.risk_labels(dev, ids, method, risk.SAFETY_BUDGET)
    assert set(labels) == {0}
    s = ids[0]
    dev["per_scene"][s][method].update(invisible_false_accept=11, invisible_false_accept_rate=.11)
    labels = risk.risk_labels(dev, ids, method, risk.SAFETY_BUDGET)
    assert labels[0] == 1
    assert sum(labels) == 1


def test_heldout_labels_cannot_change_fold_predictions():
    dev, selected, cfg, ids = inputs()
    method = selected["selected_method"]
    feature = risk.extract_features(dev, ids)
    first = risk.crossfit(dev, ids, method, feature, risk.IMAGE_FEATURES)
    # The 0th fold is withheld; changing its *labels* cannot change its scores.
    changed = copy.deepcopy(dev)
    for sid in ids[:6]:
        row = changed["per_scene"][sid][method]
        row.update(invisible_false_accept=100, invisible_false_accept_rate=1.0)
    second = risk.crossfit(changed, ids, method, feature, risk.IMAGE_FEATURES)
    assert [r["prob"] for r in first["rows"][:6]] == [r["prob"] for r in second["rows"][:6]]
    assert first["folds"][0]["alert_scene"] == second["folds"][0]["alert_scene"]


def test_train_only_standardization_isolated_from_heldout_features():
    x = np.arange(40, dtype=float).reshape(10, 4)
    train_y = [0] * 7 + [1] * 3
    pred, meta = risk.fit_predict(x, train_y, np.array([[1000., 0., 0., 0.]]))
    pred2, meta2 = risk.fit_predict(x, train_y, np.array([[-1000., 0., 0., 0.]]))
    assert meta["train_feature_means"] == meta2["train_feature_means"]
    assert meta["train_feature_std"] == meta2["train_feature_std"]
    assert len(pred) == 1 and len(pred2) == 1


def test_all_safe_train_fold_produces_finite_probabilities():
    p, meta = risk.fit_predict(
        np.ones((10, 4)), [0] * 10, np.ones((2, 4))
    )
    assert np.all(np.isfinite(p))
    assert np.all((p > 0) & (p < 1))
    assert meta["train_violations"] == 0


def test_all_unsafe_train_fold_produces_finite_probabilities():
    p, meta = risk.fit_predict(
        np.ones((10, 4)), [1] * 10, np.ones((2, 4))
    )
    assert np.all(np.isfinite(p))
    assert np.all((p > 0) & (p < 1))
    assert meta["train_violations"] == 10


@pytest.mark.parametrize("labels", [
    [-1] * 10, [2] * 10, [0] * 9, [False] * 10,
])
def test_bad_training_labels_rejected_or_fail_closed(labels):
    x = np.ones((10, 4))
    if len(labels) != 10:
        with pytest.raises(RuntimeError, match="INVALID_FOLD_SHAPE"):
            risk.fit_predict(x, labels, x)
    else:
        with pytest.raises(RuntimeError, match="INVALID_TRAIN_LABEL"):
            risk.fit_predict(x, labels, x)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), None, "a", True])
def test_bad_image_feature_fails_closed(value):
    dev, selected, cfg, ids = inputs()
    dev["scene_pixel_diagnostics"][ids[0]]["mean_luma"] = value
    with pytest.raises(RuntimeError, match="NONFINITE_OR_MISSING_FEATURE"):
        risk.extract_features(dev, ids)


def test_missing_image_scene_fails_closed():
    dev, selected, cfg, ids = inputs()
    dev["scene_pixel_diagnostics"].pop(ids[0])
    with pytest.raises(RuntimeError, match="MISSING_DIAGNOSTIC"):
        risk.extract_features(dev, ids)


def test_missing_privileged_data_only_blocks_negative_control():
    dev, selected, cfg, ids = inputs()
    dev["background_relative_diagnostics"].pop(ids[0])
    assert risk.extract_features(dev, ids).shape == (36, 4)
    with pytest.raises(RuntimeError, match="MISSING_DIAGNOSTIC"):
        risk.extract_features(dev, ids, privileged=True)


def test_fit_rejects_nan_and_bad_shapes():
    x = np.ones((10, 4))
    x[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="INVALID_NUMERIC"):
        risk.fit_predict(x, [0] * 10, np.ones((3, 4)))
    with pytest.raises(RuntimeError, match="INVALID_FOLD_SHAPE"):
        risk.fit_predict(np.ones((10, 4)), [0] * 10, np.ones((3, 2)))


def test_auc_handles_ties_and_missing_positive():
    r = [{"prob": 0.5, "label": 1}, {"prob": 0.5, "label": 0}]
    assert risk.auc_from_scores(r) == .5
    assert risk.auc_from_scores([{"prob": 0.5, "label": 0}]) is None
    assert risk.auc_from_scores([{"prob": 0.5, "label": 1}]) is None


def test_rejects_threshold_and_population_drift():
    dev, selected, cfg, ids = inputs()
    features = risk.extract_features(dev, ids)
    with pytest.raises(RuntimeError, match="SAFETY_BUDGET_DRIFT"):
        risk.crossfit(dev, ids, selected["selected_method"], features,
                      risk.IMAGE_FEATURES, safety_budget=.2)
    with pytest.raises(RuntimeError, match="INVALID_COHORT"):
        risk.crossfit(dev, ids[:-1], selected["selected_method"], features[:-1],
                      risk.IMAGE_FEATURES)
    with pytest.raises(RuntimeError, match="INVALID_COHORT"):
        risk.crossfit(dev, [ids[0]] * 36, selected["selected_method"], features,
                      risk.IMAGE_FEATURES)


def test_pinned_run_idempotent_and_tampered_dev_rejected(tmp_path, monkeypatch):
    dev, selected, cfg, ids = inputs()
    frozen = tmp_path / "frozen"
    folder = frozen / "outputs" / "kinetics_batch56"
    folder.mkdir(parents=True)
    (frozen / "configs").mkdir()
    (frozen / "configs" / "kinetics_batch56.json").write_bytes(risk.canonical(cfg))
    (folder / "dev_results.json").write_bytes(risk.canonical(dev))
    raw_sel = risk.canonical(selected)
    (folder / "dev_selection.json").write_bytes(raw_sel)
    (folder / "dev_selection.json.sha256").write_text(risk.digest(raw_sel) + "\n")
    monkeypatch.setattr(risk, "FROZEN", frozen)
    monkeypatch.setattr(risk, "IN_DIR", folder)
    monkeypatch.setattr(risk, "OUTPUT", tmp_path / "result.json")
    monkeypatch.setattr(risk.subprocess, "check_output", lambda *args, **kwargs: "a" * 40)
    result = risk.run()
    assert risk.run() == result
    assert risk.OUTPUT.with_suffix(".json.sha256").read_text().strip() == risk.digest(
        risk.OUTPUT.read_bytes()
    )
    (folder / "dev_results.json").write_bytes(
        risk.canonical({**dev, "split": "test"})
    )
    with pytest.raises(RuntimeError, match="DEV_RESULTS_PIN_INVALID"):
        risk.run()


def test_immutable_output_protection(tmp_path, monkeypatch):
    dev, selected, cfg, ids = inputs()
    folder = tmp_path / "outputs" / "kinetics_batch56"
    folder.mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (folder / "dev_results.json").write_bytes(risk.canonical(dev))
    selected["dev_results_sha256"] = risk.digest((folder / "dev_results.json").read_bytes())
    (folder / "dev_selection.json").write_bytes(risk.canonical(selected))
    (folder / "dev_selection.json.sha256").write_text(
        risk.digest(risk.canonical(selected)) + "\n"
    )
    (tmp_path / "configs" / "kinetics_batch56.json").write_bytes(risk.canonical(cfg))
    monkeypatch.setattr(risk, "FROZEN", tmp_path)
    monkeypatch.setattr(risk, "IN_DIR", folder)
    monkeypatch.setattr(risk, "OUTPUT", tmp_path / "research.json")
    monkeypatch.setattr(risk.subprocess, "check_output", lambda *args, **kwargs: "a" * 40)
    risk.run()
    monkeypatch.setattr(risk, "analyze", lambda d, s, c: {"kind": "second"})
    with pytest.raises(RuntimeError, match="IMMUTABLE_OUTPUT_CONFLICT"):
        risk.run()
