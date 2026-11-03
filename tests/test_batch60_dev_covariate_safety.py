"""Adversarial tests for Batch60's frozen DEV-only covariate stratification."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import batch60_dev_covariate_safety as audit
import batch58_dev_grouped_transfer as old


def fixture():
    cfg = json.loads((ROOT / "configs/kinetics_batch56.json").read_text())
    methods = cfg["dev_selection"]["candidate_methods"]
    ids = [f"kinetics_{i:02d}" for i in range(36)]
    pixel, relative, rows = {}, {}, {}
    for i, scene in enumerate(ids):
        pixel[scene] = {
            "mean_luma": 50 + i,
            "laplacian_variance_mean": 100 + i * i,
            "mean_adjacent_frame_absdiff": i / 3,
            "affine_translation_p50_px": i / 10,
        }
        relative[scene] = {
            "background_relative_residual_mean_px": i / 5,
            "camera_explained_fraction_mean_unclipped": (36 - i) / 36,
        }
        rows[scene] = {}
        for k, method in enumerate(methods):
            risk = 8 if (i >= 24 and k == 0) else 0
            rows[scene][method] = {
                "visible": 50,
                "accepted_visible": 40,
                "invisible": 100,
                "invisible_false_accept": risk,
                "visible_coverage": .8,
                "invisible_false_accept_rate": risk / 100,
                "jaccard_mean_1_2_4_8_16": .8 - .05 * k,
            }
    dev = {
        "split": "dev", "source_commit": "a" * 40,
        "config_sha256": audit.digest(audit.canonical(cfg)),
        "scene_count": 36, "per_scene": rows,
        "scene_pixel_diagnostics": pixel,
        "background_relative_diagnostics": relative,
    }
    dev["scene_macro"] = {
        method: {
            key: old.average([rows[s][method] for s in ids], key)
            for key in old.KEYS
        }
        for method in methods
    }
    selected_method, eligible, _ = old.choose(dev, ids, cfg["dev_selection"])
    selected = {
        "split": "dev", "source_commit": dev["source_commit"],
        "config_sha256": dev["config_sha256"],
        "selected_method": selected_method,
        "eligible_candidates": eligible,
        "rules": cfg["dev_selection"],
        "dev_results_sha256": audit.digest(audit.canonical(dev)),
    }
    return dev, selected, cfg, ids


def test_six_equal_covariate_terciles_deterministic_and_dev_only():
    dev, selected, cfg, ids = fixture()
    a = audit.analyze(dev, selected, cfg["dev_selection"], ids)
    assert a == audit.analyze(dev, selected, cfg["dev_selection"], ids)
    assert len(a["feature_reports"]) == 6
    assert len(a["methods"]) == 9
    assert a["split"] == "dev"
    assert not a["test_data_read"]
    assert not a["model_selection_changed"]
    assert not a["deployment_authorized"]
    for feature, record in a["feature_reports"].items():
        assert [len(s["scene_ids"]) for s in record["terciles"]] == [12] * 3
        assert sorted(sum((s["scene_ids"] for s in record["terciles"]), [])) == ids
        for s in record["terciles"]:
            assert set(s["methods"]) == set(cfg["dev_selection"]["candidate_methods"])


def test_strata_report_captures_high_camera_relative_residual_risk():
    dev, selected, cfg, ids = fixture()
    a = audit.analyze(dev, selected, cfg["dev_selection"], ids)
    method = a["selected_method"]
    bins = a["feature_reports"]["background_relative_residual_px"]["terciles"]
    assert bins[0]["methods"][method]["occluded_false_accept_scene_macro"] == 0
    assert bins[2]["methods"][method]["occluded_false_accept_scene_macro"] == .08
    assert bins[2]["methods"][method]["occluded_false_accept_count"] == 96
    assert bins[2]["methods"][method]["occluded_frame_count"] == 1200
    assert a["feature_reports"]["background_relative_residual_px"]["chosen_method_spearman_rho_on_supported_scenes"] > 0


def test_scene_macro_not_confused_with_micro_when_denominators_unbalanced():
    dev, selected, cfg, ids = fixture()
    method = selected["selected_method"]
    dev["per_scene"][ids[0]][method]["invisible"] = 1
    dev["per_scene"][ids[0]][method]["invisible_false_accept"] = 1
    dev["per_scene"][ids[0]][method]["invisible_false_accept_rate"] = 1.
    first = audit.summarize(dev, ids[:2], method, .1)
    assert first["occluded_false_accept_scene_macro"] == .5
    assert first["occluded_false_accept_micro"] == pytest.approx(1 / 101)
    assert first["violating_scene_count"] == 1
    assert first["low_occlusion_support_scenes_lt_20_frames"] == [ids[0]]


def test_unsupported_occlusion_does_not_claim_zero_risk():
    dev, selected, cfg, ids = fixture()
    method = selected["selected_method"]
    for scene in ids[:12]:
        row = dev["per_scene"][scene][method]
        row.update(invisible=0, invisible_false_accept=0, invisible_false_accept_rate=None)
    group = audit.summarize(dev, ids[:12], method, .1)
    assert group["scenes_without_occlusion_evidence"] == 12
    assert group["occluded_false_accept_scene_macro"] is None
    assert group["occluded_false_accept_micro"] is None
    assert group["worst_supported_scene_false_accept"] is None
    assert group["violating_scene_count"] == 0


@pytest.mark.parametrize("mutation,expected", [
    (lambda d, s, c, ids: d["scene_pixel_diagnostics"][ids[0]].pop("mean_luma"), "MISSING_COVARIATE"),
    (lambda d, s, c, ids: d["scene_pixel_diagnostics"][ids[0]].update(mean_luma=float("nan")), "INVALID_COVARIATE"),
    (lambda d, s, c, ids: d["scene_pixel_diagnostics"][ids[0]].update(mean_luma="dark"), "INVALID_COVARIATE"),
    (lambda d, s, c, ids: d["scene_pixel_diagnostics"][ids[0]].update(mean_luma=True), "INVALID_COVARIATE"),
    (lambda d, s, c, ids: d["background_relative_diagnostics"].pop(ids[0]), "INCOMPLETE_COVARIATE"),
    (lambda d, s, c, ids: c["dev_selection"].update(max_invisible_false_accept_rate=1.), "INVALID_RISK_BUDGET"),
    (lambda d, s, c, ids: s.update(selected_method="new_unregistered_method"), "UNFROZEN_METHOD"),
    (lambda d, s, c, ids: d["per_scene"][ids[0]][s["selected_method"]].update(invisible_false_accept=101), "INVALID_COUNTS"),
    (lambda d, s, c, ids: d["per_scene"][ids[0]][s["selected_method"]].update(invisible_false_accept_rate=.22), "INCONSISTENT_RATE"),
])
def test_bad_inputs_fail_closed(mutation, expected):
    dev, selected, cfg, ids = fixture()
    mutation(dev, selected, cfg, ids)
    with pytest.raises(RuntimeError, match=expected):
        audit.analyze(dev, selected, cfg["dev_selection"], ids)


def test_duplicate_scene_ids_rejected():
    dev, selected, cfg, ids = fixture()
    with pytest.raises(RuntimeError, match="INVALID_SCENES"):
        audit.analyze(dev, selected, cfg["dev_selection"], [ids[0]] * 36)


def test_spearman_tied_ranks_are_valid_and_constant_undefined():
    assert audit.spearman([1., 2., 3.], [3., 2., 1.]) == pytest.approx(-1.)
    assert audit.spearman([1., 1., 2., 2.], [1., 1., 2., 2.]) == pytest.approx(1.)
    assert audit.spearman([1., 1., 1.], [1., 2., 3.]) is None
    assert audit.spearman([1., 2.], [1., 2.]) is None


def test_pinned_io_and_read_only_frozen_source(tmp_path, monkeypatch):
    dev, selected, cfg, ids = fixture()
    frozen = tmp_path / "frozen"
    out = frozen / "outputs/kinetics_batch56"
    out.mkdir(parents=True)
    configs = frozen / "configs"
    configs.mkdir()
    (configs / "kinetics_batch56.json").write_bytes(audit.canonical(cfg))
    (out / "dev_results.json").write_bytes(audit.canonical(dev))
    raw = audit.canonical(selected)
    (out / "dev_selection.json").write_bytes(raw)
    (out / "dev_selection.json.sha256").write_text(audit.digest(raw) + "\n")
    monkeypatch.setattr(audit, "IN_DIR", out)
    monkeypatch.setattr(audit, "FROZEN", frozen)
    monkeypatch.setattr(audit, "OUTPUT", tmp_path / "research/report.json")
    monkeypatch.setattr(audit.subprocess, "check_output", lambda *args, **kwargs: "a" * 40)
    first = audit.run()
    assert audit.run() == first  # idempotent immutable output
    assert audit.OUTPUT.with_suffix(".json.sha256").read_text().strip() == audit.digest(audit.OUTPUT.read_bytes())
    (out / "dev_results.json").write_bytes(audit.canonical({**dev, "split": "test"}))
    with pytest.raises(RuntimeError, match="DEV_RESULT_PIN_INVALID"):
        audit.run()


def test_pinned_selection_tamper_rejected(tmp_path, monkeypatch):
    dev, selected, cfg, ids = fixture()
    frozen = tmp_path / "frozen"
    out = frozen / "outputs/kinetics_batch56"
    out.mkdir(parents=True)
    (frozen / "configs").mkdir()
    (frozen / "configs/kinetics_batch56.json").write_bytes(audit.canonical(cfg))
    (out / "dev_results.json").write_bytes(audit.canonical(dev))
    raw = audit.canonical(selected)
    (out / "dev_selection.json").write_bytes(raw + b" ")
    (out / "dev_selection.json.sha256").write_text(audit.digest(raw) + "\n")
    monkeypatch.setattr(audit, "IN_DIR", out)
    monkeypatch.setattr(audit, "FROZEN", frozen)
    with pytest.raises(RuntimeError, match="PIN_INVALID"):
        audit.run()
