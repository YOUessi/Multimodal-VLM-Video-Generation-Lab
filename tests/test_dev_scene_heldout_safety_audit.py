"""Pure synthetic tests; never mount or open sealed TEST examples."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

FILE = Path(__file__).resolve().parents[1] / "experiments/dev_scene_heldout_safety_audit.py"
SPEC = importlib.util.spec_from_file_location("dev52_safety", FILE)
M = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(M)


def fixture():
    cfg = {
        "dev_selection": {
            "candidate_methods": ["a", "b"],
            "fallback_method": "b",
            "min_visible_coverage": .3,
            "max_invisible_false_accept_rate": .1,
        }
    }
    canonical = (json.dumps(cfg, sort_keys=True, ensure_ascii=False,
                            indent=2) + "\n").encode()
    cfg_hash = hashlib.sha256(canonical).hexdigest()
    names = [f"kinetics_{i:02d}" for i in range(6)]
    scene = {}
    for name in names:
        scene[name] = {}
        for method, j in (("a", .5), ("b", .2)):
            scene[name][method] = {
                "visible": 40, "invisible": 20,
                "invisible_false_accept": 1,
                "visible_coverage": .5,
                "invisible_false_accept_rate": .05,
                "jaccard_mean_1_2_4_8_16": j,
            }
    results = {
        "split": "dev", "source_commit": "source-sha",
        "manifest_sha256": "manifest-hash", "config_sha256": cfg_hash,
        "prediction_sha256": "prediction-hash",
        "scene_count": 6, "per_scene": scene, "methods": ["a", "b"],
    }
    predictions = {
        "split": "dev", "source_commit": "source-sha",
        "manifest_sha256": "manifest-hash", "config_sha256": cfg_hash,
        "records": [{"name": n} for n in names],
        "scene_aux": {n: {"pixel_diagnostics": {"mean_luma": 80}}
                      for n in names},
    }
    return results, predictions, cfg


def call(inputs):
    return M.audit(*inputs[:3], "result-hash", "prediction-hash")


def test_full_dev_and_all_heldout_choose_a():
    out = call(fixture())
    assert out["full_dev_selected"] == "a"
    assert out["fold_selected_counts"] == {"a": 6}
    assert out["heldout_failed_safety"] == 0
    assert out["heldout_unevaluable_safety"] == 0
    assert out["split"] == "dev"
    assert len(out["folds"]) == 6
    assert all(f["training_scene_count"] == 5 for f in out["folds"])


def test_holdout_outlier_not_used_for_its_own_selection():
    inp = fixture()
    # The held-out scene heavily favors b; five training scenes still favor a.
    inp[0]["per_scene"]["kinetics_05"]["b"]["jaccard_mean_1_2_4_8_16"] = 1.0
    inp[0]["per_scene"]["kinetics_05"]["a"]["jaccard_mean_1_2_4_8_16"] = 0.0
    out = call(inp)
    last = next(f for f in out["folds"] if f["heldout"] == "kinetics_05")
    assert last["selection_from_train_only"] == "a"
    assert last["heldout_scene_jaccard"] == 0.0


def test_heldout_safety_failure_detected():
    inp = fixture()
    m = inp[0]["per_scene"]["kinetics_03"]["a"]
    m.update(invisible_false_accept=12, invisible_false_accept_rate=.6)
    out = call(inp)
    third = next(f for f in out["folds"] if f["heldout"] == "kinetics_03")
    assert third["heldout_safety_verdict"] == "FAIL"
    assert out["heldout_failed_safety"] >= 1


def test_zero_occlusion_support_is_not_claimed_safe():
    inp = fixture()
    for method in "ab":
        inp[0]["per_scene"]["kinetics_02"][method].update(
            invisible=0, invisible_false_accept=0, invisible_false_accept_rate=None)
    out = call(inp)
    fold = next(f for f in out["folds"] if f["heldout"] == "kinetics_02")
    assert fold["heldout_safety_verdict"] == "NO_INVISIBLE_SUPPORT"
    assert out["heldout_unevaluable_safety"] == 1


def test_no_eligible_candidate_uses_declared_fallback():
    inp = fixture()
    for scene in inp[0]["per_scene"].values():
        for method in "ab":
            scene[method]["visible_coverage"] = .1
    out = call(inp)
    assert out["full_dev_selected"] == "b"
    assert out["full_dev_eligible"] == []


@pytest.mark.parametrize("field", ["results", "predictions"])
def test_test_split_rejected(field):
    inp = fixture()
    inp[0 if field == "results" else 1]["split"] = "test"
    with pytest.raises(ValueError, match="DEV_ONLY_SPLIT"):
        call(inp)


def test_mixed_prediction_hash_rejected():
    inp = fixture()
    inp[0]["prediction_sha256"] = "new-invalid-hash"
    with pytest.raises(ValueError, match="MISMATCHED_PINNED"):
        call(inp)


def test_different_frozen_commit_rejected():
    inp = fixture()
    inp[1]["source_commit"] = "other"
    with pytest.raises(ValueError, match="MISMATCHED_PINNED"):
        call(inp)


def test_config_modified_after_freeze_rejected():
    inp = fixture()
    inp[2]["dev_selection"]["min_visible_coverage"] = .99
    with pytest.raises(ValueError, match="CONFIG_HASH"):
        call(inp)


def test_non_dev_scene_fails_closed():
    inp = fixture()
    inp[0]["per_scene"]["kinetics_05"]["a"]["visible"] = 50
    inp[0]["per_scene"]["kinetics_09"] = inp[0]["per_scene"].pop("kinetics_05")
    inp[1]["records"][-1]["name"] = "kinetics_09"
    inp[1]["scene_aux"]["kinetics_09"] = inp[1]["scene_aux"].pop("kinetics_05")
    with pytest.raises(ValueError, match="TEST_SCENE"):
        call(inp)


def test_duplicate_predictions_rejected():
    inp = fixture()
    inp[1]["records"][-1]["name"] = "kinetics_00"
    with pytest.raises(ValueError, match="PREDICTION_SCENE"):
        call(inp)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True, "0.8"])
def test_bad_rate_fails_closed(value):
    inp = fixture()
    inp[0]["per_scene"]["kinetics_00"]["a"]["visible_coverage"] = value
    with pytest.raises(ValueError, match="INVALID_RATE"):
        call(inp)


def test_missing_method_rejected():
    inp = fixture()
    del inp[0]["per_scene"]["kinetics_00"]["a"]
    with pytest.raises(ValueError, match="SCENE_METHOD"):
        call(inp)


def test_false_accept_greater_than_support_rejected():
    inp = fixture()
    inp[0]["per_scene"]["kinetics_00"]["a"]["invisible_false_accept"] = 200
    with pytest.raises(ValueError, match="FALSE_ACCEPT_EXCEEDS_SUPPORT"):
        call(inp)


def test_invalid_candidates_rejected():
    inp = fixture()
    inp[2]["dev_selection"]["candidate_methods"] = ["a", "a"]
    # Validator checks config integrity first.
    with pytest.raises(ValueError, match="CONFIG_HASH"):
        call(inp)


def test_write_once_and_pin(tmp_path):
    path = tmp_path / "nested" / "report.json"
    expected = M.write_once(path, {"split": "dev", "folds": []})
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    assert (tmp_path / "nested" / "report.json.sha256").read_text().strip() == expected
    with pytest.raises(FileExistsError, match="IMMUTABLE"):
        M.write_once(path, {"split": "dev", "folds": []})


def test_read_only_input_loader(tmp_path):
    path = tmp_path / "dev_results.json"
    path.write_text('{"split":"dev"}', encoding="utf8")
    obj, checksum = M.load_json(path)
    assert obj["split"] == "dev"
    assert checksum == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(list(tmp_path.iterdir())) == 1
