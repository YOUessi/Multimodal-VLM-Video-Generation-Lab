"""Independence, prior-selection pinning, and one-time evaluation firewall tests."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import kinetics_batch46_data as data
import kinetics_batch46_eval as evaluation
import kinetics_batch46_report as report


def _manifest(path, labels, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "scenes": [{"action_class": c, "video_id": i} for c, i in zip(labels, ids)]
    }))


def test_historical_classes_require_both_manifests(tmp_path, monkeypatch):
    a, b = tmp_path / "old1.json", tmp_path / "old2.json"
    _manifest(a, ["a"], ["clip_a"])
    _manifest(b, ["b"], ["clip_b"])
    monkeypatch.setattr(data, "OLD_MANIFESTS", (a, b))
    assert data.historical_classes() == {"a", "b"}
    b.unlink()
    with pytest.raises(RuntimeError, match="BATCH46_REQUIRED_HISTORY_MISSING"):
        data.historical_classes()


def test_never_reselect_old_action_class(tmp_path, monkeypatch):
    a, b = tmp_path / "old1.json", tmp_path / "old2.json"
    _manifest(a, ["class_000", "class_001"], ["id000", "id001"])
    _manifest(b, ["class_002"], ["id002"])
    monkeypatch.setattr(data, "OLD_MANIFESTS", (a, b))
    labels = [f"class_{i:03d}" for i in range(700)]
    candidates = {label: [{"video_id": f"id{i:03d}", "label": label,
                            "time_start": "0", "time_end": "10",
                            "track_rows": 5, "query_candidates": 4}]
                  for i, label in enumerate(labels)}
    urls = [f"https://example.test/k700_val_{i:03d}.tar.gz"
            for i in range(1, 701)]
    hashes = {k: "x" for k in ("source_csv_sha256", "kinetics_val_metadata_sha256",
                             "val_ids_sha256", "path_list_sha256")}
    selected = data.new_class_selection(candidates, labels, {"id005"},
                                        {"class_003"}, urls, hashes)
    chosen_labels = {x["label"] for x in selected["videos"]}
    chosen_ids = {x["video_id"] for x in selected["videos"]}
    assert len(chosen_labels) == len(chosen_ids) == 12
    assert not chosen_labels & {"class_000", "class_001", "class_002", "class_003"}
    assert "id005" not in chosen_ids
    assert {"class_000", "class_001", "class_002"} <= set(
        selected["previous_action_classes_excluded"])


def test_previous_dev_selection_sha_fails_closed(tmp_path, monkeypatch):
    prior = tmp_path / "old_selection.json"
    prior.write_text('{"selected_method":"affine_stabilized_klt"}')
    monkeypatch.setattr(evaluation, "PRIOR_SELECTION", prior)
    with pytest.raises(RuntimeError, match="BATCH46_PREVIOUS_DEV_SELECTION_SHA_DRIFT"):
        evaluation.verify_prior_selection()


def test_transfer_selection_cannot_be_changed_by_new_dev_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "OUT", tmp_path)
    monkeypatch.setattr(evaluation, "verify_prior_selection", lambda: {})
    monkeypatch.setattr(evaluation.previous, "gitsha", lambda: "frozen46")
    cfg = json.loads((Path(__file__).resolve().parents[1] /
                      "configs/kinetics_batch46.json").read_text())
    config_sha = evaluation.previous.sha_bytes(evaluation.previous.json_bytes(cfg))
    (tmp_path / "dev_results.json").write_text(json.dumps({
        "source_commit": "frozen46", "split": "dev", "config_sha256": config_sha,
        "scene_macro": {"klt_forward": {"jaccard_mean_1_2_4_8_16": 1.0},
                        "affine_stabilized_klt": {"jaccard_mean_1_2_4_8_16": 0.0}},
    }))
    record = evaluation.select_pre_registered(cfg)
    assert record["selected_method"] == "affine_stabilized_klt"
    assert record["selection_source"] == "PREVIOUS_BATCH45E_DEV_ONLY_FROZEN"
    evaluation.previous.core.require_pin(tmp_path / "dev_selection.json")


def test_final_report_pin_refuses_changed_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "OUT", tmp_path)
    (tmp_path / "test_predictions.json").write_text("changed")
    with pytest.raises(RuntimeError, match="BATCH46_OUTPUT_PIN_DRIFT"):
        report.require_pinned("test_predictions.json")
    p = tmp_path / "test_predictions.json"
    p.with_suffix(".json.sha256").write_text(report.digest(p) + "\n")
    assert report.require_pinned("test_predictions.json") == report.digest(p)
    p.write_text("edited")
    with pytest.raises(RuntimeError, match="BATCH46_OUTPUT_PIN_DRIFT"):
        report.require_pinned("test_predictions.json")


def test_report_transfer_method_constant():
    assert report.METHOD == evaluation.FROZEN_METHOD
    assert report.METHOD in evaluation.previous.METHODS

def test_frozen_plan_download_timeout_covers_all_shard_attempt_budgets():
    plan = json.loads((Path(__file__).resolve().parents[1] /
                       "automation/plan.json").read_text())
    stage = next(x for x in plan["stages"]
                 if x["id"] == "R00_SELECT_AND_DOWNLOAD_12_FRESH_CLASS_SHARDS")
    # Twelve sequential shard attempts can each consume the 2,460 s curl cap.
    assert stage["timeout_seconds"] >= 12 * 2460 + 600


def test_frozen_plan_paths_match_new_dataset_and_outputs():
    plan = json.loads((Path(__file__).resolve().parents[1] /
                       "automation/plan.json").read_text())
    entries = json.dumps(plan["stages"])
    assert "kinetics_batch46e" not in entries
    assert "data/tapvid_kinetics_batch46/batch46_frozen" in entries
    assert "outputs/kinetics_batch46/test_predictions.json.sha256" in entries
    assert "data/tapvid_kinetics_batch45e/preflight/k700_2020_val_path.txt" in entries
    assert all(stage.get("max_attempts") == 1 for stage in plan["stages"])
