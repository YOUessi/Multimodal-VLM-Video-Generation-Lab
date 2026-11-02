"""Batch56: fail-closed independence, selection, VLM and write-once safeguards."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import kinetics_batch45_data as historical_module
import kinetics_batch56_data as data
import kinetics_batch56_eval as evaluation
import kinetics_batch56_vlm as vlm
import kinetics_batch56_report as report


def _selection():
    return {"videos": [
        {"video_id": f"NEW_UNSEEN_{i}", "label": f"new_action_{i}",
         "split": "dev" if i < 36 else "test"}
        for i in range(42)
    ]}


def test_historical_data_module_not_mutated_by_batch56_import():
    assert data.base is not historical_module
    assert historical_module.DATA == ROOT / "data/tapvid_kinetics_batch45e"
    assert data.DATA == ROOT / "data/tapvid_kinetics_batch56"
    assert data.FROZEN != historical_module.FROZEN
    assert data.base.SALT.startswith("MVVG-BATCH56")


def test_prior_history_has_60_unique_action_classes():
    labels, ids = data.historical_identifiers()
    assert len(labels) == 72 and len(ids) == 72
    assert data.historical_classes() == labels
    previous = json.loads((ROOT / "data/tapvid_kinetics_batch48/batch48_frozen/manifest.json").read_text())
    assert {v["action_class"] for v in previous["scenes"]}.issubset(labels)
    assert {v["video_id"] for v in previous["scenes"]}.issubset(ids)


def test_missing_historical_manifest_blocks_before_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "OLD_MANIFESTS", (tmp_path / "MISSING.json",))
    with pytest.raises(RuntimeError, match="REQUIRED_HISTORY_MISSING"):
        data.audit_preflight_disjointness(_selection())


def test_malformed_prior_manifest_blocks(tmp_path, monkeypatch):
    old = tmp_path / "prior.json"
    old.write_text(json.dumps({"scenes": [{"video_id": "a", "action_class": "b"}]}))
    monkeypatch.setattr(data, "OLD_MANIFESTS", (old,))
    with pytest.raises(RuntimeError, match="HISTORY_INVALID"):
        data.historical_identifiers()


def test_fresh_selection_succeeds_and_old_ids_fail():
    data.audit_preflight_disjointness(_selection())
    classes, ids = data.historical_identifiers()
    both = _selection()
    both["videos"][0]["video_id"] = sorted(ids)[0]
    with pytest.raises(RuntimeError, match="HISTORICAL_VIDEO_LEAK"):
        data.audit_preflight_disjointness(both)
    both = _selection()
    both["videos"][0]["label"] = sorted(classes)[0]
    with pytest.raises(RuntimeError, match="HISTORICAL_CLASS_LEAK"):
        data.audit_preflight_disjointness(both)


@pytest.mark.parametrize("change,error", [
    (lambda s: s["videos"].pop(), "COUNT"),
    (lambda s: s["videos"][2].update(video_id=s["videos"][1]["video_id"]), "DUPLICATE"),
    (lambda s: s["videos"][1].update(label=s["videos"][0]["label"]), "DUPLICATE"),
    (lambda s: s["videos"][0].update(split="test"), "SPLIT"),
])
def test_bad_selection_rejected(change, error):
    s = _selection()
    change(s)
    with pytest.raises(RuntimeError, match=error):
        data.audit_preflight_disjointness(s)


def test_selection_uses_all_historical_exclusions(monkeypatch):
    prior_classes, prior_ids = data.historical_identifiers()
    seen = {}

    def stub(candidates, labels, prior_videos, prior_labels, lines, hashes):
        seen["ids"] = set(prior_videos)
        seen["classes"] = set(prior_labels)
        return _selection()

    monkeypatch.setattr(data, "ORIGINAL_SELECT", stub)
    result = data.new_class_selection({}, [], set(), set(), [], {})
    assert result["videos"] == _selection()["videos"]
    assert prior_classes.issubset(seen["classes"])
    assert prior_ids.issubset(seen["ids"])


def test_guarded_existing_selection_checked_without_download(monkeypatch):
    monkeypatch.setattr(data, "ORIGINAL_BUILD_SELECTION",
                        lambda _: {**_selection(), "videos": [{**r} for r in _selection()["videos"]]})
    selected = data.guarded_build_selection([])
    assert len(selected["videos"]) == 42
    broken = _selection()
    broken["videos"][0]["label"] = sorted(data.historical_classes())[0]
    monkeypatch.setattr(data, "ORIGINAL_BUILD_SELECTION", lambda _: broken)
    with pytest.raises(RuntimeError, match="HISTORICAL_CLASS_LEAK"):
        data.guarded_build_selection([])


def test_historical_audit_blocks_old_test_class(tmp_path, monkeypatch):
    selected = _selection()
    selected["videos"][1]["label"] = sorted(data.historical_classes())[0]
    data_root = tmp_path / "frozen"
    data_root.mkdir()
    doc = {"scenes": [{"video_id": row["video_id"], "action_class": row["label"]}
                      for row in selected["videos"]]}
    (data_root / "manifest.json").write_text(json.dumps(doc))
    monkeypatch.setattr(data, "FROZEN", data_root)
    with pytest.raises(RuntimeError, match="CLASS_REUSE"):
        data.audit_historical_disjointness()


def test_batch56_eval_and_vlm_target_new_paths():
    assert evaluation.DATA == data.FROZEN
    assert evaluation.OUT == ROOT / "outputs/kinetics_batch56"
    assert evaluation.baseline.src is data.base
    assert vlm.reference.DATA == data.FROZEN
    assert vlm.reference.OUT == evaluation.OUT
    assert vlm.reference.ev is evaluation.baseline


def test_predeclared_safety_choice_and_ineligible_fallback(tmp_path, monkeypatch):
    core = evaluation.baseline.core
    cfg = json.loads((ROOT / "configs/kinetics_batch56.json").read_text())
    methods = cfg["dev_selection"]["candidate_methods"]
    # Safe candidate beats higher-Jaccard unsafe candidate; old TEST not referenced.
    scores = {m: {"visible_coverage": 0.61, "invisible_false_accept_rate": 0.3,
                  "jaccard_mean_1_2_4_8_16": 0.8}
              for m in methods}
    good = methods[3]
    scores[good] = {"visible_coverage": 0.40, "invisible_false_accept_rate": 0.08,
                    "jaccard_mean_1_2_4_8_16": 0.25}
    monkeypatch.setattr(core, "OUT", tmp_path)
    monkeypatch.setattr(core, "gitsha", lambda: "a" * 40)
    report_bytes = core.json_bytes({
        "split": "dev", "source_commit": "a" * 40,
        "config_sha256": core.sha_bytes(core.json_bytes(cfg)), "scene_macro": scores,
    })
    (tmp_path / "dev_results.json").write_bytes(report_bytes)
    chosen = core.select_dev(cfg)
    assert chosen["selected_method"] == good
    assert chosen["eligible_candidates"] == [good]
    assert chosen["selection_source"] == "DEV only"


def test_fallback_is_ineligible_when_no_dev_method_passes(tmp_path, monkeypatch):
    core = evaluation.baseline.core
    cfg = json.loads((ROOT / "configs/kinetics_batch56.json").read_text())
    scores = {m: {"visible_coverage": 0.01, "invisible_false_accept_rate": 0.8,
                  "jaccard_mean_1_2_4_8_16": 0.01}
              for m in cfg["dev_selection"]["candidate_methods"]}
    monkeypatch.setattr(core, "OUT", tmp_path)
    monkeypatch.setattr(core, "gitsha", lambda: "b" * 40)
    cfg_hash = core.sha_bytes(core.json_bytes(cfg))
    (tmp_path / "dev_results.json").write_bytes(core.json_bytes({
        "split": "dev", "source_commit": "b" * 40,
        "config_sha256": cfg_hash, "scene_macro": scores,
    }))
    selection = core.select_dev(cfg)
    assert not selection["eligible_candidates"]
    assert selection["selected_method"] == cfg["dev_selection"]["fallback_method"]


def test_immutable_plan_never_opens_test_score_before_predictions_and_dev(tmp_path):
    plan = json.loads((ROOT / "automation/plans/batch56_kinetics_72_class_disjoint_2026-11-03.json").read_text())
    stages = plan["stages"]
    identifiers = [s["id"] for s in stages]
    assert len(stages) == len(set(identifiers)) == 23
    positions = {s["id"].split("_")[0]: i for i, s in enumerate(stages)}
    assert positions["R06"] < positions["R07"] < positions["R11"] < positions["R13"]
    assert positions["R10"] < positions["R12"] < positions["R14"]
    assert stages[19]["requires_files"] == [
        "tests/test_kinetics_batch56.py",
        "outputs/kinetics_batch56/BATCH56_FRESH_SAFETY_REPORT.md",
    ]
    for s in stages:
        assert s["max_attempts"] == 1
    assert "no opened TEST retuning" in plan["description"]


def test_research_config_excludes_old_test_promotion():
    cfg = json.loads((ROOT / "configs/kinetics_batch56.json").read_text())
    assert cfg["dev_videos"] == 36 and cfg["test_videos"] == 6
    assert cfg["queries_per_video"] == 4
    assert cfg["dev_selection"]["max_invisible_false_accept_rate"] == 0.1
    assert cfg["dev_selection"]["min_visible_coverage"] == 0.3
    assert cfg["dev_selection"]["selection_source"] == "NEW_BATCH56_DEV_ONLY"
    assert cfg["dev_selection"]["fallback_method"] in cfg["dev_selection"]["candidate_methods"]
    assert report.REPORT.name == "BATCH56_FRESH_SAFETY_REPORT.md"


def test_independent_report_recompute_catches_tampered_dev_choice():
    cfg = json.loads((ROOT / "configs/kinetics_batch56.json").read_text())
    methods = cfg["dev_selection"]["candidate_methods"]
    metric = {name: {
        "visible_coverage": .48, "invisible_false_accept_rate": .28,
        "jaccard_mean_1_2_4_8_16": .5,
    } for name in methods}
    metric[methods[0]] = {
        "visible_coverage": .41, "invisible_false_accept_rate": .07,
        "jaccard_mean_1_2_4_8_16": .23,
    }
    metric[methods[1]] = {
        "visible_coverage": .51, "invisible_false_accept_rate": .06,
        "jaccard_mean_1_2_4_8_16": .29,
    }
    selected = {"eligible_candidates": methods[:2],
                "selected_method": methods[1],
                "rules": cfg["dev_selection"]}
    report.validate_dev_only_choice(cfg, {"scene_macro": metric}, selected)
    with pytest.raises(RuntimeError, match="SELECTION_RECOMPUTE"):
        report.validate_dev_only_choice(
            cfg, {"scene_macro": metric}, {**selected, "selected_method": methods[0]})
    with pytest.raises(RuntimeError, match="SELECTION_RECOMPUTE"):
        report.validate_dev_only_choice(
            cfg, {"scene_macro": metric}, {**selected, "eligible_candidates": methods[:1]})
    with pytest.raises(RuntimeError, match="SELECTION_RECOMPUTE"):
        report.validate_dev_only_choice(
            cfg, {"scene_macro": metric}, {**selected, "rules": {"fallback_method": "TEST"}})


def test_historic_batch53b_is_part_of_60_class_exclusion():
    labels, ids = data.historical_identifiers()
    path = ROOT / "data/tapvid_kinetics_batch53/batch53_frozen/manifest.json"
    old = json.loads(path.read_text())
    assert len(old["scenes"]) == 12
    assert all(s["action_class"] in labels and s["video_id"] in ids for s in old["scenes"])


def test_current_batch_preserves_6_6_split_and_old_global_module():
    assert data.base is not historical_module
    assert data.base.DEV_VIDEOS == 36 and data.base.TEST_VIDEOS == 6
    assert historical_module.SALT.startswith("MVVG-BATCH45")
    assert data.base.SALT.startswith("MVVG-BATCH56")


def test_disk_preflight_fails_before_dataset_download(monkeypatch):
    import shutil
    from types import SimpleNamespace
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=9 * 1024 ** 3))
    monkeypatch.setattr(sys, "argv", ["kinetics_batch56_data.py", "download"])
    monkeypatch.setattr(data.base, "main", lambda: pytest.fail("download invoked despite disk guard"))
    with pytest.raises(RuntimeError, match="BATCH56_DISK_PREFLIGHT_GUARD"):
        data.main()


def test_plan_exact_one_writer_all_gpu_stages_guarded():
    plan = json.loads((ROOT / "automation/plans/batch56_kinetics_72_class_disjoint_2026-11-03.json").read_text())
    assert plan["max_stages_per_tick"] == 2
    assert len(plan["stages"]) == 23
    assert all(s["max_attempts"] == 1 for s in plan["stages"])
    assert all(s.get("min_free_mib", 0) >= 11000 for s in plan["stages"] if s.get("gpu"))
    assert plan["stages"][12]["id"].startswith("R11_TEST_CPU_PREDICTIONS_PIN")
    assert plan["stages"][14]["id"].startswith("R13_TEST_CPU_SCORING_ONCE")


def test_batch56_new_history_explicitly_contains_batch55_id_and_classes():
    labels, ids = data.historical_identifiers()
    prior = json.loads((ROOT / "data/tapvid_kinetics_batch55/batch55_frozen/manifest.json").read_text())
    assert len(prior["scenes"]) == 12
    assert {x["action_class"] for x in prior["scenes"]} <= labels
    assert {x["video_id"] for x in prior["scenes"]} <= ids


def test_batch56_frozen_population_is_36_dev_6_test():
    cfg = json.loads((ROOT / "configs/kinetics_batch56.json").read_text())
    assert (cfg["dev_videos"], cfg["test_videos"], cfg["queries_per_video"]) == (36, 6, 4)
    assert data.base.DEV_VIDEOS == 36
    assert data.base.TEST_VIDEOS == 6
    assert [v["split"] for v in _selection()["videos"]] == ["dev"] * 36 + ["test"] * 6


def test_frozen_plan_audits_independent_dev_scenes_before_test():
    plan = json.loads((ROOT / "automation/plans/batch56_kinetics_72_class_disjoint_2026-11-03.json").read_text())
    names=[x["id"] for x in plan["stages"]]
    assert len(names)==23
    assert names.index("R07_NEW_DEV_SAFETY_METHOD_SELECTION") < names.index("R07A_DEV_SCENE_CLUSTER_RISK_AUDIT")
    assert names.index("R07A_DEV_SCENE_CLUSTER_RISK_AUDIT") < names.index("R11_TEST_CPU_PREDICTIONS_PIN")
    assert "test" not in plan["stages"][8]["argv"][-1].lower()
