"""Auditor accounting and fail-closed provenance checks use synthetic fixtures."""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import parallax_gate_risk_audit as audit
from domain_shift_motion_data import LABELS


def build_example():
    ordered = [LABELS[x] for x in ("down_left", "down_right", "stationary", "up_right", "up_left")]
    examples = []
    # all: 2 independent scenes and 4 dependent forward/reverse clips.
    examples += [
        ("0f", "s0", "up_left", "E", "A", True, "up_left"),  # rescue
        ("0r", "s0", "down_right", "B", "A", True, "up_left"), # harm
        ("1f", "s1", "up_right", "D", "D", False, None),    # same
        ("1r", "s1", "down_left", "A", "B", True, "up_left"), # wrong-both
    ]
    rows, cv, model = [], {}, {}
    for identifier, pair_id, truth, gold_letter, vlm, accepted, direction in examples:
        rows.append({"id": identifier, "pair_id": pair_id, "gold_direction": truth,
                     "gold_letter": gold_letter, "choices": ordered, "variant": "example"})
        cv[identifier] = {"predictions": {"affine": {
            "accepted": accepted, "gated_direction": direction}}}
        model[(identifier, 3)] = {"prediction": vlm}
    # In this fixture the choices and letters are consistent with truth.
    return rows, cv, model


def test_rescue_harm_accounting_and_full_denominator():
    rows, cv, models = build_example()
    r = audit.risk_counts(rows, cv, models, "affine", 3)
    assert r["n"] == 4
    assert r["vlm_correct"] == 1
    assert r["accepted"] == 3
    assert r["rescue"] == 1
    assert r["harm"] == 0  # second clip: VLM A was also wrong; not a harm
    assert r["wrong_both"] == 2
    assert r["gate_wrong"] == 2
    assert r["hybrid_correct"] == 2
    assert r["net_change_correct"] == 1


def test_accepted_direction_without_valid_choice_is_not_silently_ignored():
    rows, cv, models = build_example()
    cv["0f"]["predictions"]["affine"]["gated_direction"] = "nonsense"
    with pytest.raises(RuntimeError, match="ACCEPTED_GATE_WITHOUT_VALID_DIRECTION"):
        audit.risk_counts(rows, cv, models, "affine", 3)


def test_missing_or_duplicate_prediction_rows_fail_closed():
    doc = {"records": [{"id": "a"}, {"id": "a"}]}
    with pytest.raises(RuntimeError, match="COVERAGE_OR_DUPLICATE"):
        audit.complete_records(doc, {"a", "b"}, lambda r: r["id"], "TEST")


def test_file_hash_check_and_non_git_tree_fail_closed(tmp_path):
    c = {"source_commit": "bad", "artifact_sha256": {"file": "deadbeef"}}
    with pytest.raises(RuntimeError, match="SOURCE_ROOT_NOT_GIT"):
        audit.verify_frozen_source(tmp_path, c)
    f = tmp_path / "one.txt"
    f.write_text("content")
    assert len(audit.sha256(f)) == 64


def test_risk_report_never_includes_prediction_media(tmp_path):
    r = {"source_commit": "sha", "independent_scene_pairs": 1, "test_clips": 2, "model_risks": {
        "dummy": {"3": {"affine": {
            "all": {"vlm_correct": 1, "n": 2, "hybrid_correct": 2, "accepted": 1,
                    "gate_wrong": 0, "rescue": 1, "harm": 0, "net_change_correct": 1},
            "paired_scene_delta": {"cluster_bootstrap_95ci_pp": [0, 100]},
        }}},
    }}
    audit.save_report(r, tmp_path)
    assert "post-hoc" in (tmp_path / "RISK_AUDIT.md").read_text()
    assert not list(tmp_path.glob("*.mp4"))
