"""Batch48 independent TEST report: validate provenance, DEV choice and all pins."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import kinetics_batch48_data as fresh
import kinetics_batch48_eval as experiment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/kinetics_batch48"
REPORT = OUT / "BATCH48_FRESH_SAFETY_REPORT.md"
METHOD_COLUMNS = ("pck8_visible_unconditional", "visible_coverage",
                  "accepted_precision8", "invisible_false_accept_rate",
                  "jaccard_mean_1_2_4_8_16", "cpu_ms_per_target_frame_p50")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(filename: str):
    return json.loads((OUT / filename).read_text(encoding="utf-8"))


def required_pin(name: str) -> str:
    path = OUT / name
    tag = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not tag.is_file() or tag.read_text().strip() != sha(path):
        raise RuntimeError("BATCH48_PIN_MISMATCH:" + name)
    return sha(path)


def validate_dev_only_choice(cfg: dict, dev_results: dict, selected: dict) -> None:
    """Independently recompute policy eligibility and ranking without TEST scores."""
    rules = cfg["dev_selection"]
    methods = set(rules["candidate_methods"])
    dev_macro = dev_results["scene_macro"]
    expected_eligible = [
        name for name in rules["candidate_methods"]
        if (dev_macro[name]["visible_coverage"] is not None
            and dev_macro[name]["invisible_false_accept_rate"] is not None
            and dev_macro[name]["visible_coverage"] >= rules["min_visible_coverage"]
            and dev_macro[name]["invisible_false_accept_rate"] <= rules["max_invisible_false_accept_rate"])
    ]
    expected_method = (
        max(expected_eligible, key=lambda name: (
            dev_macro[name]["jaccard_mean_1_2_4_8_16"],
            dev_macro[name]["visible_coverage"]))
        if expected_eligible else rules["fallback_method"]
    )
    if (selected.get("eligible_candidates") != expected_eligible
            or selected.get("selected_method") != expected_method
            or selected.get("rules") != rules):
        raise RuntimeError("BATCH48_DEV_ONLY_SELECTION_RECOMPUTE_MISMATCH")


def validate() -> dict:
    manifest = fresh.base.audit_full()
    fresh.audit_historical_disjointness()
    if len(manifest["scenes"]) != 12:
        raise RuntimeError("BATCH48_WRONG_SCENE_COUNT")
    cfg = read_config()
    config_sha = experiment.baseline.sha_bytes(experiment.baseline.json_bytes(cfg))
    pinned = {name: required_pin(name) for name in (
        "dev_predictions.json", "test_predictions.json", "dev_selection.json",
        "dev_vlm_predictions.json", "test_vlm_predictions.json", "dev_vlm_gate.json",
    )}
    docs = {name: read(name) for name in (
        "dev_results.json", "test_results.json", "dev_selection.json",
        "dev_vlm_results.json", "test_vlm_results.json", "dev_vlm_gate.json",
        "test_predictions.json",
    )}
    source_sha = experiment.baseline.gitsha()
    for name, obj in docs.items():
        if obj.get("source_commit") != source_sha or obj.get("config_sha256") != config_sha:
            raise RuntimeError("BATCH48_SOURCE_OR_CONFIG_DRIFT:" + name)
    selected = docs["dev_selection.json"]
    validate_dev_only_choice(cfg, docs["dev_results.json"], selected)
    methods = set(cfg["dev_selection"]["candidate_methods"])
    if (selected.get("selection_source") != "DEV only"
            or selected.get("selected_method") not in methods
            or not set(selected.get("eligible_candidates", [])).issubset(methods)
            or selected.get("dev_results_sha256") != sha(OUT / "dev_results.json")
            or docs["test_predictions.json"].get("dev_selection_sha256") != pinned["dev_selection.json"]):
        raise RuntimeError("BATCH48_METHOD_SELECTION_FIREWALL")
    for split in ("dev", "test"):
        result = docs[f"{split}_results.json"]
        if result.get("split") != split or result.get("prediction_sha256") != pinned[f"{split}_predictions.json"]:
            raise RuntimeError("BATCH48_PREDICTION_BEFORE_SCORE_PIN_MISMATCH:" + split)
    gate = docs["dev_vlm_gate.json"]
    if gate["status"] not in ("BLOCK", "PASS"):
        raise RuntimeError("BATCH48_VLM_UNKNOWN_DEV_GATE")
    if gate["status"] == "PASS" and docs["test_vlm_results.json"].get("status") != "SCORED":
        raise RuntimeError("BATCH48_PASSED_VLM_TEST_SCORE_MISSING")
    if gate["status"] == "BLOCK":
        test_vlm = docs["test_vlm_results.json"]
        if (test_vlm.get("status") != "SKIPPED_DEV_GATE_BLOCKED"
                or test_vlm.get("test_oracle_read") is not False):
            raise RuntimeError("BATCH48_BLOCKED_VLM_TEST_LEAK")
    return {"manifest": manifest, "cfg": cfg, "selected": selected, "docs": docs,
            "pins": pinned, "source_sha": source_sha}


def read_config() -> dict:
    return json.loads((ROOT / "configs/kinetics_batch48.json").read_text(encoding="utf-8"))


def _fmt(key: str, value) -> str:
    if value is None:
        return "n/a"
    if key in ("pck8_visible_unconditional", "visible_coverage",
               "accepted_precision8", "invisible_false_accept_rate"):
        return f"{value:.1%}"
    return f"{value:.3f}"


def render(evidence: dict) -> str:
    d = evidence["docs"]
    selected = evidence["selected"]
    method = selected["selected_method"]
    eligible = selected["eligible_candidates"]
    policy = "DEV_ADMITTED" if method in eligible else "FALLBACK_NOT_DEV_ADMITTED"
    dev = d["dev_results.json"]["scene_macro"][method]
    test = d["test_results.json"]["scene_macro"][method]
    header = [
        "# Batch48 — new class-disjoint Kinetics DEV selection and sealed TEST",
        "",
        f"- Frozen GitHub source SHA: `{evidence['source_sha']}`.",
        "- 12 fresh class-disjoint Kinetics natural videos, 6 DEV and 6 TEST, four official point queries each.",
        "- Excludes all video IDs and action classes in Batch42, Batch45E, and Batch46E by historical metadata.",
        f"- Method selected before TEST from fresh DEV: **{method}**; status **{policy}**.",
        f"- Frozen DEV constraints: coverage >= {evidence['cfg']['dev_selection']['min_visible_coverage']:.0%}; "
        f"invisible false accept <= {evidence['cfg']['dev_selection']['max_invisible_false_accept_rate']:.0%}.",
        f"- Eligible on DEV: {', '.join(eligible) if eligible else 'NONE; fallback is not a safety pass'}.",
        f"- TEST point prediction pin: `{evidence['pins']['test_predictions.json']}`.",
        f"- New manifest pin: `{sha(fresh.FROZEN / 'manifest.json')}`.",
        "",
        "## Predeclared selected method — DEV and sealed TEST",
        "",
        "| Metric | DEV | TEST |",
        "|---|---:|---:|",
    ]
    for key in METHOD_COLUMNS:
        header.append(f"| {key} | {_fmt(key, dev[key])} | {_fmt(key, test[key])} |")
    header.extend([
        "",
        "## VLM reference and admission",
        "",
        f"- Frozen SmolVLM2 three-image DEV gate: **{d['dev_vlm_gate.json']['status']}**.",
        f"- DEV balanced accuracy: {_fmt('accuracy', d['dev_vlm_results.json']['vlm']['balanced_accuracy'])}.",
        f"- TEST VLM status: **{d['test_vlm_results.json']['status']}**.",
        "",
        "## Interpretation",
        "",
        "- All methods appear in the frozen TEST result as post-hoc diagnostics; do not choose a new method based on these opened scores.",
        "- A method that fails the DEV constraints is explicitly not admitted even if its later TEST metrics appear favorable.",
        "- Sparse scenes and changed visibility prevalence can produce large DEV/TEST differences.",
        "- Jaccard, coverage, precision and occluded false acceptance must be considered jointly.",
        "- CPU figures exclude shared affine preprocessing, decoding and full end-to-end latency.",
        "- Estimated affine background-relative image-plane residual is not calibrated camera pose, true physical velocity or 3D movement.",
        "- VLM uses three still images, whereas CPU trackers see dense intermediate frames.",
        "- Videos, frames, private oracle, query predictions, weights and secrets remain on Tang; only aggregate statistics can be published.",
        "",
    ])
    return "\n".join(header)


def build() -> dict:
    report = render(validate()).encode("utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    if REPORT.is_file():
        if REPORT.read_bytes() != report:
            raise RuntimeError("BATCH48_REPORT_IMMUTABILITY_BREACH")
    else:
        REPORT.write_bytes(report)
    digest = hashlib.sha256(report).hexdigest()
    tag = REPORT.with_suffix(REPORT.suffix + ".sha256")
    if tag.exists():
        if tag.read_text().strip() != digest:
            raise RuntimeError("BATCH48_REPORT_PIN_MISMATCH")
    else:
        tag.write_text(digest + "\n")
    return {"report_sha256": digest, "bytes": len(report)}


def verify() -> dict:
    digest = required_pin(REPORT.name)
    expected = render(validate()).encode("utf-8")
    if REPORT.read_bytes() != expected:
        raise RuntimeError("BATCH48_REPORT_RECOMPUTE_MISMATCH")
    return {"result": "BATCH48_REPORT_AUDIT_PASS", "sha256": digest}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("build", "verify"))
    args = p.parse_args()
    print(json.dumps(build() if args.action == "build" else verify(), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
