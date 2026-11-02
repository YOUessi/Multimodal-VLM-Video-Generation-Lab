"""DEV-only scene-cluster occlusion risk audit for prospective Batch59.

Never reads TEST results, TEST predictions or private oracles. It quantifies
how many distinct natural videos support a scene-level safety conclusion.
The confidence bound requires unverified independent/exchangeable scene draws.
"""
from __future__ import annotations
import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/kinetics_batch59"
CONFIG = ROOT / "configs/kinetics_batch59.json"
ALPHA = 0.05


def canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_pinned(path: Path) -> dict:
    tag = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not tag.is_file() or tag.read_text().strip() != digest(path):
        raise RuntimeError("BATCH59_MISSING_OR_INVALID_PIN:" + path.name)
    return json.loads(path.read_text())


def cdf_leq(n: int, k: int, p: float) -> float:
    if p <= 0: return 1.0
    if p >= 1: return 0.0 if k < n else 1.0
    return sum(math.comb(n, i) * p**i * (1-p)**(n-i) for i in range(k+1))


def one_sided_upper(violations: int, supported: int, alpha: float) -> float | None:
    if not (0 < alpha < 1) or not (0 <= violations <= supported):
        raise ValueError("BATCH59_BINOMIAL_INPUT_INVALID")
    if supported == 0:
        return None
    if violations == supported:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(90):
        mid = (lo+hi)/2
        if cdf_leq(supported, violations, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def compute(dev: dict, selected: dict, cfg: dict) -> dict:
    methods = cfg["dev_selection"]["candidate_methods"]
    if dev.get("split") != "dev" or selected.get("split") != "dev":
        raise RuntimeError("BATCH59_DEV_ONLY_FIREWALL")
    scenes = dev.get("per_scene")
    expected = {f"kinetics_{i:02d}" for i in range(36)}
    if not isinstance(scenes, dict) or set(scenes) != expected:
        raise RuntimeError("BATCH59_DEV_SCENE_POPULATION_INVALID")
    if selected.get("selected_method") not in methods:
        raise RuntimeError("BATCH59_SELECTED_METHOD_INVALID")
    if selected.get("dev_results_sha256") != hashlib.sha256(canonical(dev)).hexdigest():
        raise RuntimeError("BATCH59_DEV_SELECTION_NOT_BOUND_TO_RESULTS")
    threshold = cfg["dev_selection"]["max_invisible_false_accept_rate"]
    if not 0 <= threshold <= 1 or len(set(methods)) != len(methods):
        raise RuntimeError("BATCH59_POLICY_INVALID")
    summary = {}
    for method in methods:
        supported = bad = 0
        unsupported = []
        counts = []
        for scene, row in sorted(scenes.items()):
            record = row.get(method)
            if not isinstance(record,dict):
                raise RuntimeError("BATCH59_MISSING_METHOD:" + scene + ":" + method)
            n, false_accept = record.get("invisible"), record.get("invisible_false_accept")
            if (not isinstance(n,int) or isinstance(n,bool) or
                not isinstance(false_accept,int) or isinstance(false_accept,bool) or
                n < 0 or false_accept < 0 or false_accept > n):
                raise RuntimeError("BATCH59_BAD_SCENE_COUNT:" + scene)
            if n == 0:
                unsupported.append(scene)
                continue
            rate = false_accept / n
            if abs(rate - record["invisible_false_accept_rate"]) > 1e-8:
                raise RuntimeError("BATCH59_SCENE_RATE_INCONSISTENT:" + scene)
            supported += 1
            bad += rate > threshold
            counts.append((n, false_accept))
        summary[method] = {
            "supported_scenes": supported,
            "unsupported_scenes": len(unsupported),
            "unsupported_scene_ids": unsupported,
            "violating_scenes": bad,
            "max_observed_scene_false_accept": max((f/n for n,f in counts), default=None),
            "nominal_95_upper_scene_violation_probability": one_sided_upper(bad,supported,ALPHA),
            "nine_method_familywise_95_upper": one_sided_upper(bad,supported,ALPHA/len(methods)),
            "supported_scene_occlusion_frames": sum(n for n,_ in counts),
        }
    return {
        "kind": "DEV_ONLY_SCENE_CLUSTER_RISK_DIAGNOSTIC",
        "split": "dev",
        "source_commit": dev["source_commit"],
        "config_sha256": dev.get("config_sha256"),
        "policy_threshold": threshold,
        "method_family_size": len(methods),
        "selected_method": selected["selected_method"],
        "selected_method_dev_admitted": selected["selected_method"] in selected["eligible_candidates"],
        "methods": summary,
        "interpretation": "Bounds refer to future scene-level rate-budget violations, NOT frame-level false-accept risk. Requires scene exchangeability/i.i.d., not established for Kinetics; not a safety certificate.",
        "test_oracle_read": False,
        "test_predictions_read": False,
        "historical_test_scores_used": False,
    }


def audit() -> dict:
    cfg = json.loads(CONFIG.read_text())
    dev_path = OUT / "dev_results.json"
    dev = json.loads(dev_path.read_text())
    selected = read_pinned(OUT / "dev_selection.json")
    source = subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    if dev.get("source_commit") != source or selected.get("source_commit") != source:
        raise RuntimeError("BATCH59_SOURCE_DRIFT")
    from kinetics_batch59_eval import baseline
    if dev.get("config_sha256") != baseline.sha_bytes(baseline.json_bytes(cfg)):
        raise RuntimeError("BATCH59_CONFIG_DRIFT")
    if selected.get("dev_results_sha256") != digest(dev_path):
        raise RuntimeError("BATCH59_DEV_RESULT_PIN_MISMATCH")
    return compute(dev,selected,cfg)


def main() -> None:
    result = audit()
    dst = OUT / "dev_scene_risk_audit.json"
    payload = canonical(result)
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists() and dst.read_bytes() != payload:
        raise RuntimeError("BATCH59_SCENE_RISK_IMMUTABLE_CONFLICT")
    if not dst.exists():
        dst.write_bytes(payload)
    tag = dst.with_suffix(".json.sha256")
    sha = hashlib.sha256(payload).hexdigest()
    if tag.exists() and tag.read_text().strip() != sha:
        raise RuntimeError("BATCH59_SCENE_RISK_PIN_MISMATCH")
    if not tag.exists():
        tag.write_text(sha+"\n")
    summary = result["methods"][result["selected_method"]]
    print("BATCH59_DEV_SCENE_RISK_PINNED",sha,json.dumps(summary,sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
