"""Conservative DEV-only clip-cluster safety support audit.

Never opens TEST data. An independent video/action class, NOT a target frame,
is the analysis unit. Bounds remain descriptive because natural video classes
are not a random i.i.d. sample and method selection already used DEV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ALPHA = 0.05


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def exact_one_sided_binomial_upper(failures: int, scenes: int, alpha: float) -> float | None:
    """Clopper-Pearson P(unsafe *scene*) upper bound, not a frame-level bound."""
    if isinstance(failures, bool) or isinstance(scenes, bool):
        raise ValueError("INVALID_INTEGER_TYPE")
    if not isinstance(failures, int) or not isinstance(scenes, int):
        raise ValueError("INVALID_INTEGER_TYPE")
    if scenes < 0 or not 0 <= failures <= scenes or not 0 < alpha < 1:
        raise ValueError("INVALID_BINOMIAL_ARGUMENT")
    if scenes == 0:
        return None
    if failures == scenes:
        return 1.0
    if failures == 0:
        return 1.0 - alpha ** (1.0 / scenes)

    def cdf(p: float) -> float:
        return sum(
            math.comb(scenes, j) * p ** j * (1.0 - p) ** (scenes - j)
            for j in range(failures + 1)
        )

    low, high = 0.0, 1.0
    for _ in range(90):
        mid = (low + high) / 2
        if cdf(mid) > alpha:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def zero_failure_scenes_needed(alpha: float, maximum_risk: float) -> int:
    if not 0 < alpha < 1 or not 0 < maximum_risk < 1:
        raise ValueError("INVALID_TARGET_RISK")
    return math.ceil(math.log(alpha) / math.log1p(-maximum_risk))


def read_pinned_dev(batch_root: Path) -> tuple[dict, dict, dict, dict]:
    """Read fixed DEV paths only, with frozen commit and SHA provenance."""
    base = batch_root.resolve()
    state = json.loads((base / ".automation/state.json").read_text())
    if not isinstance(state.get("frozen_sha"), str):
        raise ValueError("FROZEN_SHA_REQUIRED")
    version = state.get("plan_version", "")
    # Explicit allow-list; no dynamic path based on untrusted JSON fields.
    if version.startswith("batch48-"):
        identifier = "48"
    elif version.startswith("batch53b-") or version.startswith("batch53-"):
        identifier = "53"
    else:
        raise ValueError("UNKNOWN_FROZEN_DEV_BATCH")
    directory = base / "outputs" / f"kinetics_batch{identifier}"
    results_raw = (directory / "dev_results.json").read_bytes()
    selection_raw = (directory / "dev_selection.json").read_bytes()
    config_raw = (base / "configs" / f"kinetics_batch{identifier}.json").read_bytes()
    results = json.loads(results_raw)
    selected = json.loads(selection_raw)
    config = json.loads(config_raw)
    expected_cfg_sha = sha(canonical_json(config))
    if results.get("split") != "dev" or selected.get("split") != "dev":
        raise ValueError("DEV_ONLY_INPUT_REQUIRED")
    if selected.get("selection_source") != "DEV only":
        raise ValueError("DEV_ONLY_SELECTION_REQUIRED")
    if (results.get("source_commit") != state["frozen_sha"]
            or selected.get("source_commit") != state["frozen_sha"]):
        raise ValueError("SOURCE_SHA_MISMATCH")
    if (results.get("config_sha256") != expected_cfg_sha
            or selected.get("config_sha256") != expected_cfg_sha
            or selected.get("dev_results_sha256") != sha(results_raw)):
        raise ValueError("PIN_HASH_MISMATCH")
    if config["dev_selection"] != selected.get("rules"):
        raise ValueError("FROZEN_RULES_CHANGED")
    return results, selected, config, {
        "results_sha256": sha(results_raw), "selection_sha256": sha(selection_raw),
        "config_sha256": expected_cfg_sha,
        "frozen_source_sha": state["frozen_sha"],
    }


def calculate(results: dict, selected: dict, config: dict, provenance: dict) -> dict:
    rules = config["dev_selection"]
    risk_limit = rules["max_invisible_false_accept_rate"]
    if not isinstance(risk_limit, (float, int)) or isinstance(risk_limit, bool) or not 0 < risk_limit < 1:
        raise ValueError("INVALID_FROZEN_SAFETY_THRESHOLD")
    methods = rules["candidate_methods"]
    if (not isinstance(methods, list) or not methods
            or len(set(methods)) != len(methods)):
        raise ValueError("INVALID_FROZEN_METHODS")
    selected_method = selected["selected_method"]
    if selected_method not in set(methods) | {rules["fallback_method"]}:
        raise ValueError("BAD_SELECTED_METHOD")
    if selected_method != rules["fallback_method"] and selected_method not in selected["eligible_candidates"]:
        raise ValueError("SELECTED_METHOD_NOT_ELIGIBLE")
    scenes = results.get("per_scene")
    names = [f"kinetics_0{i}" for i in range(6)]
    if (results.get("split") != "dev" or results.get("scene_count") != 6
            or not isinstance(scenes, dict) or set(scenes) != set(names)):
        raise ValueError("SIX_DEV_SCENES_ONLY")
    if not set(methods).issubset(set(results.get("methods", []))):
        raise ValueError("MISSING_FROZEN_METHOD")

    summaries = {}
    for method in methods:
        supported, breaches, zero_support, scene_details = 0, 0, 0, []
        for name in names:
            item = scenes[name][method]
            n = item.get("invisible")
            k = item.get("invisible_false_accept")
            rate = item.get("invisible_false_accept_rate")
            if (isinstance(n, bool) or not isinstance(n, int) or n < 0
                    or isinstance(k, bool) or not isinstance(k, int)
                    or not 0 <= k <= n):
                raise ValueError("INVALID_OCCLUSION_SUPPORT")
            if n == 0:
                if rate is not None:
                    raise ValueError("ZERO_SUPPORT_RATE_MUST_BE_NULL")
                zero_support += 1
                scene_details.append({"scene": name, "occluded_frame_samples": 0,
                                      "false_accepts": 0, "rate": None,
                                      "status": "NO_OCCLUSION_EVIDENCE"})
                continue
            if (not isinstance(rate, (float, int)) or isinstance(rate, bool)
                    or not math.isfinite(rate)
                    or not math.isclose(rate, k / n, rel_tol=0.0, abs_tol=1e-10)):
                raise ValueError("INVALID_OCCLUSION_RATE")
            supported += 1
            breach = float(rate) > risk_limit
            breaches += int(breach)
            scene_details.append({
                "scene": name, "occluded_frame_samples": n,
                "false_accepts": k, "rate": rate,
                "status": "EXCEEDS_FROZEN_10_PERCENT_RISK" if breach else "BELOW_OR_EQUAL_RISK",
            })
        upper = exact_one_sided_binomial_upper(breaches, supported, ALPHA)
        simultaneous_alpha = ALPHA / len(methods)
        family_upper = exact_one_sided_binomial_upper(breaches, supported, simultaneous_alpha)
        summaries[method] = {
            "supported_independent_scene_count": supported,
            "scenes_without_occlusion_support": zero_support,
            "observed_scene_risk_exceedances": breaches,
            "nominal_95pct_scene_violation_probability_upper": upper,
            "bonferroni_95pct_familywise_scene_violation_probability_upper": family_upper,
            "nominal_upper_leq_frozen_risk_limit": upper is not None and upper <= risk_limit,
            "familywise_upper_leq_frozen_risk_limit": family_upper is not None and family_upper <= risk_limit,
            "per_scene": scene_details,
        }

    winner = summaries[selected_method]
    return {
        "kind": "DEV_ONLY_SCENE_CLUSTER_OCCLUSION_SUPPORT_AUDIT",
        "split": "dev",
        "source_commit": provenance["frozen_source_sha"],
        "pinned_inputs": provenance,
        "chosen_on_DEV_prior_to_this_audit": selected_method,
        "original_DEV_eligible_candidates": selected["eligible_candidates"],
        "frozen_occlusion_false_accept_limit": risk_limit,
        "method_family_count": len(methods),
        "one_sided_alpha": ALPHA,
        "zero_failures_min_supported_scenes_at_nominal_95pct":
            zero_failure_scenes_needed(ALPHA, risk_limit),
        "zero_failures_min_supported_scenes_familywise_95pct":
            zero_failure_scenes_needed(ALPHA / len(methods), risk_limit),
        "selected_method_cluster_support": winner,
        "all_frozen_candidate_methods": summaries,
        "selected_diagnostic_verdict": (
            "OBSERVED_SCENE_RISK_BREACH" if winner["observed_scene_risk_exceedances"]
            else "INSUFFICIENT_TO_CERTIFY_CROSS_SCENE_SAFETY"
            if not winner["familywise_upper_leq_frozen_risk_limit"]
            else "ONLY_ASSUMPTION_DEPENDENT_STATISTICAL_EVIDENCE"
        ),
        "interpretation": (
            "The event is a natural scene's false-accept RATE exceeding the frozen "
            "frame-rate budget, not an individual-frame false accept. Frames within "
            "a video are correlated and MUST NOT be counted as independent scenes. "
            "The binomial upper bounds require future scene exchangeability/i.i.d., "
            "which natural action classes do not guarantee, and are not deployment "
            "certificates. The familywise bound protects post-hoc selection among "
            "the PREDECLARED method family under that assumption. Prior TEST labels, "
            "TEST predictions and TEST metrics are never read by this audit."
        ),
    }


def run(batch_root: Path, output: Path) -> str:
    results, selected, config, provenance = read_pinned_dev(batch_root)
    audit = calculate(results, selected, config, provenance)
    raw = canonical_json(audit)
    if output.exists() or Path(str(output) + ".sha256").exists():
        raise FileExistsError("IMMUTABLE_AUDIT_EXISTS")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as file:
        file.write(raw)
    with Path(str(output) + ".sha256").open("x", encoding="ascii") as file:
        file.write(sha(raw) + "\n")
    return sha(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-batch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pin = run(args.frozen_batch_root, args.output)
    print("DEV_CLUSTER_OCCLUSION_AUDIT_PINNED", pin, args.output)


if __name__ == "__main__":
    main()
