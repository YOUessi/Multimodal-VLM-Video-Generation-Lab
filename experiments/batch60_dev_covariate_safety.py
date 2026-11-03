"""Batch60: TEST-blind environment-proxy vs occlusion-risk audit on frozen Batch56 DEV.

Strictly descriptive: covariate terciles contain natural DEV scenes selected WITHOUT
their labels, but they are neither independent domains nor causal interventions.
Image-plane affine diagnostics do not identify static background or 3-D camera motion.
Does not alter any approved / frozen experimental runner or method selection.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

import batch58_dev_grouped_transfer as base

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT.parent / "repo-batch56b-tested"
IN_DIR = FROZEN / "outputs" / "kinetics_batch56"
OUTPUT = ROOT / "outputs" / "research" / "batch60_dev_covariate_safety.json"

# Predeclared proxy features, without access to method outcomes.
# NOTE: the last two are 2-D affine/background-relative PROXIES, not ground-truth camera pose.
FEATURES = {
    "illumination_mean_luma": ("scene_pixel_diagnostics", "mean_luma"),
    "texture_laplacian_variance": ("scene_pixel_diagnostics", "laplacian_variance_mean"),
    "frame_appearance_change": ("scene_pixel_diagnostics", "mean_adjacent_frame_absdiff"),
    "dominant_affine_translation_px": ("scene_pixel_diagnostics", "affine_translation_p50_px"),
    "background_relative_residual_px": ("background_relative_diagnostics", "background_relative_residual_mean_px"),
    "dominant_affine_explained_fraction": ("background_relative_diagnostics", "camera_explained_fraction_mean_unclipped"),
}
STRATA = ("lower", "middle", "upper")
MIN_CONTEXT_FRAMES = 20


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(obj: object) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _finite_number(x: object, label: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise RuntimeError("BATCH60_INVALID_COVARIATE:" + label)
    return float(x)


def feature_values(dev: dict, scene_ids: list[str]) -> dict[str, dict[str, float]]:
    if len(scene_ids) % 3 or len(scene_ids) < 6 or len(set(scene_ids)) != len(scene_ids):
        raise RuntimeError("BATCH60_INVALID_SCENES_FOR_TERCILES")
    extracted: dict[str, dict[str, float]] = {}
    for feature, (section, key) in FEATURES.items():
        obj = dev.get(section)
        if not isinstance(obj, dict) or set(obj) != set(scene_ids):
            raise RuntimeError("BATCH60_INCOMPLETE_COVARIATE:" + feature)
        values = {}
        for scene in scene_ids:
            row = obj[scene]
            if not isinstance(row, dict) or key not in row:
                raise RuntimeError("BATCH60_MISSING_COVARIATE:" + feature + ":" + scene)
            values[scene] = _finite_number(row[key], feature + ":" + scene)
        extracted[feature] = values
    return extracted


def summarize(dev: dict, scenes: list[str], method: str, threshold: float) -> dict:
    if not scenes:
        raise RuntimeError("BATCH60_EMPTY_STRATUM")
    rows = [dev["per_scene"][scene][method] for scene in scenes]
    for scene, r in zip(scenes, rows):
        base._check_rate(r, scene, method)
    supported = [(scene, row) for scene, row in zip(scenes, rows) if row["invisible"] > 0]
    totals = base.metrics(dev, scenes, method)
    total_invisible = sum(r["invisible"] for r in rows)
    false_accept = sum(r["invisible_false_accept"] for r in rows)
    return {
        "scene_count": len(scenes),
        "scenes_with_occlusion_evidence": len(supported),
        "scenes_without_occlusion_evidence": len(scenes) - len(supported),
        "low_occlusion_support_scenes_lt_20_frames": [
            scene for scene, r in supported if r["invisible"] < MIN_CONTEXT_FRAMES
        ],
        "occluded_frame_count": total_invisible,
        "occluded_false_accept_count": false_accept,
        "occluded_false_accept_micro": false_accept / total_invisible if total_invisible else None,
        "occluded_false_accept_scene_macro": totals["invisible_false_accept_rate"],
        "violating_scene_count": sum(
            r["invisible_false_accept_rate"] > threshold for _, r in supported
        ),
        "violating_scene_ids": [
            s for s, r in supported if r["invisible_false_accept_rate"] > threshold
        ],
        "worst_supported_scene_false_accept": totals["worst_supported_scene_false_accept_rate"],
        "visible_coverage_scene_macro": totals["visible_coverage"],
        "jaccard_scene_macro": totals["jaccard_mean_1_2_4_8_16"],
    }


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Tie-aware Spearman, descriptive only; no independent-p-value fiction."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        left = 0
        while left < len(order):
            right = left + 1
            while right < len(order) and values[order[right]] == values[order[left]]:
                right += 1
            mean_rank = (left + right - 1) / 2
            for i in range(left, right):
                out[order[i]] = mean_rank
            left = right
        return out

    a, b = ranks(xs), ranks(ys)
    ma, mb = statistics.mean(a), statistics.mean(b)
    cross = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    ssx = sum((x - ma) ** 2 for x in a)
    ssy = sum((y - mb) ** 2 for y in b)
    return cross / math.sqrt(ssx * ssy) if ssx and ssy else None


def analyze(dev: dict, selected: dict, rules: dict, scene_ids: list[str]) -> dict:
    values = feature_values(dev, scene_ids)
    methods = rules["candidate_methods"]
    chosen = selected["selected_method"]
    if chosen not in methods or len(set(methods)) != len(methods):
        raise RuntimeError("BATCH60_UNFROZEN_METHOD")
    threshold = float(rules["max_invisible_false_accept_rate"])
    if not 0 < threshold < 1:
        raise RuntimeError("BATCH60_INVALID_RISK_BUDGET")
    if len(scene_ids) != 36 or len(methods) != 9:
        raise RuntimeError("BATCH60_EXPECTED_FROZEN_36_SCENES_9_METHODS")

    all_methods = {
        method: summarize(dev, scene_ids, method, threshold) for method in methods
    }
    feature_reports = {}
    for feature, cov in values.items():
        # Terciles derived only from unsupervised image covariates. Equal values
        # receive deterministic scene-id tie-breaks, explicitly flagged below.
        ordered = sorted(scene_ids, key=lambda scene: (cov[scene], scene))
        n = len(scene_ids) // 3
        groups = [ordered[i * n:(i + 1) * n] for i in range(3)]
        partitions = []
        for label, scenes in zip(STRATA, groups):
            partitions.append({
                "stratum": label,
                "scene_ids": scenes,
                "feature_min": min(cov[s] for s in scenes),
                "feature_max": max(cov[s] for s in scenes),
                "methods": {
                    method: summarize(dev, scenes, method, threshold) for method in methods
                },
            })
        supported = [
            scene for scene in scene_ids
            if dev["per_scene"][scene][chosen]["invisible"] > 0
        ]
        corr = spearman(
            [cov[s] for s in supported],
            [dev["per_scene"][s][chosen]["invisible_false_accept_rate"] for s in supported],
        )
        feature_reports[feature] = {
            "terciles": partitions,
            "sample_size_with_occlusion_evidence": len(supported),
            "chosen_method_spearman_rho_on_supported_scenes": corr,
            "stratum_cutoff_ties_possible": (
                cov[ordered[n - 1]] == cov[ordered[n]]
                or cov[ordered[2 * n - 1]] == cov[ordered[2 * n]]
            ),
        }
    return {
        "kind": "BATCH60_FROZEN_DEV_COVARIATE_STRATIFIED_RISK_DIAGNOSTIC",
        "split": "dev",
        "source_frozen_commit": dev["source_commit"],
        "config_sha256": dev["config_sha256"],
        "selected_method": chosen,
        "methods": all_methods,
        "feature_reports": feature_reports,
        "safety_threshold": threshold,
        "n_dev_scenes": len(scene_ids),
        "model_selection_changed": False,
        "test_data_read": False,
        "deployment_authorized": False,
        "limitations": [
            "Previous Batch56 DEV results have already been inspected: EXPLORATORY, not independent blind evaluation.",
            "Unsupervised covariate terciles are based on the current 36 DEV scenes; associations are noncausal.",
            "Frame-level micro and scene-macro rates must not be conflated, particularly under unbalanced occlusion.",
            "No annotated occluded frames means unknown risk, never a safe zero.",
            "Image-plane affine translation/residual and explained fraction are imperfect proxies, NOT observed 3D camera/background motion.",
            "Spearman associations have no valid IID p-values with this selected, possibly correlated video collection.",
            "Differences across strata and nine methods involve multiplicity; no threshold tuning from these results.",
        ],
    }


def run() -> dict:
    # Only DEV results and their frozen selection/config pins are opened.
    raw = (IN_DIR / "dev_results.json").read_bytes()
    selected = base.load_pinned(IN_DIR / "dev_selection.json")
    if digest(raw) != selected.get("dev_results_sha256"):
        raise RuntimeError("BATCH60_DEV_RESULT_PIN_INVALID")
    dev = json.loads(raw)
    cfg = json.loads((FROZEN / "configs" / "kinetics_batch56.json").read_bytes())
    if digest(canonical(cfg)) != selected.get("config_sha256"):
        raise RuntimeError("BATCH60_CONFIG_PIN_INVALID")
    frozen_sha = subprocess.check_output(
        ["git", "-C", str(FROZEN), "rev-parse", "HEAD"], text=True
    ).strip()
    if frozen_sha != selected["source_commit"]:
        raise RuntimeError("BATCH60_FROZEN_SOURCE_DRIFT")
    scene_ids = base.validate(dev, selected, cfg)
    result = analyze(dev, selected, cfg["dev_selection"], scene_ids)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    content = canonical(result)
    pin = OUTPUT.with_suffix(".json.sha256")
    if OUTPUT.exists() and OUTPUT.read_bytes() != content:
        raise RuntimeError("BATCH60_IMMUTABLE_RESULT_CONFLICT")
    if pin.exists() and pin.read_text().strip() != digest(content):
        raise RuntimeError("BATCH60_RESULT_PIN_MISMATCH")
    if not OUTPUT.exists():
        OUTPUT.write_bytes(content)
    if not pin.exists():
        pin.write_text(digest(content) + "\n")
    print("BATCH60_COMPLETE", "selected", result["selected_method"],
          "features", len(result["feature_reports"]),
          "methods", len(result["methods"]),
          "sha256", digest(content), flush=True)
    for feat, row in result["feature_reports"].items():
        lines = [
            (s["stratum"],
             s["methods"][result["selected_method"]]["occluded_false_accept_scene_macro"],
             s["methods"][result["selected_method"]]["violating_scene_count"])
            for s in row["terciles"]
        ]
        print(feat, "strata", lines, "spearman", row["chosen_method_spearman_rho_on_supported_scenes"])
    return result


if __name__ == "__main__":
    run()
