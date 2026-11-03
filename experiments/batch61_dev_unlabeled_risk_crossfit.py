"""Batch61: DEV-only out-of-fold test of observable scene-risk proxies.

No TEST or private oracle files, no model/threshold update, no GPU, no deployment.
The selected tracker itself was chosen on all DEV: this is EXPLORATORY,
not a fresh prospective validation. An explicitly privileged measurement is
included ONLY as a negative-control illustration of potential target leakage.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

import batch58_dev_grouped_transfer as base

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT.parent / "repo-batch56b-tested"
IN_DIR = FROZEN / "outputs" / "kinetics_batch56"
OUTPUT = ROOT / "outputs" / "research" / "batch61_dev_unlabeled_risk_crossfit.json"

# All IMAGE_FEATURES are computed from frames without target trajectories.
# The privileged residual, by contrast, uses annotated DEV target trajectories.
IMAGE_FEATURES = (
    "mean_luma",
    "laplacian_variance_mean",
    "mean_adjacent_frame_absdiff",
    "affine_translation_p50_px",
)
PRIVILEGED_FEATURES = ("background_relative_residual_mean_px",)
BLOCK_SIZE = 6
RIDGE = 5.0
MAX_ITER = 60
ONE_ALERT_PER_HELDOUT_BLOCK = 1
SAFETY_BUDGET = 0.10


def canonical(obj: object) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def extract_features(dev: dict, scene_ids: list[str], *, privileged: bool = False) -> np.ndarray:
    section = "background_relative_diagnostics" if privileged else "scene_pixel_diagnostics"
    names = PRIVILEGED_FEATURES if privileged else IMAGE_FEATURES
    diagnostic = dev.get(section)
    if not isinstance(diagnostic, dict) or set(diagnostic) != set(scene_ids):
        raise RuntimeError("BATCH61_MISSING_DIAGNOSTIC:" + section)
    data = []
    for scene in scene_ids:
        row = diagnostic[scene]
        if not isinstance(row, dict):
            raise RuntimeError("BATCH61_INVALID_SCENE_DIAGNOSTIC:" + scene)
        values = []
        for name in names:
            val = row.get(name)
            if isinstance(val, bool) or not isinstance(val, (float, int)) or not math.isfinite(val):
                raise RuntimeError("BATCH61_NONFINITE_OR_MISSING_FEATURE:" + scene + ":" + name)
            values.append(float(val))
        data.append(values)
    matrix = np.asarray(data, dtype=float)
    if matrix.shape != (len(scene_ids), len(names)):
        raise RuntimeError("BATCH61_FEATURE_SHAPE")
    return matrix


def risk_labels(dev: dict, ids: list[str], selected: str, budget: float) -> list[int | None]:
    labels = []
    for scene in ids:
        row = dev["per_scene"][scene][selected]
        base._check_rate(row, scene, selected)
        # A scene with no annotated occlusion is UNKNOWN, not a negative.
        labels.append(int(row["invisible_false_accept_rate"] > budget)
                      if row["invisible"] > 0 else None)
    return labels


def fit_predict(x_train: np.ndarray, labels: list[int], x_test: np.ndarray) -> tuple[np.ndarray, dict]:
    """L2-shrunk binomial logit. Preprocessing fits on TRAIN ONLY."""
    n, k = x_train.shape
    if n != len(labels) or n == 0 or x_test.shape[1] != k:
        raise RuntimeError("BATCH61_INVALID_FOLD_SHAPE")
    if any(type(label) is not int or label not in (0, 1) for label in labels):
        raise RuntimeError("BATCH61_INVALID_TRAIN_LABEL")
    x_train = np.asarray(x_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise RuntimeError("BATCH61_INVALID_NUMERIC")
    means = x_train.mean(axis=0)
    std = np.maximum(x_train.std(axis=0), 1e-8)
    z = np.column_stack((np.ones(n), (x_train - means) / std))
    z_test = np.column_stack((np.ones(len(x_test)), (x_test - means) / std))
    y = np.asarray(labels, dtype=float)
    # Laplace smoothed intercept avoids infinite logits with all-zero labels.
    p0 = (y.sum() + 1.0) / (n + 2.0)
    w = np.zeros(k + 1, dtype=float)
    w[0] = math.log(p0 / (1.0 - p0))
    penalty = np.diag([0.0] + [RIDGE] * k)
    for _ in range(MAX_ITER):
        eta = np.clip(z @ w, -30, 30)
        pred = 1 / (1 + np.exp(-eta))
        h = z.T @ ((pred * (1.0 - pred))[:, None] * z) + penalty
        grad = z.T @ (pred - y) + penalty @ w
        step = np.linalg.solve(h, grad)
        w = w - step
        if np.linalg.norm(step) < 1e-8:
            break
    score = 1 / (1 + np.exp(-np.clip(z_test @ w, -30, 30)))
    if not np.isfinite(score).all():
        raise RuntimeError("BATCH61_NONFINITE_SCORE")
    return score, {
        "train_occlusion_supported": n,
        "train_violations": int(y.sum()),
        "train_feature_means": means.tolist(),
        "train_feature_std": std.tolist(),
        "weights": w.tolist(),
        "train_only_preprocessing": True,
    }


def auc_from_scores(items: list[dict]) -> float | None:
    pos = [a["prob"] for a in items if a["label"] == 1]
    neg = [a["prob"] for a in items if a["label"] == 0]
    if not pos or not neg:
        return None
    return sum((p > n) + .5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def summarize(rows: list[dict], feature_names: tuple[str, ...], *, privileged: bool) -> dict:
    supported = [r for r in rows if r["label"] is not None]
    positives = [r for r in supported if r["label"] == 1]
    alerts = [r for r in rows if r["alert"]]
    found = [r for r in alerts if r["label"] == 1]
    return {
        "feature_names": list(feature_names),
        "privileged_oracle_dependent_feature": privileged,
        "deployable": False,
        "scene_count": len(rows),
        "supported_scene_count": len(supported),
        "unsupported_unknown_scene_count": len(rows) - len(supported),
        "violation_count": len(positives),
        "heldout_auc_descriptive": auc_from_scores(supported),
        "heldout_brier": (sum((r["prob"] - r["label"]) ** 2 for r in supported) /
                          len(supported) if supported else None),
        "heldout_train_prior_brier": (
            sum((r["prior"] - r["label"]) ** 2 for r in supported) /
            len(supported) if supported else None),
        "alerts": len(alerts),
        "alerts_with_unknown_occlusion_status": sum(r["label"] is None for r in alerts),
        "alerts_with_violation": len(found),
        "alerts_without_violation": sum(r["label"] == 0 for r in alerts),
        "violation_recall_with_fixed_alert_budget": len(found) / len(positives) if positives else None,
        "alert_scene_ids": [r["scene"] for r in alerts],
        "unsafe_scene_ids": [r["scene"] for r in positives],
        "rows": rows,
    }


def crossfit(dev: dict, ids: list[str], selected: str,
             features: np.ndarray, names: tuple[str, ...], *,
             privileged: bool = False, safety_budget: float = SAFETY_BUDGET) -> dict:
    if len(ids) != 36 or len(set(ids)) != 36 or features.shape != (36, len(names)):
        raise RuntimeError("BATCH61_INVALID_COHORT")
    if safety_budget != SAFETY_BUDGET:
        raise RuntimeError("BATCH61_SAFETY_BUDGET_DRIFT")
    labels = risk_labels(dev, ids, selected, safety_budget)
    rows: list[dict] = []
    folds = []
    for fold in range(6):
        withheld = list(range(fold * BLOCK_SIZE, (fold + 1) * BLOCK_SIZE))
        train = [i for i in range(36) if i not in withheld and labels[i] is not None]
        if not train:
            raise RuntimeError("BATCH61_NO_SUPPORTED_TRAIN_SCENES")
        y = [labels[i] for i in train]
        assert all(v is not None for v in y)
        pred, meta = fit_predict(features[train], y, features[withheld])
        prior = (sum(y) + 1) / (len(y) + 2)
        # Fixed alert budget: one highest predicted-risk scene per 6-scene block;
        # lexical ID breaks exact ties. Do not choose alerts using heldout labels.
        best = min(range(BLOCK_SIZE), key=lambda j: (-float(pred[j]), ids[withheld[j]]))
        block_rows = []
        for j, i in enumerate(withheld):
            r = {
                "scene": ids[i],
                "fold": fold,
                "label": labels[i],
                "prob": float(pred[j]),
                "prior": prior,
                "alert": j == best,
            }
            rows.append(r)
            block_rows.append(r)
        folds.append({
            "fold": fold,
            "heldout_scenes": [ids[i] for i in withheld],
            "supported_train_scenes": len(train),
            "train_violation_count": sum(y),
            "train_details": meta,
            "alert_scene": ids[withheld[best]],
            "heldout_supported_scenes": sum(r["label"] is not None for r in block_rows),
        })
    result = summarize(rows, names, privileged=privileged)
    result["folds"] = folds
    return result


def analyze(dev: dict, selected: dict, cfg: dict) -> dict:
    ids = base.validate(dev, selected, cfg)
    method = selected["selected_method"]
    image_features = extract_features(dev, ids)
    image_report = crossfit(dev, ids, method, image_features, IMAGE_FEATURES)
    # This is a NEGATIVE CONTROL: depends on GT target trajectories and cannot
    # qualify as a sensor-only risk gate or model-selection signal.
    privileged_features = extract_features(dev, ids, privileged=True)
    privileged_report = crossfit(
        dev, ids, method, privileged_features, PRIVILEGED_FEATURES, privileged=True
    )
    return {
        "kind": "BATCH61_EXPLORATORY_IMAGE_ONLY_PROXY_CROSSFIT_WITH_ORACLE_NEGATIVE_CONTROL",
        "split": "dev",
        "selected_method": method,
        "source_frozen_commit": dev["source_commit"],
        "dev_results_sha256": selected["dev_results_sha256"],
        "config_sha256": dev["config_sha256"],
        "fixed_scene_blocks": 6,
        "fixed_alerts_per_block": ONE_ALERT_PER_HELDOUT_BLOCK,
        "safety_budget": SAFETY_BUDGET,
        "image_only": image_report,
        "privileged_negative_control": privileged_report,
        "test_data_read": False,
        "model_selection_changed": False,
        "frozen_run_changed": False,
        "deployment_authorized": False,
        "limitations": [
            "Prior frozen DEV has been examined repeatedly; results are exploratory, not blind or independent.",
            "The tracker itself was selected using all 36 DEV scenes: cross-fitting the risk score does not remove selection bias.",
            "All folds have overlapping training sets; the 36 scenes are not proven independent deployments.",
            "Only four scene violations; AUC and alert recall are highly unstable.",
            "A scene without occlusion labels is UNKNOWN, never a confirmed safe scene.",
            "No per-frame abstention is implemented: a one-scene alert is diagnostic only, not a tracker gate.",
            "Privileged control uses GT target trajectories and is NEVER available to an online inference system.",
            "2-D affine motion is not proof of static background or calibrated 3-D camera motion.",
        ],
    }


def run() -> dict:
    dev_path = IN_DIR / "dev_results.json"
    raw = dev_path.read_bytes()
    selected = base.load_pinned(IN_DIR / "dev_selection.json")
    if digest(raw) != selected.get("dev_results_sha256"):
        raise RuntimeError("BATCH61_DEV_RESULTS_PIN_INVALID")
    cfg = json.loads((FROZEN / "configs" / "kinetics_batch56.json").read_bytes())
    if digest(canonical(cfg)) != selected.get("config_sha256"):
        raise RuntimeError("BATCH61_CONFIG_PIN_INVALID")
    frozen_sha = subprocess.check_output(
        ["git", "-C", str(FROZEN), "rev-parse", "HEAD"], text=True
    ).strip()
    if frozen_sha != selected.get("source_commit"):
        raise RuntimeError("BATCH61_FROZEN_SOURCE_DRIFT")
    report = analyze(json.loads(raw), selected, cfg)
    content = canonical(report)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pin = OUTPUT.with_suffix(".json.sha256")
    if OUTPUT.is_file() and OUTPUT.read_bytes() != content:
        raise RuntimeError("BATCH61_IMMUTABLE_OUTPUT_CONFLICT")
    if pin.is_file() and pin.read_text().strip() != digest(content):
        raise RuntimeError("BATCH61_OUTPUT_PIN_DRIFT")
    if not OUTPUT.is_file():
        OUTPUT.write_bytes(content)
    if not pin.is_file():
        pin.write_text(digest(content) + "\n")
    for label in ("image_only", "privileged_negative_control"):
        r = report[label]
        print("BATCH61", label, "AUC", r["heldout_auc_descriptive"],
              "brier", r["heldout_brier"], "train_prior_brier", r["heldout_train_prior_brier"],
              "captured", r["alerts_with_violation"], "/", r["violation_count"],
              "unknown_alerts", r["alerts_with_unknown_occlusion_status"], flush=True)
    print("BATCH61_OUTPUT_PIN", digest(content), flush=True)
    return report


if __name__ == "__main__":
    run()
