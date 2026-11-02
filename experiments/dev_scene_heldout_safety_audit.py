"""DEV-only leave-one-scene-out safety/coverage audit of a *finished frozen* batch.

A selection stability diagnostic, not a new tracker, TEST outcome, or model
promotion. It does not import the frozen evaluation module or read any TEST
oracle, frames, videos, or predictions. Results are pinned write-once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

RATES = ("visible_coverage", "invisible_false_accept_rate",
         "jaccard_mean_1_2_4_8_16")
PIXEL_KEYS = ("mean_luma", "luma_std", "laplacian_variance_mean",
              "mean_adjacent_frame_absdiff", "orb_keypoints_median",
              "affine_pair_acceptance", "affine_translation_p50_px")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("EXPECTED_JSON_OBJECT:" + path.name)
    return obj, sha256(raw)


def checked_rate(value: object, key: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("INVALID_RATE_TYPE:" + key)
    number = float(value)
    if not math.isfinite(number) or not (0.0 <= number <= 1.0):
        raise ValueError("INVALID_RATE_VALUE:" + key)
    return number


def macro(scenes: dict, names: list[str], method: str, key: str) -> float | None:
    vals = [checked_rate(scenes[name][method][key], key) for name in names]
    valid = [value for value in vals if value is not None]
    return sum(valid) / len(valid) if valid else None


def choose(scenes: dict, training_names: list[str], rules: dict) -> tuple[str, list[str]]:
    """Match the frozen selection's scene-macro scoring and candidate order."""
    candidates = rules["candidate_methods"]
    if not isinstance(candidates, list) or len(set(candidates)) != len(candidates):
        raise ValueError("INVALID_CANDIDATES")
    eligible = []
    for method in candidates:
        c = macro(scenes, training_names, method, "visible_coverage")
        f = macro(scenes, training_names, method, "invisible_false_accept_rate")
        j = macro(scenes, training_names, method, "jaccard_mean_1_2_4_8_16")
        if c is not None and f is not None and j is not None and (
            c >= rules["min_visible_coverage"] and
            f <= rules["max_invisible_false_accept_rate"]
        ):
            eligible.append(method)
    if not eligible:
        return rules["fallback_method"], []
    return max(eligible, key=lambda method: (
        macro(scenes, training_names, method, "jaccard_mean_1_2_4_8_16"),
        macro(scenes, training_names, method, "visible_coverage"),
    )), eligible


def _validate_inputs(results: dict, predictions: dict, cfg: dict,
                     result_sha: str, pred_sha: str,
                     rules: dict) -> list[str]:
    if results.get("split") != "dev" or predictions.get("split") != "dev":
        raise ValueError("DEV_ONLY_SPLIT_REQUIRED")
    if (results.get("source_commit") != predictions.get("source_commit") or
            results.get("manifest_sha256") != predictions.get("manifest_sha256") or
            results.get("config_sha256") != predictions.get("config_sha256") or
            results.get("prediction_sha256") != pred_sha):
        raise ValueError("MISMATCHED_PINNED_PROVENANCE")
    # Frozen evaluators hash config as sorted, indented JSON with trailing newline.
    cfg_sha = sha256((json.dumps(cfg, sort_keys=True, ensure_ascii=False,
                                  indent=2) + "\n").encode())
    if cfg_sha != results.get("config_sha256"):
        raise ValueError("CONFIG_HASH_MISMATCH")
    if results.get("scene_count") != 6 or len(predictions.get("records", [])) != 6:
        raise ValueError("EXPECTED_SIX_DEV_SCENES")
    scenes = results.get("per_scene")
    if not isinstance(scenes, dict) or len(scenes) != 6:
        raise ValueError("INVALID_SCENES")
    names = sorted(scenes)
    records = predictions["records"]
    if (len({record.get("name") for record in records}) != 6 or
            set(names) != {record.get("name") for record in records}):
        raise ValueError("PREDICTION_SCENE_MISMATCH")
    if not all(name.startswith("kinetics_0") and int(name.rsplit("_", 1)[1]) < 6
               for name in names):
        raise ValueError("TEST_SCENE_IN_DEV_INPUT")
    methods = set(results.get("methods", []))
    if not set(rules["candidate_methods"]).issubset(methods):
        raise ValueError("CANDIDATE_METHOD_MISSING")
    if rules["fallback_method"] not in methods:
        raise ValueError("FALLBACK_METHOD_MISSING")
    for name in names:
        if set(rules["candidate_methods"]) - set(scenes[name]):
            raise ValueError("SCENE_METHOD_MISSING:" + name)
        for method in rules["candidate_methods"]:
            for key in RATES:
                checked_rate(scenes[name][method][key], key)
            for key in ("visible", "invisible", "invisible_false_accept"):
                v = scenes[name][method][key]
                if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                    raise ValueError("INVALID_SUPPORT:" + key)
            if scenes[name][method]["invisible_false_accept"] > scenes[name][method]["invisible"]:
                raise ValueError("FALSE_ACCEPT_EXCEEDS_SUPPORT")
    return names


def audit(results: dict, predictions: dict, cfg: dict,
          result_sha: str, pred_sha: str) -> dict:
    rules = cfg["dev_selection"]
    names = _validate_inputs(results, predictions, cfg, result_sha, pred_sha, rules)
    scenes = results["per_scene"]
    full_selected, full_eligible = choose(scenes, names, rules)
    pix = {name: predictions["scene_aux"][name]["pixel_diagnostics"] for name in names}
    folds = []
    for held in names:
        train = [name for name in names if name != held]
        selected, eligible = choose(scenes, train, rules)
        observation = scenes[held][selected]
        coverage = checked_rate(observation["visible_coverage"], "visible_coverage")
        false_accept = checked_rate(observation["invisible_false_accept_rate"],
                                    "invisible_false_accept_rate")
        supported = int(observation["invisible"]) > 0
        folds.append({
            "heldout": held, "training_scene_count": len(train),
            "selection_from_train_only": selected,
            "eligible_train_only": eligible,
            "heldout_visible_support": observation["visible"],
            "heldout_invisible_support": observation["invisible"],
            "heldout_visible_coverage": coverage,
            "heldout_invisible_false_accept_rate": false_accept,
            "heldout_invisible_false_accept_count": observation["invisible_false_accept"],
            "heldout_scene_jaccard": checked_rate(
                observation["jaccard_mean_1_2_4_8_16"], "jaccard"),
            "heldout_safety_verdict": (
                "NO_INVISIBLE_SUPPORT" if not supported else
                "FAIL" if false_accept is None or
                false_accept > rules["max_invisible_false_accept_rate"] or
                coverage is None or coverage < rules["min_visible_coverage"] else "PASS"
            ),
            "pixel_diagnostics": {key: pix[held].get(key) for key in PIXEL_KEYS},
            "pixel_note": "2D brightness/texture/image motion proxies; not physical camera pose",
        })
    return {
        "kind": "DEV_ONLY_LEAVE_ONE_NATURAL_SCENE_OUT_SAFETY_AUDIT",
        "split": "dev",
        "source_commit": results["source_commit"],
        "dev_results_sha256": result_sha,
        "dev_predictions_sha256": pred_sha,
        "config_sha256": results["config_sha256"],
        "manifest_sha256": results["manifest_sha256"],
        "rule_source": "frozen batch48 dev_selection constraints; no retuning",
        "rules": rules,
        "full_dev_selected": full_selected,
        "full_dev_eligible": full_eligible,
        "folds": folds,
        "fold_selected_counts": {method: sum(f["selection_from_train_only"] == method
                                             for f in folds)
                                 for method in sorted({f["selection_from_train_only"] for f in folds})},
        "heldout_failed_safety": sum(f["heldout_safety_verdict"] == "FAIL" for f in folds),
        "heldout_unevaluable_safety": sum(f["heldout_safety_verdict"] == "NO_INVISIBLE_SUPPORT"
                                          for f in folds),
        "limitations": (
            "Six DEV natural videos only; folds share training scenes and are not "
            "a new independent TEST; no physically identified static background. "
            "A scene without invisible ground-truth support is unevaluable for "
            "occlusion safety. CPU timings in source are tracker-only, exclude decoding."
        ),
    }


def write_once(path: Path, payload: dict) -> str:
    raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    pin = sha256(raw)
    if path.exists() or Path(str(path) + ".sha256").exists():
        raise FileExistsError("IMMUTABLE_OUTPUT_EXISTS")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
    with Path(str(path) + ".sha256").open("x", encoding="ascii") as handle:
        handle.write(pin + "\n")
    return pin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", required=True, type=Path,
                        help="Frozen batch48b workspace. Only DEV files are opened.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    batch = args.batch_root.resolve()
    results_path = batch / "outputs/kinetics_batch48/dev_results.json"
    pred_path = batch / "outputs/kinetics_batch48/dev_predictions.json"
    config_path = batch / "configs/kinetics_batch48.json"
    if (results_path.name != "dev_results.json" or pred_path.name != "dev_predictions.json"):
        raise ValueError("DEV_FILENAMES_REQUIRED")
    results, rhash = load_json(results_path)
    predictions, phash = load_json(pred_path)
    cfg, _ = load_json(config_path)
    pred_pin = Path(str(pred_path) + ".sha256").read_text().strip()
    if pred_pin != phash:
        raise ValueError("DEV_PREDICTION_PIN_MISMATCH")
    result = audit(results, predictions, cfg, rhash, phash)
    checksum = write_once(args.output, result)
    print("DEV52_HELDOUT_AUDIT_WRITTEN", args.output, checksum,
          "failed", result["heldout_failed_safety"],
          "unevaluable", result["heldout_unevaluable_safety"])


if __name__ == "__main__":
    main()
