"""Batch65: label-blind structural identity and spatial ambiguity veto on DEV.

Research-only on previously studied Batch56 DEV. The image feature phase must
not open any scored result/selection/private oracle. Both phases must not read
TEST labels/predictions. A coordinate match is *not* object-visibility proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from experiments import batch62_dev_consensus_veto as b62
from experiments import batch63_dev_temporal_consensus as b63
from experiments import batch64_dev_rgb_identity as b64

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs/research"
FEATURES = OUTPUT_DIR / "batch65_unlabeled_dev_structural_identity.json"
REPORT = OUTPUT_DIR / "batch65_dev_structural_identity_report.json"
PATCH_RADIUS = 5
SEARCH_RADIUS = 13
EXCLUSION_RADIUS = 7
MIN_PATCH_STD = 5.0
# Fixed before inspecting feature-scored DEV labels. Diagnostic, no deployment.
VARIANTS = (
    "baseline", "template_4px", "ncc_050", "template_ncc_050",
    "template_ncc_margin_050_005", "template_ncc_margin_070_010",
)
POLICIES = {
    "ncc_050": (False, 0.50, None),
    "template_ncc_050": (True, 0.50, None),
    "template_ncc_margin_050_005": (True, 0.50, 0.05),
    "template_ncc_margin_070_010": (True, 0.70, 0.10),
}


def grayscale_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.shape != (256, 256, 3) or frame.dtype != np.uint8:
        raise RuntimeError("BATCH65_INVALID_RGB_SHAPE")
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def point_int(point: object) -> tuple[int, int] | None:
    p = b62.xy(point)
    if p is None:
        return None
    if not (0 <= p[0] < 256 and 0 <= p[1] < 256):
        return None
    return round(p[0]), round(p[1])


def query_template(gray: np.ndarray, query_xy: object) -> np.ndarray | None:
    p = point_int(query_xy)
    if p is None:
        return None
    x, y = p
    r = PATCH_RADIUS
    if x-r < 0 or y-r < 0 or x+r >= 256 or y+r >= 256:
        return None
    patch = gray[y-r:y+r+1, x-r:x+r+1].copy()
    if patch.shape != (2*r+1, 2*r+1):
        raise RuntimeError("BATCH65_PATCH_DIMENSION")
    if float(patch.std()) < MIN_PATCH_STD:
        return None  # Low-information query patch -> abstain, never assume 1.0 NCC.
    return patch


def structural_evidence(gray: np.ndarray, ref: np.ndarray | None,
                        predicted: object) -> dict[str, float] | None:
    """Current frame only, compare reference to candidate and separated rivals.

    A separated rival must be >= EXCLUSION_RADIUS from the predicted center.
    NCC score is ZNCC via OpenCV, in [-1, 1]. No future or label inputs.
    """
    p = point_int(predicted)
    if p is None or ref is None:
        return None
    x, y = p
    r, margin = PATCH_RADIUS, SEARCH_RADIUS
    if (x-r-margin < 0 or y-r-margin < 0 or
            x+r+margin >= 256 or y+r+margin >= 256):
        return None
    win = gray[y-r-margin:y+r+margin+1, x-r-margin:x+r+margin+1]
    corr = cv2.matchTemplate(win, ref, cv2.TM_CCOEFF_NORMED)
    if corr.shape != (2*margin+1, 2*margin+1):
        raise RuntimeError("BATCH65_SEARCH_DIMENSION")
    if not np.isfinite(corr).all():
        raise RuntimeError("BATCH65_NONFINITE_CORRELATION")
    center = float(corr[margin, margin])
    ys, xs = np.ogrid[-margin:margin+1, -margin:margin+1]
    outside = xs*xs + ys*ys >= EXCLUSION_RADIUS*EXCLUSION_RADIUS
    strongest_rival = float(np.max(corr[outside]))
    # A negative margin indicates a better match at another local coordinate.
    evidence = {"candidate_ncc": center, "rival_ncc": strongest_rival,
                "distinctiveness_margin": center - strongest_rival}
    if any(not math.isfinite(v) or abs(v) > 2.1 for v in evidence.values()):
        raise RuntimeError("BATCH65_OUT_OF_RANGE_EVIDENCE")
    return evidence


def per_scene_features(frames: np.ndarray, queries: list[dict],
                       pred_scene: dict) -> dict:
    n = pred_scene["nframes"]
    if frames.shape != (n, 256, 256, 3) or frames.dtype != np.uint8:
        raise RuntimeError("BATCH65_FRAMES_INVALID")
    qmap = {q["id"]: q for q in queries}
    if len(qmap) != 4 or set(qmap) != set(pred_scene["tracks"]):
        raise RuntimeError("BATCH65_PUBLIC_QUERY_DRIFT")
    gray = [grayscale_uint8(f) for f in frames]
    by_query = {}
    for qid, q in qmap.items():
        t0 = q["query_time"]
        tr = pred_scene["tracks"][qid][b62.BASE]
        if type(t0) is not int or not 0 <= t0 < n-1 or len(tr) != n or any(
                p is not None for p in tr[:t0]):
            raise RuntimeError("BATCH65_TRACK_TIME_DRIFT")
        template = query_template(gray[t0], q["query_xy"])
        row = [None] * n
        for t in range(t0+1, n):
            row[t] = structural_evidence(gray[t], template, tr[t])
        by_query[qid] = row
    return by_query


def extract(pred: dict, queries: dict, manifest: dict) -> dict:
    # b64 validates only *public* query, manifest and unscored DEV predictions.
    b64.validate_public(pred, queries, manifest)
    records = {r["name"]: r for r in pred["records"]}
    expected = {r["name"]: r for r in manifest["scenes"] if r["split"] == "dev"}
    evidence, sha_by_scene, timings = {}, {}, {}
    for scene in b62.SCENES:
        path = b62.DATA / "frames" / (scene + ".npy")
        digest = b64.sha_file(path)
        if digest != expected[scene]["frame_sha256"]:
            raise RuntimeError("BATCH65_FRAME_DRIFT:" + scene)
        sha_by_scene[scene] = digest
        start = time.perf_counter()
        evidence[scene] = per_scene_features(
            np.load(path, mmap_mode="r"), queries[scene], records[scene])
        timings[scene] = (time.perf_counter() - start) * 1000
    return {
        "kind": "BATCH65_LABEL_BLIND_STRUCTURAL_IDENTITY",
        "split": "dev", "exploratory": True,
        "source_commit": pred["source_commit"],
        "public_query_sha256": b64.sha_file(b62.DATA / "queries-public.json"),
        "manifest_sha256": b64.sha_file(b62.DATA / "manifest.json"),
        "prediction_sha256": b64.sha_file(b62.IN / "dev_predictions.json"),
        "fixed_parameters": {
            "patch_radius": PATCH_RADIUS, "search_radius": SEARCH_RADIUS,
            "exclusion_radius": EXCLUSION_RADIUS, "min_patch_std": MIN_PATCH_STD,
            "variant_policies": POLICIES,
        },
        "frame_sha256_by_scene": sha_by_scene,
        "feature_extraction_ms_per_scene": timings,
        "evidence_by_scene": evidence,
        "dev_oracle_read_during_extraction": False,
        "test_oracle_read": False, "test_predictions_read": False,
    }


def accepted(track: dict, value: dict | None, t: int, variant: str) -> bool:
    if variant in ("baseline", "template_4px"):
        return b62.pass_veto(track, t, variant)
    if variant not in POLICIES:
        raise RuntimeError("BATCH65_UNKNOWN_VARIANT")
    require_template, min_ncc, min_margin = POLICIES[variant]
    if not b62.pass_veto(track, t, "template_4px" if require_template else "baseline"):
        return False
    return bool(value is not None and
                value["candidate_ncc"] >= min_ncc and
                (min_margin is None or value["distinctiveness_margin"] >= min_margin))


def score_scene(pred_scene: dict, queries: list[dict], oracle_scene: dict,
                features: dict) -> dict:
    n = pred_scene["nframes"]
    qmap = {q["id"]: q for q in queries}
    if len(qmap) != 4 or set(qmap) != set(oracle_scene["tracks"]) or set(qmap) != set(features):
        raise RuntimeError("BATCH65_SCORING_QUERY_MISMATCH")
    counters = {k: b62.counts_empty() for k in VARIANTS}
    episodes = {k: [] for k in VARIANTS}
    for qid, q in qmap.items():
        t0, tr = q["query_time"], pred_scene["tracks"][qid]
        gt = oracle_scene["tracks"][qid]
        feature_row = features[qid]
        if len(feature_row) != n or len(gt["xy"]) != n or len(gt["occluded"]) != n or any(
                v is not None for v in feature_row[:t0+1]):
            raise RuntimeError("BATCH65_SCORE_SHAPE_DRIFT")
        for variant in VARIANTS:
            flags = [False] * n
            c = counters[variant]
            for t in range(t0+1, n):
                occ = gt["occluded"][t]
                if type(occ) is not bool:
                    raise RuntimeError("BATCH65_OCCLUSION_TYPE")
                c["invisible" if occ else "visible"] += 1
                flags[t] = accepted(tr, feature_row[t], t, variant)
                if flags[t]:
                    c["predicted_visible"] += 1
                    if occ:
                        c["invisible_false_accept"] += 1
                    else:
                        c["accepted_visible"] += 1
                        err = math.dist(b62.xy(tr[b62.BASE][t]), b62.xy(gt["xy"][t]))
                        c["correct8"] += int(err <= 8)
                        for radius in b62.RADII:
                            c[f"tp{radius}"] += int(err <= radius)
            episodes[variant].extend(b63.false_accept_runs(flags, gt["occluded"], t0))
    return {
        k: {**b62.rates(counters[k]),
            "longest_false_accept_episode": max(episodes[k], default=0)}
        for k in VARIANTS
    }


def evaluate(pred: dict, sel: dict, dev: dict, queries: dict,
             features: dict, oracle: dict) -> dict:
    b62.validate_sources(pred, sel, dev, queries)
    if (features.get("kind") != "BATCH65_LABEL_BLIND_STRUCTURAL_IDENTITY"
            or features.get("split") != "dev"
            or features.get("source_commit") != pred["source_commit"]
            or features.get("prediction_sha256") != b64.sha_file(b62.IN / "dev_predictions.json")
            or features.get("manifest_sha256") != b64.sha_file(b62.DATA / "manifest.json")
            or features.get("public_query_sha256") != b64.sha_file(b62.DATA / "queries-public.json")
            or features.get("fixed_parameters", {}).get("variant_policies") !=
            json.loads(json.dumps(POLICIES))
            or set(features.get("evidence_by_scene", ())) != set(b62.SCENES)
            or set(oracle) != set(b62.SCENES)
            or features.get("dev_oracle_read_during_extraction") is not False
            or features.get("test_oracle_read") is not False):
        raise RuntimeError("BATCH65_SPLIT_OR_FEATURE_PIN_DRIFT")
    records = {r["name"]: r for r in pred["records"]}
    scenes = {}
    for s in b62.SCENES:
        rows = score_scene(records[s], queries[s], oracle[s],
                           features["evidence_by_scene"][s])
        # Exact baseline reproduction; no silent metric or axis drift.
        old = dev["per_scene"][s][b62.BASE]
        for k in b62.counts_empty():
            if old[k] != rows["baseline"][k]:
                raise RuntimeError("BATCH65_BASELINE_REPLAY_DRIFT:" + s + ":" + k)
        # Veto-only: never create detections or change base track coordinates.
        for variant in VARIANTS:
            for k in ("predicted_visible", "accepted_visible", "invisible_false_accept", "correct8"):
                if rows[variant][k] > rows["baseline"][k]:
                    raise RuntimeError("BATCH65_NONVETO_FAILURE")
        for variant in POLICIES:
            if POLICIES[variant][0]:
                if rows[variant]["predicted_visible"] > rows["template_4px"]["predicted_visible"]:
                    raise RuntimeError("BATCH65_NOT_TEMPLATE_SUBSET")
        scenes[s] = rows
    summary = {}
    for variant in VARIANTS:
        rows = [scenes[s][variant] for s in b62.SCENES]
        supported = [r for r in rows if r["invisible"]]
        summary[variant] = {
            "scene_macro_visible_coverage": b62.average(rows, "visible_coverage"),
            "scene_macro_occluded_false_accept_rate": b62.average(rows, "invisible_false_accept_rate"),
            "scene_macro_jaccard": b62.average(rows, "jaccard_mean_1_2_4_8_16"),
            "violating_scenes": sum(r["invisible_false_accept_rate"] > b62.BUDGET for r in supported),
            "occlusion_supported_scenes": len(supported),
            "worst_scene_false_accept": max((r["invisible_false_accept_rate"] for r in supported), default=None),
            "longest_false_accept_episode": max(r["longest_false_accept_episode"] for r in rows),
            "paired_false_accept_vetoed_vs_template": sum(
                scenes[s]["template_4px"]["invisible_false_accept"] -
                scenes[s][variant]["invisible_false_accept"] for s in b62.SCENES),
            "paired_correct_lost_vs_template": sum(
                scenes[s]["template_4px"]["correct8"] -
                scenes[s][variant]["correct8"] for s in b62.SCENES),
        }
    timings = list(features["feature_extraction_ms_per_scene"].values())
    return {
        "kind": "BATCH65_EXPLORATORY_STRUCTURAL_IDENTITY_FALSIFICATION",
        "split": "dev", "exploratory": True, "deployable": False,
        "test_oracle_read": False, "test_predictions_read": False,
        "features_pinned_before_dev_oracle_read": True,
        "method_selection_changed": False, "frozen_batch59_unchanged": True,
        "scene_count": len(scenes), "query_count": 144,
        "feature_extraction_ms_sum": sum(timings),
        "feature_extraction_ms_median_per_scene": statistics.median(timings),
        "timing_scope": "offline feature-only; excludes tracker, video decode, hash, orchestration",
        "interpretation_limit": "2D NCC matches do not prove identity, visibility, or 3D camera motion",
        "summary": summary,
        "kinetics_17": scenes["kinetics_17"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("features", "score"))
    args = parser.parse_args()
    pred = b62.pinned_json(b62.IN / "dev_predictions.json")
    queries = json.loads((b62.DATA / "queries-public.json").read_bytes())
    if args.phase == "features":
        manifest = json.loads((b62.DATA / "manifest.json").read_bytes())
        obj = extract(pred, queries, manifest)
        digest = b64.pin_write(FEATURES, obj)
        print(json.dumps({"phase": "features", "sha256": digest,
                          "scenes": len(obj["evidence_by_scene"])}))
    else:
        # Refuse scoring until immutable and verified label-blind features exist.
        feature = b62.pinned_json(FEATURES)
        sel = b62.pinned_json(b62.IN / "dev_selection.json")
        raw = (b62.IN / "dev_results.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != sel["dev_results_sha256"]:
            raise RuntimeError("BATCH65_DEV_SCORE_PIN_DRIFT")
        dev = json.loads(raw)
        oracle = b62.load_dev_only_oracle(b62.DATA / "oracle-private.json", b62.SCENES)
        report = evaluate(pred, sel, dev, queries, feature, oracle)
        digest = b64.pin_write(REPORT, report)
        print(json.dumps({"phase": "score", "sha256": digest,
                          "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
