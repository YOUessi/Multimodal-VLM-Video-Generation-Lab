"""Batch45: fresh action-class-disjoint Kinetics holdout and illumination-robust point gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

import kinetics_batch42_eval as core
import kinetics_batch45_data as src

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/tapvid_kinetics_batch45e/batch45e_frozen"
OUT = ROOT / "outputs/kinetics_batch45e"
_BASE_METHODS = tuple(core.METHODS)
_BASE_PREDICT_QUERY = core.predict_query
GRADIENT_METHODS = ("gradient_fb_050", "gradient_fb_065", "gradient_fb_080")
METHODS = _BASE_METHODS + GRADIENT_METHODS


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def doc(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()


def gitsha() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def load_config(path: str | Path | None = None) -> dict:
    return doc(Path(path) if path else ROOT / "configs/kinetics_batch45.json")


def gradient_descriptor(patch: np.ndarray) -> np.ndarray | None:
    """Unit-normalized Sobel vector; invariant to additive brightness and positive gain."""
    value = np.asarray(patch, dtype=np.uint8)
    if value.ndim != 2 or min(value.shape) < 5:
        return None
    gx = cv2.Sobel(value, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(value, cv2.CV_32F, 0, 1, ksize=3)
    vector = np.concatenate((gx.reshape(-1), gy.reshape(-1))).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-4:
        return None
    return vector / norm


def gradient_similarity(a: np.ndarray, b: np.ndarray) -> float | None:
    da = gradient_descriptor(a)
    db = gradient_descriptor(b)
    if da is None or db is None or da.shape != db.shape:
        return None
    return float(np.clip(np.dot(da, db), -1.0, 1.0))


def gradient_gate(gray: list[np.ndarray], query: dict, base_trace: list,
                  threshold: float, patch_size: int) -> tuple[list, list[float | None]]:
    t = int(query["query_time"])
    anchor = core.patch_at(gray[t], query["query_xy"], patch_size)
    anchor_desc = gradient_descriptor(anchor) if anchor is not None else None
    out = list(base_trace)
    scores: list[float | None] = [None] * len(gray)
    if anchor_desc is None:
        return [out[i] if i <= t else None for i in range(len(out))], scores
    for i in range(t + 1, len(gray)):
        if out[i] is None:
            continue
        patch = core.patch_at(gray[i], out[i], patch_size)
        candidate = gradient_descriptor(patch) if patch is not None else None
        similarity = (
            float(np.clip(np.dot(anchor_desc, candidate), -1.0, 1.0))
            if candidate is not None and candidate.shape == anchor_desc.shape else None
        )
        scores[i] = similarity
        if similarity is None or similarity < threshold:
            out[i] = None
    return out, scores


def predict_query(gray: list[np.ndarray], query: dict, matrices: list,
                  cfg: dict) -> tuple[dict, dict]:
    active_methods = core.METHODS
    core.METHODS = _BASE_METHODS
    try:
        predictions, timings = _BASE_PREDICT_QUERY(gray, query, matrices, cfg)
    finally:
        core.METHODS = active_methods
    diagnostics = {}
    suffixes = ("050", "065", "080")
    for threshold, suffix in zip(cfg["gradient_thresholds"], suffixes):
        started = time.perf_counter()
        trace, scores = gradient_gate(
            gray, query, predictions["klt_fb"], float(threshold), int(cfg["patch_size"])
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        method = "gradient_fb_" + suffix
        predictions[method] = trace
        timings[method] = round(float(timings["klt_fb"]) + elapsed, 3)
        diagnostics[method] = scores
    predictions["_gradient_similarity"] = diagnostics
    for method in METHODS:
        trace = predictions[method]
        if len(trace) != len(gray):
            raise RuntimeError("KINETICS_BATCH45_TRACE_LENGTH:" + method)
        if trace[:int(query["query_time"])] != [None] * int(query["query_time"]):
            raise RuntimeError("KINETICS_BATCH45_PREQUERY_TRACE:" + method)
        if trace[int(query["query_time"])] is None:
            raise RuntimeError("KINETICS_BATCH45_QUERY_POINT_MISSING:" + method)
    return predictions, timings


def bind_core(public_only: bool) -> None:
    core.DATA = DATA
    core.OUT = OUT
    core.METHODS = METHODS
    core.gitsha = gitsha
    core.load_config = load_config
    core.predict_query = predict_query
    core.src.audit = src.audit_public if public_only else src.audit_full



def final_audit(cfg: dict) -> None:
    manifest = src.audit_full()
    config_sha = sha_bytes(json_bytes(cfg))
    manifest_sha = sha_bytes((DATA / "manifest.json").read_bytes())
    if manifest.get("source_commit") != gitsha():
        raise RuntimeError("KINETICS_BATCH45E_MANIFEST_SOURCE_MISMATCH")
    for split in ("dev", "test"):
        prediction_path = OUT / f"{split}_predictions.json"
        core.require_pin(prediction_path)
        saved = core.doc(prediction_path)
        if (saved.get("source_commit") != gitsha()
                or saved.get("manifest_sha256") != manifest_sha
                or saved.get("config_sha256") != config_sha
                or saved.get("oracle_unread_during_prediction") is not True):
            raise RuntimeError("KINETICS_BATCH45E_FINAL_PREDICTION_AUDIT:" + split)
        result = core.doc(OUT / f"{split}_results.json")
        if (result.get("prediction_sha256") != sha_bytes(prediction_path.read_bytes())
                or result.get("source_commit") != gitsha()
                or result.get("config_sha256") != config_sha
                or result.get("split") != split):
            raise RuntimeError("KINETICS_BATCH45E_FINAL_SCORE_AUDIT:" + split)
    selection_path = OUT / "dev_selection.json"
    core.require_pin(selection_path)
    selection = core.doc(selection_path)
    if (selection.get("source_commit") != gitsha()
            or selection.get("config_sha256") != config_sha
            or selection.get("split") != "dev"):
        raise RuntimeError("KINETICS_BATCH45E_DEV_SELECTION_AUDIT")
    test_saved = core.doc(OUT / "test_predictions.json")
    if test_saved.get("dev_selection_sha256") != sha_bytes(selection_path.read_bytes()):
        raise RuntimeError("KINETICS_BATCH45E_TEST_DEV_SELECTION_SHA_MISMATCH")
    print("KINETICS_BATCH45E_EVALUATION_FINAL_AUDIT_PASS", gitsha(), flush=True)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=(
        "audit-public", "audit", "predict-dev", "score-dev", "select-dev",
        "predict-test", "score-test", "final-audit",
    ))
    parser.add_argument("--config", default="configs/kinetics_batch45.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cv2.ocl.setUseOpenCL(False)
    if cv2.ocl.useOpenCL():
        raise RuntimeError("KINETICS_BATCH45_OPENCV_OPENCL_MUST_BE_DISABLED")
    if args.action == "audit-public":
        src.audit_public()
        return
    if args.action == "audit":
        src.audit_full()
        return
    if args.action.startswith("predict-"):
        bind_core(public_only=True)
        core.predict(args.action.split("-")[1], cfg)
    elif args.action.startswith("score-"):
        bind_core(public_only=False)
        core.score(args.action.split("-")[1], cfg)
    elif args.action == "select-dev":
        bind_core(public_only=True)
        core.select_dev(cfg)
    else:
        bind_core(public_only=False)
        final_audit(cfg)


if __name__ == "__main__":
    main()
