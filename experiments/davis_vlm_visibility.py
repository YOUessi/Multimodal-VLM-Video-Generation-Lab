"""SmolVLM2 binary visibility/abstention baseline for DAVIS initial-mask queries."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import cv2
import numpy as np
from PIL import Image

import davis_real_data as raw
import davis_real_eval as base
import davis_occlusion_eval as tracker

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.environ.get("MVVG_VLM_MODEL", str(ROOT / "models/SmolVLM2-500M-Video-Instruct")))
PROMPT = (
    "Image A is the first frame. A translucent red mask highlights the exact object instance "
    "selected by the initial annotation. Image B is a later frame from the same video. "
    "Decide whether any part of this same object instance is still visible in Image B. "
    "Do not count a similar-looking object. Reply with exactly one token: VISIBLE, ABSENT, "
    "or UNCERTAIN. Use ABSENT only if the entire selected instance is fully hidden or outside "
    "the image. If you cannot tell, use UNCERTAIN. No explanation."
)
PROMPT_SHA = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
OUT = ROOT / "outputs/davis_occlusion_v3"


def parse_visibility(reply: str) -> tuple[str | None, str]:
    text = reply.strip().strip("`").strip().upper()
    if not text:
        return None, "invalid_format"
    if "UNCERTAIN" in text or "UNSURE" in text:
        return "UNCERTAIN", "valid_uncertain"
    if ("NOT VISIBLE" in text or "NO LONGER VISIBLE" in text
            or "FULLY OCCLUDED" in text or "OUT OF FRAME" in text
            or re.search(r"\bABSENT\b", text)):
        return "ABSENT", "valid_absent"
    if re.search(r"\bVISIBLE\b", text):
        return "VISIBLE", "valid_visible"
    return None, "invalid_format"


def highlighted_reference(frame_bgr: np.ndarray, mask_path: Path) -> Image.Image:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError("DAVIS_VLM_INITIAL_MASK_MISSING")
    if mask.shape != frame_bgr.shape[:2]:
        raise RuntimeError("DAVIS_VLM_INITIAL_MASK_SHAPE_MISMATCH")
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    overlay = rgb.astype(np.float32)
    active = mask > 0
    red = np.zeros_like(overlay)
    red[..., 0] = 255.0
    overlay[active] = 0.35 * overlay[active] + 0.65 * red[active]
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
    return Image.fromarray(overlay)


def save_pair(dataset: Path, row: dict, query: dict, target_index: int) -> tuple[Path, Path]:
    image_dir = OUT / "vlm_visibility_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    reference_path = image_dir / f"{row['name']}_frame{target_index:02d}_reference.png"
    target_path = image_dir / f"{row['name']}_frame{target_index:02d}_target.png"
    if not reference_path.is_file():
        first = cv2.imread(str(dataset / row["assets"][0]["jpg"]), cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError("DAVIS_VLM_REFERENCE_FRAME_DECODE_FAILED")
        highlighted_reference(first, dataset / query["query_mask"]).save(reference_path)
    if not target_path.is_file():
        target = cv2.imread(str(dataset / row["assets"][target_index]["jpg"]), cv2.IMREAD_COLOR)
        if target is None:
            raise RuntimeError("DAVIS_VLM_TARGET_FRAME_DECODE_FAILED")
        Image.fromarray(cv2.cvtColor(target, cv2.COLOR_BGR2RGB)).save(target_path)
    return reference_path, target_path


def infer(model, processor, reference_path: Path, target_path: Path, dtype) -> tuple[str, float]:
    import torch
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Image A, highlighted initial object instance:"},
            {"type": "image", "path": str(reference_path)},
            {"type": "text", "text": "Image B, later frame:"},
            {"type": "image", "path": str(target_path)},
            {"type": "text", "text": PROMPT},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to("cuda", dtype=dtype)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()
    reply = processor.decode(
        output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip()
    return reply, round(time.perf_counter() - started, 4)


def source_sha() -> str:
    return tracker.current_sha()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predict(cfg: dict, split: str) -> None:
    dataset = ROOT / cfg["dataset_dir"]
    manifest = raw.check(cfg, include_future_masks_and_gold=False)
    rows = [r for r in manifest["rows"] if r["split"] == split]
    mh = raw.sha256((dataset / "manifest.json").read_bytes())
    target_indices = [int(x) for x in cfg["vlm_target_frame_indices"]]
    gate_path = OUT / "vlm_visibility_dev_gate.json"
    gate = None
    gate_sha = None
    if split == "test":
        if not gate_path.is_file():
            raise RuntimeError("DAVIS_VLM_DEV_GATE_MISSING")
        gate = json.loads(gate_path.read_text())
        if gate.get("source_commit") != source_sha() or gate.get("status") != "passed":
            raise RuntimeError("DAVIS_VLM_DEV_GATE_SOURCE_OR_STATUS_DRIFT")
        gate_sha = raw.sha256(gate_path.read_bytes())
    output = OUT / f"vlm_visibility_{split}_predictions.json"
    if output.is_file():
        base.require_pinned_prediction(output)
        old = json.loads(output.read_text())
        old_weights = file_sha256(MODEL / "model.safetensors")
        if (old.get("source_commit") != source_sha()
                or old.get("manifest_sha256") != mh
                or old.get("prompt_sha256") != PROMPT_SHA
                or old.get("model_weights_sha256") != old_weights
                or old.get("model_id") != cfg["vlm_model_id"]
                or old.get("dev_gate_sha256") != gate_sha
                or old.get("split") != split
                or old.get("target_frame_indices") != target_indices
                or old.get("query_count") != len(rows) * len(target_indices)):
            raise RuntimeError("DAVIS_VLM_VISIBILITY_STALE_PREDICTION")
        return
    weights_path = MODEL / "model.safetensors"
    if not weights_path.is_file():
        raise RuntimeError("DAVIS_VLM_LOCAL_MODEL_MISSING:" + str(MODEL))
    weights_sha = file_sha256(weights_path)
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    if not torch.cuda.is_available():
        raise RuntimeError("DAVIS_VLM_CUDA_REQUIRED")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL), local_files_only=True, torch_dtype=dtype
    ).to("cuda").eval()
    records = []
    queries = json.loads((dataset / "initial_queries.json").read_text())
    for row in rows:
        query = queries[row["name"]]
        for target_index in target_indices:
            if not 1 <= target_index < len(row["assets"]):
                raise RuntimeError("DAVIS_VLM_TARGET_INDEX_OUT_OF_RANGE")
            reference_path, target_path = save_pair(dataset, row, query, target_index)
            reply, seconds = infer(model, processor, reference_path, target_path, dtype)
            label, status = parse_visibility(reply)
            records.append({
                "scene": row["name"], "split": split, "target_index": target_index,
                "prediction": label, "status": status, "reply": reply,
                "inference_seconds": seconds,
            })
            print("DAVIS_VLM_VIS_QUERY", split, row["name"], target_index, status, flush=True)
    payload = {
        "split": split, "source_commit": source_sha(), "manifest_sha256": mh,
        "model_id": cfg["vlm_model_id"], "model_weights_sha256": weights_sha,
        "prompt_sha256": PROMPT_SHA, "target_frame_indices": target_indices,
        "dev_gate_sha256": gate_sha,
        "query_count": len(records), "oracle_unread_during_prediction": True,
        "records": records,
    }
    base.write_once(output, payload)
    base.pin_prediction_bytes(output)
    del model
    torch.cuda.empty_cache()
    print("DAVIS_VLM_VIS_PREDICTIONS_PINNED", split, len(records), flush=True)


def score(cfg: dict, split: str) -> dict:
    dataset = ROOT / cfg["dataset_dir"]
    manifest = raw.check(cfg, include_future_masks_and_gold=True)
    rows = [r for r in manifest["rows"] if r["split"] == split]
    output = OUT / f"vlm_visibility_{split}_predictions.json"
    base.require_pinned_prediction(output)
    sealed = json.loads(output.read_text())
    mh = raw.sha256((dataset / "manifest.json").read_bytes())
    weights_path = MODEL / "model.safetensors"
    if not weights_path.is_file():
        raise RuntimeError("DAVIS_VLM_LOCAL_MODEL_MISSING_AT_SCORE")
    gate_path = OUT / "vlm_visibility_dev_gate.json"
    gate_sha = None
    if split == "test":
        if not gate_path.is_file():
            raise RuntimeError("DAVIS_VLM_VIS_TEST_GATE_MISSING_AT_SCORE")
        gate = json.loads(gate_path.read_text())
        if gate.get("source_commit") != source_sha() or gate.get("status") != "passed":
            raise RuntimeError("DAVIS_VLM_VIS_TEST_GATE_STALE_AT_SCORE")
        gate_sha = raw.sha256(gate_path.read_bytes())
    if (sealed.get("source_commit") != source_sha()
            or sealed.get("dev_gate_sha256") != gate_sha
            or sealed.get("manifest_sha256") != mh
            or sealed.get("prompt_sha256") != PROMPT_SHA
            or sealed.get("model_weights_sha256") != file_sha256(weights_path)
            or sealed.get("model_id") != cfg["vlm_model_id"]
            or sealed.get("split") != split
            or sealed.get("target_frame_indices") != [int(x) for x in cfg["vlm_target_frame_indices"]]
            or sealed.get("oracle_unread_during_prediction") is not True):
        raise RuntimeError("DAVIS_VLM_VIS_SCORE_SOURCE_OR_DATA_DRIFT")
    gold = json.loads((dataset / "gold-private.json").read_text())
    row_names = {r["name"] for r in rows}
    target_indices = [int(x) for x in cfg["vlm_target_frame_indices"]]
    expected_pairs = {(scene, index) for scene in row_names for index in target_indices}
    actual_pairs = {(r["scene"], int(r["target_index"])) for r in sealed["records"]}
    if actual_pairs != expected_pairs or len(actual_pairs) != len(sealed["records"]):
        raise RuntimeError("DAVIS_VLM_VIS_SCENE_OR_QUERY_COVERAGE_DRIFT")
    if sealed["query_count"] != len(expected_pairs):
        raise RuntimeError("DAVIS_VLM_VIS_QUERY_COUNT_DRIFT")
    counts = {
        "visible_total": 0, "absent_total": 0, "visible_predicted": 0,
        "absent_predicted": 0, "uncertain": 0, "invalid": 0,
        "visible_correct": 0, "absent_correct": 0,
    }
    for record in sealed["records"]:
        truth_visible = bool(gold[record["scene"]]["points"][record["target_index"]]["visible"])
        pred = record["prediction"]
        if truth_visible:
            counts["visible_total"] += 1
            counts["visible_predicted"] += pred == "VISIBLE"
            counts["visible_correct"] += pred == "VISIBLE"
        else:
            counts["absent_total"] += 1
            counts["absent_predicted"] += pred == "VISIBLE"
            counts["absent_correct"] += pred == "ABSENT"
        counts["uncertain"] += pred == "UNCERTAIN"
        counts["invalid"] += record["status"] == "invalid_format"
    visible_recall = (counts["visible_correct"] / counts["visible_total"]
                      if counts["visible_total"] else None)
    absent_recall = (counts["absent_correct"] / counts["absent_total"]
                     if counts["absent_total"] else None)
    decided = len(sealed["records"]) - counts["uncertain"] - counts["invalid"]
    correct = counts["visible_correct"] + counts["absent_correct"]
    balanced = ((visible_recall + absent_recall) / 2
                if visible_recall is not None and absent_recall is not None else None)
    result = {
        "split": split, "source_commit": source_sha(),
        "manifest_sha256": mh, "sealed_predictions_sha256": raw.sha256(output.read_bytes()),
        "model_weights_sha256": sealed["model_weights_sha256"],
        "prompt_sha256": PROMPT_SHA, "query_count": len(sealed["records"]),
        **counts,
        "visible_recall": visible_recall,
        "absent_recall": absent_recall,
        "balanced_accuracy_unconditional": balanced,
        "coverage": decided / len(sealed["records"]) if sealed["records"] else None,
        "selective_accuracy": correct / decided if decided else None,
        "absent_false_accept_rate": (
            counts["absent_predicted"] / counts["absent_total"]
            if counts["absent_total"] else None
        ),
    }
    base.write_once(OUT / f"vlm_visibility_{split}_results.json", result)
    print("DAVIS_VLM_VIS_SCORED", split, json.dumps(result, sort_keys=True), flush=True)
    return result


def dev_gate(cfg: dict) -> dict:
    path = OUT / "vlm_visibility_dev_results.json"
    if not path.is_file():
        raise RuntimeError("DAVIS_VLM_VIS_DEV_RESULTS_MISSING")
    result = json.loads(path.read_text())
    expected = int(cfg["dev_scenes"]) * len(cfg["vlm_target_frame_indices"])
    if result.get("split") != "dev" or result.get("source_commit") != source_sha():
        raise RuntimeError("DAVIS_VLM_VIS_DEV_RESULTS_SOURCE_DRIFT")
    invalid_rate = result["invalid"] / expected
    if (result["query_count"] != expected
            or invalid_rate > float(cfg["vlm_dev_gate"]["max_invalid_rate"])
            or result["coverage"] < float(cfg["vlm_dev_gate"]["min_coverage"])):
        raise RuntimeError("DAVIS_VLM_VIS_DEV_INTERFACE_GATE_FAILED")
    doc = {
        "status": "passed", "source_commit": source_sha(),
        "dev_results_sha256": raw.sha256(path.read_bytes()),
        "query_count": expected, "invalid_rate": invalid_rate,
        "coverage": result["coverage"],
        "selection_basis": "format and decision coverage only; no TEST labels or accuracy-based tuning",
    }
    base.write_once(OUT / "vlm_visibility_dev_gate.json", doc)
    print("DAVIS_VLM_VIS_DEV_GATE_PASS", json.dumps(doc, sort_keys=True), flush=True)
    return doc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=(
        "predict-dev", "score-dev", "dev-gate", "predict-test", "score-test"))
    parser.add_argument("--config", default="configs/davis_occlusion_v3.json")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    if args.action == "dev-gate":
        dev_gate(cfg)
    elif args.action.startswith("predict-"):
        predict(cfg, args.action.split("-")[1])
    else:
        score(cfg, args.action.split("-")[1])


if __name__ == "__main__":
    main()
