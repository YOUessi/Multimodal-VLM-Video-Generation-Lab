"""Batch41 SmolVLM2 point-visibility baseline with a DEV-only interface gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

import robotap_batch41_data as src
import robotap_batch41_eval as ev

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/robotap_batch41.json"
MODEL = Path(os.environ.get(
    "MVVG_VLM_MODEL", str(ROOT / "models/SmolVLM2-500M-Video-Instruct")
))
PROMPT = (
    "Image A shows a specific physical point marked with a small red cross. "
    "Image B is a later frame from the same video. Decide whether that exact "
    "physical point on the same surface is still visually visible in Image B. "
    "Do not answer based only on whether a similar object is present. "
    "Reply with exactly one token: VISIBLE, ABSENT, or UNCERTAIN. "
    "Use ABSENT if the point is hidden or outside the image. "
    "If you cannot tell, use UNCERTAIN. No explanation."
)
PROMPT_SHA = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()


def parse_visibility(reply: str) -> tuple[str | None, str]:
    text = reply.strip().strip("`").strip().upper()
    if text in {"VISIBLE", "ABSENT", "UNCERTAIN"}:
        return text, "valid_" + text.lower()
    return None, "invalid_format"


def query_image(frame: np.ndarray, xy: list[float]) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    x, y = float(xy[0]), float(xy[1])
    draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=(255, 0, 0), width=3)
    draw.line((x - 12, y, x + 12, y), fill=(255, 0, 0), width=2)
    draw.line((x, y - 12, x, y + 12), fill=(255, 0, 0), width=2)
    return image


def save_pair(frames: np.ndarray, scene: str, query: dict, target_index: int,
              output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{scene}_{query['id']}_t{target_index:04d}"
    reference_path = output_dir / (prefix + "_query.png")
    target_path = output_dir / (prefix + "_target.png")
    if not reference_path.exists():
        query_image(frames[int(query["query_time"])], query["query_xy"]).save(reference_path)
    if not target_path.exists():
        Image.fromarray(np.asarray(frames[target_index], dtype=np.uint8), mode="RGB").save(target_path)
    return reference_path, target_path


def infer(model, processor, reference_path: Path, target_path: Path, dtype) -> tuple[str, float]:
    import torch
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Query frame with the marked point:"},
            {"type": "image", "path": str(reference_path)},
            {"type": "text", "text": "Later frame:"},
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


def weights_sha() -> str:
    path = MODEL / "model.safetensors"
    if not path.is_file():
        raise RuntimeError("ROBOTAP_LOCAL_VLM_WEIGHTS_MISSING:" + str(path))
    return src.sha_file(path)


def predict(split: str, cfg: dict) -> None:
    manifest = src.audit(cfg)
    frozen = ROOT / cfg["dataset_dir"] / "batch41_frozen"
    queries = ev.load_json(frozen / "queries-public.json")
    output_dir = ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"vlm_visibility_{split}_predictions.json"
    gate_path = output_dir / "vlm_visibility_dev_gate.json"
    gate = None
    gate_sha = None
    if split == "test":
        if not gate_path.is_file():
            raise RuntimeError("ROBOTAP_VLM_DEV_GATE_MISSING")
        gate = ev.load_json(gate_path)
        gate_sha = ev.sha_file(gate_path)
        if gate.get("source_commit") != ev.current_sha():
            raise RuntimeError("ROBOTAP_VLM_DEV_GATE_SOURCE_DRIFT")
        if gate.get("status") != "passed":
            payload = {
                "status": "skipped_dev_gate", "split": split,
                "source_commit": ev.current_sha(),
                "manifest_sha256": ev.sha_file(frozen / "manifest.json"),
                "public_query_sha256": ev.sha_file(frozen / "queries-public.json"),
                "dev_gate_sha256": gate_sha, "model_id": cfg["vlm_model_id"],
                "prompt_sha256": PROMPT_SHA, "oracle_unread_during_prediction": True,
                "records": [], "skip_reason": "DEV interface gate did not pass",
            }
            ev.write_once(output, payload)
            ev.pin(output)
            print("ROBOTAP_VLM_TEST_SKIPPED_DEV_GATE", flush=True)
            return
    if output.exists():
        ev.require_pin(output)
        old = ev.load_json(output)
        if (old.get("source_commit") != ev.current_sha()
                or old.get("split") != split
                or old.get("manifest_sha256") != ev.sha_file(frozen / "manifest.json")
                or old.get("dev_gate_sha256") != gate_sha):
            raise RuntimeError("ROBOTAP_VLM_EXISTING_PREDICTION_STALE")
        return
    checkpoint_sha = weights_sha()
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    if not torch.cuda.is_available():
        raise RuntimeError("ROBOTAP_VLM_CUDA_REQUIRED")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL), local_files_only=True, torch_dtype=dtype
    ).to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    image_dir = output_dir / "vlm_private_query_images"
    records = []
    target_fractions = [float(x) for x in cfg["vlm_target_fractions"]]
    for scene in manifest["scenes"]:
        if scene["split"] != split:
            continue
        frames = np.load(frozen / scene["frames"], mmap_mode="r", allow_pickle=False)
        for query in queries[scene["name"]]:
            query_time = int(query["query_time"])
            for fraction in target_fractions:
                target_index = query_time + int(round(
                    (len(frames) - 1 - query_time) * fraction
                ))
                if target_index <= query_time:
                    continue
                reference_path, target_path = save_pair(
                    frames, scene["name"], query, target_index, image_dir
                )
                reply, seconds = infer(
                    model, processor, reference_path, target_path, dtype
                )
                prediction, status = parse_visibility(reply)
                records.append({
                    "scene": scene["name"], "query_id": query["id"],
                    "query_time": query_time, "target_index": target_index,
                    "target_fraction": fraction, "prediction": prediction,
                    "status": status, "reply": reply,
                    "inference_seconds": seconds,
                })
                print("ROBOTAP_VLM_QUERY_DONE", split, scene["name"],
                      query["id"], target_index, status, flush=True)
    payload = {
        "status": "predicted", "split": split,
        "source_commit": ev.current_sha(),
        "manifest_sha256": ev.sha_file(frozen / "manifest.json"),
        "public_query_sha256": ev.sha_file(frozen / "queries-public.json"),
        "dev_gate_sha256": gate_sha,
        "model_id": cfg["vlm_model_id"],
        "model_weights_sha256": checkpoint_sha,
        "prompt_sha256": PROMPT_SHA,
        "target_fractions": target_fractions,
        "oracle_unread_during_prediction": True,
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "records": records,
    }
    ev.write_once(output, payload)
    ev.pin(output)
    del model
    torch.cuda.empty_cache()
    print("ROBOTAP_VLM_PREDICTIONS_PINNED", split, len(records), ev.sha_file(output), flush=True)


def score(split: str, cfg: dict) -> dict:
    manifest = src.audit(cfg)
    frozen = ROOT / cfg["dataset_dir"] / "batch41_frozen"
    output_dir = ROOT / cfg["output_dir"]
    prediction_path = output_dir / f"vlm_visibility_{split}_predictions.json"
    ev.require_pin(prediction_path)
    sealed = ev.load_json(prediction_path)
    if sealed.get("status") == "skipped_dev_gate":
        result = {
            "status": "skipped_dev_gate", "split": split,
            "source_commit": ev.current_sha(),
            "prediction_sha256": ev.sha_file(prediction_path),
            "reason": sealed.get("skip_reason"),
        }
        result_path = output_dir / f"vlm_visibility_{split}_results.json"
        ev.write_once(result_path, result)
        ev.pin(result_path)
        print("ROBOTAP_VLM_SCORE_SKIPPED", split, flush=True)
        return result
    gate_path = output_dir / "vlm_visibility_dev_gate.json"
    gate_sha = ev.sha_file(gate_path) if split == "test" else None
    if split == "test":
        gate = ev.load_json(gate_path)
        if gate.get("status") != "passed" or gate.get("source_commit") != ev.current_sha():
            raise RuntimeError("ROBOTAP_VLM_TEST_SCORE_GATE_DRIFT")
    if (sealed.get("source_commit") != ev.current_sha()
            or sealed.get("split") != split
            or sealed.get("manifest_sha256") != ev.sha_file(frozen / "manifest.json")
            or sealed.get("public_query_sha256") != ev.sha_file(frozen / "queries-public.json")
            or sealed.get("dev_gate_sha256") != gate_sha
            or sealed.get("prompt_sha256") != PROMPT_SHA
            or sealed.get("model_weights_sha256") != weights_sha()
            or sealed.get("oracle_unread_during_prediction") is not True):
        raise RuntimeError("ROBOTAP_VLM_SCORE_SOURCE_OR_ORACLE_DRIFT")
    gold = src.load_gold(cfg)
    queries = ev.load_json(frozen / "queries-public.json")
    expected = set()
    for scene in manifest["scenes"]:
        if scene["split"] != split:
            continue
        for query in queries[scene["name"]]:
            qtime = int(query["query_time"])
            for fraction in [float(x) for x in cfg["vlm_target_fractions"]]:
                target = qtime + int(round((scene["nframes"] - 1 - qtime) * fraction))
                if target > qtime:
                    expected.add((scene["name"], query["id"], target))
    actual = [(r["scene"], r["query_id"], int(r["target_index"])) for r in sealed["records"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError("ROBOTAP_VLM_QUERY_COVERAGE_DRIFT")
    counts = {
        "visible_total": 0, "absent_total": 0, "visible_correct": 0,
        "absent_correct": 0, "visible_predicted": 0, "absent_false_accept": 0,
        "uncertain": 0, "invalid": 0,
    }
    for record in sealed["records"]:
        truth_visible = not bool(
            gold[record["scene"]]["tracks"][record["query_id"]]["occluded"][int(record["target_index"])]
        )
        pred = record["prediction"]
        if truth_visible:
            counts["visible_total"] += 1
            counts["visible_predicted"] += int(pred == "VISIBLE")
            counts["visible_correct"] += int(pred == "VISIBLE")
        else:
            counts["absent_total"] += 1
            counts["absent_false_accept"] += int(pred == "VISIBLE")
            counts["absent_correct"] += int(pred == "ABSENT")
        counts["uncertain"] += int(pred == "UNCERTAIN")
        counts["invalid"] += int(record["status"] == "invalid_format")
    total = len(sealed["records"])
    decided = total - counts["uncertain"] - counts["invalid"]
    correct = counts["visible_correct"] + counts["absent_correct"]
    visible_recall = (
        counts["visible_correct"] / counts["visible_total"]
        if counts["visible_total"] else None
    )
    absent_recall = (
        counts["absent_correct"] / counts["absent_total"]
        if counts["absent_total"] else None
    )
    result = {
        "status": "scored", "split": split,
        "source_commit": ev.current_sha(),
        "manifest_sha256": ev.sha_file(frozen / "manifest.json"),
        "prediction_sha256": ev.sha_file(prediction_path),
        "model_weights_sha256": sealed["model_weights_sha256"],
        "prompt_sha256": PROMPT_SHA, "query_count": total,
        **counts,
        "visible_recall": visible_recall, "absent_recall": absent_recall,
        "balanced_accuracy": (
            (visible_recall + absent_recall) / 2
            if visible_recall is not None and absent_recall is not None else None
        ),
        "coverage": decided / total if total else None,
        "selective_accuracy": correct / decided if decided else None,
        "absent_false_accept_rate": (
            counts["absent_false_accept"] / counts["absent_total"]
            if counts["absent_total"] else None
        ),
        "invalid_rate": counts["invalid"] / total if total else None,
    }
    result_path = output_dir / f"vlm_visibility_{split}_results.json"
    ev.write_once(result_path, result)
    ev.pin(result_path)
    print("ROBOTAP_VLM_SCORED", split, json.dumps(result, sort_keys=True), flush=True)
    return result


def dev_gate(cfg: dict) -> dict:
    output_dir = ROOT / cfg["output_dir"]
    path = output_dir / "vlm_visibility_dev_results.json"
    if not path.is_file():
        raise RuntimeError("ROBOTAP_VLM_DEV_RESULTS_MISSING")
    result = ev.load_json(path)
    expected = sum(
        1 for record in result.get("records", [])
    )
    if result.get("status") != "scored" or result.get("source_commit") != ev.current_sha():
        raise RuntimeError("ROBOTAP_VLM_DEV_RESULTS_SOURCE_DRIFT")
    invalid_rate = float(result["invalid_rate"] or 0.0)
    coverage = float(result["coverage"] or 0.0)
    passed = (
        expected > 0
        and invalid_rate <= float(cfg["vlm_dev_gate"]["max_invalid_rate"])
        and coverage >= float(cfg["vlm_dev_gate"]["min_coverage"])
    )
    gate = {
        "status": "passed" if passed else "blocked",
        "source_commit": ev.current_sha(),
        "dev_results_sha256": ev.sha_file(path),
        "query_count": expected, "invalid_rate": invalid_rate,
        "coverage": coverage,
        "selection_basis": "format and decision coverage only; no accuracy-based tuning",
    }
    gate_path = output_dir / "vlm_visibility_dev_gate.json"
    ev.write_once(gate_path, gate)
    ev.pin(gate_path)
    print("ROBOTAP_VLM_DEV_GATE", gate["status"], json.dumps(gate, sort_keys=True), flush=True)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=(
        "predict-dev", "score-dev", "dev-gate",
        "predict-test", "score-test",
    ))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    cfg = src.load_config(Path(args.config))
    if args.action == "dev-gate":
        dev_gate(cfg)
    elif args.action.startswith("predict-"):
        predict(args.action.split("-")[1], cfg)
    else:
        score(args.action.split("-")[1], cfg)


if __name__ == "__main__":
    main()
