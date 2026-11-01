"""Batch 39 SmolVLM2-500M two-image point/ABSENT endpoint baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

import tapvidmv_data as src
import tapvidmv_eval as ev

ROOT = Path(__file__).resolve().parents[1]
DATA = src.FROZEN
OUT = ROOT / "outputs/tapvidmv_droid_batch39c"
MODEL = Path(os.environ.get(
    "MVVG_VLM_MODEL",
    str(ROOT / "models/SmolVLM2-500M-Video-Instruct"),
))
MAX_NEW_TOKENS = 32
PROMPT_TEMPLATE = (
    "Image A is the query frame. A red cross marks one exact physical surface point. "
    "Image B is a later frame from the same camera. Find the SAME physical point, "
    "not a similar object or its center. If the point is occluded or outside Image B, "
    "reply exactly ABSENT. Otherwise reply with exactly two integer pixel coordinates "
    "as x y, where x is from 0 to {width_minus_one} and y is from 0 to {height_minus_one}. "
    "No explanation, no labels, no other text."
)


def prompt_for(width: int, height: int) -> str:
    return PROMPT_TEMPLATE.format(
        width_minus_one=width - 1,
        height_minus_one=height - 1,
    )


def parse_reply(reply: str, width: int, height: int):
    text = reply.strip().strip("`").strip()
    if re.fullmatch(r"(?i)ABSENT[.!]?", text):
        return None, "explicit_absent"
    pattern = (
        r"\s*[\(\[\{]?\s*(?:x\s*[:=]\s*)?([0-9]{1,4})"
        r"\s*[,;\s]\s*(?:y\s*[:=]\s*)?([0-9]{1,4})"
        r"\s*[\)\]\}]?\s*[.!]?\s*"
    )
    match = re.fullmatch(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None, "invalid_format"
    xy = [int(match.group(1)), int(match.group(2))]
    if not (0 <= xy[0] < width and 0 <= xy[1] < height):
        return None, "out_of_bounds"
    return xy, "valid_xy"


def query_image(frame: np.ndarray, xy: list[float]) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    x, y = [float(v) for v in xy]
    draw.line((x - 8, y, x + 8, y), fill=(255, 0, 0), width=3)
    draw.line((x, y - 8, x, y + 8), fill=(255, 0, 0), width=3)
    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 255, 0))
    return image


def infer(model, processor, query_path: Path, target_path: Path, *, dtype,
          width: int, height: int):
    import torch

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Image A (marked query point):"},
            {"type": "image", "path": str(query_path)},
            {"type": "text", "text": "Image B (target frame):"},
            {"type": "image", "path": str(target_path)},
            {"type": "text", "text": prompt_for(width, height)},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda", dtype=dtype)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    torch.cuda.synchronize()
    answer = processor.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()
    return answer, round(time.perf_counter() - started, 4)


def save_pair(frames: np.ndarray, query: dict, scene: str) -> tuple[Path, Path]:
    image_dir = OUT / "vlm_private_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    qid = query["id"]
    query_frame = int(query["query_frame"])
    target_frame = int(query["target_frame"])
    query_path = image_dir / f"{scene}_{qid}_query.png"
    target_path = image_dir / f"{scene}_{qid}_target.png"
    if not query_path.exists():
        query_image(frames[query_frame], query["query_xy"]).save(query_path)
    if not target_path.exists():
        Image.fromarray(np.asarray(frames[target_frame]), "RGB").save(target_path)
    return query_path, target_path


def gitsha() -> str:
    return ev.gitsha()


def predict(split: str) -> None:
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    manifest = src.audit_public()
    queries = ev.doc(DATA / "queries-public.json")
    OUT.mkdir(parents=True, exist_ok=True)
    output_path = OUT / f"{split}_vlm_predictions.json"
    manifest_sha = ev.sha_bytes((DATA / "manifest.json").read_bytes())
    errata_sha = ev.sha_bytes((DATA / "manifest-errata.json").read_bytes())
    public_sha = ev.sha_bytes((DATA / "queries-public.json").read_bytes())
    if output_path.exists():
        ev.require_pin(output_path)
        old = ev.doc(output_path)
        if (old.get("source_commit") != gitsha()
                or old.get("manifest_sha256") != manifest_sha
                or old.get("manifest_errata_sha256") != errata_sha
                or old.get("public_query_sha256") != public_sha):
            raise RuntimeError("TAPVIDMV_VLM_STALE_PIN")
        print("TAPVIDMV_VLM_EXISTING_PIN_OK", split, flush=True)
        return
    weights_path = MODEL / "model.safetensors"
    if not weights_path.is_file():
        raise RuntimeError("FROZEN_LOCAL_SMOLVLM2_500M_CHECKPOINT_MISSING:" + str(MODEL))
    weights_sha = src.sha256_file(weights_path)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL), local_files_only=True, torch_dtype=dtype
    ).to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    records = []
    for scene in manifest["scenes"]:
        if scene["split"] != split:
            continue
        selected = []
        for camera_id in range(3):
            candidates = sorted(
                [q for q in queries[scene["name"]] if int(q["camera_id"]) == camera_id],
                key=lambda q: q["id"],
            )
            if not candidates:
                raise RuntimeError("VLM_QUERY_MISSING_FOR_CAMERA")
            selected.append(candidates[0])
        for query in selected:
            camera_id = int(query["camera_id"])
            frames = np.load(
                DATA / scene["frame_files"][str(camera_id)],
                mmap_mode="r",
                allow_pickle=False,
            )
            query_path, target_path = save_pair(frames, query, scene["name"])
            width, height = scene["resolution_by_camera"][str(camera_id)]
            answer, seconds = infer(
                model, processor, query_path, target_path, dtype=dtype,
                width=width, height=height,
            )
            xy, status = parse_reply(answer, width, height)
            prompt = prompt_for(width, height)
            records.append({
                "name": scene["name"],
                "query_id": query["id"],
                "camera_id": camera_id,
                "target_frame": int(query["target_frame"]),
                "reply": answer,
                "format": status,
                "xy": xy,
                "inference_seconds": seconds,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            })
            print("TAPVIDMV_VLM_QUERY_DONE", split, scene["name"], query["id"],
                  status, seconds, flush=True)
    payload = {
        "dataset": "TAPVid-MV DROID public Rerun viewer subset",
        "split": split,
        "source_commit": gitsha(),
        "manifest_sha256": manifest_sha,
        "manifest_errata_sha256": errata_sha,
        "public_query_sha256": public_sha,
        "model_id": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "model_weights_sha256": weights_sha,
        "prompt_template_sha256": hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest(),
        "label_space": "2D point coordinate or ABSENT at one fixed endpoint",
        "visual_input": "query RGB frame with red cross + target RGB frame; no intermediate frames",
        "oracle_unread_during_prediction": True,
        "queries_per_scene": 3,
        "dtype": str(dtype),
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "records": records,
    }
    ev.once(output_path, payload)
    ev.pin(output_path)
    print("TAPVIDMV_VLM_PREDICTIONS_PINNED", split, len(records),
          ev.sha_bytes(output_path.read_bytes()), flush=True)


def score(split: str) -> None:
    manifest = src.audit()
    queries = ev.doc(DATA / "queries-public.json")
    vlm_path = OUT / f"{split}_vlm_predictions.json"
    cpu_path = OUT / f"{split}_predictions.json"
    ev.require_pin(vlm_path)
    ev.require_pin(cpu_path)
    vlm = ev.doc(vlm_path)
    cpu = ev.doc(cpu_path)
    manifest_sha = ev.sha_bytes((DATA / "manifest.json").read_bytes())
    errata_sha = ev.sha_bytes((DATA / "manifest-errata.json").read_bytes())
    public_sha = ev.sha_bytes((DATA / "queries-public.json").read_bytes())
    if (vlm.get("source_commit") != gitsha()
            or cpu.get("source_commit") != gitsha()
            or vlm.get("manifest_sha256") != manifest_sha
            or cpu.get("manifest_sha256") != manifest_sha
            or vlm.get("manifest_errata_sha256") != errata_sha
            or cpu.get("manifest_errata_sha256") != errata_sha
            or vlm.get("public_query_sha256") != public_sha
            or cpu.get("public_query_sha256") != public_sha
            or vlm.get("oracle_unread_during_prediction") is not True
            or cpu.get("oracle_unread_during_prediction") is not True
            or vlm.get("prompt_template_sha256")
                != hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()):
        raise RuntimeError("TAPVIDMV_VLM_SCORE_FIREWALL_OR_SOURCE_DRIFT")
    scenes = {x["name"]: x for x in manifest["scenes"] if x["split"] == split}
    vlm_rows = {(x["name"], x["query_id"]): x for x in vlm["records"]}
    cpu_rows = {(x["name"], x["query_id"]): x for x in cpu["records"]}
    expected_vlm_keys = set()
    for scene_name in scenes:
        for camera_id in range(3):
            candidates = sorted(
                [q for q in queries[scene_name] if int(q["camera_id"]) == camera_id],
                key=lambda q: q["id"],
            )
            if not candidates:
                raise RuntimeError("TAPVIDMV_EXPECTED_VLM_QUERY_MISSING")
            expected_vlm_keys.add((scene_name, candidates[0]["id"]))
    if (set(vlm_rows) != expected_vlm_keys
            or not set(vlm_rows).issubset(set(cpu_rows))):
        raise RuntimeError("TAPVIDMV_VLM_CPU_QUERY_SET_MISMATCH")
    totals = {
        method: {
            "visible": 0, "visible_correct8": 0, "visible_accepted": 0,
            "occluded": 0, "occluded_false_accept": 0, "occluded_correct_absent": 0,
            "out": 0, "out_false_accept": 0, "out_correct_absent": 0,
            "unknown": 0, "missing": 0, "format_invalid": 0,
        } for method in ["vlm_500m_endpoint"] + list(ev.METHODS)
    }
    per_scene = {}
    status_names = {value: key for key, value in src.STATUS.items()}
    for scene_name, scene in scenes.items():
        oracle_path = DATA / scene["oracle_file"]
        with np.load(oracle_path, allow_pickle=False) as oracle:
            qids = [str(x) for x in oracle["qids"].tolist()]
            q_index = {qid: i for i, qid in enumerate(qids)}
            scene_total = {
                method: {key: 0 for key in totals[method]}
                for method in totals
            }
            for key, row in vlm_rows.items():
                if key[0] != scene_name:
                    continue
                qid = row["query_id"]
                if qid not in q_index:
                    raise RuntimeError("TAPVIDMV_VLM_ORACLE_QUERY_MISSING")
                qi = q_index[qid]
                query = next(q for q in queries[scene_name] if q["id"] == qid)
                target = int(query["target_frame"])
                if row["target_frame"] != target:
                    raise RuntimeError("TAPVIDMV_VLM_ENDPOINT_FRAME_DRIFT")
                width, height = scene["resolution_by_camera"][str(query["camera_id"])]
                expected_prompt_sha = hashlib.sha256(
                    prompt_for(width, height).encode()
                ).hexdigest()
                if row["prompt_sha256"] != expected_prompt_sha:
                    raise RuntimeError("TAPVIDMV_VLM_PROMPT_HASH_DRIFT")
                status = int(oracle["depth_status"][qi, target])
                gt = np.asarray(oracle["gt_xy"][qi, target], dtype=np.float64)
                methods = {
                    "vlm_500m_endpoint": row["xy"],
                    **{
                        method: cpu_rows[key]["traces"][method][target]
                        for method in ev.METHODS
                    },
                }
                for method, xy in methods.items():
                    counts = scene_total[method]
                    counts[status_names[status]] += 1
                    if method == "vlm_500m_endpoint" and row["format"] in (
                        "invalid_format", "out_of_bounds"
                    ):
                        counts["format_invalid"] += 1
                    if status == src.STATUS["visible"]:
                        counts["visible_accepted"] += int(xy is not None)
                        if xy is not None and np.isfinite(gt).all():
                            counts["visible_correct8"] += int(
                                np.linalg.norm(np.asarray(xy, dtype=np.float64) - gt) <= 8
                            )
                    elif status == src.STATUS["occluded"]:
                        counts["occluded_false_accept"] += int(xy is not None)
                        counts["occluded_correct_absent"] += int(xy is None)
                    elif status == src.STATUS["out"]:
                        counts["out_false_accept"] += int(xy is not None)
                        counts["out_correct_absent"] += int(xy is None)
        per_scene[scene_name] = scene_total
        for method, counts in scene_total.items():
            for key, value in counts.items():
                totals[method][key] += value
    metrics = {}
    for method, counts in totals.items():
        metrics[method] = {
            **counts,
            "pck8_depth_visible": (
                counts["visible_correct8"] / counts["visible"]
                if counts["visible"] else None
            ),
            "coverage_depth_visible": (
                counts["visible_accepted"] / counts["visible"]
                if counts["visible"] else None
            ),
            "precision8_depth_visible": (
                counts["visible_correct8"] / counts["visible_accepted"]
                if counts["visible_accepted"] else None
            ),
            "false_accept_occluded": (
                counts["occluded_false_accept"] / counts["occluded"]
                if counts["occluded"] else None
            ),
            "false_accept_out": (
                counts["out_false_accept"] / counts["out"]
                if counts["out"] else None
            ),
        }
    result = {
        "dataset": "TAPVid-MV DROID public Rerun viewer subset",
        "source_commit": gitsha(),
        "split": split,
        "query_count": len(vlm_rows),
        "manifest_sha256": manifest_sha,
        "vlm_prediction_sha256": ev.sha_bytes(vlm_path.read_bytes()),
        "cpu_prediction_sha256": ev.sha_bytes(cpu_path.read_bytes()),
        "model_id": vlm["model_id"],
        "model_weights_sha256": vlm["model_weights_sha256"],
        "prompt_template_sha256": vlm["prompt_template_sha256"],
        "metrics": metrics,
        "per_scene": per_scene,
        "limitations": [
            "SmolVLM2-500M receives only the query and endpoint RGB images; CPU methods receive every intermediate frame.",
            "The endpoint is a fixed cloud-snapshot frame chosen from metadata, not selected from TEST outcomes.",
            "Depth-visible and depth-occluded are derived from the exact released point-cloud snapshot; unknown/missing are not scored as absent.",
            "The tolerant coordinate parser is fixed before TEST; no TEST reply was used to change it.",
        ],
    }
    output_path = OUT / f"{split}_vlm_results.json"
    ev.once(output_path, result)
    print("TAPVIDMV_VLM_SCORED", split, json.dumps(metrics, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("predict-dev", "score-dev", "predict-test", "score-test"))
    args = parser.parse_args()
    operation, split = args.action.split("-")
    if operation == "predict":
        predict(split)
    else:
        score(split)


if __name__ == "__main__":
    main()
