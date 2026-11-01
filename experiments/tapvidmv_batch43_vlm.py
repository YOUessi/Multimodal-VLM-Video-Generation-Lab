"""Batch43 SmolVLM2 endpoint visibility baseline with a DEV-only TEST admission gate."""
from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image, ImageDraw

import tapvidmv_batch43_data as src
import tapvidmv_batch43_eval as ev

ROOT = Path(__file__).resolve().parents[1]
DATA = src.FROZEN
OUT = ev.OUT
PROMPT = (
    "Image A and Image B are from the same fixed camera at different times. "
    "A red cross in Image A marks one exact physical 3-D surface point. "
    "Decide whether that same physical point is visible in Image B. "
    "VISIBLE means you can identify that exact point in Image B. "
    "NOT_VISIBLE means the point is hidden by an occluder or outside the image. "
    "Do not answer merely because the same person, object, or scene is present. "
    "If the evidence is insufficient, answer UNCERTAIN. "
    "Your entire response must be exactly one token: VISIBLE, NOT_VISIBLE, or UNCERTAIN. "
    "No explanation."
)
LABELS = ("VISIBLE", "NOT_VISIBLE", "UNCERTAIN", "INVALID")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def doc(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def once(path: Path, payload: dict) -> None:
    ev.once(path, payload)


def pin(path: Path) -> None:
    src.write_once(path.with_suffix(path.suffix + ".sha256"),
                   (sha_bytes(path.read_bytes()) + "\n").encode())


def require_pin(path: Path) -> None:
    tag = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not tag.is_file():
        raise RuntimeError("TAPVIDMV_VLM_MISSING_PIN:" + path.name)
    if tag.read_text(encoding="utf-8").strip() != sha_bytes(path.read_bytes()):
        raise RuntimeError("TAPVIDMV_VLM_PIN_DRIFT:" + path.name)


def parse_label(raw: str) -> tuple[str, str]:
    normalized = str(raw).strip().upper()
    normalized = normalized.strip(" \t\r\n`'\".,:;!?()[]{}")
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = normalized.replace("-", "_")
    if normalized == "VISIBLE":
        return "VISIBLE", "valid"
    if normalized in ("NOT_VISIBLE", "OCCLUDED", "OUT_OF_FRAME"):
        return "NOT_VISIBLE", "valid"
    if normalized == "UNCERTAIN":
        return "UNCERTAIN", "valid"
    return "INVALID", "invalid_format"


def marked_query_image(frame: np.ndarray, xy: list[float]) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    x, y = [float(value) for value in xy]
    draw.ellipse((x - 11, y - 11, x + 11, y + 11), outline=(255, 0, 0), width=3)
    draw.line((x - 8, y, x + 8, y), fill=(255, 0, 0), width=3)
    draw.line((x, y - 8, x, y + 8), fill=(255, 0, 0), width=3)
    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 255, 0))
    return image


def save_png_once(path: Path, image: Image.Image) -> None:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    src.write_once(path, buffer.getvalue())


def prepare_images(frames: np.ndarray, query: dict, scene: str) -> tuple[Path, Path]:
    image_dir = OUT / "vlm_private_inputs"
    image_dir.mkdir(parents=True, exist_ok=True)
    key = f"{scene}_{query['id']}_t{int(query['target_frame']):04d}"
    query_path = image_dir / f"{key}_query.png"
    target_path = image_dir / f"{key}_target.png"
    save_png_once(query_path, marked_query_image(frames[0], query["query_xy"]))
    save_png_once(target_path, Image.fromarray(np.asarray(frames[int(query["target_frame"])], dtype=np.uint8)))
    return query_path, target_path


def infer(model, processor, image_a: Path, image_b: Path, max_new_tokens: int) -> tuple[str, float]:
    import torch

    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Image A: frame-0 query point marked with a red cross."},
        {"type": "image", "path": str(image_a)},
        {"type": "text", "text": "Image B: the exact public target-frame snapshot."},
        {"type": "image", "path": str(image_b)},
        {"type": "text", "text": PROMPT},
    ]}]
    dtype = next(model.parameters()).dtype
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to("cuda", dtype=dtype)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    answer = processor.decode(
        outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip()
    return answer, round(time.perf_counter() - started, 4)


def config_sha() -> str:
    return sha_bytes(src.json_bytes(src.config()))


def gate_path() -> Path:
    return OUT / "dev_vlm_gate.json"


def predict(split: str) -> None:
    if split not in ("dev", "test"):
        raise ValueError("INVALID_VLM_SPLIT")
    cfg = src.config()
    manifest = src.audit_public()
    queries = doc(DATA / "queries-public.json")
    manifest_sha = sha_bytes((DATA / "manifest.json").read_bytes())
    public_sha = sha_bytes((DATA / "queries-public.json").read_bytes())
    cfg_sha = config_sha()
    outfile = OUT / f"{split}_vlm_predictions.json"
    OUT.mkdir(parents=True, exist_ok=True)
    gate_sha = None
    if split == "test":
        gate_file = gate_path()
        require_pin(gate_file)
        gate = doc(gate_file)
        gate_sha = sha_bytes(gate_file.read_bytes())
        if gate.get("config_sha256") != cfg_sha or gate.get("source_commit") != ev.gitsha():
            raise RuntimeError("TAPVIDMV_VLM_GATE_SOURCE_MISMATCH")
        if gate.get("status") != "PASS":
            payload = {
                "dataset": "TAPVid-MV EgoExo4D + PACE",
                "split": "test",
                "status": "SKIPPED_DEV_GATE_BLOCKED",
                "source_commit": ev.gitsha(),
                "config_sha256": cfg_sha,
                "manifest_sha256": manifest_sha,
                "public_query_sha256": public_sha,
                "dev_gate_sha256": gate_sha,
                "oracle_never_read": True,
                "records": [],
            }
            once(outfile, payload)
            pin(outfile)
            print("TAPVIDMV_VLM_TEST_SKIPPED_DEV_GATE_BLOCKED", flush=True)
            return
    if outfile.exists():
        require_pin(outfile)
        old = doc(outfile)
        if (old.get("source_commit") != ev.gitsha()
                or old.get("config_sha256") != cfg_sha
                or old.get("manifest_sha256") != manifest_sha
                or old.get("public_query_sha256") != public_sha
                or old.get("split") != split
                or old.get("dev_gate_sha256") != gate_sha):
            raise RuntimeError("TAPVIDMV_VLM_STALE_PINNED_PREDICTION")
        print("TAPVIDMV_VLM_EXISTING_PIN_OK", split, flush=True)
        return

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_cfg = cfg["vlm"]
    model_path = (ROOT / model_cfg["model_path"]).resolve()
    weights = model_path / "model.safetensors"
    if not weights.is_file():
        raise RuntimeError("TAPVIDMV_LOCAL_SMOLVLM2_CHECKPOINT_MISSING")
    model_sha = src.sha256_file(weights)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path), local_files_only=True, dtype=dtype
    ).to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()

    records = []
    for scene in manifest["scenes"]:
        if scene["split"] != split:
            continue
        camera_frames = {}
        for camera_id in range(3):
            relative = scene["frame_files"][str(camera_id)]
            camera_frames[camera_id] = np.load(DATA / relative, mmap_mode="r", allow_pickle=False)
        for query in queries[scene["name"]]:
            camera_id = int(query["camera_id"])
            frames = camera_frames[camera_id]
            image_a, image_b = prepare_images(frames, query, scene["name"])
            answer, seconds = infer(
                model, processor, image_a, image_b, int(model_cfg["max_new_tokens"])
            )
            label, fmt = parse_label(answer)
            records.append({
                "scene": scene["name"],
                "query_id": query["id"],
                "camera_id": camera_id,
                "cross_view_group": query["cross_view_group"],
                "target_frame": int(query["target_frame"]),
                "raw_reply": answer,
                "label": label,
                "format": fmt,
                "inference_seconds": seconds,
            })
            print("TAPVIDMV_VLM_QUERY_DONE", split, scene["name"], query["id"],
                  label, fmt, seconds, flush=True)
    payload = {
        "dataset": "TAPVid-MV EgoExo4D + PACE",
        "split": split,
        "status": "PREDICTED",
        "source_commit": ev.gitsha(),
        "config_sha256": cfg_sha,
        "manifest_sha256": manifest_sha,
        "public_query_sha256": public_sha,
        "dev_gate_sha256": gate_sha,
        "model_id": model_cfg["model_id"],
        "model_weights_sha256": model_sha,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "input": "frame-0 red-cross query image plus exact target-frame image from the same camera",
        "oracle_never_read": True,
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "records": records,
    }
    once(outfile, payload)
    pin(outfile)
    print("TAPVIDMV_VLM_PREDICTIONS_PINNED", split, len(records),
          sha_bytes(outfile.read_bytes()), flush=True)


def score(split: str) -> dict:
    if split not in ("dev", "test"):
        raise ValueError("INVALID_VLM_SCORE_SPLIT")
    prediction_path = OUT / f"{split}_vlm_predictions.json"
    require_pin(prediction_path)
    saved = doc(prediction_path)
    cfg_sha = config_sha()
    if saved.get("status") == "SKIPPED_DEV_GATE_BLOCKED":
        result = {
            "dataset": "TAPVid-MV EgoExo4D + PACE",
            "split": split,
            "status": "SKIPPED_DEV_GATE_BLOCKED",
            "source_commit": ev.gitsha(),
            "config_sha256": cfg_sha,
            "prediction_sha256": sha_bytes(prediction_path.read_bytes()),
            "test_oracle_read": False,
            "reason": "DEV VLM gate did not pass; TEST VLM inference and label scoring were skipped.",
        }
        result_path = OUT / f"{split}_vlm_results.json"
        once(result_path, result)
        pin(result_path)
        print("TAPVIDMV_VLM_SCORE_SKIPPED", split, flush=True)
        return result
    if saved.get("status") != "PREDICTED":
        raise RuntimeError("TAPVIDMV_VLM_PREDICTION_STATUS_INVALID")
    if split == "test":
        require_pin(gate_path())
        gate = doc(gate_path())
        if gate.get("status") != "PASS" or saved.get("dev_gate_sha256") != sha_bytes(gate_path().read_bytes()):
            raise RuntimeError("TAPVIDMV_VLM_TEST_GATE_MISMATCH")

    cpu_path = OUT / f"{split}_predictions.json"
    ev.require_pin(cpu_path)
    cpu = doc(cpu_path)
    if cpu.get("split") != split:
        raise RuntimeError("TAPVIDMV_VLM_CPU_SPLIT_MISMATCH")
    manifest_sha = sha_bytes((DATA / "manifest.json").read_bytes())
    public_sha = sha_bytes((DATA / "queries-public.json").read_bytes())
    if (saved.get("source_commit") != ev.gitsha()
            or saved.get("config_sha256") != cfg_sha
            or saved.get("manifest_sha256") != manifest_sha
            or saved.get("public_query_sha256") != public_sha
            or saved.get("oracle_never_read") is not True
            or saved.get("prompt_sha256") != hashlib.sha256(PROMPT.encode()).hexdigest()
            or saved.get("split") != split
            or cpu.get("source_commit") != ev.gitsha()
            or cpu.get("config_sha256") != cfg_sha):
        raise RuntimeError("TAPVIDMV_VLM_SCORE_FIREWALL_FAILED")

    manifest = src.audit_private(split)
    queries = doc(DATA / "queries-public.json")
    scenes = [x for x in manifest["scenes"] if x["split"] == split]
    expected = {
        (scene["name"], query["id"])
        for scene in scenes for query in queries[scene["name"]]
    }
    records = saved.get("records", [])
    keys = [(row.get("scene"), row.get("query_id")) for row in records]
    if len(keys) != len(set(keys)) or set(keys) != expected or len(keys) != len(expected):
        raise RuntimeError("TAPVIDMV_VLM_QUERY_SET_MISMATCH")
    cpu_by_key = {(row["name"], row["query_id"]): row for row in cpu["records"]}
    query_by_key = {
        (scene["name"], query["id"]): query
        for scene in scenes for query in queries[scene["name"]]
    }
    eligible_labels = []
    eligible_statuses = []
    eligible_formats = []
    cpu_endpoint = {
        method: {
            "visible": 0, "visible_correct8": 0, "visible_accepted": 0,
            "occluded": 0, "occluded_false_accept": 0,
            "out": 0, "out_false_accept": 0,
        } for method in ev.METHODS
    }
    per_subset = {}
    for scene in scenes:
        oracle_path = DATA / scene["oracle_file"]
        with np.load(oracle_path, allow_pickle=False) as oracle:
            qids = [str(value) for value in oracle["qids"].tolist()]
            q_index = {qid: index for index, qid in enumerate(qids)}
            for row in [x for x in records if x["scene"] == scene["name"]]:
                key = (scene["name"], row["query_id"])
                query = query_by_key[key]
                if (row["target_frame"] != int(query["target_frame"])
                        or row["camera_id"] != int(query["camera_id"])
                        or row["query_id"] not in q_index):
                    raise RuntimeError("TAPVIDMV_VLM_QUERY_METADATA_DRIFT")
                qi = q_index[row["query_id"]]
                target = int(row["target_frame"])
                status = int(oracle["depth_status"][qi, target])
                if status not in (src.STATUS["visible"], src.STATUS["occluded"], src.STATUS["out"]):
                    continue
                eligible_labels.append(row["label"])
                eligible_statuses.append(status)
                eligible_formats.append(row["format"])
                subset_rows = per_subset.setdefault(scene["subset"], {"labels": [], "statuses": []})
                subset_rows["labels"].append(row["label"])
                subset_rows["statuses"].append(status)
                cpu_row = cpu_by_key[key]
                gt = np.asarray(oracle["gt_xy"][qi, target], dtype=np.float64)
                for method in ev.METHODS:
                    point = cpu_row["traces"][method][target]
                    metric = cpu_endpoint[method]
                    if status == src.STATUS["visible"]:
                        metric["visible"] += 1
                        metric["visible_accepted"] += int(point is not None)
                        if point is not None and np.isfinite(gt).all():
                            metric["visible_correct8"] += int(
                                np.linalg.norm(np.asarray(point, dtype=np.float64) - gt) <= 8
                            )
                    elif status == src.STATUS["occluded"]:
                        metric["occluded"] += 1
                        metric["occluded_false_accept"] += int(point is not None)
                    else:
                        metric["out"] += 1
                        metric["out_false_accept"] += int(point is not None)

    visible_mask = np.asarray([x == src.STATUS["visible"] for x in eligible_statuses], dtype=bool)
    predicted_visible = np.asarray([x == "VISIBLE" for x in eligible_labels], dtype=bool)
    predicted_not_visible = np.asarray([x == "NOT_VISIBLE" for x in eligible_labels], dtype=bool)
    invalid = np.asarray([x == "INVALID" for x in eligible_labels], dtype=bool)
    uncertain = np.asarray([x == "UNCERTAIN" for x in eligible_labels], dtype=bool)
    nonvisible_mask = ~visible_mask
    visible_support = int(visible_mask.sum())
    nonvisible_support = int(nonvisible_mask.sum())
    visible_recall = float((predicted_visible & visible_mask).sum() / visible_support) if visible_support else None
    not_visible_recall = float((predicted_not_visible & nonvisible_mask).sum() / nonvisible_support) if nonvisible_support else None
    balanced = (
        float((visible_recall + not_visible_recall) / 2)
        if visible_recall is not None and not_visible_recall is not None else None
    )
    accepted = predicted_visible | predicted_not_visible
    correct = (predicted_visible & visible_mask) | (predicted_not_visible & nonvisible_mask)
    occluded_mask = np.asarray([x == src.STATUS["occluded"] for x in eligible_statuses], dtype=bool)
    out_mask = np.asarray([x == src.STATUS["out"] for x in eligible_statuses], dtype=bool)
    false_accept_occluded = int((predicted_visible & occluded_mask).sum())
    false_accept_out = int((predicted_visible & out_mask).sum())
    false_accept_nonvisible = false_accept_occluded + false_accept_out
    eligible_count = len(eligible_labels)
    metrics = {
        "eligible_count": eligible_count,
        "visible_support": visible_support,
        "nonvisible_support": nonvisible_support,
        "occluded_support": int(occluded_mask.sum()),
        "out_support": int(out_mask.sum()),
        "correct": int(correct.sum()),
        "accuracy_over_eligible": float(correct.sum() / eligible_count) if eligible_count else None,
        "balanced_accuracy": balanced,
        "visible_recall": visible_recall,
        "not_visible_recall": not_visible_recall,
        "coverage": float(accepted.mean()) if eligible_count else None,
        "invalid_count": int(invalid.sum()),
        "invalid_rate": float(invalid.mean()) if eligible_count else None,
        "uncertain_count": int(uncertain.sum()),
        "false_accept_occluded": false_accept_occluded,
        "false_accept_occluded_rate": float(false_accept_occluded / occluded_mask.sum()) if occluded_mask.any() else None,
        "false_accept_out": false_accept_out,
        "false_accept_out_rate": float(false_accept_out / out_mask.sum()) if out_mask.any() else None,
        "false_accept_nonvisible_rate": float(false_accept_nonvisible / nonvisible_support) if nonvisible_support else None,
        "unknown_or_missing_excluded": len(records) - eligible_count,
    }
    cpu_comparison = {}
    for method, metric in cpu_endpoint.items():
        cpu_comparison[method] = {
            **metric,
            "visible_pck8": metric["visible_correct8"] / metric["visible"] if metric["visible"] else None,
            "visible_coverage": metric["visible_accepted"] / metric["visible"] if metric["visible"] else None,
            "false_accept_occluded_rate": metric["occluded_false_accept"] / metric["occluded"] if metric["occluded"] else None,
            "false_accept_out_rate": metric["out_false_accept"] / metric["out"] if metric["out"] else None,
        }
    subset_metrics = {}
    for subset, values in per_subset.items():
        mask = np.asarray([x == src.STATUS["visible"] for x in values["statuses"]], dtype=bool)
        labels = values["labels"]
        visible_n = int(mask.sum())
        nonvisible_n = int((~mask).sum())
        subset_metrics[subset] = {
            "count": len(labels),
            "visible_support": visible_n,
            "nonvisible_support": nonvisible_n,
            "accuracy": float(np.mean([
                (label == "VISIBLE") == bool(is_visible) and label in ("VISIBLE", "NOT_VISIBLE")
                for label, is_visible in zip(labels, mask)
            ])) if labels else None,
        }
    result = {
        "dataset": "TAPVid-MV EgoExo4D + PACE",
        "split": split,
        "status": "SCORED",
        "source_commit": ev.gitsha(),
        "config_sha256": cfg_sha,
        "prediction_sha256": sha_bytes(prediction_path.read_bytes()),
        "cpu_prediction_sha256": sha_bytes(cpu_path.read_bytes()),
        "model_id": saved["model_id"],
        "model_weights_sha256": saved["model_weights_sha256"],
        "prompt_sha256": saved["prompt_sha256"],
        "vlm": metrics,
        "cpu_endpoint_comparison_at_same_target_frames": cpu_comparison,
        "per_subset": subset_metrics,
        "test_oracle_read": split == "test",
        "limitations": [
            "The VLM receives only frame 0 with a red-cross query and one exact target-frame image; it does not see intermediate frames.",
            "Depth visibility is inferred from released point-cloud snapshots and is not an official binary visibility label.",
            "UNKNOWN and MISSING oracle states are excluded from the binary visibility gate.",
            "The VLM baseline predicts visibility only; it does not output point coordinates or a 3-D position.",
            "TEST VLM inference is permitted only if all predeclared DEV gate checks pass.",
        ],
    }
    result_path = OUT / f"{split}_vlm_results.json"
    once(result_path, result)
    pin(result_path)
    print("TAPVIDMV_VLM_SCORED", split, json.dumps({
        "vlm": metrics, "cpu_endpoint_comparison": cpu_comparison
    }, sort_keys=True), flush=True)
    return result


def make_dev_gate() -> dict:
    result_path = OUT / "dev_vlm_results.json"
    if not result_path.is_file():
        raise RuntimeError("TAPVIDMV_DEV_VLM_RESULTS_MISSING")
    require_pin(result_path)
    result = doc(result_path)
    cfg = src.config()
    cfg_sha = config_sha()
    if (result.get("source_commit") != ev.gitsha()
            or result.get("config_sha256") != cfg_sha
            or result.get("split") != "dev"
            or result.get("status") != "SCORED"):
        raise RuntimeError("TAPVIDMV_VLM_DEV_GATE_SOURCE_MISMATCH")
    metrics = result["vlm"]
    rules = cfg["vlm"]["dev_gate"]
    checks = {
        "balanced_accuracy": metrics["balanced_accuracy"] is not None
            and metrics["balanced_accuracy"] >= rules["min_balanced_accuracy"],
        "class_support": metrics["visible_support"] >= rules["min_visible_support"]
            and metrics["nonvisible_support"] >= rules["min_nonvisible_support"],
        "coverage": metrics["coverage"] is not None and metrics["coverage"] >= rules["min_coverage"],
        "invalid_rate": metrics["invalid_rate"] is not None and metrics["invalid_rate"] <= rules["max_invalid_rate"],
        "false_accept_nonvisible_rate": metrics["false_accept_nonvisible_rate"] is not None
            and metrics["false_accept_nonvisible_rate"] <= rules["max_false_accept_nonvisible_rate"],
    }
    payload = {
        "dataset": "TAPVid-MV EgoExo4D + PACE",
        "split": "dev",
        "status": "PASS" if all(checks.values()) else "BLOCK",
        "source_commit": ev.gitsha(),
        "config_sha256": cfg_sha,
        "dev_vlm_results_sha256": sha_bytes(result_path.read_bytes()),
        "metrics": metrics,
        "rules": rules,
        "checks": checks,
        "test_policy": "TEST VLM prediction is skipped unless every DEV check passes.",
    }
    path = gate_path()
    once(path, payload)
    pin(path)
    print("TAPVIDMV_VLM_DEV_GATE", payload["status"], json.dumps(checks, sort_keys=True), flush=True)
    return payload


def final_audit() -> None:
    src.audit_public()
    cfg_sha = config_sha()
    for split in ("dev", "test"):
        prediction = OUT / f"{split}_vlm_predictions.json"
        result = OUT / f"{split}_vlm_results.json"
        require_pin(prediction)
        if not result.is_file():
            raise RuntimeError("TAPVIDMV_VLM_FINAL_RESULT_MISSING:" + split)
        require_pin(result)
        saved = doc(prediction)
        scored = doc(result)
        if (saved.get("source_commit") != ev.gitsha()
                or saved.get("config_sha256") != cfg_sha
                or scored.get("source_commit") != ev.gitsha()
                or scored.get("config_sha256") != cfg_sha
                or scored.get("prediction_sha256") != sha_bytes(prediction.read_bytes())):
            raise RuntimeError("TAPVIDMV_VLM_FINAL_PIN_AUDIT_FAILED:" + split)
    require_pin(gate_path())
    gate = doc(gate_path())
    test_saved = doc(OUT / "test_vlm_predictions.json")
    if test_saved.get("dev_gate_sha256") != sha_bytes(gate_path().read_bytes()):
        raise RuntimeError("TAPVIDMV_VLM_FINAL_GATE_PIN_MISMATCH")
    test_result = doc(OUT / "test_vlm_results.json")
    if gate["status"] != "PASS" and (
        test_saved.get("status") != "SKIPPED_DEV_GATE_BLOCKED"
        or test_result.get("status") != "SKIPPED_DEV_GATE_BLOCKED"
        or test_result.get("test_oracle_read") is not False
    ):
        raise RuntimeError("TAPVIDMV_VLM_BLOCKED_TEST_WAS_NOT_SKIPPED")
    if gate["status"] == "PASS" and (
        test_saved.get("status") != "PREDICTED"
        or test_result.get("status") != "SCORED"
        or test_result.get("test_oracle_read") is not True
    ):
        raise RuntimeError("TAPVIDMV_VLM_PASSED_GATE_TEST_NOT_SCORED")
    print("TAPVIDMV_VLM_FINAL_AUDIT_PASS", gate["status"], ev.gitsha(), flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=(
        "predict-dev", "score-dev", "dev-gate",
        "predict-test", "score-test", "final-audit",
    ))
    args = parser.parse_args()
    if args.action.startswith("predict-"):
        predict(args.action.split("-", 1)[1])
    elif args.action.startswith("score-"):
        score(args.action.split("-", 1)[1])
    elif args.action == "dev-gate":
        make_dev_gate()
    else:
        final_audit()


if __name__ == "__main__":
    main()
