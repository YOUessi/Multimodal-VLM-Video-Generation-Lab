"""Batch45 SmolVLM2-2.2B visibility baseline with a query-point reference crop and DEV-only gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import kinetics_batch45_data as src
import kinetics_batch45_eval as ev

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/tapvid_kinetics_batch45/batch45_frozen"
OUT = ROOT / "outputs/kinetics_batch45"
PROMPT = (
    "Image A is the first-visible query frame. A red cross marks one exact physical "
    "surface point. Image B is a crop centered on that same marked point and is only "
    "a reference for its local texture and appearance. Image C is a later frame from "
    "the same video. Decide whether the exact same physical point is visible in Image C. "
    "Do not answer VISIBLE merely because the same object or a similar object appears. "
    "If the point is hidden or outside the image, answer NOT_VISIBLE. If its visibility "
    "cannot be determined, answer UNCERTAIN. Reply with exactly one token: VISIBLE, "
    "NOT_VISIBLE, or UNCERTAIN. No explanation."
)
LABELS = ("VISIBLE", "NOT_VISIBLE", "UNCERTAIN", "INVALID")
WEIGHT_FILES = (
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def doc(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def json_bytes(value: dict) -> bytes:
    return ev.json_bytes(value)


def once(path: Path, payload: dict) -> None:
    src.atomic_once(path, json_bytes(payload))


def pin(path: Path) -> None:
    src.atomic_once(path.with_suffix(path.suffix + ".sha256"),
                    (sha_bytes(path.read_bytes()) + "\n").encode())


def require_pin(path: Path) -> None:
    tag = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not tag.is_file():
        raise RuntimeError("KINETICS_BATCH45_MISSING_PIN:" + path.name)
    if tag.read_text(encoding="utf-8").strip() != sha_bytes(path.read_bytes()):
        raise RuntimeError("KINETICS_BATCH45_PIN_DRIFT:" + path.name)


def parse_label(text: str) -> tuple[str, str]:
    raw = str(text).strip().upper()
    tokens = re.findall(r"NOT[\s_-]*VISIBLE|VISIBLE|UNCERTAIN", raw)
    if len(tokens) != 1:
        return "UNCERTAIN", "invalid_format"
    token = re.sub(r"[\s_-]+", "_", tokens[0])
    if token == "VISIBLE":
        return "VISIBLE", "valid"
    if token == "NOT_VISIBLE":
        return "NOT_VISIBLE", "valid"
    if token == "UNCERTAIN":
        return "UNCERTAIN", "valid"
    return "UNCERTAIN", "invalid_format"


def query_image(frame: np.ndarray, xy: list[float]) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    x, y = [float(v) for v in xy]
    draw.line((x - 8, y, x + 8, y), fill=(255, 0, 0), width=3)
    draw.line((x, y - 8, x, y + 8), fill=(255, 0, 0), width=3)
    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 255, 0))
    return image


def reference_crop(frame: np.ndarray, xy: list[float], size: int) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    x, y = [float(v) for v in xy]
    radius = max(8, int(size) // 2)
    crop = image.crop((
        int(round(x)) - radius, int(round(y)) - radius,
        int(round(x)) + radius, int(round(y)) + radius,
    ))
    return ImageOps.pad(crop, (int(size), int(size)), method=Image.Resampling.BICUBIC)


def prepare_images(frames: np.ndarray, query: dict, target_time: int,
                   scene: str, fraction: float, crop_size: int) -> tuple[Path, Path, Path]:
    image_dir = OUT / "vlm_private_inputs"
    image_dir.mkdir(parents=True, exist_ok=True)
    key = f"{scene}_{query['id']}_{int(round(fraction * 100)):02d}"
    query_path = image_dir / f"{key}_query.png"
    crop_path = image_dir / f"{key}_reference_crop.png"
    target_path = image_dir / f"{key}_target.png"
    qtime = int(query["query_time"])
    if not query_path.exists():
        query_image(frames[qtime], query["query_xy"]).save(query_path)
    if not crop_path.exists():
        reference_crop(frames[qtime], query["query_xy"], crop_size).save(crop_path)
    if not target_path.exists():
        Image.fromarray(np.asarray(frames[target_time], dtype=np.uint8)).save(target_path)
    return query_path, crop_path, target_path


def infer(model, processor, image_a: Path, image_b: Path, image_c: Path,
          max_new_tokens: int) -> tuple[str, float]:
    import torch

    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Image A: query frame with the exact point marked."},
        {"type": "image", "path": str(image_a)},
        {"type": "text", "text": "Image B: reference crop around the exact point."},
        {"type": "image", "path": str(image_b)},
        {"type": "text", "text": "Image C: later target frame."},
        {"type": "image", "path": str(image_c)},
        {"type": "text", "text": PROMPT},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to("cuda", dtype=next(model.parameters()).dtype)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    torch.cuda.synchronize()
    answer = processor.decode(
        outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip()
    return answer, round(time.perf_counter() - started, 4)


def model_path_and_sha(cfg: dict) -> tuple[Path, str]:
    model_path = (ROOT / cfg["vlm"]["model_path"]).resolve()
    hashes = []
    for name in WEIGHT_FILES:
        path = model_path / name
        if not path.is_file():
            raise RuntimeError("KINETICS_BATCH45_LOCAL_2_2B_WEIGHT_MISSING:" + name)
        hashes.append((name, sha_file(path)))
    return model_path, sha_bytes(json_bytes({"files": hashes}))


def dev_gate_path() -> Path:
    return OUT / "dev_vlm_gate.json"


def predict(split: str, cfg: dict) -> None:
    manifest = src.audit_public()
    queries = doc(DATA / "queries-public.json")
    manifest_sha = sha_bytes((DATA / "manifest.json").read_bytes())
    public_sha = sha_bytes((DATA / "queries-public.json").read_bytes())
    config_sha = sha_bytes(json_bytes(cfg))
    outfile = OUT / f"{split}_vlm_predictions.json"
    if outfile.exists():
        require_pin(outfile)
        old = doc(outfile)
        if (old.get("source_commit") != ev.gitsha()
                or old.get("manifest_sha256") != manifest_sha
                or old.get("public_query_sha256") != public_sha
                or old.get("config_sha256") != config_sha):
            raise RuntimeError("KINETICS_BATCH45_VLM_STALE_SOURCE")
        print("KINETICS_BATCH45_VLM_EXISTING_PIN_OK", split, flush=True)
        return

    gate_sha = None
    if split == "test":
        gate_path = dev_gate_path()
        require_pin(gate_path)
        gate = doc(gate_path)
        gate_sha = sha_bytes(gate_path.read_bytes())
        if gate.get("config_sha256") != config_sha:
            raise RuntimeError("KINETICS_BATCH45_VLM_GATE_CONFIG_MISMATCH")
        if gate.get("status") != "PASS":
            payload = {
                "dataset": "TAP-Vid-Kinetics fresh action-class holdout",
                "split": "test",
                "status": "SKIPPED_DEV_GATE_BLOCKED",
                "source_commit": ev.gitsha(),
                "manifest_sha256": manifest_sha,
                "public_query_sha256": public_sha,
                "config_sha256": config_sha,
                "dev_gate_sha256": gate_sha,
                "oracle_never_read": True,
                "records": [],
            }
            once(outfile, payload)
            pin(outfile)
            print("KINETICS_BATCH45_VLM_TEST_SKIPPED_DEV_GATE_BLOCKED", flush=True)
            return

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_path, model_sha = model_path_and_sha(cfg)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype=dtype
    ).to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()

    records = []
    fractions = [float(x) for x in cfg["vlm"]["target_fractions"]]
    for scene in manifest["scenes"]:
        if scene["split"] != split:
            continue
        frames = np.load(ROOT / scene["frame_file"], mmap_mode="r", allow_pickle=False)
        for query in queries[scene["name"]]:
            qtime = int(query["query_time"])
            for fraction in fractions:
                target_time = qtime + int(round(fraction * (len(frames) - 1 - qtime)))
                if target_time <= qtime:
                    continue
                image_a, image_b, image_c = prepare_images(
                    frames, query, target_time, scene["name"], fraction,
                    int(cfg["vlm"]["reference_crop_size"]),
                )
                answer, seconds = infer(
                    model, processor, image_a, image_b, image_c,
                    int(cfg["vlm"]["max_new_tokens"]),
                )
                label, fmt = parse_label(answer)
                records.append({
                    "scene": scene["name"],
                    "query_id": query["id"],
                    "target_fraction": fraction,
                    "target_time": target_time,
                    "raw_reply": answer,
                    "label": label,
                    "format": fmt,
                    "inference_seconds": seconds,
                })
                print("KINETICS_BATCH45_VLM_QUERY_DONE", split, scene["name"],
                      query["id"], fraction, label, fmt, seconds, flush=True)

    payload = {
        "dataset": "TAP-Vid-Kinetics fresh natural-video holdout",
        "split": split,
        "status": "PREDICTED",
        "source_commit": ev.gitsha(),
        "manifest_sha256": manifest_sha,
        "public_query_sha256": public_sha,
        "config_sha256": config_sha,
        "dev_gate_sha256": gate_sha,
        "model_id": cfg["vlm"]["model_id"],
        "model_weights_sha256": model_sha,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "target_fractions": fractions,
        "label_space": "VISIBLE / NOT_VISIBLE / UNCERTAIN",
        "input": "marked first-visible query frame + reference crop + fixed later target frame",
        "oracle_never_read": True,
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "records": records,
    }
    once(outfile, payload)
    pin(outfile)
    print("KINETICS_BATCH45_VLM_PREDICTIONS_PINNED", split, len(records),
          sha_bytes(outfile.read_bytes()), flush=True)


def summarize_binary(labels: list[str], truth: list[bool], formats: list[str]) -> dict:
    if not labels:
        return {"count": 0}
    visible = np.asarray(truth, dtype=bool)
    predicted_visible = np.asarray([x == "VISIBLE" for x in labels], dtype=bool)
    uncertain = np.asarray([x == "UNCERTAIN" for x in labels], dtype=bool)
    valid = np.asarray([x == "valid" for x in formats], dtype=bool)
    vis_support = int(visible.sum())
    not_support = int((~visible).sum())
    vis_recall = float((predicted_visible & visible).sum() / vis_support) if vis_support else None
    not_recall = float(((~predicted_visible & ~uncertain) & ~visible).sum() / not_support) if not_support else None
    balanced = float((vis_recall + not_recall) / 2) if vis_recall is not None and not_recall is not None else None
    accepted = ~uncertain & valid
    correct = (predicted_visible == visible) & accepted
    false_accept = int((predicted_visible & ~visible).sum())
    return {
        "count": len(labels),
        "visible_support": vis_support,
        "not_visible_support": not_support,
        "correct": int(correct.sum()),
        "accuracy_over_all": float(correct.sum() / len(labels)),
        "balanced_accuracy": balanced,
        "visible_recall": vis_recall,
        "not_visible_recall": not_recall,
        "coverage": float(accepted.mean()),
        "invalid_count": int((~valid).sum()),
        "invalid_rate": float((~valid).mean()),
        "uncertain_count": int(uncertain.sum()),
        "not_visible_false_accept": false_accept,
        "not_visible_false_accept_rate": float(false_accept / not_support) if not_support else None,
    }


def score(split: str, cfg: dict) -> dict:
    prediction_path = OUT / f"{split}_vlm_predictions.json"
    require_pin(prediction_path)
    saved = doc(prediction_path)
    if saved.get("status") == "SKIPPED_DEV_GATE_BLOCKED":
        result = {
            "dataset": "TAP-Vid-Kinetics fresh holdout",
            "split": split,
            "status": "SKIPPED_DEV_GATE_BLOCKED",
            "source_commit": ev.gitsha(),
            "config_sha256": sha_bytes(json_bytes(cfg)),
            "prediction_sha256": sha_bytes(prediction_path.read_bytes()),
            "test_oracle_read": False,
            "reason": "DEV VLM gate did not pass; TEST labels were not opened.",
        }
        once(OUT / f"{split}_vlm_results.json", result)
        print("KINETICS_BATCH45_VLM_SCORE_SKIPPED", split, flush=True)
        return result

    cpu_path = OUT / f"{split}_predictions.json"
    require_pin(cpu_path)
    cpu = doc(cpu_path)
    expected_manifest = sha_bytes((DATA / "manifest.json").read_bytes())
    expected_public = sha_bytes((DATA / "queries-public.json").read_bytes())
    gate_sha = None
    if split == "test":
        require_pin(dev_gate_path())
        gate_sha = sha_bytes(dev_gate_path().read_bytes())
    if (saved.get("source_commit") != ev.gitsha()
            or saved.get("config_sha256") != sha_bytes(json_bytes(cfg))
            or (split == "test" and saved.get("dev_gate_sha256") != gate_sha)
            or cpu.get("source_commit") != ev.gitsha()
            or saved.get("manifest_sha256") != expected_manifest
            or saved.get("public_query_sha256") != expected_public
            or saved.get("oracle_never_read") is not True
            or saved.get("prompt_sha256") != hashlib.sha256(PROMPT.encode()).hexdigest()
            or saved.get("split") != split):
        raise RuntimeError("KINETICS_BATCH45_VLM_SCORE_FIREWALL_FAILED")

    # Both VLM and CPU predictions are pinned before opening the private oracle.
    manifest = src.audit_full()
    queries = doc(DATA / "queries-public.json")
    oracle = doc(DATA / "oracle-private.json")
    if manifest.get("oracle_sha256") != sha_bytes((DATA / "oracle-private.json").read_bytes()):
        raise RuntimeError("KINETICS_BATCH45_VLM_ORACLE_SHA_DRIFT")

    cpu_by_scene = {r["name"]: r for r in cpu["records"]}
    selected_method = None
    selection_path = OUT / "dev_selection.json"
    require_pin(selection_path)
    selection = doc(selection_path)
    if selection.get("source_commit") != ev.gitsha():
        raise RuntimeError("KINETICS_BATCH45_VLM_TRACKER_SELECTION_SOURCE")
    selected_method = selection["selected_method"]

    labels, truth, formats = [], [], []
    cpu_labels, cpu_truth = [], []
    seen = set()
    for row in saved["records"]:
        key = (row["scene"], row["query_id"], row["target_time"])
        if key in seen:
            raise RuntimeError("KINETICS_BATCH45_VLM_DUPLICATE_QUERY")
        seen.add(key)
        query = next(q for q in queries[row["scene"]] if q["id"] == row["query_id"])
        gold = oracle[row["scene"]]["tracks"][row["query_id"]]
        target_time = int(row["target_time"])
        if not (int(query["query_time"]) < target_time < len(gold["occluded"])):
            raise RuntimeError("KINETICS_BATCH45_VLM_TARGET_TIME_INVALID")
        actual_visible = not bool(gold["occluded"][target_time])
        labels.append(row["label"])
        truth.append(actual_visible)
        formats.append(row["format"])
        point = cpu_by_scene[row["scene"]]["tracks"][row["query_id"]][selected_method][target_time]
        cpu_labels.append("VISIBLE" if point is not None else "NOT_VISIBLE")
        cpu_truth.append(actual_visible)

    vlm_metrics = summarize_binary(labels, truth, formats)
    cpu_metrics = summarize_binary(cpu_labels, cpu_truth, ["valid"] * len(cpu_labels))
    result = {
        "dataset": "TAP-Vid-Kinetics fresh natural-video holdout",
        "split": split,
        "status": "SCORED",
        "source_commit": ev.gitsha(),
        "config_sha256": sha_bytes(json_bytes(cfg)),
        "prediction_sha256": sha_bytes(prediction_path.read_bytes()),
        "cpu_prediction_sha256": sha_bytes(cpu_path.read_bytes()),
        "selected_cpu_method": selected_method,
        "model_weights_sha256": saved["model_weights_sha256"],
        "prompt_sha256": saved["prompt_sha256"],
        "target_fractions": saved["target_fractions"],
        "vlm": vlm_metrics,
        "selected_cpu_visibility_at_same_target_frames": cpu_metrics,
        "raw_replies": [
            {"scene": r["scene"], "query_id": r["query_id"],
             "target_fraction": r["target_fraction"], "raw_reply": r["raw_reply"],
             "label": r["label"], "format": r["format"]}
            for r in saved["records"]
        ],
        "limitations": [
            "VLM sees a marked query frame, a local reference crop and one later frame; CPU trackers see every intermediate 25-fps frame.",
            "Visibility classification is not point-coordinate accuracy or persistent identity proof.",
            "DEV gate controls whether TEST VLM inference is allowed; a blocked gate means no TEST VLM labels are opened.",
        ],
    }
    once(OUT / f"{split}_vlm_results.json", result)
    print("KINETICS_BATCH45_VLM_SCORED", split,
          json.dumps({"vlm": vlm_metrics, "selected_cpu": cpu_metrics}, sort_keys=True), flush=True)
    return result


def make_dev_gate(cfg: dict) -> dict:
    result_path = OUT / "dev_vlm_results.json"
    if not result_path.is_file():
        raise RuntimeError("KINETICS_BATCH45_VLM_DEV_SCORE_REQUIRED")
    result = doc(result_path)
    if (result.get("split") != "dev" or result.get("source_commit") != ev.gitsha()
            or result.get("config_sha256") != sha_bytes(json_bytes(cfg))):
        raise RuntimeError("KINETICS_BATCH45_VLM_DEV_GATE_SOURCE_MISMATCH")
    metrics = result["vlm"]
    rules = cfg["vlm"]["dev_gate"]
    support_ok = (
        metrics["visible_support"] >= int(rules["min_visible_support"])
        and metrics["not_visible_support"] >= int(rules["min_not_visible_support"])
    )
    checks = {
        "class_support": support_ok,
        "invalid_rate": metrics["invalid_rate"] <= float(rules["max_invalid_rate"]),
        "coverage": metrics["coverage"] >= float(rules["min_coverage"]),
        "balanced_accuracy": metrics["balanced_accuracy"] is not None
            and metrics["balanced_accuracy"] >= float(rules["min_balanced_accuracy"]),
        "not_visible_false_accept_rate": metrics["not_visible_false_accept_rate"] is not None
            and metrics["not_visible_false_accept_rate"] <= float(rules["max_not_visible_false_accept_rate"]),
    }
    status = "PASS" if all(checks.values()) else "BLOCK"
    payload = {
        "source_commit": ev.gitsha(),
        "split": "dev",
        "config_sha256": sha_bytes(json_bytes(cfg)),
        "dev_vlm_results_sha256": sha_bytes(result_path.read_bytes()),
        "status": status,
        "checks": checks,
        "metrics": metrics,
        "rules": rules,
        "test_policy": "TEST VLM inference is skipped unless every DEV check passes",
    }
    path = dev_gate_path()
    once(path, payload)
    pin(path)
    print("KINETICS_BATCH45_VLM_DEV_GATE", status, json.dumps(checks, sort_keys=True), flush=True)
    return payload


def final_audit(cfg: dict) -> None:
    for split in ("dev", "test"):
        pred = OUT / f"{split}_vlm_predictions.json"
        result = OUT / f"{split}_vlm_results.json"
        require_pin(pred)
        saved = doc(pred)
        if (saved.get("source_commit") != ev.gitsha()
                or saved.get("config_sha256") != sha_bytes(json_bytes(cfg))
                or saved.get("oracle_never_read") is not True):
            raise RuntimeError("KINETICS_BATCH45_VLM_FINAL_PREDICTION_AUDIT")
        if not result.is_file():
            raise RuntimeError("KINETICS_BATCH45_VLM_FINAL_RESULT_MISSING")
        result_doc = doc(result)
        if (result_doc.get("source_commit") != ev.gitsha()
                or result_doc.get("config_sha256") != sha_bytes(json_bytes(cfg))):
            raise RuntimeError("KINETICS_BATCH45_VLM_FINAL_RESULT_SOURCE_MISMATCH")
    require_pin(dev_gate_path())
    gate = doc(dev_gate_path())
    test_saved = doc(OUT / "test_vlm_predictions.json")
    if test_saved.get("dev_gate_sha256") != sha_bytes(dev_gate_path().read_bytes()):
        raise RuntimeError("KINETICS_BATCH45_VLM_TEST_GATE_SHA_MISMATCH")
    if gate["status"] != "PASS" and test_saved.get("status") != "SKIPPED_DEV_GATE_BLOCKED":
        raise RuntimeError("KINETICS_BATCH45_VLM_TEST_RAN_AFTER_BLOCKED_DEV_GATE")
    print("KINETICS_BATCH45_VLM_FINAL_AUDIT_PASS", gate["status"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=(
        "predict-dev", "score-dev", "dev-gate",
        "predict-test", "score-test", "final-audit",
    ))
    parser.add_argument("--config", default="configs/kinetics_batch45.json")
    args = parser.parse_args()
    cfg = ev.load_config(args.config)
    if args.action.startswith("predict-"):
        predict(args.action.split("-")[1], cfg)
    elif args.action.startswith("score-"):
        score(args.action.split("-")[1], cfg)
    elif args.action == "dev-gate":
        make_dev_gate(cfg)
    else:
        final_audit(cfg)


if __name__ == "__main__":
    main()
