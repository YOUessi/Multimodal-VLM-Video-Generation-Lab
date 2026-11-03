"""Batch69: newly generated, sealed procedural video stress test of frozen trackers.

These are synthetic pixel-plane scenes, NOT new natural videos and NOT calibrated
3D camera ground truth. The already running Batch59 natural-video protocol is
unmodified. No tracker threshold is fit to this dataset, including its DEV.
Run: prepare -> predict-dev -> score-dev -> predict-test -> score-test -> audit.
Private oracles are never opened in predict paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kinetics_batch42_eval as core

WIDTH = 256
FRAMES = 40
REGIMES = (
    "clean", "camera_pan", "illumination", "crossing_occluder",
    "shared_motion_occluder", "pan_shared_motion_occluder",
)
DEV_PER_REGIME = 4
TEST_PER_REGIME = 3
METHODS = ("appearance_fb_055", "klt_fb", "template_recover4",
           "affine_stabilized_klt", "template_consensus_4px")
BASE = "appearance_fb_055"
# No model selection or new parameter search is permitted in either split.
PCK_RADIUS = 8.0
PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT.parent / "data" / "batch69_prospective_occlusion_v1"
OUT = PROJECT / "outputs" / "batch69_prospective_occlusion_v1"
CONFIG = PROJECT / "configs" / "kinetics_batch56.json"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj: object) -> bytes:
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2)
            + "\n").encode("utf-8")


def write_once(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as fp:
        fp.write(data)
    return digest(data)


def save_json(path: Path, obj: object) -> str:
    result = write_once(path, canonical(obj))
    write_once(path.with_suffix(path.suffix + ".sha256"), (result + "\n").encode())
    return result


def read_pinned(path: Path) -> dict:
    tag = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not tag.is_file():
        raise RuntimeError("BATCH69_MISSING_PIN:" + path.name)
    data = path.read_bytes()
    if digest(data) != tag.read_text().strip():
        raise RuntimeError("BATCH69_HASH_DRIFT:" + path.name)
    return json.loads(data)


def source_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True
    ).strip()


def assert_clean_source(expected: str) -> None:
    if source_sha() != expected:
        raise RuntimeError("BATCH69_FROZEN_SOURCE_SHA_CHANGED")
    dirty = subprocess.check_output(
        ["git", "-C", str(PROJECT), "status", "--porcelain"],
        text=True
    )
    if dirty.strip():
        raise RuntimeError("BATCH69_FROZEN_SOURCE_DIRTY")


def specifications() -> list[dict]:
    records = []
    for split, count, seed_base in (
        ("dev", DEV_PER_REGIME, 690_100),
        ("test", TEST_PER_REGIME, 699_000),
    ):
        for group_idx, regime in enumerate(REGIMES):
            for j in range(count):
                records.append({
                    "id": f"{split}_{group_idx:02d}_{j:02d}",
                    "split": split, "regime": regime,
                    "seed": seed_base + group_idx * 101 + j * 7,
                    "nframes": FRAMES,
                    "query_time": 0,
                })
    return records


def geometry(spec: dict, t: int) -> tuple[tuple[float, float], tuple[float, float]]:
    """Known 2D image-plane trajectory: screen=world+global background pan."""
    seed = spec["seed"]
    world = (70 + seed % 13 + 1.5*t, 96 + seed % 11 + 0.55*t)
    pan = spec["regime"] in ("camera_pan", "pan_shared_motion_occluder")
    camera = ((0.7*t, -0.4*t) if pan else (0.0, 0.0))
    return world, camera


def occluder_rect(spec: dict, t: int, center: tuple[int, int]
                  ) -> tuple[int, int, int, int] | None:
    regime = spec["regime"]
    if regime == "crossing_occluder":
        # Stationary image-plane occluder: target crosses it naturally.
        return (110, 103, 155, 150)
    if regime in ("shared_motion_occluder", "pan_shared_motion_occluder"):
        if 10 <= t <= 32:
            x, y = center
            # Deliberately correlated adversarial occluder motion.
            return (x - 22, y - 21, x + 24, y + 25)
    return None


def render(spec: dict) -> tuple[np.ndarray, dict]:
    """Only called during prepare. Returned labels are written into private/.

    Predict reads public .npy frames and the public query only, not this function.
    """
    rng = np.random.default_rng(spec["seed"])
    bg = rng.integers(0, 255, (WIDTH, WIDTH), dtype=np.uint8)
    bg = cv2.GaussianBlur(bg, (5, 5), 0.8)
    # Distinct, fixed background structure + small textured target.
    for _ in range(17):
        x, y = rng.integers(0, WIDTH, size=2)
        cv2.circle(bg, (int(x), int(y)), int(rng.integers(2, 11)),
                   int(rng.integers(25, 225)), 1)
    target = rng.integers(20, 235, (19, 19), dtype=np.uint8)
    blocker = rng.integers(70, 215, (64, 64), dtype=np.uint8)
    blocker = cv2.GaussianBlur(blocker, (3, 3), 0.7)
    frames = np.empty((FRAMES, WIDTH, WIDTH), dtype=np.uint8)
    xy = []
    occluded = []
    world_xy = []
    camera_xy = []
    for t in range(FRAMES):
        world, camera = geometry(spec, t)
        cx = int(round(world[0]+camera[0]))
        cy = int(round(world[1]+camera[1]))
        shift = np.float32([[1, 0, camera[0]], [0, 1, camera[1]]])
        canvas = cv2.warpAffine(bg, shift, (WIDTH, WIDTH),
                                borderMode=cv2.BORDER_REFLECT_101)
        canvas[cy-9:cy+10, cx-9:cx+10] = target
        bounds = occluder_rect(spec, t, (cx, cy))
        blocked = False
        if bounds is not None:
            x0, y0, x1, y1 = bounds
            # Withdrawing the target point when the occluding layer covers it
            # corresponds to point visibility, not full-object visibility.
            blocked = x0 <= cx < x1 and y0 <= cy < y1
            h, w = y1-y0, x1-x0
            canvas[y0:y1, x0:x1] = blocker[:h, :w]
        if spec["regime"] == "illumination":
            gain = 0.70 + 0.25 * math.sin(t/6.0)
            canvas = np.clip(canvas.astype(np.float32)*gain + 20, 0, 255
                             ).astype(np.uint8)
        frames[t] = canvas
        xy.append([float(cx), float(cy)])
        occluded.append(bool(blocked))
        world_xy.append([float(v) for v in world])
        camera_xy.append([float(v) for v in camera])
    oracle = {"xy": xy, "occluded": occluded, "world_xy": world_xy,
              "background_translation_xy": camera_xy}
    return frames, oracle


def assert_manifest(man: dict) -> None:
    if (man.get("kind") != "BATCH69_SYNTHETIC_PRE_REGISTERED_V1"
            or man.get("protocol") != "fixed-untuned-five-methods-v1"
            or man.get("scenes") is None
            or len(man["scenes"]) != len(specifications())
            or man.get("source_sha") is None):
        raise RuntimeError("BATCH69_MANIFEST_PROTOCOL_INVALID")
    expected = specifications()
    for a, b in zip(man["scenes"], expected):
        if any(a.get(key) != b[key] for key in b):
            raise RuntimeError("BATCH69_SPLIT_OR_SEED_DRIFT")
        if not isinstance(a.get("frames_sha256"), str):
            raise RuntimeError("BATCH69_FRAME_SHA_MISSING")


def prepare(data: Path = DATA) -> None:
    if (data / "manifest.json").exists() or (data / "public").exists():
        raise RuntimeError("BATCH69_DATASET_ALREADY_EXISTS")
    specs = specifications()
    records = []
    for spec in specs:
        frames, truth = render(spec)
        if frames.shape != (FRAMES, WIDTH, WIDTH):
            raise RuntimeError("BATCH69_BAD_GENERATED_SHAPE")
        public_path = data / "public" / (spec["id"] + ".npy")
        public_path.parent.mkdir(parents=True, exist_ok=True)
        with public_path.open("xb") as fp:
            np.save(fp, frames, allow_pickle=False)
        raw_sha = digest(public_path.read_bytes())
        save_json(data / "private" / (spec["id"] + ".json"), truth)
        rec = dict(spec)
        rec["frames_sha256"] = raw_sha
        rec["query_xy"] = truth["xy"][0]
        records.append(rec)
    manifest = {
        "kind": "BATCH69_SYNTHETIC_PRE_REGISTERED_V1",
        "protocol": "fixed-untuned-five-methods-v1",
        "note": "Synthetic 2D world/background pan only, never 3D camera GT",
        "source_sha": source_sha(),
        "config_sha256": digest(CONFIG.read_bytes()),
        "scenes": records,
    }
    assert_manifest(manifest)
    print("BATCH69_DATA_FROZEN", save_json(data / "manifest.json", manifest),
          "scenes", len(records), flush=True)


def load_public(data: Path, scene: dict) -> list[np.ndarray]:
    path = data / "public" / (scene["id"] + ".npy")
    if digest(path.read_bytes()) != scene["frames_sha256"]:
        raise RuntimeError("BATCH69_PUBLIC_FRAME_HASH_DRIFT:" + scene["id"])
    frames = np.load(path, allow_pickle=False)
    if frames.dtype != np.uint8 or frames.shape != (FRAMES, WIDTH, WIDTH):
        raise RuntimeError("BATCH69_PUBLIC_FRAME_FORMAT_DRIFT")
    return [np.ascontiguousarray(f) for f in frames]


def agree(trace_a: list, trace_b: list, idx: int) -> list | None:
    a, b = trace_a[idx], trace_b[idx]
    if a is None or b is None:
        return None
    return a if math.dist(a, b) <= 4.0 else None


def predict(split: str, data: Path = DATA, output: Path = OUT) -> dict:
    if split not in ("dev", "test"):
        raise RuntimeError("BATCH69_UNKNOWN_SPLIT")
    man = read_pinned(data / "manifest.json")
    assert_manifest(man)
    assert_clean_source(man["source_sha"])
    if digest(CONFIG.read_bytes()) != man["config_sha256"]:
        raise RuntimeError("BATCH69_TRACKER_CONFIG_DRIFT")
    if split == "test":
        read_pinned(output / "dev_results.json")
    if (output / f"{split}_predictions.json").exists():
        raise RuntimeError("BATCH69_PREDICTION_ALREADY_FROZEN")
    cfg = json.loads(CONFIG.read_text())["tracker"]
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    records = []
    for scene in man["scenes"]:
        if scene["split"] != split:
            continue
        gray = load_public(data, scene)
        query = {"query_time": 0, "query_xy": scene["query_xy"]}
        t0 = time.perf_counter()
        matrices, _, _ = core.estimate_affines(gray, cfg)
        traces, timings = core.predict_query(gray, query, matrices, cfg)
        traces["template_consensus_4px"] = [
            agree(traces[BASE], traces["template_recover4"], i)
            for i in range(FRAMES)
        ]
        elapsed_ms = 1000*(time.perf_counter()-t0)
        if (set(METHODS) - set(traces)
                or any(len(traces[k]) != FRAMES for k in METHODS)):
            raise RuntimeError("BATCH69_TRACKER_OUTPUT_DRIFT")
        records.append({
            "id": scene["id"], "regime": scene["regime"],
            "traces": {k: traces[k] for k in METHODS},
            "cpu_wall_ms": round(elapsed_ms, 3),
            "component_cpu_ms": {k: timings.get(k) for k in METHODS},
        })
    saved = {
        "kind": "BATCH69_LABEL_BLIND_PREDICTIONS", "split": split,
        "source_sha": man["source_sha"],
        "manifest_sha256": digest((data / "manifest.json").read_bytes()),
        "config_sha256": man["config_sha256"],
        "private_oracle_read_during_prediction": False,
        "predictions": records,
    }
    print("BATCH69_PREDICTIONS_FROZEN", split,
          save_json(output / f"{split}_predictions.json", saved),
          len(records), flush=True)
    return saved


def summarize(counts: dict) -> dict:
    visible = counts["visible"]
    hidden = counts["hidden"]
    return {
        **counts,
        "visible_coverage": counts["visible_accepted"]/visible if visible else None,
        "pck8_full_visible": counts["correct_visible"]/visible if visible else None,
        "pck8_accepted_precision": counts["correct_visible"]/counts["visible_accepted"]
            if counts["visible_accepted"] else None,
        "occluded_false_accept_rate": counts["hidden_accepted"]/hidden if hidden else None,
    }


def score_scene(traces: dict, oracle: dict) -> dict:
    if (len(oracle["xy"]) != FRAMES or len(oracle["occluded"]) != FRAMES
            or len(oracle["background_translation_xy"]) != FRAMES):
        raise RuntimeError("BATCH69_ORACLE_LENGTH_DRIFT")
    metrics = {}
    for method in METHODS:
        trace = traces.get(method)
        if trace is None or len(trace) != FRAMES:
            raise RuntimeError("BATCH69_TRACE_INVALID:" + method)
        c = {"visible": 0, "hidden": 0, "visible_accepted": 0,
             "correct_visible": 0, "hidden_accepted": 0,
             "max_contiguous_hidden_accept": 0,
             "world_relative_error_sum_px": 0.0,
             "world_relative_support": 0}
        run = 0
        for t in range(1, FRAMES):
            occ = oracle["occluded"][t]
            if type(occ) is not bool:
                raise RuntimeError("BATCH69_VISIBILITY_INVALID")
            point = trace[t]
            if point is not None and (len(point) != 2
                    or not all(math.isfinite(v) for v in point)):
                raise RuntimeError("BATCH69_NONFINITE_TRACK")
            if occ:
                c["hidden"] += 1
                if point is not None:
                    c["hidden_accepted"] += 1
                    run += 1
                else:
                    run = 0
                c["max_contiguous_hidden_accept"] = max(
                    run, c["max_contiguous_hidden_accept"])
            else:
                run = 0
                c["visible"] += 1
                if point is not None:
                    c["visible_accepted"] += 1
                    if math.dist(point, oracle["xy"][t]) <= PCK_RADIUS:
                        c["correct_visible"] += 1
                        # Error after removing *known synthetic* background shift.
                        # This is evaluation-only, NOT privileged tracker input.
                        shift = oracle["background_translation_xy"][t]
                        a = [point[j]-trace[0][j]-shift[j] for j in range(2)]
                        ref = [oracle["world_xy"][t][j]-oracle["world_xy"][0][j]
                               for j in range(2)]
                        c["world_relative_error_sum_px"] += math.dist(a, ref)
                        c["world_relative_support"] += 1
        metrics[method] = summarize(c)
    return metrics


def aggregate(scenes: list[dict]) -> dict:
    summary = {}
    for method in METHODS:
        grouped = {}
        for regime in REGIMES:
            rows = [s["metrics"][method] for s in scenes if s["regime"] == regime]
            if not rows:
                continue
            counts = {k: sum(row[k] for row in rows) for k in (
                "visible", "hidden", "visible_accepted", "correct_visible",
                "hidden_accepted", "world_relative_support")}
            counts["max_contiguous_hidden_accept"] = max(
                row["max_contiguous_hidden_accept"] for row in rows)
            counts["world_relative_error_sum_px"] = sum(
                row["world_relative_error_sum_px"] for row in rows)
            grouped[regime] = summarize(counts)
        total = {k: sum(x[k] for x in grouped.values()) for k in (
            "visible", "hidden", "visible_accepted", "correct_visible",
            "hidden_accepted", "world_relative_support")}
        total["max_contiguous_hidden_accept"] = max(
            x["max_contiguous_hidden_accept"] for x in grouped.values())
        total["world_relative_error_sum_px"] = sum(
            x["world_relative_error_sum_px"] for x in grouped.values())
        summary[method] = {"overall": summarize(total), "by_regime": grouped}
    return summary


def score(split: str, data: Path = DATA, output: Path = OUT) -> dict:
    if split not in ("dev", "test"):
        raise RuntimeError("BATCH69_UNKNOWN_SPLIT")
    man = read_pinned(data / "manifest.json")
    assert_manifest(man)
    assert_clean_source(man["source_sha"])
    if split == "test":
        read_pinned(output / "dev_results.json")
    pred = read_pinned(output / f"{split}_predictions.json")
    if (pred["split"] != split or pred["source_sha"] != man["source_sha"]
            or pred["manifest_sha256"] != digest((data / "manifest.json").read_bytes())
            or pred["config_sha256"] != man["config_sha256"]
            or pred["private_oracle_read_during_prediction"] is not False):
        raise RuntimeError("BATCH69_PREDICTION_FIREWALL_OR_PIN_DRIFT")
    if (output / f"{split}_results.json").exists():
        raise RuntimeError("BATCH69_SCORE_ALREADY_FROZEN")
    specs = [x for x in man["scenes"] if x["split"] == split]
    if len(pred["predictions"]) != len(specs):
        raise RuntimeError("BATCH69_SPLIT_POPULATION_DRIFT")
    results = []
    for rec, sc in zip(pred["predictions"], specs):
        if rec["id"] != sc["id"] or rec["regime"] != sc["regime"]:
            raise RuntimeError("BATCH69_SCENE_ALIGNMENT_DRIFT")
        gt = read_pinned(data / "private" / (sc["id"] + ".json"))
        results.append({
            "id": sc["id"], "regime": sc["regime"],
            "metrics": score_scene(rec["traces"], gt),
            "cpu_wall_ms": rec["cpu_wall_ms"],
        })
    cpu = [r["cpu_wall_ms"] for r in results]
    answer = {
        "kind": "BATCH69_SYNTHETIC_STRESS_RESULTS", "split": split,
        "source_sha": man["source_sha"],
        "prediction_sha256": digest((output / f"{split}_predictions.json").read_bytes()),
        "synthetic_not_natural": True, "thresholds_tuned_on_dataset": False,
        "scenes": results, "metrics": aggregate(results),
        "cpu_wall_ms_median": statistics.median(cpu),
        "cpu_wall_ms_p95": float(np.percentile(cpu, 95)),
    }
    print("BATCH69_SCORE_FROZEN", split,
          save_json(output / f"{split}_results.json", answer), flush=True)
    for key in (BASE, "template_consensus_4px"):
        row = answer["metrics"][key]["overall"]
        print(key, {name: row[name] for name in (
            "visible_coverage", "pck8_full_visible",
            "occluded_false_accept_rate", "max_contiguous_hidden_accept")}, flush=True)
    return answer


def audit(data: Path = DATA, output: Path = OUT) -> None:
    man = read_pinned(data / "manifest.json")
    assert_manifest(man)
    assert_clean_source(man["source_sha"])
    for sc in man["scenes"]:
        if digest((data / "public" / (sc["id"] + ".npy")).read_bytes()) != sc["frames_sha256"]:
            raise RuntimeError("BATCH69_FINAL_FRAME_DRIFT")
        read_pinned(data / "private" / (sc["id"] + ".json"))
    for split in ("dev", "test"):
        pred = read_pinned(output / f"{split}_predictions.json")
        result = read_pinned(output / f"{split}_results.json")
        if (result["prediction_sha256"]
                != digest((output / f"{split}_predictions.json").read_bytes())
                or pred["private_oracle_read_during_prediction"] is not False):
            raise RuntimeError("BATCH69_FINAL_LINKAGE_FAILURE")
    print("BATCH69_FINAL_AUDIT_PASS", man["source_sha"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "predict-dev", "score-dev",
                                         "predict-test", "score-test", "audit"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.data)
    elif args.action.startswith("predict-"):
        predict(args.action[8:], args.data, args.output)
    elif args.action.startswith("score-"):
        score(args.action[6:], args.data, args.output)
    else:
        audit(args.data, args.output)


if __name__ == "__main__":
    main()
