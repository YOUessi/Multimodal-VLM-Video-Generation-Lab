"""Independent, frozen synthetic affine-camera and ambiguous-target motion benchmark.

World-relative labels derive from scene construction, never from pixel tracks.
Four balanced nuisance conditions: translation, rotation, zoom, same-color
distractor with partial occlusion and moving camera. Forward/reverse clips
share a scene-pair ID. This is 2D rendered video, NOT real camera footage.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
import cv2
import numpy as np

import domain_shift_motion_data as original

ROOT = Path(__file__).resolve().parents[1]
MODES = ("camera_pan", "camera_rotation", "camera_zoom", "color_ambiguity")
DIRECTIONS = original.DIRECTIONS
OPPOSITE = original.OPPOSITE
LABELS = original.LABELS


def affine_for_frame(props, t, height=256, width=256):
    """World frame to screen, never used by an inference-side estimator."""
    scale = 1.0 + t * props["zoom_per_frame"]
    degrees = t * props["rotation_per_frame"]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), degrees, scale)
    matrix[0, 2] += t * props["pan_velocity"][0]
    matrix[1, 2] += t * props["pan_velocity"][1]
    return matrix


def scene(seed, mode, direction):
    rng = random.Random(seed ^ 0xA6FF1E3)
    base_mode = "occlusion" if mode == "color_ambiguity" else "clean"
    p = original.scene(seed, base_mode, direction)
    p["camera_velocity"] = [0, 0]  # world renderer has no implicit camera
    p["study_mode"] = mode
    p["pan_velocity"] = [0, 0]
    p["rotation_per_frame"] = 0.0
    p["zoom_per_frame"] = 0.0
    if mode == "camera_pan":
        p["pan_velocity"] = [rng.choice((-3, -2, 2, 3)), rng.choice((-3, -2, 2, 3))]
    elif mode == "camera_rotation":
        p["rotation_per_frame"] = rng.choice((-2.4, -1.7, 1.7, 2.4))
        p["pan_velocity"] = [rng.choice((-1, 1)), rng.choice((-1, 1))]
    elif mode == "camera_zoom":
        p["zoom_per_frame"] = rng.choice((-0.022, -0.016, 0.016, 0.022))
        p["pan_velocity"] = [rng.choice((-1, 1)), rng.choice((-1, 1))]
    elif mode == "color_ambiguity":
        p["distractor_color"] = p["target_color"]
        p["rotation_per_frame"] = rng.choice((-1.5, 1.5))
        p["pan_velocity"] = [rng.choice((-2, 2)), rng.choice((-2, 2))]
        # Contrast between target and distractor shapes; query must identify
        # by both color and shape; HSV-only selection is intentionally weak.
        p["distractor_shape"] = next(s for s in original.SHAPES if s != p["target_shape"])
    else:
        raise ValueError("Unknown nuisance family")
    return p


def render(props, count, height, width):
    base = original.render(props, count, height, width)
    return [
        cv2.warpAffine(frame, affine_for_frame(props, i, height, width),
                       (width, height), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)
        for i, frame in enumerate(base)
    ]


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def build(cfg):
    if cfg["dataset_dir"] == cfg["output_dir"] or not cfg["dataset_dir"].startswith("data/"):
        raise RuntimeError("SOURCE_MUST_LIVE_IN_PRIVATE_DATA_ROOT")
    root = ROOT / cfg["dataset_dir"]
    frozen = root / "manifest.json"
    if frozen.exists():
        prior = json.loads(frozen.read_text())
        if prior["protocol_version"] != cfg["version"] or prior["config_sha256"] != config_hash(cfg):
            raise RuntimeError("AFFINE_DATASET_CONFIG_DRIFT")
        for row in prior["rows"]:
            p = ROOT / row["video"]
            if not p.is_file() or original.digest(p) != row["content_sha256"]:
                raise RuntimeError("AFFINE_VIDEO_BYTE_DRIFT")
        print("AFFINE_FROZEN_DATA_REUSED", prior["source_sha256"], flush=True)
        return prior
    rows, specs = [], []
    grid = [(mode, direction) for mode in MODES for direction in DIRECTIONS]
    for split, pairs, seed in (
        ("dev", cfg["dev_pairs"], cfg["dev_seed"]),
        ("test", cfg["test_pairs"], cfg["test_seed"]),
    ):
        if pairs % len(grid):
            raise ValueError("Split pair count must balance all mode x direction combinations")
        combinations = grid * (pairs // len(grid))
        rng = random.Random(seed)
        rng.shuffle(combinations)
        for i, (mode, direction) in enumerate(combinations):
            props = scene(rng.randrange(2**31), mode, direction)
            images = render(props, cfg["frames"], cfg["height"], cfg["width"])
            choices = list(DIRECTIONS)
            random.Random(props["seed"] ^ 0x8173F).shuffle(choices)
            pair_id = f"{split}_{i:04d}"
            specs.append({"pair_id": pair_id, "split": split, "variant": mode,
                          "scene_seed": props["seed"], "camera_params": {
                              "pan_velocity": props["pan_velocity"],
                              "rotation_per_frame": props["rotation_per_frame"],
                              "zoom_per_frame": props["zoom_per_frame"]}})
            for reverse in (False, True):
                suffix = "reverse" if reverse else "forward"
                path = root / "clips" / split / f"{pair_id}_{suffix}.mp4"
                if path.exists():
                    raise RuntimeError("ORPHAN_CLIP_REQUIRES_NEW_FROZEN_BATCH")
                original.write_video(path, images[::-1] if reverse else images, cfg["fps"])
                gold = OPPOSITE[direction] if reverse else direction
                rows.append({
                    "id": f"{pair_id}_{suffix}", "pair_id": pair_id, "split": split,
                    "variant": mode, "reversed": reverse,
                    "video": str(path.relative_to(ROOT)),
                    "content_sha256": original.digest(path),
                    "query": (f"Relative to the stationary textured world background, "
                              f"which direction does the {props['target_color']} "
                              f"{props['target_shape']} move?"),
                    "choices": [LABELS[d] for d in choices],
                    "gold_direction": gold,
                    "gold_letter": "ABCDE"[choices.index(gold)],
                    "frames": cfg["frames"],
                    "source_of_gold": "world target velocity, never screen displacement",
                })
        print("AFFINE_DATA_GENERATED", split, pairs, "SCENE_PAIRS", flush=True)
    dev_hashes = {r["content_sha256"] for r in rows if r["split"] == "dev"}
    test_hashes = {r["content_sha256"] for r in rows if r["split"] == "test"}
    if dev_hashes & test_hashes:
        raise RuntimeError("AFFINE_CROSS_SPLIT_BYTE_LEAKAGE")
    doc = {
        "protocol_version": cfg["version"],
        "config_sha256": config_hash(cfg),
        "source_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "synthetic_only": True, "not_natural_video": True,
        "dev_pairs": cfg["dev_pairs"], "test_pairs": cfg["test_pairs"],
        "dev_test_content_overlap": 0, "total_video_clips": len(rows),
        "modes": list(MODES), "directions": list(DIRECTIONS),
        "rows": rows, "source_pairs": specs,
    }
    original.save(frozen, doc)
    print("AFFINE_DATA_FROZEN", doc["source_sha256"], "CLIPS", len(rows), flush=True)
    return doc


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--config", default="configs/affine_camera_motion_v1.json")
    a = arg.parse_args()
    build(json.loads((ROOT / a.config).read_text()))
