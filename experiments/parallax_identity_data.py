"""New 2.5D synthetic parallax/target-identity benchmark; not natural footage.

The labels are world-relative directions. Camera depth layers and object
world-velocities exist ONLY in the private generator/manifest, never in model
or tracker inputs. Paired reverse clips are one independent source scene.
This independent study must NOT be tuned on the scored Batch 32/33 tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np

import domain_shift_motion_data as shared

ROOT = Path(__file__).resolve().parents[1]
MODES = ("parallax_dual_depth", "color_shape_distractor",
         "same_shape_ambiguity", "shadow_low_texture")
DIRECTIONS = shared.DIRECTIONS
OPPOSITE = shared.OPPOSITE
LABELS = shared.LABELS


def configuration_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def world_scene(seed, mode, direction):
    rng = random.Random(seed ^ 0xB34D2026)
    color = rng.choice(tuple(shared.COLOR_RGB))
    shape = rng.choice(shared.SHAPES)
    if mode in ("color_shape_distractor", "same_shape_ambiguity"):
        other_color = color
    else:
        other_color = rng.choice([c for c in shared.COLOR_RGB if c != color])
    if mode == "same_shape_ambiguity":
        other_shape = shape
    elif mode == "color_shape_distractor":
        other_shape = rng.choice([s for s in shared.SHAPES if s != shape])
    else:
        other_shape = rng.choice(shared.SHAPES)
    return {
        "seed": seed, "variant": mode, "direction": direction,
        "target_color": color, "target_shape": shape,
        "distractor_color": other_color, "distractor_shape": other_shape,
        "world_target_velocity": list(shared.DXDY[direction]),
        "target_speed": rng.choice([5, 6, 7]),
        "target_xy": [rng.randint(107, 145), rng.randint(104, 146)],
        "distractor_xy": [rng.choice([rng.randint(35, 65), rng.randint(188, 223)]),
                          rng.choice([rng.randint(35, 65), rng.randint(190, 224)])],
        "distractor_velocity": list(shared.DXDY[rng.choice(DIRECTIONS)]),
        "distractor_speed": rng.choice([1, 2, 3]),
        "camera_velocity": [rng.choice([-3.9, -3.0, 3.0, 3.9]),
                            rng.choice([-3.2, -2.7, 2.7, 3.2])],
        "world_texture_seed": rng.randrange(2**31),
        "foreground_texture_seed": rng.randrange(2**31),
        "near_depth_factor": rng.choice([1.65, 1.85, 2.0]),
        "far_depth_factor": rng.choice([0.48, 0.58, 0.68]),
        "object_depth_factor": rng.choice([1.0, 1.1]),
    }


def textured_plane(height, width, seed, low_texture=False):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 9 if low_texture else 23, (height, width))
    gray = np.clip((180 if low_texture else 165) + noise, 60, 239).astype(np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if not low_texture:
        rand = random.Random(seed)
        for _ in range(105):
            x, y = rand.randrange(width), rand.randrange(height)
            z = rand.choice([1, 2, 3])
            cv2.line(bgr, (x-z, y-z), (x+z, y+z),
                     (rand.choice([87, 102, 213]),)*3, 1)
        for x in range(9, width, 27):
            cv2.line(bgr, (x, 0), (x, height-1), (127,)*3, 1)
    return bgr


def translate(image, dx, dy):
    height, width = image.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (width, height),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)


def frames_for_scene(scene, count, height, width):
    near_tex = textured_plane(height, width, scene["foreground_texture_seed"])
    far_tex = textured_plane(height, width, scene["world_texture_seed"],
                             scene["variant"] == "shadow_low_texture")
    near_mask = np.zeros((height, width), dtype=np.uint8)
    # Two separate depth planes with hard boundaries, creating genuinely
    # nonuniform parallax. The target is on a third depth layer.
    cv2.rectangle(near_mask, (11, 20), (79, height-18), 255, -1)
    cv2.rectangle(near_mask, (width-75, 12), (width-11, height-26), 255, -1)
    d = scene["world_target_velocity"]
    o = scene["distractor_velocity"]
    cx, cy = scene["camera_velocity"]
    output = []
    for t in range(count):
        far = translate(far_tex, -cx*t*scene["far_depth_factor"],
                        -cy*t*scene["far_depth_factor"])
        near = translate(near_tex, -cx*t*scene["near_depth_factor"],
                         -cy*t*scene["near_depth_factor"])
        mask = translate(near_mask, -cx*t*scene["near_depth_factor"],
                         -cy*t*scene["near_depth_factor"])
        frame = np.where(mask[:, :, None] >= 128, near, far).copy()
        x = scene["target_xy"][0] + d[0]*scene["target_speed"]*t - cx*t*scene["object_depth_factor"]
        y = scene["target_xy"][1] + d[1]*scene["target_speed"]*t - cy*t*scene["object_depth_factor"]
        u = scene["distractor_xy"][0] + o[0]*scene["distractor_speed"]*t - cx*t*scene["object_depth_factor"]
        v = scene["distractor_xy"][1] + o[1]*scene["distractor_speed"]*t - cy*t*scene["object_depth_factor"]
        shared.draw_object(frame, u, v, scene["distractor_color"],
                           scene["distractor_shape"], 12)
        shared.draw_object(frame, x, y, scene["target_color"],
                           scene["target_shape"], 13)
        if scene["variant"] in ("color_shape_distractor", "same_shape_ambiguity") and 3 <= t <= 6:
            # Occasional partial target occlusion by a moving foreground.
            ox = int(round(x))
            oy = int(round(y))
            cv2.rectangle(frame, (ox-7, oy-19), (ox+8, oy+6),
                          (134, 131, 126), -1)
        if scene["variant"] == "shadow_low_texture":
            yy, xx = np.indices((height, width))
            shadow = (xx > (45 + t*14)) & (xx < (139 + t*14))
            factor = 0.56 if t <= count // 2 else 0.74
            frame = np.clip(frame.astype(np.float32) *
                            np.where(shadow[..., None], factor, 0.96),
                            0, 255).astype(np.uint8)
        output.append(frame)
    return output


def build(cfg):
    if not cfg["dataset_dir"].startswith("data/"):
        raise RuntimeError("PRIVATE_DATA_DIRECTORY_REQUIRED")
    source = ROOT / cfg["dataset_dir"]
    manifest = source / "manifest.json"
    cfg_hash = configuration_hash(cfg)
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old["config_sha256"] != cfg_hash:
            raise RuntimeError("FROZEN_2P5D_CONFIG_DRIFT")
        for row in old["rows"]:
            if shared.digest(ROOT / row["video"]) != row["content_sha256"]:
                raise RuntimeError("FROZEN_2P5D_VIDEO_DRIFT")
        print("PARALLAX_DATA_ALREADY_FROZEN", old["source_sha256"], flush=True)
        return old
    rows, pairs = [], []
    conditions = [(mode, direction) for mode in MODES for direction in DIRECTIONS]
    for split, count, seed in (
        ("dev", cfg["dev_pairs"], cfg["dev_seed"]),
        ("test", cfg["test_pairs"], cfg["test_seed"]),
    ):
        if count % len(conditions):
            raise ValueError("Source-pair count must balance all nuisance x direction cells")
        grid = conditions * (count // len(conditions))
        rng = random.Random(seed)
        rng.shuffle(grid)
        for i, (mode, direction) in enumerate(grid):
            p = world_scene(rng.randrange(2**31), mode, direction)
            images = frames_for_scene(p, cfg["frames"], cfg["height"], cfg["width"])
            pair_id = f"{split}_{i:04d}"
            choices = list(DIRECTIONS)
            random.Random(p["seed"] ^ 0xB341CD).shuffle(choices)
            pairs.append({"id": pair_id, "split": split, "mode": mode, "seed": p["seed"],
                          "camera_depths": [p["far_depth_factor"],
                                            p["object_depth_factor"],
                                            p["near_depth_factor"]]})
            for reverse in (False, True):
                name = f"{pair_id}_{'reverse' if reverse else 'forward'}"
                file = source / "clips" / split / (name + ".mp4")
                if file.exists():
                    raise RuntimeError("ORPHAN_VIDEO_WITHOUT_FROZEN_MANIFEST")
                shared.write_video(file, images[::-1] if reverse else images, cfg["fps"])
                gold = OPPOSITE[direction] if reverse else direction
                rows.append({
                    "id": name, "pair_id": pair_id, "split": split,
                    "variant": mode, "reversed": reverse,
                    "video": str(file.relative_to(ROOT)),
                    "content_sha256": shared.digest(file),
                    "query": f"Relative to the stationary world (not apparent screen motion), "
                             f"which direction does the {p['target_color']} {p['target_shape']} move?",
                    "choices": [LABELS[c] for c in choices],
                    "gold_direction": gold,
                    "gold_letter": "ABCDE"[choices.index(gold)],
                    "frames": cfg["frames"],
                    "source_of_gold": "target world velocity, not image-plane motion",
                })
        print("PARALLAX_GENERATED", split, count, "scene pairs", flush=True)
    dev = {r["content_sha256"] for r in rows if r["split"] == "dev"}
    test = {r["content_sha256"] for r in rows if r["split"] == "test"}
    if dev & test or len(set(r["id"] for r in rows)) != len(rows):
        raise RuntimeError("PARALLAX_SPLIT_OR_ID_LEAK")
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    document = {
        "protocol_version": cfg["version"],
        "config_sha256": cfg_hash,
        "source_sha256": digest,
        "synthetic_only": True,
        "not_natural_video": True,
        "scene_depth_layers": 3,
        "pairs": pairs,
        "rows": rows,
        "dev_pairs": cfg["dev_pairs"], "test_pairs": cfg["test_pairs"],
        "total_video_clips": len(rows), "dev_test_content_overlap": 0,
        "modes": list(MODES), "directions": list(DIRECTIONS),
    }
    shared.save(manifest, document)
    print("PARALLAX_FROZEN", digest, len(rows), flush=True)
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/parallax_identity_v1.json")
    args = parser.parse_args()
    build(json.loads((ROOT / args.config).read_text()))
