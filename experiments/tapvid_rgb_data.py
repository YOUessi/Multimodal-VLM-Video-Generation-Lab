"""Batch 38: first-query-only TAP-Vid RGB-Stacking acquisition and oracle firewall.

This is a *synthetic robotics* benchmark, deliberately independent of DAVIS.
Future point tracks and occlusion labels stay in a distinct private file.
No algorithm can use them to select templates, reject frames or re-identify.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from zipfile import ZipFile

from tapvid_download import verify, digest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/tapvid_rgb_stacking_v1"
SALT = "BATCH38-INDEPENDENT-RGB-STACKING-20261102-SEALED-47"
DEV = 4
TEST = 8
STRIDE = 2
QUERIES = 4


def atomic_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError("FROZEN_FILE_CONFLICT:" + str(path))
        return
    with path.open("xb") as f:
        f.write(data)


def as_json(doc: dict) -> bytes:
    return (json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()


def official_records(path: Path):
    verify(path)
    with ZipFile(path) as z:
        members = [n for n in z.namelist() if n.endswith(".pkl")]
        if len(members) != 1:
            raise RuntimeError("OFFICIAL_PICKLE_MEMBER_COUNT_MISMATCH")
        # NEVER unpickle arbitrary user files; origin MD5/CRC must be verified.
        with z.open(members[0]) as f:
            payload = pickle.load(f)
    if isinstance(payload, dict):
        return sorted(payload.items())
    if isinstance(payload, (tuple, list)):
        return [(f"video_{i:04d}", v) for i, v in enumerate(payload)]
    raise RuntimeError("UNRECOGNIZED_OFFICIAL_DATASET_STRUCTURE")


def selected_indices(keys):
    keys = list(keys)
    if len(keys) < DEV + TEST:
        raise RuntimeError("NOT_ENOUGH_OFFICIAL_VIDEOS")
    if len(keys) != len(set(keys)):
        raise RuntimeError("DUPLICATE_VIDEO_NAMES")
    def rank(k):
        return hashlib.sha256((SALT + "|VIDEO|" + str(k)).encode()).hexdigest()
    return sorted(keys, key=rank)[:DEV + TEST]


def extract_query(scene_key: str, track_id: int, points: np.ndarray, occluded: np.ndarray) -> dict | None:
    visible = np.flatnonzero(~occluded[track_id])
    if not len(visible):
        return None
    time = int(visible[0])
    xy = [float(v) for v in points[track_id, time]]
    if not np.isfinite(xy).all():
        raise RuntimeError("QUERY_NONFINITE")
    return {"id": f"p{track_id:04d}", "index": track_id, "query_time": time,
            "query_xy": xy, "query_source": "first-annotated-visible-point"}


def content_key(path: Path) -> str:
    return digest(path)


def normalized_to_pixel_points(points: np.ndarray, occluded: np.ndarray, height: int, width: int) -> np.ndarray:
    """Official TAP-Vid stores (x,y) in 0..1; OpenCV and PCK expect raster pixels.

    Follow the upstream evaluation_datasets.py convention: multiply by [W,H],
    not by [W-1,H-1]. Occluded-point coordinates are not scored for geometry.
    Never infer coordinate units from (private) TEST scores.
    """
    if not (0 < height and 0 < width and points.ndim == 3 and points.shape[-1] == 2
            and occluded.shape == points.shape[:2]):
        raise ValueError("INVALID_TAPVID_COORDINATE_SHAPE")
    visible = np.asarray(points, dtype=np.float64)[~occluded]
    if (not np.isfinite(visible).all()
            or np.any(visible < 0) or np.any(visible > 1)):
        raise RuntimeError("TAPVID_VISIBLE_POINTS_NOT_NORMALIZED")
    return np.asarray(points, dtype=np.float64) * np.array([width, height], dtype=np.float64)


def prepare() -> None:
    official = DATA / "official.zip"
    records = dict(official_records(official))
    keys = selected_indices(records)
    target = DATA / "batch38_frozen"
    manifest_path = target / "manifest.json"
    if manifest_path.exists():
        audit()
        print("TAPVID_RGB_FROZEN_DATA_REUSED", flush=True)
        return
    target.mkdir(parents=True, exist_ok=True)
    public: dict[str, list] = {}
    private: dict[str, dict] = {}
    scenes = []
    for slot, key in enumerate(keys):
        sample = records[key]
        if not isinstance(sample, dict):
            raise RuntimeError("DATASET_SAMPLE_NOT_DICT:" + key)
        frames = np.asarray(sample["video"])
        points = np.asarray(sample["points"], dtype=np.float64)
        occluded = np.asarray(sample["occluded"], dtype=bool)
        if (frames.ndim != 4 or frames.shape[-1] != 3
                or frames.dtype != np.uint8 or points.ndim != 3
                or points.shape[-1] != 2
                or occluded.shape != points.shape[:2]
                or frames.shape[0] != points.shape[1]):
            raise RuntimeError("OFFICIAL_SAMPLE_SHAPE_MISMATCH:" + key)
        subframes = np.ascontiguousarray(frames[::STRIDE])
        suboccluded = occluded[:, ::STRIDE]
        subpoints = normalized_to_pixel_points(
            points[:, ::STRIDE, :], suboccluded, int(frames.shape[1]), int(frames.shape[2]))
        filename = f"frames/clip_{slot:02d}.npy"
        filepath = target / filename
        if not filepath.exists():
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with filepath.open("xb") as f:
                np.save(f, subframes, allow_pickle=False)
        else:
            existing = np.load(filepath, mmap_mode="r", allow_pickle=False)
            if existing.shape != subframes.shape or not np.array_equal(existing, subframes):
                raise RuntimeError("FROZEN_FRAME_CONFLICT")
        qsort = sorted(range(len(subpoints)), key=lambda idx:
            hashlib.sha256((SALT + "|QUERY|" + key + "|" + str(idx)).encode()).hexdigest())
        qs = []
        for idx in qsort[:QUERIES]:
            q = extract_query(key, idx, subpoints, suboccluded)
            if q is not None:
                qs.append(q)
        if not qs:
            raise RuntimeError("NO_QUERY_AT_FIRST_VISIBLE_SAMPLE:" + key)
        clip_id = f"clip_{slot:02d}"
        split = "dev" if slot < DEV else "test"
        public[clip_id] = qs
        private[clip_id] = {
            "tracks": {
                q["id"]: {
                    "xy": [[float(z) for z in pt] for pt in subpoints[q["index"]]],
                    "occluded": [bool(v) for v in suboccluded[q["index"]]],
                } for q in qs
            }
        }
        scenes.append({
            "name": clip_id, "official_scene": str(key), "split": split,
            "frames": filename, "frames_sha256": content_key(filepath),
            "nframes": int(len(subframes)),
            "nqueries": len(qs), "size": list(subframes.shape[1:3]),
        })
        print("FROZEN_SCENE", clip_id, split, key, len(subframes), "queries", len(qs), flush=True)
    manifest = {
        "version": "BATCH38-2026-11-02-RGB-STACKING-v1",
        "dataset": "Official TAP-Vid RGB-Stacking, synthetic robotics (not real-world DAVIS)",
        "dataset_origin_md5": digest(official, "md5"),
        "dataset_origin_sha256": digest(official),
        "selection_salt": SALT, "sample_stride": STRIDE,
        "n_total_official": len(records), "n_dev": DEV, "n_test": TEST,
        "query_selection": "hash-only point ids, first visible query on sampled frames",
        "coordinates": "official normalized (x,y) multiplied by (frame_width,frame_height) into pixels; upstream TAP-Vid reader convention",
        "oracle_sha256": content_key(target / "oracle-private.json"),
        "public_queries_sha256": content_key(target / "queries-public.json"),
        "scenes": scenes,
    }
    # Write oracle before manifest, so an interrupted write cannot appear complete.
    atomic_once(target / "oracle-private.json", as_json(private))
    atomic_once(target / "queries-public.json", as_json(public))
    atomic_once(manifest_path, as_json(manifest))
    audit()
    print("TAPVID_RGB_PREPARED", len(scenes), "source_sha256", digest(official), flush=True)


def audit() -> dict:
    official = DATA / "official.zip"
    verify(official)
    target = DATA / "batch38_frozen"
    manifest = json.loads((target / "manifest.json").read_text())
    queries = json.loads((target / "queries-public.json").read_text())
    if manifest["dataset_origin_sha256"] != digest(official):
        raise RuntimeError("FROZEN_SOURCE_DIGEST_DRIFT")
    scenes = manifest["scenes"]
    if (len(scenes) != DEV + TEST or len({x["name"] for x in scenes}) != len(scenes)
            or [s["split"] for s in scenes] != ["dev"] * DEV + ["test"] * TEST
            or [s["official_scene"] for s in scenes]
                != selected_indices(s["official_scene"] for s in scenes)):
        raise RuntimeError("FROZEN_SPLIT_DRIFT")
    for scene in scenes:
        name = scene["name"]
        if content_key(target / scene["frames"]) != scene["frames_sha256"]:
            raise RuntimeError("FROZEN_FRAME_HASH_DRIFT:" + name)
        if len(queries[name]) != scene["nqueries"]:
            raise RuntimeError("FROZEN_QUERY_COUNT_DRIFT")
        for q in queries[name]:
            if (not 0 <= q["query_time"] < scene["nframes"]
                    or not np.isfinite(q["query_xy"]).all()):
                raise RuntimeError("INVALID_FROZEN_QUERY")
    print("TAPVID_RGB_AUDIT_PASS", len(scenes), "scenes", flush=True)
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "audit"])
    a = p.parse_args()
    if a.action == "prepare":
        prepare()
    else:
        audit()


if __name__ == "__main__":
    main()
