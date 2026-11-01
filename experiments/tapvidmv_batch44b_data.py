"""Batch 43 sequence-disjoint TAPVid-MV EgoExo4D/PACE RRD acquisition and 3-D oracle firewall."""
from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import urllib.request

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DATA = WORKSPACE / "data/tapvidmv_multiview_batch44bb"
SOURCE = DATA / "source"
RRD_DIR = SOURCE / "rrd"
FROZEN = DATA / "batch44bb_frozen"
CONFIG = ROOT / "configs/tapvidmv_batch44bb.json"
STATUS = {"unknown": 0, "visible": 1, "occluded": 2, "out": 3, "missing": 4}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_bytes(doc: dict) -> bytes:
    return (json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256_file(path) != sha256_bytes(data):
            raise RuntimeError("FROZEN_FILE_CONFLICT:" + str(path))
        return
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        raise RuntimeError("PRESERVED_PARTIAL_FILE_REQUIRES_REVIEW:" + str(tmp))
    with tmp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_array_once(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = np.load(path, mmap_mode="r", allow_pickle=False)
        if old.shape != array.shape or old.dtype != array.dtype or not np.array_equal(old, array):
            raise RuntimeError("FROZEN_ARRAY_CONFLICT:" + str(path))
        return
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        raise RuntimeError("PRESERVED_PARTIAL_ARRAY_REQUIRES_REVIEW:" + str(tmp))
    with tmp.open("xb") as f:
        np.save(f, array, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_npz_once(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(arrays):
                raise RuntimeError("FROZEN_ORACLE_CONFLICT:" + str(path))
            for key, value in arrays.items():
                expected = np.asarray(value)
                if old[key].shape != expected.shape:
                    raise RuntimeError("FROZEN_ORACLE_CONFLICT:" + str(path) + ":" + key)
                equal = (np.array_equal(old[key], expected, equal_nan=True)
                         if expected.dtype.kind in "fc" else np.array_equal(old[key], expected))
                if not equal:
                    raise RuntimeError("FROZEN_ORACLE_CONFLICT:" + str(path) + ":" + key)
        return
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        raise RuntimeError("PRESERVED_PARTIAL_ORACLE_REQUIRES_REVIEW:" + str(tmp))
    with tmp.open("xb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def npz_bytes(**arrays) -> bytes:
    import io
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def parse_index(text: str) -> dict[str, dict]:
    records = {}
    sections = re.findall(
        r"<h2 id='([^']+)'[^>]*>.*?</h2>(.*?)(?=<h2 id='|\Z)",
        text, flags=re.S,
    )
    for subset, section in sections:
        if subset not in {"egoexo4d", "pace"}:
            continue
        rows = re.findall(r"<tr>(.*?)</tr>", section, flags=re.S)
        for row in rows[1:]:
            cells = re.findall(r"<td>(.*?)</td>", row, flags=re.S)
            if len(cells) < 5:
                continue
            values = [html.unescape(re.sub(r"<.*?>", "", cell)).strip() for cell in cells]
            links = re.findall(r"href='([^']+)'", row)
            sequence = values[0]
            if not sequence or len(links) < 2:
                raise RuntimeError("MALFORMED_TAPVIDMV_INDEX_ROW")
            if subset == "egoexo4d":
                group = re.sub(r"_[0-9]+$", "", sequence)
            else:
                group = sequence.split("_video", 1)[0]
            records[sequence] = {
                "sequence": sequence, "subset": subset,
                "group": f"{subset}:{group}",
                "views": int(values[1]), "frames": int(values[2]),
                "tracks_metadata": int(values[3]), "size_label": values[4],
                "url": links[-1],
            }
    if len(records) < 80:
        raise RuntimeError("TAPVIDMV_EGOEXO4D_PACE_INDEX_INCOMPLETE:" + str(len(records)))
    return records


def deterministic_selection(records: dict[str, dict], cfg: dict) -> list[dict]:
    result = []
    for expected in cfg["sequences"]:
        record = records.get(expected["sequence"])
        if record is None:
            raise RuntimeError("TAPVIDMV_SELECTED_SEQUENCE_MISSING:" + expected["sequence"])
        for key in ("subset", "group", "frames", "tracks_metadata"):
            if record[key] != expected[key]:
                raise RuntimeError("TAPVIDMV_INDEX_METADATA_DRIFT:" + expected["sequence"] + ":" + key)
        if record["views"] != 3:
            raise RuntimeError("TAPVIDMV_EXPECTED_THREE_VIEWS:" + expected["sequence"])
        result.append({**record, "split": expected["split"]})
    if len(result) != cfg["dev_groups"] + cfg["test_groups"]:
        raise RuntimeError("TAPVIDMV_SELECTED_SCENE_COUNT_DRIFT")
    if len({x["sequence"] for x in result}) != len(result):
        raise RuntimeError("TAPVIDMV_DUPLICATE_SEQUENCE")
    group_splits = {}
    for item in result:
        group_splits.setdefault(item["group"], set()).add(item["split"])
    if any(len(splits) > 1 for splits in group_splits.values()):
        raise RuntimeError("TAPVIDMV_CROSS_SPLIT_COLLECTION_GROUP_LEAKAGE")
    expected_splits = ["dev"] * cfg["dev_groups"] + ["test"] * cfg["test_groups"]
    if [x["split"] for x in result] != expected_splits:
        raise RuntimeError("TAPVIDMV_SPLIT_ORDER_DRIFT")
    counts = {}
    for item in result:
        counts.setdefault(item["subset"], {"dev": 0, "test": 0})[item["split"]] += 1
    if counts != cfg["subset_split_counts"]:
        raise RuntimeError("TAPVIDMV_SUBSET_SPLIT_BALANCE_DRIFT:" + json.dumps(counts, sort_keys=True))
    return result


def download() -> None:
    cfg = config()
    SOURCE.mkdir(parents=True, exist_ok=True)
    RRD_DIR.mkdir(parents=True, exist_ok=True)
    index_path = SOURCE / "index.html"
    if index_path.exists():
        index_sha = sha256_file(index_path)
    else:
        with urllib.request.urlopen(cfg["source_index_url"], timeout=60) as response:
            index_data = response.read()
        write_once(index_path, index_data)
        index_sha = sha256_file(index_path)
    if index_sha != cfg["source_index_sha256"]:
        raise RuntimeError("TAPVIDMV_SOURCE_INDEX_SHA_MISMATCH:" + index_sha)
    records = parse_index(index_path.read_text(encoding="utf-8"))
    selected = deterministic_selection(records, cfg)
    import shutil
    expected_bytes = 0
    for item in selected:
        match = re.search(r"([0-9.]+)\s*(KiB|MiB|GiB|KB|MB|GB)", item["size_label"], re.I)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            factor = {"kib": 1024, "mib": 1024**2, "gib": 1024**3,
                      "kb": 1000, "mb": 1000**2, "gb": 1000**3}[unit]
            expected_bytes += int(value * factor)
    free_bytes = shutil.disk_usage(DATA).free
    if free_bytes < expected_bytes * 1.25 + 2 * 1024**3:
        raise RuntimeError("TAPVIDMV_INSUFFICIENT_DISK_SPACE:" + str(free_bytes))
    existing_manifest_path = SOURCE / "source_manifest.json"
    if existing_manifest_path.exists():
        existing = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        expected_keys = [(x["split"], x["sequence"]) for x in selected]
        actual_keys = [(x["split"], x["sequence"]) for x in existing["sequences"]]
        if existing.get("source_index_sha256") != index_sha or actual_keys != expected_keys:
            raise RuntimeError("TAPVIDMV_EXISTING_SOURCE_MANIFEST_CONFLICT")
        for record in existing["sequences"]:
            source_path = DATA / record["rrd"]
            if not source_path.is_file() or sha256_file(source_path) != record["rrd_sha256"]:
                raise RuntimeError("TAPVIDMV_EXISTING_SOURCE_RRD_HASH_DRIFT:" + record["sequence"])
        print("TAPVIDMV_SOURCE_REUSED", len(existing["sequences"]), index_sha, flush=True)
        return
    manifest = {
        "version": cfg["version"],
        "source_index_sha256": index_sha,
        "rrd_version": cfg["rrd_version"],
        "selection_salt": cfg["selection_salt"],
        "groups_disjoint": True,
        "subsets": ["egoexo4d", "pace"],
        "sequences": [],
    }
    for item in selected:
        name = safe_name(item["sequence"])
        target = RRD_DIR / (name + ".rrd")
        part = target.with_suffix(".rrd.part")
        if not target.exists():
            command = ["curl", "-fL", "--retry", "3", "--connect-timeout", "20"]
            if part.exists() and part.stat().st_size > 0:
                command += ["-C", "-"]
            command += [item["url"], "-o", str(part)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
            if result.returncode:
                raise RuntimeError(
                    "RRD_DOWNLOAD_FAILED:" + item["sequence"] + ":" + result.stderr[-1200:]
                )
            os.replace(part, target)
        if target.stat().st_size < 1024 * 1024:
            raise RuntimeError("RRD_FILE_TOO_SMALL:" + item["sequence"])
        manifest["sequences"].append({
            "split": item["split"],
            "subset": item["subset"],
            "group": item["group"],
            "sequence": item["sequence"],
            "rrd": str(target.relative_to(DATA)),
            "rrd_sha256": sha256_file(target),
            "bytes": target.stat().st_size,
            "frames_index": item["frames"],
            "tracks_metadata": item["tracks_metadata"],
            "url": item["url"],
        })
        print("RRD_READY", item["split"], item["group"], item["sequence"],
              target.stat().st_size, sha256_file(target), flush=True)
    write_once(SOURCE / "source_manifest.json", json_bytes(manifest))
    print("TAPVIDMV_SOURCE_FROZEN", len(selected), index_sha, flush=True)


def safe_name(sequence: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", sequence).strip("_")


def scalar(value):
    return np.asarray(value).reshape(-1)[0].item()


def text_scalar(value) -> str:
    item = scalar(value)
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def rows_by_frame(chunks, entity: str, component: str) -> dict[int, object]:
    output = {}
    for chunk in chunks:
        if chunk.entity_path != entity:
            continue
        data = chunk.to_record_batch().to_pydict()
        if component not in data or "frame" not in data:
            continue
        for index, frame in enumerate(data["frame"]):
            frame_id = int(np.asarray(frame).reshape(-1)[0])
            output[frame_id] = data[component][index]
    return output


def static_component(chunks, entity: str, component: str):
    for chunk in chunks:
        if chunk.entity_path != entity:
            continue
        data = chunk.to_record_batch().to_pydict()
        if component in data:
            return data[component][0]
    raise RuntimeError("RRD_STATIC_COMPONENT_MISSING:" + entity + ":" + component)


def project_point(point: np.ndarray, rotation: np.ndarray, translation: np.ndarray,
                  intrinsics: np.ndarray) -> tuple[np.ndarray, float]:
    camera_xyz = rotation @ (np.asarray(point, dtype=np.float64) - translation)
    depth = float(camera_xyz[2])
    if not math.isfinite(depth) or depth <= 1e-8:
        return np.array([np.nan, np.nan], dtype=np.float64), depth
    normalized = np.array([camera_xyz[0] / depth, camera_xyz[1] / depth, 1.0])
    pixel = normalized @ intrinsics
    return np.asarray(pixel[:2], dtype=np.float64), depth


def project_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray,
                   intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera_xyz = (rotation @ (np.asarray(points, dtype=np.float64) - translation).T).T
    depth = camera_xyz[:, 2]
    safe_depth = np.where(depth > 1e-8, depth, np.nan)
    normalized = np.column_stack([
        camera_xyz[:, 0] / safe_depth,
        camera_xyz[:, 1] / safe_depth,
        np.ones(len(camera_xyz)),
    ])
    pixels = normalized @ intrinsics
    return pixels[:, :2], depth


def make_depth_buffer(cloud: np.ndarray, pose: tuple, intrinsics: np.ndarray,
                      width: int, height: int) -> np.ndarray:
    rotation, translation = pose
    pixels, depth = project_points(cloud, rotation, translation, intrinsics)
    good = (
        (depth > 1e-8) & np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    x = np.clip(np.rint(pixels[good, 0]).astype(np.int64), 0, width - 1)
    y = np.clip(np.rint(pixels[good, 1]).astype(np.int64), 0, height - 1)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    np.minimum.at(depth_buffer, (y, x), depth[good].astype(np.float32))
    return depth_buffer


def classify_depth(pixel: np.ndarray, point_depth: float, depth_buffer: np.ndarray,
                   radius: int, margin: float) -> int:
    height, width = depth_buffer.shape
    x, y = float(pixel[0]), float(pixel[1])
    if (not np.isfinite(pixel).all() or point_depth <= 1e-8
            or x < 0 or x >= width or y < 0 or y >= height):
        return STATUS["out"]
    ix = int(np.clip(round(x), 0, width - 1))
    iy = int(np.clip(round(y), 0, height - 1))
    patch = depth_buffer[
        max(0, iy - radius):min(height, iy + radius + 1),
        max(0, ix - radius):min(width, ix + radius + 1),
    ]
    nearest = float(np.min(patch))
    if not math.isfinite(nearest):
        return STATUS["unknown"]
    delta = point_depth - nearest
    if delta > margin:
        return STATUS["occluded"]
    if abs(delta) <= margin:
        return STATUS["visible"]
    return STATUS["unknown"]


def read_rrd(path: Path, expected_sequence: str) -> dict:
    try:
        import rerun as rr
    except Exception as exc:
        raise RuntimeError("RERUN_SDK_0_35_REQUIRED") from exc
    if rr.__version__ != "0.35.0":
        raise RuntimeError("RERUN_VERSION_MISMATCH:" + rr.__version__)
    reader = rr.experimental.RrdReader(str(path))
    recordings = reader.recordings()
    if len(recordings) != 1:
        raise RuntimeError("RRD_RECORDING_COUNT_MISMATCH")
    chunks = reader.store(store=recordings[0]).stream().to_chunks()
    world = None
    for chunk in chunks:
        if chunk.entity_path == "/world":
            data = chunk.to_record_batch().to_pydict()
            if "frames" in data:
                world = data
                break
    if world is None:
        raise RuntimeError("RRD_WORLD_METADATA_MISSING")
    nframes = int(scalar(world["frames"][0]))
    sequence = text_scalar(world["sequence"][0])
    views = int(scalar(world["views"][0]))
    tracks_metadata = int(scalar(world["tracks"][0]))
    if sequence != expected_sequence or views != 3 or nframes < 20:
        raise RuntimeError("RRD_METADATA_MISMATCH:" + sequence)
    cameras = {}
    images = {}
    resolutions = {}
    for camera_id in range(3):
        entity = f"/world/cam{camera_id}"
        image_entity = entity + "/image"
        pose_rows = rows_by_frame(chunks, entity, "Transform3D:mat3x3")
        translation_rows = rows_by_frame(chunks, entity, "Transform3D:translation")
        if set(pose_rows) != set(range(nframes)) or set(translation_rows) != set(range(nframes)):
            raise RuntimeError("RRD_CAMERA_POSE_FRAME_GAP:" + str(camera_id))
        rotations = np.stack([
            np.asarray(pose_rows[i], dtype=np.float64).reshape(3, 3)
            for i in range(nframes)
        ])
        translations = np.stack([
            np.asarray(translation_rows[i], dtype=np.float64).reshape(3)
            for i in range(nframes)
        ])
        intrinsics = np.asarray(
            static_component(chunks, image_entity, "Pinhole:image_from_camera"),
            dtype=np.float64,
        ).reshape(3, 3)
        resolution = np.asarray(
            static_component(chunks, image_entity, "Pinhole:resolution"),
            dtype=np.float64,
        ).reshape(-1)
        cam_width, cam_height = int(resolution[0]), int(resolution[1])
        resolutions[camera_id] = (cam_width, cam_height)
        image_rows = rows_by_frame(chunks, image_entity, "EncodedImage:blob")
        if set(image_rows) != set(range(nframes)):
            raise RuntimeError("RRD_IMAGE_FRAME_GAP:" + str(camera_id))
        frames = []
        for frame_id in range(nframes):
            blob = bytes(np.asarray(image_rows[frame_id], dtype=np.uint8).reshape(-1))
            rgb = np.asarray(Image.open(BytesIO(blob)).convert("RGB"), dtype=np.uint8)
            if rgb.shape != (cam_height, cam_width, 3):
                raise RuntimeError("RRD_DECODED_IMAGE_SHAPE_MISMATCH")
            frames.append(rgb)
        cameras[camera_id] = {
            "rotation": rotations,
            "translation": translations,
            "intrinsics": intrinsics,
        }
        images[camera_id] = np.stack(frames)
    pos_rows = rows_by_frame(chunks, "/world/tracks/head", "Points3D:positions")
    color_rows = rows_by_frame(chunks, "/world/tracks/head", "Points3D:colors")
    if set(pos_rows) != set(range(nframes)) or set(color_rows) != set(range(nframes)):
        raise RuntimeError("RRD_TRACK_HEAD_FRAME_GAP")
    track_maps = {}
    all_ids = set()
    max_display = 0
    for frame_id in range(nframes):
        positions = np.asarray(pos_rows[frame_id], dtype=np.float64).reshape(-1, 3)
        colors = np.asarray(color_rows[frame_id], dtype=np.uint32).reshape(-1)
        ids = (colors >> 8).astype(np.uint32)
        if len(ids) != len(positions) or len(set(ids.tolist())) != len(ids):
            raise RuntimeError("RRD_TRACK_ID_COLLISION_OR_SHAPE")
        track_maps[frame_id] = {
            int(track_id): point for track_id, point in zip(ids, positions)
        }
        all_ids.update(int(x) for x in ids)
        max_display = max(max_display, len(ids))
    cloud_rows = rows_by_frame(chunks, "/world/cloud", "Points3D:positions")
    clouds = {
        int(frame_id): np.asarray(value, dtype=np.float64).reshape(-1, 3)
        for frame_id, value in cloud_rows.items()
    }
    if 0 not in clouds:
        raise RuntimeError("RRD_FRAME_ZERO_DEPTH_CLOUD_MISSING")
    return {
        "sequence": sequence,
        "nframes": nframes,
        "tracks_metadata": tracks_metadata,
        "views": views,
        "resolutions": resolutions,
        "cameras": cameras,
        "images": images,
        "track_maps": track_maps,
        "clouds": clouds,
        "display_track_id_union": len(all_ids),
        "display_tracks_per_frame_max": max_display,
    }


def validate_public_calibration(arrays: dict, nframes: int) -> bool:
    required = {"rotations", "translations", "intrinsics", "resolution"}
    if set(arrays) != required:
        raise RuntimeError("PUBLIC_CALIBRATION_SCHEMA_LEAK_OR_DRIFT")
    rotations = np.asarray(arrays["rotations"])
    translations = np.asarray(arrays["translations"])
    intrinsics = np.asarray(arrays["intrinsics"])
    resolution = np.asarray(arrays["resolution"])
    if (rotations.shape != (3, nframes, 3, 3)
            or translations.shape != (3, nframes, 3)
            or intrinsics.shape != (3, 3, 3)
            or resolution.shape != (3, 2)):
        raise RuntimeError("PUBLIC_CALIBRATION_SHAPE_DRIFT")
    if not all(np.isfinite(x).all() for x in (rotations, translations, intrinsics)):
        raise RuntimeError("PUBLIC_CALIBRATION_NONFINITE")
    if np.any(resolution <= 0):
        raise RuntimeError("PUBLIC_CALIBRATION_INVALID_RESOLUTION")
    return True


def choose_query_frame(common_counts: dict[int, int], nframes: int,
                       min_future_frames: int, min_common: int) -> int:
    for frame_id in sorted(common_counts):
        if (0 <= int(frame_id) <= nframes - min_future_frames
                and int(common_counts[frame_id]) >= min_common):
            return int(frame_id)
    raise RuntimeError("NO_COMMON_THREE_VIEW_VISIBLE_QUERY_FRAME")


def visible_tracks_for_frame(raw: dict, depth_buffers: dict, resolutions: dict,
                             frame_id: int, cfg: dict) -> dict:
    visible_by_camera = {}
    for camera_id in range(3):
        camera = raw["cameras"][camera_id]
        frame_pose = (camera["rotation"][frame_id], camera["translation"][frame_id])
        visible = {}
        for track_id, point in raw["track_maps"][frame_id].items():
            pixel, depth = project_point(point, frame_pose[0], frame_pose[1], camera["intrinsics"])
            status = classify_depth(
                pixel, depth, depth_buffers[camera_id][frame_id],
                cfg["depth_visibility"]["radius_px"], cfg["depth_visibility"]["depth_margin_m"],
            )
            if status == STATUS["visible"]:
                visible[int(track_id)] = pixel
        visible_by_camera[camera_id] = visible
    return visible_by_camera


def extract_scene(item: dict, source_record: dict, scene_slot: int, cfg: dict) -> dict:
    rrd_path = DATA / source_record["rrd"]
    raw = read_rrd(rrd_path, item["sequence"])
    if raw["nframes"] != item["frames"] or raw["tracks_metadata"] != item["tracks_metadata"]:
        raise RuntimeError("RRD_INDEX_METADATA_MISMATCH:" + item["sequence"])
    nframes = raw["nframes"]
    resolutions = raw["resolutions"]
    cloud_frames = sorted(raw["clouds"])
    if not cloud_frames:
        raise RuntimeError("QUERY_FRAME_DEPTH_SNAPSHOT_REQUIRED")
    depth_buffers = {camera_id: {} for camera_id in range(3)}
    for camera_id in range(3):
        camera = raw["cameras"][camera_id]
        width, height = resolutions[camera_id]
        for frame_id in cloud_frames:
            pose = (camera["rotation"][frame_id], camera["translation"][frame_id])
            depth_buffers[camera_id][frame_id] = make_depth_buffer(
                raw["clouds"][frame_id], pose, camera["intrinsics"], width, height
            )
    candidate_frames = sorted(set(cloud_frames) & set(raw["track_maps"]))
    common_counts = {}
    for candidate_frame in candidate_frames:
        visible_at_frame = visible_tracks_for_frame(raw, depth_buffers, resolutions, candidate_frame, cfg)
        common_counts[candidate_frame] = len(set.intersection(
            *(set(visible_at_frame[c].keys()) for c in range(3))
        ))
    query_frame = choose_query_frame(
        common_counts, nframes,
        int(cfg["query_frame_selection"]["min_future_frames"]),
        int(cfg["queries_per_camera"]),
    )
    visible_by_camera = visible_tracks_for_frame(raw, depth_buffers, resolutions, query_frame, cfg)
    common_ids = set.intersection(*(set(visible_by_camera[c].keys()) for c in range(3)))
    future_cloud_frames = [f for f in cloud_frames if f > query_frame]
    if not future_cloud_frames:
        raise RuntimeError("NO_FUTURE_DEPTH_SNAPSHOT_AFTER_QUERY_FRAME")
    target_frame = min(future_cloud_frames, key=lambda f: abs(f - cfg["target_fraction"] * (nframes - 1)))
    queries = []
    query_private = []
    ranked_ids = sorted(
        common_ids,
        key=lambda track_id: hashlib.sha256(
            f"{cfg['selection_salt']}|cross-view-query|{item['sequence']}|{track_id}".encode()
        ).hexdigest(),
    )
    if len(ranked_ids) < cfg["queries_per_camera"]:
        raise RuntimeError(
            "NOT_ENOUGH_COMMON_THREE_VIEW_VISIBLE_TRACKS:" + item["sequence"]
            + ":" + str(len(ranked_ids))
        )
    for track_slot, track_id in enumerate(ranked_ids[:cfg["queries_per_camera"]]):
        cross_view_group = f"g{track_slot:02d}"
        for camera_id in range(3):
            pixel = visible_by_camera[camera_id][track_id]
            qid = f"q{track_slot:02d}_c{camera_id}"
            queries.append({
                "id": qid, "camera_id": camera_id, "query_frame": int(query_frame),
                "query_xy": [float(pixel[0]), float(pixel[1])],
                "target_frame": int(target_frame),
                "cross_view_group": cross_view_group,
                "track_slot": int(track_slot),
            })
            query_private.append((qid, camera_id, track_id, pixel))
    qcount = len(queries)
    gt_xy = np.full((qcount, nframes, 2), np.nan, dtype=np.float32)
    track_present = np.zeros((qcount, nframes), dtype=np.bool_)
    in_frame = np.zeros((qcount, nframes), dtype=np.bool_)
    depth_status = np.full((qcount, nframes), STATUS["unknown"], dtype=np.uint8)
    world_xyz = np.full((qcount, nframes, 3), np.nan, dtype=np.float32)
    camera_only_xy = np.full((qcount, nframes, 2), np.nan, dtype=np.float32)
    world_only_xy = np.full((qcount, nframes, 2), np.nan, dtype=np.float32)
    camera_translation_m = np.zeros((qcount, nframes), dtype=np.float32)
    camera_rotation_deg = np.zeros((qcount, nframes), dtype=np.float32)
    camera_rotation_matrix = np.zeros((qcount, nframes, 3, 3), dtype=np.float32)
    camera_translation_xyz = np.zeros((qcount, nframes, 3), dtype=np.float32)
    camera_intrinsics = np.zeros((qcount, 3, 3), dtype=np.float32)
    query_track_ids = np.zeros(qcount, dtype=np.uint32)
    query_camera_ids = np.zeros(qcount, dtype=np.uint8)
    query_depth_m = np.zeros(qcount, dtype=np.float32)
    query_target_frames = np.full(qcount, target_frame, dtype=np.int32)
    for qi, (qid, camera_id, track_id, query_pixel) in enumerate(query_private):
        camera = raw["cameras"][camera_id]
        width, height = resolutions[camera_id]
        query_track_ids[qi] = track_id
        query_camera_ids[qi] = camera_id
        initial_point = raw["track_maps"][query_frame][track_id]
        initial_rotation = camera["rotation"][query_frame]
        initial_translation = camera["translation"][query_frame]
        _, initial_depth = project_point(
            initial_point, initial_rotation, initial_translation, camera["intrinsics"]
        )
        query_depth_m[qi] = initial_depth
        camera_intrinsics[qi] = camera["intrinsics"].astype(np.float32)
        for frame_id in range(query_frame, nframes):
            rotation = camera["rotation"][frame_id]
            translation = camera["translation"][frame_id]
            camera_rotation_matrix[qi, frame_id] = rotation.astype(np.float32)
            camera_translation_xyz[qi, frame_id] = translation.astype(np.float32)
            camera_translation_m[qi, frame_id] = np.linalg.norm(
                translation - initial_translation
            )
            relative_rotation = rotation @ initial_rotation.T
            cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
            camera_rotation_deg[qi, frame_id] = math.degrees(math.acos(float(cosine)))
            static_pixel, _ = project_point(
                initial_point, rotation, translation, camera["intrinsics"]
            )
            camera_only_xy[qi, frame_id] = static_pixel.astype(np.float32)
            point = raw["track_maps"][frame_id].get(track_id)
            if point is None:
                depth_status[qi, frame_id] = STATUS["missing"]
                continue
            track_present[qi, frame_id] = True
            world_xyz[qi, frame_id] = point.astype(np.float32)
            pixel, point_depth = project_point(
                point, rotation, translation, camera["intrinsics"]
            )
            if np.isfinite(pixel).all():
                gt_xy[qi, frame_id] = pixel.astype(np.float32)
            inside = (
                np.isfinite(pixel).all() and point_depth > 1e-8
                and 0 <= pixel[0] < width and 0 <= pixel[1] < height
            )
            in_frame[qi, frame_id] = bool(inside)
            world_pixel, _ = project_point(
                point, initial_rotation, initial_translation, camera["intrinsics"]
            )
            world_only_xy[qi, frame_id] = world_pixel.astype(np.float32)
            if not inside:
                depth_status[qi, frame_id] = STATUS["out"]
            elif frame_id in depth_buffers[camera_id]:
                depth_status[qi, frame_id] = classify_depth(
                    pixel, point_depth, depth_buffers[camera_id][frame_id],
                    cfg["depth_visibility"]["radius_px"],
                    cfg["depth_visibility"]["depth_margin_m"],
                )
            else:
                depth_status[qi, frame_id] = STATUS["unknown"]
    scene_id = f"scene_{scene_slot:02d}"
    calibration_relative = f"calibration-public/{scene_id}.npz"
    calibration_arrays = {
        "rotations": np.stack([raw["cameras"][c]["rotation"] for c in range(3)]).astype(np.float32),
        "translations": np.stack([raw["cameras"][c]["translation"] for c in range(3)]).astype(np.float32),
        "intrinsics": np.stack([raw["cameras"][c]["intrinsics"] for c in range(3)]).astype(np.float32),
        "resolution": np.asarray([resolutions[c] for c in range(3)], dtype=np.int32),
    }
    validate_public_calibration(calibration_arrays, nframes)
    calibration_path = FROZEN / calibration_relative
    save_npz_once(calibration_path, **calibration_arrays)
    frame_paths = {}
    frame_hashes = {}
    for camera_id in range(3):
        relative = f"frames/{scene_id}_cam{camera_id}.npy"
        frame_path = FROZEN / relative
        save_array_once(frame_path, raw["images"][camera_id])
        frame_paths[str(camera_id)] = relative
        frame_hashes[str(camera_id)] = sha256_file(frame_path)
    oracle_relative = f"oracle-private/{scene_id}.npz"
    oracle_path = FROZEN / oracle_relative
    save_npz_once(
        oracle_path,
        qids=np.asarray([x[0] for x in query_private], dtype="U16"),
        query_track_ids=query_track_ids,
        query_camera_ids=query_camera_ids,
        query_depth_m=query_depth_m,
        query_target_frames=query_target_frames,
        gt_xy=gt_xy,
        track_present=track_present,
        in_frame=in_frame,
        depth_status=depth_status,
        world_xyz=world_xyz,
        camera_only_xy=camera_only_xy,
        world_only_xy=world_only_xy,
        camera_translation_m=camera_translation_m,
        camera_rotation_deg=camera_rotation_deg,
        camera_rotation_matrix=camera_rotation_matrix,
        camera_translation_xyz=camera_translation_xyz,
        camera_intrinsics=camera_intrinsics,
    )
    return {
        "name": scene_id,
        "sequence": item["sequence"],
        "subset": item["subset"],
        "group": item["group"],
        "split": item["split"],
        "nframes": nframes,
        "views": 3,
        "resolution_by_camera": {
            str(camera_id): list(resolutions[camera_id]) for camera_id in range(3)
        },
        "tracks_metadata": raw["tracks_metadata"],
        "display_track_id_union": raw["display_track_id_union"],
        "display_tracks_per_frame_max": raw["display_tracks_per_frame_max"],
        "cloud_snapshot_frames": cloud_frames,
        "frame_files": frame_paths,
        "frame_sha256": frame_hashes,
        "calibration_file": calibration_relative,
        "calibration_sha256": sha256_file(calibration_path),
        "oracle_file": oracle_relative,
        "oracle_sha256": sha256_file(oracle_path),
        "query_count": len(queries),
        "queries": queries,
        "rrd_sha256": source_record["rrd_sha256"],
        "rrd_bytes": source_record["bytes"],
    }


def prepare() -> None:
    cfg = config()
    source_manifest_path = SOURCE / "source_manifest.json"
    if not source_manifest_path.is_file():
        raise RuntimeError("SOURCE_MANIFEST_MISSING_RUN_DOWNLOAD_FIRST")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("source_index_sha256") != cfg["source_index_sha256"]:
        raise RuntimeError("SOURCE_MANIFEST_INDEX_SHA_DRIFT")
    if (FROZEN / "manifest.json").exists():
        audit()
        print("TAPVIDMV_FROZEN_DATA_REUSED", flush=True)
        return
    source_by_sequence = {x["sequence"]: x for x in source_manifest["sequences"]}
    scenes = []
    public_queries = {}
    for slot, item in enumerate(cfg["sequences"]):
        source_record = source_by_sequence.get(item["sequence"])
        if (source_record is None or source_record["split"] != item["split"]
                or source_record["subset"] != item["subset"]
                or source_record["group"] != item["group"]):
            raise RuntimeError("SOURCE_SEQUENCE_SPLIT_OR_GROUP_MISMATCH")
        scene = extract_scene(item, source_record, slot, cfg)
        scenes.append({k: v for k, v in scene.items() if k != "queries"})
        public_queries[scene["name"]] = scene["queries"]
        print("TAPVIDMV_SCENE_EXTRACTED", scene["name"], scene["split"],
              scene["sequence"], scene["nframes"], scene["query_count"], flush=True)
    public_data = json_bytes(public_queries)
    write_once(FROZEN / "queries-public.json", public_data)
    manifest = {
        "version": cfg["version"],
        "config_sha256": sha256_bytes(json_bytes(cfg)),
        "dataset": cfg["dataset"],
        "source_index_sha256": cfg["source_index_sha256"],
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "selection_salt": cfg["selection_salt"],
        "split_contract": "11 DEV + 10 TEST sequences; PACE scene06 is DEV-only and scene07 is TEST-only; EgoExo4D collection groups are sequence-disjoint",
        "queries_per_camera": int(cfg["queries_per_camera"]),
        "query_contract": (
            f"{cfg['queries_per_camera']} hash-ranked 3-D track IDs visible in all 3 cameras at the "
            "earliest feasible exact depth snapshot; each shared cross-view point queried once per camera"
        ),
        "query_frame_selection": cfg["query_frame_selection"],
        "future_oracle_private": True,
        "public_calibration_contract": "Camera poses, intrinsics and image resolutions only; no 3-D point tracks, track IDs, depth clouds or future labels.",
        "depth_visibility": cfg["depth_visibility"],
        "camera_projection": cfg["camera_projection"],
        "rrd_version": cfg["rrd_version"],
        "source_license": cfg["source_license"],
        "public_queries_sha256": sha256_bytes(public_data),
        "scenes": scenes,
        "limitations": [
            "Rerun viewer recordings subsample point tracks for display; these are not the full metadata-declared tracks.",
            "Per-view visibility is inferred from exact depth point-cloud snapshots, not a separate official binary occlusion annotation.",
            "Depth visibility is unknown on frames without an exact cloud snapshot.",
        ],
    }
    write_once(FROZEN / "manifest.json", json_bytes(manifest))
    audit()
    print("TAPVIDMV_PREPARED", len(scenes), "scenes", "queries",
          sum(x["query_count"] for x in scenes), flush=True)

def audit_public() -> dict:
    """Verify public source/index/frames/queries without opening any private oracle."""
    cfg = config()
    index_path = SOURCE / "index.html"
    source_manifest_path = SOURCE / "source_manifest.json"
    manifest_path = FROZEN / "manifest.json"
    query_path = FROZEN / "queries-public.json"
    if not all(x.is_file() for x in (index_path, source_manifest_path, manifest_path, query_path)):
        raise RuntimeError("TAPVIDMV_FROZEN_MANIFEST_MISSING")
    if sha256_file(index_path) != cfg["source_index_sha256"]:
        raise RuntimeError("TAPVIDMV_INDEX_SHA_DRIFT")
    index_records = parse_index(index_path.read_text(encoding="utf-8"))
    expected_selection = deterministic_selection(index_records, cfg)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    queries = json.loads(query_path.read_text(encoding="utf-8"))
    if source_manifest.get("source_index_sha256") != cfg["source_index_sha256"]:
        raise RuntimeError("TAPVIDMV_SOURCE_MANIFEST_INDEX_SHA_DRIFT")
    if manifest.get("config_sha256") != sha256_bytes(json_bytes(cfg)):
        raise RuntimeError("TAPVIDMV_CONFIG_SHA_DRIFT")
    expected_contract = (
        f"{cfg['queries_per_camera']} hash-ranked 3-D track IDs visible in all 3 cameras at the "
        "earliest feasible exact depth snapshot; each shared cross-view point queried once per camera"
    )
    if (manifest.get("future_oracle_private") is not True
            or manifest.get("public_calibration_contract") != "Camera poses, intrinsics and image resolutions only; no 3-D point tracks, track IDs, depth clouds or future labels."
            or manifest.get("query_frame_selection") != cfg["query_frame_selection"]
            or manifest.get("rrd_version") != cfg["rrd_version"]
            or manifest.get("queries_per_camera") != cfg["queries_per_camera"]
            or manifest.get("query_contract") != expected_contract
            or manifest.get("depth_visibility") != cfg["depth_visibility"]
            or manifest.get("camera_projection") != cfg["camera_projection"]
            or manifest.get("source_license") != cfg["source_license"]):
        raise RuntimeError("TAPVIDMV_MANIFEST_PROTOCOL_DRIFT")
    if manifest.get("source_index_sha256") != cfg["source_index_sha256"]:
        raise RuntimeError("TAPVIDMV_INDEX_SHA_DRIFT")
    if manifest.get("source_manifest_sha256") != sha256_file(source_manifest_path):
        raise RuntimeError("TAPVIDMV_SOURCE_MANIFEST_DRIFT")
    if manifest.get("public_queries_sha256") != sha256_file(query_path):
        raise RuntimeError("TAPVIDMV_PUBLIC_QUERY_HASH_DRIFT")
    expected_source = [(x["split"], x["subset"], x["group"], x["sequence"]) for x in expected_selection]
    actual_source = [(x["split"], x["subset"], x["group"], x["sequence"])
                     for x in source_manifest.get("sequences", [])]
    if actual_source != expected_source:
        raise RuntimeError("TAPVIDMV_SOURCE_SELECTION_DRIFT")
    if len(manifest.get("scenes", [])) != cfg["dev_groups"] + cfg["test_groups"]:
        raise RuntimeError("TAPVIDMV_SCENE_COUNT_DRIFT")
    expected_scenes = [(x["split"], x["subset"], x["group"], x["sequence"]) for x in expected_selection]
    actual_scenes = [(x["split"], x["subset"], x["group"], x["sequence"])
                     for x in manifest["scenes"]]
    if actual_scenes != expected_scenes:
        raise RuntimeError("TAPVIDMV_MANIFEST_SEQUENCE_SPLIT_DRIFT")
    group_splits = {}
    for scene in manifest["scenes"]:
        group_splits.setdefault(scene["group"], set()).add(scene["split"])
    if any(len(splits) > 1 for splits in group_splits.values()):
        raise RuntimeError("TAPVIDMV_GROUP_LEAKAGE")
    subset_counts = {}
    for scene in manifest["scenes"]:
        subset_counts.setdefault(scene["subset"], {"dev": 0, "test": 0})[scene["split"]] += 1
    if subset_counts != cfg["subset_split_counts"]:
        raise RuntimeError("TAPVIDMV_MANIFEST_SUBSET_SPLIT_DRIFT")
    source_by_sequence = {x["sequence"]: x for x in source_manifest["sequences"]}
    for scene in manifest["scenes"]:
        source_record = source_by_sequence.get(scene["sequence"])
        if source_record is None:
            raise RuntimeError("TAPVIDMV_SCENE_NOT_IN_SOURCE_MANIFEST")
        if (source_record["rrd_sha256"] != scene["rrd_sha256"]
                or source_record["subset"] != scene["subset"]
                or source_record["group"] != scene["group"]):
            raise RuntimeError("TAPVIDMV_RRD_HASH_OR_GROUP_DRIFT")
        rrd_path = DATA / source_record["rrd"]
        if not rrd_path.is_file() or sha256_file(rrd_path) != source_record["rrd_sha256"]:
            raise RuntimeError("TAPVIDMV_RRD_SOURCE_HASH_DRIFT:" + scene["sequence"])
        if scene["query_count"] != 3 * int(cfg["queries_per_camera"]):
            raise RuntimeError("TAPVIDMV_SCENE_QUERY_COUNT_DRIFT")
        if not str(scene["oracle_file"]).startswith("oracle-private/"):
            raise RuntimeError("TAPVIDMV_ORACLE_PATH_NOT_PRIVATE")
        for camera_id, relative in scene["frame_files"].items():
            frame_path = FROZEN / relative
            if FROZEN.resolve() not in frame_path.resolve().parents:
                raise RuntimeError("TAPVIDMV_FRAME_PATH_ESCAPE")
            if not frame_path.is_file() or sha256_file(frame_path) != scene["frame_sha256"][camera_id]:
                raise RuntimeError("TAPVIDMV_FRAME_HASH_DRIFT:" + relative)
            frames = np.load(frame_path, mmap_mode="r", allow_pickle=False)
            width, height = scene["resolution_by_camera"][camera_id]
            if frames.shape != (scene["nframes"], height, width, 3) or frames.dtype != np.uint8:
                raise RuntimeError("TAPVIDMV_FRAME_SHAPE_DRIFT:" + relative)
        calibration_relative = scene.get("calibration_file", "")
        if not calibration_relative.startswith("calibration-public/"):
            raise RuntimeError("TAPVIDMV_PUBLIC_CALIBRATION_PATH_NOT_PUBLIC")
        calibration_path = FROZEN / calibration_relative
        if FROZEN.resolve() not in calibration_path.resolve().parents:
            raise RuntimeError("TAPVIDMV_PUBLIC_CALIBRATION_PATH_ESCAPE")
        if (not calibration_path.is_file()
                or sha256_file(calibration_path) != scene.get("calibration_sha256")):
            raise RuntimeError("TAPVIDMV_PUBLIC_CALIBRATION_HASH_DRIFT")
        with np.load(calibration_path, allow_pickle=False) as calibration:
            validate_public_calibration(
                {key: calibration[key] for key in calibration.files}, int(scene["nframes"])
            )
        rows = queries.get(scene["name"])
        if not isinstance(rows, list) or len(rows) != scene["query_count"]:
            raise RuntimeError("TAPVIDMV_QUERY_COUNT_DRIFT")
        if len({q["id"] for q in rows}) != len(rows):
            raise RuntimeError("TAPVIDMV_DUPLICATE_QUERY_ID")
        per_camera = {camera_id: 0 for camera_id in range(3)}
        by_group = {}
        for q in rows:
            if set(q) != {"id", "camera_id", "query_frame", "query_xy", "target_frame", "cross_view_group", "track_slot"}:
                raise RuntimeError("PUBLIC_QUERY_LEAKS_PRIVATE_FIELDS")
            camera_id = int(q["camera_id"])
            if camera_id not in (0, 1, 2):
                raise RuntimeError("INVALID_TAPVIDMV_CAMERA_ID")
            per_camera[camera_id] += 1
            x, y = [float(v) for v in q["query_xy"]]
            width, height = scene["resolution_by_camera"][str(camera_id)]
            query_frame = int(q["query_frame"])
            target_frame = int(q["target_frame"])
            min_future = int(cfg["query_frame_selection"]["min_future_frames"])
            if (query_frame not in scene["cloud_snapshot_frames"]
                    or query_frame > scene["nframes"] - min_future
                    or not np.isfinite([x, y]).all()
                    or not (0 <= x < width and 0 <= y < height)
                    or not (query_frame < target_frame < scene["nframes"])
                    or target_frame not in scene["cloud_snapshot_frames"]):
                raise RuntimeError("INVALID_TAPVIDMV_PUBLIC_QUERY")
            by_group.setdefault(q["cross_view_group"], []).append(q)
        if per_camera != {camera_id: int(cfg["queries_per_camera"]) for camera_id in range(3)}:
            raise RuntimeError("TAPVIDMV_PER_CAMERA_QUERY_COUNT_DRIFT")
        if len(by_group) != int(cfg["queries_per_camera"]):
            raise RuntimeError("TAPVIDMV_CROSS_VIEW_GROUP_COUNT_DRIFT")
        for group_rows in by_group.values():
            if (len(group_rows) != 3
                    or {int(q["camera_id"]) for q in group_rows} != {0, 1, 2}
                    or len({int(q["track_slot"]) for q in group_rows}) != 1
                    or len({int(q["target_frame"]) for q in group_rows}) != 1
                    or len({int(q["query_frame"]) for q in group_rows}) != 1):
                raise RuntimeError("TAPVIDMV_CROSS_VIEW_QUERY_ALIGNMENT_DRIFT")
    if set(queries) != {x["name"] for x in manifest["scenes"]}:
        raise RuntimeError("TAPVIDMV_PUBLIC_QUERY_SCENE_SET_DRIFT")
    print("TAPVIDMV_PUBLIC_AUDIT_PASS", len(manifest["scenes"]),
          "scenes", sum(x["query_count"] for x in manifest["scenes"]), "queries", flush=True)
    return manifest

def audit_private(split: str | None = None) -> dict:
    """Verify oracle hashes for one split only; never decode labels during prediction."""
    if split not in (None, "dev", "test"):
        raise ValueError("INVALID_AUDIT_SPLIT")
    manifest = audit_public()
    checked = []
    for scene in manifest["scenes"]:
        if split is not None and scene["split"] != split:
            continue
        oracle_path = FROZEN / scene["oracle_file"]
        if FROZEN.resolve() not in oracle_path.resolve().parents:
            raise RuntimeError("TAPVIDMV_ORACLE_PATH_ESCAPE")
        if not oracle_path.is_file() or sha256_file(oracle_path) != scene["oracle_sha256"]:
            raise RuntimeError("TAPVIDMV_ORACLE_HASH_DRIFT:" + scene["name"])
        query_rows = doc(FROZEN / "queries-public.json")[scene["name"]]
        with np.load(oracle_path, allow_pickle=False) as oracle:
            qids = [str(x) for x in oracle["qids"].tolist()]
            q_index = {qid: i for i, qid in enumerate(qids)}
            for query in query_rows:
                qi = q_index[query["id"]]
                query_frame = int(query["query_frame"])
                if (int(oracle["depth_status"][qi, query_frame]) != STATUS["visible"]
                        or not bool(oracle["track_present"][qi, query_frame])
                        or not bool(oracle["in_frame"][qi, query_frame])):
                    raise RuntimeError("QUERY_FRAME_INITIAL_VISIBILITY_ORACLE_MISMATCH")
        checked.append(scene["name"])
    print("TAPVIDMV_PRIVATE_ORACLE_HASH_AUDIT_PASS", split or "all", len(checked), "scenes", flush=True)
    return manifest


def audit() -> dict:
    return audit_private(None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("download", "prepare", "audit-public", "audit-dev-oracle", "audit-test-oracle", "audit")
    )
    args = parser.parse_args()
    if args.action == "download":
        download()
    elif args.action == "prepare":
        prepare()
    elif args.action == "audit-public":
        audit_public()
    elif args.action == "audit-dev-oracle":
        audit_private("dev")
    elif args.action == "audit-test-oracle":
        audit_private("test")
    else:
        audit()


if __name__ == "__main__":
    main()
