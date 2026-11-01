"""Batch 39 TAPVid-MV DROID RRD acquisition, extraction and oracle firewall."""
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
DATA = WORKSPACE / "data/tapvidmv_droid_batch39"
SOURCE = DATA / "source"
RRD_DIR = SOURCE / "rrd"
FROZEN = DATA / "batch39b_frozen"
CONFIG = ROOT / "configs/tapvidmv_droid_batch39b.json"
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


def build_manifest_errata(
    manifest: dict, queries: dict, cfg: dict,
    manifest_sha256: str, public_queries_sha256: str,
) -> dict:
    """Describe a metadata-only correction without mutating the original frozen manifest."""
    per_camera = int(cfg["queries_per_camera"])
    if set(queries) != {scene["name"] for scene in manifest["scenes"]}:
        raise RuntimeError("TAPVIDMV_ERRATA_SCENE_SET_MISMATCH")
    for scene in manifest["scenes"]:
        rows = queries[scene["name"]]
        counts = {camera_id: sum(int(q["camera_id"]) == camera_id for q in rows)
                  for camera_id in range(3)}
        if (len(rows) != 3 * per_camera
                or counts != {camera_id: per_camera for camera_id in range(3)}
                or int(scene["query_count"]) != len(rows)):
            raise RuntimeError("TAPVIDMV_ERRATA_QUERY_COUNT_MISMATCH:" + scene["name"])
    corrected = (
        f"hash-ranked frame-0 track IDs; {per_camera} depth-visible query points per camera"
    )
    return {
        "schema": "tapvidmv-manifest-errata-v1",
        "original_manifest_sha256": manifest_sha256,
        "public_queries_sha256": public_queries_sha256,
        "original_query_contract": manifest.get("query_contract"),
        "corrected_queries_per_camera": per_camera,
        "corrected_query_contract": corrected,
        "reason": (
            "The original human-readable manifest query_contract was stale; "
            "the frozen config and public query payload both contain exactly "
            f"{per_camera} queries per camera. Original frames, queries and private oracle are unchanged."
        ),
    }


def write_manifest_errata() -> dict:
    manifest_path = FROZEN / "manifest.json"
    query_path = FROZEN / "queries-public.json"
    if not (manifest_path.is_file() and query_path.is_file()):
        raise RuntimeError("TAPVIDMV_ERRATA_SOURCE_FILES_MISSING")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    queries = json.loads(query_path.read_text(encoding="utf-8"))
    errata = build_manifest_errata(
        manifest, queries, config(), sha256_file(manifest_path), sha256_file(query_path)
    )
    write_once(FROZEN / "manifest-errata.json", json_bytes(errata))
    return errata


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
    marker = "<h2 id='droid'>"
    if marker not in text:
        raise RuntimeError("TAPVIDMV_DROID_SECTION_MISSING")
    section = text.split(marker, 1)[1].split("<h2", 1)[0]
    rows = re.findall(r"<tr>(.*?)</tr>", section, flags=re.S)
    records = {}
    for row in rows[1:]:
        cells = re.findall(r"<td>(.*?)</td>", row, flags=re.S)
        if len(cells) < 5:
            continue
        values = [html.unescape(re.sub(r"<.*?>", "", cell)).strip() for cell in cells]
        links = re.findall(r"href='([^']+)'", row)
        sequence = values[0]
        if not sequence or len(links) < 2:
            raise RuntimeError("MALFORMED_TAPVIDMV_INDEX_ROW")
        records[sequence] = {
            "sequence": sequence,
            "group": sequence.split("+", 1)[0],
            "views": int(values[1]),
            "frames": int(values[2]),
            "tracks_metadata": int(values[3]),
            "size_label": values[4],
            "url": links[-1],
        }
    if len(records) != 50:
        raise RuntimeError("TAPVIDMV_DROID_SEQUENCE_COUNT_MISMATCH:" + str(len(records)))
    return records


def deterministic_selection(records: dict[str, dict], cfg: dict) -> list[dict]:
    salt = cfg["selection_salt"]
    groups = {}
    for record in records.values():
        groups.setdefault(record["group"], []).append(record)
    ordered_groups = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{salt}|group|{group}".encode()).hexdigest(),
    )
    selected_groups = ordered_groups[: cfg["dev_groups"] + cfg["test_groups"]]
    result = []
    for slot, group in enumerate(selected_groups):
        ranked = sorted(
            groups[group],
            key=lambda record: hashlib.sha256(
                f"{salt}|sequence|{record['sequence']}".encode()
            ).hexdigest(),
        )
        chosen = next(
            (r for r in ranked if r["sequence"] != cfg["excluded_preflight_sequence"]),
            None,
        )
        if chosen is None:
            raise RuntimeError("NO_SEQUENCE_AFTER_PREFLIGHT_EXCLUSION:" + group)
        result.append({
            **chosen,
            "split": "dev" if slot < cfg["dev_groups"] else "test",
        })
    expected = cfg["sequences"]
    if [(x["split"], x["sequence"]) for x in result] != [
        (x["split"], x["sequence"]) for x in expected
    ]:
        raise RuntimeError("HASH_GROUP_SPLIT_OR_SEQUENCE_SELECTION_DRIFT")
    if len({x["group"] for x in result}) != len(result):
        raise RuntimeError("CROSS_SPLIT_GROUP_LEAKAGE")
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


def extract_scene(item: dict, source_record: dict, scene_slot: int, cfg: dict) -> dict:
    rrd_path = DATA / source_record["rrd"]
    raw = read_rrd(rrd_path, item["sequence"])
    if raw["nframes"] != item["frames"] or raw["tracks_metadata"] != item["tracks_metadata"]:
        raise RuntimeError("RRD_INDEX_METADATA_MISMATCH:" + item["sequence"])
    nframes = raw["nframes"]
    resolutions = raw["resolutions"]
    cloud_frames = sorted(raw["clouds"])
    if 0 not in cloud_frames:
        raise RuntimeError("QUERY_FRAME_DEPTH_SNAPSHOT_REQUIRED")
    target_frame = min(
        (f for f in cloud_frames if f > 0),
        key=lambda f: abs(f - cfg["target_fraction"] * (nframes - 1)),
    )
    depth_buffers = {camera_id: {} for camera_id in range(3)}
    for camera_id in range(3):
        camera = raw["cameras"][camera_id]
        width, height = resolutions[camera_id]
        for frame_id in cloud_frames:
            pose = (camera["rotation"][frame_id], camera["translation"][frame_id])
            depth_buffers[camera_id][frame_id] = make_depth_buffer(
                raw["clouds"][frame_id], pose, camera["intrinsics"], width, height
            )
    queries = []
    query_private = []
    query_slot = 0
    for camera_id in range(3):
        camera = raw["cameras"][camera_id]
        frame_pose = (camera["rotation"][0], camera["translation"][0])
        candidates = []
        for track_id, point in raw["track_maps"][0].items():
            pixel, depth = project_point(
                point, frame_pose[0], frame_pose[1], camera["intrinsics"]
            )
            status = classify_depth(
                pixel, depth, depth_buffers[camera_id][0],
                cfg["depth_visibility"]["radius_px"],
                cfg["depth_visibility"]["depth_margin_m"],
            )
            if status == STATUS["visible"]:
                rank = hashlib.sha256(
                    f"{cfg['selection_salt']}|query|{item['sequence']}|{camera_id}|{track_id}".encode()
                ).hexdigest()
                candidates.append((rank, int(track_id), pixel))
        candidates.sort(key=lambda x: x[0])
        if len(candidates) < cfg["queries_per_camera"]:
            raise RuntimeError(
                "NOT_ENOUGH_VISIBLE_FIRST_FRAME_QUERIES:"
                + item["sequence"] + ":cam" + str(camera_id) + ":" + str(len(candidates))
            )
        for _, track_id, pixel in candidates[:cfg["queries_per_camera"]]:
            qid = f"q{query_slot:02d}"
            query_slot += 1
            queries.append({
                "id": qid,
                "camera_id": camera_id,
                "query_frame": 0,
                "query_xy": [float(pixel[0]), float(pixel[1])],
                "target_frame": int(target_frame),
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
    query_track_ids = np.zeros(qcount, dtype=np.uint32)
    query_camera_ids = np.zeros(qcount, dtype=np.uint8)
    query_depth_m = np.zeros(qcount, dtype=np.float32)
    query_target_frames = np.full(qcount, target_frame, dtype=np.int32)
    for qi, (qid, camera_id, track_id, query_pixel) in enumerate(query_private):
        camera = raw["cameras"][camera_id]
        width, height = resolutions[camera_id]
        query_track_ids[qi] = track_id
        query_camera_ids[qi] = camera_id
        initial_point = raw["track_maps"][0][track_id]
        initial_rotation = camera["rotation"][0]
        initial_translation = camera["translation"][0]
        _, initial_depth = project_point(
            initial_point, initial_rotation, initial_translation, camera["intrinsics"]
        )
        query_depth_m[qi] = initial_depth
        for frame_id in range(nframes):
            rotation = camera["rotation"][frame_id]
            translation = camera["translation"][frame_id]
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
    )
    return {
        "name": scene_id,
        "sequence": item["sequence"],
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
    if source_manifest["source_index_sha256"] != cfg["source_index_sha256"]:
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
        if source_record is None or source_record["split"] != item["split"]:
            raise RuntimeError("SOURCE_SEQUENCE_SPLIT_MISMATCH")
        scene = extract_scene(item, source_record, slot, cfg)
        scenes.append({k: v for k, v in scene.items() if k != "queries"})
        public_queries[scene["name"]] = scene["queries"]
        print("TAPVIDMV_SCENE_EXTRACTED", scene["name"], scene["split"],
              scene["sequence"], scene["nframes"], scene["query_count"], flush=True)
    public_data = json_bytes(public_queries)
    write_once(FROZEN / "queries-public.json", public_data)
    manifest = {
        "version": cfg["version"],
        "dataset": cfg["dataset"],
        "source_index_sha256": cfg["source_index_sha256"],
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "selection_salt": cfg["selection_salt"],
        "split_contract": "4 DEV + 8 TEST, distinct collection-group prefixes",
        "queries_per_camera": int(cfg["queries_per_camera"]),
        "query_contract": (
            f"hash-ranked frame-0 track IDs; {cfg['queries_per_camera']} "
            "depth-visible query points per camera"
        ),
        "future_oracle_private": True,
        "depth_visibility": cfg["depth_visibility"],
        "camera_projection": cfg["camera_projection"],
        "rrd_version": cfg["rrd_version"],
        "source_license": cfg["source_license"],
        "public_queries_sha256": sha256_bytes(public_data),
        "scenes": scenes,
        "limitations": [
            "Rerun viewer recordings subsample point tracks for display; this is not the full released track array.",
            "Per-view visibility is inferred from exact depth point-cloud snapshots, not a separate official binary occlusion annotation.",
            "Depth visibility is unknown on frames without an exact cloud snapshot.",
        ],
    }
    write_once(FROZEN / "manifest.json", json_bytes(manifest))
    write_manifest_errata()
    audit()
    print("TAPVIDMV_PREPARED", len(scenes), "scenes", "queries",
          sum(x["query_count"] for x in scenes), flush=True)


def audit_public() -> dict:
    """Verify only public frames and queries; never open private oracle arrays."""
    cfg = config()
    source_manifest_path = SOURCE / "source_manifest.json"
    manifest_path = FROZEN / "manifest.json"
    query_path = FROZEN / "queries-public.json"
    if not (source_manifest_path.is_file() and manifest_path.is_file() and query_path.is_file()):
        raise RuntimeError("TAPVIDMV_FROZEN_MANIFEST_MISSING")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    queries = json.loads(query_path.read_text(encoding="utf-8"))
    errata_path = FROZEN / "manifest-errata.json"
    if not errata_path.is_file():
        raise RuntimeError("TAPVIDMV_MANIFEST_ERRATA_MISSING_RUN_CORRECT_MANIFEST")
    errata = json.loads(errata_path.read_text(encoding="utf-8"))
    expected_errata = build_manifest_errata(
        manifest, queries, cfg, sha256_file(manifest_path), sha256_file(query_path)
    )
    if errata != expected_errata:
        raise RuntimeError("TAPVIDMV_MANIFEST_ERRATA_DRIFT")
    manifest = dict(manifest)
    manifest["queries_per_camera"] = errata["corrected_queries_per_camera"]
    manifest["query_contract"] = errata["corrected_query_contract"]
    manifest["manifest_errata_sha256"] = sha256_file(errata_path)
    if manifest["source_index_sha256"] != cfg["source_index_sha256"]:
        raise RuntimeError("TAPVIDMV_INDEX_SHA_DRIFT")
    if manifest["source_manifest_sha256"] != sha256_file(source_manifest_path):
        raise RuntimeError("TAPVIDMV_SOURCE_MANIFEST_DRIFT")
    if manifest["public_queries_sha256"] != sha256_file(query_path):
        raise RuntimeError("TAPVIDMV_PUBLIC_QUERY_HASH_DRIFT")
    if len(manifest["scenes"]) != cfg["dev_groups"] + cfg["test_groups"]:
        raise RuntimeError("TAPVIDMV_SCENE_COUNT_DRIFT")
    groups = [x["group"] for x in manifest["scenes"]]
    if len(groups) != len(set(groups)):
        raise RuntimeError("TAPVIDMV_GROUP_LEAKAGE")
    expected_splits = ["dev"] * cfg["dev_groups"] + ["test"] * cfg["test_groups"]
    if [x["split"] for x in manifest["scenes"]] != expected_splits:
        raise RuntimeError("TAPVIDMV_SPLIT_ORDER_DRIFT")
    source_by_sequence = {x["sequence"]: x for x in source_manifest["sequences"]}
    for scene in manifest["scenes"]:
        if scene["sequence"] not in source_by_sequence:
            raise RuntimeError("TAPVIDMV_SCENE_NOT_IN_SOURCE_MANIFEST")
        source_record = source_by_sequence[scene["sequence"]]
        if source_record["rrd_sha256"] != scene["rrd_sha256"]:
            raise RuntimeError("TAPVIDMV_RRD_HASH_DRIFT")
        for camera_id, relative in scene["frame_files"].items():
            frame_path = FROZEN / relative
            if sha256_file(frame_path) != scene["frame_sha256"][camera_id]:
                raise RuntimeError("TAPVIDMV_FRAME_HASH_DRIFT:" + relative)
            frames = np.load(frame_path, mmap_mode="r", allow_pickle=False)
            width, height = scene["resolution_by_camera"][camera_id]
            if frames.shape != (scene["nframes"], height, width, 3) or frames.dtype != np.uint8:
                raise RuntimeError("TAPVIDMV_FRAME_SHAPE_DRIFT:" + relative)
        if scene["name"] not in queries or len(queries[scene["name"]]) != scene["query_count"]:
            raise RuntimeError("TAPVIDMV_QUERY_COUNT_DRIFT")
        for q in queries[scene["name"]]:
            if set(q) != {"id", "camera_id", "query_frame", "query_xy", "target_frame"}:
                raise RuntimeError("PUBLIC_QUERY_LEAKS_PRIVATE_FIELDS")
            x, y = q["query_xy"]
            width, height = scene["resolution_by_camera"][str(q["camera_id"])]
            if (q["query_frame"] != 0 or q["camera_id"] not in (0, 1, 2)
                    or not (0 <= x < width and 0 <= y < height)
                    or not (0 < q["target_frame"] < scene["nframes"])):
                raise RuntimeError("INVALID_TAPVIDMV_PUBLIC_QUERY")
    if set(queries) != {x["name"] for x in manifest["scenes"]}:
        raise RuntimeError("TAPVIDMV_PUBLIC_QUERY_SCENE_SET_DRIFT")
    print("TAPVIDMV_PUBLIC_AUDIT_PASS", len(manifest["scenes"]),
          "scenes", sum(x["query_count"] for x in manifest["scenes"]), "queries", flush=True)
    return manifest


def audit() -> dict:
    manifest = audit_public()
    for scene in manifest["scenes"]:
        oracle_path = FROZEN / scene["oracle_file"]
        if sha256_file(oracle_path) != scene["oracle_sha256"]:
            raise RuntimeError("TAPVIDMV_ORACLE_HASH_DRIFT:" + scene["name"])
    print("TAPVIDMV_FULL_AUDIT_PASS", len(manifest["scenes"]), "scenes", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("download", "prepare", "correct-manifest", "audit")
    )
    args = parser.parse_args()
    if args.action == "download":
        download()
    elif args.action == "prepare":
        prepare()
    elif args.action == "correct-manifest":
        errata = write_manifest_errata()
        print("TAPVIDMV_MANIFEST_ERRATA_WRITTEN", json.dumps(errata, sort_keys=True), flush=True)
        audit_public()
    else:
        audit()


if __name__ == "__main__":
    main()
