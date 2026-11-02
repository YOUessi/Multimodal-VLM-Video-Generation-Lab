"""Batch45: isolated natural-video TAP-Vid-Kinetics point/visibility freeze.

The annotations are official TAP-Vid Kinetics CSV rows (250 samples/track).
Videos are decoded at 25 fps, truncated exactly as the upstream generator does,
then resized to 256x256 for the benchmark protocol. TEST oracle data is written
separately and is never opened by prediction code.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/tapvid_kinetics_batch45e"
SOURCE_CSV = ROOT / "data/tapvid_kinetics_batch41/tapvid_kinetics/tapvid_kinetics.csv"
VAL_METADATA = ROOT / "data/tapvid_kinetics_batch41/metadata/k700_2020_val.csv"
VAL_IDS = ROOT / "data/tapvid_kinetics_batch41/tapvid_kinetics/val.txt"
FROZEN = DATA / "batch45e_frozen"
PATH_LIST = DATA / "preflight/k700_2020_val_path.txt"
DOWNLOAD_MANIFEST = DATA / "preflight/download_manifest.json"
PATH_LIST_URL = "https://s3.amazonaws.com/kinetics/700_2020/val/k700_2020_val_path.txt"
SALT = "MVVG-BATCH45-KINETICS-FRESH-CLASS-DISJOINT-20261102"
DEV_VIDEOS = 6
TEST_VIDEOS = 6
QUERIES_PER_VIDEO = 4
FPS = 25
MAX_FRAMES = 250
HEIGHT = 256
WIDTH = 256
ROW_COLUMNS = 3 + 3 * 250


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()


def atomic_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError("FROZEN_FILE_CONFLICT:" + str(path))
        return
    with path.open("xb") as f:
        f.write(payload)


def load_selection() -> dict:
    path = DATA / "preflight/selection.json"
    selection = json.loads(path.read_text(encoding="utf-8"))
    if len(selection.get("videos", [])) != DEV_VIDEOS + TEST_VIDEOS:
        raise RuntimeError("KINETICS_SELECTION_COUNT_MISMATCH")
    if [x.get("split") for x in selection["videos"]] != (
        ["dev"] * DEV_VIDEOS + ["test"] * TEST_VIDEOS
    ):
        raise RuntimeError("KINETICS_SELECTION_SPLIT_ORDER_MISMATCH")
    if len({x["video_id"] for x in selection["videos"]}) != len(selection["videos"]):
        raise RuntimeError("KINETICS_DUPLICATE_VIDEO_ID")
    if len({x["label"] for x in selection["videos"]}) != len(selection["videos"]):
        raise RuntimeError("KINETICS_ACTION_CLASS_NOT_UNIQUE")
    return selection


def verify_sources(selection: dict) -> None:
    expected = {
        SOURCE_CSV: selection["source_csv_sha256"],
        VAL_METADATA: selection["kinetics_val_metadata_sha256"],
        VAL_IDS: selection["val_ids_sha256"],
        PATH_LIST: selection["path_list_sha256"],
    }
    for path, digest in expected.items():
        if not path.is_file() or sha_file(path) != digest:
            raise RuntimeError("KINETICS_SOURCE_SHA_MISMATCH:" + str(path))


def selected_csv_rows(selection: dict) -> dict[str, list[list[str]]]:
    selected = {x["video_id"]: x for x in selection["videos"]}
    result: dict[str, list[list[str]]] = {key: [] for key in selected}
    with SOURCE_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0] not in selected:
                continue
            if len(row) != ROW_COLUMNS:
                raise RuntimeError("KINETICS_CSV_ROW_WIDTH:" + str(len(row)))
            result[row[0]].append(row)
    for video_id, meta in selected.items():
        rows = result[video_id]
        if len(rows) != int(meta["track_rows"]):
            raise RuntimeError(
                f"KINETICS_TRACK_ROW_COUNT:{video_id}:{len(rows)}!={meta['track_rows']}"
            )
        expected_key = (str(meta["time_start"]), str(meta["time_end"]))
        for row in rows:
            if (row[1], row[2]) != expected_key:
                raise RuntimeError("KINETICS_VIDEO_TIME_KEY_MISMATCH:" + video_id)
    return result


def parse_track_row(row: list[str]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(row[3:], dtype=np.float64).reshape(250, 3)
    xy = raw[:, :2]
    occ_raw = raw[:, 2]
    if not np.isfinite(raw).all() or not np.isin(occ_raw, [0.0, 1.0]).all():
        raise RuntimeError("KINETICS_TRACK_NONFINITE_OR_BAD_OCCLUSION")
    occluded = occ_raw.astype(bool)
    visible_xy = xy[~occluded]
    if visible_xy.size and (
        np.any(visible_xy < 0.0) or np.any(visible_xy > 1.0)
    ):
        raise RuntimeError("KINETICS_VISIBLE_COORDINATE_NOT_NORMALIZED")
    return xy, occluded


def normalized_to_resized_pixels(
    points: np.ndarray, original_width: int, original_height: int
) -> np.ndarray:
    """Match upstream generate_tapvid.py's half-pixel correction then 256 resize."""
    if points.ndim != 2 or points.shape[1] != 2 or original_width <= 0 or original_height <= 0:
        raise ValueError("KINETICS_COORDINATE_SHAPE_INVALID")
    scale = np.asarray([WIDTH / original_width, HEIGHT / original_height], dtype=np.float64)
    return np.asarray(points, dtype=np.float64) * np.asarray([WIDTH, HEIGHT]) - 0.5 * scale


def first_visible_query(track_index: int, xy: np.ndarray, occluded: np.ndarray,
                        width: int, height: int) -> dict | None:
    visible = np.flatnonzero(~occluded)
    if not len(visible):
        return None
    t = int(visible[0])
    point = normalized_to_resized_pixels(xy[t:t + 1], width, height)[0]
    # Upstream TAP-Vid applies a half-pixel correction; edge points may be -0.1 px.
    if not np.isfinite(point).all() or not (-0.5 <= point[0] < WIDTH and -0.5 <= point[1] < HEIGHT):
        raise RuntimeError("KINETICS_QUERY_POINT_OUT_OF_FRAME")
    return {
        "id": f"p{track_index:04d}",
        "track_index": int(track_index),
        "query_time": t,
        "query_xy": [round(float(point[0]), 5), round(float(point[1]), 5)],
        "query_protocol": "official_first_visible_point",
    }


def probe_video(path: Path) -> dict:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,duration,nb_frames",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True, timeout=60,
    )
    streams = json.loads(proc.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError("KINETICS_VIDEO_STREAM_COUNT")
    stream = streams[0]
    width, height = int(stream["width"]), int(stream["height"])
    if width < 256 or height < 144:
        raise RuntimeError("KINETICS_SOURCE_RESOLUTION_TOO_SMALL")
    return {
        "width": width,
        "height": height,
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "duration_seconds": float(stream.get("duration", 0.0)),
        "nb_frames": int(stream["nb_frames"]) if stream.get("nb_frames", "").isdigit() else None,
    }


def decode_frames(path: Path) -> np.ndarray:
    """Use the official 25-fps sampling rule; scale to TAP-Vid's 256x256 reader size."""
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
            "-an", "-sn", "-dn", "-vf",
            f"fps={FPS},scale={WIDTH}:{HEIGHT}:flags=area",
            "-frames:v", str(MAX_FRAMES), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ],
        capture_output=True, check=True, timeout=300,
    )
    frame_bytes = HEIGHT * WIDTH * 3
    if not proc.stdout or len(proc.stdout) % frame_bytes:
        raise RuntimeError("KINETICS_FFMPEG_RAW_FRAME_BYTES_INVALID")
    count = len(proc.stdout) // frame_bytes
    if count < 2 or count > MAX_FRAMES:
        raise RuntimeError("KINETICS_DECODED_FRAME_COUNT_INVALID:" + str(count))
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(count, HEIGHT, WIDTH, 3).copy()


def archive_member_sha(meta: dict) -> str:
    shard = DATA / "shards" / f"k700_val_{int(meta['shard_index']):03d}.tar.gz"
    member_name = meta["archive_member"]
    with tarfile.open(shard, "r:gz") as archive:
        member = archive.getmember(member_name)
        if not member.isfile():
            raise RuntimeError("KINETICS_ARCHIVE_MEMBER_NOT_REGULAR_FILE")
        stream = archive.extractfile(member)
        if stream is None:
            raise RuntimeError("KINETICS_ARCHIVE_MEMBER_UNREADABLE")
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
        return h.hexdigest()


def extract_member(meta: dict, destination: Path) -> None:
    shard = DATA / "shards" / f"k700_val_{int(meta['shard_index']):03d}.tar.gz"
    member_name = meta["archive_member"]
    if Path(member_name).is_absolute() or ".." in Path(member_name).parts:
        raise RuntimeError("KINETICS_ARCHIVE_MEMBER_PATH_TRAVERSAL")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha_file(destination) != archive_member_sha(meta):
            raise RuntimeError("KINETICS_EXTRACTED_VIDEO_MEMBER_SHA_MISMATCH")
        return
    tmp = destination.with_suffix(destination.suffix + ".partial")
    if tmp.exists():
        raise RuntimeError("KINETICS_PARTIAL_EXTRACTION_PRESERVED:" + str(tmp))
    with tarfile.open(shard, "r:gz") as archive:
        member = archive.getmember(member_name)
        if not member.isfile():
            raise RuntimeError("KINETICS_ARCHIVE_MEMBER_NOT_REGULAR_FILE")
        stream = archive.extractfile(member)
        if stream is None:
            raise RuntimeError("KINETICS_ARCHIVE_MEMBER_UNREADABLE")
        with tmp.open("xb") as output:
            shutil.copyfileobj(stream, output)
    os.replace(tmp, destination)
    if sha_file(destination) != archive_member_sha(meta):
        raise RuntimeError("KINETICS_EXTRACTED_VIDEO_MEMBER_SHA_MISMATCH")



def gitsha() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def manifest_provenance(selection: dict, source_commit: str) -> dict:
    return {
        "path_list_sha256": sha_file(PATH_LIST),
        "download_manifest_sha256": sha_file(DOWNLOAD_MANIFEST),
        "previous_video_ids_sha256": selection["previous_video_ids_sha256"],
        "source_commit": source_commit,
    }


def validate_split_disjointness(scenes: list[dict], previous_ids: set[str],
                                previous_labels: set[str]) -> bool:
    video_ids = [x["video_id"] for x in scenes]
    labels = [x["action_class"] for x in scenes]
    if len(set(video_ids)) != len(video_ids):
        raise RuntimeError("KINETICS_SCENE_ID_OVERLAP")
    if len(set(labels)) != len(labels):
        raise RuntimeError("KINETICS_ACTION_CLASS_OVERLAP")
    if set(video_ids) & previous_ids:
        raise RuntimeError("KINETICS_PREVIOUS_VIDEO_ID_OVERLAP")
    if set(labels) & previous_labels:
        raise RuntimeError("KINETICS_BATCH42_ACTION_CLASS_OVERLAP")
    return True



def prepare() -> None:
    selection = load_selection()
    verify_sources(selection)
    rows_by_video = selected_csv_rows(selection)
    target = FROZEN
    manifest_path = target / "manifest.json"
    if manifest_path.exists():
        audit()
        print("KINETICS_BATCH45_FROZEN_REUSED", flush=True)
        return

    public: dict[str, list] = {}
    private: dict[str, dict] = {}
    scenes = []
    for slot, meta in enumerate(selection["videos"]):
        video_id = meta["video_id"]
        scene = f"kinetics_{slot:02d}"
        video_path = DATA / "videos" / f"{video_id}.mp4"
        extract_member(meta, video_path)
        video_sha = sha_file(video_path)
        if video_sha != archive_member_sha(meta):
            raise RuntimeError("KINETICS_EXTRACTED_VIDEO_MEMBER_SHA_MISMATCH")
        probe = probe_video(video_path)
        frames = decode_frames(video_path)
        frame_path = target / "frames" / f"{scene}.npy"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        if frame_path.exists():
            old = np.load(frame_path, mmap_mode="r", allow_pickle=False)
            if old.shape != frames.shape or not np.array_equal(old, frames):
                raise RuntimeError("KINETICS_FROZEN_FRAME_CONFLICT:" + scene)
        else:
            with frame_path.open("xb") as f:
                np.save(f, frames, allow_pickle=False)

        all_tracks = []
        for track_index, row in enumerate(rows_by_video[video_id]):
            xy_norm, occluded = parse_track_row(row)
            n = len(frames)
            xy = normalized_to_resized_pixels(xy_norm[:n], probe["width"], probe["height"])
            occ = occluded[:n]
            visible_xy = xy[~occ]
            if visible_xy.size and (
                np.any(visible_xy < -0.5) or np.any(visible_xy[:, 0] >= WIDTH)
                or np.any(visible_xy[:, 1] >= HEIGHT)
            ):
                raise RuntimeError("KINETICS_VISIBLE_TRACK_OUT_OF_FRAME:" + video_id)
            all_tracks.append((track_index, xy, occ, xy_norm[:n]))

        candidates = []
        for track_index, xy, occ, xy_norm in all_tracks:
            query = first_visible_query(track_index, xy_norm, occ,
                                         probe["width"], probe["height"])
            if query is not None and query["query_time"] < len(frames) - 1:
                rank = hashlib.sha256(
                    f"{SALT}|{video_id}|{track_index}".encode()
                ).hexdigest()
                candidates.append((rank, query, xy, occ))
        candidates.sort(key=lambda item: item[0])
        selected = candidates[:QUERIES_PER_VIDEO]
        if len(selected) != QUERIES_PER_VIDEO:
            raise RuntimeError("KINETICS_NOT_ENOUGH_VALID_QUERY_TRACKS:" + video_id)

        queries = []
        gold_tracks = {}
        for _, query, xy, occ in selected:
            queries.append(query)
            gold_tracks[query["id"]] = {
                "xy": [[round(float(v), 5) for v in point] for point in xy],
                "occluded": [bool(v) for v in occ],
            }
        public[scene] = queries
        private[scene] = {"tracks": gold_tracks}
        scenes.append({
            "name": scene,
            "video_id": video_id,
            "action_class": meta["label"],
            "split": meta["split"],
            "archive_member": meta["archive_member"],
            "shard_index": int(meta["shard_index"]),
            "shard_sha256": sha_file(DATA / "shards" / f"k700_val_{int(meta['shard_index']):03d}.tar.gz"),
            "video_file": str(video_path.relative_to(ROOT)),
            "video_sha256": video_sha,
            "source_width": probe["width"],
            "source_height": probe["height"],
            "source_fps": probe["avg_frame_rate"],
            "source_duration_seconds": probe["duration_seconds"],
            "source_nb_frames": probe["nb_frames"],
            "frame_file": str(frame_path.relative_to(ROOT)),
            "frame_sha256": sha_file(frame_path),
            "frame_count": int(len(frames)),
            "frame_rate_sampling": FPS,
            "frame_resolution": [HEIGHT, WIDTH],
            "track_rows": len(all_tracks),
            "query_count": len(queries),
        })
        print("KINETICS_FROZEN_SCENE", scene, meta["split"], video_id,
              "frames", len(frames), "queries", len(queries), flush=True)

    selection_sha = sha_file(DATA / "preflight/selection.json")
    manifest = {
        "version": "BATCH45E-TAPVID-KINETICS-FRESH-CLASS-HOLDOUT-2026-11-02-v5",
        "dataset": "TAP-Vid-Kinetics; natural YouTube/Kinetics video clips",
        "annotation_license": "CC BY 4.0",
        "video_rights": "source videos remain local and retain their individual creator/platform terms",
        "selection_protocol": selection["protocol"],
        "selection_sha256": selection_sha,
        "source_csv_sha256": sha_file(SOURCE_CSV),
        "kinetics_val_metadata_sha256": sha_file(VAL_METADATA),
        "val_ids_sha256": sha_file(VAL_IDS),
        **manifest_provenance(selection, gitsha()),
        "selection_salt": SALT,
        "dev_videos": DEV_VIDEOS,
        "test_videos": TEST_VIDEOS,
        "queries_per_video": QUERIES_PER_VIDEO,
        "query_protocol": "official TAP-Vid first-visible query; deterministic hash selection of track ids",
        "frame_protocol": "ffmpeg fps=25, first 250 frames, 256x256 resize; shorter clips truncate tracks",
        "coordinate_protocol": "upstream half-pixel correction then scale to 256x256 raster coordinates",
        "oracle_sha256": sha_bytes(json_bytes(private)),
        "public_queries_sha256": sha_bytes(json_bytes(public)),
        "scenes": scenes,
    }
    atomic_once(target / "oracle-private.json", json_bytes(private))
    atomic_once(target / "queries-public.json", json_bytes(public))
    atomic_once(manifest_path, json_bytes(manifest))
    audit()
    print("KINETICS_BATCH45_PREPARED", len(scenes), "scenes",
          "source_csv_sha256", manifest["source_csv_sha256"], flush=True)


def rows_norm_for_query(row: list[str]) -> np.ndarray:
    raw = np.asarray(row[3:], dtype=np.float64).reshape(250, 3)
    return raw[:, :2]


def rows_for_query(row: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return parse_track_row(row)


def rows_norm_for_query_legacy(row: list[str]) -> np.ndarray:
    return rows_norm_for_query(row)


def audit() -> dict:
    selection = load_selection()
    verify_sources(selection)
    manifest = json.loads((FROZEN / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("source_commit") != gitsha():
        raise RuntimeError("KINETICS_SOURCE_COMMIT_DRIFT")
    public_path = FROZEN / "queries-public.json"
    oracle_path = FROZEN / "oracle-private.json"
    if manifest.get("selection_sha256") != sha_file(DATA / "preflight/selection.json"):
        raise RuntimeError("KINETICS_SELECTION_SHA_DRIFT")
    if manifest.get("source_csv_sha256") != sha_file(SOURCE_CSV):
        raise RuntimeError("KINETICS_CSV_SHA_DRIFT")
    if manifest.get("kinetics_val_metadata_sha256") != sha_file(VAL_METADATA):
        raise RuntimeError("KINETICS_METADATA_SHA_DRIFT")
    if manifest.get("val_ids_sha256") != sha_file(VAL_IDS):
        raise RuntimeError("KINETICS_VAL_IDS_SHA_DRIFT")
    if manifest.get("path_list_sha256") != sha_file(PATH_LIST):
        raise RuntimeError("KINETICS_PATH_LIST_SHA_DRIFT")
    if manifest.get("download_manifest_sha256") != sha_file(DOWNLOAD_MANIFEST):
        raise RuntimeError("KINETICS_DOWNLOAD_MANIFEST_SHA_DRIFT")
    if manifest.get("oracle_sha256") != sha_file(oracle_path):
        raise RuntimeError("KINETICS_PRIVATE_ORACLE_SHA_DRIFT")
    if manifest.get("public_queries_sha256") != sha_file(public_path):
        raise RuntimeError("KINETICS_PUBLIC_QUERY_SHA_DRIFT")
    if len(manifest.get("scenes", [])) != DEV_VIDEOS + TEST_VIDEOS:
        raise RuntimeError("KINETICS_SCENE_COUNT_DRIFT")
    if [s["split"] for s in manifest["scenes"]] != (
        ["dev"] * DEV_VIDEOS + ["test"] * TEST_VIDEOS
    ):
        raise RuntimeError("KINETICS_SPLIT_DRIFT")
    previous_ids = collect_previous_video_ids()
    batch42_manifest = ROOT / "data/tapvid_kinetics_batch42/batch42_frozen/manifest.json"
    old_labels = set()
    if batch42_manifest.is_file():
        old = json.loads(batch42_manifest.read_text(encoding="utf-8"))
        old_labels = {x.get("action_class") for x in old.get("scenes", []) if x.get("action_class")}
    validate_split_disjointness(manifest["scenes"], previous_ids, old_labels)
    public = json.loads(public_path.read_text(encoding="utf-8"))
    if set(public) != {s["name"] for s in manifest["scenes"]}:
        raise RuntimeError("KINETICS_PUBLIC_SCENE_SET_DRIFT")
    for scene in manifest["scenes"]:
        video = ROOT / scene["video_file"]
        frames = ROOT / scene["frame_file"]
        shard = DATA / "shards" / f"k700_val_{scene['shard_index']:03d}.tar.gz"
        if sha_file(video) != scene["video_sha256"] or sha_file(shard) != scene["shard_sha256"]:
            raise RuntimeError("KINETICS_SOURCE_MEDIA_SHA_DRIFT:" + scene["name"])
        selected_meta = next(x for x in selection["videos"] if x["video_id"] == scene["video_id"])
        if archive_member_sha(selected_meta) != scene["video_sha256"]:
            raise RuntimeError("KINETICS_ARCHIVE_MEMBER_CONTENT_DRIFT:" + scene["name"])
        if sha_file(frames) != scene["frame_sha256"]:
            raise RuntimeError("KINETICS_FRAME_SHA_DRIFT:" + scene["name"])
        arr = np.load(frames, mmap_mode="r", allow_pickle=False)
        if arr.shape != (scene["frame_count"], HEIGHT, WIDTH, 3) or arr.dtype != np.uint8:
            raise RuntimeError("KINETICS_FRAME_SHAPE_DRIFT:" + scene["name"])
        if len(public[scene["name"]]) != QUERIES_PER_VIDEO:
            raise RuntimeError("KINETICS_QUERY_COUNT_DRIFT:" + scene["name"])
        for query in public[scene["name"]]:
            if not (0 <= query["query_time"] < scene["frame_count"] - 1):
                raise RuntimeError("KINETICS_QUERY_TIME_INVALID")
            xy = np.asarray(query["query_xy"], dtype=np.float64)
            if (xy.shape != (2,) or not np.isfinite(xy).all()
                    or np.any(xy < -0.5) or np.any(xy >= [WIDTH, HEIGHT])):
                raise RuntimeError("KINETICS_QUERY_COORDINATE_INVALID")
    print("KINETICS_BATCH45_AUDIT_PASS", len(manifest["scenes"]), "scenes",
          "private_oracle_bytes_sha_only", flush=True)
    return manifest




def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(json_bytes(value))
    os.replace(tmp, path)


def ensure_path_list() -> list[str]:
    PATH_LIST.parent.mkdir(parents=True, exist_ok=True)
    if not PATH_LIST.is_file():
        tmp = PATH_LIST.with_suffix(".txt.partial")
        if tmp.exists():
            raise RuntimeError("KINETICS_PATH_LIST_PARTIAL_PRESERVED")
        proc = subprocess.run(
            ["curl", "-fL", "--retry", "3", "--connect-timeout", "30",
             "--max-time", "600", "-o", str(tmp), PATH_LIST_URL],
            capture_output=True, text=True, timeout=660,
        )
        if proc.returncode:
            raise RuntimeError("KINETICS_PATH_LIST_DOWNLOAD_FAILED:" + proc.stderr[-500:])
        os.replace(tmp, PATH_LIST)
    lines = [x.strip() for x in PATH_LIST.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(lines) != 700:
        raise RuntimeError("KINETICS_PATH_LIST_COUNT:" + str(len(lines)))
    for i, url in enumerate(lines, start=1):
        if f"k700_val_{i:03d}.tar.gz" not in url or not url.startswith("https://"):
            raise RuntimeError("KINETICS_PATH_LIST_ORDER_OR_URL_INVALID:" + str(i))
    return lines


def collect_previous_video_ids() -> set[str]:
    ids: set[str] = set()
    data_root = (ROOT / "data").resolve()
    batch45_data = DATA.resolve()
    for path in data_root.rglob("manifest.json"):
        try:
            if batch45_data in path.parents:
                continue
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        stack = [doc]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in ("video_id", "youtube_id") and isinstance(item, str):
                        ids.add(item)
                    if isinstance(item, (dict, list)):
                        stack.append(item)
            elif isinstance(value, list):
                stack.extend(value)
    return ids


def deterministic_class_selection(
    candidates: dict[str, list[dict]],
    all_labels: list[str],
    previous_ids: set[str],
    previous_labels: set[str],
    path_lines: list[str],
    source_hashes: dict,
) -> dict:
    labels = sorted(all_labels)
    if len(labels) != 700:
        raise RuntimeError("KINETICS_CLASS_COUNT_NOT_700:" + str(len(labels)))
    class_index = {label: i + 1 for i, label in enumerate(labels)}
    if len(path_lines) != 700:
        raise RuntimeError("KINETICS_PATH_LIST_COUNT_MISMATCH")
    eligible = []
    for label, items in candidates.items():
        if label in previous_labels:
            continue
        clean = [x for x in items if x["video_id"] not in previous_ids]
        if not clean:
            continue
        clean.sort(key=lambda x: hashlib.sha256(
            f"{SALT}|{label}|{x['video_id']}".encode()
        ).hexdigest())
        chosen = clean[0]
        shard_index = class_index[label]
        url = path_lines[shard_index - 1]
        if f"k700_val_{shard_index:03d}.tar.gz" not in url:
            raise RuntimeError("KINETICS_CLASS_TO_SHARD_MAPPING_INVALID:" + label)
        start, end = int(chosen["time_start"]), int(chosen["time_end"])
        eligible.append((
            hashlib.sha256(f"{SALT}|CLASS|{label}".encode()).hexdigest(),
            label,
            {
                **chosen,
                "split": "pending",
                "shard_index": shard_index,
                "archive_member": f"{label}/{chosen['video_id']}_{start:06d}_{end:06d}.mp4",
                "shard_url": url,
            },
        ))
    eligible.sort(key=lambda x: x[0])
    if len(eligible) < DEV_VIDEOS + TEST_VIDEOS:
        raise RuntimeError("KINETICS_NOT_ENOUGH_FRESH_CLASSES:" + str(len(eligible)))
    chosen = []
    for i, (_, label, item) in enumerate(eligible[:DEV_VIDEOS + TEST_VIDEOS]):
        item["split"] = "dev" if i < DEV_VIDEOS else "test"
        chosen.append(item)
    return {
        "protocol": (
            "TAP-Vid-Kinetics natural YouTube clips; one previously unused video per "
            "previously unused Kinetics action class; deterministic SHA-256 selection; "
            "6 DEV / 6 TEST; no video ID or action class overlap with Batch42"
        ),
        "selection_salt": SALT,
        "source_csv_sha256": source_hashes["source_csv_sha256"],
        "kinetics_val_metadata_sha256": source_hashes["kinetics_val_metadata_sha256"],
        "val_ids_sha256": source_hashes["val_ids_sha256"],
        "path_list_sha256": source_hashes["path_list_sha256"],
        "previous_video_ids_sha256": sha_bytes(("\n".join(sorted(previous_ids)) + "\n").encode()),
        "previous_video_id_count": len(previous_ids),
        "previous_action_classes_excluded": sorted(previous_labels),
        "videos": chosen,
    }


def build_selection(path_lines: list[str]) -> dict:
    selection_path = DATA / "preflight/selection.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        verify_sources(selection)
        return selection

    metadata: dict[str, dict] = {}
    with VAL_METADATA.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split") == "validate":
                metadata[row["youtube_id"]] = row
    all_labels = sorted({row["label"] for row in metadata.values()})
    if len(all_labels) != 700:
        raise RuntimeError("KINETICS_METADATA_CLASS_COUNT:" + str(len(all_labels)))

    previous_ids = collect_previous_video_ids()
    previous_labels: set[str] = set()
    batch42_manifest = ROOT / "data/tapvid_kinetics_batch42/batch42_frozen/manifest.json"
    if batch42_manifest.is_file():
        old = json.loads(batch42_manifest.read_text(encoding="utf-8"))
        previous_labels = {x.get("action_class") for x in old.get("scenes", []) if x.get("action_class")}

    stats: dict[str, dict] = {}
    with SOURCE_CSV.open("r", newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0] not in metadata or row[0] in previous_ids:
                continue
            if len(row) != ROW_COLUMNS:
                raise RuntimeError("KINETICS_CSV_ROW_WIDTH:" + str(len(row)))
            rec = stats.setdefault(row[0], {"track_rows": 0, "query_candidates": 0})
            rec["track_rows"] += 1
            raw = np.asarray(row[3:], dtype=np.float64).reshape(250, 3)
            if not np.isfinite(raw).all() or not np.isin(raw[:, 2], [0.0, 1.0]).all():
                continue
            visible = np.flatnonzero(raw[:MAX_FRAMES - 1, 2] == 0.0)
            if len(visible):
                xy = raw[int(visible[0]), :2]
                if np.isfinite(xy).all() and np.all(xy >= 0.0) and np.all(xy <= 1.0):
                    rec["query_candidates"] += 1

    candidates: dict[str, list[dict]] = {}
    for video_id, rec in stats.items():
        if rec["query_candidates"] < QUERIES_PER_VIDEO:
            continue
        meta = metadata[video_id]
        label = meta["label"]
        candidates.setdefault(label, []).append({
            "video_id": video_id,
            "label": label,
            "time_start": meta["time_start"],
            "time_end": meta["time_end"],
            "track_rows": rec["track_rows"],
            "query_candidates": rec["query_candidates"],
        })

    hashes = {
        "source_csv_sha256": sha_file(SOURCE_CSV),
        "kinetics_val_metadata_sha256": sha_file(VAL_METADATA),
        "val_ids_sha256": sha_file(VAL_IDS),
        "path_list_sha256": sha_file(PATH_LIST),
    }
    selection = deterministic_class_selection(
        candidates, all_labels, previous_ids, previous_labels, path_lines, hashes
    )
    atomic_once(selection_path, json_bytes(selection))
    print("KINETICS_BATCH45_SELECTION_FROZEN", len(selection["videos"]),
          "classes", "dev", DEV_VIDEOS, "test", TEST_VIDEOS,
          "sha256", sha_file(selection_path), flush=True)
    return selection


# One shard gets a bounded total budget, subdivided into resumable attempts.
# Never discard a failed download: previous .partial bytes are preserved.
SHARD_BUDGET_SECONDS = 2460
SHARD_CURL_SLICE_SECONDS = 420
SHARD_MAX_ATTEMPTS = 16
SHARD_STALL_SECONDS = 75
SHARD_MIN_BYTES_PER_SECOND = 2048


def remote_shard_identity(url: str) -> tuple[int, str]:
    """Freeze the immutable S3 object's size and ETag while resuming byte ranges."""
    with urlopen(Request(url, method="HEAD"), timeout=25) as response:
        length = response.headers.get("Content-Length")
        etag = response.headers.get("ETag")
        ranges = response.headers.get("Accept-Ranges", "").lower()
        if response.status != 200 or not length or not etag or ranges != "bytes":
            raise RuntimeError("KINETICS_SHARD_REMOTE_IDENTITY_MISSING")
        size = int(length)
        if size <= 0:
            raise RuntimeError("KINETICS_SHARD_REMOTE_SIZE_INVALID")
        return size, etag


def verify_complete_gzip(path: Path) -> None:
    """Read the entire compressed archive, validating gzip CRC and trailer."""
    try:
        with gzip.open(path, "rb") as stream:
            for _ in iter(lambda: stream.read(1024 * 1024), b""):
                pass
    except (OSError, EOFError, ValueError) as exc:
        raise RuntimeError(
            "KINETICS_SHARD_GZIP_INTEGRITY_FAILED:" + path.name
        ) from exc


def download_one_shard(shard: Path, url: str) -> tuple[int, str]:
    """Bounded retries with *persistent* partial bytes and fail-closed integrity.

    A failed attempt may append bytes, but it never publishes a final .tar.gz.
    curl -C - is used ONLY when a nonempty partial already exists. Verified
    content length and ETag prevent silent acceptance of truncated/mixed data.
    """
    size, etag = remote_shard_identity(url)
    if shard.is_file():
        if shard.stat().st_size != size:
            raise RuntimeError("KINETICS_SHARD_FINAL_SIZE_MISMATCH:" + shard.name)
        return size, etag

    tmp = shard.with_suffix(".tar.gz.partial")  # keep prior batch's exact path
    if tmp.is_file() and tmp.stat().st_size > size:
        raise RuntimeError("KINETICS_SHARD_PARTIAL_OVERSIZE:" + tmp.name)
    started = time.monotonic()
    last_error = ""
    consecutive_stalls = 0
    for attempt in range(1, SHARD_MAX_ATTEMPTS + 1):
        budget_left = SHARD_BUDGET_SECONDS - (time.monotonic() - started)
        if budget_left < 30:
            break
        before = tmp.stat().st_size if tmp.is_file() else 0
        if before == size:
            # curl might have received the final bytes just before a timeout.
            # Verify the entire archive rather than abandoning a valid partial.
            if remote_shard_identity(url) != (size, etag):
                raise RuntimeError("KINETICS_SHARD_REMOTE_IDENTITY_DRIFT:" + shard.name)
            verify_complete_gzip(tmp)
            os.replace(tmp, shard)
            return size, etag
        max_seconds = min(SHARD_CURL_SLICE_SECONDS, int(budget_left) - 20)
        command = [
            "curl", "-fL", "--retry", "0", "--connect-timeout", "25",
            "--max-time", str(max_seconds), "--speed-limit",
            str(SHARD_MIN_BYTES_PER_SECOND), "--speed-time",
            str(SHARD_STALL_SECONDS),
        ]
        if before:
            command += ["-C", "-"]
        command += ["-o", str(tmp), url]
        try:
            proc = subprocess.run(
                command, capture_output=True, text=True, timeout=max_seconds + 12
            )
            code = proc.returncode
            last_error = proc.stderr[-350:]
        except subprocess.TimeoutExpired as exc:
            code = 28
            last_error = "subprocess_timeout:" + str(exc.timeout)
        after = tmp.stat().st_size if tmp.is_file() else 0
        if after > size or after < before:
            raise RuntimeError(
                f"KINETICS_SHARD_RESUME_SIZE_INVALID:{shard.name}:{before}->{after}/{size}"
            )
        if code == 0:
            if after != size:
                raise RuntimeError(
                    f"KINETICS_SHARD_TRUNCATED_SUCCESS:{shard.name}:{after}/{size}"
                )
            if remote_shard_identity(url) != (size, etag):
                raise RuntimeError("KINETICS_SHARD_REMOTE_IDENTITY_DRIFT:" + shard.name)
            verify_complete_gzip(tmp)
            os.replace(tmp, shard)
            return size, etag
        # Only transient connection/partial-transfer/timeout errors are retryable.
        if code not in (7, 18, 28, 52, 55, 56):
            raise RuntimeError(
                f"KINETICS_SHARD_NONRETRYABLE:{shard.name}:rc={code}:{last_error}"
            )
        print(
            f"KINETICS_SHARD_RESUME_PROGRESS {shard.name} "
            f"attempt={attempt} rc={code} bytes={after}/{size}",
            flush=True,
        )
        # Three *consecutive* zero-progress attempts signal a dead source.
        # Keep all bytes and fail without burning another 40 minutes spinning.
        consecutive_stalls = consecutive_stalls + 1 if after == before else 0
        if consecutive_stalls >= 3:
            raise RuntimeError(
                f"KINETICS_SHARD_NO_PROGRESS:{shard.name}:{last_error}"
            )
    raise RuntimeError(
        f"KINETICS_SHARD_RETRY_EXHAUSTED:{shard.name}:"
        f"{tmp.stat().st_size if tmp.is_file() else 0}/{size}:{last_error}"
    )


def download_shards(selection: dict) -> dict:
    DATA.joinpath("shards").mkdir(parents=True, exist_ok=True)
    current = {"version": "BATCH45-SHARD-DOWNLOADS-v1", "shards": {}}
    if DOWNLOAD_MANIFEST.is_file():
        current = json.loads(DOWNLOAD_MANIFEST.read_text(encoding="utf-8"))
    for item in selection["videos"]:
        index = int(item["shard_index"])
        key = f"{index:03d}"
        shard = DATA / "shards" / f"k700_val_{key}.tar.gz"
        size, etag = download_one_shard(shard, item["shard_url"])
        try:
            with tarfile.open(shard, "r:gz") as archive:
                member = archive.getmember(item["archive_member"])
                if not member.isfile():
                    raise RuntimeError("KINETICS_ARCHIVE_MEMBER_NOT_REGULAR_FILE")
        except Exception as exc:
            raise RuntimeError("KINETICS_SHARD_OR_MEMBER_INVALID:" + key + ":" + repr(exc))
        digest = sha_file(shard)
        old = current["shards"].get(key)
        if old and (
            old.get("sha256") != digest
            or old.get("url") != item["shard_url"]
            or old.get("archive_member") != item["archive_member"]
        ):
            raise RuntimeError("KINETICS_SHARD_PIN_CONFLICT:" + key)
        current["shards"][key] = {
            "url": item["shard_url"],
            "sha256": digest,
            "archive_member": item["archive_member"],
            "content_length": size,
            "etag": etag,
        }
        write_json_atomic(DOWNLOAD_MANIFEST, current)
        print("KINETICS_BATCH45_SHARD_READY", key, digest, flush=True)
    return current


def download() -> None:
    path_lines = ensure_path_list()
    selection = build_selection(path_lines)
    manifest = download_shards(selection)
    print("KINETICS_BATCH45_DOWNLOAD_READY", len(manifest["shards"]),
          "shards", "selection", sha_file(DATA / "preflight/selection.json"),
          "download_manifest", sha_file(DOWNLOAD_MANIFEST), flush=True)


def audit_public() -> dict:
    selection = load_selection()
    verify_sources(selection)
    manifest = json.loads((FROZEN / "manifest.json").read_text(encoding="utf-8"))
    public_path = FROZEN / "queries-public.json"
    if manifest.get("selection_sha256") != sha_file(DATA / "preflight/selection.json"):
        raise RuntimeError("KINETICS_SELECTION_SHA_DRIFT")
    if manifest.get("source_csv_sha256") != sha_file(SOURCE_CSV):
        raise RuntimeError("KINETICS_CSV_SHA_DRIFT")
    if manifest.get("kinetics_val_metadata_sha256") != sha_file(VAL_METADATA):
        raise RuntimeError("KINETICS_METADATA_SHA_DRIFT")
    if manifest.get("val_ids_sha256") != sha_file(VAL_IDS):
        raise RuntimeError("KINETICS_VAL_IDS_SHA_DRIFT")
    if manifest.get("path_list_sha256") != sha_file(PATH_LIST):
        raise RuntimeError("KINETICS_PATH_LIST_SHA_DRIFT")
    if manifest.get("download_manifest_sha256") != sha_file(DOWNLOAD_MANIFEST):
        raise RuntimeError("KINETICS_DOWNLOAD_MANIFEST_SHA_DRIFT")
    if manifest.get("public_queries_sha256") != sha_file(public_path):
        raise RuntimeError("KINETICS_PUBLIC_QUERY_SHA_DRIFT")
    if len(manifest.get("scenes", [])) != DEV_VIDEOS + TEST_VIDEOS:
        raise RuntimeError("KINETICS_SCENE_COUNT_DRIFT")
    if [s["split"] for s in manifest["scenes"]] != ["dev"] * DEV_VIDEOS + ["test"] * TEST_VIDEOS:
        raise RuntimeError("KINETICS_SPLIT_DRIFT")
    if len({s["video_id"] for s in manifest["scenes"]}) != len(manifest["scenes"]):
        raise RuntimeError("KINETICS_SCENE_ID_OVERLAP")
    if len({s["action_class"] for s in manifest["scenes"]}) != len(manifest["scenes"]):
        raise RuntimeError("KINETICS_ACTION_CLASS_OVERLAP")
    public = json.loads(public_path.read_text(encoding="utf-8"))
    if set(public) != {s["name"] for s in manifest["scenes"]}:
        raise RuntimeError("KINETICS_PUBLIC_SCENE_SET_DRIFT")
    download_manifest = json.loads(DOWNLOAD_MANIFEST.read_text(encoding="utf-8"))
    for scene in manifest["scenes"]:
        video = ROOT / scene["video_file"]
        frames = ROOT / scene["frame_file"]
        shard = DATA / "shards" / f"k700_val_{scene['shard_index']:03d}.tar.gz"
        if sha_file(video) != scene["video_sha256"] or sha_file(shard) != scene["shard_sha256"]:
            raise RuntimeError("KINETICS_SOURCE_MEDIA_SHA_DRIFT:" + scene["name"])
        if download_manifest["shards"].get(f"{scene['shard_index']:03d}", {}).get("sha256") != scene["shard_sha256"]:
            raise RuntimeError("KINETICS_DOWNLOAD_PIN_DRIFT:" + scene["name"])
        selected_meta = next(x for x in selection["videos"] if x["video_id"] == scene["video_id"])
        if archive_member_sha(selected_meta) != scene["video_sha256"]:
            raise RuntimeError("KINETICS_ARCHIVE_MEMBER_CONTENT_DRIFT:" + scene["name"])
        if sha_file(frames) != scene["frame_sha256"]:
            raise RuntimeError("KINETICS_FRAME_SHA_DRIFT:" + scene["name"])
        arr = np.load(frames, mmap_mode="r", allow_pickle=False)
        if arr.shape != (scene["frame_count"], HEIGHT, WIDTH, 3) or arr.dtype != np.uint8:
            raise RuntimeError("KINETICS_FRAME_SHAPE_DRIFT:" + scene["name"])
        if len(public[scene["name"]]) != QUERIES_PER_VIDEO:
            raise RuntimeError("KINETICS_QUERY_COUNT_DRIFT:" + scene["name"])
        for query in public[scene["name"]]:
            if not (0 <= query["query_time"] < scene["frame_count"] - 1):
                raise RuntimeError("KINETICS_QUERY_TIME_INVALID")
            xy = np.asarray(query["query_xy"], dtype=np.float64)
            if xy.shape != (2,) or not np.isfinite(xy).all() or np.any(xy < -0.5) or np.any(xy >= [WIDTH, HEIGHT]):
                raise RuntimeError("KINETICS_QUERY_COORDINATE_INVALID")
    print("KINETICS_BATCH45_PUBLIC_AUDIT_PASS", len(manifest["scenes"]), "scenes", flush=True)
    return manifest


def audit_full() -> dict:
    manifest = audit_public()
    oracle_path = FROZEN / "oracle-private.json"
    if manifest.get("oracle_sha256") != sha_file(oracle_path):
        raise RuntimeError("KINETICS_PRIVATE_ORACLE_SHA_DRIFT")
    print("KINETICS_BATCH45_FULL_AUDIT_PASS", len(manifest["scenes"]), "scenes", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("download", "prepare", "audit-public", "audit"))
    action = parser.parse_args().action
    if action == "download":
        download()
    elif action == "prepare":
        prepare()
    elif action == "audit-public":
        audit_public()
    else:
        audit_full()


if __name__ == "__main__":
    main()
