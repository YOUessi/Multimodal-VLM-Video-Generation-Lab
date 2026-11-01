"""Batch42: isolated natural-video TAP-Vid-Kinetics point/visibility freeze.

The annotations are official TAP-Vid Kinetics CSV rows (250 samples/track).
Videos are decoded at 25 fps, truncated exactly as the upstream generator does,
then resized to 256x256 for the benchmark protocol. TEST oracle data is written
separately and is never opened by prediction code.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/tapvid_kinetics_batch42"
SOURCE_CSV = ROOT / "data/tapvid_kinetics_batch41/tapvid_kinetics/tapvid_kinetics.csv"
VAL_METADATA = ROOT / "data/tapvid_kinetics_batch41/metadata/k700_2020_val.csv"
VAL_IDS = ROOT / "data/tapvid_kinetics_batch41/tapvid_kinetics/val.txt"
FROZEN = DATA / "batch42_frozen"
SALT = "MVVG-BATCH42-KINETICS-FIRST12-20261102"
DEV_VIDEOS = 4
TEST_VIDEOS = 8
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


def prepare() -> None:
    selection = load_selection()
    verify_sources(selection)
    rows_by_video = selected_csv_rows(selection)
    target = FROZEN
    manifest_path = target / "manifest.json"
    if manifest_path.exists():
        audit()
        print("KINETICS_BATCH42_FROZEN_REUSED", flush=True)
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
        "version": "BATCH42-TAPVID-KINETICS-2026-11-02-v1",
        "dataset": "TAP-Vid-Kinetics; natural YouTube/Kinetics video clips",
        "annotation_license": "CC BY 4.0",
        "video_rights": "source videos remain local and retain their individual creator/platform terms",
        "selection_protocol": selection["protocol"],
        "selection_sha256": selection_sha,
        "source_csv_sha256": sha_file(SOURCE_CSV),
        "kinetics_val_metadata_sha256": sha_file(VAL_METADATA),
        "val_ids_sha256": sha_file(VAL_IDS),
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
    print("KINETICS_BATCH42_PREPARED", len(scenes), "scenes",
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
    if len({s["video_id"] for s in manifest["scenes"]}) != len(manifest["scenes"]):
        raise RuntimeError("KINETICS_SCENE_ID_OVERLAP")
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
    print("KINETICS_BATCH42_AUDIT_PASS", len(manifest["scenes"]), "scenes",
          "private_oracle_bytes_sha_only", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "audit"))
    action = parser.parse_args().action
    if action == "prepare":
        prepare()
    else:
        audit()


if __name__ == "__main__":
    main()
