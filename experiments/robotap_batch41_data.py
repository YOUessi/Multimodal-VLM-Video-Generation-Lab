"""Batch 41: immutable RoboTAP shard acquisition and point-query firewall.

RoboTAP is synthetic robotics video, not natural camera footage. The official
TAP-Vid annotations and RoboTAP videos are CC BY 4.0. Only two official ZIP
members are range-extracted; the 12.6 GB archive is never downloaded whole.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import urllib.request
import zipfile
import zlib

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/robotap_batch41.json"
SALT = "MVVG-BATCH41-ROBOTAP-VIDEO-SPLIT-20261102-cedar-482"
BLOCK_BYTES = 8 * 1024 * 1024


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError("FROZEN_FILE_CONFLICT:" + str(path))
        return
    temp = path.with_name(path.name + ".partial")
    if temp.exists():
        raise RuntimeError("PRESERVED_PARTIAL_FILE_REQUIRES_REVIEW:" + str(temp))
    with temp.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only HTTP range view used by zipfile for one ZIP member."""

    def __init__(self, url: str, expected_size: int, expected_etag: str,
                 block_bytes: int = BLOCK_BYTES, cache_blocks: int = 3):
        super().__init__()
        self.url = url
        self.size = int(expected_size)
        self.block_bytes = int(block_bytes)
        self.cache_blocks = int(cache_blocks)
        self.position = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.request_count = 0
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            actual_size = int(response.headers.get("Content-Length", "-1"))
            actual_etag = response.headers.get("ETag", "")
            ranges = response.headers.get("Accept-Ranges", "").lower()
        if actual_size != self.size:
            raise RuntimeError("ROBOTAP_ARCHIVE_SIZE_MISMATCH")
        if actual_etag != expected_etag:
            raise RuntimeError("ROBOTAP_ARCHIVE_ETAG_MISMATCH")
        if ranges != "bytes":
            raise RuntimeError("ROBOTAP_ARCHIVE_RANGE_NOT_SUPPORTED")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            target = int(offset)
        elif whence == io.SEEK_CUR:
            target = self.position + int(offset)
        elif whence == io.SEEK_END:
            target = self.size + int(offset)
        else:
            raise ValueError("INVALID_SEEK_MODE")
        if target < 0:
            raise ValueError("NEGATIVE_SEEK")
        self.position = target
        return target

    def _get_block(self, index: int) -> bytes:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        start = index * self.block_bytes
        end = min(self.size, start + self.block_bytes) - 1
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{end}"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            if getattr(response, "status", None) != 206:
                raise RuntimeError("ROBOTAP_RANGE_HTTP_STATUS_NOT_206")
            content_range = response.headers.get("Content-Range", "")
            expected_prefix = f"bytes {start}-{end}/{self.size}"
            if content_range != expected_prefix:
                raise RuntimeError("ROBOTAP_RANGE_CONTENT_RANGE_MISMATCH")
            data = response.read()
        if len(data) != end - start + 1:
            raise RuntimeError("ROBOTAP_RANGE_LENGTH_MISMATCH")
        self.request_count += 1
        self.cache[index] = data
        while len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return data

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.position
        size = min(int(size), self.size - self.position)
        if size <= 0:
            return b""
        output = bytearray()
        remaining = size
        while remaining:
            block_index = self.position // self.block_bytes
            block_offset = self.position % self.block_bytes
            block = self._get_block(block_index)
            take = min(remaining, len(block) - block_offset)
            if take <= 0:
                break
            output.extend(block[block_offset:block_offset + take])
            self.position += take
            remaining -= take
        return bytes(output)


def crc32_file(path: Path) -> int:
    value = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value = zlib.crc32(chunk, value)
    return value & 0xFFFFFFFF


def download_members(cfg: dict) -> dict:
    data_dir = ROOT / cfg["dataset_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    remote = HttpRangeFile(
        cfg["archive_url"], cfg["archive_size"], cfg["archive_etag"]
    )
    members = {}
    with zipfile.ZipFile(remote) as archive:
        for member in cfg["archive_members"]:
            info = archive.getinfo(member)
            output = data_dir / Path(member).name
            if output.exists():
                if output.stat().st_size != info.file_size:
                    raise RuntimeError("ROBOTAP_EXISTING_MEMBER_SIZE_MISMATCH:" + member)
                if crc32_file(output) != info.CRC:
                    raise RuntimeError("ROBOTAP_EXISTING_MEMBER_CRC_MISMATCH:" + member)
                print("ROBOTAP_MEMBER_ALREADY_VERIFIED", member, flush=True)
            else:
                partial = output.with_name(output.name + ".partial")
                if partial.exists():
                    raise RuntimeError("ROBOTAP_PARTIAL_MEMBER_PRESERVED:" + str(partial))
                copied = 0
                with archive.open(info, "r") as source, partial.open("xb") as target:
                    while True:
                        chunk = source.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        target.write(chunk)
                        copied += len(chunk)
                        if copied % (256 * 1024 * 1024) < len(chunk):
                            print("ROBOTAP_MEMBER_EXTRACTED_MIB", member,
                                  copied // 1048576, flush=True)
                    target.flush()
                    os.fsync(target.fileno())
                if copied != info.file_size or crc32_file(partial) != info.CRC:
                    raise RuntimeError("ROBOTAP_MEMBER_CRC_OR_SIZE_FAILURE:" + member)
                os.replace(partial, output)
                print("ROBOTAP_MEMBER_EXTRACTED_OK", member, copied, flush=True)
            members[output.name] = {
                "member": member, "size": info.file_size,
                "crc32": f"{info.CRC:08x}", "sha256": sha_file(output),
            }
    doc = {
        "archive_url": cfg["archive_url"],
        "archive_size": cfg["archive_size"],
        "archive_etag": cfg["archive_etag"],
        "license": "CC BY 4.0; upstream source videos remain attributed to RoboTAP",
        "members": members,
    }
    path = data_dir / "download_manifest.json"
    atomic_once(path, json_bytes(doc))
    print("ROBOTAP_DOWNLOAD_MANIFEST_PINNED", sha_file(path), flush=True)
    return doc


def load_official_records(cfg: dict) -> list[tuple[str, dict]]:
    data_dir = ROOT / cfg["dataset_dir"]
    download_manifest_path = data_dir / "download_manifest.json"
    if not download_manifest_path.is_file():
        raise RuntimeError("ROBOTAP_DOWNLOAD_MANIFEST_MISSING")
    download_manifest = json.loads(download_manifest_path.read_text())
    records = []
    for filename, metadata in sorted(download_manifest["members"].items()):
        pkl_path = data_dir / filename
        if (not pkl_path.is_file()
                or pkl_path.stat().st_size != metadata["size"]
                or sha_file(pkl_path) != metadata["sha256"]):
            raise RuntimeError("ROBOTAP_SOURCE_MEMBER_HASH_DRIFT:" + filename)
        with pkl_path.open("rb") as stream:
            payload = pickle.load(stream)
        if isinstance(payload, dict):
            items = sorted(payload.items(), key=lambda item: str(item[0]))
        elif isinstance(payload, (list, tuple)):
            items = [(f"video_{i:04d}", value) for i, value in enumerate(payload)]
        else:
            raise RuntimeError("ROBOTAP_PICKLE_ROOT_TYPE_UNSUPPORTED:" + filename)
        for key, sample in items:
            scene_key = filename + "::" + str(key)
            if not isinstance(sample, dict):
                raise RuntimeError("ROBOTAP_SAMPLE_NOT_A_DICT:" + scene_key)
            records.append((scene_key, sample))
    if len({key for key, _ in records}) != len(records):
        raise RuntimeError("ROBOTAP_DUPLICATE_SCENE_KEYS")
    return records


def decode_video(sample: dict) -> np.ndarray:
    if "video" not in sample:
        raise RuntimeError("ROBOTAP_VIDEO_FIELD_MISSING")
    video = np.asarray(sample["video"])
    if video.ndim == 4 and video.shape[-1] == 3 and video.dtype == np.uint8:
        return np.ascontiguousarray(video)
    if video.ndim == 1 and video.dtype.kind in ("S", "O"):
        frames = []
        for encoded in video:
            raw = bytes(encoded)
            decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise RuntimeError("ROBOTAP_ENCODED_FRAME_DECODE_FAILED")
            frames.append(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
        return np.stack(frames, axis=0)
    raise RuntimeError("ROBOTAP_VIDEO_ARRAY_SHAPE_OR_DTYPE_UNSUPPORTED")


def normalized_to_pixels(points: np.ndarray, occluded: np.ndarray,
                         height: int, width: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    occluded = np.asarray(occluded, dtype=bool)
    if (points.ndim != 3 or points.shape[-1] != 2
            or occluded.shape != points.shape[:2]
            or height <= 0 or width <= 0):
        raise ValueError("ROBOTAP_POINT_ARRAY_SHAPE_INVALID")
    visible = points[~occluded]
    if (not np.isfinite(visible).all()
            or np.any(visible < 0) or np.any(visible > 1)):
        raise RuntimeError("ROBOTAP_VISIBLE_POINTS_NOT_NORMALIZED")
    pixels = points * np.array([width, height], dtype=np.float64)
    if not np.isfinite(pixels[~occluded]).all():
        raise RuntimeError("ROBOTAP_VISIBLE_PIXEL_POINT_NONFINITE")
    return pixels


def first_visible_query(track_id: int, points_px: np.ndarray,
                        occluded: np.ndarray, width: int, height: int) -> dict | None:
    visible = np.flatnonzero(~np.asarray(occluded, dtype=bool))
    if not len(visible):
        return None
    query_time = int(visible[0])
    xy = np.asarray(points_px[query_time], dtype=np.float64)
    if (not np.isfinite(xy).all()
            or not 0 <= xy[0] < width or not 0 <= xy[1] < height):
        raise RuntimeError("ROBOTAP_FIRST_VISIBLE_QUERY_OUT_OF_BOUNDS")
    return {
        "id": f"p{track_id:04d}", "track_index": int(track_id),
        "query_time": query_time,
        "query_xy": [round(float(x), 4) for x in xy],
        "query_rule": "first annotated visible point; no future labels passed to predictor",
    }


def selected_video_keys(records: list[tuple[str, dict]], total: int) -> list[str]:
    keys = [key for key, _ in records]
    if len(keys) < total:
        raise RuntimeError("ROBOTAP_NOT_ENOUGH_SCENES:" + str(len(keys)))
    if len(keys) != len(set(keys)):
        raise RuntimeError("ROBOTAP_SCENE_KEY_DUPLICATE")
    rank = lambda key: hashlib.sha256((SALT + "|VIDEO|" + key).encode()).hexdigest()
    return sorted(keys, key=rank)[:total]


def prepare(cfg: dict) -> dict:
    data_dir = ROOT / cfg["dataset_dir"]
    frozen = data_dir / "batch41_frozen"
    manifest_path = frozen / "manifest.json"
    if manifest_path.exists():
        return audit(cfg)
    download_manifest = json.loads((data_dir / "download_manifest.json").read_text())
    records = load_official_records(cfg)
    total = int(cfg["dev_videos"]) + int(cfg["test_videos"])
    chosen_keys = selected_video_keys(records, total)
    by_key = dict(records)
    public: dict[str, list] = {}
    private: dict[str, dict] = {}
    scenes = []
    for slot, key in enumerate(chosen_keys):
        sample = by_key[key]
        frames = decode_video(sample)
        points = np.asarray(sample.get("points"), dtype=np.float64)
        occluded = np.asarray(sample.get("occluded"), dtype=bool)
        if (frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8
                or points.ndim != 3 or points.shape[-1] != 2
                or occluded.shape != points.shape[:2]
                or frames.shape[0] != points.shape[1]):
            raise RuntimeError("ROBOTAP_SAMPLE_SHAPE_MISMATCH:" + key)
        height, width = frames.shape[1:3]
        points_px = normalized_to_pixels(points, occluded, height, width)
        track_order = sorted(
            range(points.shape[0]),
            key=lambda idx: hashlib.sha256(
                (SALT + "|TRACK|" + key + "|" + str(idx)).encode()
            ).hexdigest(),
        )
        selected_tracks = track_order[:int(cfg["max_tracks_per_video"])]
        queries = []
        private_tracks = {}
        for track_id in selected_tracks:
            query = first_visible_query(track_id, points_px[track_id],
                                        occluded[track_id], width, height)
            if query is None or query["query_time"] >= len(frames) - 1:
                continue
            queries.append(query)
            private_xy = []
            for frame_index, xy in enumerate(points_px[track_id]):
                if (bool(occluded[track_id, frame_index])
                        or not np.isfinite(xy).all()):
                    private_xy.append(None)
                else:
                    private_xy.append([float(v) for v in xy])
            private_tracks[query["id"]] = {
                "xy": private_xy,
                "occluded": [bool(v) for v in occluded[track_id]],
            }
        if not queries:
            raise RuntimeError("ROBOTAP_SCENE_HAS_NO_VALID_FIRST_VISIBLE_QUERY:" + key)
        scene_id = f"scene_{slot:03d}"
        split = "dev" if slot < int(cfg["dev_videos"]) else "test"
        frame_rel = f"frames/{scene_id}.npy"
        frame_path = frozen / frame_rel
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        if not frame_path.exists():
            temp = frame_path.with_name(frame_path.name + ".partial")
            if temp.exists():
                raise RuntimeError("ROBOTAP_FRAME_PARTIAL_PRESERVED:" + str(temp))
            with temp.open("xb") as stream:
                np.save(stream, frames, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, frame_path)
        else:
            existing = np.load(frame_path, mmap_mode="r", allow_pickle=False)
            if (existing.shape != frames.shape or existing.dtype != frames.dtype
                    or not np.array_equal(existing, frames)):
                raise RuntimeError("ROBOTAP_FROZEN_FRAME_CONFLICT:" + scene_id)
        public[scene_id] = queries
        private[scene_id] = {"tracks": private_tracks}
        scenes.append({
            "name": scene_id, "source_key": key, "split": split,
            "frames": frame_rel, "frames_sha256": sha_file(frame_path),
            "nframes": int(len(frames)), "height": int(height), "width": int(width),
            "source_track_count": int(points.shape[0]), "nqueries": len(queries),
            "selected_track_ids": [int(q["track_index"]) for q in queries],
        })
        print("ROBOTAP_SCENE_FROZEN", scene_id, split, key,
              "frames", len(frames), "queries", len(queries), flush=True)
    manifest = {
        "version": cfg["version"],
        "dataset": "Official RoboTAP / TAP-Vid synthetic robotics videos",
        "source_type": "synthetic robotics; not natural camera footage",
        "license": "CC BY 4.0 for RoboTAP videos and TAP-Vid annotations",
        "source_archive_url": cfg["archive_url"],
        "source_archive_etag": cfg["archive_etag"],
        "source_members": download_manifest["members"],
        "download_manifest_sha256": sha_file(data_dir / "download_manifest.json"),
        "selection_salt": SALT,
        "selection_rule": "video-id hash only; track ids hash only; first-visible query protocol",
        "dev_videos": int(cfg["dev_videos"]), "test_videos": int(cfg["test_videos"]),
        "oracle_sha256": sha_bytes(json_bytes(private)),
        "public_queries_sha256": sha_bytes(json_bytes(public)),
        "coordinate_rule": "official normalized (x,y) in [0,1] multiplied by frame (W,H)",
        "scenes": scenes,
    }
    frozen.mkdir(parents=True, exist_ok=True)
    atomic_once(frozen / "queries-public.json", json_bytes(public))
    atomic_once(frozen / "oracle-private.json", json_bytes(private))
    atomic_once(manifest_path, json_bytes(manifest))
    audit(cfg)
    print("ROBOTAP_PREPARED", len(scenes), "scenes",
          "manifest_sha256", sha_file(manifest_path), flush=True)
    return manifest


def audit(cfg: dict, verify_sources: bool = False) -> dict:
    data_dir = ROOT / cfg["dataset_dir"]
    frozen = data_dir / "batch41_frozen"
    manifest_path = frozen / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("ROBOTAP_FROZEN_MANIFEST_MISSING")
    manifest = json.loads(manifest_path.read_text())
    queries_path = frozen / "queries-public.json"
    if not queries_path.is_file():
        raise RuntimeError("ROBOTAP_PUBLIC_QUERIES_MISSING")
    queries = json.loads(queries_path.read_text())
    if sha_file(data_dir / "download_manifest.json") != manifest["download_manifest_sha256"]:
        raise RuntimeError("ROBOTAP_DOWNLOAD_MANIFEST_DRIFT")
    if sha_bytes(json_bytes(queries)) != manifest["public_queries_sha256"]:
        raise RuntimeError("ROBOTAP_PUBLIC_QUERY_HASH_DRIFT")
    scenes = manifest["scenes"]
    expected = int(cfg["dev_videos"]) + int(cfg["test_videos"])
    if (len(scenes) != expected
            or [s["split"] for s in scenes]
            != ["dev"] * int(cfg["dev_videos"]) + ["test"] * int(cfg["test_videos"])
            or len({s["source_key"] for s in scenes}) != len(scenes)):
        raise RuntimeError("ROBOTAP_SCENE_SPLIT_OR_IDENTITY_DRIFT")
    for scene in scenes:
        frame_path = frozen / scene["frames"]
        if not frame_path.is_file() or sha_file(frame_path) != scene["frames_sha256"]:
            raise RuntimeError("ROBOTAP_FRAME_SHA_DRIFT:" + scene["name"])
        if scene["name"] not in queries or len(queries[scene["name"]]) != scene["nqueries"]:
            raise RuntimeError("ROBOTAP_QUERY_COVERAGE_DRIFT:" + scene["name"])
        frames = np.load(frame_path, mmap_mode="r", allow_pickle=False)
        if (frames.ndim != 4 or frames.dtype != np.uint8
                or frames.shape[0] != scene["nframes"]
                or list(frames.shape[1:3]) != [scene["height"], scene["width"]]):
            raise RuntimeError("ROBOTAP_FRAME_SHAPE_DRIFT:" + scene["name"])
        for query in queries[scene["name"]]:
            xy = np.asarray(query["query_xy"], dtype=np.float64)
            if (not 0 <= int(query["query_time"]) < scene["nframes"] - 1
                    or xy.shape != (2,) or not np.isfinite(xy).all()
                    or not 0 <= xy[0] < scene["width"]
                    or not 0 <= xy[1] < scene["height"]):
                raise RuntimeError("ROBOTAP_INVALID_PUBLIC_QUERY")
    if verify_sources:
        dm = json.loads((data_dir / "download_manifest.json").read_text())
        for filename, metadata in dm["members"].items():
            path = data_dir / filename
            if (not path.is_file() or path.stat().st_size != metadata["size"]
                    or sha_file(path) != metadata["sha256"]):
                raise RuntimeError("ROBOTAP_SOURCE_FILE_SHA_DRIFT:" + filename)
    print("ROBOTAP_PUBLIC_AUDIT_PASS", len(scenes), "scenes",
          "verify_sources", verify_sources, flush=True)
    return manifest


def load_gold(cfg: dict) -> dict:
    frozen = ROOT / cfg["dataset_dir"] / "batch41_frozen"
    manifest = json.loads((frozen / "manifest.json").read_text())
    path = frozen / "oracle-private.json"
    if sha_file(path) != manifest["oracle_sha256"]:
        raise RuntimeError("ROBOTAP_PRIVATE_ORACLE_SHA_DRIFT")
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("download", "prepare", "audit", "final-audit"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    if args.action == "download":
        download_members(cfg)
    elif args.action == "prepare":
        prepare(cfg)
    elif args.action == "audit":
        audit(cfg)
    else:
        audit(cfg, verify_sources=True)


if __name__ == "__main__":
    main()
