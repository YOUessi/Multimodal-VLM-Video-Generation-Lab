"""Read a *selected* set of official DAVIS 2017 real camera images and masks by HTTP ZIP ranges.

No full archive extraction, zip-slip, weights, secrets, or commits of media.
All private annotations are created once, and predictor reads only legal initial query.
These are real camera VIDEO FRAMES, not world-motion or 3D motion labels.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import struct
import time
import urllib.request
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CENTRAL_HEADER = "<IHHHHHHIIIHHHHHII"
EOCD_HEADER = "<IHHHHIIH"
LOCAL_HEADER = "<IHHHHHIIIHH"
VALID_SUFFIXES = (".jpg", ".png", ".txt", ".md")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def cfg_hash(cfg: dict) -> str:
    return sha256(json.dumps(cfg, sort_keys=True).encode("utf8"))


def range_get(url: str, start: int, end: int, *, attempts: int = 3) -> bytes:
    """Require verifiable 206 bytes; never silently accept an entire archive."""
    if not 0 <= start <= end:
        raise ValueError("BAD_HTTP_RANGE")
    for retry in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "MVVG-DAVIS-range-audit/1"}
            )
            with urllib.request.urlopen(request, timeout=35) as response:
                data = response.read()
                cr = response.headers.get("Content-Range", "")
                if response.status != 206 or not cr.startswith(f"bytes {start}-{end}/"):
                    raise RuntimeError("HTTP_RANGE_NOT_HONORED")
                if len(data) != end - start + 1:
                    raise RuntimeError("PARTIAL_RANGE_READ")
                return data
        except (OSError, RuntimeError):
            if retry + 1 == attempts:
                raise
            time.sleep(1 + retry)
    raise RuntimeError("HTTP_RETRY_EXHAUSTED")


def zip_index(url: str, *, workers: int = 8) -> dict:
    """Read ZIP directory from tail in bounded ranges, verifying every directory entry."""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=25) as response:
        size = int(response.headers["Content-Length"])
    if not 1000 < size < 10_000_000_000:
        raise RuntimeError("SUSPICIOUS_ARCHIVE_SIZE")
    tail_start = size - 4096
    tail = range_get(url, tail_start, size - 1)
    index = tail.rfind(b"PK\x05\x06")
    if index == -1:
        raise RuntimeError("ZIP_END_NOT_FOUND")
    signature, _, _, n_disk, n_total, cd_size, cd_start, comment = struct.unpack_from(EOCD_HEADER, tail, index)
    if signature != 0x06054B50 or n_total != n_disk or not (0 < cd_size <= 10_000_000):
        raise RuntimeError("ZIP_DIRECTORY_INVALID")
    # The ZIP central directory normally ends immediately before EOCD in
    # the final 4 KiB; overlapping the *tail read* is expected, not an error.
    if cd_start <= 0 or cd_start + cd_size > tail_start + index:
        raise RuntimeError("ZIP_DIRECTORY_OUTSIDE_ARCHIVE")
    chunks = [(p, min(p + 131071, cd_start + cd_size - 1))
              for p in range(cd_start, cd_start + cd_size, 131072)]
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        buf = b"".join(pool.map(lambda ab: range_get(url, *ab), chunks))
    if len(buf) != cd_size:
        raise RuntimeError("ZIP_DIRECTORY_LENGTH_INVALID")
    entries = {}
    offset = 0
    for _ in range(n_total):
        if buf[offset:offset + 4] != b"PK\x01\x02":
            raise RuntimeError("ZIP_DIRECTORY_TRUNCATED")
        header = struct.unpack_from(CENTRAL_HEADER, buf, offset)
        (sig, made_by, required, flags, method, _, _, crc, compressed, uncompressed,
         name_len, extra_len, comment_len, disk, internal, external, local_offset) = header
        raw_name = buf[offset + 46:offset + 46 + name_len]
        name = raw_name.decode("utf8")
        offset += 46 + name_len + extra_len + comment_len
        if name.startswith("/") or ".." in Path(name).parts or "\\" in name:
            raise RuntimeError("ZIP_MEMBER_PATH_TRAVERSAL")
        if name in entries:
            raise RuntimeError("ZIP_DUPLICATE_ENTRY")
        if flags & 1 or method not in (0, 8) or uncompressed > 30_000_000:
            continue
        if name.endswith(VALID_SUFFIXES):
            entries[name] = {
                "local_offset": local_offset, "compressed": compressed, "uncompressed": uncompressed,
                "crc32": crc, "method": method,
            }
    if offset != len(buf):
        raise RuntimeError("ZIP_DIRECTORY_TRAILING_BYTES")
    return {"url": url, "archive_bytes": size, "directory_sha256": sha256(buf), "entries": entries}


def member_bytes(index: dict, name: str) -> bytes:
    meta = index["entries"][name]
    off = meta["local_offset"]
    total = 30 + 1024 + meta["compressed"]
    if total > 30_002_000:
        raise RuntimeError("ZIP_MEMBER_TOO_LARGE")
    # 1024 bytes of local-header/name slack avoids a second GET per asset.
    block = range_get(index["url"], off, min(index["archive_bytes"] - 1, off + total - 1))
    sig, version, flags, method, _, _, crc, csize, usize, name_len, extra_len = struct.unpack_from(
        LOCAL_HEADER, block, 0
    )
    start = 30 + name_len + extra_len
    if (sig != 0x04034B50 or block[30:30 + name_len].decode("utf8") != name
            or start + meta["compressed"] > len(block)):
        raise RuntimeError("ZIP_LOCAL_HEADER_MISMATCH")
    compressed = block[start:start + meta["compressed"]]
    raw = zlib.decompress(compressed, -15) if meta["method"] == 8 else compressed
    if len(raw) != meta["uncompressed"] or zlib.crc32(raw) != meta["crc32"]:
        raise RuntimeError("ZIP_MEMBER_CRC_OR_SIZE_MISMATCH")
    return raw


def usable_sequences(index: dict) -> dict:
    jpg = {}
    png = {}
    for name in index["entries"]:
        m = re.search(r"/JPEGImages/480p/([^/]+)/([0-9]{5})\.jpg$", name)
        if m:
            jpg.setdefault(m.group(1), {})[m.group(2)] = name
        m = re.search(r"/Annotations/480p/([^/]+)/([0-9]{5})\.png$", name)
        if m:
            png.setdefault(m.group(1), {})[m.group(2)] = name
    common = {}
    for scene in sorted(set(jpg) & set(png)):
        frames = sorted(set(jpg[scene]) & set(png[scene]))
        if len(frames) >= 22:
            common[scene] = {"frames": frames, "jpg": jpg[scene], "png": png[scene]}
    return common


def selection(index: dict, cfg: dict) -> list:
    scenes = usable_sequences(index)
    # Batch37+ may require longer sequences and explicit *previous-scene*
    # exclusion. Both constraints are metadata-only, never future gold masks.
    required = cfg["source_frames"] * cfg["source_stride"]
    excluded = set(cfg.get("exclude_scenes", []))
    scenes = {name: entry for name, entry in scenes.items()
              if name not in excluded and len(entry["frames"]) >= required}
    candidates = sorted(scenes, key=lambda name: sha256((cfg["split_salt"] + "/" + name).encode("utf8")))
    amount = cfg["dev_scenes"] + cfg["test_scenes"]
    if len(candidates) < amount:
        raise RuntimeError("INSUFFICIENT_DISTINCT_REAL_CAMERA_SCENES")
    selected = []
    for k, scene in enumerate(candidates[:amount]):
        # Decimation is fixed *before* masks are read.
        ids = scenes[scene]["frames"]
        frames = ids[:cfg["source_frames"] * cfg["source_stride"]:cfg["source_stride"]]
        if len(frames) != cfg["source_frames"]:
            raise RuntimeError("FRAME_SELECTION_INCOMPLETE")
        selected.append({"name": scene, "split": "dev" if k < cfg["dev_scenes"] else "test",
                         "frame_ids": frames,
                         "members": [{"frame": f, "jpg": scenes[scene]["jpg"][f],
                                      "png": scenes[scene]["png"][f]} for f in frames]})
    return selected


def save_if_frozen(path: Path, raw: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError("EXISTING_LOCAL_MEDIA_BYTES_DRIFT:" + str(path))
    else:
        tmp = path.with_suffix(path.suffix + ".partial")
        tmp.write_bytes(raw)
        tmp.replace(path)
    return sha256(raw)


def extract_one(job):
    index, name, dest = job
    raw = member_bytes(index, name)
    return str(dest), save_if_frozen(dest, raw)


def query_and_gold(dataset: Path, rows: list) -> tuple:
    """Never expose future masks as an inference input."""
    queries, gold = {}, {}
    for row in rows:
        scene = row["name"]
        points = []
        chosen = None
        for k, item in enumerate(row["assets"]):
            # DAVIS uses indexed palette PNGs: cv2.imread would map index IDs to
            # RGB colors and lose the original integer instance labels.
            with Image.open(dataset / item["png"]) as image:
                mask = np.asarray(image)
            if mask.ndim != 2:
                raise RuntimeError("DAVIS_INSTANCE_MASK_DECODE_FAILURE")
            if k == 0:
                labels, counts = np.unique(mask, return_counts=True)
                counts = [(int(v), int(c)) for v, c in zip(labels, counts) if 1 <= int(v) <= 254]
                if not counts:
                    raise RuntimeError("DAVIS_FIRST_FRAME_NO_FOREGROUND")
                chosen = max(counts, key=lambda x: (x[1], -x[0]))[0]
                only = ((mask == chosen) * 255).astype(np.uint8)
                query_path = Path("queries") / f"{scene}.png"
                # Initial user-provided mask is the *only* ground truth seen at inference.
                ok, encoded = cv2.imencode(".png", only)
                if not ok:
                    raise RuntimeError("QUERY_MASK_ENCODING_FAILED")
                save_if_frozen(dataset / query_path, encoded.tobytes())
                ys, xs = np.where(only != 0)
                queries[scene] = {"query_mask": str(query_path),
                                  "center": [float(xs.mean()), float(ys.mean())],
                                  "frames": [x["jpg"] for x in row["assets"]]}
            coords = np.where(mask == chosen)
            if len(coords[0]):
                points.append({"visible": True,
                               "center": [float(coords[1].mean()), float(coords[0].mean())],
                               "pixels": int(len(coords[0]))})
            else:
                points.append({"visible": False, "center": None, "pixels": 0})
        gold[scene] = {"split": row["split"], "points": points, "object_id_private": chosen}
    return queries, gold


def acquire(cfg: dict) -> None:
    dataset = ROOT / cfg["dataset_dir"]
    dataset.mkdir(parents=True, exist_ok=True)
    manifest_file = dataset / "manifest.json"
    if manifest_file.exists():
        check(cfg)
        print("DAVIS_REAL_ALREADY_FROZEN", manifest_file, flush=True)
        return
    index = zip_index(cfg["official_url"])
    rows = selection(index, cfg)
    tasks = []
    for row in rows:
        row["assets"] = []
        for item in row["members"]:
            assets = {"frame": item["frame"]}
            for typ in ("jpg", "png"):
                src = item[typ]
                dest = Path("images" if typ == "jpg" else "masks") / row["name"] / (item["frame"] + "." + typ)
                tasks.append((index, src, dataset / dest))
                assets[typ] = str(dest)
            row["assets"].append(assets)
        del row["members"]
    print("DAVIS_RANGE_ACQUIRE",len(rows),"scenes",len(tasks),"assets",flush=True)
    with cf.ThreadPoolExecutor(max_workers=cfg["download_workers"]) as pool:
        paths = list(pool.map(extract_one, tasks))
    digests = {str(Path(p).relative_to(dataset)): h for p, h in paths}
    queries, gold = query_and_gold(dataset, rows)
    save_if_frozen(dataset / "initial_queries.json", json.dumps(queries, sort_keys=True, indent=2).encode())
    save_if_frozen(dataset / "gold-private.json", json.dumps(gold, sort_keys=True, indent=2).encode())
    manifest = {
        "config_sha256": cfg_hash(cfg), "official_url": cfg["official_url"],
        "archive_bytes": index["archive_bytes"], "directory_sha256": index["directory_sha256"],
        "natural_camera_video": True, "world_relative_motion_gold_available": False,
        "source_frames_per_scene": cfg["source_frames"], "rows": rows, "sha256": digests,
    }
    save_if_frozen(manifest_file, json.dumps(manifest, sort_keys=True, indent=2).encode())
    check(cfg)
    print("DAVIS_REAL_FROZEN", len(rows), len(paths), sha256(manifest_file.read_bytes()), flush=True)


def check(cfg: dict, *, include_future_masks_and_gold: bool = True) -> dict:
    dataset = ROOT / cfg["dataset_dir"]
    doc = json.loads((dataset / "manifest.json").read_text())
    if doc["config_sha256"] != cfg_hash(cfg):
        raise RuntimeError("FROZEN_DAVIS_CONFIG_CHANGED")
    if len(doc["rows"]) != cfg["dev_scenes"] + cfg["test_scenes"]:
        raise RuntimeError("FROZEN_DAVIS_SCENE_COUNT_CHANGED")
    rows = doc["rows"]
    if len(set(r["name"] for r in rows)) != len(rows):
        raise RuntimeError("CROSS_SPLIT_DAVIS_SCENE_DUPLICATED")
    if len([r for r in rows if r["split"] == "test"]) != cfg["test_scenes"]:
        raise RuntimeError("DAVIS_TEST_COUNT_DRIFT")
    initial_masks = {row["assets"][0]["png"] for row in rows}
    for path, expected in doc["sha256"].items():
        # The first annotation is the user-supplied query; do not read future masks.
        if (not include_future_masks_and_gold and path.startswith("masks/")
                and path not in initial_masks):
            continue
        candidate = (dataset / path).resolve()
        if not candidate.is_relative_to(dataset.resolve()) or sha256(candidate.read_bytes()) != expected:
            raise RuntimeError("DAVIS_SOURCE_ASSET_DRIFT:" + path)
    queries = json.loads((dataset / "initial_queries.json").read_text())
    if set(queries) != {r["name"] for r in rows}:
        raise RuntimeError("DAVIS_QUERY_COVERAGE_MISMATCH")
    for row in rows:
        q = queries[row["name"]]
        mask_path = (dataset / q["query_mask"]).resolve()
        if (not mask_path.is_relative_to(dataset.resolve()) or
                not mask_path.is_file()):
            raise RuntimeError("DAVIS_QUERY_MASK_MISSING_OR_UNSAFE_PATH")
        # Rebuild the query deterministically from the FIRST mask only.
        with Image.open(dataset / row["assets"][0]["png"]) as first:
            source_ids = np.asarray(first)
        if source_ids.ndim != 2:
            raise RuntimeError("DAVIS_INITIAL_PALETTE_MASK_INVALID")
        labels, counts = np.unique(source_ids, return_counts=True)
        fg = [(int(v), int(n)) for v, n in zip(labels, counts)
              if 1 <= int(v) <= 254]
        if not fg:
            raise RuntimeError("DAVIS_INITIAL_FOREGROUND_MISSING")
        chosen = max(fg, key=lambda pair: (pair[1], -pair[0]))[0]
        expected_mask = ((source_ids == chosen) * 255).astype(np.uint8)
        query_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if (query_mask is None or query_mask.shape != expected_mask.shape or
                not np.array_equal(query_mask, expected_mask)):
            raise RuntimeError("DAVIS_FROZEN_INITIAL_QUERY_MASK_DRIFT")
        ys, xs = np.where(expected_mask != 0)
        if (len(q.get("center", [])) != 2 or
                not np.allclose(q["center"], [xs.mean(), ys.mean()], rtol=0, atol=1e-5) or
                q.get("frames") != [item["jpg"] for item in row["assets"]]):
            raise RuntimeError("DAVIS_FROZEN_INITIAL_QUERY_METADATA_DRIFT")
    if include_future_masks_and_gold:
        gold = json.loads((dataset / "gold-private.json").read_text())
        if set(queries) != set(gold):
            raise RuntimeError("DAVIS_PRIVATE_LABEL_QUERY_MISMATCH")
    print("DAVIS_BYTES_AUDIT_OK",len(doc["sha256"]),"assets",flush=True)
    return doc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("acquire", "audit"))
    p.add_argument("--config", default="configs/davis_real_v1.json")
    args = p.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    acquire(cfg) if args.action == "acquire" else check(cfg)


if __name__ == "__main__":
    main()
