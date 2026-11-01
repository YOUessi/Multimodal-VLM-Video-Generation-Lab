"""Resumable, verifiable parallel HTTP-Range download of official TAP-Vid RGB-Stacking.

No archive contents are deserialized before origin MD5/ZIP CRC verification.
Only pinned, official HTTPS endpoint is permitted. No media enters Git.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile

SOURCE = "https://storage.googleapis.com/dm-tapnet/tapvid_rgb_stacking.zip"
SIZE = 187291581
MD5 = "46ec560a2387a819a4856a4ac5b2e27d"
CHUNK = 2 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def range_bytes(start: int, end: int, *, retries: int = 5) -> bytes:
    assert 0 <= start <= end < SIZE
    for attempt in range(retries):
        try:
            req = Request(SOURCE, headers={
                "Range": f"bytes={start}-{end}",
                "User-Agent": "MVVG-TAPVid-RGB-Frozen-Ranges/1",
            })
            with urlopen(req, timeout=90) as response:
                body = response.read()
                expected = f"bytes {start}-{end}/{SIZE}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise RuntimeError("SOURCE_RANGE_NOT_HONORED")
            if len(body) != end - start + 1:
                raise RuntimeError("SOURCE_RANGE_LENGTH_MISMATCH")
            return body
        except (OSError, RuntimeError):
            if attempt + 1 == retries:
                raise
            time.sleep(min(2 ** attempt, 12))
    raise RuntimeError("UNREACHABLE")


def download(directory: Path, workers: int = 18) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / "official.zip"
    if final.is_file():
        verify(final)
        return final
    partial = directory / "official.zip.range.part"
    receipt = directory / "completed_ranges.jsonl"
    total = (SIZE + CHUNK - 1) // CHUNK
    if not partial.exists():
        with partial.open("wb") as f:
            f.truncate(SIZE)
    if partial.stat().st_size != SIZE:
        raise RuntimeError("PARTIAL_SIZE_DRIFT")
    good: dict[int, str] = {}
    if receipt.exists():
        for line in receipt.read_text().splitlines():
            r = json.loads(line)
            i = int(r["chunk"])
            if 0 <= i < total:
                good[i] = str(r["sha256"])
    # Never trust a receipt without re-reading actual bytes.
    fd = os.open(partial, os.O_RDWR)
    lock = threading.Lock()
    completed = set()
    try:
        for i, expected in good.items():
            start = i * CHUNK
            b = os.pread(fd, min(CHUNK, SIZE-start), start)
            if hashlib.sha256(b).hexdigest() == expected:
                completed.add(i)
        print(f"RANGE_RESUME valid_chunks={len(completed)}/{total}", flush=True)

        def job(i: int) -> tuple[int, str]:
            start = i * CHUNK
            data = range_bytes(start, min(SIZE - 1, start + CHUNK - 1))
            sha = hashlib.sha256(data).hexdigest()
            if os.pwrite(fd, data, start) != len(data):
                raise RuntimeError("SHORT_RANGE_WRITE")
            with lock:
                with receipt.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"chunk": i, "sha256": sha}) + "\n")
                completed.add(i)
                if len(completed) % 10 == 0 or len(completed) == total:
                    print(f"RANGE_PROGRESS {len(completed)}/{total}", flush=True)
            return i, sha

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(job, i): i for i in range(total) if i not in completed}
            for f in as_completed(futures):
                f.result()
        os.fsync(fd)
    finally:
        os.close(fd)
    if len(completed) != total:
        raise RuntimeError("SOURCE_RANGES_INCOMPLETE")
    verify(partial)
    os.replace(partial, final)
    print("RANGE_DOWNLOAD_COMPLETE", final, "sha256", digest(final), flush=True)
    return final


def verify(path: Path) -> None:
    if path.stat().st_size != SIZE:
        raise RuntimeError("SOURCE_ARCHIVE_SIZE_MISMATCH")
    if digest(path, "md5") != MD5:
        raise RuntimeError("OFFICIAL_MD5_MISMATCH")
    with ZipFile(path) as z:
        names = z.namelist()
        if any("/../" in n or n.startswith("/") for n in names):
            raise RuntimeError("ZIP_MEMBER_TRAVERSAL")
        if z.testzip() is not None:
            raise RuntimeError("OFFICIAL_ZIP_CRC_FAILURE")
    print("OFFICIAL_SOURCE_VERIFIED", "md5", MD5, "sha256", digest(path), "members", names, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["download", "verify"])
    p.add_argument("--dest", required=True)
    p.add_argument("--workers", type=int, default=18)
    a = p.parse_args()
    path = Path(a.dest)
    if a.action == "download":
        download(path, a.workers)
    else:
        verify(path / "official.zip")


if __name__ == "__main__":
    main()
