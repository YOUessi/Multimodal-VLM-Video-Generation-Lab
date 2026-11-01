"""Batch36 real-camera provenance, frozen split, label blindness and geometry regressions."""
from __future__ import annotations

import io
import json
import pathlib
import sys
import zipfile

import cv2
import numpy as np
import pytest
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import davis_real_data as raw
import davis_real_eval as exp


def config():
    return json.loads((ROOT / "configs/davis_real_v1.json").read_text())


def test_source_and_split_protocol_precommitted():
    cfg = config()
    assert cfg["source_frames"] == 8 and cfg["source_stride"] == 3
    assert (cfg["dev_scenes"], cfg["test_scenes"]) == (4, 8)
    assert cfg["official_url"].startswith("https://data.vision.ee.ethz.ch/")
    assert cfg["dataset_dir"].startswith("data/")
    assert cfg["output_dir"].startswith("outputs/")
    assert cfg["initial_mask_only"] and not cfg["world_relative_gold"]
    assert exp.METHODS == ("stationary_pixel", "klt_forward", "klt_forward_backward")


def test_split_is_scene_disjoint_and_stable(monkeypatch):
    scenes = {
        f"scene_{i:02d}": {
            "frames": [f"{f:05d}" for f in range(30)],
            "jpg": {f"{f:05d}": f"IMG/{i}/{f:05d}.jpg" for f in range(30)},
            "png": {f"{f:05d}": f"ANN/{i}/{f:05d}.png" for f in range(30)},
        } for i in range(20)
    }
    monkeypatch.setattr(raw, "usable_sequences", lambda index: scenes)
    a = raw.selection({"entries": {}}, config())
    b = raw.selection({"entries": {}}, config())
    assert a == b
    assert len(a) == 12
    assert len({r["name"] for r in a}) == 12
    assert [r["split"] for r in a] == ["dev"] * 4 + ["test"] * 8
    assert all([int(x) for x in row["frame_ids"]] == list(range(0,24,3))
               for row in a)


def test_range_http_requires_exact_partial_content(monkeypatch):
    class Reply:
        def __init__(self, code, span):
            self.status = code
            self.headers = {"Content-Range": span}
        def __enter__(self):
            return self
        def __exit__(self, *unused):
            return False
        def read(self):
            return b"1234"
    monkeypatch.setattr(raw.urllib.request, "urlopen",
                        lambda request, timeout: Reply(200, ""))
    with pytest.raises(RuntimeError, match="HTTP_RANGE_NOT_HONORED"):
        raw.range_get("https://example.com/zip", 0, 3, attempts=1)
    monkeypatch.setattr(raw.urllib.request, "urlopen",
                        lambda request, timeout: Reply(206, "bytes 0-3/100"))
    assert raw.range_get("https://example.com/zip", 0, 3, attempts=1) == b"1234"
    with pytest.raises(ValueError, match="BAD_HTTP_RANGE"):
        raw.range_get("https://example.com/zip", 3, 1)


def test_zip_member_read_verifies_bytes_and_crc(monkeypatch):
    # Tiny trusted-memory ZIP, never open untrusted pickles or network.
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("DAVIS/JPEGImages/480p/a/00000.jpg", b"abcdef" * 100)
    content = bio.getvalue()
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        meta = zf.infolist()[0]
    def fetch(url, start, end, attempts=3):
        assert end < len(content)
        return content[start:end + 1]
    monkeypatch.setattr(raw, "range_get", fetch)
    entry = {"url": "memory://archive", "archive_bytes": len(content),
             "entries": {meta.filename: {"local_offset": meta.header_offset,
                                         "compressed": meta.compress_size,
                                         "uncompressed": meta.file_size,
                                         "crc32": meta.CRC,
                                         "method": meta.compress_type}}}
    assert raw.member_bytes(entry, meta.filename) == b"abcdef" * 100
    entry["entries"][meta.filename]["crc32"] ^= 123
    with pytest.raises(RuntimeError, match="CRC_OR_SIZE"):
        raw.member_bytes(entry, meta.filename)


def test_palette_instance_identity_and_first_mask_only(tmp_path):
    data = tmp_path
    mask0 = np.zeros((64, 64), np.uint8)
    mask0[16:34, 20:38] = 1
    mask1 = np.zeros((64, 64), np.uint8)
    mask1[20:38, 25:43] = 1
    palette = [0, 0, 0, 250, 10, 160] + [0] * (768 - 6)
    items = []
    for k, mask in enumerate((mask0, mask1)):
        where = pathlib.Path(f"masks/test/{k:05d}.png")
        (data / where).parent.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(mask, mode="P")
        image.putpalette(palette)
        image.save(data / where)
        items.append({"jpg": f"images/test/{k:05d}.jpg", "png": str(where)})
    queries, gold = raw.query_and_gold(data, [{"name": "test", "split": "dev",
                                                "assets": items}])
    q = queries["test"]
    assert abs(q["center"][0] - 28.5) < 0.1
    assert abs(q["center"][1] - 24.5) < 0.1
    assert gold["test"]["object_id_private"] == 1
    assert gold["test"]["points"][1]["center"] == pytest.approx([33.5, 28.5])
    initial = cv2.imread(str(data / q["query_mask"]), cv2.IMREAD_GRAYSCALE)
    assert initial[20,25] == 255 and initial[0,0] == 0


def scene_frames():
    rng = np.random.default_rng(20261101)
    base = rng.integers(35, 230, (140, 180, 3), dtype=np.uint8)
    # Foreground image patch with many trackable corners; the background is also textured.
    cv2.rectangle(base, (46, 37), (102, 94), (230, 230, 230), -1)
    for y in range(42, 92, 7):
        for x in range(51, 100, 7):
            cv2.circle(base, (x, y), 2, (10, 20, 10), -1)
    mask = np.zeros((140, 180), np.uint8)
    mask[37:95, 46:103] = 255
    frames = [cv2.warpAffine(base, np.float32([[1, 0, 3*k], [0, 1, 2*k]]),
                             (180, 140)) for k in range(8)]
    return frames, mask, [74., 65.]


def test_klt_on_video_frames_tracks_screen_motion_without_oracle():
    frames, mask, initial = scene_frames()
    static = [initial] * len(frames)
    track = exp.klt_track(frames, mask, initial, use_fb=True)
    assert len(track) == len(frames)
    assert track[-1] is not None
    assert np.linalg.norm(np.subtract(track[-1], [initial[0]+21,initial[1]+14])) < 5
    assert np.linalg.norm(np.subtract(static[-1], [initial[0]+21,initial[1]+14])) > 20


def test_no_texture_causes_abstention_not_imagined_direction():
    frames = [np.full((64, 64, 3), 120, np.uint8) for _ in range(8)]
    mask = np.full((64, 64), 255, np.uint8)
    tr = exp.klt_track(frames, mask, [32,32], use_fb=True)
    assert tr[0] == [32,32] and tr[1:] == [None]*7


def test_unconditional_score_counts_abstention_and_false_acceptance():
    rows = [{"name": "a"}]
    gold = {"a": {"points": [{"visible":True,"center":[0,0]},
                             {"visible":True,"center":[2,0]},
                             {"visible":True,"center":[4,0]},
                             {"visible":False,"center":None}]}}
    records = {"a": {"trajectories": {"klt_forward_backward": [[0,0],[2,0],None,[5,5]]},
                     "cpu_method_ms": {"klt_forward_backward": 12}}}
    out = exp.summarize(gold, rows, records, "klt_forward_backward")
    assert out["visible_evaluations"] == 2
    assert out["visible_accepted"] == 1
    assert out["visible_abstained"] == 1
    assert out["absent_false_accepted"] == 1
    assert out["visible_pck8_unconditional"] == 0.5
    assert out["accepted_precision_at_8px"] == 0.5


def test_prediction_does_not_open_future_gold(monkeypatch, tmp_path):
    dataset = tmp_path / "data" / "davis_real_v1"
    dataset.mkdir(parents=True)
    image = np.full((50, 50, 3), 100, np.uint8)
    cv2.imwrite(str(dataset / "first.jpg"), image)
    mask = np.full((50,50), 255,np.uint8)
    cv2.imwrite(str(dataset / "query.png"), mask)
    (dataset / "manifest.json").write_text('{"reference":"not a gold label"}')
    (dataset / "initial_queries.json").write_text(json.dumps({
        "demo":{"query_mask":"query.png","center":[25,25]}}))
    row={"name":"demo","split":"test","assets":[{"jpg":"first.jpg"}]*3}
    def fake_check(cfg, *, include_future_masks_and_gold=True):
        assert not include_future_masks_and_gold, "would read future annotation"
        return {"rows":[row]}
    monkeypatch.setattr(raw,"ROOT",tmp_path)
    monkeypatch.setattr(exp,"ROOT",tmp_path)
    monkeypatch.setattr(raw,"check",fake_check)
    original = exp.document
    def guard(p):
        assert "gold" not in p.name and "mask" not in p.name
        return original(p)
    monkeypatch.setattr(exp,"document",guard)
    exp.predict({"dataset_dir":"data/davis_real_v1","output_dir":"outputs/davis_real_v1"},"test")
    found = exp.document(tmp_path / "outputs/davis_real_v1/test_predictions.json")
    assert found["no_future_gold_in_inference"] is True
    assert found["nscenes"] == 1
