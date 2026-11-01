"""Batch36 prospective checks: the only inference annotation is the first frame."""
from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np
import pytest
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import davis_real_data as raw
import davis_real_eval as exp


def fixture_in_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(raw, "ROOT", tmp_path)
    base = tmp_path / "data" / "davis"
    for kind in ("images", "masks"):
        (base / kind / "mini").mkdir(parents=True, exist_ok=True)
    assets = []
    digests = {}
    for k in range(2):
        img = np.full((32, 40, 3), 120, np.uint8)
        photo = pathlib.Path("images") / "mini" / f"{k:05d}.jpg"
        assert cv2.imwrite(str(base / photo), img)
        mask = np.zeros((32, 40), np.uint8)
        mask[9:17, 12 + k:20 + k] = 1
        label = pathlib.Path("masks") / "mini" / f"{k:05d}.png"
        Image.fromarray(mask).save(base / label)
        assets.append({"jpg": str(photo), "png": str(label)})
        digests[str(photo)] = raw.sha256((base / photo).read_bytes())
        digests[str(label)] = raw.sha256((base / label).read_bytes())
    row = {"name": "mini", "split": "dev", "assets": assets}
    queries, gold = raw.query_and_gold(base, [row])
    (base / "initial_queries.json").write_text(json.dumps(queries))
    config = {"dataset_dir": "data/davis", "dev_scenes": 1, "test_scenes": 0}
    (base / "manifest.json").write_text(json.dumps({
        "config_sha256": raw.cfg_hash(config), "rows": [row], "sha256": digests
    }))
    # Explicitly remove future labels: a legal prediction audit must still pass.
    (base / assets[1]["png"]).unlink()
    return base, config, queries


def test_prediction_audit_verifies_initial_mask_but_never_future_mask(tmp_path, monkeypatch):
    base, config, queries = fixture_in_memory(tmp_path, monkeypatch)
    assert len(raw.check(config, include_future_masks_and_gold=False)["rows"]) == 1

    query_path = base / queries["mini"]["query_mask"]
    before = query_path.read_bytes()
    assert cv2.imwrite(str(query_path), np.zeros((32, 40), np.uint8))
    with pytest.raises(RuntimeError, match="DAVIS_FROZEN_INITIAL_QUERY_MASK_DRIFT"):
        raw.check(config, include_future_masks_and_gold=False)
    query_path.write_bytes(before)

    altered = json.loads((base / "initial_queries.json").read_text())
    altered["mini"]["center"][0] += 2.0
    (base / "initial_queries.json").write_text(json.dumps(altered))
    with pytest.raises(RuntimeError, match="DAVIS_FROZEN_INITIAL_QUERY_METADATA_DRIFT"):
        raw.check(config, include_future_masks_and_gold=False)


def test_scores_require_bytes_frozen_prior_to_oracle(tmp_path):
    path = tmp_path / "test_predictions.json"
    path.write_text('{"sample": 1}', encoding="utf8")
    with pytest.raises(RuntimeError, match="UNPINNED_PREDICTION_REFUSED"):
        exp.require_pinned_prediction(path)
    exp.pin_prediction_bytes(path)
    exp.require_pinned_prediction(path)
    path.write_text('{"sample": 2}', encoding="utf8")
    with pytest.raises(RuntimeError, match="PREDICTION_SHA256_DRIFT"):
        exp.require_pinned_prediction(path)
    with pytest.raises(RuntimeError, match="EXISTING_LOCAL_MEDIA_BYTES_DRIFT"):
        exp.pin_prediction_bytes(path)
