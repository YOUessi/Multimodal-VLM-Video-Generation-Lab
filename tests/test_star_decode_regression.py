"""Offline regression coverage for STAR real video metadata/decoding mismatch."""
from pathlib import Path
import importlib.util
import subprocess
import pytest

FILE=Path(__file__).resolve().parents[1]/"experiments"/"star_real_sft.py"
SPEC=importlib.util.spec_from_file_location("star_sft_decode_fix",FILE)
module=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

def test_ffprobe_uses_actual_decoded_count(monkeypatch):
    recorded={}
    def fake_run(command,**options):
        recorded["args"]=command
        assert options["check"] is True
        return subprocess.CompletedProcess(command,0,stdout="437\n",stderr="")
    monkeypatch.setattr(module.subprocess,"run",fake_run)
    count=module.decoded_frame_count(Path("/fake/metadata-reports-595.mp4"))
    assert count==437
    assert "-count_frames" in recorded["args"]
    assert any("nb_read_frames" in x for x in recorded["args"])

def test_ffprobe_rejects_invalid_counts(monkeypatch):
    def fake_run(cmd,**options):
        return subprocess.CompletedProcess(cmd,0,stdout="N/A\n",stderr="")
    monkeypatch.setattr(module.subprocess,"run",fake_run)
    with pytest.raises(RuntimeError,match="INVALID_DECODED_FRAME_COUNT"):
        module.decoded_frame_count(Path("/tmp/broken.mp4"))

def test_sample_with_actual_decoded_length(tmp_path):
    import cv2,numpy as np
    clip=tmp_path/"frames.mp4"
    writer=cv2.VideoWriter(str(clip),cv2.VideoWriter_fourcc(*"mp4v"),6,(96,64))
    assert writer.isOpened()
    for i in range(30):
        img=np.full((64,96,3),i*6,dtype=np.uint8)
        writer.write(img)
    writer.release()
    assert module.decoded_frame_count(clip)==30
    images=module.extract_frames(clip,3,decoded_frames=30)
    assert len(images)==3
    assert all(im.size==(96,64) for im in images)

def test_star_round2_config_pin():
    import json
    cfg=json.loads((FILE.parents[1]/"configs/star_real_sft_round2.json").read_text())
    assert cfg["split_sizes"]=={"train":120,"val":40,"test":40}
    assert cfg["source_annotations_sha256"]=="c0379b130cef91bd2ccbe8db062437a564fc31a5c5f4807d9a7d8674bd0fa4ef"
    assert cfg["output_dir"]=="outputs/star_sft_round2"
