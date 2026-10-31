"""Diffusers UNet3D may expose but not actually support checkpointing."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_pinned_3d_unet_checkpointing_compatibility():
    cfg=json.loads((ROOT/"configs/zeroscope_lora_smoke_nockpt.json").read_text())
    assert cfg["gradient_checkpointing"] is False
    assert cfg["output_dir"]=="outputs/zeroscope_video_lora_v2"
    assert cfg["version"].endswith("-nockpt")
    assert cfg["optimizer_steps"]==12
    assert "UNet3DConditionModel" in cfg["compatibility_note"]
    assert cfg["dataset_dir"]=="data/zeroscope_paired_motion_v1"
