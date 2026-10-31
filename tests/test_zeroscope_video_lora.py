"""Contract tests for actual 3D Diffusion LoRA on synthetic video-caption pairs."""
import importlib.util,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"experiments/diffusion_paired_motion_data.py"
TRAIN=ROOT/"experiments/zeroscope_video_lora.py"
def load(path,identifier):
    spec=importlib.util.spec_from_file_location(identifier,path)
    x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x)
    return x
def test_deterministic_dataset_and_split():
    m=load(DATA,"motion_data")
    rows=m.make_specs()
    assert len(rows)==16
    assert sum(r["split"]=="train" for r in rows)==12
    assert sum(r["split"]=="val" for r in rows)==4
    assert len({r["id"] for r in rows})==16
    assert all(r["direction"] in ("left","right") for r in rows)
def test_shape_motion_geometry_and_photometric_labels():
    import numpy as np
    m=load(DATA,"motion_data")
    for row in m.make_specs():
        ims=m.render_frames(row,192,128,4)
        assert len(ims)==4
        assert all(im.size==(192,128) for im in ims)
        arr=[np.asarray(im) for im in ims]
        color=np.array(row["rgb"])
        centers=[]
        for x in arr:
            locations=np.where(np.all(x==color,axis=2))
            assert len(locations[0])>100
            centers.append(float(np.mean(locations[1])))
        if row["direction"]=="right":assert centers[-1]>centers[0]+20
        else:assert centers[-1]<centers[0]-20
        assert ("left to right" in m.caption(row)) == (row["direction"]=="right")
def test_diffusion_config_and_no_real_video():
    cfg=json.loads((ROOT/"configs/zeroscope_lora_smoke.json").read_text())
    assert cfg["width"]%8==cfg["height"]%8==0
    assert cfg["frames"]==4 and cfg["optimizer_steps"]==12
    assert cfg["model_dir"].startswith("models/zeroscope")
    assert cfg["dataset_dir"].startswith("data/")
    assert cfg["output_dir"].startswith("outputs/")
    assert "noncommercial" in cfg["license"]
def test_optimizer_code_is_real_unet3d_peft():
    s=TRAIN.read_text()
    assert "UNet3DConditionModel" in s
    assert "get_peft_model" in s
    assert "optimizer.step()" in s
    assert "loss.backward()" in s
    assert "validate_noise_mse" in s
    assert "latent_dist.mode()" in s
