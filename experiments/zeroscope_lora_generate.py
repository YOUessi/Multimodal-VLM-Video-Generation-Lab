"""Matched ZeroScope pretrained vs actual UNet3D LoRA video generation.

This is an engineering capability test on a tiny procedural motion domain.
It does NOT claim improved human perceived quality or general video generation.
All local MP4s, PEFT weights and data remain on Tang.
"""
from __future__ import annotations
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROMPTS=[
 {"id":"red_moves_right","color":"red","expected_direction":"right",
  "text":"A simple flat animation of a solid red square moving left to right on a uniform light-gray background.",
  "seed":20261102},
 {"id":"blue_moves_left","color":"blue","expected_direction":"left",
  "text":"A simple flat animation of a solid blue square moving right to left on a uniform light-gray background.",
  "seed":20261103},
]

def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    tmp.replace(path)

def direction_proxy(rgb_frames,color,min_pixels=25,dead_zone_px=5):
    """Lightweight color-centroid motion diagnostic, NOT image quality assessment."""
    import numpy as np
    centers=[]
    for arr in rgb_frames:
        rgb=np.asarray(arr,dtype=np.float32)
        if rgb.max()<=1.01:rgb=rgb*255
        red,green,blue=rgb[...,0],rgb[...,1],rgb[...,2]
        if color=="red":
            mask=(red>110)&(red>green*1.35)&(red>blue*1.3)
        elif color=="blue":
            mask=(blue>110)&(blue>red*1.25)&(blue>green*1.20)
        else:raise ValueError("Unsupported eval color")
        yy,xx=np.where(mask)
        centers.append(round(float(xx.mean()),3) if len(xx)>=min_pixels else None)
    if any(x is None for x in centers):
        return {"measurable":False,"direction":None,"centroids":centers,
                "proxy_failure":"No reliable color segment in one or more generated frames"}
    delta=centers[-1]-centers[0]
    direction="right" if delta>dead_zone_px else "left" if delta<-dead_zone_px else "stationary"
    return {"measurable":True,"direction":direction,
            "centroids":centers,"first_last_pixel_delta":round(delta,3),
            "warning":"Color centroid is a proxy, not human-judged semantic content"}

def decode_mp4(path,expected_frames,width,height):
    import cv2
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise RuntimeError("Cannot decode generated file")
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if (w,h)!=(width,height) or n<expected_frames-1:
        raise RuntimeError("Generated MP4 format mismatch")
    return {"width":w,"height":h,"frames":n}

def build_pipeline(cfg,adapter=False):
    import torch
    from diffusers import DiffusionPipeline,DPMSolverMultistepScheduler
    model=ROOT/cfg["model_dir"]
    pipe=DiffusionPipeline.from_pretrained(str(model),local_files_only=True,
            use_safetensors=True,torch_dtype=torch.float16)
    pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if adapter:
        from peft import PeftModel
        path=ROOT/cfg["output_dir"]/"adapter"
        if not (path/"adapter_model.safetensors").is_file():
            raise RuntimeError("MISSING_LORA_ADAPTER_WEIGHTS")
        pipe.unet=PeftModel.from_pretrained(pipe.unet,str(path),is_trainable=False).eval()
    pipe.enable_model_cpu_offload()
    if hasattr(pipe.unet,"enable_forward_chunking"):
        pipe.unet.enable_forward_chunking(chunk_size=1,dim=1)
    if hasattr(pipe,"enable_vae_slicing"):pipe.enable_vae_slicing()
    return pipe

def run(cfg):
    import numpy as np, cv2, torch
    if not torch.cuda.is_available():raise RuntimeError("GPU_REQUIRED")
    training=json.loads((ROOT/cfg["output_dir"]/"training_report.json").read_text())
    if training["train_steps"]!=cfg["optimizer_steps"]:
        raise RuntimeError("MODEL_TRAINING_STEPS_DRIFT")
    out=ROOT/cfg["output_dir"]/"generated_comparison"
    if (out/"generation_report.json").exists():
        raise RuntimeError("Refuse overwrite of existing matched generation")
    out.mkdir(parents=True,exist_ok=True)
    clips=[]
    for variant in ("base","lora"):
        pipe=build_pipeline(cfg,adapter=(variant=="lora"))
        for prompt in PROMPTS:
            torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            frame_set=pipe(prompt["text"],num_inference_steps=cfg["generation_steps"],
                    width=cfg["width"],height=cfg["height"],
                    num_frames=cfg["frames"],guidance_scale=cfg["generation_guidance"],
                    generator=torch.Generator(device="cuda").manual_seed(prompt["seed"]))
            frames=frame_set.frames[0]
            if len(frames)!=cfg["frames"]:raise RuntimeError("MODEL_FRAME_COUNT_CHANGED")
            rgb=[]
            file=out/variant/(prompt["id"]+".mp4")
            file.parent.mkdir(parents=True,exist_ok=True)
            writer=cv2.VideoWriter(str(file),cv2.VideoWriter_fourcc(*"mp4v"),
                                   cfg["fps"],(cfg["width"],cfg["height"]))
            if not writer.isOpened():raise RuntimeError("Could not open MP4 writer")
            for image in frames:
                array=np.asarray(image.convert("RGB") if hasattr(image,"convert") else image)
                if array.dtype!=np.uint8:
                    array=np.clip(array*255 if array.max()<=1.0 else array,0,255).astype(np.uint8)
                rgb.append(array)
                writer.write(cv2.cvtColor(array,cv2.COLOR_RGB2BGR))
            writer.release()
            specs=decode_mp4(file,cfg["frames"],cfg["width"],cfg["height"])
            proxy=direction_proxy(rgb,prompt["color"])
            measurement={
              "variant":variant,"prompt_id":prompt["id"],"prompt":prompt["text"],
              "seed":prompt["seed"],"output_file":str(file.relative_to(ROOT)),
              "runtime_seconds":round(time.perf_counter()-start,4),
              "peak_allocated_vram_gib":round(torch.cuda.max_memory_allocated()/2**30,3),
              "model":"ZeroScope pretrained UNet3D, PEFT LoRA only for variant=lora",
              "width":specs["width"],"height":specs["height"],"frames":specs["frames"],
              "diffusion_steps":cfg["generation_steps"],
              "expected_synthetic_direction":prompt["expected_direction"],
              "color_motion_proxy":proxy,
              "blind_human_quality":None,"human_prompt_adherence":None}
            clips.append(measurement)
            print("VIDEO_DIFFUSION_GENERATION",json.dumps(measurement,ensure_ascii=False),flush=True)
        del pipe
        torch.cuda.empty_cache()
    if len(clips)!=4:raise RuntimeError("Expected 2x2 generated comparison")
    write_json(out/"generation_report.json",{"timestamp_utc":datetime.now(timezone.utc).isoformat(),
       "source":"Tiny synthetic video-caption domain, ZeroScope CC-BY-NC-4.0",
       "method":"Matched base/LoRA model weights, fixed prompts, seeds, frame budget and steps",
       "baseline_denoising_val":training["baseline_validation_denoising"],
       "lora_denoising_val":training["after_validation_denoising"],
       "video_outputs":clips,"blind_human_quality_complete":False,
       "note":"Color centroid is only a crude diagnostic. No general video quality improvement claim."})
    print("MATCHED_BASE_VS_LORA_GENERATION_COMPLETE",len(clips),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/zeroscope_lora_smoke.json")
    args=p.parse_args()
    cfg=json.loads((ROOT/args.config).read_text(encoding="utf8"))
    run(cfg)

if __name__=="__main__":
    main()
