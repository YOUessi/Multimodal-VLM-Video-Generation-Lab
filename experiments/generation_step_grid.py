"""Matched Diffusion inference-step grid, with locally cached safetensors only.

Generate 8/20 step variants of two fixed prompts; reuse/copy frozen 14-step
baseline videos for the 8/14/20 comparison. Files remain on Tang. All human
quality scores stay NULL; this measures successful generation and cost only.
"""
from __future__ import annotations
import argparse, json, shutil, time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")
    temp.replace(path)

def read_config(path):
    return json.loads((ROOT/path).read_text(encoding="utf8"))

def verify_video(path,expected_frames,width,height):
    import cv2
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise RuntimeError("Cannot decode "+str(path))
    got=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if got<expected_frames-1 or w!=width or h!=height:
        raise RuntimeError(f"Generated MP4 mismatch {path}: {got} frames {w}x{h}")
    return got

def run(config):
    import cv2,numpy as np,torch
    from diffusers import DiffusionPipeline,DPMSolverMultistepScheduler
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    model=ROOT/config["model_dir"]
    required=[model/"model_index.json",model/"text_encoder/model.safetensors",
       model/"vae/diffusion_pytorch_model.safetensors",model/"unet/diffusion_pytorch_model.safetensors"]
    if not all(p.is_file() for p in required):
        raise RuntimeError("PINNED_WEIGHTS_NOT_PRESENT: "+str([str(p) for p in required if not p.is_file()]))
    old=json.loads((ROOT/config["baseline_report"]).read_text(encoding="utf8"))
    assert old["status"]=="REAL_LOCAL_GPU_DIFFUSION_INFERENCE"
    assert old["model_revision"]==config["model_revision"]
    pairs={p["id"]:p for p in old["samples"]}
    assert set(pairs)==set(config["prompts"])
    for name,info in config["prompts"].items():
        original=pairs[name]
        assert original["prompt"]==info["text"] and original["seed"]==info["seed"]
        assert original["diffusion_steps"]==config["baseline_steps"]
        assert original["width"]==config["width"] and original["height"]==config["height"]
        assert original["frames"]==config["frames"]
    out=ROOT/config["output_dir"]
    out.mkdir(parents=True,exist_ok=True)
    results=[]
    # Include the exactly pre-existing and frozen 14-step baseline without rerunning.
    for promptid,info in config["prompts"].items():
        olditem=pairs[promptid]
        target=out/(f"steps_{config['baseline_steps']}")/(promptid+".mp4")
        target.parent.mkdir(parents=True,exist_ok=True)
        source=ROOT/olditem["output"]
        if not target.exists():shutil.copy2(source,target)
        verify_video(target,config["frames"],config["width"],config["height"])
        results.append({**olditem,"output":str(target.relative_to(ROOT)),
                        "reused_from_frozen_baseline":True,
                        "quality_score":None,"temporal_consistency_human_score":None})
    start=time.perf_counter()
    pipe=DiffusionPipeline.from_pretrained(str(model),torch_dtype=torch.float16,
           use_safetensors=True,local_files_only=True)
    pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    if hasattr(pipe.unet,"enable_forward_chunking"):
        pipe.unet.enable_forward_chunking(chunk_size=1,dim=1)
    if hasattr(pipe,"vae") and hasattr(pipe.vae,"enable_slicing"):
        pipe.vae.enable_slicing()
    print("DIFFUSION_GRID_MODEL_READY",round(time.perf_counter()-start,3),flush=True)
    for steps in config["additional_steps"]:
        if steps==config["baseline_steps"] or not 4<=steps<=32:raise ValueError("Unapproved bounded step count")
        for promptid,info in config["prompts"].items():
            target=out/f"steps_{steps}"/(promptid+".mp4")
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                raise RuntimeError("Refusing to overwrite existing generated clip: "+str(target))
            torch.cuda.reset_peak_memory_stats()
            t=time.perf_counter()
            generator=torch.Generator(device="cuda").manual_seed(info["seed"])
            video=pipe(info["text"],width=config["width"],height=config["height"],
                       num_frames=config["frames"],num_inference_steps=steps,
                       guidance_scale=config["guidance_scale"],generator=generator)
            frames=video.frames[0]
            assert len(frames)==config["frames"]
            writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*"mp4v"),
                                   config["fps"],(config["width"],config["height"]))
            if not writer.isOpened():raise RuntimeError("MP4 writer failed")
            means=[]
            for frame in frames:
                rgb=np.asarray(frame.convert("RGB") if hasattr(frame,"convert") else frame)
                if rgb.dtype!=np.uint8:
                    rgb=np.clip(rgb*255,0,255).astype(np.uint8) if rgb.max()<=1.0 else np.clip(rgb,0,255).astype(np.uint8)
                means.append(rgb.astype(np.float32))
                writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            writer.release()
            verified=verify_video(target,config["frames"],config["width"],config["height"])
            elapsed=time.perf_counter()-t
            adjacent=[float(np.abs(means[i]-means[i-1]).mean()) for i in range(1,len(means))]
            record={"id":f"{promptid}_steps_{steps}","prompt":info["text"],
              "prompt_group":promptid,"seed":info["seed"],"output":str(target.relative_to(ROOT)),
              "frames":verified,"width":config["width"],"height":config["height"],
              "diffusion_steps":steps,"guidance_scale":config["guidance_scale"],
              "elapsed_sec":round(elapsed,3),
              "gpu_peak_gib":round(torch.cuda.max_memory_allocated()/2**30,3),
              "mean_adjacent_pixel_difference":round(float(np.mean(adjacent)),4),
              "quality_score":None,"temporal_consistency_human_score":None,
              "reused_from_frozen_baseline":False}
            results.append(record)
            print("GEN_GRID_COMPLETE",json.dumps(record,ensure_ascii=False),flush=True)
    doc={"timestamp_utc":datetime.now(timezone.utc).isoformat(),
         "model":"cerspense/zeroscope_v2_576w","model_revision":config["model_revision"],
         "license":"cc-by-nc-4.0 (research-only demonstration; noncommercial)",
         "status":"REAL_LOCAL_GPU_DIFFUSION_INFERENCE",
         "disclaimer":"Actual 8/14/20-step video outputs, not verified semantic or human-rated quality",
         "samples":results}
    write(out/"generation_report.json",doc)
    print("GENERATION_STEP_GRID_DONE",len(results),"CLIPS",out/"generation_report.json",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/diffusion_steps_grid.json")
    args=p.parse_args()
    run(read_config(args.config))
