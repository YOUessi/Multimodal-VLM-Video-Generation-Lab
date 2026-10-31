"""Text-to-video diffusion baseline on Tang. Uses SAFETENSORS-ONLY locally cached ZeroScope weights.

This is NOT Wan generation. No subjective/video quality scores are fabricated.
"""
import argparse, csv, json, os, time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_MODEL=ROOT/"models"/"zeroscope_v2_576w_safe"
OUT=ROOT/"outputs"/"diffusion_round1"
PROMPTS=[
    {"id":"rolling_ball","seed":42,
     "text":"A bright red toy ball slowly rolls across a clean wooden table, static camera, natural lighting"},
    {"id":"turning_robot","seed":31415,
     "text":"A small white toy robot carefully turns to face the camera inside a simple studio, smooth movement, fixed viewpoint"}
]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=Path,default=DEFAULT_MODEL)
    p.add_argument("--steps",type=int,default=14)
    p.add_argument("--frames",type=int,default=12)
    p.add_argument("--height",type=int,default=320)
    p.add_argument("--width",type=int,default=576)
    args=p.parse_args()
    if args.steps<=0 or args.frames<2 or args.frames>24:raise ValueError("Invalid bounded generation settings")
    required=[args.model/"model_index.json",args.model/"text_encoder/model.safetensors",
              args.model/"unet/diffusion_pytorch_model.safetensors",
              args.model/"vae/diffusion_pytorch_model.safetensors"]
    missing=[str(x) for x in required if not x.is_file()]
    if missing:raise RuntimeError("Missing complete safe model; do not silently download: "+str(missing))
    import cv2, numpy as np, torch
    from diffusers import DiffusionPipeline,DPMSolverMultistepScheduler
    if not torch.cuda.is_available():raise RuntimeError("GPU required")
    OUT.mkdir(exist_ok=True,parents=True)
    t0=time.monotonic()
    pipe=DiffusionPipeline.from_pretrained(str(args.model),torch_dtype=torch.float16,
                                          use_safetensors=True,local_files_only=True)
    pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    if hasattr(pipe.unet,"enable_forward_chunking"):
        pipe.unet.enable_forward_chunking(chunk_size=1,dim=1)
    if hasattr(pipe,"enable_vae_slicing"):
        pipe.enable_vae_slicing()
    print("MODEL_READY",round(time.monotonic()-t0,2),flush=True)
    results=[]
    for item in PROMPTS:
        torch.cuda.reset_peak_memory_stats()
        begin=time.monotonic()
        generator=torch.Generator(device="cuda").manual_seed(item["seed"])
        result=pipe(item["text"],num_inference_steps=args.steps,
                    width=args.width,height=args.height,num_frames=args.frames,
                    generator=generator,guidance_scale=7.0)
        frames=result.frames[0]
        path=OUT/(item["id"]+".mp4")
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),6,(args.width,args.height))
        if not writer.isOpened():raise RuntimeError("Cannot open MP4 writer")
        arrays=[]
        for frame in frames:
            rgb=np.asarray(frame.convert("RGB") if hasattr(frame,"convert") else frame)
            if rgb.dtype!=np.uint8:
                rgb=np.clip(rgb*255,0,255).astype(np.uint8) if rgb.max()<=1.0 else np.clip(rgb,0,255).astype(np.uint8)
            arrays.append(rgb)
            writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        writer.release()
        cap=cv2.VideoCapture(str(path))
        decoded=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.release()
        if decoded < args.frames-1:raise RuntimeError("Video decode verification failed: "+str(path))
        # Diagnostics only: adjacent frame pixel difference is not a human quality metric.
        avg_diff=float(np.mean([np.abs(arrays[j].astype("float32")-arrays[j-1].astype("float32")).mean()
                                for j in range(1,len(arrays))]))
        rec={"id":item["id"],"prompt":item["text"],"seed":item["seed"],
             "output":str(path.relative_to(ROOT)),"frames":len(arrays),
             "height":args.height,"width":args.width,"diffusion_steps":args.steps,
             "elapsed_sec":round(time.monotonic()-begin,3),
             "gpu_peak_gib":round(torch.cuda.max_memory_allocated()/2**30,3),
             "mean_adjacent_pixel_difference":round(avg_diff,3),
             "quality_score":None,"temporal_consistency_human_score":None}
        results.append(rec)
        print("GENERATED_VIDEO",json.dumps(rec,ensure_ascii=False),flush=True)
    report={"timestamp_utc":datetime.now(timezone.utc).isoformat(),
            "model":"cerspense/zeroscope_v2_576w",
            "model_revision":"1f35e3dc3350d649f342bb7618fe34ac8fae5c9e",
            "license":"cc-by-nc-4.0 (research-only demonstration; not commercial use)",
            "status":"REAL_LOCAL_GPU_DIFFUSION_INFERENCE",
            "disclaimer":"ZeroScope baseline, not Wan; temporal pixel differences are descriptive, not quality ratings",
            "samples":results}
    target=OUT/"generation_report.json";target.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    csv_path=OUT/"human_review_blank.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as fh:
        w=csv.writer(fh)
        w.writerow(["video_id","reviewer_id","prompt_alignment_1to5","visual_quality_1to5","temporal_coherence_1to5","failure_notes"])
        for item in PROMPTS:
            w.writerow([item["id"],"","","","",""])
    print("SAVED",target,csv_path,flush=True)

if __name__=="__main__":main()
