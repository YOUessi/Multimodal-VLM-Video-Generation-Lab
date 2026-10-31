"""Real ZeroScope UNet3D video Diffusion LoRA optimization (synthetic-domain study).

Trainable PEFT LoRA modules are attached to the actual pretrained 3D denoiser;
VAE and CLIP text encoder are frozen and cached on CPU. A fixed-val denoising
MSE uses identical latent/noise/timestep before and after optimization.
No natural-video generalization or visual-quality gain is inferred.
"""
from __future__ import annotations
import argparse,hashlib,json,random,statistics,time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def save(path,item):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(item,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    tmp.replace(path)

def configuration(filename):
    return json.loads((ROOT/filename).read_text(encoding="utf8"))

def check_manifest(cfg):
    dataset=ROOT/cfg["dataset_dir"]
    manifest=json.loads((dataset/"manifest.json").read_text(encoding="utf8"))
    if manifest["train_count"]!=12 or manifest["val_count"]!=4:
        raise RuntimeError("PAIRED_VIDEO_SPLIT_CARDINALITY_DRIFT")
    if manifest["train_val_content_sha_overlap"]!=0:
        raise RuntimeError("SYNTHETIC_TRAIN_VAL_LEAK")
    expected=hashlib.sha256(json.dumps(manifest["samples"],sort_keys=True).encode()).hexdigest()
    if expected!=manifest["rows_sha256"]:raise RuntimeError("TRAINING_DATA_MANIFEST_SHA_DRIFT")
    for row in manifest["samples"]:
        file=ROOT/row["video"]
        digest=hashlib.sha256(file.read_bytes()).hexdigest()
        if digest!=row["sha256"]:raise RuntimeError("TRAINING_CLIP_CONTENT_CHANGED")
    return manifest

def model_integrity(cfg):
    root=ROOT/cfg["model_dir"]
    for part in cfg["required_weights"]:
        if not (root/part).is_file():
            raise RuntimeError("MISSING_PINNED_PRETRAINED_WEIGHT:"+part)
    return root

def cache_features(cfg):
    import cv2,numpy as np,torch
    from diffusers import AutoencoderKL
    from transformers import CLIPTokenizer,CLIPTextModel
    torch.manual_seed(cfg["seed"])
    manifest=check_manifest(cfg)
    model_path=model_integrity(cfg)
    out=ROOT/cfg["output_dir"]
    cache=out/"cached_features"
    cache.mkdir(parents=True,exist_ok=True)
    print("CACHE_PREPARE_LOAD_VAE_CLIP_ON_CPU",flush=True)
    vae=AutoencoderKL.from_pretrained(str(model_path),subfolder="vae",
             local_files_only=True,torch_dtype=torch.float32).eval().cpu()
    encoder=CLIPTextModel.from_pretrained(str(model_path),subfolder="text_encoder",
             local_files_only=True,torch_dtype=torch.float32).eval().cpu()
    tokenizer=CLIPTokenizer.from_pretrained(str(model_path/"tokenizer"),local_files_only=True)
    scale=float(vae.config.scaling_factor)
    import time
    t=time.perf_counter()
    cached=[]
    for index,row in enumerate(manifest["samples"]):
        dest=cache/(row["id"]+".pt")
        if dest.exists():
            value=torch.load(dest,map_location="cpu",weights_only=True)
            if (value["sample_id"]!=row["id"] or
                value["source_video_sha"]!=row["sha256"]):
                raise RuntimeError("CACHED_FEATURE_SOURCE_DRIFT")
            cached.append(dest.name)
            continue
        cap=cv2.VideoCapture(str(ROOT/row["video"]))
        if not cap.isOpened():raise RuntimeError("CANNOT_READ_SYNTHETIC_VIDEO")
        frames=[]
        try:
            while True:
                ok,frame=cap.read()
                if not ok:break
                arr=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).astype(np.float32)/127.5-1.0
                frames.append(torch.from_numpy(arr).permute(2,0,1))
        finally:cap.release()
        if len(frames)!=cfg["frames"]:raise RuntimeError("SYNTHETIC_CLIP_WRONG_FRAMES")
        image=torch.stack(frames,dim=0)
        if list(image.shape)!=[cfg["frames"],3,cfg["height"],cfg["width"]]:
            raise RuntimeError("BAD_SYNTHETIC_FRAME_SHAPE")
        text=tokenizer(row["caption"],padding="max_length",truncation=True,
                       max_length=tokenizer.model_max_length,return_tensors="pt")
        with torch.no_grad():
            latents=vae.encode(image).latent_dist.mode()*scale
            embedding=encoder(text.input_ids)[0]
        latents=latents.permute(1,0,2,3).contiguous()
        if latents.shape[:2]!=(4,cfg["frames"]):
            raise RuntimeError("UNEXPECTED_DIFFUSION_LATENT_SHAPE")
        record={"sample_id":row["id"],"source_video_sha":row["sha256"],
                "split":row["split"],"caption":row["caption"],
                "latents":latents,"text_embeddings":embedding}
        torch.save(record,dest)
        cached.append(dest.name)
        print("DIFFUSION_FEATURE_CACHED",index+1,"/",len(manifest["samples"]),flush=True)
    save(out/"feature_cache_manifest.json",{
        "dataset_sha":manifest["rows_sha256"],"count":len(cached),
        "vae_scaling_factor":scale,"files":cached,"cache_prep_seconds":round(time.perf_counter()-t,2),
        "policy":"CPU frozen VAE and CLIP, source SHA checked, no test-video gradients"})
    print("DIFFUSION_FEATURE_CACHE_COMPLETE",len(cached),flush=True)

def load_cached(cfg,split):
    import torch
    manifest=check_manifest(cfg)
    out=ROOT/cfg["output_dir"]
    cached=json.loads((out/"feature_cache_manifest.json").read_text())
    if cached["dataset_sha"]!=manifest["rows_sha256"]:
        raise RuntimeError("FROZEN_DIFFUSION_DATASET_CACHE_DRIFT")
    rows=[]
    for row in manifest["samples"]:
        if row["split"]!=split:continue
        d=torch.load(out/"cached_features"/(row["id"]+".pt"),
                     map_location="cpu",weights_only=True)
        if d["source_video_sha"]!=row["sha256"] or d["sample_id"]!=row["id"]:
            raise RuntimeError("CACHE_ENTRY_CHANGED")
        rows.append(d)
    return rows

def load_trainable(cfg):
    import torch
    from diffusers import UNet3DConditionModel
    from peft import LoraConfig,get_peft_model
    path=model_integrity(cfg)
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model=UNet3DConditionModel.from_pretrained(
        str(path),subfolder="unet",local_files_only=True,torch_dtype=dtype).to("cuda")
    model.requires_grad_(False)
    if cfg.get("gradient_checkpointing"):
        model.enable_gradient_checkpointing()
    model=get_peft_model(model,LoraConfig(
        r=cfg["rank"],lora_alpha=cfg["alpha"],lora_dropout=0.0,
        target_modules=["to_q","to_v"]))
    params=[p for p in model.parameters() if p.requires_grad]
    if not params:raise RuntimeError("PEFT_ADAPTER_HAS_NO_TRAINABLE_PARAMETERS")
    print("UNET3D_LORA_READY",
          "trainable",sum(p.numel() for p in params),
          "total",sum(p.numel() for p in model.parameters()),
          "dtype",str(dtype),flush=True)
    return model,params,dtype

def validate_noise_mse(model,scheduler,examples,dtype,cfg):
    import torch
    model.eval()
    results=[]
    for row in examples:
        # Same independently seeded noise + timestep for both measurements.
        seed=int(hashlib.sha256((row["sample_id"]+str(cfg["seed"])).encode()).hexdigest()[:8],16)
        g=torch.Generator(device="cuda").manual_seed(seed)
        lat=row["latents"].unsqueeze(0).to("cuda",dtype=dtype)
        emb=row["text_embeddings"].to("cuda",dtype=dtype)
        noise=torch.randn(lat.shape,device="cuda",dtype=dtype,generator=g)
        step=torch.tensor([cfg["validation_timestep"]],device="cuda",dtype=torch.int64)
        noisy=scheduler.add_noise(lat,noise,step)
        with torch.no_grad():
            predicted=model(noisy,step,encoder_hidden_states=emb).sample
            error=torch.nn.functional.mse_loss(predicted.float(),noise.float()).item()
        results.append(float(error))
    model.train()
    return {"per_example_mse":[round(v,6) for v in results],
            "mean_denoising_mse":round(statistics.mean(results),6),
            "count":len(results),"fixed_timestep":cfg["validation_timestep"]}

def train(cfg):
    import torch
    from diffusers import DDPMScheduler
    if not torch.cuda.is_available():raise RuntimeError("CUDA_REQUIRED")
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    examples=load_cached(cfg,"train")
    validation=load_cached(cfg,"val")
    if len(examples)!=12 or len(validation)!=4:raise RuntimeError("TRAIN_VAL_COUNT_DRIFT")
    scheduler=DDPMScheduler.from_pretrained(str(model_integrity(cfg)),subfolder="scheduler",
                                             local_files_only=True)
    if scheduler.config.prediction_type!="epsilon":
        raise RuntimeError("UNET_PREDICTION_TYPE_UNSUPPORTED")
    model,params,dtype=load_trainable(cfg)
    optimizer=torch.optim.AdamW(params,lr=cfg["learning_rate"],weight_decay=0.0)
    baseline=validate_noise_mse(model,scheduler,validation,dtype,cfg)
    print("BASELINE_VALIDATION_NOISE_MSE",baseline["mean_denoising_mse"],flush=True)
    records=[]
    for i in range(cfg["optimizer_steps"]):
        row=examples[i%len(examples)]
        latents=row["latents"].unsqueeze(0).to("cuda",dtype=dtype)
        embeddings=row["text_embeddings"].to("cuda",dtype=dtype)
        noise=torch.randn_like(latents)
        steps=torch.randint(low=0,high=scheduler.config.num_train_timesteps,
                            size=(1,),device="cuda",dtype=torch.int64)
        x=scheduler.add_noise(latents,noise,steps)
        optimizer.zero_grad(set_to_none=True)
        started=time.perf_counter()
        predictions=model(x,steps,encoder_hidden_states=embeddings).sample
        loss=torch.nn.functional.mse_loss(predictions.float(),noise.float())
        if not torch.isfinite(loss):raise RuntimeError("NONFINITE_UNET_DIFFUSION_MSE")
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(params,max_norm=1.0)
        optimizer.step()
        torch.cuda.synchronize()
        record={"step":i+1,"train_mse":round(float(loss.detach().cpu()),6),
                "grad_norm":round(float(grad.detach().cpu()),6),
                "seconds":round(time.perf_counter()-started,3),
                "clip":row["sample_id"]}
        records.append(record)
        print("ACTUAL_UNET3D_LORA_STEP",json.dumps(record),flush=True)
    tuned=validate_noise_mse(model,scheduler,validation,dtype,cfg)
    out=ROOT/cfg["output_dir"]
    adapter=out/"adapter"
    model.save_pretrained(str(adapter))
    if not (adapter/"adapter_model.safetensors").is_file():
        raise RuntimeError("UNET_LORA_ADAPTER_NOT_SAVED")
    stats={
      "time_utc":datetime.now(timezone.utc).isoformat(),
      "model":"cerspense/zeroscope_v2_576w",
      "model_revision":cfg["model_revision"],
      "license":"CC-BY-NC-4.0 research/noncommercial",
      "experiment":"Actual UNet3D LoRA optimization on synthetic paired 4-frame motion clips",
      "train_examples":len(examples),"validation_examples":len(validation),
      "train_steps":cfg["optimizer_steps"],"rank":cfg["rank"],
      "baseline_validation_denoising":baseline,"after_validation_denoising":tuned,
      "validation_mse_delta":round(tuned["mean_denoising_mse"]-baseline["mean_denoising_mse"],6),
      "trace":records,
      "peak_allocated_vram_gib":round(torch.cuda.max_memory_allocated()/2**30,3),
      "adapter":str(adapter.relative_to(ROOT)),
      "limitations":["Synthetic moving-shape clips only, not natural-video generation training",
                     "Denoising MSE is not a human perceptual quality metric",
                     "Small validation set and a single seed cannot establish generalization"]
    }
    save(out/"training_report.json",stats)
    print("UNET3D_VIDEO_DIFFUSION_LORA_COMPLETE",stats["validation_mse_delta"],
          "PEAK_VRAM",stats["peak_allocated_vram_gib"],flush=True)

def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("stage",choices=["cache","train"])
    cli.add_argument("--config",default="configs/zeroscope_lora_smoke.json")
    args=cli.parse_args()
    cfg=configuration(args.config)
    if args.stage=="cache":cache_features(cfg)
    else:train(cfg)

if __name__=="__main__":
    main()
