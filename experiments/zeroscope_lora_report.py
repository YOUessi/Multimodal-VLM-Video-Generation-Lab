"""Evidence-only report for true 3D-UNet LoRA and synthetic motion video generation.

Creates an anonymous, balanced four-video review UI and a *private* arm mapping
to allow independent human ratings without exposing base/LoRA labels. No scores
are automatically assigned. Outputs and weights stay on Tang.
"""
from __future__ import annotations
import hashlib,json,random,shutil,statistics
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    tmp.replace(path)

def run(config):
    cfg=json.loads((ROOT/config).read_text(encoding="utf8"))
    folder=ROOT/cfg["output_dir"]
    train=json.loads((folder/"training_report.json").read_text(encoding="utf8"))
    gen=json.loads((folder/"generated_comparison/generation_report.json").read_text(encoding="utf8"))
    if train["train_steps"]!=cfg["optimizer_steps"] or len(gen["video_outputs"])!=4:
        raise RuntimeError("DIFFUSION_LORA_EVIDENCE_INCOMPLETE")
    if gen["blind_human_quality_complete"] is not False:
        raise RuntimeError("UNVERIFIED_HUMAN_QUALITY_SCORE")
    for row in gen["video_outputs"]:
        if row["blind_human_quality"] is not None or row["human_prompt_adherence"] is not None:
            raise RuntimeError("UNVERIFIED_QUALITY_SCORE_PRESENT")
        if not (ROOT/row["output_file"]).is_file():
            raise FileNotFoundError(row["output_file"])
    if len({(z["prompt_id"],z["variant"]) for z in gen["video_outputs"]})!=4:
        raise RuntimeError("MISSING_MATCHED_BASE_LORA_OUTPUT")
    review=folder/"blind_review"
    private=folder/"blind_private"
    review.mkdir(parents=True,exist_ok=True)
    private.mkdir(parents=True,exist_ok=True)
    rows=list(gen["video_outputs"])
    random.Random(20261101).shuffle(rows)
    blind=[];mapping=[]
    for index,row in enumerate(rows):
        ident="sample_"+str(index+1).zfill(2)
        mp4=review/"clips"/(ident+".mp4")
        mp4.parent.mkdir(parents=True,exist_ok=True)
        source=ROOT/row["output_file"]
        if not mp4.is_file():shutil.copy2(source,mp4)
        if hashlib.sha256(source.read_bytes()).hexdigest()!=hashlib.sha256(mp4.read_bytes()).hexdigest():
            raise RuntimeError("ANONYMIZED_VIDEO_BYTES_CHANGED")
        blind.append({"id":ident,"prompt":row["prompt"],"seed":row["seed"],
          "output":str(mp4.relative_to(ROOT)),"frames":row["frames"],
          "width":row["width"],"height":row["height"],
          "elapsed_sec":row["runtime_seconds"],"gpu_peak_gib":row["peak_allocated_vram_gib"]})
        mapping.append({"blind_id":ident,"variant":row["variant"],"prompt_id":row["prompt_id"]})
    private_map={"created_utc":datetime.now(timezone.utc).isoformat(),
       "review_blinding":"variant identity not displayed in reviewer HTML",
       "mapping":mapping}
    map_path=private/"variant_mapping.json"
    if map_path.is_file() and json.loads(map_path.read_text())["mapping"]!=mapping:
        raise RuntimeError("BLIND_GROUP_MAPPING_DRIFT")
    save(map_path,private_map)
    reviewdoc={"status":"REAL_LOCAL_GPU_DIFFUSION_INFERENCE",
       "samples":blind,"warning":"Reviewer UI hides model arm and no quality scores are pre-filled"}
    manifest=review/"generation_report.json"
    save(manifest,reviewdoc)
    import sys
    sys.path.insert(0,str(ROOT/"experiments"))
    from generation_review import prepare
    prepare(manifest)
    b=train["baseline_validation_denoising"]["mean_denoising_mse"]
    l=train["after_validation_denoising"]["mean_denoising_mse"]
    report={
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "source":"pretrained ZeroScope v2 576w, research-only noncommercial",
        "actual_unet3d_lora_updates":cfg["optimizer_steps"],
        "train_examples":train["train_examples"],
        "validation_examples":train["validation_examples"],
        "mean_validation_epsilon_mse_base":b,
        "mean_validation_epsilon_mse_lora":l,
        "relative_validation_mse_change_percent":round((l/b-1)*100,3),
        "trained_adapter_file":str((folder/"adapter/adapter_model.safetensors").relative_to(ROOT)),
        "generated_videos":gen["video_outputs"],
        "blind_review_html":str((review/"review.html").relative_to(ROOT)),
        "blind_variant_mapping_local_only":str(map_path.relative_to(ROOT)),
        "human_review_complete":False,
        "limitations":["All videos and labels were procedurally generated",
          "Validation is only four heldout synthetic examples",
          "Denoising MSE is not perceptual video quality",
          "Color centroid motion is an imperfect diagnostic",
          "No real-world motion generalization or benchmark improvement is claimed"]
    }
    save(folder/"EVIDENCE_REPORT.json",report)
    lines=["# Actual video Diffusion 3D-UNet LoRA experiment","",
        "ZeroScope v2 576w (CC-BY-NC-4.0), rank "+str(cfg["rank"])+
        ", real "+str(cfg["optimizer_steps"])+" optimizer steps.",
        "12 procedurally paired motion videos for training; 4 unique heldout synthetic motion clips for validation.",
        "","| Frozen validation | MSE epsilon denoising |",
        "|---|---:|",
        f"| Before LoRA | {b:.6f} |",
        f"| After LoRA | {l:.6f} |",
        f"| Relative change | {(l/b-1)*100:+.2f}% |",
        "","## Generated video examples","",
        "| Variant | Caption | Duration | Seconds | Peak GPU GiB | Color-direction proxy |",
        "|---|---|---|---:|---:|---|"]
    for row in gen["video_outputs"]:
        proxy=row["color_motion_proxy"]
        category=(proxy["direction"] if proxy["measurable"] else "NOT MEASURABLE")
        lines.append("| "+row["variant"]+" | "+row["prompt"]+" | "+str(row["frames"])+
            " frames | "+str(row["runtime_seconds"])+" | "+str(row["peak_allocated_vram_gib"])+
            " | "+category+" |")
    lines+=["","**No human prompt alignment, visual quality, or temporal-coherence scores have been collected.**",
      "Anonymous offline human review page: "+str((review/"review.html").relative_to(ROOT)),
      "","## Limitations"]+[("- "+x) for x in report["limitations"]]
    (folder/"EVIDENCE_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("REAL_DIFFUSION_LORA_EXPERIMENT_REPORT_READY",folder/"EVIDENCE_REPORT.md",flush=True)

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/zeroscope_lora_smoke.json")
    args=p.parse_args()
    run(args.config)
