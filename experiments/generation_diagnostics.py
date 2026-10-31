"""Video generation diagnostics: descriptive frame-change metrics, NOT quality ratings."""
import argparse,json,statistics
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def inspect_video(path):
    import cv2,numpy as np
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise RuntimeError("Video decoder could not open "+str(path))
    images=[]
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            images.append(frame)
    finally:cap.release()
    if len(images)<2:raise RuntimeError("At least two frames required")
    lum=[float(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).mean()) for im in images]
    sharp=[float(cv2.Laplacian(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()) for im in images]
    absdiff=[];flows=[]
    for a,b in zip(images[:-1],images[1:]):
        absdiff.append(float(np.abs(a.astype(np.float32)-b.astype(np.float32)).mean()))
        g1=cv2.cvtColor(cv2.resize(a,(144,80)),cv2.COLOR_BGR2GRAY)
        g2=cv2.cvtColor(cv2.resize(b,(144,80)),cv2.COLOR_BGR2GRAY)
        f=cv2.calcOpticalFlowFarneback(g1,g2,None,.5,3,13,3,5,1.1,0)
        flows.append(float(np.median(np.sqrt(f[...,0]**2+f[...,1]**2))))
    return {"frames":len(images),"width":images[0].shape[1],"height":images[0].shape[0],
      "adjacent_rgb_mae":round(statistics.mean(absdiff),4),
      "adjacent_rgb_mae_std":round(statistics.pstdev(absdiff),4),
      "near_identical_pairs":sum(x<.35 for x in absdiff),
      "mean_brightness":round(statistics.mean(lum),4),
      "brightness_delta_std":round(statistics.pstdev([b-a for a,b in zip(lum[:-1],lum[1:])]),4),
      "median_flow_magnitude_at_144x80":round(statistics.median(flows),4),
      "mean_laplacian_sharpness":round(statistics.mean(sharp),4)}
def run(config):
    cfg=json.loads((ROOT/config).read_text())
    source=json.loads((ROOT/cfg["generation_report"]).read_text())
    if source["status"]!="REAL_LOCAL_GPU_DIFFUSION_INFERENCE":raise RuntimeError("Generation result not verified")
    results=[]
    for row in source["samples"]:
        path=ROOT/row["output"]
        if not path.is_file():raise FileNotFoundError(path)
        m=inspect_video(path)
        assert (m["frames"]>=row["frames"]-1 and m["width"]==row["width"]
                and m["height"]==row["height"]), "Video metadata mismatch"
        item={"id":row["id"],"prompt_group":row.get("prompt_group",row["id"]),
              "steps":row["diffusion_steps"],"seed":row["seed"],
              "prompt":row["prompt"],"seconds":row["elapsed_sec"],"vram_gib":row["gpu_peak_gib"],
              "human_quality":None,"human_prompt_alignment":None,"human_temporal_coherence":None,
              "relative_video_file":row["output"],**m}
        results.append(item)
        print("VIDEO_DIAGNOSTICS",row["id"],row["diffusion_steps"],flush=True)
    assert len(results)==6
    lines=["# ZeroScope Diffusion inference-cost / motion diagnostics","",
        "Fixed prompts and seeds. 8/14/20 inference steps; actual GPU generation.","",
        "Frame changes and optical flow are pixel proxies, NOT semantic/visual quality.",
        "","| Prompt | Steps | Seconds | Peak VRAM GiB | Mean RGB change | Median flow (low resolution) |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in sorted(results,key=lambda a:(a["prompt_group"],a["steps"])):
        lines.append(f"| {r['prompt_group']} | {r['steps']} | {r['seconds']:.2f} | {r['vram_gib']:.3f} | {r['adjacent_rgb_mae']:.3f} | {r['median_flow_magnitude_at_144x80']:.3f} |")
    lines+=["","Human scores are deliberately blank. Use outputs/diffusion_grid/review.html and two independent reviewers before reporting quality.",
            "Video generation checkpoint is research-only CC-BY-NC-4.0."]
    out=ROOT/cfg["output_dir"];out.mkdir(parents=True,exist_ok=True)
    report={"timestamp_utc":datetime.now(timezone.utc).isoformat(),
            "model":source["model"],"model_revision":source["model_revision"],
            "human_quality_review_complete":False,
            "meaning":"Descriptive low-level motion and flicker proxies, NOT model quality",
            "results":results}
    (out/"diagnostics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    (out/"DIAGNOSTICS.md").write_text("\n".join(lines)+"\n")
    print("GENERATION_DIAGNOSTICS_COMPLETE",len(results),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/diffusion_steps_grid.json")
    args=p.parse_args();run(args.config)
