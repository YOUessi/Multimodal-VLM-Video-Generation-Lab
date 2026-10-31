"""Aggregate independent frozen CPU sampling-cost and 8/14/20-step generation evidence."""
import hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def load(relative):
    p=ROOT/relative
    if not p.is_file():raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf8"))

def run():
    cost=load("outputs/frame_selection_cost_v1/selection_cost.json")
    frames=load("outputs/adaptive_sampling_v2/comparison.json")
    grid=load("outputs/diffusion_grid/generation_report.json")
    diagnostics=load("outputs/diffusion_grid/diagnostics.json")
    star=load("outputs/star_sft_round2/manifest.json")
    plan=load("outputs/adaptive_sampling_v2/sampling_plan.json")
    assert plan["star_manifest_sha256"]==star["sha256"]
    assert cost["sample_count"]==40 and cost["repeats"]==3
    assert len(grid["samples"])==6 and len(diagnostics["results"])==6
    assert grid["status"]=="REAL_LOCAL_GPU_DIFFUSION_INFERENCE"
    assert diagnostics["human_quality_review_complete"] is False
    assert sorted({s["diffusion_steps"] for s in grid["samples"]})==[8,14,20]
    assert {r["id"] for r in grid["samples"]}=={r["id"] for r in diagnostics["results"]}
    assert (ROOT/"outputs/diffusion_grid/review.html").is_file()
    sha=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                       capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    lines=["# Project 4 — frame-selection cost and Diffusion generation step grid","",
        "Created UTC: "+datetime.now(timezone.utc).isoformat(),
        "Frozen batch source: "+sha,"",
        "## CPU video sampling cost on 40 independent STAR test clips","",
        "Uniform frame selection and label-blind adaptive motion/hue selection are matched at 3 and 5 sampled frames.",
        "Each approach was timed three times in alternating order. CPU selection overhead is reported separately from prior GPU inference.",
        "","| Frames | Uniform CPU median ms | Adaptive CPU median ms | Mean cost multiplier |",
        "|---|---:|---:|---:|"]
    for n in ("3","5"):
        r=cost["by_budget"][n]
        lines.append(f"| {n} | {r['uniform']['median_s']*1000:.1f} | {r['adaptive']['median_s']*1000:.1f} | {r['relative_overhead_ratio']:.2f}x |")
    lines+=["","## Frozen video answer correctness","",
        "Paired accuracy comparison is reused from the previous unchanged 40-video test (no hyperparameter changes).",
        "","| Model | Frames | Uniform correct | Adaptive correct | Delta pp |",
        "|---|---:|---:|---:|---:|"]
    for r in frames["comparisons"]:
        lines.append(f"| {r['model']} | {r['frames']} | {r['uniform']['correct']}/40 | {r['adaptive']['correct']}/40 | {r['delta_pp']:+.2f} |")
    lines+=["","## Real Diffusion generation: matched 8, 14, 20 inference steps","",
        "ZeroScope v2 576w model, two fixed prompts and seeds; 14-step clips are the original baseline copied, not regenerated.",
        "","| Prompt group | Steps | Wall time s | Peak VRAM GiB | Frame RGB change (nonsemantic) |",
        "|---|---:|---:|---:|---:|"]
    for r in sorted(diagnostics["results"],key=lambda x:(x["prompt_group"],x["steps"])):
        lines.append(f"| {r['prompt_group']} | {r['steps']} | {r['seconds']:.2f} | {r['vram_gib']:.3f} | {r['adjacent_rgb_mae']:.3f} |")
    lines+=["","Generated videos are 576x320 and 12 frames. Human prompt alignment, visual quality and temporal coherence remain **unscored**.",
       "The offline review page at outputs/diffusion_grid/review.html supports independent reviewer CSV export.",
       "","## Interpretability and limitations",
       "- Temporal frame selection improvement is not statistically established; paired 95% intervals contain zero.",
       "- The probe-based adaptive method pays additional CPU time, so accuracy and inference cost need joint optimization.",
       "- Optical-flow, RGB difference, brightness and Laplacian variance are mechanical proxies, not semantic quality.",
       "- Video generation uses a noncommercial checkpoint, not Wan2.1. Commercial use is not authorized.",
       "- This is a frozen experimental cost/quality grid, not a human-validated preference model."]
    out=ROOT/"outputs/batch10";out.mkdir(parents=True,exist_ok=True)
    (out/"BATCH10_RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    rawfiles=["outputs/frame_selection_cost_v1/selection_cost.json",
      "outputs/adaptive_sampling_v2/comparison.json",
      "outputs/diffusion_grid/generation_report.json",
      "outputs/diffusion_grid/diagnostics.json"]
    data={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"frozen_git_sha":sha,
          "cpu_sampling_cost":cost["by_budget"],"adaptive_accuracy":frames["comparisons"],
          "generated_videos":len(grid["samples"]),"diffusion_steps":[8,14,20],
          "model":"ZeroScope research-only, not Wan",
          "human_quality_scores":None,
          "sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in rawfiles}}
    (out/"BATCH10_MANIFEST.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
    print("BATCH10_REPORT_COMPLETE",out/"BATCH10_RESULTS.md",flush=True)

if __name__=="__main__":
    run()
