"""Assemble independently verifiable artifacts into a job-oriented technical report."""
import argparse,json,subprocess,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(rel):
    path=ROOT/rel
    if not path.is_file():raise FileNotFoundError("Missing expected artifact: "+str(path))
    return json.loads(path.read_text(encoding="utf8"))
def build():
    mv=read("outputs/mvbench_round1/comparison.json")
    manifest=read("outputs/mvbench_round1/manifest.json")
    seed=read("outputs/multi_seed/REPLICATION.json")
    synt=read("outputs/temporal_round2/comparison.json")
    generation=read("outputs/diffusion_round1/generation_report.json")
    wan=read("outputs/temporal_round2/wan_preflight.json")
    assert mv["manifest_sha256"]==manifest["sha256"],"Real-video manifest drift"
    assert len(seed["records"])==3 and len({r["seed"] for r in seed["records"]})==3
    assert generation["status"]=="REAL_LOCAL_GPU_DIFFUSION_INFERENCE"
    assert len(generation["samples"])>=2
    assert wan["status"]=="PREFLIGHT_ONLY_NO_VIDEO_GENERATED"
    records=manifest["rows"]
    assert len(records)==80 and len({r["video"] for r in records})==80
    git=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True,check=True,timeout=8).stdout.strip()
    lines=["# Multimodal VLM & Video Generation Lab: measured results","",
        "Generated UTC: "+datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "Frozen local SHA: "+git,
        "","## Public real video understanding: MVBench action_antonym",
        "80 distinct videos selected with frozen seed from a public annotation set of 200 videos.",
        "No MVBench video or answer was used during LoRA training.",
        "","| Frames | Base | LoRA trained on synthetic colors | Delta pp | Bootstrap 95% CI pp |",
        "|---|---:|---:|---:|---|"]
    for n in ("1","3","5"):
        pair=mv["stats"][n]
        a=pair["base"]; b=pair["synthetic_lora"]
        lines.append(f"| {n} | {a['correct']}/{a['total']} | {b['correct']}/{b['total']} | {pair['delta_pp']:+.2f} | {pair['paired_bootstrap_95_ci_pp']} |")
    lines+=["","This experiment uses official MVBench annotations on publicly distributed clips.",
           "The source calls this subset split 'train', but this project reserves it as evaluation only.",
           "Multiple-choice accuracy on one subset is not overall VLM performance.",
           "","## Synthetic temporal-video LoRA: multi-seed replication","",
           "Three LoRA seeds, the SAME frozen synthetic train/val/test MP4 split.",
           "Every adapter is trained only on synthetic training videos.",
           "","| Input frames | Mean delta pp | Stddev pp | Per-seed delta pp |",
           "|---|---:|---:|---|"]
    for n,v in seed["stats"].items():
        lines.append(f"| {n} | {v['delta_mean_pp']:+.2f} | {v['delta_stddev_pp']:.2f} | {v['per_seed_deltas_pp']} |")
    lines+=["","Synthetic benchmark has a limited six-color vocabulary and a trivial visual style. No real-world benefit is inferred.",
           "","## Actual local text-to-video Diffusion generation",
           "","| Clip | Frames | Size | Seconds | Peak VRAM GiB | Human quality score |",
           "|---|---:|---|---:|---:|---|"]
    for item in generation["samples"]:
        lines.append(f"| {item['id']} | {item['frames']} | {item['width']}x{item['height']} | {item['elapsed_sec']} | {item['gpu_peak_gib']} | Pending independent human review |")
    lines+=["","Generation model: ZeroScope v2 576w safetensors, research-only noncommercial license.",
           "Generated videos stay on Tang. See local outputs/diffusion_round1/review.html for an offline blind scoring interface.",
           "The review page intentionally ships with NO prefilled scores.",
           "","## Wan full-precision feasibility",
           f"- Weight estimate: {wan['metadata_based_weights_gib']} GiB",
           f"- Available system RAM at gate: {wan['available_ram_gib']} GiB",
           f"- Safe full-precision plan: {wan['safe_full_precision_execution']}",
           "- Wan video generation was NOT executed. A smaller Diffusion model was used for the actual videos.",
           "","## Reproducibility and limits",
           "- Source code and frozen configs in GitHub; local machine outputs and weights excluded.",
           "- Synthetic task ≠ real actions; MVBench action-antonym only; public benchmark may have model pretraining contamination.",
           "- Only one public real-video evaluation sample selection; no real-video LoRA training and no human-verified custom video QA set.",
           "- No claim of generation quality without blinded human ratings.",
           "- Additional larger-scale multi-task real-video evaluation and model improvements remain future research."]
    out=ROOT/"outputs"/"round3"
    out.mkdir(parents=True,exist_ok=True)
    path=out/"PROJECT_REPORT.md"
    path.write_text("\n".join(lines)+"\n",encoding="utf8")
    summary={"timestamp_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "frozen_local_git_sha":git,"real_video_dataset":"PKU-Alignment/MVBench action_antonym",
             "real_video_count":80,"real_video_selection_sha256":manifest["sha256"],
             "real_video_comparison":mv["stats"],"synthetic_seed_replication":seed["stats"],
             "generated_video_count":len(generation["samples"]),
             "generated_video_model":"ZeroScope (not Wan)","blind_generation_scores":None,
             "wan_executed":False,"warnings":["no claims about natural-video LoRA gains","no human review completed"]}
    (out/"PROJECT_REPORT.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("REPORT_READY",path,"REAL_VIDEO_COUNT",len(records),flush=True)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["build"])
    parser.parse_args();build()
if __name__=="__main__":main()
