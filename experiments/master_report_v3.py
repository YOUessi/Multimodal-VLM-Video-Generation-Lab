"""Collect audited real STAR fine-tuning results plus prior corrected benchmarks."""
from datetime import datetime,timezone
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(rel):
    f=ROOT/rel
    if not f.exists():raise FileNotFoundError(rel)
    return json.loads(f.read_text(encoding="utf8"))
def main():
    star=read("outputs/star_sft_round2/test_comparison.json")
    manifest=read("outputs/star_sft_round2/manifest.json")
    chosen=read("outputs/star_sft_round2/chosen_seed.json")
    original=read("outputs/mvbench_round1/rescored_v2/comparison_scored.json")
    bias=read("outputs/choice_bias_round1/comparison.json")
    generation=read("outputs/diffusion_round1/generation_report.json")
    synthetic=read("outputs/multi_seed/REPLICATION.json")
    assert len(manifest["rows"])==200
    assert len({r["video"] for r in manifest["rows"]})==200
    assert star["selected_seed"]==chosen["chosen_seed"]
    assert original["parser_version"]=="mcq_v2"
    assert len(bias["comparisons"])==4
    assert len(generation["samples"])==2 and len(synthetic["records"])==3
    sha=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True,check=True,timeout=9).stdout.strip()
    results=star["stats"]
    rows=["# Multimodal VLM & Video Generation Lab — evidence report","",
       "Generated (UTC): "+datetime.now(timezone.utc).isoformat(),
       "Frozen source SHA: "+sha,"",
       "## Experiment 1: REAL-VIDEO SFT on STAR video actions","",
       "Official MVBench/STAR action_sequence annotation set, 120 train / 40 validation / 40 heldout test, video-ID disjoint.",
       "Only train videos used for optimizer gradients. Seed selected using validation 3-frame QA; test not used for checkpoint selection.",
       "The optimized STAR MP4s are pre-trimmed; original start/end metadata are retained but not used as absolute crop coordinates.",
       "","| Frames | Base | Selected real-video LoRA | Delta percentage points | Bootstrap 95% CI pp |",
       "|---|---:|---:|---:|---|"]
    for n in ("1","3","5"):
        record=results[n]
        rows.append(f"| {n} | {record['base']['correct']}/40 | {record['lora']['correct']}/40 | {record['delta_pp']:+.2f} | {record['bootstrap_95_ci_pp']} |")
    rows+=["","Chosen LoRA seed by validation: "+str(chosen["chosen_seed"]),
           "","## Experiment 2: 80 real MVBench action-antonym videos (synthetic LoRA transfer)","",
           "Results use the CORRECTED MCQ parser. The initial parser-v1 output was invalid and superseded.",
           "","| Frames | Base | Synthetic-only LoRA | Change pp |",
           "|---|---:|---:|---:|"]
    for n in ("1","3","5"):
        x=original["stats"][n]
        rows.append(f"| {n} | {x['base']['correct']}/80 | {x['synthetic_lora']['correct']}/80 | {x['delta_pp']:+.2f} |")
    rows+=["","## Experiment 3: untouched 120 real videos with option-order rotations","",
           "Rotation majority was preregistered; negative results are preserved.",
           "","| Model | Frames | Original ordering | Rotated vote | Delta pp |",
           "|---|---:|---:|---:|---:|"]
    for x in bias["comparisons"]:
        y=x["summary"]
        rows.append(f"| {x['variant']} | {x['frames']} | {y['original_correct']}/120 | {y['majority_correct']}/120 | {x['delta_pp']:+.2f} |")
    rows+=["","## Experiment 4: synthetic color video multi-seed LoRA","",
           "Three seeds reproducibly solved a simple synthetic task. NOT general real-world temporal reasoning improvement.",
           "","| Frames | Mean delta pp | Stddev pp |",
           "|---|---:|---:|"]
    for n,x in synthetic["stats"].items():
        rows.append(f"| {n} | {x['delta_mean_pp']:+.2f} | {x['delta_stddev_pp']:.2f} |")
    rows+=["","## Experiment 5: GPU Diffusion video generation","",
           "Actual generation used ZeroScope (not Wan), with research-only licensing.",
           "","| Clip | Frames | Size | Elapsed seconds | Peak allocated VRAM |",
           "|---|---:|---|---:|---:|"]
    for sample in generation["samples"]:
        rows.append(f"| {sample['id']} | {sample['frames']} | {sample['width']}x{sample['height']} | {sample['elapsed_sec']} | {sample['gpu_peak_gib']} GiB |")
    rows+=["","Human blind quality review remains pending, not filled automatically.",
           "","## Limitations",
           "- Public benchmarks can be contaminated by pretraining. 40-sample heldout test yields wide uncertainty.",
           "- STAR real-video supervised LoRA is task-specific; it is not evidence of video generation ability.",
           "- ZeroScope generation works but no human quality gain or Wan generation is claimed.",
           "- Video assets, adapters and weights remain on Tang; only code, configs and evidence summaries are published.",
           "- No test-based model selection or posthoc selective reporting."]
    output=ROOT/"outputs/final_project"
    output.mkdir(parents=True,exist_ok=True)
    (output/"RESULTS.md").write_text("\n".join(rows)+"\n",encoding="utf8")
    paths=["outputs/star_sft_round2/test_comparison.json",
           "outputs/mvbench_round1/rescored_v2/comparison_scored.json",
           "outputs/choice_bias_round1/comparison.json",
           "outputs/multi_seed/REPLICATION.json",
           "outputs/diffusion_round1/generation_report.json"]
    report={"upstream_frozen_local_sha":sha,"generated_utc":datetime.now(timezone.utc).isoformat(),
            "star_test":results,"chosen_val_seed":chosen,"public_real_video_transfer":original["stats"],
            "real120_order_bias":bias["comparisons"],"synthetic_seed":synthetic["stats"],
            "video_generation_count":len(generation["samples"]),
            "human_quality_scored":False,"wan_generated":False,
            "sha256":{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in paths}}
    (output/"MANIFEST.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("FINAL_EVIDENCE_REPORT_READY",output/"RESULTS.md",flush=True)
if __name__=="__main__":main()
