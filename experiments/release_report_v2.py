"""Publish an auditable final research report using corrected raw-output rescoring.

The legacy real-video v1 parser is INVALID for scoring and must never be used
as evidence. Reports explicitly separate synthetic video, public real video,
and video-generation capability from genuine human quality assessment.
"""
from __future__ import annotations
import hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def artifact(relative):
    p=ROOT/relative
    if not p.is_file():raise FileNotFoundError("Missing required artifact: "+str(relative))
    return json.loads(p.read_text(encoding="utf8"))
def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def build():
    corrected=artifact("outputs/mvbench_round1/rescored_v2/comparison_scored.json")
    order=artifact("outputs/choice_bias_round1/comparison.json")
    synthetic=artifact("outputs/multi_seed/REPLICATION.json")
    generation=artifact("outputs/diffusion_round1/generation_report.json")
    wan=artifact("outputs/temporal_round2/wan_preflight.json")
    first=artifact("outputs/mvbench_round1/manifest.json")
    untouched=artifact("outputs/choice_bias_round1/manifest.json")
    assert corrected["parser_version"]=="mcq_v2"
    assert corrected["manifest_sha256"]==first["sha256"]
    assert untouched["original_80_manifest_hash"]==first["sha256"]
    assert order["manifest_sha256"]==untouched["sha256"]
    assert len({r["id"] for r in first["rows"]} & {r["id"] for r in untouched["rows"]})==0
    assert len(first["rows"])==80 and len(untouched["rows"])==120
    assert len(synthetic["records"])==3
    assert len(generation["samples"])==2
    assert generation["status"]=="REAL_LOCAL_GPU_DIFFUSION_INFERENCE"
    assert wan["status"]=="PREFLIGHT_ONLY_NO_VIDEO_GENERATED"
    assert (ROOT/"outputs/diffusion_round1/review.html").is_file()
    head=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,
                        text=True,check=True,timeout=10).stdout.strip()
    reviewed=(ROOT/"outputs/diffusion_round1/human_scores.json").is_file()
    lines=["# Project 4 — measured research results","",
           "Generated UTC: "+datetime.now(timezone.utc).isoformat(),
           "Frozen source git SHA: "+head,"",
           "## 1. Public REAL video action-understanding evaluation (80 clips)","",
           "Source: PKU-Alignment/MVBench action_antonym. Public official labels; no real videos used for LoRA training.",
           "The original scoring script incorrectly marked replies like C. description invalid.",
           "All values below use the corrected MCQ-v2 parser rescored from SAVED original outputs.",
           "","| Frames | Base | Synthetic-LoRA transfer | Paired delta pp | Paired bootstrap CI pp |",
           "|---:|---:|---:|---:|---|"]
    for n in ("1","3","5"):
        x=corrected["stats"][n]
        lines.append(f"| {n} | {x['base']['correct']}/{x['base']['total']} | {x['synthetic_lora']['correct']}/{x['synthetic_lora']['total']} | {x['delta_pp']:+.2f} | {x['bootstrap_95ci_pp']} |")
    lines+=["","## 2. Previously untouched REAL videos: 120-clip option-order study","",
           "All three cyclic option rotations were preregistered. Semantic candidate indices, not option letters, are aggregated.",
           "Ties fall back deterministically to the first rotation.",
           "","| Variant | Frames | Original option order | Rotation majority | Delta pp | 95% CI pp |",
           "|---|---:|---:|---:|---:|---|"]
    for rec in order["comparisons"]:
        summary=rec["summary"]
        lines.append(f"| {rec['variant']} | {rec['frames']} | {summary['original_correct']}/{summary['total']} | {summary['majority_correct']}/{summary['total']} | {rec['delta_pp']:+.2f} | {rec['bootstrap_95ci_pp']} |")
    lines+=["","## 3. Synthetic temporal color video LoRA replication","",
           "FROZEN train/test split, three LoRA training seeds. This exercise does NOT measure natural video understanding.",
           "","| Frames | Mean improvement pp | Stddev pp | Per-seed improvements |",
           "|---:|---:|---:|---|"]
    for n,v in synthetic["stats"].items():
        lines.append(f"| {n} | {v['delta_mean_pp']:+.2f} | {v['delta_stddev_pp']:.2f} | {v['per_seed_deltas_pp']} |")
    lines+=["","## 4. Actual GPU text-to-video Diffusion generation","",
           "Model: cerspense/zeroscope_v2_576w; research-only noncommercial use.",
           "","| Clip | Frames | Resolution | Elapsed seconds | Peak allocated GPU memory |",
           "|---|---:|---|---:|---:|"]
    for r in generation["samples"]:
        lines.append(f"| {r['id']} | {r['frames']} | {r['width']}x{r['height']} | {r['elapsed_sec']} | {r['gpu_peak_gib']} GiB |")
    lines+=["", "**Human evaluation:** "+("Completed, inspect human_scores.json" if reviewed else "NOT YET DONE; video review page exists but ratings are blank."),
            "","## 5. Wan2.1 feasibility",
            f"Full-weight preflight RAM gate: {wan['safe_full_precision_execution']}. "
            "Full-weight Wan inference was NOT executed. The actual generated videos are from ZeroScope.",
            "","## Limitations and research integrity","",
            "- Synthetic test result is not a real-world model gain.",
            "- Public MVBench subset may be present in large-scale pretraining data.",
            "- Public action-antonym benchmark only; not full MVBench. An 80-sample evaluation and its 120 unseen complement are independently tracked.",
            "- Training used synthetic colors only; we do not claim meaningful natural action generalization from 48 updates.",
            "- No manual quality ratings have been fabricated; model outputs and private inputs stay on Tang.",
            "- Model answer-position bias is reported and investigated; do not cherry-pick failed prompt configurations.",
            "- Source archive from the earlier CPU prototype remains a separate artifact until imported and re-tested."]
    target=ROOT/"outputs/round4";target.mkdir(parents=True,exist_ok=True)
    (target/"FINAL_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    files=["outputs/mvbench_round1/rescored_v2/comparison_scored.json",
           "outputs/choice_bias_round1/comparison.json",
           "outputs/multi_seed/REPLICATION.json","outputs/diffusion_round1/generation_report.json"]
    manifest={"time_utc":datetime.now(timezone.utc).isoformat(),"frozen_source_sha":head,
              "real_video_eval_80":corrected["stats"],"unseen_120_choice_bias":order["comparisons"],
              "synthetic_multiseed":synthetic["stats"],"generation_model":"ZeroScope (not Wan)",
              "human_video_quality_reviewed":reviewed,"wan_executed":False,
              "artifact_sha256":{f:sha256(ROOT/f) for f in files}}
    (target/"FINAL_REPORT.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    print("RELEASE_REPORT_READY",target/"FINAL_REPORT.md",flush=True)
def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument("command",choices=["build"]);p.parse_args()
    build()
if __name__=="__main__":main()
