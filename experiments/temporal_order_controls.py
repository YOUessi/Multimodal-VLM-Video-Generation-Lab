"""Frozen matched-frame temporal evidence controls (no training).

Two ablations on the same 400 public MVBench task examples:
A. Repeat the single midpoint image 3/5 times (identical visual token budget,
   but no observed temporal progression). Original labels remain applicable.
B. Reverse the order of the 3/5 original chronological frames. This changes
   video semantics, so the original labels are INVALID for this input. Only
   report whether the model's selected option changes, never a 'reverse accuracy'.

All examples and budgets are predefined before inference; no prompt tuning.
"""
from __future__ import annotations
import argparse,json,random,statistics,time,hashlib
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
from mvbench_capacity_temporal import (load_model,open_frames,build_message,infer,
                                       parsed_answer,paired_difference)

def save_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    tmp.replace(path)

def build_images(row,frames,method):
    if frames not in (3,5):raise ValueError("Only matched 3/5 frame inputs allowed")
    if method=="center_repeated":
        middle=open_frames(row,1)[0]
        return [middle.copy() for _ in range(frames)]
    if method=="reverse_order":
        images=open_frames(row,frames)
        return list(reversed(images))
    raise ValueError("Unapproved fixed video-control method")

def manifest(config):
    baseline=json.loads((ROOT/config["source_manifest"]).read_text(encoding="utf8"))
    if baseline["manifest_sha256"]!=config["pinned_source_manifest_sha256"]:
        raise RuntimeError("SOURCE_MVBENCH_MANIFEST_HASH_CHANGED")
    if len(baseline["rows"])!=400:
        raise RuntimeError("EXPECTED_400_PUBLIC_VIDEOS")
    rows=baseline["rows"]
    tasks=Counter(x["task"] for x in rows)
    if set(tasks)!={"moving_direction","scene_transition"} or any(x!=200 for x in tasks.values()):
        raise RuntimeError("SOURCE_PUBLIC_TASK_COUNT_DRIFT")
    reference=json.loads((ROOT/config["uniform_model_results"]).read_text(encoding="utf8"))
    if reference["manifest_sha256"]!=baseline["manifest_sha256"]:
        raise RuntimeError("FROZEN_REFERENCE_MODEL_MANIFEST_DRIFT")
    if reference["model_name"]!=config["model_name"]:
        raise RuntimeError("FROZEN_REFERENCE_MODEL_ID_DRIFT")
    if config["frame_counts"]!=[3,5] or config["methods"]!=["center_repeated","reverse_order"]:
        raise RuntimeError("APPROVED_VIDEO_CAUSALITY_PROTOCOL_CHANGED")
    return baseline,reference

def run(config):
    import torch
    data,source=manifest(config)
    model_name=config["model_name"]
    details=config["models"][model_name]
    out=ROOT/config["output_dir"]
    out.mkdir(parents=True,exist_ok=True)
    dest=out/"controlled_predictions.json"
    if dest.exists():
        existing=json.loads(dest.read_text(encoding="utf8"))
        if existing["manifest_sha256"]!=data["manifest_sha256"]:
            raise RuntimeError("COMPLETE_CONTROL_MANIFEST_DRIFT")
        print("ALREADY_COMPLETE",dest,flush=True)
        return
    progress=out/"controlled_predictions.partial.json"
    existing={}
    if progress.exists():
        old=json.loads(progress.read_text(encoding="utf8"))
        if old["manifest_sha256"]!=data["manifest_sha256"]:
            raise RuntimeError("PARTIAL_CONTROL_MANIFEST_DRIFT")
        existing={x["method"]+"|"+x["task"]+"|"+x["key"]+"|"+str(x["frames"]):x for x in old["predictions"]}
    model,proc,dtype=load_model(details["model_dir"],details)
    torch.cuda.reset_peak_memory_stats()
    total=len(data["rows"])*len(config["frame_counts"])*len(config["methods"])
    for method in config["methods"]:
        for n in config["frame_counts"]:
            for row in data["rows"]:
                key=method+"|"+row["task"]+"|"+row["key"]+"|"+str(n)
                if key in existing:continue
                started=time.perf_counter()
                images=build_images(row,n,method)
                preprocessing=round(time.perf_counter()-started,4)
                raw,elapsed=infer(model,proc,dtype,build_message(row,images),config["max_new_tokens"])
                answer=parsed_answer(raw,len(row["choices"]))
                item={
                    "task":row["task"],"key":row["key"],
                    "frames":n,"method":method,
                    "pred":answer,"raw":raw,
                    "gold_for_original_video":row["gold"],
                    "valid_gold_for_this_control":method=="center_repeated",
                    "correct_if_valid":(answer==row["gold"]) if method=="center_repeated" else None,
                    "preprocessing_seconds":preprocessing,"generation_seconds":elapsed}
                existing[key]=item
                if len(existing)%20==0:
                    save_json(progress,{"manifest_sha256":data["manifest_sha256"],
                        "model_name":model_name,
                        "predictions":list(existing.values()),
                        "note":"Partial results; reversed videos have NO valid gold label"})
                    print("TEMPORAL_CONTROL_PROGRESS",len(existing),"/",total,
                          method,n,row["task"],flush=True)
    if len(existing)!=total:raise RuntimeError("INCOMPLETE_TEMPORAL_VIDEO_CONTROLS")
    final={"timestamp_utc":datetime.now(timezone.utc).isoformat(),
           "model_name":model_name,"revision":details["revision"],
           "manifest_sha256":data["manifest_sha256"],
           "total_predictions":len(existing),
           "max_allocated_gpu_gib":round(torch.cuda.max_memory_allocated()/2**30,3),
           "input_contract":{"frame_counts":config["frame_counts"],
              "methods":config["methods"],
              "repeated":"identical centered frame copied N times",
              "reverse":"original N frames, reversed order, NO ground truth accuracy"},
           "predictions":list(existing.values())}
    save_json(dest,final)
    print("TEMPORAL_CONTROL_GPU_EVALUATION_COMPLETE",len(existing),
          "PEAK_VRAM",final["max_allocated_gpu_gib"],flush=True)

def summarize(config):
    base,source=manifest(config)
    raw=json.loads((ROOT/config["output_dir"]/"controlled_predictions.json").read_text(encoding="utf8"))
    if raw["manifest_sha256"]!=base["manifest_sha256"]:
        raise RuntimeError("TEMPORAL_CONTROL_SOURCE_DRIFT")
    reference={(p["task"],p["key"],p["frames"]):p for p in source["predictions"]}
    controlled={(p["task"],p["key"],p["frames"],p["method"]):p for p in raw["predictions"]}
    result={"model":config["model_name"],"source_manifest_sha256":base["manifest_sha256"],
            "input_contract":raw["input_contract"],"results":[]}
    lines=["# Frozen temporal visual evidence controls","",
           "Strong SmolVLM2-2.2B, no task training. Two separate public MVBench tasks, 200 videos each.",
           "Original uniform 3/5-frame results are compared with repeated mid-frame and reversed-order video inputs.",
           "**Reversed videos have different temporal semantics, so no reverse accuracy is computed.**",
           "","| Task | Frames | Uniform correct | Repeat-center correct | Delta pp (95% paired CI) | Reverse-choice change rate |",
           "|---|---:|---:|---:|---|---:|"]
    for task in ("moving_direction","scene_transition"):
        for n in (3,5):
            rows=[r for r in base["rows"] if r["task"]==task]
            originals={r["key"]:reference[(task,r["key"],n)] for r in rows}
            repeats={r["key"]:controlled[(task,r["key"],n,"center_repeated")] for r in rows}
            reversed_frames={r["key"]:controlled[(task,r["key"],n,"reverse_order")] for r in rows}
            if any(p["valid_gold_for_this_control"] is not True for p in repeats.values()):
                raise RuntimeError("INVALID_REPEAT_SCORE")
            if any(p["valid_gold_for_this_control"] is not False or p["correct_if_valid"] is not None for p in reversed_frames.values()):
                raise RuntimeError("ILLEGAL_REVERSE_FRAME_ACCURACY_SCORE")
            orig_correct={k:p["correct"] for k,p in originals.items()}
            rep_correct={k:p["correct_if_valid"] for k,p in repeats.items()}
            paired=paired_difference(orig_correct,rep_correct,seed=config["bootstrap_seed"]+n)
            changed=sum(originals[k]["pred"]!=reversed_frames[k]["pred"] for k in originals)
            reverse_null=sum(reversed_frames[k]["pred"] is None for k in originals)
            orig_hits=sum(orig_correct.values());rep_hits=sum(rep_correct.values())
            rec={"task":task,"frames":n,
                 "uniform_correct":orig_hits,"repeat_center_correct":rep_hits,"total":200,
                 "repeat_minus_uniform":paired,
                 "reversed_pred_changed":changed,
                 "reverse_changed_rate":round(changed/200,4),
                 "reverse_invalid_prediction_count":reverse_null}
            result["results"].append(rec)
            lines.append(f"| {task} | {n} | {orig_hits}/200 | {rep_hits}/200 | "
                         f"{paired['delta_pp']:+.2f} pp, {paired['bootstrap_95_ci_pp']} | "
                         f"{changed}/200 |")
    lines+=["","## Interpretation and limits",
         "- A repeated mid-frame input preserves image count and approximate token budget but removes visible temporal variation; original QA labels remain valid.",
         "- For a reversed clip, the answer label generally changes. We report choice changes ONLY, not reverse accuracy.",
         "- Little choice change under reversal may suggest insensitivity to chronological order, but does not prove which attention or positional mechanism failed.",
         "- CLEVRER moving_direction is simulated video, unlike the separate scene_transition camera footage.",
         "- Public benchmark pretraining contamination remains possible. No test-driven prompt or model selection.",
         "- This is a descriptive fixed-control audit, not a new trained algorithm."]
    out=ROOT/config["output_dir"]
    save_json(out/"temporal_controls_report.json",result)
    (out/"TEMPORAL_CONTROLS_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("TEMPORAL_CONTROLS_REPORT_DONE",json.dumps(result["results"]),flush=True)

def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("stage",choices=["preflight","infer","report"])
    cli.add_argument("--config",default="configs/temporal_controls_v1.json")
    args=cli.parse_args()
    cfg=json.loads((ROOT/args.config).read_text(encoding="utf8"))
    if args.stage=="preflight":
        m,_=manifest(cfg)
        print("TEMPORAL_INPUT_CONTRACT_FROZEN",m["manifest_sha256"],
              "400 examples, two methods, two frame budgets",flush=True)
    elif args.stage=="infer":run(cfg)
    else:summarize(cfg)

if __name__=="__main__":
    main()
