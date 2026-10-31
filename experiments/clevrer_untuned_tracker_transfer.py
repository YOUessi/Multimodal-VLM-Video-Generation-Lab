"""Untuned external CLEVRER domain-transfer check of a procedural color tracker.

The tracker was developed on DIFFERENT locally rendered moving shapes.
No colors, thresholds, confidence gates, or direction mappings are tuned on
CLEVRER labels. The complete 200 previously inspected public MVBench moving
direction questions are evaluated. Unsupported color queries, missing target
centroids and ambiguous displacements ABSTAIN; they never disappear from the
denominator. Fallback predictions come only from already frozen 500M/2.2B
MVBench 3/5-frame raw model evaluation, so no new model inference is performed.

This is a diagnostic on an already-used benchmark, NOT an independent blind
real-video test, and CLEVRER is simulated, not natural footage.
"""
from __future__ import annotations
import argparse,json,statistics,time
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
from target_motion_tracking import measured_trace,infer_direction
from mvbench_capacity_temporal import paired_difference
from object_motion_counterfactual_data import LABELS

def classify_option(text):
    s=" ".join(str(text).lower().strip(" .!?").split())
    if "stationary" in s:return "stationary"
    if "up" in s and "right" in s:return "up_right"
    if "up" in s and "left" in s:return "up_left"
    if "down" in s and "right" in s:return "down_right"
    if "down" in s and "left" in s:return "down_left"
    return None

def choose_index(options,direction):
    labels=[classify_option(x) for x in options]
    matches=[i for i,x in enumerate(labels) if x==direction]
    return matches[0] if len(matches)==1 else None

def track_external(row,threshold_px,gate_px):
    start=time.perf_counter()
    try:
        trace=measured_trace(ROOT/row["video"],row["question"],3)
        guess=infer_direction(trace,threshold_px)
        if guess is None:
            reason=("TARGET_NOT_FOUND" if trace["missing_points"] else "UNCERTAIN_IMAGE_DISPLACEMENT")
            accepted=False
        else:
            dx,dy=trace["delta_xy"]
            # Conservative production tool gate: skip stationary and small
            # source-image motions (which could be camera jitter or mislocalization).
            accepted=(guess!="stationary" and
                      abs(dx)>=gate_px and abs(dy)>=gate_px and trace["missing_points"]==0)
            reason="ACCEPTED_MOVING" if accepted else "GATE_REJECTED"
        index=choose_index(row["choices"],guess) if guess is not None else None
        if index is None:
            accepted=False
            if reason=="ACCEPTED_MOVING":reason="DIRECTION_NOT_IN_CHOICES"
        return {"pred_direction":guess,"pred_letter":"ABCDEFGH"[index] if index is not None else None,
            "accepted":accepted,"reason":reason,
            "pixel_delta_xy":trace["delta_xy"],"missing":trace["missing_points"],
            "elapsed_seconds":round(time.perf_counter()-start,5)}
    except Exception as error:
        return {"pred_direction":None,"pred_letter":None,"accepted":False,
            "reason":type(error).__name__,"error_detail":str(error)[:180],
            "pixel_delta_xy":None,"missing":None,
            "elapsed_seconds":round(time.perf_counter()-start,5)}

def audit(cfg):
    official=json.loads((ROOT/cfg["public_manifest"]).read_text(encoding="utf8"))
    if official["manifest_sha256"]!=cfg["pinned_manifest_sha"]:
        raise RuntimeError("PUBLIC_SOURCE_SHA_DRIFT")
    source=[r for r in official["rows"] if r["task"]=="moving_direction"]
    if len(source)!=200:raise RuntimeError("EXPECTED_FULL_200_CLEVRER_SOURCE")
    ids=[r["key"] for r in source]
    if len(set(ids))!=200:raise RuntimeError("DUPLICATE_PUBLIC_SOURCE_ID")
    if any(classify_option(r["choices"]["0"] if isinstance(r["choices"],dict) else r["choices"][0]) is None
           and not any(classify_option(x) is not None for x in r["choices"]) for r in source):
        raise RuntimeError("UNRECOGNIZED_PUBLIC_OPTIONS")
    prior={}
    for name in ("500m","2b"):
        result=json.loads((ROOT/cfg["model_results_dir"]/(name+".json")).read_text(encoding="utf8"))
        if result["manifest_sha256"]!=official["manifest_sha256"]:
            raise RuntimeError("MODEL_COMPARISON_SOURCE_SHA_DRIFT")
        model={(r["key"],r["frames"]):r for r in result["predictions"]
               if r["task"]=="moving_direction" and r["frames"] in (3,5)}
        if len(model)!=400:raise RuntimeError("FROZEN_MODEL_OUTPUT_COUNT_INCOMPLETE")
        prior[name]=model
    rows=[]
    for i,row in enumerate(source):
        recorded=track_external(row,cfg["direction_deadzone_px"],cfg["min_gate_displacement_px"])
        gold_letter=row["gold"]
        score=recorded["pred_letter"]==gold_letter
        record={"key":row["key"],"source_video":row["video"],
            "source_question":row["question"],"target_color_query":row["question"].lower(),
            "gold_direction":classify_option(row["choices"][ord(gold_letter)-ord("A")]),
            "cv_prediction":recorded,"cv_correct":score,
            "baseline_model":{},"gated_model":{}}
        for model_name,preds in prior.items():
            record["baseline_model"][model_name]={}
            record["gated_model"][model_name]={}
            for n in (3,5):
                base=preds[(row["key"],n)]["pred"]
                gated=(recorded["pred_letter"] if recorded["accepted"] else base)
                record["baseline_model"][model_name][str(n)]=base
                record["gated_model"][model_name][str(n)]=gated
        rows.append(record)
        if (i+1)%40==0:print("UNTUNED_CLEVRER_TRANSFER",i+1,"/",len(source),flush=True)
    out=ROOT/cfg["output_dir"]
    out.mkdir(parents=True,exist_ok=True)
    counters=Counter(x["cv_prediction"]["reason"] for x in rows)
    accepted=sum(x["cv_prediction"]["accepted"] for x in rows)
    attempted=sum(x["cv_prediction"]["pred_letter"] is not None for x in rows)
    correct=sum(x["cv_correct"] for x in rows)
    summary={"protocol":"Untuned pixel-color tracker from unrelated 2D shape test",
        "source":"MVBench moving_direction, 200 public CLEVRER simulated videos",
        "newly_blind":False,"no_external_label_tuning":True,
        "denominator":200,"classification_attempted":attempted,
        "gated_tool_accepts":accepted,"cv_only_correct_total":correct,
        "errors_and_abstentions":dict(counters),
        "cpu_median_seconds":round(statistics.median(x["cv_prediction"]["elapsed_seconds"] for x in rows),5),
        "paired_model_tests":{},"rows":rows}
    lines=["# Untuned source-pixel tracker transfer to CLEVRER (all 200 official videos)","",
       "This is a **previously inspected public benchmark** and CLEVRER contains simulated videos.",
       "The tracker and confidence gate were frozen using separate locally generated 2D shapes; they were not adapted to CLEVRER labels.",
       "","| Tracker property | Result |","|---|---:|",
       f"| Source QA videos | {len(rows)} |",
       f"| Tracker produced a labeled option | {attempted} |",
       f"| Conservative gate accepted source-pixel direction | {accepted} |",
       f"| CV direct correct over ALL source videos (abstention = wrong) | {correct}/200 |",
       f"| Median CPU tracker time | {summary['cpu_median_seconds']:.3f} s |",
       "","Failure/abstention reasons: "+str(dict(counters)), "",
       "## Gated fallback vs frozen VLM (not newly trained or rerun)","",
       "| Model | Frame budget | Base VLM correct | CV-gated VLM correct | Difference pp | Paired CI pp |",
       "|---|---:|---:|---:|---:|---|"]
    for name in ("500m","2b"):
        for n in (3,5):
            original={r["key"]:r["baseline_model"][name][str(n)]==next(x for x in source if x["key"]==r["key"])["gold"] for r in rows}
            gated={r["key"]:r["gated_model"][name][str(n)]==next(x for x in source if x["key"]==r["key"])["gold"] for r in rows}
            paired=paired_difference(original,gated,seed=cfg["bootstrap_seed"]+n+int(name=="2b")*11)
            base_hits=sum(original.values());gated_hits=sum(gated.values())
            summary["paired_model_tests"][name+"_"+str(n)]={"base_correct":base_hits,
                "gated_correct":gated_hits,**paired}
            lines.append(f"| {name} | {n} | {base_hits}/200 | {gated_hits}/200 | {paired['delta_pp']:+.2f} | {paired['bootstrap_95_ci_pp']} |")
    lines+=["","## Limitations",
      "- The input domain differs from the training-friendly synthetic RGB palette. Unsupported target colors trigger explicit abstention rather than silent sample filtering.",
      "- This tracker uses no object detector, camera-motion compensation, data-driven palette learning, or shape reasoning.",
      "- A color-centroid pixel displacement is not the same as physical object motion under perspective projection.",
      "- Gate thresholds and palette remain frozen. Do not inspect the scored labels and tune this public test in place.",
      "- CLEVRER results cannot substantiate general performance on real captured video.",
      "- CV-gated fallback is an engineering demonstration, not proof that the VLM internally reasons about motion."]
    (out/"UNTUNED_CLEVRER_TRANSFER.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    (out/"UNTUNED_CLEVRER_TRANSFER.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("UNTUNED_CLEVRER_TRANSFER_COMPLETE",
          json.dumps({"attempted":attempted,"accepted":accepted,"cv_correct":correct,
              "gated":summary["paired_model_tests"],"reasons":dict(counters)}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/clevrer_untuned_tracker_v1.json")
    args=p.parse_args()
    config=json.loads((ROOT/args.config).read_text(encoding="utf8"))
    audit(config)
