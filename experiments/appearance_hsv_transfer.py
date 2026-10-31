"""Frozen generic-HSV tracker transfer and conservative VLM tool-fallback audit.

All 200 original CLEVRER Moving Direction questions are evaluated, with full
denominator and explicit abstentions. The tracker parameters are frozen
without observing CLEVRER labels; only already saved 500M/2.2B model outputs
are used, and no new VLM inference or training occurs.

This benchmark was previously inspected. Results are exploratory, not a new
independent unseen sample.
"""
from __future__ import annotations
import argparse,json,statistics,time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from appearance_hsv_tracker import measured_hsv_trace,full_dev_sanity
from clevrer_untuned_tracker_transfer import classify_option,choose_index
from mvbench_capacity_temporal import paired_difference

def audit(cfg):
    manifest=json.loads((ROOT/cfg["public_manifest"]).read_text())
    if manifest["manifest_sha256"]!=cfg["pinned_public_manifest_sha"]:
        raise RuntimeError("EXTERNAL_PUBLIC_SOURCE_DRIFT")
    rows=[x for x in manifest["rows"] if x["task"]=="moving_direction"]
    if len(rows)!=200 or len({x["key"] for x in rows})!=200:
        raise RuntimeError("EXPECTED_COMPLETE_200_MVBENCH_MOVEMENT_ITEMS")
    dev_counts=full_dev_sanity(cfg)
    originals={}
    for name in ("500m","2b"):
        r=json.loads((ROOT/cfg["frozen_vlm_dir"]/(name+".json")).read_text())
        if r["manifest_sha256"]!=manifest["manifest_sha256"]:
            raise RuntimeError("SOURCE_VLM_OUTPUT_DRIFT")
        subset={(x["key"],x["frames"]):x["pred"] for x in r["predictions"]
                if x["task"]=="moving_direction" and x["frames"] in (3,5)}
        if len(subset)!=400:raise RuntimeError("MISSING_FROZEN_BASELINE_ITEMS")
        originals[name]=subset
    records=[];confusion=Counter()
    for i,row in enumerate(rows):
        now=time.perf_counter()
        try:
            trace=measured_hsv_trace(ROOT/row["video"],row["question"],cfg,3)
            movement=trace["direction"]
            idx=choose_index(row["choices"],movement) if movement else None
            letter=("ABCDEFGH"[idx] if idx is not None else None)
            if movement is None:
                why=trace["reason"]
            elif idx is None:
                why="PREDICTION_DIRECTION_NOT_PRESENT_IN_CHOICES"
            else:why="DIRECTION_RECOGNIZED"
            accepted=(movement is not None and movement!="stationary" and idx is not None and
                 trace["missing"]==0 and
                 abs(trace["delta_xy"][0])>=cfg["gate_min_dxdy"] and
                 abs(trace["delta_xy"][1])>=cfg["gate_min_dxdy"])
            if not accepted and why=="DIRECTION_RECOGNIZED":
                why="LOW_DISPLACEMENT_OR_STATIONARY_GATE"
            loc={"target_color":trace["target_color"],"points":trace["points"],
                 "delta_xy":trace["delta_xy"],"direction":movement,
                 "pred_letter":letter,"accepted":accepted,"reason":why}
        except Exception as e:
            loc={"target_color":None,"points":None,"delta_xy":None,
                 "direction":None,"pred_letter":None,"accepted":False,
                 "reason":type(e).__name__,"error":str(e)[:160]}
        elapsed=time.perf_counter()-now
        confusion[loc["reason"]]+=1
        record={"key":row["key"],"qa_source_index":row["source_index"],
           "tracking":loc,"tracking_cpu_s":round(elapsed,5),
           "cv_only_correct":loc["pred_letter"]==row["gold"],
           "fallback_predictions":{}}
        for name in ("500m","2b"):
            record["fallback_predictions"][name]={}
            for n in (3,5):
                base=originals[name][(row["key"],n)]
                gated=loc["pred_letter"] if loc["accepted"] else base
                record["fallback_predictions"][name][str(n)]={
                    "model_raw_answer":base,"gated_answer":gated}
        records.append(record)
        if (i+1)%40==0:
            print("HSV_DOMAIN_TRANSFER_PROGRESS",i+1,"/200",flush=True)
    if len(records)!=200:raise RuntimeError("SOURCE_VIDEO_FILTERED")
    considered=sum(r["tracking"]["pred_letter"] is not None for r in records)
    accepted=sum(r["tracking"]["accepted"] for r in records)
    correct=sum(r["cv_only_correct"] for r in records)
    stats={"source":"full 200 CLEVRER moving_direction public questions",
           "newly_blind":False,"no_external_label_calibration":True,
           "dev_only_tracker":dev_counts,"total":200,
           "tracker_direction_option_count":considered,
           "confidence_gate_accept_count":accepted,
           "cv_only_correct_all_200":correct,
           "reason_counts":dict(confusion),
           "median_tracking_cpu_seconds":round(statistics.median(r["tracking_cpu_s"] for r in records),5),
           "model_comparisons":{},"records":records}
    lines=["# Generic HSV tracker transfer to CLEVRER — no label tuning","",
           "All 200 public CLEVRER moving-direction simulated videos. Data previously inspected; exploratory only.",
           "Fixed generic named-color hue templates, component association, full-denominator abstention.",
           "No CLEVRER label-based calibration of any tracker rule or gate.",
           "","| Metric | Value |","|---|---:|",
           f"| HSV dev tracker success / 40 | {dev_counts['matched']} |",
           f"| CLEVRER source videos | 200 |",
           f"| Pixel tracker produced an option | {considered} |",
           f"| Confidence gate accepted | {accepted} |",
           f"| CV-only correct (abstentions count as unhandled) | {correct}/200 |",
           f"| Median CPU tracker time | {stats['median_tracking_cpu_seconds']:.4f} s |",
           "","Tracking reasons: "+str(dict(confusion)),
           "","## Frozen VLM vs gated HSV tool fallback","",
           "| Model | Frame budget | Original VLM | Gated hybrid | Delta pp | Paired bootstrap CI pp |",
           "|---|---:|---:|---:|---:|---|"]
    for name in ("500m","2b"):
        for n in (3,5):
            a={};b={}
            for row,item in zip(rows,records):
                pred=item["fallback_predictions"][name][str(n)]
                a[row["key"]]=pred["model_raw_answer"]==row["gold"]
                b[row["key"]]=pred["gated_answer"]==row["gold"]
            delta=paired_difference(a,b,cfg["bootstrap_seed"]+n+int(name=="2b")*31)
            base_hits=sum(a.values());new_hits=sum(b.values())
            stats["model_comparisons"][name+"_"+str(n)]={"base_correct":base_hits,
               "gated_correct":new_hits,**delta}
            lines.append(f"| {name} | {n} | {base_hits}/200 | {new_hits}/200 | {delta['delta_pp']:+.2f} | {delta['bootstrap_95_ci_pp']} |")
    lines+=["","## Validity and restrictions",
      "- HSV thresholds and geometric confidence gate were fixed before CLEVRER scoring and were NOT updated from its labels.",
      "- A hue-name model and simple connected components can confuse target with similar colors, background or occlusion.",
      "- Accepted tool predictions may be WRONG; gate avoids some errors but is not calibrated on real natural videos.",
      "- The public CLEVRER simulated-video subset is not an unseen generalization test after earlier analysis.",
      "- All 200 source questions, including abstentions and tracking failures, stay in the denominator.",
      "- Do not claim external quality improvements unless the paired confidence interval and coverage justify it."]
    out=ROOT/cfg["output_dir"]
    out.mkdir(parents=True,exist_ok=True)
    (out/"HSV_TRANSFER.json").write_text(json.dumps(stats,indent=2,ensure_ascii=False)+"\n")
    (out/"HSV_TRANSFER.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("GENERIC_HSV_CLEVRER_TRANSFER_COMPLETE",
          json.dumps({k:stats[k] for k in ["dev_only_tracker","total","tracker_direction_option_count",
          "confidence_gate_accept_count","cv_only_correct_all_200","reason_counts",
          "model_comparisons"]},ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/appearance_hsv_transfer_v1.json")
    args=p.parse_args()
    audit(json.loads((ROOT/args.config).read_text(encoding="utf8")))
