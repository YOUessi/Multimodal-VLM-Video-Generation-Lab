"""Matched object-centric temporal evidence experiment (frozen synthetic counterfactuals).

Conditions for each frozen test video and pretrained VLM:
  text_only, rgb3, positions_only, rgb3_positions, rgb3_wrong_order_positions.
The latter is an intentional conflicting-evidence control, NOT a placebo
representing physically valid motion. The trajectory numbers are extracted
from the SOURCE PIXELS with a deterministic RGB tracker, not from ground truth.
Ground truth is accessed only by summary/scoring code, never by prompts/tracker.

This is a controlled synthetic test; simple CV is a strong baseline, and the
experiment makes no claim of natural-video generalization or model novelty.
"""
from __future__ import annotations
import argparse, hashlib, json, random, statistics, time
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
from object_motion_counterfactual_data import LABELS, DIRECTIONS, OPPOSITE
from target_motion_tracking import (measured_trace,infer_direction,
                                    numeric_evidence,video_rgb_frames,run_preflight)
from mvbench_capacity_temporal import load_model,infer,parsed_answer,paired_difference

CONDITIONS=("text_only","rgb3","positions_only","rgb3_positions",
            "rgb3_wrong_order_positions")
SPLIT="test"

def atomic(path,item):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(item,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    tmp.replace(path)

def config_doc(cfgpath):
    return json.loads((ROOT/cfgpath).read_text(encoding="utf8"))

def sha256(file):
    h=hashlib.sha256()
    with Path(file).open("rb") as f:
        for part in iter(lambda:f.read(2**20),b""):
            h.update(part)
    return h.hexdigest()

def check_dataset(cfg,strict_sha=False):
    dataset=ROOT/cfg["dataset_dir"]
    manifest=json.loads((dataset/"manifest.json").read_text(encoding="utf8"))
    if manifest["config_version"]!=cfg["version"]:
        raise RuntimeError("DATASET_CONFIG_VERSION_DRIFT")
    rows=manifest["rows"]
    if len(rows)!=2*(cfg["dev_pairs"]+cfg["test_pairs"]):
        raise RuntimeError("GENERATED_DATASET_SIZE_CHANGED")
    if manifest["rows_sha256"]!=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest():
        raise RuntimeError("GENERATED_VIDEO_MANIFEST_SHA_CHANGED")
    test=[x for x in rows if x["split"]==SPLIT]
    if len(test)!=cfg["test_pairs"]*2:raise RuntimeError("EXPECTED_FROZEN_TEST_PAIR_COUNT")
    test_pairs=defaultdict(list)
    for row in test:
        test_pairs[row["pair_id"]].append(row)
    if len(test_pairs)!=cfg["test_pairs"] or any(len(v)!=2 for v in test_pairs.values()):
        raise RuntimeError("COUNTERFACTUAL_PAIR_INCOMPLETE")
    if any({x["reversed"] for x in v}!={True,False} for v in test_pairs.values()):
        raise RuntimeError("COUNTERFACTUAL_PAIR_DIRECTION_MISSING")
    if any(v[0]["options"]!=v[1]["options"] for v in test_pairs.values()):
        raise RuntimeError("COUNTERFACTUAL_CHOICES_NOT_MATCHED")
    for pair in test_pairs.values():
        original=next(x for x in pair if not x["reversed"])
        reverse=next(x for x in pair if x["reversed"])
        if reverse["gold_direction"]!=OPPOSITE[original["gold_direction"]]:
            raise RuntimeError("INVERTED_LABEL_NOT_COUNTERFACTUAL")
    hashes={x["video_sha256"] for x in rows if x["split"]=="dev"}
    if hashes & {x["video_sha256"] for x in test}:
        raise RuntimeError("DEV_TEST_VIDEO_SHA_LEAK")
    if strict_sha:
        for i,row in enumerate(rows):
            path=ROOT/row["video"]
            if sha256(path)!=row["video_sha256"]:
                raise RuntimeError("VIDEO_CONTENT_SHA_DRIFT:"+row["video"])
            if (i+1)%80==0:print("FROZEN_VIDEO_SHA_VERIFIED",i+1,len(rows),flush=True)
    return manifest,test,test_pairs

def mode_images_and_hint(row,mode,trace=None):
    if mode not in CONDITIONS:raise ValueError("Not an approved condition")
    has_visual=mode in ("rgb3","rgb3_positions","rgb3_wrong_order_positions")
    has_numbers=mode in ("positions_only","rgb3_positions","rgb3_wrong_order_positions")
    frames=video_rgb_frames(ROOT/row["video"],3) if has_visual else []
    if has_numbers:
        if trace is None:
            trace=measured_trace(ROOT/row["video"],row["question"],3)
        hint=numeric_evidence(trace,reverse=mode=="rgb3_wrong_order_positions")
    else:hint=None
    return frames,hint

def make_prompt(row,mode,images,hint):
    payload=[{"type":"text","text":"Choose one option using available evidence. Do not assume that every observation is reliable."}]
    if images:
        payload.append({"type":"text","text":"Three RGB video frames in chronological order follow:"})
        for k,img in enumerate(images):
            payload.extend([{"type":"text","text":f"Video frame {k+1} of {len(images)}:"},
                            {"type":"image","image":img}])
    else:
        payload.append({"type":"text","text":"No RGB video frames are supplied."})
    if hint is not None:
        payload.append({"type":"text","text":"Auxiliary automatically measured (potentially erroneous) visual coordinates: "+hint})
    elif not images:
        payload.append({"type":"text","text":"No motion measurements are supplied."})
    choices="\n".join(chr(ord("A")+i)+". "+str(v)
                      for i,v in enumerate(row["options"]))
    payload.append({"type":"text","text":row["question"]+"\n"+choices+
        "\nRespond with exactly one line: ANSWER: followed by the selected option letter."})
    return [{"role":"user","content":payload}]

def validate_tracker_and_models(cfg):
    run_preflight("configs/object_motion_v1.json",dev_only=True)
    dev=[r for r in json.loads((ROOT/cfg["dataset_dir"]/"manifest.json").read_text())["rows"]
         if r["split"]=="dev"]
    assert len(dev)==2*cfg["dev_pairs"]
    for name,details in cfg["models"].items():
        model,processor,dtype=load_model(details["model_dir"],details)
        # Pure execution checks; no hyperparameter tuning based on any
        # outputs. Do not inspect/test on the final heldout here.
        test=dev[0]
        for mode in CONDITIONS:
            images,hint=mode_images_and_hint(test,mode)
            raw,seconds=infer(model,processor,dtype,
                              make_prompt(test,mode,images,hint),cfg["max_new_tokens"])
            print("PREDECLARED_MODEL_PROTOCOL_SMOKE",name,mode,repr(raw[:110]),
                  round(seconds,3),flush=True)
        del model
        import torch
        torch.cuda.empty_cache()

def model_eval(cfg,modelname):
    import torch
    if not torch.cuda.is_available():raise RuntimeError("CUDA_NOT_AVAILABLE")
    if modelname not in cfg["models"]:raise ValueError("Unapproved model")
    manifest,test,_=check_dataset(cfg,strict_sha=False)
    target=ROOT/cfg["output_dir"]
    output=target/(modelname+".json")
    partial=target/(modelname+".partial.json")
    total=len(test)*len(CONDITIONS)
    if output.exists():
        existing=json.loads(output.read_text(encoding="utf8"))
        if existing["dataset_manifest_sha"]!=manifest["rows_sha256"] or len(existing["predictions"])!=total:
            raise RuntimeError("COMPLETED_PREDICTIONS_DID_NOT_MATCH_FROZEN_DATASET")
        print("MODEL_EVALUATION_ALREADY_COMPLETED",modelname,total,flush=True)
        return
    processed={}
    if partial.exists():
        checkpoint=json.loads(partial.read_text(encoding="utf8"))
        if (checkpoint["dataset_manifest_sha"]!=manifest["rows_sha256"] or
            checkpoint["model_name"]!=modelname):
            raise RuntimeError("PARTIAL_PREDICTION_SOURCE_DRIFT")
        processed={(r["id"],r["mode"]):r for r in checkpoint["predictions"]}
    model,processor,dtype=load_model(cfg["models"][modelname]["model_dir"],cfg["models"][modelname])
    torch.cuda.reset_peak_memory_stats()
    for mode in CONDITIONS:
        for row in test:
            key=(row["id"],mode)
            if key in processed:continue
            started=time.perf_counter()
            trace=measured_trace(ROOT/row["video"],row["question"],3) if mode in (
                "positions_only","rgb3_positions","rgb3_wrong_order_positions") else None
            images,hint=mode_images_and_hint(row,mode,trace)
            preprocess_time=round(time.perf_counter()-started,4)
            message=make_prompt(row,mode,images,hint)
            raw,inference_s=infer(model,processor,dtype,message,cfg["max_new_tokens"])
            pred=parsed_answer(raw,len(row["options"]))
            # Inference NEVER receives gold labels or metadata. The record
            # does not score anything until the final summary stage.
            processed[key]={"id":row["id"],"pair_id":row["pair_id"],
                "mode":mode,"pred_letter":pred,"raw":raw,"frames":len(images),
                "evidence_was_computed_from_pixels":trace is not None,
                "tracker_missing_count":trace["missing_points"] if trace else None,
                "tracker_delta_xy":trace["delta_xy"] if trace else None,
                "preprocess_s":preprocess_time,"inference_s":inference_s}
            if len(processed)%20==0:
                atomic(partial,{"dataset_manifest_sha":manifest["rows_sha256"],
                    "model_name":modelname,"predictions":list(processed.values()),
                    "status":"PARTIAL_NOT_SCORED"})
                print("OBJECT_TRAJECTORY_GPU_PROGRESS",modelname,len(processed),"/",total,
                      "mode",mode,flush=True)
    if len(processed)!=total:
        raise RuntimeError("MODEL_INFERENCE_COUNT_MISMATCH")
    doc={"time_utc":datetime.now(timezone.utc).isoformat(),
         "model_name":modelname,
         "source_model_revision":cfg["models"][modelname]["revision"],
         "dataset_manifest_sha":manifest["rows_sha256"],
         "conditions":list(CONDITIONS),
         "test_counterfactual_pairs":cfg["test_pairs"],
         "predictions":list(processed.values()),
         "model_peak_allocated_gpu_gib":round(torch.cuda.max_memory_allocated()/2**30,3),
         "condition_interpretation":"wrong_order_positions intentionally conflicts with RGB frame chronology"}
    atomic(output,doc)
    print("OBJECT_TRAJECTORY_MODEL_INFERENCE_DONE",modelname,total,flush=True)

def direction_from_letter(row,letter):
    if letter not in ("A","B","C","D","E"):return None
    from object_motion_counterfactual_data import LABELS
    choice=row["options"][ord(letter)-ord("A")]
    for name,description in LABELS.items():
        if description==choice:return name
    return None

def group_bootstrap_delta(original,new,group_by_id,seed,iterations=4000):
    groups=defaultdict(list)
    if set(original)!=set(new):raise RuntimeError("PAIRED_QA_SOURCE_MISMATCH")
    for key in sorted(original):
        groups[group_by_id[key]].append(int(new[key])-int(original[key]))
    ids=sorted(groups)
    point=sum(sum(x) for x in groups.values())/sum(len(x) for x in groups.values())
    rng=random.Random(seed)
    distribution=[]
    for _ in range(iterations):
        selected=rng.choices(ids,k=len(ids))
        vals=[v for ident in selected for v in groups[ident]]
        distribution.append(100*sum(vals)/len(vals))
    distribution.sort()
    return {"delta_pp":round(point*100,3),
            "cluster_bootstrap_95ci_pp":[round(distribution[int(.025*iterations)],3),
                                         round(distribution[int(.975*iterations)],3)],
            "resampling_unit":"paired original/reversed scene ID"}

def score(cfg):
    manifest,test,pairs=check_dataset(cfg,strict_sha=False)
    source={r["id"]:r for r in test}
    summary={"timestamp_utc":datetime.now(timezone.utc).isoformat(),
      "dataset_manifest_sha":manifest["rows_sha256"],
      "test_videos":len(test),"counterfactual_pairs":len(pairs),
      "controlled_synthetic_only":True,"models":{}}
    lines=["# Question-grounded temporal trajectory evidence on NEW synthetic counterfactual videos","",
       "120 independently generated base test scenes, each also reversed: 240 heldout clips.",
       "Motion classes: stationary / up-right / up-left / down-right / down-left, balanced by generation; target color and shape named in each question.",
       "Additional distractor object moves independently. All model inputs come from source pixels and question/option text, not ground-truth scene_specs.",
       "**A simple query-grounded CV color-centroid tracker is an explicit supervised-by-design domain baseline; generalization to natural videos is not demonstrated.**",
       "","| Model | Condition | Correct /240 | Parse failures | Forward/reverse causal consistency |",
       "|---|---|---:|---:|---:|"]
    for modelname in cfg["models"]:
        report=json.loads((ROOT/cfg["output_dir"]/(modelname+".json")).read_text(encoding="utf8"))
        if report["dataset_manifest_sha"]!=manifest["rows_sha256"]:
            raise RuntimeError("MODEL_DATASET_MANIFEST_DRIFT")
        predictions=report["predictions"]
        if len(predictions)!=len(test)*len(CONDITIONS):
            raise RuntimeError("MODEL_PREDICTION_COUNT_INCOMPLETE")
        results={}
        for row in predictions:
            source_row=source[row["id"]]
            correct=(row["pred_letter"]==source_row["gold_letter"])
            direction=direction_from_letter(source_row,row["pred_letter"])
            r={**row,"correct":bool(correct),"pred_direction":direction}
            results[(row["id"],row["mode"])]=r
        stats={}
        for mode in CONDITIONS:
            rows=[results[(entry["id"],mode)] for entry in test]
            if len(rows)!=240:raise RuntimeError("INCOMPLETE_MODE")
            consistent=0;both_correct=0
            for pair in pairs.values():
                fwd=next(r for r in pair if not r["reversed"])
                rev=next(r for r in pair if r["reversed"])
                l=results[(fwd["id"],mode)]
                r=results[(rev["id"],mode)]
                if l["pred_direction"] is not None and r["pred_direction"]==OPPOSITE[l["pred_direction"]]:
                    consistent+=1
                if l["correct"] and r["correct"]:
                    both_correct+=1
            stats[mode]={"correct":sum(r["correct"] for r in rows),
                         "total":len(rows),"invalid":sum(r["pred_letter"] is None for r in rows),
                         "counterfactual_consistent_pairs":consistent,
                         "both_pair_ends_correct":both_correct,
                         "median_generate_seconds":round(statistics.median(r["inference_s"] for r in rows),4)}
            lines.append(f"| {modelname} | {mode} | {stats[mode]['correct']}/240 | "+
                         f"{stats[mode]['invalid']} | {consistent}/120 |")
        base={r["id"]:results[(r["id"],"rgb3")]["correct"] for r in test}
        groups={r["id"]:r["pair_id"] for r in test}
        changes={}
        for mode in CONDITIONS:
            if mode=="rgb3":continue
            now_map={r["id"]:results[(r["id"],mode)]["correct"] for r in test}
            changes[mode]=group_bootstrap_delta(base,now_map,groups,cfg["bootstrap_seed"]+len(mode))
        summary["models"][modelname]={"stats":stats,"paired_changes_vs_rgb3":changes,
             "gpu_peak_gib":report["model_peak_allocated_gpu_gib"]}
    # Compute baseline tracker directly from source RGB pixels for every clip
    # and contrast with model results. This is NOT an oracle label feed.
    cv_res=[];tracker_wrong=[]
    for row in test:
        trace=measured_trace(ROOT/row["video"],row["question"],3)
        guess=infer_direction(trace,cfg["tracker_threshold_px"])
        cv_res.append({"id":row["id"],"expected":row["gold_direction"],
                       "cv_pred":guess,"correct":guess==row["gold_direction"],
                       "source_pixel_delta":trace["delta_xy"],
                       "missing":trace["missing_points"]})
        if guess!=row["gold_direction"]:tracker_wrong.append(row["id"])
    summary["deterministic_cv"]={"correct":len(cv_res)-len(tracker_wrong),
                                "total":len(cv_res),
                                "wrong_ids":tracker_wrong,
                                "records":cv_res}
    lines+=["","## Direct classical CV baseline","",
      f"Grounded color-centroid source-image tracker: {len(cv_res)-len(tracker_wrong)}/{len(cv_res)} correct.",
      "This tracker does not consult the answer labels. It is allowed to rely on known synthetic palette colors named in the question.",
      "","## Paired changes versus RGB-only 3-frame VLM input",
      ""]
    for name,item in summary["models"].items():
        for mode,values in item["paired_changes_vs_rgb3"].items():
            lines.append(f"- {name} / {mode}: {values['delta_pp']:+.1f} pp, paired-cluster CI {values['cluster_bootstrap_95ci_pp']}")
    lines+=["","## Important limitations",
      "- These are procedurally rendered colored shapes with a fixed camera, not natural footage or CLEVRER as a heldout unseen external benchmark.",
      "- The color tracker uses domain-specific exact hue references; this is not a general visual object detector.",
      "- Position-only prompts expose computed displacement numerically; improvement from numeric hints is not proof the VLM learned to track objects itself.",
      "- Reversed-hint condition intentionally contains conflicting data; its numerical accuracy is a stress test, not a legitimate perception method.",
      "- Bootstrap samples paired source scenes, not all individual clips as independent data.",
      "- No test driven tuning of rendering, threshold, prompts or VLM weights should follow this result."]
    out=ROOT/cfg["output_dir"]
    atomic(out/"comparison.json",summary)
    (out/"COUNTERFACTUAL_MOTION_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("COUNTERFACTUAL_MOTION_STUDY_DONE",
          json.dumps({name:details["stats"] for name,details in summary["models"].items()},ensure_ascii=False),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["audit","preflight","evaluate","report"])
    parser.add_argument("--model",choices=["500m","2b"])
    parser.add_argument("--config",default="configs/object_motion_v1.json")
    args=parser.parse_args()
    cfg=config_doc(args.config)
    if args.stage=="audit":
        manifest,test,pairs=check_dataset(cfg,strict_sha=True)
        print("COUNTERFACTUAL_SOURCE_FROZEN",manifest["rows_sha256"],"TEST_CLIPS",len(test),
              "TEST_PAIRS",len(pairs),flush=True)
    elif args.stage=="preflight":validate_tracker_and_models(cfg)
    elif args.stage=="evaluate":
        if args.model is None:raise ValueError("--model required")
        model_eval(cfg,args.model)
    else:score(cfg)

if __name__=="__main__":main()
