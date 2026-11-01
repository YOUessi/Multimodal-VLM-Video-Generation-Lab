"""New frozen camera/illumination/occlusion shift study.

Predeclared comparisons: original HSV source-pixel tracker vs HSV with
stationary-background phase-correlation correction; frozen VLM (SmolVLM2)
3/5-frame baselines; gate-and-fallback hybrids. Truth is WORLD relative to
stationary textured background, not screen displacement. The motion tools
only receive video pixels + question, never ground truth, scene type or camera
parameters. Label-based analysis occurs in the final separate scoring step.

The test is a NEW procedurally generated synthetic holdout. It does not prove
performance on natural human-captured video or arbitrary camera motion.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,random,statistics,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from domain_shift_motion_data import MODES,DIRECTIONS,OPPOSITE,LABELS,digest
from domain_shift_camera_tracker import tracker,gated_label
from mvbench_capacity_temporal import load_model,infer,build_message,parsed_answer

def put(path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(record,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    tmp.replace(path)
def read(path):
    return json.loads((ROOT/path).read_text(encoding="utf8"))
def cfg_load(path):
    return read(path)

def manifest(cfg,verify_bytes=False):
    source=read(Path(cfg["dataset_dir"])/"manifest.json")
    if source["protocol_version"]!=cfg["version"]:
        raise RuntimeError("SOURCE_DOMAIN_SHIFT_VERSION_DRIFT")
    rows=source["rows"]
    if len(rows)!=2*(cfg["dev_pairs"]+cfg["test_pairs"]):
        raise RuntimeError("SOURCE_ROW_COUNT_DRIFT")
    expected=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    if expected!=source["source_sha256"]:
        raise RuntimeError("SOURCE_VIDEO_MANIFEST_SHA_DRIFT")
    for name,pairs in (("dev",cfg["dev_pairs"]),("test",cfg["test_pairs"])):
        subset=[r for r in rows if r["split"]==name]
        if len(subset)!=pairs*2:raise RuntimeError("SPLIT_ROW_COUNT_DRIFT")
        original=[r for r in subset if not r["reversed"]]
        combos=collections.Counter((r["variant"],r["gold_direction"]) for r in original)
        if len(combos)!=20 or any(v!=pairs//20 for v in combos.values()):
            raise RuntimeError("NUISANCE_DIRECTION_GRID_UNBALANCED")
    pairset=collections.defaultdict(list)
    for row in rows:
        pairset[row["pair_id"]].append(row)
    if len(pairset)!=cfg["dev_pairs"]+cfg["test_pairs"]:
        raise RuntimeError("PAIR_COUNT_DRIFT")
    for group in pairset.values():
        if len(group)!=2:raise RuntimeError("PAIR_REVERSED_SAMPLE_MISSING")
        a=next(x for x in group if x["reversed"] is False)
        b=next(x for x in group if x["reversed"] is True)
        if (a["choices"]!=b["choices"] or a["variant"]!=b["variant"]
            or b["gold_direction"]!=OPPOSITE[a["gold_direction"]]):
            raise RuntimeError("REVERSED_COUNTERFACTUAL_GOLD_CONTRACT_FAILED")
    devsha={r["content_sha256"] for r in rows if r["split"]=="dev"}
    testsha={r["content_sha256"] for r in rows if r["split"]=="test"}
    if devsha&testsha:raise RuntimeError("DEV_TEST_EXACT_DUPLICATE_LEAKAGE")
    if verify_bytes:
        for i,row in enumerate(rows):
            file=ROOT/row["video"]
            if not file.is_file() or digest(file)!=row["content_sha256"]:
                raise RuntimeError("SOURCE_MOTION_VIDEO_CHANGED")
            if (i+1)%80==0:print("DOMAIN_SHIFT_SHA_VERIFIED",i+1,len(rows),flush=True)
    return source

def cv_prediction(video,question,cfg,stable):
    start=time.perf_counter()
    try:
        record=tracker(ROOT/video,question,cfg,stable=stable)
        choice,reason=gated_label(record,cfg)
        output={"direction":record["direction"],
                "raw_delta_xy":record["screen_delta_xy"],
                "world_delta_xy":record["delta_xy"],
                "missing":record["missing"],"reason":record["reason"],
                "gate_reason":reason,"gated_direction":choice,
                "accepted":choice is not None,
                "bg_registration":record["registration"]}
    except Exception as error:
        output={"direction":None,"raw_delta_xy":None,"world_delta_xy":None,
               "missing":None,"reason":type(error).__name__,
               "error_detail":str(error)[:180],"gate_reason":"TRACKER_EXCEPTION",
               "gated_direction":None,"accepted":False,"bg_registration":None}
    output["cpu_seconds"]=round(time.perf_counter()-start,5)
    return output

def dev(cfg):
    source=manifest(cfg)
    rows=[r for r in source["rows"] if r["split"]=="dev"]
    results={"plain":collections.Counter(),"stabilized":collections.Counter()}
    count=collections.Counter()
    for row in rows:
        for name,stable in (("plain",False),("stabilized",True)):
            prediction=cv_prediction(row["video"],row["query"],cfg,stable)
            count[(name,row["variant"])]+=1
            if prediction["direction"] is None:results[name]["abstain"]+=1
            elif prediction["direction"]==row["gold_direction"]:
                results[name]["correct"]+=1
            else:results[name]["wrong"]+=1
            if prediction["accepted"]:results[name]["gate_accepted"]+=1
    print("SHIFT_DEVELOPMENT_ONLY",json.dumps({n:dict(r) for n,r in results.items()}),flush=True)
    # Development values may guide future hypothesis formation, but all
    # thresholds in this batch are frozen; no test-time optimization.
    return results

def scorecv(cfg):
    source=manifest(cfg,verify_bytes=False)
    test=[r for r in source["rows"] if r["split"]=="test"]
    out=ROOT/cfg["output_dir"];out.mkdir(parents=True,exist_ok=True)
    recordfile=out/"cv_predictions.json"
    old=None
    if recordfile.exists():
        old=read(Path(cfg["output_dir"])/"cv_predictions.json")
        if old["source_sha256"]!=source["source_sha256"]:
            raise RuntimeError("PREVIOUS_CV_SOURCE_SHA_DRIFT")
        if len(old["records"])!=len(test):
            raise RuntimeError("INCOMPLETE_FROZEN_CV_REPORT")
        return old
    records=[]
    for i,row in enumerate(test):
        records.append({"id":row["id"],"pair_id":row["pair_id"],
           "plain":cv_prediction(row["video"],row["query"],cfg,False),
           "stabilized":cv_prediction(row["video"],row["query"],cfg,True)})
        if (i+1)%40==0:print("DOMAIN_SHIFT_TRACKER_PROGRESS",i+1,len(test),flush=True)
    doc={"source_sha256":source["source_sha256"],
         "sample_count":len(test),"tracker_blind_to_labels":True,
         "methods":["plain","stabilized"],"records":records}
    put(recordfile,doc)
    print("FROZEN_TRACKER_PREDICTIONS_SAVED",len(records),flush=True)
    return doc

def frame_ids(count,n):
    if count<n:raise RuntimeError("TOO_FEW_VIDEO_FRAMES")
    lo=round(.05*(count-1));hi=round(.95*(count-1))
    return [lo+round((hi-lo)*i/(n-1)) for i in range(n)]

def read_frames(video,n):
    import cv2
    from PIL import Image
    cap=cv2.VideoCapture(str(ROOT/video))
    if not cap.isOpened():raise RuntimeError("VLM_SOURCE_VIDEO_NOT_OPENABLE")
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ids=frame_ids(count,n)
    images=[]
    try:
        for i in ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES,i)
            valid,bgr=cap.read()
            if not valid:raise RuntimeError("VLM_FRAME_DECODE_FAILED")
            images.append(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
    finally:cap.release()
    return images

def evaluate(cfg,model_name):
    import torch
    source=manifest(cfg)
    rows=[r for r in source["rows"] if r["split"]=="test"]
    config=cfg["models"][model_name]
    budgets=cfg["frame_budgets"][model_name]
    output=ROOT/cfg["output_dir"]
    path=output/(model_name+".json")
    partial=output/(model_name+".partial.json")
    total=len(rows)*len(budgets)
    if path.exists():
        doc=json.loads(path.read_text())
        if doc["source_sha256"]!=source["source_sha256"] or len(doc["records"])!=total:
            raise RuntimeError("MODEL_SOURCE_OR_RESULT_COUNT_DRIFT")
        print("VLM_SHIFT_ALREADY_COMPLETE",model_name,flush=True)
        return
    done={}
    if partial.exists():
        prev=json.loads(partial.read_text())
        if prev["source_sha256"]!=source["source_sha256"]:
            raise RuntimeError("PARTIAL_VLM_SHIFT_SOURCE_DRIFT")
        done={(r["id"],r["frames"]):r for r in prev["records"]}
    model,processor,dtype=load_model(config["model_dir"],config)
    torch.cuda.reset_peak_memory_stats()
    for budget in budgets:
        for row in rows:
            k=(row["id"],budget)
            if k in done:continue
            started=time.perf_counter()
            frames=read_frames(row["video"],budget)
            prepare=round(time.perf_counter()-started,5)
            message_row={"question":row["query"],"choices":row["choices"]}
            raw,seconds=infer(model,processor,dtype,
                             build_message(message_row,frames),cfg["max_new_tokens"])
            answer=parsed_answer(raw,len(row["choices"]))
            done[k]={"id":row["id"],"pair_id":row["pair_id"],"frames":budget,
                 "prediction":answer,"raw":raw,"decode_cpu_s":prepare,
                 "generate_gpu_s":seconds}
            if len(done)%20==0:
                put(partial,{"source_sha256":source["source_sha256"],"records":list(done.values()),
                             "status":"PARTIAL_NO_GROUND_TRUTH_SCORES"})
                print("DOMAIN_SHIFT_MODEL_PROGRESS",model_name,len(done),total,flush=True)
    if len(done)!=total:raise RuntimeError("MODEL_RESULTS_INCOMPLETE")
    put(path,{"model_name":model_name,
       "source_sha256":source["source_sha256"],
       "model_revision":config["revision"],
       "records":list(done.values()),
       "peak_allocated_GPU_gib":round(torch.cuda.max_memory_allocated()/2**30,3)})
    print("DOMAIN_SHIFT_MODEL_GPU_DONE",model_name,total,flush=True)

def bootstrap_scene(base,new,group_ids,seed,iterations=3000):
    if set(base)!=set(new) or set(base)!=set(group_ids):
        raise RuntimeError("PAIRED_SOURCES_NOT_IDENTICAL")
    groups=collections.defaultdict(list)
    for k in sorted(base):
        groups[group_ids[k]].append(int(new[k])-int(base[k]))
    pairids=sorted(groups)
    if not pairids:raise RuntimeError("EMPTY_BOOTSTRAP")
    mean=100*sum(sum(v) for v in groups.values())/sum(len(v) for v in groups.values())
    rng=random.Random(seed)
    sampled=[]
    for _ in range(iterations):
        choices=rng.choices(pairids,k=len(pairids))
        deltas=[x for c in choices for x in groups[c]]
        sampled.append(100*sum(deltas)/len(deltas))
    sampled.sort()
    return {"delta_pp":round(mean,3),
       "cluster_bootstrap_95ci_pp":[round(sampled[int(.025*iterations)],3),
                                    round(sampled[int(.975*iterations)],3)],
       "unit":"independent forward/reversed source scene pair"}

def choice_for_direction(row,direction):
    if direction is None:return None
    try:return "ABCDE"[row["choices"].index(LABELS[direction])]
    except (ValueError,KeyError):return None

def report(cfg):
    source=manifest(cfg)
    test=[r for r in source["rows"] if r["split"]=="test"]
    gt={r["id"]:r for r in test}
    cv=read(Path(cfg["output_dir"])/"cv_predictions.json")
    if cv["source_sha256"]!=source["source_sha256"] or len(cv["records"])!=len(test):
        raise RuntimeError("TRACKER_SOURCE_UNVERIFIED")
    cvbyid={r["id"]:r for r in cv["records"]}
    models={}
    for name in cfg["models"]:
        record=read(Path(cfg["output_dir"])/(name+".json"))
        if record["source_sha256"]!=source["source_sha256"]:
            raise RuntimeError("MODEL_SOURCE_DRIFT")
        models[name]={(r["id"],r["frames"]):r for r in record["records"]}
    summary={"source_sha256":source["source_sha256"],
         "newly_frozen_synthetic_test":True,
         "natural_video_generalization":False,
         "paired_source_scenes":cfg["test_pairs"],
         "test_clips":len(test),
         "metrics":{},"models":{},"paired":{}}
    lines=["# New independently frozen render-shift motion benchmark","",
      "This is generated 2D video, not natural camera footage. Each of 100 test source scenes has normal and reversed MP4s.",
      "Five motion classes × four appearance/camera nuisance modes, balanced; both 3-frame tool estimators use the SAME HSV color-region tracker.",
      "The second estimator removes global background shift using phase correlation, with predeclared thresholds and abstentions.",
      "Ground truth was hidden from tracker+VLM inference, and evaluated here AFTER frozen outputs were saved.",
      "","## Direct pixel-tracker coverage and correctness","",
      "| Tracker | Shift mode | Correct/total (including abstentions) | Produced direction | Gated accepted | Gated correct | Gated wrong |",
      "|---|---|---:|---:|---:|---:|---:|"]
    grouped=collections.defaultdict(list)
    for r in test:grouped[r["variant"]].append(r)
    for method in ("plain","stabilized"):
        summary["metrics"][method]={}
        for variant in MODES:
            sampled=grouped[variant]
            predictions=[cvbyid[r["id"]][method] for r in sampled]
            exact=sum(p["direction"]==r["gold_direction"] for p,r in zip(predictions,sampled))
            attempted=sum(p["direction"] is not None for p in predictions)
            accepted=sum(p["accepted"] for p in predictions)
            correctgate=sum(p["accepted"] and p["gated_direction"]==r["gold_direction"]
                           for p,r in zip(predictions,sampled))
            wronggate=accepted-correctgate
            result={"total":len(sampled),"direct_correct":exact,
                    "produced_direction":attempted,"gate_accepted":accepted,
                    "gate_correct":correctgate,"gate_wrong":wronggate}
            summary["metrics"][method][variant]=result
            lines.append(f"| {method} | {variant} | {exact}/{len(sampled)} | {attempted} | {accepted} | {correctgate} | {wronggate} |")
    lines+=["","## Frozen VLM-only and gated tool fallback", "",
       "| Model | Frames | VLM correct/200 | Plain HSV gate+VLM | Camera-stabilized gate+VLM |",
       "|---|---:|---:|---:|---:|"]
    pairid={r["id"]:r["pair_id"] for r in test}
    for name,budgets in cfg["frame_budgets"].items():
        summary["models"][name]={}
        for n in budgets:
            base={r["id"]:models[name][(r["id"],n)]["prediction"]==r["gold_letter"] for r in test}
            row={"model":name,"frames":n,"base_correct":sum(base.values())}
            for method in ("plain","stabilized"):
                gated={}
                for r in test:
                    pred=cvbyid[r["id"]][method]
                    tool=choice_for_direction(r,pred["gated_direction"]) if pred["accepted"] else None
                    choice=tool if tool is not None else models[name][(r["id"],n)]["prediction"]
                    gated[r["id"]]=choice==r["gold_letter"]
                paired=bootstrap_scene(base,gated,pairid,seed=cfg["bootstrap_seed"]+n+len(method)+len(name))
                row[method]={"correct":sum(gated.values()),**paired}
            summary["models"][name][str(n)]=row
            lines.append(f"| {name} | {n} | {row['base_correct']}/200 | {row['plain']['correct']}/200 | {row['stabilized']['correct']}/200 |")
            for method in ("plain","stabilized"):
                change=row[method]
                lines.append(f"  - {name} {n}frames {method}: {change['delta_pp']:+.2f} pp; scene-cluster 95% CI {change['cluster_bootstrap_95ci_pp']}")
    lines+=["","## Protocol caveats",
      "- Test samples were procedurally generated with known color templates, not human natural videos. No external real-world generalization claim.",
      "- Even correct-tracking tools may fail on shadows, camera rotations, zoom, parallax, occlusions or unseen object colors.",
      "- Camera registration assumes a dominant stationary textured plane and global image translation. No general 3D camera understanding.",
      "- The gate can override the VLM incorrectly, so gated precision/coverage are essential and cannot be reduced to accuracy.",
      "- Statistical intervals are scene-cluster bootstraps, counting original/reversed video pairs as one independent unit.",
      "- The model does not read ground truth or scene generation metadata; test parameters were frozen prior to outcome inspection."]
    out=ROOT/cfg["output_dir"]
    put(out/"results.json",summary)
    (out/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("DOMAIN_SHIFT_REPORT_FINAL",out/"RESULTS.md",
          "sample_count",summary["test_clips"],flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["audit","dev","tracker","model","report"])
    p.add_argument("--model",choices=["500m","2b"])
    p.add_argument("--config",default="configs/domain_shift_motion_v1.json")
    args=p.parse_args()
    cfg=cfg_load(args.config)
    if args.stage=="audit":
        x=manifest(cfg,verify_bytes=True)
        print("NEW_DOMAIN_SHIFT_MANIFEST_FROZEN",x["source_sha256"],flush=True)
    elif args.stage=="dev":dev(cfg)
    elif args.stage=="tracker":scorecv(cfg)
    elif args.stage=="model":
        if not args.model:raise ValueError("Requires --model")
        evaluate(cfg,args.model)
    else:report(cfg)
if __name__=="__main__":main()
