"""Motion-aware visual-only frame selection with FIXED matched VLM input budgets.

The sampling rule is frozen before new GPU inference, never reads QA labels,
uses 12 visual probes per video, and emits 3/5 frames including 5%/95%
endpoints. Compare to historical uniform n=3/5 inputs on identical videos.
The entire 400-item public benchmark has been inspected in earlier experiments:
this is an EXPLORATORY ablation, NOT a never-seen blind generalization test.
"""
from __future__ import annotations
import argparse, hashlib, json, random, statistics, time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from mvbench_capacity_temporal import (load_model,build_message,infer,
    parsed_answer,paired_difference)

def save_json(path,item):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(item,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    temp.replace(path)

def probe_indices(decoded,number):
    if decoded<6 or number<5:raise ValueError("Insufficient frame/probe count")
    lo=round(.05*(decoded-1))
    hi=round(.95*(decoded-1))
    pts=sorted(set(lo+round((hi-lo)*k/(number-1)) for k in range(number)) | {lo,hi})
    if len(pts)<5 or pts[0]!=lo or pts[-1]!=hi:
        raise RuntimeError("INVALID_PROBE_POSITIONS")
    return pts

def features(video,decoded,candidates):
    import cv2,numpy as np
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise RuntimeError("VIDEO_OPEN_FAILED: "+str(video))
    grays=[];hists=[]
    try:
        for idx in candidates:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx))
            ok,raw=cap.read()
            if not ok:raise RuntimeError(f"VIDEO_DECODE_FAILED {video} at {idx}")
            image=cv2.resize(raw,(96,72))
            grays.append(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
            hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            hist=cv2.calcHist([hsv],[0],None,[24],[0,180])
            cv2.normalize(hist,hist)
            hists.append(hist)
    finally:cap.release()
    motion=[0.0];color=[0.0]
    for k in range(1,len(candidates)):
        motion.append(float(np.mean(np.abs(
            grays[k].astype(np.float32)-grays[k-1].astype(np.float32)))))
        color.append(float(cv2.compareHist(hists[k-1],hists[k],cv2.HISTCMP_BHATTACHARYYA)))
    def unit_scale(vals):
        top=max(vals)
        return [v/top if top>1e-10 else 0.0 for v in vals]
    m=unit_scale(motion);h=unit_scale(color)
    scores=[0.7*a+0.3*b for a,b in zip(m,h)]
    return {"scores":scores,"peak_motion":round(max(motion),4),
            "peak_color_delta":round(max(color),4)}

def choose_positions(positions,score,n,decoded):
    if n not in (3,5) or len(positions)!=len(score):
        raise ValueError("Only frozen n=3 and n=5 conditions supported")
    selected=[positions[0],positions[-1]]
    remaining=sorted(zip(positions[1:-1],score[1:-1]),
                    key=lambda x:(-x[1],x[0]))
    min_separation=max(1,round(decoded*(.14 if n==3 else .08)))
    while len(selected)<n:
        valid=[idx for idx,value in remaining if idx not in selected
               and all(abs(idx-old)>=min_separation for old in selected)]
        if not valid:
            valid=[idx for idx,_ in remaining if idx not in selected]
        if not valid:raise RuntimeError("INSUFFICIENT_DISTINCT_FRAMES")
        selected.append(valid[0])
    return sorted(selected)

def select_frame_data(row,config):
    video=ROOT/row["video"]
    count=row["decoded_frames"]
    started=time.perf_counter()
    points=probe_indices(count,config["probe_count"])
    visual=features(video,count,points)
    selected={str(n):choose_positions(points,visual["scores"],n,count)
              for n in config["frame_counts"]}
    duration=time.perf_counter()-started
    return {"key":row["key"],"task":row["task"],
         "video":row["video"],"video_sha256":row["video_sha256"],
         "decoded_frames":count,"n_probe_positions":len(points),
         "candidate_positions":points,"frame_indices":selected,
         "peak_motion":visual["peak_motion"],
         "peak_color_delta":visual["peak_color_delta"],
         "selection_seconds":round(duration,6)}

def load_config(filename):
    return json.loads((ROOT/filename).read_text(encoding="utf8"))

def plan(config):
    source=json.loads((ROOT/config["source_manifest"]).read_text(encoding="utf8"))
    if source["manifest_sha256"]!=config["pinned_source_sha"]:
        raise RuntimeError("PUBLIC_EVAL_DATASET_SHA_DRIFT")
    if len(source["rows"])!=400:raise RuntimeError("EXPECTED_FULL_400_ITEM_DATASET")
    out=ROOT/config["output_dir"]
    target=out/"frame_plan.json"
    if target.is_file():
        old=json.loads(target.read_text(encoding="utf8"))
        if old["source_manifest_sha"]!=source["manifest_sha256"] or old["algorithm_version"]!=config["version"]:
            raise RuntimeError("FROZEN_MOTION_SAMPLE_PLAN_DRIFT")
        fingerprint=hashlib.sha256(json.dumps(old["records"],sort_keys=True).encode()).hexdigest()
        if old["frame_plan_sha"]!=fingerprint or len(old["records"])!=400:
            raise RuntimeError("CACHED_PLAN_SHA_OR_COUNT_DRIFT")
        return source,old
    rows=[]
    for i,row in enumerate(source["rows"]):
        rows.append(select_frame_data(row,config))
        if (i+1)%40==0:
            print("MOTION_PROBE_PREPARED",i+1,"/400",flush=True)
    selected={r["key"] for r in rows}
    if len(selected)!=400:raise RuntimeError("REPEATED_QA_SOURCE")
    digest=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    output={"algorithm_version":config["version"],
            "source_manifest_sha":source["manifest_sha256"],
            "frame_plan_sha":digest,"frame_counts":config["frame_counts"],
            "weights":{"luma_motion":0.7,"hue_histogram_change":0.3},
            "probe_count":config["probe_count"],"question_labels_used_for_sampling":False,
            "is_new_unseen_test":False,
            "records":rows}
    save_json(target,output)
    print("FROZEN_LABEL_BLIND_MOTION_PLAN",digest,flush=True)
    return source,output

def frames_at_positions(video,positions):
    import cv2
    from PIL import Image
    cap=cv2.VideoCapture(str(ROOT/video))
    if not cap.isOpened():raise RuntimeError("CANNOT_OPEN_VIDEO")
    images=[]
    try:
        for ix in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(ix))
            valid,frame=cap.read()
            if not valid:raise RuntimeError("VIDEO_FRAME_FAILED:"+str(ix))
            image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            image.thumbnail((640,480))
            images.append(image)
    finally:cap.release()
    return images

def evaluate(config,modelname):
    import torch
    source,plan_=plan(config)
    cfg=config["models"][modelname]
    output=ROOT/config["output_dir"]
    complete=output/(modelname+".json")
    progress=output/(modelname+".partial.json")
    if complete.exists():
        doc=json.loads(complete.read_text(encoding="utf8"))
        if doc["frame_plan_sha"]!=plan_["frame_plan_sha"]:
            raise RuntimeError("COMPLETE_MODEL_PLAN_SHA_DRIFT")
        if len(doc["predictions"])!=800:raise RuntimeError("INCOMPLETE_BUT_FINAL_FILE")
        print("MOTION_EVAL_ALREADY_COMPLETE",modelname,flush=True)
        return
    processed={}
    if progress.exists():
        previous=json.loads(progress.read_text(encoding="utf8"))
        if previous["frame_plan_sha"]!=plan_["frame_plan_sha"]:raise RuntimeError("PARTIAL_MODEL_PLAN_CHANGED")
        processed={(r["key"],r["frames"]):r for r in previous["predictions"]}
    bykey={r["key"]:r for r in source["rows"]}
    model,processor,dtype=load_model(cfg["model_dir"],cfg)
    torch.cuda.reset_peak_memory_stats()
    for n in config["frame_counts"]:
        for record in plan_["records"]:
            key=(record["key"],n)
            if key in processed:continue
            original=bykey[record["key"]]
            if original["video_sha256"]!=record["video_sha256"]:
                raise RuntimeError("FROZEN_VIDEO_SHA_CHANGED")
            started=time.perf_counter()
            pictures=frames_at_positions(record["video"],record["frame_indices"][str(n)])
            decoded=round(time.perf_counter()-started,4)
            answer,t=infer(model,processor,dtype,build_message(original,pictures),
                           config["max_new_tokens"])
            choice=parsed_answer(answer,len(original["choices"]))
            processed[key]={"key":record["key"],"task":record["task"],
                            "frames":n,"gold":original["gold"],
                            "raw":answer,"pred":choice,"correct":choice==original["gold"],
                            "decode_seconds":decoded,"generation_seconds":t,
                            "positions":record["frame_indices"][str(n)]}
            if len(processed)%20==0:
                save_json(progress,{"frame_plan_sha":plan_["frame_plan_sha"],
                    "model_name":modelname,"predictions":list(processed.values())})
                print("MOTION_SAMPLING_GPU_PROGRESS",modelname,len(processed),"/800",flush=True)
    if len(processed)!=800:raise RuntimeError("MOTION_EVAL_INCOMPLETE")
    save_json(complete,{"frame_plan_sha":plan_["frame_plan_sha"],
       "model_name":modelname,"model_revision":cfg["revision"],
       "peak_gpu_allocated_gib":round(torch.cuda.max_memory_allocated()/2**30,3),
       "predictions":list(processed.values()),"no_task_specific_training":True})
    print("MOTION_SAMPLING_GPU_EVAL_DONE",modelname,len(processed),flush=True)

def paired_stats(reference,saliency,n,task,seed):
    lhs={r["key"]:r["correct"] for r in reference if r["frames"]==n and r["task"]==task}
    rhs={r["key"]:r["correct"] for r in saliency if r["frames"]==n and r["task"]==task}
    if len(lhs)!=200 or len(rhs)!=200:
        raise RuntimeError("PAIRED_TASK_ITEM_COUNT_DRIFT")
    return paired_difference(lhs,rhs,seed)

def report(config):
    _,source=plan(config)
    uniform_root=ROOT/config["uniform_output_dir"]
    directory=ROOT/config["output_dir"]
    result={"method":"12-probe grayscale motion and hue-change label-blind frame selection",
       "frame_plan_sha":source["frame_plan_sha"],"task_results":[],
       "warning":"posthoc experimental preprocessing evaluated on publicly previously inspected tasks, not new blind validation"}
    lines=["# Frozen motion-aware frame sampling ablation","",
        "Compared to fixed uniform 3/5-frame selection at exactly equal visual-frame budgets.",
        "12 low-resolution candidate probes, 0.7 grayscale change + 0.3 HSV hue change.",
        "5%/95% endpoints and time diversity, NO QA labels, NO task training, NO test-time tuning.",
        "**These tasks were already evaluated in earlier experiments: results are exploratory, not new blind generalization.**",
        "","| Model | Task | Frames | Uniform correct | Motion-selected correct | Difference pp | Paired 95% CI pp |",
        "|---|---|---:|---:|---:|---:|---|"]
    for name in config["models"]:
        original=json.loads((uniform_root/(name+".json")).read_text())
        new=json.loads((directory/(name+".json")).read_text())
        if len(new["predictions"])!=800 or new["frame_plan_sha"]!=source["frame_plan_sha"]:
            raise RuntimeError("INCOMPLETE_SOURCE_MODEL_COMPARISON")
        for task in ("moving_direction","scene_transition"):
            for n in config["frame_counts"]:
                u=[x for x in original["predictions"] if x["task"]==task and x["frames"]==n]
                m=[x for x in new["predictions"] if x["task"]==task and x["frames"]==n]
                stats=paired_stats(u,m,n,task,config["bootstrap_seed"]+n+int(name=="2b")*100)
                before=sum(r["correct"] for r in u);after=sum(r["correct"] for r in m)
                record={"model":name,"task":task,"frames":n,"uniform_correct":before,
                    "motion_correct":after,**stats}
                result["task_results"].append(record)
                lines.append(f"| {name} | {task} | {n} | {before}/200 | {after}/200 | {stats['delta_pp']:+.1f} | {stats['bootstrap_95_ci_pp']} |")
    timing=[r["selection_seconds"] for r in source["records"]]
    result["mean_probe_seconds"]=round(statistics.mean(timing),5)
    result["median_probe_seconds"]=round(statistics.median(timing),5)
    lines+=["",f"Mean per-video 12-probe CPU selection overhead: {result['mean_probe_seconds']:.3f} seconds.",
             f"Median per-video overhead: {result['median_probe_seconds']:.3f} seconds.",
             "This CPU overhead is ADDITIONAL to GPU inference and final selected-frame decoding.",
             "","## Research limitations","",
             "- The 200 CLEVRER moving-direction videos are synthetic, not natural real camera footage.",
             "- The public scene-transition task is not a fully new unseen test after prior exploration.",
             "- The frame-selection rule was frozen before this ablation; **no answer labels are used** to select frames.",
             "- Visually maximizing motion does not guarantee task-relevant evidence, particularly for scene transitions.",
             "- More candidate probes add decoding cost and may not pay off in accuracy."]
    save_json(directory/"comparison.json",result)
    (directory/"MOTION_SAMPLING_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("MOTION_SAMPLING_PAIRED_REPORT_COMPLETE",json.dumps(result["task_results"]),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["prepare","evaluate","report"])
    parser.add_argument("--model",choices=["500m","2b"])
    parser.add_argument("--config",default="configs/motion_sampling_v1.json")
    args=parser.parse_args()
    cfg=load_config(args.config)
    if args.stage=="prepare":plan(cfg)
    elif args.stage=="evaluate":
        if not args.model:raise ValueError("MODEL_REQUIRED")
        evaluate(cfg,args.model)
    else:report(cfg)

if __name__=="__main__":
    main()
