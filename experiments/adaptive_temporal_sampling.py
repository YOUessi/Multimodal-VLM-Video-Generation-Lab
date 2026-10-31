"""Blind adaptive temporal frame sampling vs frozen uniform 5/50/95% and 5-frame baselines.

No labels enter frame selection. Scores use video pixels (motion and appearance
change), with a deterministic endpoint guard and temporal diversity. All 40
heldout STAR videos are evaluated; no data-dependent tuning on the test set.
"""
from __future__ import annotations
import argparse,hashlib,json,random,statistics,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
from star_real_sft import load_model,prompt,parse_answer

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    temp.replace(path)

def indices_for_signals(frame_indices,signals,count,min_distance):
    """Deterministic greedy top-event frames with coverage and fixed endpoints."""
    if count<3:raise ValueError("Expected 3+ frames")
    if len(frame_indices)!=len(signals):raise ValueError("Length mismatch")
    first,last=frame_indices[0],frame_indices[-1]
    if last-first<count:raise ValueError("Not enough candidate positions")
    chosen=[first,last]
    pool=sorted(zip(frame_indices[1:-1],signals[1:-1]),key=lambda z:(-z[1],z[0]))
    while len(chosen)<count:
        eligible=[z for z in pool if z[0] not in chosen and
                  all(abs(z[0]-picked)>=min_distance for picked in chosen)]
        if not eligible:
            eligible=[z for z in pool if z[0] not in chosen]
        if not eligible:raise RuntimeError("Not enough unique positions")
        picked=eligible[0][0]
        chosen.append(picked)
    return sorted(chosen)

def select_video(video,decoded,count,probe_count=48):
    import cv2
    if decoded<6:raise ValueError("Video must have >=6 decoded frames")
    endpoints=[round(.05*(decoded-1)),round(.95*(decoded-1))]
    samples=sorted(set([round((.05+.90*i/(probe_count-1))*(decoded-1))
                        for i in range(probe_count)]+endpoints))
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise RuntimeError("Cannot open "+str(video))
    frames=[]
    try:
        for idx in samples:
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
            ok,img=cap.read()
            if not ok:raise RuntimeError(f"Cannot decode frame {idx} of {video}")
            gray=cv2.cvtColor(cv2.resize(img,(96,72)),cv2.COLOR_BGR2GRAY)
            hsv=cv2.cvtColor(cv2.resize(img,(96,72)),cv2.COLOR_BGR2HSV)
            hist=cv2.calcHist([hsv],[0],None,[24],[0,180])
            cv2.normalize(hist,hist)
            frames.append((gray,hist))
    finally:
        cap.release()
    raw_motion=[0.]
    raw_color=[0.]
    for i in range(1,len(frames)):
        raw_motion.append(float(np.mean(np.abs(frames[i][0].astype(np.float32)-frames[i-1][0].astype(np.float32)))))
        raw_color.append(float(cv2.compareHist(frames[i-1][1],frames[i][1],cv2.HISTCMP_BHATTACHARYYA)))
    def robust_scale(v):
        hi=max(v)
        return [float(t/hi) if hi>1e-9 else 0.0 for t in v]
    motion=robust_scale(raw_motion);change=robust_scale(raw_color)
    signals=[.65*m+.35*c for m,c in zip(motion,change)]
    gap=max(1,round(decoded*(.07 if count==5 else .10)))
    picked=indices_for_signals(samples,signals,count,gap)
    if picked[0]!=endpoints[0] or picked[-1]!=endpoints[1]:
        raise RuntimeError("ENDPOINT_GUARD_FAILED")
    return picked,{"motion_max":max(raw_motion),"color_max":max(raw_color),"min_separation":gap,
                    "probe_count":len(samples)}

def load_pil(video,frame_ids):
    import cv2
    from PIL import Image
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise RuntimeError("Cannot open for eval")
    out=[]
    try:
        for ix in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(ix))
            ok,arr=cap.read()
            if not ok:raise RuntimeError("Cannot decode chosen frame "+str(ix))
            img=Image.fromarray(cv2.cvtColor(arr,cv2.COLOR_BGR2RGB));img.thumbnail((640,480))
            out.append(img)
    finally:
        cap.release()
    return out

def config(path):
    return json.loads((ROOT/path).read_text(encoding="utf8"))

def prepare(cfg):
    src=json.loads((ROOT/cfg["star_manifest"]).read_text())
    selected=[r for r in src["rows"] if r["split"]=="test"]
    assert len(selected)==40 and len({r["video"] for r in selected})==40
    record=[]
    for i,row in enumerate(selected):
        methods={}
        for n in cfg["frame_counts"]:
            indexes,stats=select_video(ROOT/row["video"],row["decoded_frames"],n,cfg["probe_count"])
            methods[str(n)]={"frame_indices":indexes,"frame_stats":stats}
        record.append({"id":row["source_index"],"video":row["video"],
                       "gold":row["answer"],"frames":methods})
        if (i+1)%10==0:print("SAMPLER_PREFLIGHT",i+1,"/",len(selected),flush=True)
    digest=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
    target=ROOT/cfg["output_dir"]/"sampling_plan.json"
    doc={"sampling_version":"adaptive_v1_preregistered","star_manifest_sha256":src["sha256"],
         "plan_sha256":digest,"rules":{"motion_weight":.65,"hue_change_weight":.35,
         "endpoints":"5% and 95% of ACTUAL decoded frames","min_spacing":"7% for n5, 10% for n3",
         "candidate_count":cfg["probe_count"]},"selection_is_label_blind":True,"records":record}
    if target.exists():
        old=json.loads(target.read_text())
        if old["plan_sha256"]!=digest:raise RuntimeError("FRAME_SAMPLING_SELECTION_DRIFT")
    else:write(target,doc)
    return doc

def evaluate(cfg,variant):
    import torch
    plan=json.loads((ROOT/cfg["output_dir"]/"sampling_plan.json").read_text())
    manifest=json.loads((ROOT/cfg["star_manifest"]).read_text())
    assert plan["star_manifest_sha256"]==manifest["sha256"]
    rows={r["source_index"]:r for r in manifest["rows"] if r["split"]=="test"}
    if variant=="selected_lora":
        chosen=json.loads((ROOT/cfg["chosen_seed"]).read_text())
        adapter=ROOT/cfg["star_output"]/(f"adapter_{chosen['chosen_seed']}")
    else:adapter=None
    model,proc,dtype=load_model({"model_dir":cfg["model_dir"]},adapter)
    dest=ROOT/cfg["output_dir"]/(variant+".json")
    result=[]
    for n in cfg["frame_counts"]:
        for i,r in enumerate(plan["records"]):
            original=rows[r["id"]]
            indexes=r["frames"][str(n)]["frame_indices"]
            frames=load_pil(ROOT[r["video"]],indexes)
            batch=proc.apply_chat_template(prompt(original,frames),
                        add_generation_prompt=True,tokenize=True,
                        return_tensors="pt",return_dict=True).to("cuda",dtype=dtype)
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode():
                seq=model.generate(**batch,max_new_tokens=cfg["max_new_tokens"],do_sample=False)
            torch.cuda.synchronize()
            raw=proc.decode(seq[0][batch["input_ids"].shape[-1]:],skip_special_tokens=True).strip()
            pred=parse_answer(raw)
            result.append({"id":r["id"],"frames":n,"gold":r["gold"],"pred":pred,
                           "correct":pred==r["gold"],"raw":raw,"chosen_frames":indexes,
                           "inference_seconds":round(time.perf_counter()-start,4)})
            if (i+1)%10==0:
                print("ADAPTIVE_EVAL",variant,n,i+1,
                      "correct",sum(row["correct"] for row in result if row["frames"]==n),flush=True)
    metrics={}
    for n in cfg["frame_counts"]:
        subset=[r for r in result if r["frames"]==n]
        metrics[str(n)]={"correct":sum(r["correct"] for r in subset),"total":len(subset),
                          "unparseable":sum(r["pred"] is None for r in subset)}
    write(dest,{"variant":variant,"source":"STAR action_sequence independent test split",
                "sampling_plan_sha256":plan["plan_sha256"],"metrics":metrics,
                "predictions":result,"peak_vram_gib":round(torch.cuda.max_memory_allocated()/2**30,3)})
    print("ADAPTIVE_EVAL_COMPLETE",variant,metrics,flush=True)

def paired_improvement(uniform,adaptive,n):
    a={r["id"]:int(r["correct"]) for r in uniform["results"] if r["frames"]==n}
    b={r["id"]:int(r["correct"]) for r in adaptive["predictions"] if r["frames"]==n}
    assert set(a)==set(b) and len(a)==40
    diff=[b[i]-a[i] for i in sorted(a)]
    rng=random.Random(88441+n)
    boot=sorted(sum(rng.choices(diff,k=len(diff)))/len(diff)*100 for _ in range(4000))
    return {"delta_pp":round(sum(diff)/len(diff)*100,2),
            "bootstrap_95ci_pp":[round(boot[100],2),round(boot[3900],2)]}

def compare(cfg):
    base=ROOT/cfg["star_output"];new=ROOT/cfg["output_dir"]
    table=["# Change-aware frame selection: frozen natural video evaluation",
           "","Unseen STAR 40-video test; identical model, question, and 3/5 frame budgets.",
           "Motion+hue-change are label blind; endpoints fixed. No tuning was performed on the test.",
           "","| Model | Frame budget | Uniform | Adaptive | Delta pp | Paired 95% CI pp |",
           "|---|---:|---:|---:|---:|---|"]
    summary=[]
    for variant,orig in [("base","test_base"),("selected_lora","test_lora")]:
        old=json.loads((base/(orig+".json")).read_text())
        newd=json.loads((new/(variant+".json")).read_text())
        for n in cfg["frame_counts"]:
            ref=old["metrics"][str(n)];target=newd["metrics"][str(n)]
            stats=paired_improvement(old,newd,n)
            summary.append({"model":variant,"frames":n,"uniform":ref,"adaptive":target,**stats})
            table.append(f"| {variant} | {n} | {ref['correct']}/{ref['total']} | {target['correct']}/{target['total']} | {stats['delta_pp']:+.2f} | {stats['bootstrap_95ci_pp']} |")
    table+=["","Sampling algorithm must not use question labels or choose model based on heldout performance.",
            "The input budget is matched by number of frames, but CPU frame-selection computation adds extra latency.",
            "This limited action_sequence split and 40-sample test is not proof of broad VLM gain."]
    (new/"ADAPTIVE_SAMPLING_REPORT.md").write_text("\n".join(table)+"\n")
    write(new/"comparison.json",{"source":"STAR video-id disjoint test","predeclared_rules":True,"comparisons":summary})
    print("ADAPTIVE_COMPARISON_COMPLETE",json.dumps(summary),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["prepare","base","selected_lora","compare"])
    p.add_argument("--config",default="configs/adaptive_sampling_v1.json")
    args=p.parse_args();cfg=config(args.config)
    if args.stage=="prepare":prepare(cfg)
    elif args.stage=="compare":compare(cfg)
    else:evaluate(cfg,args.stage)
if __name__=="__main__":main()
