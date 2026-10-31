"""Real-video supervised LoRA on 200 MVBench STAR action-sequence clips.

Train/val/test videos are disjoint by ID. Uses only pre-trimmed STAR action_sequence
MP4s, not the other same-name files in the STAR zip. No test labels seen during
training/validation model selection. All raw media stay on Tang.
"""
from __future__ import annotations
import argparse,hashlib,json,random,re,statistics,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LETTERS="ABCD"
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    tmp.replace(path)

def get_config(path):
    return json.loads((ROOT/path).read_text(encoding="utf8"))

def decoded_frame_count(path):
    """ffprobe counts actual decodable frames; MP4 metadata may overstate count.

    The optimized STAR clips can report CAP_PROP_FRAME_COUNT=595 despite only
    437 frames actually decoding. Never sample near the metadata-reported end.
    """
    result=subprocess.run(
        ["ffprobe","-v","error","-count_frames","-select_streams","v:0",
         "-show_entries","stream=nb_read_frames",
         "-of","default=noprint_wrappers=1:nokey=1",str(path)],
        capture_output=True,text=True,check=True,timeout=40)
    values=[line.strip() for line in result.stdout.splitlines()
            if line.strip().isdigit()]
    if len(values)!=1 or int(values[0])<3:
        raise RuntimeError("INVALID_DECODED_FRAME_COUNT: "+str(path)+" raw="+repr(result.stdout))
    return int(values[0])

def manifest(cfg):
    data=ROOT/cfg["data_dir"]
    ann=json.loads((data/"text/action_sequence.json").read_text(encoding="utf8"))
    prov=json.loads((data/"provenance.json").read_text(encoding="utf8"))
    assert prov["archive_sha256"]==cfg["star_archive_sha256"] and prov["exact_video_count"]==200
    assert len(ann)==200 and len({r["video"] for r in ann})==200
    rng=random.Random(cfg["split_seed"])
    indices=list(range(200));rng.shuffle(indices)
    sizes=cfg["split_sizes"]
    assert sum(sizes.values())==200 and all(i>0 for i in sizes.values())
    splits={};offset=0
    for split,count in sizes.items():
        splits[split]=sorted(indices[offset:offset+count]);offset+=count
    assert not(set(splits["train"])&set(splits["val"])) and not(set(splits["train"])&set(splits["test"]))
    target=ROOT/cfg["output_dir"]/"manifest.json"
    if target.is_file():
        saved=json.loads(target.read_text(encoding="utf8"))
        if (saved.get("format_version")!=2 or
            saved.get("split_seed")!=cfg["split_seed"] or
            saved.get("split_sizes")!=sizes or
            saved.get("video_archive_sha256")!=cfg["star_archive_sha256"] or
            saved.get("source_video_manifest_sha256")!=prov["video_manifest_sha256"]):
            raise RuntimeError("FROZEN_STAR_MANIFEST_CONFIG_OR_SOURCE_DRIFT")
        expected=hashlib.sha256(json.dumps(saved["rows"],sort_keys=True).encode()).hexdigest()
        if expected!=saved["sha256"] or len(saved["rows"])!=200:
            raise RuntimeError("FROZEN_STAR_MANIFEST_CONTENT_DRIFT")
        for entry in saved["rows"]:
            if not (ROOT/entry["video"]).is_file() or int(entry["decoded_frames"])<3:
                raise RuntimeError("FROZEN_STAR_VIDEO_MISSING_OR_CORRUPT: "+str(entry["video"]))
        return expected,saved["rows"]
    rows=[]
    for split,ids in splits.items():
        for i in ids:
            x=ann[i];path=data/"clips"/x["video"]
            if not path.exists():raise FileNotFoundError(str(path))
            if len(x["candidates"])!=4 or x["answer"] not in x["candidates"]:raise RuntimeError("Invalid official answer")
            count=decoded_frame_count(path)
            rows.append({"source_index":i,"split":split,"video":str(path.relative_to(ROOT)),
                         "decoded_frames":count,
                         "question":x["question"],"choices":x["candidates"],
                         "answer":LETTERS[x["candidates"].index(x["answer"])],
                         "original_start":x.get("start"),"original_end":x.get("end")})
    digest=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    save(target,{"format_version":2,"sha256":digest,"source":"MVBench action_sequence from STAR",
            "revision":cfg["source_revision"],"video_archive_sha256":cfg["star_archive_sha256"],
            "source_video_manifest_sha256":prov["video_manifest_sha256"],
            "split_seed":cfg["split_seed"],"split_sizes":sizes,"rows":rows,
            "note":"Decode counts measured with ffprobe -count_frames; STAR clips are already trimmed, so original_start/end only metadata."})
    return digest,rows

def extract_frames(path,n,decoded_frames):
    import cv2
    from PIL import Image
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise RuntimeError("Failed to open "+str(path))
    length=int(decoded_frames)
    if length<3:raise RuntimeError("Not enough truly decoded frames: "+str(path))
    ts=[.5] if n==1 else [.05+.9*i/(n-1) for i in range(n)]
    imgs=[]
    try:
        for fraction in ts:
            ix=min(length-1,max(0,round(fraction*(length-1))))
            cap.set(cv2.CAP_PROP_POS_FRAMES,ix)
            ok,bgr=cap.read()
            if not ok:
                raise RuntimeError("DECODE_FRAME_FAIL path="+str(path)+
                                   " actual_frames="+str(length)+" index="+str(ix))
            im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            im.thumbnail((640,480))
            imgs.append(im)
    finally:
        cap.release()
    return imgs

def prompt(row,images):
    items=[{"type":"text","text":"Observe these chronological frames of an already trimmed video. Answer the action sequence multiple choice question."}]
    for i,image in enumerate(images):
        items.extend([{"type":"text","text":f"Video frame {i+1}/{len(images)}:"},
                      {"type":"image","image":image}])
    choices="\n".join(f"{LETTERS[i]}. {s}" for i,s in enumerate(row["choices"]))
    items.append({"type":"text","text":row["question"]+"\n"+choices+
                  "\nReply with a single selected option letter on the final line: ANSWER: followed by that letter."})
    return [{"role":"user","content":items}]

def parse_answer(s):
    x=str(s or "").strip()
    found=re.findall(r"(?i)\bANSWER\s*:\s*([A-D])\b(?!/)",x)
    if found:return found[-1].upper()
    m=re.match(r"^\s*([A-D])\s*[.)\]:-]\s*(?:\S|$)",x,re.I)
    if m:return m.group(1).upper()
    m=re.match(r"^\s*(?:option|choice)\s*([A-D])\b",x,re.I)
    if m:return m.group(1).upper()
    return x.upper() if x.upper() in LETTERS else None

def load_model(cfg,adapter=None):
    import torch
    from transformers import AutoProcessor,AutoModelForImageTextToText
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    local=ROOT/cfg["model_dir"]
    processor=AutoProcessor.from_pretrained(local,local_files_only=True)
    processor.image_processor.do_image_splitting=False
    model=AutoModelForImageTextToText.from_pretrained(local,dtype=dtype,local_files_only=True)
    if adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
    return model.to("cuda").eval(),processor,dtype

def run_eval(cfg,rows,split,tag,adapter=None,counts=(3,)):
    import torch
    model,proc,dtype=load_model(cfg,adapter)
    data=[r for r in rows if r["split"]==split]
    results=[]
    for frames in counts:
        for i,row in enumerate(data):
            imgs=extract_frames(ROOT/row["video"],frames,row["decoded_frames"])
            batch=proc.apply_chat_template(prompt(row,imgs),add_generation_prompt=True,
               tokenize=True,return_tensors="pt",return_dict=True).to("cuda",dtype=dtype)
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode():
                outputs=model.generate(**batch,max_new_tokens=cfg["max_new_tokens"],do_sample=False)
            torch.cuda.synchronize()
            suffix=outputs[0][batch["input_ids"].shape[-1]:]
            raw=proc.decode(suffix,skip_special_tokens=True).strip()
            pred=parse_answer(raw)
            record={"id":row["source_index"],"frames":frames,"gold":row["answer"],
                    "pred":pred,"correct":pred==row["answer"],"raw":raw,
                    "seconds":round(time.perf_counter()-start,4)}
            results.append(record)
            if (i+1)%10==0:print("EVAL",tag,frames,i+1,len(data),
                                  sum(r["correct"] for r in results if r["frames"]==frames),flush=True)
    metrics={}
    for n in counts:
        items=[r for r in results if r["frames"]==n]
        metrics[str(n)]={"correct":sum(r["correct"] for r in items),"total":len(items),
                          "invalid":sum(r["pred"] is None for r in items)}
    result={"model":tag,"split":split,"source":"STAR MVBench action_sequence",
            "trained_on":["train"] if adapter else [],"metrics":metrics,
            "results":results,"peak_gpu_gib":round(torch.cuda.max_memory_allocated()/2**30,3)}
    out=ROOT/cfg["output_dir"]
    save(out/(tag+".json"),result)
    del model,proc;torch.cuda.empty_cache()
    return result

def train(cfg,rows,seed):
    import torch
    from peft import LoraConfig,get_peft_model
    torch.manual_seed(seed)
    base,proc,dtype=load_model(cfg)
    model=get_peft_model(base,LoraConfig(r=cfg["rank"],lora_alpha=cfg["alpha"],
                           target_modules=["q_proj","v_proj"],lora_dropout=0.0,task_type="CAUSAL_LM"))
    model.train()
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=cfg["learning_rate"])
    examples=[r for r in rows if r["split"]=="train"]
    rng=random.Random(seed);rng.shuffle(examples)
    trace=[]
    for step in range(cfg["train_steps"]):
        row=examples[step%len(examples)]
        imgs=extract_frames(ROOT/row["video"],cfg["training_frames"],row["decoded_frames"])
        user=prompt(row,imgs)
        full=user+[{"role":"assistant","content":[{"type":"text","text":"ANSWER: "+row["answer"]}]}]
        start_batch=proc.apply_chat_template(user,add_generation_prompt=True,
                              tokenize=True,return_tensors="pt",return_dict=True)
        target=proc.apply_chat_template(full,add_generation_prompt=False,
                              tokenize=True,return_tensors="pt",return_dict=True)
        labels=target["input_ids"].clone()
        labels[:,:min(start_batch["input_ids"].shape[-1],labels.shape[-1]-1)]=-100
        features=target.to("cuda",dtype=dtype)
        features["labels"]=labels.to("cuda")
        if int((labels!=-100).sum())==0:raise RuntimeError("EMPTY_TRAIN_LABELS")
        opt.zero_grad(set_to_none=True)
        loss=model(**features).loss
        if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss")
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.0)
        opt.step()
        if step%10==0 or step==cfg["train_steps"]-1:
            info={"step":step+1,"loss":round(float(loss.detach().cpu()),5),
                  "grad_norm":round(float(grad.detach().cpu()),4)}
            trace.append(info);print("TRAIN",seed,info,flush=True)
    out=ROOT/cfg["output_dir"]
    dest=out/f"adapter_{seed}";dest.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(str(dest))
    save(out/f"train_{seed}.json",{"seed":seed,"steps":cfg["train_steps"],"train_count":len(examples),
         "trace":trace,"peak_gpu_gib":round(torch.cuda.max_memory_allocated()/2**30,3),
         "warning":"Real video fine-tuning on 120 training videos; heldout split never in gradients."})
    print("TRAIN_COMPLETE",seed,dest,flush=True)

def paired_ci(a,b,n):
    aa={r["id"]:int(r["correct"]) for r in a["results"] if r["frames"]==n}
    bb={r["id"]:int(r["correct"]) for r in b["results"] if r["frames"]==n}
    assert set(aa)==set(bb)
    diff=[bb[i]-aa[i] for i in sorted(aa)]
    rng=random.Random(91001+n)
    distribution=sorted(100*sum(rng.choices(diff,k=len(diff)))/len(diff) for _ in range(4000))
    return {"delta_pp":round(100*sum(diff)/len(diff),2),
            "bootstrap_95_ci_pp":[round(distribution[100],2),round(distribution[3900],2)]}

def validate(cfg,rows):
    out=ROOT/cfg["output_dir"]
    scores=[]
    for seed in cfg["training_seeds"]:
        result=run_eval(cfg,rows,"val",f"val_{seed}",out/f"adapter_{seed}",counts=(3,))
        scores.append((result["metrics"]["3"]["correct"],seed))
    # Precommitted tie break: smaller seed on ties; never inspect test labels to choose seed.
    chosen=sorted(scores,key=lambda z:(-z[0],z[1]))[0][1]
    save(out/"chosen_seed.json",{"chosen_seed":chosen,"rule":"highest val 3-frame hits; ties smaller seed",
                                 "val_results":scores,"test_unseen":True})
    print("SELECTED_SEED_BY_VAL_ONLY",chosen,flush=True)

def test(cfg,rows):
    out=ROOT/cfg["output_dir"]
    selector=json.loads((out/"chosen_seed.json").read_text())
    seed=selector["chosen_seed"]
    baseline=run_eval(cfg,rows,"test","test_base",None,counts=(1,3,5))
    tuned=run_eval(cfg,rows,"test","test_lora",out/f"adapter_{seed}",counts=(1,3,5))
    stats={}
    lines=["# Real-video LoRA train/val/test result","",
           "Task: MVBench STAR action_sequence. This project uses its 200 official samples, with video-ID-disjoint 120/40/40 partitions.",
           "Training on real video labels is now genuinely performed; this is task-specific supervised fine-tuning.",
           "Validation, not test, selects the best among two training seeds.",
           "","| Frames | Base | Selected LoRA | Delta pp | Paired bootstrap 95% CI pp |","|---:|---:|---:|---:|---|"]
    for n in (1,3,5):
        diff=paired_ci(baseline,tuned,n)
        stats[str(n)]={"base":baseline["metrics"][str(n)],
                       "lora":tuned["metrics"][str(n)],**diff}
        lines.append(f"| {n} | {baseline['metrics'][str(n)]['correct']}/40 | {tuned['metrics'][str(n)]['correct']}/40 | {diff['delta_pp']:+.2f} | {diff['bootstrap_95_ci_pp']} |")
    lines+=["","Selected seed: "+str(seed),
            "","Reminder: 40 test samples are small, source data are public, and all clips are from one action-sequence task.",
            "No claims about general-purpose video reasoning without independent external benchmark."]
    (out/"REAL_VIDEO_SFT_REPORT.md").write_text("\n".join(lines)+"\n")
    save(out/"test_comparison.json",{"selected_seed":seed,"stats":stats})
    print("REAL_VIDEO_SFT_TEST_COMPLETE",stats,flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["prepare","train","validate","test"])
    p.add_argument("--config",default="configs/star_real_sft_round1.json")
    a=p.parse_args();cfg=get_config(a.config)
    digest,rows=manifest(cfg)
    if a.stage=="prepare":
        print("STAR_MANIFEST_FROZEN",digest,"sample_count",len(rows),flush=True)
        for i,row in enumerate(rows):
            extract_frames(ROOT/row["video"],cfg["training_frames"],row["decoded_frames"])
            if (i+1)%40==0:print("STAR_DECODE_VERIFIED",i+1,"/",len(rows),flush=True)
        print("STAR_SPLIT_AND_ALL_FRAMES_VERIFIED",len(rows),flush=True)
    elif a.stage=="train":
        for seed in cfg["training_seeds"]:train(cfg,rows,seed)
    elif a.stage=="validate":validate(cfg,rows)
    else:test(cfg,rows)
if __name__=="__main__":main()
