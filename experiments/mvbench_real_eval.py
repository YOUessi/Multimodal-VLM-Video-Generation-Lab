"""Frozen public MVBench real-video evaluation, base and synthetic-LoRA transfer.
The 200 source QA labels are publicly supplied, not assistant-generated. No samples
from this public set are used to train adapters. Raw videos remain on Tang.
"""
import argparse, hashlib, json, random, re, statistics, time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LETTERS="ABC"
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    tmp.replace(path)
def manifest(cfg):
    data=ROOT/cfg["data_dir"]
    rows=json.loads((data/"text/action_antonym.json").read_text())
    prov=json.loads((data/"dataset_provenance.json").read_text())
    assert len(rows)==200 and prov["matched"]==200
    assert prov["video_archive_sha256"]==cfg["archive_sha256"],"DATASET_SHA_DRIFT"
    assert 0<cfg["sample_count"]<=200
    rng=random.Random(cfg["seed"]);chosen=sorted(rng.sample(range(200),cfg["sample_count"]))
    selected=[]
    for i in chosen:
        entry=rows[i];files=list((data/"clips").rglob(entry["video"]))
        assert len(files)==1,(entry["video"],len(files))
        assert len(entry["candidates"])==3 and entry["answer"] in entry["candidates"]
        selected.append({"id":i,"video":str(files[0].relative_to(ROOT)),
            "question":entry["question"],"choices":entry["candidates"],
            "gold":LETTERS[entry["candidates"].index(entry["answer"])]})
    digest=hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest()
    out=ROOT/cfg["output_dir"];target=out/"manifest.json"
    if target.exists():
        old=json.loads(target.read_text());assert digest==old["sha256"],"MANIFEST_DRIFT"
    else:save(target,{"source":"PKU-Alignment/MVBench action_antonym",
        "source_revision":cfg["revision"],"seed":cfg["seed"],"sha256":digest,
        "sample_count":len(selected),"rows":selected,
        "policy":"official public labels; evaluation-only; not manual video review"})
    return digest,selected
def frames(path,n):
    import cv2
    from PIL import Image
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise RuntimeError("VIDEO_READ_FAILED: "+str(path))
    length=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); assert length>2
    positions=[.5] if n==1 else [.05+.90*i/(n-1) for i in range(n)]
    imgs=[]
    for t in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES,min(length-1,round(t*(length-1))))
        ok,frame=cap.read()
        if not ok:raise RuntimeError("VIDEO_FRAME_READ_FAILED")
        img=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        img.thumbnail((640,480))
        imgs.append(img)
    cap.release();return imgs
def prompt(row,imgs):
    content=[{"type":"text","text":"These are chronological video frames. Answer from the visible action only."}]
    for i,img in enumerate(imgs):
        content.extend([{"type":"text","text":f"Video frame {i+1} of {len(imgs)}:"},
                        {"type":"image","image":img}])
    choices="\n".join(f"{LETTERS[k]}. {x}" for k,x in enumerate(row["choices"]))
    content.append({"type":"text","text":row["question"]+"\n"+choices+
                    "\nChoose exactly one option. Reply on a single line: ANSWER: A or ANSWER: B or ANSWER: C."})
    return [{"role":"user","content":content}]
def parse_choice(s):
    found=re.findall(r"(?i)\bANSWER\s*:\s*([A-C])\b",s)
    if found:return found[-1].upper()
    v=s.strip().upper().strip(" .:()[]")
    return v if v in LETTERS else None
def evaluate(cfg,rows,digest,variant):
    import torch
    from transformers import AutoProcessor,AutoModelForImageTextToText
    out=ROOT/cfg["output_dir"];dest=out/(variant+".json")
    if dest.exists():
        existing=json.loads(dest.read_text())
        if existing["manifest_sha256"]==digest and len(existing["predictions"])==len(rows)*len(cfg["frame_counts"]):
            print("CACHE_VERIFIED",dest,flush=True);return
    model_path=ROOT/cfg["model_path"]
    proc=AutoProcessor.from_pretrained(model_path,local_files_only=True)
    proc.image_processor.do_image_splitting=False
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model=AutoModelForImageTextToText.from_pretrained(model_path,local_files_only=True,dtype=dtype)
    if variant=="synthetic_lora":
        from peft import PeftModel
        adapter=ROOT/cfg["adapter"]
        assert (adapter/"adapter_model.safetensors").is_file()
        model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
    model=model.to("cuda").eval()
    results=[]
    for n in cfg["frame_counts"]:
        for i,row in enumerate(rows):
            media=frames(ROOT/row["video"],n)
            message=prompt(row,media)
            try:
                x=proc.apply_chat_template(message,add_generation_prompt=True,
                    tokenize=True,return_dict=True,return_tensors="pt").to("cuda",dtype=dtype)
                torch.cuda.synchronize();start=time.perf_counter()
                with torch.inference_mode():
                    tokens=model.generate(**x,max_new_tokens=cfg["max_new_tokens"],do_sample=False)
                torch.cuda.synchronize()
                raw=proc.decode(tokens[0][x["input_ids"].shape[-1]:],skip_special_tokens=True).strip()
                pred=parse_choice(raw);seconds=round(time.perf_counter()-start,4);error=None
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as e:
                raw="";pred=None;seconds=None;error=f"{type(e).__name__}: {str(e)[:200]}"
            results.append({"id":row["id"],"frames":n,"gold":row["gold"],
                            "pred":pred,"raw":raw,"correct":pred==row["gold"],
                            "seconds":seconds,"error":error})
            if (i+1)%10==0:
                print("EVAL",variant,n,i+1,len(rows),
                    "CORRECT",sum(z["correct"] for z in results if z["frames"]==n),flush=True)
    metrics={}
    for n in cfg["frame_counts"]:
        subset=[r for r in results if r["frames"]==n]
        lat=[r["seconds"] for r in subset if r["seconds"] is not None]
        metrics[str(n)]={"correct":sum(r["correct"] for r in subset),"total":len(subset),
                         "invalid_output":sum(r["pred"] is None for r in subset),
                         "inference_errors":sum(r["error"] is not None for r in subset),
                         "median_latency_s":round(statistics.median(lat),4) if lat else None}
    save(dest,{"variant":variant,"source":"MVBench action_antonym","manifest_sha256":digest,
       "timestamp_utc":datetime.now(timezone.utc).isoformat(),
       "training":"synthetic color video ONLY" if variant=="synthetic_lora" else "base model",
       "label_policy":"unparseable/error counted wrong","metrics":metrics,
       "peak_gpu_gib":round(torch.cuda.max_memory_allocated()/2**30,3),
       "predictions":results})
    print("EVAL_COMPLETE",variant,metrics,flush=True)
def paired_ci(base,tuned,n,seed=42):
    a={x["id"]:int(x["correct"]) for x in base["predictions"] if x["frames"]==n}
    b={x["id"]:int(x["correct"]) for x in tuned["predictions"] if x["frames"]==n}
    assert set(a)==set(b)
    diffs=[b[k]-a[k] for k in sorted(a)]
    rng=random.Random(seed+n);sampled=sorted(
        sum(rng.choices(diffs,k=len(diffs)))/len(diffs) for _ in range(4000))
    return [round(sampled[100]*100,2),round(sampled[3900]*100,2)]
def compare(cfg,digest):
    out=ROOT/cfg["output_dir"]
    a=json.loads((out/"base.json").read_text())
    b=json.loads((out/"synthetic_lora.json").read_text())
    assert a["manifest_sha256"]==b["manifest_sha256"]==digest
    lines=["# Real video MVBench evaluation-only benchmark","",
        "Public officially annotated action_antonym videos, not used for LoRA training.",
        "","| Frames | Base | Synthetic-trained LoRA | Delta (pp) | Paired bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|"]
    stats={}
    for n in cfg["frame_counts"]:
        x=a["metrics"][str(n)];y=b["metrics"][str(n)]
        delta=round(100*(y["correct"]-x["correct"])/x["total"],2)
        ci=paired_ci(a,b,n)
        stats[str(n)]={"base":x,"synthetic_lora":y,"delta_pp":delta,"paired_bootstrap_95_ci_pp":ci}
        lines.append(f"| {n} | {x['correct']}/{x['total']} | {y['correct']}/{y['total']} | {delta:+.2f} | {ci} |")
    lines += ["","Manifest SHA256: "+digest,
             "","These are public benchmark annotations. No model was trained on the public video questions.",
             "The source calls this split train; here a disjoint 80-video sample is evaluation-only.",
             "No automatic model selection on this test sample and no production quality claims."]
    (out/"REAL_VIDEO_COMPARISON.md").write_text("\n".join(lines)+"\n")
    save(out/"comparison.json",{"stats":stats,"manifest_sha256":digest})
    print("COMPARISON_COMPLETE",json.dumps(stats),flush=True)
def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("stage",choices=["base","synthetic_lora","compare"])
    cli.add_argument("--config",default="configs/mvbench_round1.json")
    args=cli.parse_args()
    cfg=json.loads((ROOT/args.config).read_text())
    digest,rows=manifest(cfg)
    if args.stage=="compare":compare(cfg,digest)
    else:evaluate(cfg,rows,digest,args.stage)
if __name__=="__main__":main()
