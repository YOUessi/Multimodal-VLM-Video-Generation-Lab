"""Controlled real CUDA experiment: generated multi-scene MP4, frame-count baseline, LoRA and frozen test set."""
from __future__ import annotations
import argparse, itertools, json, random, hashlib, re, statistics, time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PALETTE={"red":(45,44,224),"blue":(225,94,42),"green":(74,160,45),
         "yellow":(44,215,235),"purple":(174,57,146),"orange":(50,143,242)}
COLORS=list(PALETTE)

def save(path,object):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(object,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    tmp.replace(path)

def paths(cfg):
    return ROOT/cfg["dataset_dir"],ROOT/cfg["output_dir"],ROOT/cfg["model_local_path"]

def make_clips(cfg):
    import cv2, numpy as np
    dat,output,_=paths(cfg); f=dat/"manifest.json"
    if f.exists():
        doc=json.loads(f.read_text())
        assert doc["seed"]==cfg["seed"] and doc["split_sizes"]==cfg["split_sizes"], "FROZEN_DATASET_CONFIG_CHANGED"
        return doc
    rng=random.Random(cfg["seed"])
    triples=[v for v in itertools.product(COLORS,repeat=3) if v[0]!=v[-1]]
    rng.shuffle(triples)
    if sum(cfg["split_sizes"].values())>len(triples):raise ValueError("Not enough distinct color sequences")
    width,height=cfg["frame_size"]; fps=cfg["video_fps"]; duration=cfg["seconds_per_color"]
    samples=[]; count=0
    for split,size in cfg["split_sizes"].items():
        for index in range(size):
            triple=triples[count];count+=1;name=f"{split}_{index:04d}"
            video=dat/"clips"/split/(name+".mp4");video.parent.mkdir(parents=True,exist_ok=True)
            w=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
            if not w.isOpened():raise RuntimeError("MP4 writer failed")
            for color in triple:
                for k in range(fps*duration):
                    arr=np.full((height,width,3),(240,240,240),dtype=np.uint8)
                    dx=(index*5+k)%12-6;dy=(index*7+k)%12-6
                    cv2.rectangle(arr,(64+dx,32+dy),(259+dx,199+dy),PALETTE[color],-1)
                    cv2.circle(arr,(23,20),8,(135,140,145),-1)
                    w.write(arr)
            w.release()
            cap=cv2.VideoCapture(str(video))
            frames=cap.get(cv2.CAP_PROP_FRAME_COUNT) if cap.isOpened() else 0
            cap.release()
            if frames<fps*duration*2.9:raise RuntimeError("Failed MP4 readback")
            samples.append({"id":name,"split":split,"video":str(video.relative_to(ROOT)),
                            "first":triple[0],"middle":triple[1],"last":triple[2],"sequence":triple})
    digest=hashlib.sha256(json.dumps([(z["id"],z["sequence"]) for z in samples]).encode()).hexdigest()
    doc={"seed":cfg["seed"],"split_sizes":cfg["split_sizes"],"hash":digest,
         "samples":samples,"warning":"Synthetic color clips, not natural video reasoning"}
    save(f,doc);print("FROZEN_DATASET",len(samples),digest,flush=True)
    return doc

def frames(sample,n,folder):
    import cv2
    from PIL import Image
    cap=cv2.VideoCapture(str(ROOT/sample["video"]))
    if not cap.isOpened():raise RuntimeError("Video open failure: "+sample["video"])
    num=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indexes=[num//2] if n==1 else [round((.04+.92*i/(n-1))*(num-1)) for i in range(n)]
    result=[]
    for i,idx in enumerate(indexes):
        cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
        ok,bgr=cap.read()
        if not ok:raise RuntimeError("Video decode failed: "+sample["video"])
        path=folder/sample["id"]/(f"{n}_{i}.png");path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).save(path)
        result.append(path)
    cap.release()
    return result

def example(sample,n,folder):
    images=frames(sample,n,folder)
    if n==1:
        gold=sample["middle"];question="What is the color of the large rectangle? Reply with one color word."
    else:
        gold=f"first: {sample['first']}; last: {sample['last']}"
        question="Which rectangle color appears first and which last? Reply: first: COLOR; last: COLOR"
    content=[{"type":"text","text":"Answer only from the following chronological video frames."}]
    for i,image in enumerate(images):
        content.extend([{"type":"text","text":f"Frame {i+1} in chronological order:"},
                        {"type":"image","path":str(image)}])
    content.append({"type":"text","text":question})
    return [{"role":"user","content":content}],gold

def normalize(answer,n):
    text=answer.lower()
    colors=r"(red|blue|green|yellow|purple|orange)"
    if n==1:
        found=re.findall(r"\b"+colors+r"\b",text)
        return found[0] if len(found)==1 else "UNPARSEABLE"
    first=re.search(r"\bfirst\s*(?:color\s*)?(?:is|:|-)?\s*"+colors+r"\b",text)
    last=re.search(r"\blast\s*(?:color\s*)?(?:is|:|-)?\s*"+colors+r"\b",text)
    return f"first: {first.group(1)}; last: {last.group(1)}" if first and last else "UNPARSEABLE"

def model_pack(path,adapter=None):
    import torch
    from transformers import AutoModelForImageTextToText,AutoProcessor
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor=AutoProcessor.from_pretrained(str(path),local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(str(path),local_files_only=True,dtype=dtype)
    if adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
    return model.to("cuda").eval(),processor,dtype

def answer(model,proc,dtype,messages,max_new_tokens):
    import torch
    batch=proc.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,
                                   return_tensors="pt",return_dict=True).to("cuda",dtype=dtype)
    torch.cuda.synchronize();start=time.perf_counter()
    with torch.inference_mode():
        output=model.generate(**batch,do_sample=False,max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    end=output[0][batch["input_ids"].shape[-1]:]
    return proc.decode(end,skip_special_tokens=True).strip(),round(time.perf_counter()-start,4)

def evaluate(cfg,doc,adapter=None,label="baseline"):
    import torch
    _,out,modelpath=paths(cfg)
    model,proc,dtype=model_pack(modelpath,adapter)
    heldout=[z for z in doc["samples"] if z["split"]=="test"][:cfg["max_test_samples"]]
    rows=[]
    for n in cfg["eval_frame_counts"]:
        for sample in heldout:
            msg,gold=example(sample,n,out/"frames")
            raw,seconds=answer(model,proc,dtype,msg,cfg["max_new_tokens"])
            pred=normalize(raw,n)
            row={"sample":sample["id"],"frames":n,"raw":raw,"gold":gold,"pred":pred,
                 "correct":pred==gold,"seconds":seconds}
            rows.append(row)
            print("EVAL",label,n,sample["id"],"OK" if row["correct"] else "MISS",repr(pred),flush=True)
    metrics={}
    for n in cfg["eval_frame_counts"]:
        subset=[r for r in rows if r["frames"]==n]
        metrics[str(n)]={"correct":sum(r["correct"] for r in subset),"total":len(subset),
                         "median_seconds":round(statistics.median(r["seconds"] for r in subset),4)}
    result={"time_utc":datetime.now(timezone.utc).isoformat(),"frozen_dataset_sha":doc["hash"],
            "model":str(modelpath),"adapter":str(adapter) if adapter else None,"metrics":metrics,
            "predictions":rows,"peak_gpu_gib":round(torch.cuda.max_memory_allocated()/2**30,3),
            "warning":"Synthetic generated videos; not real-world video understanding"}
    save(out/(label+".json"),result)
    del model,proc;torch.cuda.empty_cache()
    return result

def train(cfg,doc):
    import torch
    from peft import LoraConfig,get_peft_model
    _,out,modelpath=paths(cfg)
    if not (out/"baseline.json").is_file():raise RuntimeError("Missing baseline")
    torch.manual_seed(cfg["seed"])
    model,proc,dtype=model_pack(modelpath)
    model=get_peft_model(model,LoraConfig(r=cfg["lora_rank"],lora_alpha=cfg["lora_alpha"],
                                         lora_dropout=0.0,target_modules=["q_proj","v_proj"],task_type="CAUSAL_LM"))
    model.train()
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=cfg["train_learning_rate"])
    examples=[s for s in doc["samples"] if s["split"]=="train"]
    steps=[];num_steps=cfg["train_steps"]
    for idx in range(num_steps):
        item=examples[idx % len(examples)]
        msg,gold=example(item,3,out/"frames")
        full=msg+[{"role":"assistant","content":[{"type":"text","text":gold}]}]
        prior=proc.apply_chat_template(msg,add_generation_prompt=True,tokenize=True,
                                       return_tensors="pt",return_dict=True)
        inputs=proc.apply_chat_template(full,add_generation_prompt=False,tokenize=True,
                                        return_tensors="pt",return_dict=True)
        labels=inputs["input_ids"].clone()
        labels[:,:min(prior["input_ids"].shape[-1],labels.shape[-1]-1)]=-100
        inputs=inputs.to("cuda",dtype=dtype);inputs["labels"]=labels.to("cuda")
        optimizer.zero_grad(set_to_none=True);t=time.perf_counter()
        loss=model(**inputs).loss
        if not torch.isfinite(loss):raise RuntimeError(f"Nonfinite LoRA loss step {idx+1}")
        loss.backward()
        grad=float(torch.nn.utils.clip_grad_norm_(params,1.0))
        optimizer.step()
        if idx%4==0 or idx==num_steps-1:
            step={"step":idx+1,"loss":round(loss.item(),5),"grad_norm":round(grad,4),
                  "elapsed_sec":round(time.perf_counter()-t,3)}
            steps.append(step);print("TRAIN",json.dumps(step),flush=True)
    adapter=out/"adapter";adapter.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(str(adapter))
    result={"frozen_dataset_sha":doc["hash"],"steps":num_steps,"rank":cfg["lora_rank"],
            "training_curve":steps,"peak_gpu_gib":round(torch.cuda.max_memory_allocated()/2**30,3),
            "warning":"No quality improvement claim without heldout post-training eval"}
    save(out/"training.json",result)
    print("TRAIN_COMPLETE",adapter,flush=True)

def report(cfg,doc):
    _,out,_=paths(cfg)
    base=json.loads((out/"baseline.json").read_text())
    trainlog=json.loads((out/"training.json").read_text())
    assert base["frozen_dataset_sha"]==doc["hash"]==trainlog["frozen_dataset_sha"],"DATASET_DRIFT"
    tuned=evaluate(cfg,doc,adapter=out/"adapter",label="lora")
    stats={}
    lines=["# Frozen heldout controlled temporal benchmark","",
           "IMPORTANT: generated synthetic color scenes, not real-world model performance.","",
           "| Sampled frames | Base accuracy | LoRA accuracy | Delta percentage points |","|---|---:|---:|---:|"]
    for n in cfg["eval_frame_counts"]:
        a=base["metrics"][str(n)];b=tuned["metrics"][str(n)]
        av=100*a["correct"]/a["total"];bv=100*b["correct"]/b["total"]
        stats[str(n)]={"base":a,"lora":b,"delta_percentage_points":round(bv-av,2)}
        lines.append(f"| {n} | {a['correct']}/{a['total']} ({av:.1f}%) | {b['correct']}/{b['total']} ({bv:.1f}%) | {bv-av:+.1f} |")
    lines+=["","Dataset SHA256: "+doc["hash"],"",
            "Inspect baseline.json and lora.json for every raw prediction and latency.",
            "Do not call this experiment real-world video understanding."]
    (out/"COMPARISON.md").write_text("\n".join(lines)+"\n")
    save(out/"comparison.json",stats)
    print("FROZEN_HELDOUT_COMPARISON",json.dumps(stats),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["baseline","train","report"])
    parser.add_argument("--config",default="configs/round1.json")
    args=parser.parse_args()
    cfg=json.loads((ROOT/args.config).read_text())
    dataset=make_clips(cfg)
    _,_,modelpath=paths(cfg)
    if not (modelpath/"model.safetensors").is_file():raise RuntimeError("No cached weights: "+str(modelpath))
    import torch
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    if args.stage=="baseline":evaluate(cfg,dataset)
    elif args.stage=="train":train(cfg,dataset)
    else:report(cfg,dataset)
if __name__=="__main__":main()
