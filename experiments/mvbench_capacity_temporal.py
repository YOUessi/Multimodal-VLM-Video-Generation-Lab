"""External MVBench capacity + temporal evidence ablation.

Evaluates two predeclared models (SmolVLM2-500M, 2.2B), two public MVBench
tasks (CLEVRER moving_direction, Moments in Time scene_transition) and three
frame conditions (0 text-only, uniform 3, uniform 5) WITHOUT fine-tuning.
All 200 official QA examples per task are evaluated; no test-driven tuning.
Per-example logits are not used for model selection. Raw videos remain local.
"""
from __future__ import annotations
import argparse, hashlib, json, re, statistics, subprocess, time, random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LETTERS="ABCDEFGH"

def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda:file.read(2**20),b""):digest.update(chunk)
    return digest.hexdigest()

def save_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    tmp.replace(path)

def parsed_answer(text,num_choices):
    text=str(text or "").strip()
    permitted=LETTERS[:num_choices]
    forms=[r"\bANSWER\s*:\s*([A-H])\b(?!\s*/)",
           r"^\s*(?:Option|Choice)\s+([A-H])\b",
           r"^\s*(?:The\s+)?(?:correct\s+)?answer\s+is\s+([A-H])\b",
           r"^\s*([A-H])\s*[.)\]:-]\s*(?:\S|$)"]
    for pattern in forms:
        hits=re.findall(pattern,text,flags=re.IGNORECASE|re.MULTILINE)
        if hits:
            answer=hits[-1].upper()
            return answer if answer in permitted else None
    simple=text.upper().strip(" .:()[]")
    return simple if simple in permitted else None

def ffprobe_frames(path):
    result=subprocess.run(["ffprobe","-v","error","-count_frames",
        "-select_streams","v:0","-show_entries","stream=nb_read_frames",
        "-of","default=noprint_wrappers=1:nokey=1",str(path)],
        capture_output=True,text=True,check=True,timeout=50)
    found=[i.strip() for i in result.stdout.splitlines() if i.strip().isdigit()]
    if len(found)!=1 or int(found[0])<3:
        raise RuntimeError("DECODE_COUNT_NOT_AVAILABLE:"+str(path))
    return int(found[0])

def cache_manifest(cfg):
    output=ROOT/cfg["output_dir"]
    output.mkdir(parents=True,exist_ok=True)
    star=json.loads((ROOT/cfg["star_manifest"]).read_text(encoding="utf8"))
    assert star.get("unique_content_hashes")==193
    trained_sha={v["content_sha256"] for v in star["rows"]}
    manifest_path=output/"manifest.json"
    data=[];seen_across={}
    for task,config in cfg["tasks"].items():
        data_root=ROOT/config["data_dir"]
        labels=data_root/config["label_file"]
        if sha256_file(labels)!=config["label_sha256"]:
            raise RuntimeError("OFFICIAL_LABEL_HASH_DRIFT:"+task)
        provenance=json.loads((data_root/"provenance.json").read_text(encoding="utf8"))
        if (provenance.get("archive_sha256") or provenance.get("video_archive_sha256"))!=config["archive_sha256"]:
            raise RuntimeError("OFFICIAL_ARCHIVE_HASH_DRIFT:"+task)
        entries=json.loads(labels.read_text(encoding="utf8"))
        if len(entries)!=200:
            raise RuntimeError("UNEXPECTED_PUBLIC_TASK_SIZE:"+task)
        observed_video=set();observed_hash=set()
        for index,row in enumerate(entries):
            video_name=row["video"]
            if video_name in observed_video:
                raise RuntimeError("REPEATED_VIDEO_FILENAME:"+task)
            observed_video.add(video_name)
            file=data_root/"clips"/video_name
            if not file.is_file():
                raise FileNotFoundError(file)
            sha=sha256_file(file)
            if sha in observed_hash or sha in trained_sha:
                raise RuntimeError("EXACT_FILE_DUPLICATE_IN_TASK_OR_WITH_STAR:"+task)
            if sha in seen_across:
                raise RuntimeError("EXACT_FILE_DUPLICATE_ACROSS_EXTERNAL_TASKS")
            observed_hash.add(sha);seen_across[sha]=task
            choices=row["candidates"]
            if not 2<=len(choices)<=8 or row["answer"] not in choices:
                raise RuntimeError("INVALID_PUBLIC_MCQ:"+str(index))
            decoded=ffprobe_frames(file)
            data.append({"key":task+"_"+str(index).zfill(4),"task":task,
                "source_index":index,"video":str(file.relative_to(ROOT)),
                "video_sha256":sha,"decoded_frames":decoded,
                "question":row["question"],"choices":choices,
                "gold":LETTERS[choices.index(row["answer"])]})
            if (index+1)%40==0:
                print("DATA_PREFLIGHT",task,index+1,"/200",flush=True)
    if len(data)!=400 or len(seen_across)!=400:
        raise RuntimeError("EXPECTED_400_DISTINCT_EXTERNAL_VIDEO_FILES")
    digest=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
    frozen={
        "version":1,"timestamp_utc":now(),
        "manifest_sha256":digest,"sample_count":len(data),
        "task_counts":dict(Counter(row["task"] for row in data)),
        "total_content_sha256_groups":len(seen_across),
        "source_star_content_overlap":0,
        "input_conditions":cfg["frame_counts"],"models":list(cfg["models"]),
        "rule":"Official QA, all original examples, no train/validation or prompt tuning",
        "rows":data}
    if manifest_path.is_file():
        old=json.loads(manifest_path.read_text(encoding="utf8"))
        # Do not compare timestamps: only the frozen materialized selection.
        if old["manifest_sha256"]!=digest or old["input_conditions"]!=frozen["input_conditions"]:
            raise RuntimeError("FROZEN_CAPACITY_MANIFEST_DRIFT")
    else:save_json(manifest_path,frozen)
    print("MVBench_400_EXAMPLES_FROZEN",digest,flush=True)
    return frozen

def frame_positions(decoded,n):
    if n==0:return []
    if n==1:return [round(.5*(decoded-1))]
    return [max(0,min(decoded-1,round((.05+.90*j/(n-1))*(decoded-1)))) for j in range(n)]

def open_frames(row,n):
    import cv2
    from PIL import Image
    if n==0:return []
    path=ROOT/row["video"]
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise RuntimeError("VIDEO_OPEN_FAILED:"+str(path))
    frames=[]
    try:
        for idx in frame_positions(row["decoded_frames"],n):
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
            good,frame=cap.read()
            if not good:raise RuntimeError("VIDEO_FRAME_DECODE_FAILED:"+str(path))
            image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            image.thumbnail((640,480))
            frames.append(image)
    finally:cap.release()
    return frames

def build_message(row,images):
    n=len(images)
    content=[
      {"type":"text","text":(
        "Video frames below are in chronological order. "
        "Answer only using the actual visible video evidence." if n else
        "No video frames are provided. Answer only from the question wording and option text."
      )}]
    for idx,img in enumerate(images):
        content.append({"type":"text","text":"Chronological video frame "+str(idx+1)+"/"+str(n)+":"})
        content.append({"type":"image","image":img})
    options="\n".join(LETTERS[i]+". "+str(v) for i,v in enumerate(row["choices"]))
    content.append({"type":"text","text":str(row["question"])+"\n"+options+
                    "\nChoose exactly one option. On the final line write ANSWER: followed by ONE selected letter."})
    return [{"role":"user","content":content}]

def cached_model_integrity(model_path,details):
    path=ROOT/model_path
    listed=details["weight_files"]
    for filename,expected_bytes in listed.items():
        file=path/filename
        if not file.is_file():
            raise RuntimeError("MODEL_SHARD_NOT_CACHED:"+str(file))
        if file.stat().st_size!=expected_bytes:
            raise RuntimeError("MODEL_SHARD_SIZE_MISMATCH:"+str(file))
    return path

def load_model(model_path,details):
    import torch
    from transformers import AutoModelForImageTextToText,AutoProcessor
    path=cached_model_integrity(model_path,details)
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor=AutoProcessor.from_pretrained(str(path),local_files_only=True)
    processor.image_processor.do_image_splitting=False
    model=AutoModelForImageTextToText.from_pretrained(
        str(path),local_files_only=True,dtype=dtype,
        low_cpu_mem_usage=True).to("cuda").eval()
    print("MODEL_READY",str(path),torch.cuda.get_device_name(0),
          "allocated_gib",round(torch.cuda.memory_allocated()/2**30,3),flush=True)
    return model,processor,dtype

def infer(model,processor,dtype,messages,max_new_tokens):
    import torch
    features=processor.apply_chat_template(messages,
       add_generation_prompt=True,tokenize=True,return_tensors="pt",
       return_dict=True).to("cuda",dtype=dtype)
    torch.cuda.synchronize()
    started=time.perf_counter()
    with torch.inference_mode():
        generated=model.generate(**features,do_sample=False,max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    suffix=generated[0][features["input_ids"].shape[-1]:]
    return processor.decode(suffix,skip_special_tokens=True).strip(),round(time.perf_counter()-started,4)

def preflight(cfg,model_name):
    from PIL import Image
    import torch
    details=cfg["models"][model_name]
    model,processor,dtype=load_model(details["model_dir"],details)
    sample={"question":"What is the color of the square?","choices":["Red","Blue","Green","Yellow"]}
    for budget in cfg["frame_counts"]:
        fake=[Image.new("RGB",(256,192),color="red") for _ in range(budget)]
        ans,seconds=infer(model,processor,dtype,build_message(sample,fake),cfg["max_new_tokens"])
        print("MODEL_PROTOCOL_PREFLIGHT",model_name,budget,repr(ans),seconds,flush=True)
    print("MODEL_GPU_PREFLIGHT_DONE",model_name,flush=True)

def evaluate(cfg,manifest,model_name):
    import torch
    details=cfg["models"][model_name]
    output=ROOT/cfg["output_dir"]
    target=output/(model_name+".json")
    partial=output/(model_name+".partial.json")
    if target.is_file():
        old=json.loads(target.read_text(encoding="utf8"))
        if old["manifest_sha256"]!=manifest["manifest_sha256"]:
            raise RuntimeError("COMPLETED_RESULT_MANIFEST_DRIFT")
        print("ALREADY_COMPLETE",model_name,flush=True);return
    processed={}
    if partial.is_file():
        previous=json.loads(partial.read_text(encoding="utf8"))
        if previous["manifest_sha256"]!=manifest["manifest_sha256"]:
            raise RuntimeError("PARTIAL_RESULT_MANIFEST_DRIFT")
        if previous["model_name"]!=model_name:
            raise RuntimeError("PARTIAL_RESULT_MODEL_DRIFT")
        processed={r["task"]+"|"+r["key"]+"|"+str(r["frames"]):r for r in previous["predictions"]}
    model,processor,dtype=load_model(details["model_dir"],details)
    torch.cuda.reset_peak_memory_stats()
    total=len(manifest["rows"])*len(cfg["frame_counts"])
    for budget in cfg["frame_counts"]:
        for row in manifest["rows"]:
            key=row["task"]+"|"+row["key"]+"|"+str(budget)
            if key in processed:continue
            begin=time.perf_counter()
            images=open_frames(row,budget)
            preparation=round(time.perf_counter()-begin,4)
            messages=build_message(row,images)
            # OOM and CPU decoding errors fail this stage; no silent skipping.
            raw,generate_seconds=infer(model,processor,dtype,messages,cfg["max_new_tokens"])
            prediction=parsed_answer(raw,len(row["choices"]))
            record={"task":row["task"],"key":row["key"],"frames":budget,
                    "gold":row["gold"],"pred":prediction,"correct":prediction==row["gold"],
                    "raw":raw,"decode_seconds":preparation,"generate_seconds":generate_seconds}
            processed[key]=record
            if len(processed)%10==0:
                partial_doc={"model_name":model_name,
                  "manifest_sha256":manifest["manifest_sha256"],
                  "predictions":list(processed.values()),
                  "note":"Incomplete checkpoint, not final metrics"}
                save_json(partial,partial_doc)
                print("MODEL_EVAL_PROGRESS",model_name,len(processed),"/",total,
                      "last_task",row["task"],"frames",budget,flush=True)
    rows=list(processed.values())
    if len(rows)!=total:raise RuntimeError("INCOMPLETE_REAL_VIDEO_EVAL")
    result={
      "model_name":model_name,"model_revision":details["revision"],
      "manifest_sha256":manifest["manifest_sha256"],
      "timestamp_utc":now(),"model_loading_dtype":str(dtype),
      "frame_counts":cfg["frame_counts"],
      "peak_gpu_allocated_gib":round(torch.cuda.max_memory_allocated()/2**30,3),
      "predictions":rows,
      "note":"All 200 official items per task; invalid parsed answers count wrong; no training"}
    save_json(target,result)
    print("EVALUATION_DONE",model_name,"predictions",len(rows),
          "peak_gib",result["peak_gpu_allocated_gib"],flush=True)

def paired_difference(a,b,seed,boots=5000):
    if set(a)!=set(b):raise RuntimeError("DIFFERENT_PAIRED_SAMPLES")
    keys=sorted(a)
    delta=[int(b[k])-int(a[k]) for k in keys]
    rng=random.Random(seed)
    distribution=sorted(sum(rng.choices(delta,k=len(delta)))/len(delta)*100
                        for _ in range(boots))
    return {"delta_pp":round(sum(delta)/len(delta)*100,3),
            "bootstrap_95_ci_pp":[round(distribution[int(.025*boots)],3),
                                   round(distribution[int(.975*boots)],3)]}

def summarise(cfg,manifest):
    output=ROOT/cfg["output_dir"]
    models={}
    for name in cfg["models"]:
        result=json.loads((output/(name+".json")).read_text(encoding="utf8"))
        if result["manifest_sha256"]!=manifest["manifest_sha256"]:
            raise RuntimeError("MODEL_COMPARISON_MANIFEST_MISMATCH")
        models[name]=result
    data=manifest["rows"]
    gold={(r["task"],r["key"]):r["gold"] for r in data}
    by_model={}
    for name,values in models.items():
        groups={}
        for task in cfg["tasks"]:
            for budget in cfg["frame_counts"]:
                group=[x for x in values["predictions"]
                       if x["task"]==task and x["frames"]==budget]
                if len(group)!=200 or len({r["key"] for r in group})!=200:
                    raise RuntimeError("TASK_INCOMPLETE:"+name+task+str(budget))
                if any(r["gold"]!=gold[(r["task"],r["key"])] for r in group):
                    raise RuntimeError("SOURCE_GOLD_MISMATCH")
                lat=[r["generate_seconds"] for r in group]
                counts=Counter(r["pred"] for r in group)
                groups[task+"_"+str(budget)]={"correct":sum(r["correct"] for r in group),
                    "total":len(group),
                    "unparseable":sum(r["pred"] is None for r in group),
                    "prediction_counts":{str(key):v for key,v in counts.items()},
                    "latency_median_s":round(statistics.median(lat),4),
                    "latency_p95_s":round(sorted(lat)[int(.95*(len(lat)-1))],4)}
        by_model[name]=groups
    contrasts=[]
    name_500, name_2b=cfg["comparison_order"]
    for task in cfg["tasks"]:
        for n in cfg["frame_counts"]:
            a={r["key"]:r["correct"] for r in models[name_500]["predictions"]
               if r["task"]==task and r["frames"]==n}
            b={r["key"]:r["correct"] for r in models[name_2b]["predictions"]
               if r["task"]==task and r["frames"]==n}
            contrast=paired_difference(a,b,seed=cfg["bootstrap_seed"]+n)
            contrasts.append({"comparison":"2.2B minus 500M","task":task,"frames":n,**contrast})
        for name in cfg["models"]:
            groups={n:{r["key"]:r["correct"] for r in models[name]["predictions"]
                       if r["task"]==task and r["frames"]==n} for n in cfg["frame_counts"]}
            for early,later in ((0,3),(0,5),(3,5)):
                contrast=paired_difference(groups[early],groups[later],
                                           seed=cfg["bootstrap_seed"]+early*11+later)
                contrasts.append({"comparison":name+" "+str(later)+"f minus "+str(early)+"f",
                                  "task":task,"frames":later,**contrast})
    doc={
      "timestamp_utc":now(),"manifest_sha256":manifest["manifest_sha256"],
      "models":by_model,"contrasts":contrasts,
      "limitations":["moving_direction is simulated CLEVRER video, scene_transition contains public natural videos",
                     "public benchmark content may have been used in foundation model pretraining",
                     "question-only baseline helps estimate text/option priors but not all language leakage",
                     "one frozen inference run; no hyperparameter selection from test scores",
                     "non-significant contrasts cannot establish capacity or temporal gain"]}
    save_json(output/"comparison.json",doc)
    text=["# Independent public MVBench capacity and temporal-evidence comparison","",
          "Compared frozen SmolVLM2-500M and SmolVLM2-2.2B (no task-specific training).",
          "Moving Direction is synthetic CLEVRER video; Scene Transition is a separate public natural-video task.",
          "All 200 official questions per task, all input budgets 0, 3, 5 frames; no choice shuffling or posthoc sample filtering.",
          "","| Task | Model | Text-only /200 | 3 frames /200 | 5 frames /200 | Peak VRAM GiB |",
          "|---|---|---:|---:|---:|---:|"]
    for task in cfg["tasks"]:
        for name in cfg["models"]:
            g=by_model[name];m=models[name]
            cells=[str(g[task+"_"+str(n)]["correct"]) for n in cfg["frame_counts"]]
            text.append("| "+task+" | "+name+" | "+" | ".join(cells)+" | "+str(m["peak_gpu_allocated_gib"])+" |")
    text+=["","## Paired contrasts: percentage points and bootstrap 95% confidence intervals",
            "","| Contrast | Task | Frame condition | Delta pp | 95% CI pp |",
            "|---|---|---:|---:|---|"]
    for item in contrasts:
        text.append("| "+item["comparison"]+" | "+item["task"]+" | "+str(item["frames"])+
                    " | "+str(item["delta_pp"])+" | "+str(item["bootstrap_95_ci_pp"])+" |")
    text+=["","## Methodological limitations"]+[("* "+x) for x in doc["limitations"]]
    (output/"CAPACITY_TEMPORAL_REPORT.md").write_text("\n".join(text)+"\n",encoding="utf8")
    print("CAPACITY_TEMPORAL_REPORT_READY",output/"CAPACITY_TEMPORAL_REPORT.md",flush=True)

def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("stage",choices=["prepare","preflight","evaluate","compare"])
    cli.add_argument("--model",choices=["500m","2b"])
    cli.add_argument("--config",default="configs/capacity_temporal_v1.json")
    args=cli.parse_args()
    cfg=json.loads((ROOT/args.config).read_text(encoding="utf8"))
    if args.stage=="prepare":cache_manifest(cfg);return
    manifest=json.loads((ROOT/cfg["output_dir"]/"manifest.json").read_text(encoding="utf8"))
    if args.stage=="preflight":
        if args.model is None:raise ValueError("--model needed")
        preflight(cfg,args.model)
    elif args.stage=="evaluate":
        if args.model is None:raise ValueError("--model needed")
        evaluate(cfg,manifest,args.model)
    elif args.stage=="compare":summarise(cfg,manifest)

if __name__=="__main__":
    main()
