"""Independent cross-task evaluation: STAR action-sequence-trained LoRA -> MVBench scene transition.

All public scene-transition videos are evaluation ONLY. No real-video QA from
this task is used in optimizer gradients, checkpoint selection, or prompt tuning.
Trained LoRA seed is selected exclusively on the grouped STAR validation set.
"""
from __future__ import annotations
import argparse,hashlib,json,random,statistics,time
from collections import Counter
from pathlib import Path
from star_real_sft import extract_frames,load_model,prompt,parse_answer,decoded_frame_count,paired_ci
ROOT=Path(__file__).resolve().parents[1]
LETTERS="ABCD"
def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as file:
        for c in iter(lambda:file.read(2**20),b""):h.update(c)
    return h.hexdigest()
def save(path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(record,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    temp.replace(path)
def manifest(cfg):
    src=ROOT/cfg["source_dir"]
    labels=src/"text/scene_transition.json"
    if sha256(labels)!=cfg["labels_sha256"]:raise RuntimeError("SCENE_LABEL_SHA_CHANGED")
    archive=src/"provenance.json"
    prov=json.loads(archive.read_text())
    if prov["video_archive_sha256"]!=cfg["archive_sha256"] or prov["videos_matched"]!=200:
        raise RuntimeError("SCENE_ARCHIVE_MISMATCH")
    official=json.loads(labels.read_text())
    assert len(official)==200 and len({r["video"] for r in official})==200
    sample_count=cfg["sample_count"]
    if sample_count!=200:raise ValueError("Use all 200 external samples to prevent cherry picking")
    # Grouped STAR content excludes all scene files: check full SHA, not just names.
    training=json.loads((ROOT/cfg["grouped_star_manifest"]).read_text())
    group_hashes={r["content_sha256"] for r in training["rows"]}
    assert len(group_hashes)==193
    rows=[];contents=set()
    for i,label in enumerate(official):
        file=src/"clips"/label["video"]
        if not file.is_file():raise FileNotFoundError(file)
        h=sha256(file)
        if h in group_hashes:raise RuntimeError("SCENE_VIDEO_SEEN_IN_STAR_TRAIN_VAL_OR_TEST")
        if h in contents:raise RuntimeError("SCENE_EXACT_DUPLICATE_IN_PUBLIC_EXTERNAL_TEST")
        contents.add(h)
        if len(label["candidates"])!=4 or label["answer"] not in label["candidates"]:
            raise RuntimeError("BAD_OFFICIAL_LABEL")
        rows.append({"source_id":i,"video":str(file.relative_to(ROOT)),
            "decoded_frames":decoded_frame_count(file),
            "content_sha256":h,"question":label["question"],"choices":label["candidates"],
            "answer":LETTERS[label["candidates"].index(label["answer"])]})
        if (i+1)%40==0:print("SCENE_EXTERNAL_READ_CHECK",i+1,"/200",flush=True)
    digest=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    out=ROOT/cfg["output_dir"]/"manifest.json"
    if out.exists():
        old=json.loads(out.read_text())
        if old["rows_sha256"]!=digest:raise RuntimeError("EXTERNAL_SCENE_MANIFEST_DRIFT")
    else:save(out,{"rows_sha256":digest,"source":"PKU-Alignment/MVBench scene_transition",
        "source_revision":cfg["source_revision"],"samples":len(rows),
        "star_overlap_count":0,"source_task_used_for_training":False,"rows":rows})
    return digest,rows
def evaluate(cfg,digest,rows,variant):
    import torch
    out=ROOT/cfg["output_dir"]
    filename=out/(variant+".json")
    if variant=="grouped_lora":
        selector=json.loads((ROOT/cfg["grouped_adapter_dir"]/"chosen_seed.json").read_text())
        seed=selector["chosen_seed"]
        adapter=ROOT/cfg["grouped_adapter_dir"]/(f"adapter_{seed}")
        if not (adapter/"adapter_model.safetensors").is_file():
            raise RuntimeError("GROUPED_ADAPTER_NOT_TRAINED_YET")
    else:adapter=None
    model,proc,dtype=load_model({"model_dir":cfg["model_dir"]},adapter)
    result=[]
    for budget in cfg["frame_counts"]:
        for index,row in enumerate(rows):
            images=extract_frames(ROOT/row["video"],budget,row["decoded_frames"])
            tokens=proc.apply_chat_template(prompt(row,images),
                   add_generation_prompt=True,tokenize=True,
                   return_tensors="pt",return_dict=True).to("cuda",dtype=dtype)
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode():
                generated=model.generate(**tokens,max_new_tokens=cfg["max_new_tokens"],do_sample=False)
            torch.cuda.synchronize()
            answer=proc.decode(generated[0][tokens["input_ids"].shape[-1]:],skip_special_tokens=True).strip()
            prediction=parse_answer(answer)
            result.append({"id":row["source_id"],"frames":budget,"gold":row["answer"],
                "pred":prediction,"correct":prediction==row["answer"],
                "raw":answer,"seconds":round(time.perf_counter()-start,4)})
            if (index+1)%40==0:
                group=[x for x in result if x["frames"]==budget]
                print("EXTERNAL_SCENE_EVAL",variant,budget,index+1,len(rows),
                      "correct",sum(x["correct"] for x in group),flush=True)
    metrics={}
    for n in cfg["frame_counts"]:
        sub=[x for x in result if x["frames"]==n]
        metrics[str(n)]={"correct":sum(x["correct"] for x in sub),
            "total":len(sub),"unparseable":sum(x["pred"] is None for x in sub),
            "median_generate_seconds":round(statistics.median(x["seconds"] for x in sub),4)}
    save(filename,{"variant":variant,"task":"MVBench scene_transition",
        "task_used_for_training":False,"manifest_sha256":digest,
        "selected_adapter_from_STAR_validation":variant=="grouped_lora",
        "metrics":metrics,"predictions":result,
        "peak_gpu_gib":round(torch.cuda.max_memory_allocated()/2**30,3)})
    print("EXTERNAL_SCENE_FINISHED",variant,metrics,flush=True)
def compare(cfg,digest):
    target=ROOT/cfg["output_dir"]
    a=json.loads((target/"base.json").read_text())
    b=json.loads((target/"grouped_lora.json").read_text())
    assert a["manifest_sha256"]==b["manifest_sha256"]==digest
    lines=["# Unseen cross-task MVBench scene transition evaluation","",
           "200 original public scene-transition videos, evaluation ONLY.",
           "No training or validation was performed on scene-transition questions.",
           "All 200 local MP4 SHA-256 hashes were verified distinct from 193 STAR contents.",
           "","| Frames | Base | SHA-grouped STAR LoRA | Delta pp | 95% paired bootstrap CI pp |",
           "|---|---:|---:|---:|---|"]
    results={}
    for n in cfg["frame_counts"]:
        base=a["metrics"][str(n)];tuned=b["metrics"][str(n)]
        ci=paired_ci({"results":[{"id":x["id"],"frames":x["frames"],"correct":x["correct"]}
                                  for x in a["predictions"]]},
                     {"results":[{"id":x["id"],"frames":x["frames"],"correct":x["correct"]}
                                  for x in b["predictions"]]},n)
        results[str(n)]={"base":base,"grouped_lora":tuned,**ci}
        lines.append(f"| {n} | {base['correct']}/{base['total']} | {tuned['correct']}/{tuned['total']} | {ci['delta_pp']:+.2f} | {ci['bootstrap_95_ci_pp']} |")
    lines+=["","The action-sequence training task and scene-transition test differ;",
            "this is an external transfer test, not a scene-specific fine-tuning benchmark.",
            "Public benchmark contamination through the original pretrained model cannot be ruled out.",
            "No test prompt tuning or checkpoint reselection is allowed."]
    (target/"CROSS_TASK_REPORT.md").write_text("\n".join(lines)+"\n")
    save(target/"comparison.json",{"source":"MVBench public scene_transition",
         "samples":200,"star_overlap":0,"manifest_sha256":digest,
         "results":results,"training_on_external_test":False})
    print("CROSS_TASK_REPORT_READY",json.dumps(results),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["prepare","base","grouped_lora","compare"])
    p.add_argument("--config",default="configs/mvbench_scene_external_v1.json")
    a=p.parse_args()
    cfg=json.loads((ROOT/a.config).read_text())
    digest,rows=manifest(cfg) if a.stage=="prepare" else (
        lambda m:(m["rows_sha256"],m["rows"])
    )(json.loads((ROOT/cfg["output_dir"]/"manifest.json").read_text()))
    if a.stage=="prepare":
        print("EXTERNAL_SCENE_MANIFEST_FROZEN",digest,flush=True)
    elif a.stage=="compare":
        compare(cfg,digest)
    else:evaluate(cfg,digest,rows,a.stage)
if __name__=="__main__":main()
