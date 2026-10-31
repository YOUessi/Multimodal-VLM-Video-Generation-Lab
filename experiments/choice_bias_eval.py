"""Choice-order bias study on the 120 MVBench videos never used in the first 80-video evaluation.

PREDECLARED METHOD:
- Use exactly 3 cyclic option rotations for each question; every candidate appears once in each position.
- Map output letter back to original candidate identity, not its shifted letter.
- Majority vote if >=2 rotated predictions agree; otherwise take rotation-0 result.
- Report all errors as wrong, paired delta and bootstrap CI.
- Model adapters are trained only on synthetic color videos, never on MVBench.
"""
from __future__ import annotations
import argparse,hashlib,json,random,statistics,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LETTERS="ABC"

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    temp.replace(path)

def heldout(config):
    root=ROOT/config["data_dir"]
    original=json.loads((root/"text/action_antonym.json").read_text(encoding="utf8"))
    previous=json.loads((ROOT/"outputs/mvbench_round1/manifest.json").read_text(encoding="utf8"))
    used={row["id"] for row in previous["rows"]}
    assert len(original)==200 and len(used)==80
    remaining=[(i,entry) for i,entry in enumerate(original) if i not in used]
    assert len(remaining)==120
    samples=[]
    for i,entry in remaining:
        files=list((root/"clips").rglob(entry["video"]))
        assert len(files)==1,(i,entry["video"])
        assert len(entry["candidates"])==3 and entry["answer"] in entry["candidates"]
        samples.append({"id":i,"question":entry["question"],"choices":entry["candidates"],
                        "gold_index":entry["candidates"].index(entry["answer"]),
                        "video":str(files[0].relative_to(ROOT))})
    key=hashlib.sha256(json.dumps(samples,sort_keys=True).encode()).hexdigest()
    out=ROOT/config["output_dir"]
    dest=out/"manifest.json"
    if dest.exists():
        m=json.loads(dest.read_text(encoding="utf8"))
        assert m["sha256"]==key,"OPTIONS_STUDY_DATASET_DRIFT"
    else:
        dump(dest,{"source":"PKU-Alignment/MVBench action_antonym",
            "original_80_manifest_hash":previous["sha256"],"sha256":key,
            "sample_count":len(samples),"policy":"ALL 120 complement samples, no data-driven filtering",
            "rows":samples})
    return key,samples

def rotate_choices(choices,rotation):
    assert rotation in (0,1,2)
    return [choices[(rotation+i)%3] for i in range(3)]

def backmap(predicted_letter,rotation):
    if not isinstance(predicted_letter,str) or predicted_letter not in LETTERS:return None
    return (LETTERS.index(predicted_letter)+rotation)%3

def majority_vote(preds):
    assert len(preds)==3
    for candidate in (0,1,2):
        if preds.count(candidate)>=2:return candidate
    # Predetermined tie-break, no ground-truth access.
    return preds[0]

def prompt_for(row,images,rotation):
    choices=rotate_choices(row["choices"],rotation)
    content=[{"type":"text","text":"Answer from these chronological video frames only."}]
    for i,img in enumerate(images):
        content.extend([{"type":"text","text":f"Frame {i+1} of {len(images)}:"},
                        {"type":"image","image":img}])
    options="\n".join(f"{LETTERS[k]}. {v}" for k,v in enumerate(choices))
    content.append({"type":"text","text":row["question"]+"\n"+options+
        "\nChoose one option and give only its letter and description."})
    return [{"role":"user","content":content}]

def load_model(config,variant):
    import torch
    from transformers import AutoProcessor,AutoModelForImageTextToText
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    modeldir=ROOT/config["model_dir"]
    processor=AutoProcessor.from_pretrained(modeldir,local_files_only=True)
    processor.image_processor.do_image_splitting=False
    model=AutoModelForImageTextToText.from_pretrained(modeldir,local_files_only=True,dtype=dtype)
    if variant=="synthetic_lora":
        from peft import PeftModel
        adapter=ROOT/config["adapter"]
        model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
    return model.to("cuda").eval(),processor,dtype

def evaluate(cfg,manifest_sha,rows,variant):
    import torch
    from mvbench_real_eval import frames
    from mvbench_rescore import answer_letter
    directory=ROOT/cfg["output_dir"]
    file=directory/(variant+".json")
    model,processor,dtype=load_model(cfg,variant)
    results=[]
    for frame_count in cfg["frames"]:
        for i,row in enumerate(rows):
            imgs=frames(ROOT/row["video"],frame_count)
            letters=[];original_indices=[];times=[];raws=[]
            for rotation in (0,1,2):
                messages=prompt_for(row,imgs,rotation)
                try:
                    inputs=processor.apply_chat_template(messages,add_generation_prompt=True,
                        tokenize=True,return_tensors="pt",return_dict=True).to("cuda",dtype=dtype)
                    torch.cuda.synchronize(); t=time.perf_counter()
                    with torch.inference_mode():
                        output=model.generate(**inputs,max_new_tokens=cfg["max_new_tokens"],do_sample=False)
                    torch.cuda.synchronize()
                    result=processor.decode(output[0][inputs["input_ids"].shape[-1]:],skip_special_tokens=True).strip()
                    letter=answer_letter(result)
                    seconds=round(time.perf_counter()-t,4)
                except torch.cuda.OutOfMemoryError:
                    raise
                except Exception as e:
                    result=f"INFERENCE_ERROR: {type(e).__name__}: {str(e)[:160]}"
                    letter=None;seconds=None
                letters.append(letter)
                original_indices.append(backmap(letter,rotation))
                times.append(seconds);raws.append(result)
            selected=majority_vote(original_indices)
            results.append({"source_id":row["id"],"frames":frame_count,
                "gold_index":row["gold_index"],"raw_outputs":raws,
                "rotated_answer_letters":letters,"original_candidate_indices":original_indices,
                "raw_seconds":times,
                "original_correct":original_indices[0]==row["gold_index"],
                "majority_candidate":selected,
                "majority_correct":selected==row["gold_index"],
                "rotations_correct":[p==row["gold_index"] for p in original_indices],
                "position_disagreement":len(set(original_indices))>1})
            if (i+1)%10==0:
                subset=[v for v in results if v["frames"]==frame_count]
                print("ORDER_TEST",variant,frame_count,i+1,"/",len(rows),
                      "base",sum(v["original_correct"] for v in subset),
                      "vote",sum(v["majority_correct"] for v in subset),flush=True)
    summary={}
    for n in cfg["frames"]:
        subset=[r for r in results if r["frames"]==n]
        summary[str(n)]={
            "total":len(subset),
            "original_correct":sum(r["original_correct"] for r in subset),
            "majority_correct":sum(r["majority_correct"] for r in subset),
            "rotations_correct":[sum(r["rotations_correct"][k] for r in subset) for k in range(3)],
            "disagreement_count":sum(r["position_disagreement"] for r in subset),
            "raw_rotation0_letter_counts":{ch:sum(r["rotated_answer_letters"][0]==ch for r in subset) for ch in "ABC"},
            "missing_predictions":sum(p is None for r in subset for p in r["original_candidate_indices"])
        }
    result={"variant":variant,"manifest_sha256":manifest_sha,
            "source":"MVBench 120-video complement heldout", "model_training_data":"synthetic color only",
            "timestamp_utc":datetime.now(timezone.utc).isoformat(),
            "method":"3 cyclic option rotations; majority vote with rotation-0 deterministic tie",
            "summary":summary,"predictions":results,
            "peak_gpu_gib":round(torch.cuda.max_memory_allocated()/2**30,3)}
    dump(file,result)
    print("ORDER_STUDY_DONE",variant,summary,flush=True)

def paired_ci(records,n):
    src=[(int(r["original_correct"]),int(r["majority_correct"])) for r in records if r["frames"]==n]
    assert len(src)==120
    diffs=[b-a for a,b in src]
    seed=random.Random(20261031+n)
    samples=sorted(100*sum(seed.choices(diffs,k=len(diffs)))/len(diffs) for _ in range(4000))
    return {"delta_pp":round(sum(diffs)*100/len(diffs),2),
            "bootstrap_95ci_pp":[round(samples[100],2),round(samples[3900],2)]}

def compare(cfg,manifest_sha):
    directory=ROOT/cfg["output_dir"]
    rows=[]
    table=["# Untouched MVBench 120-video option-order study","",
      "This is the complement of the previously inspected 80 public real videos.",
      "All three cyclic rotations were preregistered, and ties fall back to original ordering.",
      "No real-video model training; all ground truth comes from MVBench official annotations.",
      "","| Model | Frames | Original order | Three-rotation vote | Paired difference pp | 95% CI pp | Position disagreement |",
      "|---|---:|---:|---:|---:|---|---:|"]
    for variant in ("base","synthetic_lora"):
        src=json.loads((directory/(variant+".json")).read_text())
        if src["manifest_sha256"]!=manifest_sha:raise RuntimeError("SAMPLE_HASH_MISMATCH")
        for n in cfg["frames"]:
            summary=src["summary"][str(n)]
            diff=paired_ci(src["predictions"],n)
            rows.append({"variant":variant,"frames":n,"summary":summary,**diff})
            table.append(f"| {variant} | {n} | {summary['original_correct']}/{summary['total']} | {summary['majority_correct']}/{summary['total']} | {diff['delta_pp']:+.2f} | {diff['bootstrap_95ci_pp']} | {summary['disagreement_count']}/{summary['total']} |")
    dump(directory/"comparison.json",{"manifest_sha256":manifest_sha,"comparisons":rows,
         "protocol":"3 cyclic rotations and majority vote fixed before evaluation"})
    (directory/"ORDER_BIAS_REPORT.md").write_text("\n".join(table)+"\n",encoding="utf8")
    print("ORDER_BIAS_COMPARISON",json.dumps(rows),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["base","synthetic_lora","compare"])
    p.add_argument("--config",default="configs/choice_bias_round1.json")
    args=p.parse_args()
    cfg=json.loads((ROOT/args.config).read_text())
    manifest_sha,rows=heldout(cfg)
    if args.stage=="compare":compare(cfg,manifest_sha)
    else:evaluate(cfg,manifest_sha,rows,args.stage)
if __name__=="__main__":main()
