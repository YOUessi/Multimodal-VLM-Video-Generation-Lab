"""Offline MCQ parser correction of immutable raw MVBench predictions.

Evaluation v1 incorrectly failed to parse the SmolVLM2 reply style "A. ...".
This rerates from original raw outputs, WITHOUT rerunning inference and WITHOUT
using gold labels to infer a model's intended choice. All original files retained.
"""
import argparse,hashlib,json,random,re,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def answer_letter(raw):
    s=str(raw or "").strip()
    # Explicit mandated form; otherwise an initial choice followed by punctuation.
    found=re.findall(r"\bANSWER\s*:\s*([A-C])\b",s,flags=re.IGNORECASE)
    if found:return found[-1].upper()
    m=re.match(r"^\s*([A-C])\s*[.)\]:-]\s*(?:\S|$)",s,flags=re.IGNORECASE)
    if m:return m.group(1).upper()
    m=re.match(r"^\s*(?:option|choice)\s*([A-C])\b",s,flags=re.IGNORECASE)
    if m:return m.group(1).upper()
    m=re.match(r"^\s*(?:the\s+)?(?:correct\s+)?answer\s+is\s+([A-C])\b",s,flags=re.IGNORECASE)
    if m:return m.group(1).upper()
    clean=s.upper().strip(" .():[]")
    return clean if clean in ("A","B","C") else None

def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n")
    tmp.replace(path)

def regrade(source_path,manifest):
    orig=json.loads(source_path.read_text(encoding="utf8"))
    if orig["manifest_sha256"]!=manifest["sha256"]:raise RuntimeError("MANIFEST_MISMATCH")
    n=len(manifest["rows"]);sizes=[1,3,5]
    assert len(orig["predictions"])==n*len(sizes)
    gold={(r["id"],k):r["gold"] for r in manifest["rows"] for k in sizes}
    predictions=[]
    for row in orig["predictions"]:
        key=(row["id"],row["frames"])
        assert key in gold
        if row["gold"]!=gold[key]:raise RuntimeError("GOLD_LABEL_MUTATION")
        new=dict(row)
        new["old_pred"]=row["pred"]
        new["pred"]=answer_letter(row["raw"]) if row["error"] is None else None
        new["correct"]=new["pred"]==row["gold"]
        predictions.append(new)
    metrics={}
    for frames in sizes:
        subset=[r for r in predictions if r["frames"]==frames]
        assert len(subset)==n
        hits=sum(r["correct"] for r in subset)
        metrics[str(frames)]={"correct":hits,"total":n,"accuracy":round(hits/n,4),
                             "invalid_output":sum(r["pred"] is None for r in subset),
                             "inference_errors":sum(r["error"] is not None for r in subset),
                             "answer_counts":{ch:sum(r["pred"]==ch for r in subset) for ch in "ABC"},
                             "gold_counts":{ch:sum(r["gold"]==ch for r in subset) for ch in "ABC"}}
    return {"variant":orig["variant"],"manifest_sha256":orig["manifest_sha256"],
            "source_file_sha256":hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "parser_version":"mcq_v2","label_policy":"Official MVBench answers; invalid counted wrong",
            "rescore_only":True,"metrics":metrics,"predictions":predictions}

def paired(a,b,n):
    av={r["id"]:int(r["correct"]) for r in a["predictions"] if r["frames"]==n}
    bv={r["id"]:int(r["correct"]) for r in b["predictions"] if r["frames"]==n}
    if set(av)!=set(bv):raise RuntimeError("NOT_PAIRED")
    delta=[bv[i]-av[i] for i in sorted(av)]
    rng=random.Random(20261031+n)
    samples=sorted(100*sum(rng.choices(delta,k=len(delta)))/len(delta) for _ in range(4000))
    return {"delta_pp":round(100*sum(delta)/len(delta),2),"bootstrap_95ci_pp":[round(samples[100],2),round(samples[3900],2)]}

def run(root):
    manifest=json.loads((root/"manifest.json").read_text())
    base=regrade(root/"base.json",manifest)
    lora=regrade(root/"synthetic_lora.json",manifest)
    output=root/"rescored_v2"
    atomic_json(output/"base_scored.json",base)
    atomic_json(output/"synthetic_lora_scored.json",lora)
    comparisons={}
    lines=["# Corrected MVBench real-video MCQ metrics","",
           "The initial parser failed to recognize answers like C. description. This v2 is rescored from saved raw model outputs, not new model inference.",
           "","| Frames | Base correct | Synthetic-trained LoRA correct | Difference pp | Paired bootstrap 95% CI pp |",
           "|---|---:|---:|---:|---|"]
    for n in [1,3,5]:
        delta=paired(base,lora,n)
        comparisons[str(n)]={"base":base["metrics"][str(n)],"synthetic_lora":lora["metrics"][str(n)],**delta}
        lines.append(f"| {n} | {base['metrics'][str(n)]['correct']}/{base['metrics'][str(n)]['total']} | {lora['metrics'][str(n)]['correct']}/{lora['metrics'][str(n)]['total']} | {delta['delta_pp']:+.2f} | {delta['bootstrap_95ci_pp']} |")
    lines+=["","Dataset manifest SHA256: "+manifest["sha256"],
             "","This is ONLY the public MVBench action_antonym subset, 80 videos. Videos were not used for training.",
             "Synthetic color LoRA is not expected to generalize to natural actions. Results may be negative.",
             "The public benchmark is not guaranteed unseen in pretraining.",
             "The original parser-v1 files remain retained for auditing."]
    atomic_json(output/"comparison_scored.json",{"parser_version":"mcq_v2",
       "real_video":True,"manifest_sha256":manifest["sha256"],"source_of_predictions":"saved raw model outputs",
       "stats":comparisons,"important":"Supersedes unparsed initial scores in ../comparison.json"})
    (output/"REAL_VIDEO_V2.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("CORRECTED_RESULTS",json.dumps(comparisons),flush=True)
    print("RESCORE_COMPLETE",output)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",default="outputs/mvbench_round1")
    a=p.parse_args()
    run(ROOT/a.output_dir)
if __name__=="__main__":main()
