"""Read-only question-only / visual evidence & answer-position bias diagnostics.

No model calls, no new prompts, no test sample deletions or training. Only
summarizes raw predictions already saved in frozen batch19.
"""
from __future__ import annotations
import hashlib,json,math
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def index_rows(predictions):
    return {(r["task"],r["key"],r["frames"]):r for r in predictions}

def paired_transitions(a,b):
    if set(a)!=set(b):raise RuntimeError("UNEQUAL_COMPARISON_SAMPLES")
    result={"items":len(a),
            "both_correct":0,"both_wrong":0,
            "video_rescues":0,"video_regressions":0,"choice_changes":0}
    for key in a:
        left,right=a[key],b[key]
        if left["gold"]!=right["gold"]:raise RuntimeError("GOLD_LABEL_DRIFT")
        if left["pred"]!=right["pred"]:
            result["choice_changes"]+=1
        if left["correct"] and right["correct"]:result["both_correct"]+=1
        elif not left["correct"] and not right["correct"]:result["both_wrong"]+=1
        elif not left["correct"] and right["correct"]:result["video_rescues"]+=1
        else:result["video_regressions"]+=1
    return result

def entropy(probs):
    n=sum(probs)
    if not n:return None
    return round(-sum((x/n)*math.log2(x/n) for x in probs if x),4)

def diagnostics():
    base=ROOT/"outputs/capacity_temporal_v1"
    manifest=json.loads((base/"manifest.json").read_text(encoding="utf8"))
    model_rows={name:index_rows(json.loads((base/(name+".json")).read_text())["predictions"])
                for name in ("500m","2b")}
    tasks=list(manifest["task_counts"])
    report={"created_utc":datetime.now(timezone.utc).isoformat(),
            "manifest_sha256":manifest["manifest_sha256"],
            "descriptive_only":True,
            "tasks":tasks,"by_task":{}}
    md=["# Frozen 0/3/5 frame evidence-use and option-bias diagnostics","",
        "This report only analyzes fixed raw predictions, no retraining, no test-driven prompt optimization.",
        "","## Text-only versus video frames","",
        "| Task | Model | Condition | Rescued from wrong | Lost former correct | Changed option |",
        "|---|---|---|---:|---:|---:|"]
    for task in tasks:
        doc={}
        items=[r for r in manifest["rows"] if r["task"]==task]
        gold=Counter(row["gold"] for row in items)
        for model in ("500m","2b"):
            index=model_rows[model]
            group={n:{row["key"]:index[(task,row["key"],n)] for row in items}
                   for n in (0,3,5)}
            pairs={}
            for lhs,rhs in ((0,3),(0,5),(3,5)):
                transitions=paired_transitions(group[lhs],group[rhs])
                name=f"{lhs}to{rhs}"
                pairs[name]=transitions
                md.append(f"| {task} | {model} | {lhs}→{rhs} frames | "
                   f"{transitions['video_rescues']} | {transitions['video_regressions']} | "
                   f"{transitions['choice_changes']}/{len(items)} |")
            bias={}
            for n in (0,3,5):
                pred=Counter(r["pred"] for r in group[n].values())
                bias[str(n)]={"prediction_counts":dict(pred),
                    "label_counts":dict(gold),
                    "prediction_entropy_bits":entropy([pred[k] for k in "ABCD"]),
                    "majority_gold_accuracy":round(max(gold.values())/len(items),4),
                    "no_parse_count":pred[None]}
            doc[model]={"pairs":pairs,"bias":bias}
        report["by_task"][task]=doc
    md+=["","## Limitations",
         "- Model choices can change without becoming correct; change rate is not evidence of video understanding.",
         "- CLEVRER Moving Direction is simulated video, not natural camera footage.",
         "- The official public benchmark may be present in the pretrained model's original corpora.",
         "- The same 200 videos are used in 0, 3 and 5-frame comparisons; paired statistics matter.",
         "- Do not tune answers, prompting or weights based on these evaluated test instances."]
    (base/"EVIDENCE_USE_DIAGNOSTICS.md").write_text("\n".join(md)+"\n",encoding="utf8")
    (base/"EVIDENCE_USE_DIAGNOSTICS.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("EVIDENCE_USE_DIAGNOSTICS_DONE",base/"EVIDENCE_USE_DIAGNOSTICS.md",flush=True)

if __name__=="__main__":
    diagnostics()
