"""Explicitly retrospective, perceptual-candidate-grouped STAR rerun report."""
from __future__ import annotations
import json,hashlib,collections,subprocess
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]

def read(relative):
    file=ROOT/relative
    if not file.exists():raise FileNotFoundError(file)
    return json.loads(file.read_text(encoding="utf8"))

def publish():
    folder=Path("outputs/star_perceptual_grouped_sft_v1")
    m=read(folder/"manifest.json")
    selected=read(folder/"chosen_seed.json")
    scores=read(folder/"test_comparison.json")
    if m["version"]!=4 or m["component_count"]!=191:
        raise RuntimeError("EXPECTED_PERCEPTUAL_COMPONENT_COUNT_MISSING")
    content=collections.defaultdict(set)
    byid={r["source_index"]:r for r in m["rows"]}
    for row in m["rows"]:
        content[row["content_sha256"]].add(row["split"])
    if len(m["rows"])!=200 or any(len(s)!=1 for s in content.values()):
        raise RuntimeError("CONTENT_SHA_CROSS_SPLIT_LEAK")
    for a,b in m["near_duplicate_edges"]:
        if byid[a]["split"]!=byid[b]["split"]:
            raise RuntimeError("PERCEPTUAL_EDGE_CROSS_SPLIT_LEAK")
    if selected["chosen_seed"]!=scores["selected_seed"] or selected["test_unseen"] is not True:
        raise RuntimeError("INVALID_MODEL_SELECTION")
    for seed in (19,37):
        if not (ROOT/folder/f"adapter_{seed}"/"adapter_model.safetensors").is_file():
            raise RuntimeError("MISSING_NEW_PERCEPTUAL_ADAPTER")
    sha=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,capture_output=True,check=True,timeout=10).stdout.strip()
    doc={
      "result_scope":"Retrospective strict perceptual candidate grouping",
      "not_newly_blind":True,"time_utc":datetime.now(timezone.utc).isoformat(),
      "frozen_source_sha":sha,
      "manifest_sha256":m["rows_sha256"],"components":m["component_count"],
      "content_groups":len(content),
      "near_candidate_edges":m["near_duplicate_edges"],
      "train_val_test_exact_content_overlap":0,
      "train_val_test_audit_edge_overlap":0,
      "selected_seed_by_val":selected["chosen_seed"],
      "result":scores["stats"],
      "limitations":[
        "STAR video set and some previous test outcomes were already inspected during development.",
        "Strict pHash/dHash candidate edges do not detect all perceptual near-duplicates.",
        "This is not new blind validation. External independent task should be used for transfer."
      ]
    }
    out=ROOT/folder
    (out/"RETROSPECTIVE_AUDIT_RESULT.json").write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n")
    lines=["# STAR perceptual+exact SHA group-disjoint SFT (retrospective)","",
      "Important: the public STAR dataset and previous scores were already analyzed; this is not a new blind benchmark.",
      "No model predictions or QA answers were used for the grouping algorithm.",
      f"Components: {m['component_count']}. Strict perceptual candidate edges: {m['near_duplicate_edges']}.",
      "Train/val/test SHA-identical groups and flagged perceptual candidate edges never cross subsets.",
      f"Validation-selected seed: {selected['chosen_seed']}.",
      "","| Frames | Base | Newly trained LoRA | Delta pp | Paired bootstrap 95% CI pp |",
      "|---:|---:|---:|---:|---|"]
    for n in ("1","3","5"):
        x=scores["stats"][n]
        lines.append(f"| {n} | {x['base']['correct']}/40 | {x['lora']['correct']}/40 | {x['delta_pp']:+.2f} | {x['bootstrap_95_ci_pp']} |")
    lines+=["","These are retrospective integrity-rerun scores, not evidence of general improvement.",
            "No automatic model-performance claims are allowed if a confidence interval contains zero."]
    (out/"RETROSPECTIVE_AUDIT_RESULT.md").write_text("\n".join(lines)+"\n")
    print("RETROSPECTIVE_PERCEPTUAL_GROUP_RESULT",json.dumps(doc["result"]),flush=True)

if __name__=="__main__":
    publish()
