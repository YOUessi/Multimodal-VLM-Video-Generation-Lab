"""Authoritative STAR content-SHA256-disjoint LoRA evaluation report.

Old video-ID-only STAR split contains cross-split duplicate video bytes.
It must not be used as independent heldout evidence.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, subprocess
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read_json(relative):
    p=ROOT/relative
    if not p.is_file():raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))

def content_sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(2**20),b""):h.update(chunk)
    return h.hexdigest()

def evaluate(base):
    manifest=read_json(base/"manifest.json")
    chosen=read_json(base/"chosen_seed.json")
    results=read_json(base/"test_comparison.json")
    groups=collections.defaultdict(set)
    group_members=collections.defaultdict(list)
    for row in manifest["rows"]:
        actual=content_sha256(ROOT/row["video"])
        if actual!=row["content_sha256"]:
            raise RuntimeError("SOURCE_VIDEO_SHA256_DRIFT")
        groups[actual].add(row["split"])
        group_members[actual].append((row["source_index"],row["split"]))
    if any(len(s)!=1 for s in groups.values()):
        raise RuntimeError("CROSS_SPLIT_DUPLICATE_VIDEO_BYTES")
    if len(groups)!=193 or manifest.get("unique_content_hashes")!=193:
        raise RuntimeError("UNEXPECTED_CONTENT_GROUP_COUNT")
    if manifest.get("train_val_test_sha256_overlap_count")!=0:
        raise RuntimeError("CONTENT_DISJOINT_CONTRACT_MISSING")
    count=collections.Counter(row["split"] for row in manifest["rows"])
    if count!={"train":120,"val":40,"test":40}:
        raise RuntimeError("DATA_SPLIT_COUNT_DRIFT")
    if chosen["chosen_seed"]!=results["selected_seed"] or chosen["test_unseen"] is not True:
        raise RuntimeError("MODEL_SELECTION_NOT_VALIDATION_ONLY")
    for seed in (19,37):
        if not (ROOT/base/("adapter_"+str(seed))/"adapter_model.safetensors").is_file():
            raise RuntimeError("MISSING_FRESH_ADAPTER")
        if not (ROOT/base/("train_"+str(seed)+".json")).is_file():
            raise RuntimeError("MISSING_FRESH_TRAIN_LOG")
    for name in ("test_base.json","test_lora.json"):
        evaluation=read_json(base/name)
        if evaluation["split"]!="test" or len(evaluation["results"])!=120:
            raise RuntimeError("INCOMPLETE_TEST_RUN")
    sha=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,check=True,
                       text=True,capture_output=True,timeout=10).stdout.strip()
    provenance={
        "report_version":3,"frozen_source_sha":sha,
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "frozen_manifest_sha256":manifest["rows_sha256"],
        "total_rows":200,"unique_content_hashes":len(groups),
        "duplicate_content_groups":sum(len(v)>1 for v in group_members.values()),
        "cross_split_exact_content_overlap":0,"split_counts":dict(count),
        "validation_selected_seed":chosen["chosen_seed"],
        "previous_round2":"INVALID: 5 duplicate groups crossed old train/val/test splits",
        "new_results":results["stats"],
        "limitations":["40 sample public heldout is small",
                       "near-duplicates not excluded by exact SHA",
                       "foundation pretraining contamination cannot be excluded"]
    }
    lines=["# STAR LoRA evaluation after SHA256 content-leakage remediation","",
           "**Previous video-ID-only STAR scores cannot be used as independent heldout evidence.**",
           "Seven exact-duplicate byte-content groups existed; five crossed old split boundaries.",
           "All identical-content videos are now assigned to the same train/val/test split.",
           "",
           "Frozen manifest SHA256: "+manifest["rows_sha256"],
           f"Unique SHA256 content groups: {len(groups)}. Cross-split duplicate groups: 0.",
           "Train / validation / test: 120 / 40 / 40 videos.",
           "Fresh LoRA seeds 19 and 37; chosen by validation only: "+str(chosen["chosen_seed"])+".",
           "","| Frames | Base | Group-disjoint LoRA | Difference (pp) | Paired bootstrap 95% CI pp |",
           "|---:|---:|---:|---:|---|"]
    for n in ("1","3","5"):
        r=results["stats"][n]
        lines.append(f"| {n} | {r['base']['correct']}/40 | {r['lora']['correct']}/40 | {r['delta_pp']:+.2f} | {r['bootstrap_95_ci_pp']} |")
    lines+=["","## Limitations",
            "- Public STAR benchmark may appear in pretraining data.",
            "- Exact-file SHA grouping does not remove perceptual near-duplicates.",
            "- Unmodified 40-video test is only a small single-task sample.",
            "- No model selection on test labels; do not overstate small gains.",
            "- All original videos, weights, trained adapters remain on Tang."]
    (ROOT/base/"AUTHORITATIVE_RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (ROOT/base/"AUTHORITATIVE_RESULTS.json").write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+"\n")
    print("AUTHORITATIVE_GROUPED_RESULTS",json.dumps({
        "splits":dict(count),"unique_hashes":len(groups),
        "cross_split_overlap":0,"selected_seed":chosen["chosen_seed"],"metrics":results["stats"]},ensure_ascii=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-dir",default="outputs/star_grouped_sft_v1")
    args=ap.parse_args()
    evaluate(Path(args.run_dir))
