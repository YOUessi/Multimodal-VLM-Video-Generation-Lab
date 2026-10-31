"""Content-SHA256 GROUP-DISJOINT STAR real-video SFT.

Remediates 2026-10-31 leakage audit: 200 distinct filenames but 193 distinct
byte contents; 5 SHA-identical groups previously crossed train/val/test.
Builds a deterministic split by full video bytes, then calls the existing
model training/validation/test code with entirely NEW adapters and outputs.
"""
from __future__ import annotations
import argparse,hashlib,json,random
from collections import defaultdict,Counter
from pathlib import Path
from star_real_sft import decoded_frame_count,extract_frames,train,validate,test

ROOT=Path(__file__).resolve().parents[1]
SPLITS=("train","val","test")
def sha256_bytes(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(2**20),b""):h.update(block)
    return h.hexdigest()

def assign_content_groups(groups,capacities,seed):
    """Deterministic grouped allocation, largest first; labels NEVER inspected."""
    cap={s:int(capacities[s]) for s in SPLITS}
    assert all(v>0 for v in cap.values())
    assert sum(len(x) for x in groups.values())==sum(cap.values())
    rng=random.Random(seed)
    items=list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda x:-len(x[1]))  # stable seeded tie order
    remaining=cap.copy()
    allocation={}
    for digest,members in items:
        options=[s for s in SPLITS if remaining[s]>=len(members)]
        if not options:raise RuntimeError("SPLIT_GROUP_CAPACITY_EXHAUSTED")
        # Balance relative remaining capacities, deterministic split-name tie.
        selected=max(options,key=lambda s:remaining[s]/cap[s])
        for idx in members:
            if idx in allocation:raise RuntimeError("DUPLICATE_SOURCE_ROW")
            allocation[idx]=selected
        remaining[selected]-=len(members)
    if any(remaining.values()):raise RuntimeError("SPLIT_SIZE_MISMATCH:"+str(remaining))
    assert len(allocation)==sum(cap.values())
    return allocation

def manifest(cfg):
    data=ROOT/cfg["data_dir"]
    annotation_file=data/"text/action_sequence.json"
    if sha256_bytes(annotation_file)!=cfg["source_annotations_sha256"]:
        raise RuntimeError("ANNOTATION_HASH_DRIFT")
    annotations=json.loads(annotation_file.read_text(encoding="utf8"))
    provenance=json.loads((data/"provenance.json").read_text(encoding="utf8"))
    assert len(annotations)==200 and len({x["video"] for x in annotations})==200
    assert provenance["archive_sha256"]==cfg["star_archive_sha256"]
    assert provenance["exact_video_count"]==200
    groups=defaultdict(list);hashed={}
    for idx,entry in enumerate(annotations):
        f=data/"clips"/entry["video"]
        if not f.is_file():raise FileNotFoundError(f)
        digest=sha256_bytes(f)
        groups[digest].append(idx)
        hashed[idx]=digest
    assert len(groups)==193, "EXPECTED_193_CONTENT_GROUPS_CHANGED: "+str(len(groups))
    assignment=assign_content_groups(groups,cfg["split_sizes"],cfg["split_seed"])
    if any(len({assignment[idx] for idx in group})!=1 for group in groups.values()):
        raise RuntimeError("CROSS_SPLIT_SHA256_LEAK")
    target=ROOT/cfg["output_dir"]/"manifest.json"
    # Inspect the frozen manifest on all follow-up stages. Never reallocate
    # a split or use test correctness to alter assignment.
    if target.is_file():
        frozen=json.loads(target.read_text(encoding="utf8"))
        if (frozen.get("format_version")!=3 or
            frozen.get("source_annotations_sha256")!=cfg["source_annotations_sha256"] or
            frozen.get("source_video_archive_sha256")!=cfg["star_archive_sha256"] or
            frozen.get("split_sizes")!=cfg["split_sizes"]):
            raise RuntimeError("GROUPED_MANIFEST_CONFIG_DRIFT")
        saved=frozen["rows"]
        assert len(saved)==200
        if frozen["rows_sha256"]!=hashlib.sha256(json.dumps(saved,sort_keys=True).encode()).hexdigest():
            raise RuntimeError("GROUPED_MANIFEST_SHA_DRIFT")
        for r in saved:
            i=r["source_index"]
            if r["split"]!=assignment[i] or r["content_sha256"]!=hashed[i]:
                raise RuntimeError("GROUPED_MANIFEST_CONTENT_SPLIT_DRIFT")
            if not (ROOT/r["video"]).is_file() or int(r["decoded_frames"])<3:
                raise RuntimeError("FROZEN_SOURCE_MISSING_OR_INVALID")
        return frozen["rows_sha256"],saved
    rows=[]
    for split in SPLITS:
        for idx in sorted(k for k,v in assignment.items() if v==split):
            entry=annotations[idx]
            if entry["answer"] not in entry["candidates"] or len(entry["candidates"])!=4:
                raise RuntimeError("ANNOTATION_INVALID")
            file=data/"clips"/entry["video"]
            count=decoded_frame_count(file)
            row={"source_index":idx,"split":split,"video":str(file.relative_to(ROOT)),
                 "content_sha256":hashed[idx],"decoded_frames":count,
                 "question":entry["question"],"choices":entry["candidates"],
                 "answer":"ABCD"[entry["candidates"].index(entry["answer"])],
                 "original_start":entry.get("start"),"original_end":entry.get("end")}
            rows.append(row)
    assert len(rows)==200
    digest=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    duplicate_groups={sha:v for sha,v in groups.items() if len(v)>1}
    document={"format_version":3,"rows_sha256":digest,
         "source":"PKU-Alignment/MVBench action_sequence",
         "source_annotations_sha256":cfg["source_annotations_sha256"],
         "source_video_archive_sha256":cfg["star_archive_sha256"],
         "source_revision":cfg["source_revision"],
         "split_seed":cfg["split_seed"],"split_sizes":cfg["split_sizes"],
         "grouping":"FULL_SOURCE_VIDEO_SHA256_GROUP_DISJOINT",
         "unique_content_hashes":len(groups),
         "content_duplicate_groups":len(duplicate_groups),
         "train_val_test_sha256_overlap_count":0,
         "rows":rows}
    target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(document,ensure_ascii=False,indent=2)+"\n")
    temp.replace(target)
    print("GROUPED_MANIFEST_CREATED",digest,"SHA_CONTENT_GROUPS",len(groups),
          "duplicate_groups",len(duplicate_groups),flush=True)
    return digest,rows

def preflight(cfg,rows,digest):
    totals=Counter(x["split"] for x in rows)
    assert totals==Counter(cfg["split_sizes"])
    groups=defaultdict(set)
    for row in rows:
        groups[row["content_sha256"]].add(row["split"])
    assert all(len(v)==1 for v in groups.values()),"CROSS_SPLIT_LEAK"
    for i,row in enumerate(rows):
        imgs=extract_frames(ROOT/row["video"],cfg["training_frames"],row["decoded_frames"])
        if len(imgs)!=cfg["training_frames"]:raise RuntimeError("IMAGE_PREP_FAIL")
        if (i+1)%40==0:print("CONTENT_GROUPED_DECODE_VERIFIED",i+1,"/200",flush=True)
    print("CONTENT_GROUPED_ZERO_OVERLAP_CONFIRMED",
          len(groups),"unique_sha_groups",digest,dict(totals),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["prepare","train","validate","test"])
    p.add_argument("--config",default="configs/star_grouped_sft_v1.json")
    a=p.parse_args()
    cfg=json.loads((ROOT/a.config).read_text())
    digest,rows=manifest(cfg)
    if a.stage=="prepare":preflight(cfg,rows,digest)
    elif a.stage=="train":
        for seed in cfg["training_seeds"]:train(cfg,rows,seed)
    elif a.stage=="validate":validate(cfg,rows)
    else:test(cfg,rows)
if __name__=="__main__":main()
