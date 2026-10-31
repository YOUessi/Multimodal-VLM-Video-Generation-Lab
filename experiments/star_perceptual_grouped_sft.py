"""Fresh STAR LoRA with exact SHA and strict perceptual-candidate CONNECTED split groups.

This is a retrospective data-audit remedy. It is not a newly blind
external test: public STAR samples and historic outcomes were inspected
during development. No model predictions or labels drive connected edges.
"""
from __future__ import annotations
import argparse,hashlib,json,random
from collections import defaultdict,Counter
from pathlib import Path
from star_content_grouped_sft import assign_content_groups,sha256_bytes
from star_real_sft import extract_frames,train,validate,test
ROOT=Path(__file__).resolve().parents[1]

def sha_file(path):
    return sha256_bytes(path)

def linked_components(rows,edges):
    parent={row["source_index"]:row["source_index"] for row in rows}
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    def union(a,b):
        x,y=find(a),find(b)
        if x!=y:parent[max(x,y)]=min(x,y)
    by_sha={}
    for r in rows:
        h=r["content_sha256"]
        if h in by_sha:union(by_sha[h],r["source_index"])
        else:by_sha[h]=r["source_index"]
    for a,b in edges:
        if a not in parent or b not in parent:raise ValueError("Unknown near-duplicate ID")
        union(a,b)
    components=defaultdict(list)
    for idx in parent:
        components[find(idx)].append(idx)
    return {str(k):sorted(v) for k,v in components.items()}

def manifest(cfg):
    original=json.loads((ROOT/cfg["sha_group_manifest"]).read_text(encoding="utf8"))
    audit_file=ROOT/cfg["perceptual_audit"]
    audit=json.loads(audit_file.read_text(encoding="utf8"))
    if original["rows_sha256"]!=audit["source_manifest_hash"]:
        raise RuntimeError("PERCEPTUAL_AUDIT_SOURCE_MISMATCH")
    if original["unique_content_hashes"]!=193 or len(original["rows"])!=200:
        raise RuntimeError("STAR_SOURCE_GROUP_CHANGED")
    if (audit["thresholds"]["phash"]!=cfg["max_phash_distance"] or
        audit["thresholds"]["dhash"]!=cfg["max_dhash_distance"]):
        raise RuntimeError("PERCEPTUAL_THRESHOLDS_CHANGED")
    edge_records=audit["cross_split_candidates"]+audit["within_split_candidates"]
    edges=[(int(r["left_id"]),int(r["right_id"])) for r in edge_records]
    if len(edges)!=2:
        raise RuntimeError("NEAR_DUP_AUDIT_EDGE_COUNT_CHANGED")
    components=linked_components(original["rows"],edges)
    if len(components)!=191:raise RuntimeError("UNEXPECTED_CONNECTED_COMPONENT_COUNT")
    choices=assign_content_groups(components,cfg["split_sizes"],cfg["split_seed"])
    rows=[]
    original_by_id={r["source_index"]:r for r in original["rows"]}
    for split in ("train","val","test"):
        for idx in sorted(i for i,s in choices.items() if s==split):
            row=dict(original_by_id[idx])
            row["split"]=split
            rows.append(row)
    groups=defaultdict(set)
    for row in rows:
        digest=sha_file(ROOT/row["video"])
        if digest!=row["content_sha256"]:raise RuntimeError("VIDEO_BYTES_CHANGED")
        groups[digest].add(row["split"])
    if any(len(s)!=1 for s in groups.values()):
        raise RuntimeError("EXACT_SHA_CROSS_SPLIT_LEAK")
    for a,b in edges:
        if choices[a]!=choices[b]:
            raise RuntimeError("PERCEPTUAL_CANDIDATE_CROSS_SPLIT_LEAK")
    digest=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    audit_digest=sha_file(audit_file)
    out=ROOT/cfg["output_dir"]/"manifest.json"
    frozen={"version":4,"rows_sha256":digest,
        "source_sha_group_manifest":original["rows_sha256"],
        "perceptual_audit_sha256":audit_digest,
        "near_duplicate_edges":edges,
        "grouping":"connected components of exact video SHA256 + fixed strict pHash/dHash candidates",
        "component_count":len(components),
        "unique_content_hashes":len(groups),
        "split_sizes":cfg["split_sizes"],
        "split_seed":cfg["split_seed"],"rows":rows}
    if out.exists():
        current=json.loads(out.read_text(encoding="utf8"))
        if current!=frozen:raise RuntimeError("FROZEN_CONNECTED_MANIFEST_DRIFT")
    else:
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(frozen,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    return digest,rows,edges

def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("stage",choices=["prepare","train","validate","test"])
    cli.add_argument("--config",default="configs/star_perceptual_grouped_v1.json")
    args=cli.parse_args()
    cfg=json.loads((ROOT/args.config).read_text())
    digest,rows,edges=manifest(cfg)
    if args.stage=="prepare":
        for row in rows:
            frames=extract_frames(ROOT/row["video"],cfg["training_frames"],row["decoded_frames"])
            if len(frames)!=cfg["training_frames"]:raise RuntimeError("FRAME_READ_FAILED")
        print("STAR_PERCEPTUAL_GROUPED_PREFLIGHT",digest,
              "rows",len(rows),"near_candidate_edges",edges,"cross_split_edges",0,flush=True)
    elif args.stage=="train":
        for seed in cfg["training_seeds"]:train(cfg,rows,seed)
    elif args.stage=="validate":validate(cfg,rows)
    else:test(cfg,rows)

if __name__=="__main__":main()
