"""Perceptual near-duplicate AUDIT of SHA256-grouped STAR video splits.

No labels or model predictions are read. Exact SHA groups were already
eliminated across splits. This complementary DCT pHash+dHash audit can
suggest likely near duplicates, but does NOT certify perceptual uniqueness.
Does not alter any training/test split or retroactively filter test examples.
"""
from __future__ import annotations
import argparse,collections,json,math
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def video_fingerprints(path,decoded,positions=(.1,.5,.9)):
    import cv2,numpy as np
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise RuntimeError("Failed video open: "+str(path))
    ph=[];dh=[];thumbs=[]
    try:
        for t in positions:
            idx=max(0,min(decoded-1,round(t*(decoded-1))))
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
            ok,frame=cap.read()
            if not ok:raise RuntimeError(f"Could not decode {path} at idx {idx}")
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            small=cv2.resize(gray,(32,32)).astype(np.float32)
            dct=cv2.dct(small)[:8,:8].flatten()
            mask=dct>float(np.median(dct[1:]))
            p=int("".join("1" if b else "0" for b in mask),2)
            grid=cv2.resize(gray,(9,8))
            pairs=(grid[:,1:] > grid[:,:-1]).flatten()
            d=int("".join("1" if b else "0" for b in pairs),2)
            ph.append(p);dh.append(d)
            thumbs.append(cv2.resize(frame,(180,110)))
    finally:cap.release()
    return ph,dh,thumbs

def hamming(a,b):
    return int(a ^ b).bit_count()

def suspicious(a,b,limit):
    """Strict aligned 10/50/90% 3-frame gate; no labels or QA."""
    if a["content_sha256"]==b["content_sha256"]:
        return None
    p=[hamming(x,y) for x,y in zip(a["p"],b["p"])]
    d=[hamming(x,y) for x,y in zip(a["d"],b["d"])]
    if all(x <= limit["phash"] for x in p) and all(x <= limit["dhash"] for x in d):
        return {"phash_distances":p,"dhash_distances":d}
    return None

def run(config):
    cfg=json.loads((ROOT/config).read_text(encoding="utf8"))
    manifest=json.loads((ROOT/cfg["manifest"]).read_text(encoding="utf8"))
    if manifest["unique_content_hashes"]!=193 or len(manifest["rows"])!=200:
        raise RuntimeError("GROUPED_CONTENT_AUDIT_SOURCE_CHANGED")
    if manifest["train_val_test_sha256_overlap_count"]!=0:
        raise RuntimeError("EXACT_SHA_LEAK_IN_SOURCE")
    limits={"phash":int(cfg["max_phash_distance"]),
            "dhash":int(cfg["max_dhash_distance"])}
    if limits["phash"]>10 or limits["dhash"]>10:
        raise RuntimeError("Threshold would be too broad for a strict audit")
    fingerprints=[]
    thumb_map={}
    for i,row in enumerate(manifest["rows"]):
        path=ROOT/row["video"]
        ph,dh,thumbs=video_fingerprints(path,int(row["decoded_frames"]))
        record={"id":row["source_index"],"split":row["split"],"video":row["video"],
                "content_sha256":row["content_sha256"],"p":ph,"d":dh}
        fingerprints.append(record)
        thumb_map[row["source_index"]]=thumbs
        if (i+1)%40==0:print("NEAR_DUP_FINGERPRINTED",i+1,"/",len(manifest["rows"]),flush=True)
    cross=[];within=[];pairs_scanned=0
    for i,a in enumerate(fingerprints):
        for b in fingerprints[i+1:]:
            pairs_scanned+=1
            condition=suspicious(a,b,limits)
            if condition is None:continue
            match={"left_id":a["id"],"right_id":b["id"],
                   "left_split":a["split"],"right_split":b["split"],
                   "left_video":a["video"],"right_video":b["video"],
                   **condition}
            if a["split"]!=b["split"]:cross.append(match)
            else:within.append(match)
    directory=ROOT/cfg["output_dir"];directory.mkdir(parents=True,exist_ok=True)
    result={"timestamp_utc":datetime.now(timezone.utc).isoformat(),
            "algorithm":"aligned 10%,50%,90% video DCT pHash 64bit and dHash 64bit",
            "thresholds":limits,"source_manifest_hash":manifest["rows_sha256"],
            "sampled_videos":len(fingerprints),"pairs_compared":pairs_scanned,
            "cross_split_suspicious":len(cross),"within_split_suspicious":len(within),
            "cross_split_candidates":cross,"within_split_candidates":within,
            "conclusion":"CANDIDATES ONLY, NOT proof of duplicates or proof of absence",
            "policy":"No labels, answer traces or performance are used; no re-split or test sample removed."}
    (directory/"NEAR_DUP_AUDIT.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    lines=["# STAR perceptual near-duplicate audit (no dataset changes)","",
         "Source: frozen SHA256-disjoint 200-video STAR action_sequence manifest.",
         "Exactly identical video byte content is already grouped; this check flags potentially similar MP4s even when encoded bytes differ.",
         "",
         f"Frame fingerprint: 10/50/90% aligned positions, 64-bit DCT pHash and dHash; max distances pHash {limits['phash']}, dHash {limits['dhash']} across ALL three aligned frames.",
         f"Pairs compared: {pairs_scanned}; potential cross-split pairs: {len(cross)}; potential within-split pairs: {len(within)}.",
         "",
         "## Potential cross-split pairs (requires human confirmation)",
         "",
         "| Video A | Split | Video B | Split | DCT distances | dHash distances |",
         "|---|---|---|---|---|---|"]
    for x in cross:
        lines.append(f"| {x['left_id']} | {x['left_split']} | {x['right_id']} | {x['right_split']} | {x['phash_distances']} | {x['dhash_distances']} |")
    lines+=["","An empty table does not imply guaranteed independence: this method cannot find all cropped, retimed, or semantically similar scenes.",
            "**Do not change the existing test split after seeing results.** Publish the audit as a limitation; any new grouping requires a fresh frozen experiment."]
    (directory/"NEAR_DUP_AUDIT.md").write_text("\n".join(lines)+"\n")
    # Human inspection contact sheet for up to 12 cross-split flagged pairs.
    if cross:
        import cv2,numpy as np
        shown=cross[:12]
        blocks=[]
        for x in shown:
            left=thumb_map[x["left_id"]][1]
            right=thumb_map[x["right_id"]][1]
            pair=np.concatenate((left,right),axis=1)
            label=f"{x['left_id']}:{x['left_split']} vs {x['right_id']}:{x['right_split']}"
            cv2.putText(pair,label,(5,15),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
            blocks.append(pair)
        canvas=np.concatenate(blocks,axis=0)
        cv2.imwrite(str(directory/"NEAR_DUP_CANDIDATES.jpg"),canvas)
    print("PERCEPTUAL_AUDIT_COMPLETE",json.dumps({
        "pairs":pairs_scanned,"cross_split_candidates":len(cross),
        "within_split_candidates":len(within),"thresholds":limits}),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/perceptual_near_dup_v1.json")
    arg=ap.parse_args();run(arg.config)
