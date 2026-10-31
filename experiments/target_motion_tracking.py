"""Question-grounded, label-blind object tracking and temporal evidence.

A deterministic color segmentation tracker identifies the object mentioned
in a question, samples three chronological frames, extracts object centroids
from actual RGB pixels, and derives displacement. It never reads ground
truth, generated scene_specs or answer choices. Valid only for this controlled
high-chroma synthetic domain, not a generic real-video object tracker.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import cv2
import numpy as np
from object_motion_counterfactual_data import COLORS, DIRECTIONS, LABELS

ROOT=Path(__file__).resolve().parents[1]

def choose_frames(decoded,n=3):
    if n not in (3,5):raise ValueError("Only 3/5 chronological samples supported")
    if decoded<n:raise RuntimeError("TOO_FEW_DECODED_FRAMES")
    lo=round(.05*(decoded-1));hi=round(.95*(decoded-1))
    return [lo+round((hi-lo)*i/(n-1)) for i in range(n)]

def question_color(question):
    matches=[c for c in COLORS if re.search(r"\b"+re.escape(c)+r"\b",question,flags=re.I)]
    if len(matches)!=1:raise RuntimeError("AMBIGUOUS_TARGET_COLOR_QUERY:"+str(matches))
    return matches[0]

def center_from_rgb(image,color,min_pixels=60):
    """Largest connected near-palette-color component from decoded pixels."""
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    target=np.array(COLORS[color],dtype=np.int16)
    diff=rgb.astype(np.int16)-target[None,None,:]
    score=np.sum(diff.astype(np.int32)**2,axis=2)
    mask=(score<45**2).astype(np.uint8)
    num,ids,stats,centroids=cv2.connectedComponentsWithStats(mask,8)
    if num<=1:return None
    order=sorted(range(1,num),key=lambda k:-stats[k,cv2.CC_STAT_AREA])
    winner=order[0]
    area=int(stats[winner,cv2.CC_STAT_AREA])
    if area<min_pixels:return None
    x,y=centroids[winner]
    return [round(float(x),3),round(float(y),3)]

def measured_trace(video_path,question,frame_count=3):
    path=Path(video_path)
    color=question_color(question)
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise RuntimeError("VIDEO_READER_FAILED:"+str(path))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Generated fixtures must decode exactly as written. Do not use GT labels.
    indexes=choose_frames(total,frame_count)
    points=[]
    frames=[]
    try:
        for index in indexes:
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,image=cap.read()
            if not ok:raise RuntimeError("OBJECT_FRAME_DECODE_FAILED:"+str(index))
            points.append(center_from_rgb(image,color))
            frames.append(image)
    finally:cap.release()
    missing=sum(p is None for p in points)
    vector=None
    if missing==0:
        vector=[round(points[-1][0]-points[0][0],3),
                round(points[-1][1]-points[0][1],3)]
    return {"target_color":color,"points":points,"indexes":indexes,
            "missing_points":missing,"delta_xy":vector,
            "image_size":[int(frames[0].shape[1]),int(frames[0].shape[0])],
            "method":"Largest connected component nearest fixed RGB target color, source pixels only"}

def infer_direction(trace,threshold_px=9):
    delta=trace.get("delta_xy")
    if delta is None:return None
    dx,dy=delta
    if abs(dx)<threshold_px and abs(dy)<threshold_px:
        return "stationary"
    if abs(dx)<threshold_px or abs(dy)<threshold_px:
        return None
    return ("up" if dy<0 else "down")+"_"+("right" if dx>0 else "left")

def numeric_evidence(trace,reverse=False):
    points=list(reversed(trace["points"])) if reverse else trace["points"]
    if any(p is None for p in points):
        return "The color tracker could not identify a reliable target in every sampled frame; no position estimate is available."
    entries=[]
    for i,(x,y) in enumerate(points):
        entries.append("sample "+str(i+1)+": x="+f"{x:.1f}"+", y="+f"{y:.1f}")
    return ("Computed target-object centroid positions at equal chronological sample intervals, "
            "from observed pixels only (x grows right, y grows downward): "+
            "; ".join(entries)+". The coordinates are observations, not ground-truth answer labels.")

def video_rgb_frames(video,n=3):
    from PIL import Image
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise RuntimeError("IMAGE_VIDEO_OPEN_FAILED")
    decoded=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    positions=choose_frames(decoded,n)
    images=[]
    try:
        for i in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES,i)
            ok,bgr=cap.read()
            if not ok:raise RuntimeError("RGB_FRAME_DECODE_FAILED")
            images.append(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
    finally:cap.release()
    return images

def assert_blind_tracking_interface():
    """No annotations, gold labels or metadata enter the tracker API."""
    import inspect
    args=list(inspect.signature(measured_trace).parameters)
    assert args==["video_path","question","frame_count"]

def run_preflight(config,dev_only=True):
    cfg=json.loads((ROOT/config).read_text(encoding="utf8"))
    manifest=json.loads((ROOT/cfg["data_dir"]/"manifest.json").read_text(encoding="utf8"))
    if manifest["config_version"]!=cfg["version"]:raise RuntimeError("MOTION_MANIFEST_VERSION_DRIFT")
    rows=[r for r in manifest["rows"] if r["split"]=="dev"] if dev_only else manifest["rows"]
    missing=0;wrong=[];stats=[]
    for row in rows:
        trace=measured_trace(ROOT/row["video"],row["question"],3)
        guess=infer_direction(trace,cfg["tracker_threshold_px"])
        # Ground truth is allowed ONLY here in evaluation, not in the tracker.
        if guess!=row["gold_direction"]:
            wrong.append((row["id"],guess,row["gold_direction"],trace["delta_xy"]))
        if trace["missing_points"]:missing+=1
        stats.append({"id":row["id"],"guess":guess,
                      "expected":row["gold_direction"],"missing":trace["missing_points"]})
    print("MOTION_TRACKER_DEV_OR_ALL_PREFLIGHT",len(rows),
          "CORRECT",len(rows)-len(wrong),"MISSING",missing,
          "ERROR_SAMPLE",wrong[:8],flush=True)
    if wrong:raise RuntimeError("TRACKER_DEV_SANITY_FAILURE: "+repr(wrong[:4]))
    return stats

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/object_motion_v1.json")
    p.add_argument("--verify-test",action="store_true")
    args=p.parse_args()
    run_preflight(args.config,dev_only=not args.verify_test)
