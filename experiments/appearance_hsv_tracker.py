"""Query-grounded appearance-robust HSV color component tracker.

Only generic named-color HSV templates and source pixels are used. It does
NOT train or tune on MVBench/CLEVRER labels, evaluate task correctness,
read generation scene_specs or use a pretrained object detector.

This heuristic is expected to abstain under complicated lighting,
occlusion, ambiguous target colors or camera motion. Such abstentions must
remain counted over the full external benchmark denominator.
"""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import cv2,numpy as np
from object_motion_counterfactual_data import COLORS as SYNTHETIC_RGB
from target_motion_tracking import choose_frames,infer_direction

ROOT=Path(__file__).resolve().parents[1]
EXTRA_RGB={"brown":(133,84,45),"gray":(135,135,135),
           "yellow":(225,196,50)}
PALETTE={**SYNTHETIC_RGB,**EXTRA_RGB}
ALIASES={"grey":"gray"}
def query_target(question):
    q=question.lower()
    found=[color for color in PALETTE if re.search(r"\b"+color+r"\b",q)]
    if "grey" in q:found.append("gray")
    found=list(dict.fromkeys(found))
    if len(found)!=1:raise RuntimeError("AMBIGUOUS_OR_UNSUPPORTED_COLOR_NAME")
    return found[0]

def template_hsv(color):
    rgb=PALETTE[color]
    patch=np.array([[tuple(reversed(rgb))]],dtype=np.uint8)
    return cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)[0,0]

def threshold_mask(frame,color,config):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    h,s,v=[hsv[:,:,i].astype(np.int16) for i in range(3)]
    ref=template_hsv(color)
    if color=="gray":
        mask=(s<=config["gray_saturation_max"]) & (v>=config["min_value"]) & (v<=config["gray_value_max"])
    else:
        hue_error=np.abs(h-int(ref[0]))
        hue_error=np.minimum(hue_error,180-hue_error)
        mask=(hue_error<=config["hue_tolerance"]) & (s>=config["min_saturation"]) & (v>=config["min_value"])
        if color=="brown":
            mask=mask & (v<=config["brown_value_max"])
    binary=mask.astype(np.uint8)
    # Remove isolated chromatic background noise, preserving large objects.
    kernel=np.ones((3,3),np.uint8)
    return cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)

def candidates(frame,color,config):
    h,w=frame.shape[:2]
    binary=threshold_mask(frame,color,config)
    n,l,stats,centers=cv2.connectedComponentsWithStats(binary,8)
    low=max(config["min_component_pixels"],int(w*h*config["min_component_fraction"]))
    high=min(config["max_component_pixels"],int(w*h*config["max_component_fraction"]))
    found=[]
    for i in range(1,n):
        area=int(stats[i,cv2.CC_STAT_AREA])
        x,y=centers[i]
        if (area<low or area>high or
            not config["roi_min_x"]*w<=x<=config["roi_max_x"]*w or
            not config["roi_min_y"]*h<=y<=config["roi_max_y"]*h):
            continue
        found.append({"xy":[round(float(x),3),round(float(y),3)],"area":area})
    found.sort(key=lambda x:(-x["area"],x["xy"][0]))
    return found

def select_component(options,previous,config):
    if not options:return None,"NO_COLOR_COMPONENT"
    if len(options)>config["max_candidate_count"]:
        return None,"AMBIGUOUS_MANY_COLOR_COMPONENTS"
    if previous is None:
        if len(options)>=2 and options[0]["area"]<=options[1]["area"]*config["first_frame_dominance_ratio"]:
            return None,"AMBIGUOUS_FIRST_FRAME_COMPONENTS"
        return options[0],"OK"
    choices=sorted(options,key=lambda z:((z["xy"][0]-previous[0])**2+(z["xy"][1]-previous[1])**2))
    best=choices[0]
    distance=((best["xy"][0]-previous[0])**2+(best["xy"][1]-previous[1])**2)**.5
    if distance>config["max_step_displacement_px"]:
        return None,"UNSTABLE_OBJECT_ASSOCIATION"
    if len(choices)>=2:
        other=((choices[1]["xy"][0]-previous[0])**2+(choices[1]["xy"][1]-previous[1])**2)**.5
        if distance+config["association_margin_px"]>=other:
            return None,"AMBIGUOUS_ASSOCIATION"
    return best,"OK"

def measured_hsv_trace(video,question,config,n=3):
    color=query_target(question)
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise RuntimeError("CANNOT_OPEN_VIDEO")
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    chosen=choose_frames(total,n)
    obs=[];reasons=[]
    previous=None
    try:
        for index in chosen:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(index))
            good,bgr=cap.read()
            if not good:raise RuntimeError("VIDEO_FRAME_UNREADABLE")
            comps=candidates(bgr,color,config)
            current,reason=select_component(comps,previous,config)
            reasons.append(reason)
            obs.append(current)
            if current is not None:previous=current["xy"]
    finally:cap.release()
    if any(x is None for x in obs):
        return {"target_color":color,"frame_indices":chosen,
            "points":[x["xy"] if x is not None else None for x in obs],
            "delta_xy":None,"missing":sum(x is None for x in obs),
            "direction":None,"reason":";".join(reasons),
            "method":"Generic named HSV template and component tracking"}
    points=[x["xy"] for x in obs]
    delta=[round(points[-1][0]-points[0][0],3),
           round(points[-1][1]-points[0][1],3)]
    estimate=infer_direction({"delta_xy":delta},config["direction_deadzone_px"])
    return {"target_color":color,"frame_indices":chosen,"points":points,
        "delta_xy":delta,"missing":0,"direction":estimate,
        "reason":"OK" if estimate else "AXIAL_OR_AMBIGUOUS_MOTION",
        "method":"Generic named HSV template and component tracking"}

def full_dev_sanity(cfg):
    source=json.loads((ROOT/cfg["development_dataset"]/"manifest.json").read_text())
    dev=[x for x in source["rows"] if x["split"]=="dev"]
    if len(dev)!=40:raise RuntimeError("DEVELOPMENT_SOURCE_CHANGED")
    counts={"matched":0,"abstained":0,"wrong":0}
    for row in dev:
        try:
            track=measured_hsv_trace(ROOT/row["video"],row["question"],cfg,3)
            direction=track["direction"]
        except Exception:
            direction=None
        if direction is None:counts["abstained"]+=1
        elif direction==row["gold_direction"]:counts["matched"]+=1
        else:counts["wrong"]+=1
    print("HSV_TRACKER_DEV_ONLY_SUMMARY",counts,flush=True)
    # Failing to solve dev is a measured limitation, not reason to tune this
    # already-frozen heuristic to external test labels.
    return counts

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/appearance_hsv_transfer_v1.json")
    args=p.parse_args()
    cfg=json.loads((ROOT/args.config).read_text())
    full_dev_sanity(cfg)
