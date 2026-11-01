"""Fixed appearance HSV tracker with optional camera-background registration.

The ONLY inputs are (video path, question text, frozen thresholds).
Ground-truth direction, scene mode, camera velocity and answer options are
unavailable to the tracker. Camera registration estimates stationary-world
BACKGROUND translation using image phase correlation. The background-relative
target delta is screen-target delta minus estimated background-screen shift.

This handles global translation, NOT homography/rotating cameras/3D parallax.
"""
from __future__ import annotations
import cv2,numpy as np
from pathlib import Path
from appearance_hsv_tracker import measured_hsv_trace
from target_motion_tracking import choose_frames,infer_direction

def grayscale_features(path,n=3):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise RuntimeError("CANNOT_OPEN_REGISTRATION_VIDEO")
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total<3:raise RuntimeError("VIDEO_TOO_SHORT")
    indexes=choose_frames(total,n)
    images=[]
    try:
        for index in (indexes[0],indexes[-1]):
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,bgr=cap.read()
            if not ok:raise RuntimeError("REGISTRATION_FRAME_DECODE_FAILURE")
            # Target is small relative to the textured background. Use edge
            # gradients to minimize luminance fluctuations.
            gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)
            images.append(gray)
    finally:cap.release()
    return images,indexes

def estimated_bg_shift(path,min_response=0.12):
    """Phase correlation returns displacement of background in image pixels."""
    (a,b),indices=grayscale_features(path)
    if a.shape!=b.shape:raise RuntimeError("FRAME_SHAPE_CHANGED")
    window=cv2.createHanningWindow((a.shape[1],a.shape[0]),cv2.CV_32F)
    shift,confidence=cv2.phaseCorrelate(a,b,window)
    dx,dy=float(shift[0]),float(shift[1])
    if not all(np.isfinite([dx,dy,confidence])):raise RuntimeError("NONFINITE_REGISTRATION")
    ok=(confidence>=min_response and abs(dx)<=a.shape[1]*.35
        and abs(dy)<=a.shape[0]*.35)
    return {"background_shift":[round(dx,4),round(dy,4)],
            "registration_response":round(float(confidence),5),
            "registration_accepted":bool(ok),
            "frame_indices":[indices[0],indices[-1]],
            "method":"cv2.phaseCorrelate stationary background, full-frame Hanning"}

def tracker(path,question,hsv_cfg,stable=False):
    base=measured_hsv_trace(Path(path),question,hsv_cfg,3)
    output={"camera_compensated":bool(stable),
        "target_color":base["target_color"],
        "target_points":base["points"],
        "screen_delta_xy":base["delta_xy"],
        "reason":base["reason"],
        "missing":base["missing"],"direction":None,
        "delta_xy":base["delta_xy"],
        "registration":None}
    if base["delta_xy"] is None:
        return output
    if stable:
        result=estimated_bg_shift(path,hsv_cfg["min_registration_response"])
        output["registration"]=result
        if not result["registration_accepted"]:
            output["direction"]=None
            output["delta_xy"]=None
            output["reason"]="CAMERA_REGISTRATION_ABSTAIN"
            return output
        corrected=[base["delta_xy"][i]-result["background_shift"][i]
                   for i in (0,1)]
        output["delta_xy"]=[round(x,3) for x in corrected]
    output["direction"]=infer_direction({"delta_xy":output["delta_xy"]},
                                      hsv_cfg["direction_deadzone_px"])
    if output["direction"] is None:
        output["reason"]="AXIAL_OR_INDETERMINATE_MOVEMENT"
    else:
        output["reason"]="ACCEPTED_TRACK"
    return output

def gated_label(track,hsv_cfg):
    """Abstain on weak motion, stationary, ambiguity or missing named option."""
    direction=track.get("direction")
    delta=track.get("delta_xy")
    if direction is None or delta is None:
        return None,"NO_STABLE_DIRECTION"
    if direction=="stationary":
        return None,"STATIONARY_DECLINED"
    if any(abs(value)<hsv_cfg["gate_min_dxdy"] for value in delta):
        return None,"WEAK_DIRECTION"
    return direction,"MOVING_ACCEPTED"
