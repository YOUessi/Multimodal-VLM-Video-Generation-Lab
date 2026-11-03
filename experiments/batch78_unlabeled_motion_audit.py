"""Batch78: DEV-only, label-free audit of image-plane motion and scene-flow proxies.

A dominant 2-D optical-flow translation is NOT true camera motion, depth,
background motion, or physical target identity. No oracle is ever opened.
Predictor comparisons are derived only from pinned public DEV tracks.
This exploratory audit must NOT select a model on TEST covariates/labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
PROTOCOL = "batch78-dev-unlabeled-dominant-image-flow-v1"
PAIR_STRIDE = 10
MAX_SIDE = 256
MAX_CORNERS = 320
MIN_TRACKS = 25
MIN_CELLS = 6
MIN_CONSENSUS = 0.60
ROUNDTRIP_TOL = 1.5
BASE_METHOD = "appearance_fb_055"


def sha256(buf: bytes) -> str:
    return hashlib.sha256(buf).hexdigest()


def finite_median(values: list[float]) -> float | None:
    values = [float(v) for v in values if math.isfinite(float(v))]
    return float(median(values)) if values else None


def write_once(path: Path, payload: dict) -> str:
    raw = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    pin = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or pin.exists():
        raise RuntimeError("BATCH78_IMMUTABLE_OUTPUT_EXISTS")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(raw)
    with pin.open("x", encoding="ascii") as f:
        f.write(sha256(raw) + "\n")
    return sha256(raw)


def read_pinned_json(path: Path) -> dict:
    if not path.is_file() or not path.with_suffix(path.suffix + ".sha256").is_file():
        raise RuntimeError("BATCH78_MISSING_PIN")
    raw = path.read_bytes()
    if sha256(raw) != path.with_suffix(path.suffix + ".sha256").read_text().strip():
        raise RuntimeError("BATCH78_PIN_DRIFT")
    return json.loads(raw)


def safe_frame_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name.endswith(".npy") or ".." in Path(name).parts:
        raise RuntimeError("BATCH78_UNSAFE_FRAME_PATH")
    resolved_root = root.resolve()
    path = (resolved_root / name).resolve()
    if not path.is_relative_to(resolved_root):
        raise RuntimeError("BATCH78_OUTSIDE_DATA_ROOT")
    return path


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise RuntimeError("BATCH78_RGB_U8_REQUIRED")
    h, w = frame.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if scale != 1.0:
        gray = cv2.resize(gray, (max(16, round(w*scale)), max(16, round(h*scale))),
                          interpolation=cv2.INTER_AREA)
    return gray


def scene_pair(previous: np.ndarray, current: np.ndarray) -> dict:
    """Estimate dominant 2-D image flow, not camera pose or world-coordinate flow."""
    if previous.shape != current.shape or previous.ndim != 2:
        raise RuntimeError("BATCH78_GRAY_PAIR_SHAPE")
    h, w = previous.shape
    brightness_delta = float(np.median(current.astype(np.float32)) -
                             np.median(previous.astype(np.float32)))
    brightness_change = float(np.median(np.abs(current.astype(np.float32) -
                                               previous.astype(np.float32))))
    texture = float(cv2.Laplacian(previous, cv2.CV_32F).var())
    common = {"brightness_delta": brightness_delta,
              "brightness_abs_change": brightness_change,
              "texture_laplacian_var": texture,
              "saturated_frac": float(np.mean((previous < 5) | (previous > 250))),
              "flow_kind": "dominant_image_flow_proxy_not_camera_motion"}
    p0 = cv2.goodFeaturesToTrack(previous, maxCorners=MAX_CORNERS, qualityLevel=0.012,
                                minDistance=6, blockSize=3)
    if p0 is None or len(p0) < MIN_TRACKS:
        return {**common, "motion_trusted": False, "reason": "LOW_FEATURE_COUNT",
                "valid_lk": 0, "inliers": 0, "inlier_cells": 0,
                "consensus_fraction": 0.0, "dx_px": None, "dy_px": None,
                "residual_median_px": None, "retention_fraction": 0.0}
    p1, status, _ = cv2.calcOpticalFlowPyrLK(
        previous, current, p0, None, winSize=(21,21), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if p1 is None:
        return {**common, "motion_trusted": False, "reason": "LK_FAILED",
                "valid_lk": 0, "inliers": 0, "inlier_cells": 0,
                "consensus_fraction": 0.0, "dx_px": None, "dy_px": None,
                "residual_median_px": None, "retention_fraction": 0.0}
    pback, back_status, _ = cv2.calcOpticalFlowPyrLK(
        current, previous, p1, None, winSize=(21,21), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if pback is None:
        valid = np.zeros(len(p0), dtype=bool)
    else:
        valid = ((status.ravel() == 1) & (back_status.ravel() == 1) &
                 (np.linalg.norm(pback.reshape(-1,2) - p0.reshape(-1,2), axis=1) <= ROUNDTRIP_TOL) &
                 np.isfinite(p1.reshape(-1,2)).all(axis=1))
    retained = int(valid.sum())
    if retained < MIN_TRACKS:
        return {**common, "motion_trusted": False, "reason": "LOW_LK_RETENTION",
                "valid_lk": retained, "inliers": 0, "inlier_cells": 0,
                "consensus_fraction": 0.0, "dx_px": None, "dy_px": None,
                "residual_median_px": None, "retention_fraction": retained / len(p0)}
    flow = (p1.reshape(-1,2) - p0.reshape(-1,2))[valid]
    points = p0.reshape(-1,2)[valid]
    center = np.median(flow, axis=0)
    residual = np.linalg.norm(flow-center, axis=1)
    tolerance = max(1.5, 2.5 * float(np.median(residual)))
    inlier = residual <= tolerance
    consensus = float(inlier.mean())
    cell_ids = {(int(min(3, 4*x/w)), int(min(3, 4*y/h)))
                for x,y in points[inlier]}
    supported = int(inlier.sum()) >= MIN_TRACKS and consensus >= MIN_CONSENSUS and len(cell_ids) >= MIN_CELLS
    return {**common, "motion_trusted": bool(supported),
            "reason": "DOMINANT_SPATIAL_SUPPORT" if supported else "INSUFFICIENT_SPATIAL_CONSENSUS",
            "valid_lk": retained, "inliers": int(inlier.sum()),
            "inlier_cells": len(cell_ids), "consensus_fraction": consensus,
            "dx_px": float(np.median(flow[inlier,0])) if supported else None,
            "dy_px": float(np.median(flow[inlier,1])) if supported else None,
            "residual_median_px": float(np.median(residual[inlier])) if supported else None,
            "retention_fraction": retained / len(p0)}


def valid_trackpoint(v: object) -> tuple[float,float] | None:
    if not isinstance(v, (list,tuple)) or len(v) != 2:
        return None
    try:
        xy = (float(v[0]), float(v[1]))
    except (TypeError,ValueError,OverflowError):
        return None
    return xy if all(math.isfinite(z) for z in xy) else None


def paired_track_motion(scene_rec: dict, pair_records: list[dict], n: int) -> dict:
    """Compares a *reported* tracker displacement with scene flow proxy.

    This is not physical target motion. Predictions can themselves be wrong.
    """
    tracks = scene_rec.get("tracks", {})
    img_magnitudes, rel_magnitudes = [], []
    for entry in pair_records:
        if not entry["motion_trusted"]:
            continue
        t0, t1 = entry["t0"], entry["t1"]
        for record in tracks.values():
            pts = record.get(BASE_METHOD)
            if not isinstance(pts,list) or len(pts)!=n:
                raise RuntimeError("BATCH78_PREDICTION_LENGTH_DRIFT")
            first, last = valid_trackpoint(pts[t0]), valid_trackpoint(pts[t1])
            if first is None or last is None:
                continue
            dx,dy = last[0]-first[0],last[1]-first[1]
            img_magnitudes.append(math.hypot(dx,dy))
            rel_magnitudes.append(math.hypot(dx-entry["dx_px"],dy-entry["dy_px"]))
    return {"comparable_samples": len(img_magnitudes),
            "reported_image_plane_displacement_px_p50": finite_median(img_magnitudes),
            "reported_scene_relative_proxy_displacement_px_p50": finite_median(rel_magnitudes),
            "identity_verified": False, "camera_motion_ground_truth": False}


def analyze_frames(frames: np.ndarray, stride: int = PAIR_STRIDE,
                   scene_rec: dict | None = None) -> dict:
    if frames.ndim!=4 or frames.shape[-1]!=3 or frames.dtype!=np.uint8:
        raise RuntimeError("BATCH78_FRAME_ARRAY_SHAPE")
    if not 1 <= stride < len(frames):
        raise RuntimeError("BATCH78_BAD_STRIDE")
    ticks=list(range(0,len(frames)-stride,stride))
    pairs=[]
    elapsed=perf_counter()
    for t in ticks:
        a,b=to_gray(frames[t]),to_gray(frames[t+stride])
        rec=scene_pair(a,b)
        # Scene flow was measured on resized grayscale images. Public tracker
        # points are in source-frame pixels: convert both 2-D displacement
        # components before comparing. Optical-flow residuals remain in
        # analysis-raster pixels and are never used as physical motion.
        if rec["motion_trusted"]:
            rec["dx_px"] *= frames.shape[2] / a.shape[1]
            rec["dy_px"] *= frames.shape[1] / a.shape[0]
        rec.update({"t0":t,"t1":t+stride})
        pairs.append(rec)
    ms=(perf_counter()-elapsed)*1000
    trusted=[r for r in pairs if r["motion_trusted"]]
    result={"frames":len(frames), "width":int(frames.shape[2]),"height":int(frames.shape[1]),
            "sampled_pairs":len(pairs), "reliable_pairs":len(trusted),
            "reliable_pair_fraction":len(trusted)/len(pairs),
            "dominant_image_translation_px_p50":finite_median([
                math.hypot(p["dx_px"],p["dy_px"]) for p in trusted]),
            "dominant_image_translation_dx_px_p50":finite_median([p["dx_px"] for p in trusted]),
            "dominant_image_translation_dy_px_p50":finite_median([p["dy_px"] for p in trusted]),
            "luminance_delta_p50":finite_median([p["brightness_delta"] for p in pairs]),
            "median_absolute_frame_change_p50":finite_median([p["brightness_abs_change"] for p in pairs]),
            "texture_laplacian_variance_p50":finite_median([p["texture_laplacian_var"] for p in pairs]),
            "lk_retention_fraction_p50":finite_median([p["retention_fraction"] for p in pairs]),
            "inlier_spatial_cells_p50":finite_median([p["inlier_cells"] for p in pairs]),
            "reason_counts":{key:sum(p["reason"]==key for p in pairs)
                             for key in sorted({p["reason"] for p in pairs})},
            "compute_ms":round(ms,3),
            "camera_motion_ground_truth":False, "background_identity_verified":False}
    if scene_rec is not None:
        result["reported_track_motion"]=paired_track_motion(scene_rec,pairs,len(frames))
    return result


def validate_manifest(manifest: dict, split: str) -> list[dict]:
    if split != "dev":
        raise RuntimeError("BATCH78_TEST_SPLIT_FORBIDDEN_DURING_DEVELOPMENT")
    if not str(manifest.get("dataset","")).startswith("TAP-Vid-Kinetics"):
        raise RuntimeError("BATCH78_UNSUPPORTED_DATASET")
    scenes=manifest.get("scenes")
    if not isinstance(scenes,list) or not scenes:
        raise RuntimeError("BATCH78_MISSING_SCENES")
    dev=[s for s in scenes if s.get("split")=="dev"]
    if not dev or len({s.get("name") for s in dev}) !=len(dev):
        raise RuntimeError("BATCH78_DUPLICATE_OR_EMPTY_DEV")
    for s in dev:
        if not isinstance(s.get("frame_sha256"),str) or len(s["frame_sha256"])!=64:
            raise RuntimeError("BATCH78_MISSING_FRAME_SHA")
    return dev


def audit(manifest_path: Path, root_data: Path, output: Path,
          source_predictions: Path | None=None, stride: int=PAIR_STRIDE) -> dict:
    if "oracle" in str(manifest_path).lower() or "oracle" in str(root_data).lower():
        raise RuntimeError("BATCH78_ORACLE_INPUT_FORBIDDEN")
    manifest_raw=manifest_path.read_bytes()
    manifest=json.loads(manifest_raw)
    dev=validate_manifest(manifest,"dev")
    prediction_map=None
    if source_predictions:
        if "oracle" in str(source_predictions).lower():
            raise RuntimeError("BATCH78_ORACLE_INPUT_FORBIDDEN")
        pred=read_pinned_json(source_predictions)
        if (pred.get("split")!="dev" or pred.get("oracle_unread_during_prediction") is not True
            or pred.get("manifest_sha256") != sha256(manifest_raw)):
            raise RuntimeError("BATCH78_PUBLIC_PREDICTION_FIREWALL")
        prediction_map={r["name"]:r for r in pred["records"]}
        if set(prediction_map)!={s["name"] for s in dev}:
            raise RuntimeError("BATCH78_PUBLIC_PREDICTION_SCENE_DRIFT")
    output_records=[]
    start=perf_counter()
    for scene in dev:
        path=safe_frame_path(root_data,scene["frame_file"])
        data=path.read_bytes()
        if sha256(data)!=scene["frame_sha256"]:
            raise RuntimeError("BATCH78_FRAME_SHA_DRIFT:"+scene["name"])
        frames=np.load(path,allow_pickle=False,mmap_mode="r")
        result=analyze_frames(frames,stride,
                              prediction_map[scene["name"]] if prediction_map else None)
        result["scene"]=scene["name"]
        result["frame_sha256"]=scene["frame_sha256"]
        output_records.append(result)
        print("BATCH78_DEV_SCENE_AUDITED",scene["name"],
              result["reliable_pairs"],result["sampled_pairs"],flush=True)
    payload={"protocol":PROTOCOL, "split":"dev", "exploratory":True,
             "not_an_independent_test":True, "test_oracle_read":False,
             "dev_oracle_read":False,"no_visibility_ground_truth":True,
             "camera_motion_ground_truth":False,
             "dominant_image_flow_is_not_camera_motion":True,
             "stride":stride,"manifest_sha256":sha256(manifest_raw),
             "public_predictions_sha256":(sha256(source_predictions.read_bytes())
                                           if source_predictions else None),
             "records":output_records,
             "summary":{"scenes":len(output_records),
                        "pairs":sum(r["sampled_pairs"] for r in output_records),
                        "reliable_pairs":sum(r["reliable_pairs"] for r in output_records),
                        "scene_reliable_fraction_p50":finite_median([
                            r["reliable_pair_fraction"] for r in output_records]),
                        "scene_2d_translation_px_p50":finite_median([
                            r["dominant_image_translation_px_p50"]
                            for r in output_records if r["dominant_image_translation_px_p50"] is not None]),
                        "scene_luminance_delta_p50":finite_median([
                            r["luminance_delta_p50"] for r in output_records]),
                        "scene_texture_var_p50":finite_median([
                            r["texture_laplacian_variance_p50"] for r in output_records]),
                        "total_scene_cpu_ms":round(sum(r["compute_ms"] for r in output_records),3),
                        "wall_ms":round((perf_counter()-start)*1000,3)}}
    payload["result_sha256"]=write_once(output,payload)
    return payload


def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--data-root",type=Path,default=PROJECT)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--source-predictions",type=Path)
    ap.add_argument("--split",default="dev",choices=["dev","test"])
    ap.add_argument("--stride",type=int,default=PAIR_STRIDE)
    args=ap.parse_args()
    if args.split!="dev":
        raise SystemExit("BATCH78_TEST_SPLIT_FORBIDDEN")
    result=audit(args.manifest,args.data_root,args.output,args.source_predictions,args.stride)
    print("BATCH78_DEV_AUDIT_PINNED",result["result_sha256"],result["summary"],flush=True)


if __name__=="__main__":
    main()
