"""Batch78 fail-closed checks for pre-label DEV-only image-flow characterization."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from experiments import batch78_unlabeled_motion_audit as m


def textured(seed=9, h=160, w=192):
    rng=np.random.default_rng(seed)
    out=rng.integers(0,256,(h,w),dtype=np.uint8)
    return out


def translated(gray,dx=3,dy=2):
    h,w=gray.shape
    return cv2.warpAffine(gray,np.float32([[1,0,dx],[0,1,dy]]),
                          (w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)


def rgb(gray):
    return np.repeat(gray[...,None],3,axis=2)


def demo_frames():
    a=textured()
    return np.stack([rgb(a),rgb(translated(a,3,2)),rgb(translated(a,6,4))])


def manifest_for(tmp_path, frames, split="dev"):
    rel="samples/v00.npy"
    file=tmp_path/rel
    file.parent.mkdir(exist_ok=True,parents=True)
    np.save(file,frames)
    raw=file.read_bytes()
    info={"dataset":"TAP-Vid-Kinetics; fixture",
          "scenes":[{"name":"v00","split":split,"frame_file":rel,
                     "frame_sha256":hashlib.sha256(raw).hexdigest()}]}
    path=tmp_path/"manifest.json"
    path.write_text(json.dumps(info))
    return path,info


def test_exact_translation_estimation():
    a=textured()
    result=m.scene_pair(a,translated(a,3,2))
    assert result["motion_trusted"] is True
    assert abs(result["dx_px"]-3)<0.7
    assert abs(result["dy_px"]-2)<0.7
    assert result["inlier_cells"]>=6
    assert result["flow_kind"].endswith("not_camera_motion")


def test_no_texture_fails_closed():
    a=np.zeros((160,192),dtype=np.uint8)
    res=m.scene_pair(a,a)
    assert not res["motion_trusted"]
    assert res["dx_px"] is None
    assert res["reason"]=="LOW_FEATURE_COUNT"


def test_luminance_change():
    a=np.full((160,192),70,dtype=np.uint8)
    b=np.full((160,192),95,dtype=np.uint8)
    r=m.scene_pair(a,b)
    assert r["brightness_delta"]==25.0
    assert r["brightness_abs_change"]==25.0


def test_bad_pair_shape_rejected():
    with pytest.raises(RuntimeError,match="GRAY_PAIR_SHAPE"):
        m.scene_pair(np.zeros((8,8),dtype=np.uint8),np.zeros((8,9),dtype=np.uint8))


def test_rgb_only():
    with pytest.raises(RuntimeError,match="RGB_U8"):
        m.to_gray(np.zeros((20,20),dtype=np.uint8))


@pytest.mark.parametrize("split",["test","unknown",""])
def test_split_is_dev_only(split):
    with pytest.raises(RuntimeError,match="TEST_SPLIT_FORBIDDEN"):
        m.validate_manifest({"dataset":"TAP-Vid-Kinetics","scenes":[]},split)


def test_scene_manifest_validation():
    _,info=manifest_for(Path(__import__("tempfile").mkdtemp()),demo_frames())
    assert len(m.validate_manifest(info,"dev"))==1


@pytest.mark.parametrize("name",["../x.npy","/tmp/o.npy","bad.mp4","sub/../../x.npy"])
def test_unsafe_path_rejected(tmp_path,name):
    with pytest.raises(RuntimeError,match="UNSAFE_FRAME_PATH|OUTSIDE_DATA_ROOT"):
        m.safe_frame_path(tmp_path,name)


def test_no_escape_symlink(tmp_path):
    (tmp_path/"link.npy").symlink_to(tmp_path.parent/"escape.npy")
    with pytest.raises(RuntimeError,match="OUTSIDE_DATA_ROOT"):
        m.safe_frame_path(tmp_path,"link.npy")


def test_empty_dev_fails():
    with pytest.raises(RuntimeError,match="DUPLICATE_OR_EMPTY_DEV"):
        m.validate_manifest({"dataset":"TAP-Vid-Kinetics","scenes":[
            {"split":"test","name":"s0","frame_sha256":"a"*64}]},"dev")


def test_duplicate_dev_fails():
    with pytest.raises(RuntimeError,match="DUPLICATE_OR_EMPTY_DEV"):
        m.validate_manifest({"dataset":"TAP-Vid-Kinetics","scenes":[
            {"split":"dev","name":"s0","frame_sha256":"a"*64},
            {"split":"dev","name":"s0","frame_sha256":"a"*64}]},"dev")


def test_valid_trackpoints():
    assert m.valid_trackpoint([1,2])==(1.,2.)
    assert m.valid_trackpoint(None) is None
    assert m.valid_trackpoint([1,float("nan")]) is None
    assert m.valid_trackpoint(["oops",1]) is None


def test_paired_track_residual():
    pair=[{"motion_trusted":True,"t0":0,"t1":1,"dx_px":3.0,"dy_px":2.0}]
    record={"tracks":{"q":{"appearance_fb_055":[[4,5],[7,7],None]}}}
    out=m.paired_track_motion(record,pair,3)
    assert out["comparable_samples"]==1
    assert out["reported_scene_relative_proxy_displacement_px_p50"]==0.0
    assert out["reported_image_plane_displacement_px_p50"]==pytest.approx(np.sqrt(13))
    assert not out["identity_verified"]


def test_paired_track_wrong_length():
    pair=[{"motion_trusted":True,"t0":0,"t1":1,"dx_px":3.0,"dy_px":2.0}]
    with pytest.raises(RuntimeError,match="PREDICTION_LENGTH_DRIFT"):
        m.paired_track_motion({"tracks":{"q":{"appearance_fb_055":[[1,2]]}}},pair,3)


def test_analyze_frames_has_no_true_camera_claim():
    r=m.analyze_frames(demo_frames(),stride=1)
    assert r["sampled_pairs"]==2
    assert r["reliable_pairs"]>=1
    assert not r["camera_motion_ground_truth"]
    assert r["dominant_image_translation_px_p50"]>1


@pytest.mark.parametrize("stride",[0,3,10])
def test_bad_stride(stride):
    with pytest.raises(RuntimeError,match="BAD_STRIDE"):
        m.analyze_frames(demo_frames(),stride=stride)


def test_actual_dev_only_audit_pins_without_oracle(tmp_path,monkeypatch):
    frames=demo_frames()
    path,info=manifest_for(tmp_path,frames)
    # Any attempt to read an oracle would fail, even if the manifest points to it.
    (tmp_path/"oracle-private.json").write_text("DO_NOT_READ")
    original=Path.read_bytes
    def blocked(self):
        if "oracle" in str(self).lower():
            raise AssertionError("Attempted to read private oracle")
        return original(self)
    monkeypatch.setattr(Path,"read_bytes",blocked)
    result=m.audit(path,tmp_path,tmp_path/"summary.json",stride=1)
    assert result["split"]=="dev"
    assert result["dev_oracle_read"] is False
    assert result["test_oracle_read"] is False
    assert result["summary"]["scenes"]==1
    assert m.read_pinned_json(tmp_path/"summary.json")["protocol"]==m.PROTOCOL
    with pytest.raises(RuntimeError,match="IMMUTABLE_OUTPUT_EXISTS"):
        m.audit(path,tmp_path,tmp_path/"summary.json",stride=1)


def test_frame_integrity_mismatch(tmp_path):
    path,info=manifest_for(tmp_path,demo_frames())
    info["scenes"][0]["frame_sha256"]="0"*64
    path.write_text(json.dumps(info))
    with pytest.raises(RuntimeError,match="FRAME_SHA_DRIFT"):
        m.audit(path,tmp_path,tmp_path/"r.json",stride=1)


def test_public_prediction_schema_firewall(tmp_path):
    path,info=manifest_for(tmp_path,demo_frames())
    p=tmp_path/"p.json"
    m.write_once(p,{"split":"test","oracle_unread_during_prediction":True,
                    "manifest_sha256":m.sha256(path.read_bytes()),"records":[]})
    with pytest.raises(RuntimeError,match="PUBLIC_PREDICTION_FIREWALL"):
        m.audit(path,tmp_path,tmp_path/"r.json",source_predictions=p,stride=1)


def test_pin_tamper_detected(tmp_path):
    p=tmp_path/"data.json"
    m.write_once(p,{"foo":1})
    p.write_text('{"foo":2}')
    with pytest.raises(RuntimeError,match="PIN_DRIFT"):
        m.read_pinned_json(p)


def test_immutable_duplicate_pin_detected(tmp_path):
    p=tmp_path/"data.json"
    p.with_suffix(".json.sha256").write_text("x")
    with pytest.raises(RuntimeError,match="IMMUTABLE_OUTPUT_EXISTS"):
        m.write_once(p,{"foo":1})


def test_oracle_input_path_forbidden(tmp_path):
    with pytest.raises(RuntimeError,match="ORACLE_INPUT_FORBIDDEN"):
        m.audit(tmp_path/"oracle-private.json",tmp_path,tmp_path/"out.json")


def test_no_nonfinite_json():
    assert m.finite_median([1,float("nan"),3])==2
    assert m.finite_median([float("inf")]) is None


def test_original_pixel_scale_preserved_after_resize():
    a=textured(seed=91,h=320,w=384)
    b=translated(a,8,4)
    frames=np.stack([rgb(a),rgb(b)])
    result=m.analyze_frames(frames,stride=1,
         scene_rec={"tracks":{"q":{m.BASE_METHOD:[[100,100],[108,104]]}}})
    assert result["reliable_pairs"]==1
    assert result["dominant_image_translation_dx_px_p50"]==pytest.approx(8,abs=1)
    assert result["dominant_image_translation_dy_px_p50"]==pytest.approx(4,abs=1)
    assert result["reported_track_motion"]["reported_scene_relative_proxy_displacement_px_p50"]<1.5
