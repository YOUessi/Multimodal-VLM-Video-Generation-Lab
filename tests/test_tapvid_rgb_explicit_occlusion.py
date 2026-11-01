"""Regression tests for independent official point tracks and visibility firewall."""
from pathlib import Path
import hashlib
import json
import numpy as np
import cv2
import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import tapvid_rgb_data as data
import tapvid_rgb_eval as ev
import tapvid_download as downloader


def textured_video(n=5, dx=1, dy=0):
    rng = np.random.default_rng(21)
    first = rng.integers(30, 225, (80, 80), dtype=np.uint8)
    frames = [
        cv2.warpAffine(first, np.float32([[1,0,k*dx],[0,1,k*dy]]),
                       (80,80), flags=cv2.INTER_LINEAR)
        for k in range(n)
    ]
    return frames


@pytest.mark.parametrize("seed", ["a", "b", "c", "a"*8])
def test_selection_is_order_independent(seed):
    names = [seed+str(i) for i in range(35)]
    assert data.selected_indices(names) == data.selected_indices(names[::-1])
    assert len(data.selected_indices(names)) == data.DEV + data.TEST
    assert len(set(data.selected_indices(names))) == data.DEV + data.TEST


def test_query_is_first_visible_and_finite():
    pts = np.zeros((1, 4, 2))
    pts[0,2] = [9., 18.]
    occl = np.array([[True, True, False, True]])
    q = data.extract_query("v0", 0, pts, occl)
    assert q["query_time"] == 2 and q["query_xy"] == [9.,18.]


def test_all_occluded_query_is_dropped():
    assert data.extract_query("v0",0,np.zeros((1,3,2)),np.ones((1,3),bool)) is None


def test_public_queries_cannot_contain_out_of_bounds():
    video=np.zeros((2,32,32,3),np.uint8)
    with pytest.raises(ValueError,match="OUTSIDE"):
        ev.check_video(video,[{"query_time":0,"query_xy":[-1,3]}])
    with pytest.raises(ValueError,match="BAD_PUBLIC"):
        ev.check_video(video,[{"query_time":3,"query_xy":[4,3]}])


@pytest.mark.parametrize("fb", [False,True])
def test_klt_tracks_known_image_plane_translation(fb):
    frames=textured_video(n=6,dx=1,dy=1)
    q={"query_time":0,"query_xy":[38.,39.]}
    tr=ev.klt_point(frames,q,fb=fb)
    assert len(tr)==6 and all(v is not None for v in tr)
    assert abs(tr[-1][0]-43)<2.2 and abs(tr[-1][1]-44)<2.2


def test_klt_query_late_never_predicts_before_query():
    frames=textured_video(n=4)
    q={"query_time":2,"query_xy":[38.,39.]}
    tr=ev.klt_point(frames,q,fb=True)
    assert tr[:2]==[None,None] and tr[2]==[38.,39.]


def test_klt_breaks_on_failure_and_never_reidentifies():
    gray=[np.zeros((40,40),np.uint8) for _ in range(8)]
    q={"query_time":0,"query_xy":[20.,20.]}
    tr=ev.klt_point(gray,q,fb=True)
    assert tr[0]==[20.,20.] and tr[1:]==[None]*7


def test_template_matches_known_image_plane_translation():
    frames=textured_video(n=5,dx=1)
    tr=ev.template_trace(frames,{"query_time":0,"query_xy":[39.,38.]})
    assert tr[0]==[39.,38.] and tr[-1] is not None
    assert abs(tr[-1][0]-43)<3


def test_template_no_texture_abstains():
    frames=[np.zeros((80,80),np.uint8) for _ in range(6)]
    tr=ev.template_trace(frames,{"query_time":0,"query_xy":[40.,40.]})
    assert tr[1:]==[None]*5


def test_stationary_baseline_has_correct_query_time():
    gray=textured_video()
    tr,ms=ev.predict_query(gray,{"query_time":2,"query_xy":[22.,34.]})
    assert tr["stationary_point"][:2] == [None,None]
    assert tr["stationary_point"][2:] == [[22.,34.]]*3
    assert set(ms)==set(ev.METHODS)


def test_pin_detects_tampered_bytes(tmp_path):
    p=tmp_path/"test_predictions.json"
    p.write_text('{"a":1}')
    ev.pin(p)
    ev.require_pin(p)
    p.write_text('{"a":2}')
    with pytest.raises(RuntimeError,match="CHANGED"):
        ev.require_pin(p)


def test_once_refuses_mutation(tmp_path):
    p=tmp_path/"private.json"
    data.atomic_once(p,b'original')
    data.atomic_once(p,b'original')
    with pytest.raises(RuntimeError,match="CONFLICT"):
        data.atomic_once(p,b'changed')


def test_range_guard_rejects_bad_indices():
    with pytest.raises(AssertionError):
        downloader.range_bytes(-1,1)
    with pytest.raises(AssertionError):
        downloader.range_bytes(4,3)


def test_archive_verifier_rejects_wrong_size(tmp_path):
    p=tmp_path/"not-official.zip"
    p.write_bytes(b"not-trusted")
    with pytest.raises(RuntimeError,match="SIZE"):
        downloader.verify(p)


def fake_scoring(tmp_path,monkeypatch,*,after_prediction_mutation=False):
    d=tmp_path/"data"
    out=tmp_path/"output"
    d.mkdir()
    out.mkdir()
    manifest={"dataset_origin_sha256":"source-good",
              "scenes":[{"name":"clip_00","split":"test","nframes":4}]}
    (d/"manifest.json").write_text(json.dumps(manifest))
    queries={"clip_00":[{"id":"p0001","query_time":0,"query_xy":[10,10]}]}
    (d/"queries-public.json").write_text(json.dumps(queries))
    truth={"clip_00":{"tracks":{"p0001":{
        "xy":[[10,10],[11,11],[12,12],[13,13]],
        "occluded":[False,False,True,False]}}}}
    (d/"oracle-private.json").write_text(json.dumps(truth))
    manifest["oracle_sha256"] = ev.sha_bytes((d/"oracle-private.json").read_bytes())
    manifest["public_queries_sha256"] = ev.sha_bytes((d/"queries-public.json").read_bytes())
    (d/"manifest.json").write_text(json.dumps(manifest))
    trace=[[10.,10.],[11.,11.],[12.,12.],None]
    records=[{"name":"clip_00","tracks":{"p0001":{m:trace for m in ev.METHODS}},
              "cpu_query_ms":{"p0001":{m:6.0 for m in ev.METHODS}}}]
    saved={
        "source_commit":"testsha","manifest_sha256":ev.sha_bytes((d/"manifest.json").read_bytes()),
        "public_query_sha256":ev.sha_bytes((d/"queries-public.json").read_bytes()),
        "oracle_unread_during_prediction":True,"models":list(ev.METHODS),
        "split":"test","records":records,
    }
    ev.once(out/"test_predictions.json",saved)
    ev.pin(out/"test_predictions.json")
    monkeypatch.setattr(ev,"DATA",d)
    monkeypatch.setattr(ev,"OUT",out)
    monkeypatch.setattr(ev,"gitsha",lambda:"testsha")
    monkeypatch.setattr(ev.src,"audit",lambda:manifest)
    if after_prediction_mutation:
        saved["source_commit"]="DIFFERENT"
        (out/"test_predictions.json").write_text(json.dumps(saved))
        # Digest is re-sealed only to ensure SHA guard catches stale source.
        (out/"test_predictions.sha256").write_text(ev.sha_bytes((out/"test_predictions.json").read_bytes()))
    return out


def test_score_counts_true_point_absence_and_wrong_accept(tmp_path,monkeypatch):
    out=fake_scoring(tmp_path,monkeypatch)
    ev.score("test")
    report=json.loads((out/"test_results.json").read_text())
    for m in ev.METHODS:
        s=report["methods"][m]
        assert s["visible"]==2 and s["occluded"]==1
        assert s["accepted_visible"]==1
        assert s["correct8"]==1
        assert s["absent_false_accepted"]==1
        assert s["pck8_unconditional"]==0.5
        assert s["absence_false_accept_rate"]==1.0


def test_source_commit_mismatch_blocks_score(tmp_path,monkeypatch):
    fake_scoring(tmp_path,monkeypatch,after_prediction_mutation=True)
    with pytest.raises(RuntimeError,match="FROZEN_ORACLE"):
        ev.score("test")


def test_unsealed_prediction_blocks_private_oracle(tmp_path,monkeypatch):
    out=fake_scoring(tmp_path,monkeypatch)
    (out/"test_predictions.sha256").unlink()
    with pytest.raises(RuntimeError,match="MISSING_ORACLE"):
        ev.score("test")


def test_inference_works_without_private_future_labels(tmp_path,monkeypatch):
    d=tmp_path/"data"
    (d/"frames").mkdir(parents=True)
    frames=np.stack([np.stack([x,x,x],axis=-1) for x in textured_video(4)])
    np.save(d/"frames/clip_00.npy",frames)
    manifest={"dataset_origin_sha256":"frozen","scenes":[
        {"name":"clip_00","split":"dev","frames":"frames/clip_00.npy",
         "nframes":4,"nqueries":1}]}
    (d/"manifest.json").write_text(json.dumps(manifest))
    (d/"queries-public.json").write_text(json.dumps({
        "clip_00":[{"id":"p0001","query_time":0,"query_xy":[40,40]}]}))
    monkeypatch.setattr(ev,"DATA",d)
    monkeypatch.setattr(ev,"OUT",tmp_path/"output")
    monkeypatch.setattr(ev,"gitsha",lambda:"testsha")
    monkeypatch.setattr(ev.src,"audit",lambda:manifest)
    assert not (d/"oracle-private.json").exists()
    ev.predict("dev")
    p=tmp_path/"output/dev_predictions.json"
    ev.require_pin(p)
    assert json.loads(p.read_text())["oracle_unread_during_prediction"] is True

def test_official_normalized_xy_to_raster_pixels_not_top_left():
    # Upstream TAP-Vid stores normalized coordinates and multiplies by (W,H).
    # In particular, a center point MUST become (128,128), not (0.5,0.5).
    unit = np.array([[[0.5,0.25],[0.0,1.0],[0.2,0.9]]],dtype=np.float64)
    occ = np.array([[False,True,False]],dtype=bool)
    result = data.normalized_to_pixel_points(unit,occ,256,256)
    assert result[0,0].tolist() == [128.0,64.0]
    assert result[0,1].tolist() == [0.0,256.0]
    assert result[0,2].tolist() == [51.2,230.4]
    np.testing.assert_array_equal(unit[0,0],[0.5,0.25])


def test_official_visible_point_out_of_unit_range_fails_closed():
    bad = np.array([[[1.2,0.4]]])
    with pytest.raises(RuntimeError,match="NOT_NORMALIZED"):
        data.normalized_to_pixel_points(bad,np.array([[False]]),256,256)
    # Occluded coordinates might be offscreen by design: should not be scored.
    assert data.normalized_to_pixel_points(bad,np.array([[True]]),256,256).shape == (1,1,2)


def test_private_oracle_mutation_is_rejected_even_when_prediction_sealed(tmp_path,monkeypatch):
    out=fake_scoring(tmp_path,monkeypatch)
    p=tmp_path/"data"/"oracle-private.json"
    truth=json.loads(p.read_text())
    truth["clip_00"]["tracks"]["p0001"]["occluded"][1]=True
    p.write_text(json.dumps(truth))
    with pytest.raises(RuntimeError,match="PRIVATE_ORACLE"):
        ev.score("test")


def test_upstream_tapvid_jaccard_counts_wrong_visible_false_positives(tmp_path,monkeypatch):
    out=fake_scoring(tmp_path,monkeypatch)
    ev.score("test")
    report=json.loads((out/"test_results.json").read_text())
    # 2 visible (1 correct, 1 abstained), 1 occluded but predicted visible.
    # Jaccard at every radius: TP / (GT visible + absent FP + bad-visible FP)
    assert report["scene_details"]["clip_00"]["stationary_point"]["jaccard_mean_1_2_4_8_16"] == round(1/3,5)

def test_preparation_manifest_hashes_generated_payloads_before_disk_write(tmp_path, monkeypatch):
    # Regression: the first frozen Batch38 attempt crashed because it hashed
    # oracle-private.json BEFORE atomic_once had written it. Test complete
    # prepare+audit without opening any real source or TEST answers.
    d = tmp_path/"shared-data"
    d.mkdir()
    (d/"official.zip").write_bytes(b"fake-source-in-unit-test")
    tracks = np.tile(np.array([[[0.5,0.25],[0.55,0.25],[0.6,0.25],[0.65,0.25]]]),
                    (2,1,1))
    samples = [(f"video_{k:04d}",{
        "video": np.zeros((4,8,8,3),dtype=np.uint8),
        "points": tracks.copy(),
        "occluded": np.zeros((2,4),dtype=bool),
    }) for k in range(data.DEV+data.TEST)]
    monkeypatch.setattr(data,"DATA",d)
    monkeypatch.setattr(data,"verify",lambda _:None)
    monkeypatch.setattr(data,"official_records",lambda _:samples)
    data.prepare()
    manifest=json.loads((d/"batch38_frozen/manifest.json").read_text())
    for key, filename in [("oracle_sha256","oracle-private.json"),
                          ("public_queries_sha256","queries-public.json")]:
        expected=hashlib.sha256((d/"batch38_frozen"/filename).read_bytes()).hexdigest()
        assert manifest[key] == expected
    query=json.loads((d/"batch38_frozen/queries-public.json").read_text())
    assert sorted(q["query_xy"][0] for q in query["clip_00"]) == [4.,4.]
    # It must audit without reading future private answers during prediction.
    (d/"batch38_frozen/oracle-private.json").rename(d/"batch38_frozen/private-unavailable")
    data.audit()
