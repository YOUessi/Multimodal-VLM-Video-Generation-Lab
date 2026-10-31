"""Shared frame-feature cache must preserve every frozen selection and pixel."""
from pathlib import Path
import importlib.util,sys,json
P=Path(__file__).resolve().parents[1]/"experiments"
sys.path.insert(0,str(P))
spec=importlib.util.spec_from_file_location("cache",P/"shared_temporal_cache.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def _video(tmp_path):
    import cv2,numpy as np
    p=tmp_path/"video.mp4"
    w=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*"mp4v"),10,(96,64))
    assert w.isOpened()
    for i in range(61):
        img=np.zeros((64,96,3),dtype=np.uint8)
        img[:,:(i*3)%95]=(i*5%255,12,120)
        w.write(img)
    w.release()
    return p
def test_same_frame_indices_and_pixels_across_cold_warm_cache(tmp_path):
    path=_video(tmp_path)
    naive,imgs=m.naive_pair(path,61,48)
    first,images,hit=m.cached_pair(path,61,48,tmp_path/"cache")
    assert first==naive and hit is False
    for a,b in zip(imgs,images):
        assert len(a)==len(b)
        for left,right in zip(a,b):
            assert left.tobytes()==right.tobytes()
    second,images2,hit=m.cached_pair(path,61,48,tmp_path/"cache")
    assert hit and first==second
    assert len(images2[0])==3 and len(images2[1])==5
def test_cache_key_tracks_real_video_bytes(tmp_path):
    file=_video(tmp_path)
    one=m.file_digest(file)
    f,_=m.read_or_build(file,61,48,tmp_path/"cache")
    assert f["decoded_frames"]==61
    with file.open("ab") as w:w.write(b"changed")
    assert m.file_digest(file)!=one
    f2,hit=m.read_or_build(file,61,48,tmp_path/"cache")
    assert not hit
def test_cached_config_frozen():
    p=P.parents[0]/"configs/shared_cache_v1.json"
    config=json.loads(p.read_text())
    assert config["probe_count"]==48 and config["frame_budgets"]==[3,5]
    assert config["warm_repeats"]==3
