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

def test_sha_identical_distinct_filenames_warm_hit_and_cold_isolated(tmp_path):
    """A second filename with identical MP4 bytes must not corrupt cold timing."""
    from shutil import copy2
    from tempfile import TemporaryDirectory
    first=_video(tmp_path)
    second=tmp_path/"same_bytes_new_filename.mp4"
    copy2(first,second)
    assert m.file_digest(first)==m.file_digest(second)
    shared=tmp_path/"shared"
    _,hit=m.read_or_build(first,61,48,shared)
    assert hit is False
    _,hit=m.read_or_build(second,61,48,shared)
    assert hit is True
    with TemporaryDirectory(dir=str(tmp_path)) as tmp:
        _,hit=m.read_or_build(second,61,48,Path(tmp))
        assert hit is False

def test_grouped_cache_config_uses_new_nonleaked_manifest():
    conf=json.loads((P.parents[0]/"configs/shared_cache_grouped_v1.json").read_text())
    assert conf["star_manifest"]=="outputs/star_grouped_sft_v1/manifest.json"
    assert conf["adaptive_plan"]=="outputs/adaptive_grouped_v1/sampling_plan.json"
    assert conf["output_dir"]=="outputs/shared_cache_grouped_v1"
