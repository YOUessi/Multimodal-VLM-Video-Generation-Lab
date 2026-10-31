"""Reuse 48-probe visual signals and unique decoded frames across n=3/n=5.

Identical selection algorithm to frozen adaptive_v2: motion(.65) + hue(.35),
5%/95% endpoints, 10%/7% separation. Only the expensive visual feature
extraction and repeated frame decoding are cached. No labels in input/output.
"""
from __future__ import annotations
import argparse,hashlib,json,random,statistics,time
from pathlib import Path
import cv2,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
from adaptive_temporal_sampling import candidate_positions,indices_for_signals,select_video,load_pil

VERSION="mvvg_adaptive_signal_v2_sharedcache_20261031"
def file_digest(video):
    h=hashlib.sha256()
    with Path(video).open("rb") as stream:
        for chunk in iter(lambda:stream.read(2**20),b""):
            h.update(chunk)
    return h.hexdigest()
def relative_or_abs(video):
    try:return str(Path(video).relative_to(ROOT))
    except ValueError:return str(video)
def compute_signals(video,decoded,probe_count):
    (_, _),positions=candidate_positions(decoded,probe_count)
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise RuntimeError("Cannot open video: "+str(video))
    evidence=[]
    try:
        for ix in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES,ix)
            ok,bgr=cap.read()
            if not ok:raise RuntimeError(f"Cannot decode frame {ix} {video}")
            thumbnail=cv2.resize(bgr,(96,72))
            gray=cv2.cvtColor(thumbnail,cv2.COLOR_BGR2GRAY)
            hsv=cv2.cvtColor(thumbnail,cv2.COLOR_BGR2HSV)
            hist=cv2.calcHist([hsv],[0],None,[24],[0,180])
            cv2.normalize(hist,hist)
            evidence.append((gray,hist))
    finally:cap.release()
    motion=[0.];color=[0.]
    for k in range(1,len(evidence)):
        a,b=evidence[k-1],evidence[k]
        motion.append(float(np.mean(np.abs(b[0].astype(np.float32)-a[0].astype(np.float32)))))
        color.append(float(cv2.compareHist(a[1],b[1],cv2.HISTCMP_BHATTACHARYYA)))
    def normal(values):
        hi=max(values)
        return [float(v/hi) if hi>1e-9 else 0.0 for v in values]
    m,c=normal(motion),normal(color)
    return {"probe_positions":positions,"signals":[.65*x+.35*y for x,y in zip(m,c)],
            "raw_motion_max":max(motion),"raw_color_max":max(color),
            "decoded_frames":int(decoded),"probe_count":int(probe_count)}

def read_or_build(video,decoded,probes,cache_dir):
    digest=file_digest(video)
    key=hashlib.sha256(f"{VERSION}:{digest}:{decoded}:{probes}".encode()).hexdigest()
    path=Path(cache_dir)/(key+".json")
    if path.is_file():
        doc=json.loads(path.read_text(encoding="utf8"))
        if (doc["version"]!=VERSION or doc["video_sha256"]!=digest or
            doc["features"]["decoded_frames"]!=decoded or
            doc["features"]["probe_count"]!=probes):
            raise RuntimeError("CACHE_KEY_OR_CONTENT_DRIFT")
        return doc["features"],True
    f=compute_signals(video,decoded,probes)
    doc={"version":VERSION,"video_sha256":digest,"local_path":relative_or_abs(video),
         "features":f}
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n")
    tmp.replace(path)
    return f,False

def choose(feature,count):
    decoded=feature["decoded_frames"]
    gap=max(1,round(decoded*(.07 if count==5 else .10)))
    return indices_for_signals(feature["probe_positions"],feature["signals"],count,gap)

def decode_shared(video,selection3,selection5):
    unique=sorted(set(selection3+selection5))
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise RuntimeError("Cannot open video")
    images={}
    try:
        for ix in unique:
            cap.set(cv2.CAP_PROP_POS_FRAMES,ix)
            ok,bgr=cap.read()
            if not ok:raise RuntimeError(f"Chosen frame decode failed {ix}")
            img=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            img.thumbnail((640,480))
            images[ix]=img
    finally:cap.release()
    return ([images[i] for i in selection3],[images[i] for i in selection5])

def cached_pair(video,decoded,probes,cache_dir):
    feature,hit=read_or_build(video,decoded,probes,cache_dir)
    three=choose(feature,3);five=choose(feature,5)
    images=decode_shared(video,three,five)
    return (three,five),images,hit

def naive_pair(video,decoded,probes):
    a,_=select_video(video,decoded,3,probes)
    b,_=select_video(video,decoded,5,probes)
    return (a,b),(load_pil(video,a),load_pil(video,b))

def _pct(items,q):
    x=sorted(items);index=(len(x)-1)*q;lo=int(index);hi=min(lo+1,len(x)-1)
    return round(x[lo]+(x[hi]-x[lo])*(index-lo),5)

def run(config):
    cfg=json.loads((ROOT/config).read_text())
    manifest=json.loads((ROOT/cfg["star_manifest"]).read_text())
    plan=json.loads((ROOT/cfg["adaptive_plan"]).read_text())
    assert manifest["sha256"]==plan["star_manifest_sha256"]
    frozen={x["id"]:x for x in plan["records"]}
    test=[r for r in manifest["rows"] if r["split"]=="test"]
    assert len(test)==40 and len(frozen)==40
    cache=ROOT/cfg["cache_dir"]; cache.mkdir(parents=True,exist_ok=True)
    raw=[]
    order=list(test)
    random.Random(cfg["order_seed"]).shuffle(order)
    for index,row in enumerate(order):
        video=ROOT/row["video"];count=row["decoded_frames"]
        expected=[frozen[row["source_index"]]["frames"][str(n)]["frame_indices"] for n in (3,5)]
        def naive():
            t=time.perf_counter();choice,imgs=naive_pair(video,count,cfg["probe_count"])
            return time.perf_counter()-t,choice,imgs,None
        def cached():
            t=time.perf_counter();choice,imgs,hit=cached_pair(video,count,cfg["probe_count"],cache)
            return time.perf_counter()-t,choice,imgs,hit
        steps=[("naive",naive),("cached_cold",cached)]
        if index%2:steps.reverse()
        runs={}
        for name,fn in steps:
            latency,selected,frames,hit=fn()
            if selected!=tuple(expected):raise RuntimeError("SHARED_CACHE_CHANGED_FROZEN_FRAME_SELECTION")
            runs[name]={"wall_seconds":round(latency,6),"images":frames,
                        "cached":hit}
        if runs["cached_cold"]["cached"] is True:raise RuntimeError("Expected a cold feature miss")
        for branch in (0,1):
            left=runs["naive"]["images"][branch]
            right=runs["cached_cold"]["images"][branch]
            if len(left)!=len(right):raise RuntimeError("Frame count changed")
            for a,b in zip(left,right):
                if a.size!=b.size or a.tobytes()!=b.tobytes():
                    raise RuntimeError("PIL_INPUT_PIXELS_CHANGED")
        warm=[]
        for repeat in range(cfg["warm_repeats"]):
            latency,choice,_,hit=cached()
            assert hit is True and choice==tuple(expected)
            warm.append(round(latency,6))
        raw.append({"id":row["source_index"],"video":relative_or_abs(video),
                    "naive_pair_seconds":runs["naive"]["wall_seconds"],
                    "cold_cached_pair_seconds":runs["cached_cold"]["wall_seconds"],
                    "warm_cached_pair_seconds":warm,
                    "selected3":expected[0],"selected5":expected[1],
                    "identical_video_inputs":True})
        if (index+1)%10==0:print("SHARED_CACHE_VERIFIED",index+1,"/40",flush=True)
    measurements={
       "naive_pair":[r["naive_pair_seconds"] for r in raw],
       "cold_cached_pair":[r["cold_cached_pair_seconds"] for r in raw],
       "warm_cached_pair":[x for r in raw for x in r["warm_cached_pair_seconds"]]}
    stats={label:{"mean_s":round(statistics.mean(vals),5),
                  "median_s":_pct(vals,.5),"p95_s":_pct(vals,.95),
                  "n":len(vals)} for label,vals in measurements.items()}
    stats["cold_speedup_mean"]=round(stats["naive_pair"]["mean_s"]/stats["cold_cached_pair"]["mean_s"],3)
    stats["warm_speedup_mean"]=round(stats["naive_pair"]["mean_s"]/stats["warm_cached_pair"]["mean_s"],3)
    summary={"source":"STAR 40 fixed heldout, identical adaptive visual selection",
       "motion_weight":.65,"hue_weight":.35,"probe_count":cfg["probe_count"],
       "warm_repeats":cfg["warm_repeats"],"sample_count":40,
       "all_cached_and_naive_images_byte_identical":True,
       "cached_file_key":"full SHA256 video bytes + algorithm/version + decoded/probe count",
       "all_data_label_blind":True,"metrics":stats,"per_video":raw,
       "limitation":"Warm cache benefits repeated videos only; no change to model accuracy is claimed"}
    out=ROOT/cfg["output_dir"];out.mkdir(parents=True,exist_ok=True)
    (out/"shared_cache.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    lines=["# Shared visual feature cache: identical frames, measured CPU savings","",
       "40 frozen STAR heldout real videos, no label access or model re-evaluation.",
       "Three/five-frame selections must be EXACTLY the same as the previously frozen adaptive plan;",
       "PIL RGB bytes were checked identical for every input frame.",
       "","| Workflow per video (both 3 and 5 frames) | Mean CPU seconds | Median seconds | P95 seconds |",
       "|---|---:|---:|---:|"]
    for label,d in stats.items():
        if not isinstance(d,dict):continue
        lines.append(f"| {label} | {d['mean_s']:.3f} | {d['median_s']:.3f} | {d['p95_s']:.3f} |")
    lines+=["",f"Mean cold-cache speedup: {stats['cold_speedup_mean']:.2f}x",
             f"Mean warm-cache speedup: {stats['warm_speedup_mean']:.2f}x",
             "","Cache retains only local video feature signals and metadata; never uploads private media.",
             "This is a system optimization of repeated frame selection, not a new model-accuracy gain.",
             "The single-budget cold-start overhead of examining 48 frames is still substantial."]
    (out/"SHARED_CACHE_REPORT.md").write_text("\n".join(lines)+"\n")
    print("SHARED_CACHE_COMPLETE",json.dumps(stats),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/shared_cache_v1.json")
    args=p.parse_args()
    run(args.config)
