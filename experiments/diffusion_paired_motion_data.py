"""Deterministic paired synthetic video-caption dataset for actual Diffusion LoRA.

This is intentionally a small TRAINING INFRASTRUCTURE study. The tiny synthetic
motion domain is not a substitute for natural, high-quality videos.
Never write training/validation videos or weights to GitHub.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(2**20),b""):h.update(chunk)
    return h.hexdigest()

def make_specs():
    specs=[]
    colors={"red":(226,43,56),"blue":(47,95,211),"green":(38,150,88)}
    for color in colors:
        for direction in ("right","left"):
            for lane in (0,1):
                specs.append({"id":"train_"+color+"_"+direction+"_"+str(lane),
                    "split":"train","color":color,"rgb":colors[color],
                    "direction":direction,"lane":lane,"distance":42})
    for color in ("red","blue"):
        for direction in ("right","left"):
            specs.append({"id":"val_"+color+"_"+direction,"split":"val",
                "color":color,"rgb":colors[color],
                "direction":direction,"lane":2,"distance":30})
    assert len(specs)==16
    return specs

def render_frames(spec,width=192,height=128,count=4):
    from PIL import Image,ImageDraw
    if width%8 or height%8 or count<3:
        raise ValueError("Resolution must be divisible by 8; at least 3 frames")
    if spec["direction"] not in ("left","right"):
        raise ValueError("Unsupported direction")
    y={0:38,1:65,2:90}[spec["lane"]]
    travel=int(spec["distance"])
    mid=width//2
    start=mid-travel//2 if spec["direction"]=="right" else mid+travel//2
    end=mid+travel//2 if spec["direction"]=="right" else mid-travel//2
    images=[]
    for frame in range(count):
        x=round(start+(end-start)*frame/(count-1))
        im=Image.new("RGB",(width,height),color=(236,236,232))
        pen=ImageDraw.Draw(im)
        pen.rectangle((x-14,y-14,x+14,y+14),fill=tuple(spec["rgb"]))
        images.append(im)
    return images

def caption(spec):
    move="left to right" if spec["direction"]=="right" else "right to left"
    return ("A simple flat animation of a solid "+spec["color"]+
            " square moving "+move+" on a uniform light-gray background.")

def save(config="configs/zeroscope_lora_smoke.json"):
    import cv2,numpy as np
    cfg=json.loads((ROOT/config).read_text(encoding="utf8"))
    out=ROOT/cfg["dataset_dir"]
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for spec in make_specs():
        images=render_frames(spec,cfg["width"],cfg["height"],cfg["frames"])
        file=out/"videos"/spec["split"]/(spec["id"]+".mp4")
        file.parent.mkdir(parents=True,exist_ok=True)
        if not file.exists():
            writer=cv2.VideoWriter(str(file),cv2.VideoWriter_fourcc(*"mp4v"),
                                  cfg["fps"],(cfg["width"],cfg["height"]))
            if not writer.isOpened():raise RuntimeError("Cannot open MP4 output")
            for im in images:
                rgb=np.asarray(im)
                writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            writer.release()
        cap=cv2.VideoCapture(str(file))
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1
        cap.release()
        if count!=cfg["frames"]:raise RuntimeError("Generated MP4 frame count mismatch")
        rows.append({"id":spec["id"],"split":spec["split"],"video":str(file.relative_to(ROOT)),
            "sha256":sha256(file),"caption":caption(spec),
            "color":spec["color"],"direction":spec["direction"],
            "lane":spec["lane"],"distance":spec["distance"]})
    assert len(rows)==16
    train={r["sha256"] for r in rows if r["split"]=="train"}
    val={r["sha256"] for r in rows if r["split"]=="val"}
    if train&val:raise RuntimeError("SYNTHETIC_DIFFUSION_SPLIT_LEAK")
    manifest={"version":1,"source":"Procedurally generated illustrative videos",
        "description":"Twelve 4-frame train and four 4-frame val square motions; NOT natural videos",
        "train_count":12,"val_count":4,"train_val_content_sha_overlap":0,
        "configuration":{"width":cfg["width"],"height":cfg["height"],
                         "frames":cfg["frames"],"fps":cfg["fps"]},
        "samples":rows}
    digest=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    manifest["rows_sha256"]=digest
    dest=out/"manifest.json"
    if dest.exists():
        old=json.loads(dest.read_text(encoding="utf8"))
        if old["rows_sha256"]!=digest:raise RuntimeError("SYNTHETIC_VIDEO_MANIFEST_DRIFT")
    else:
        dest.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("SYNTHETIC_DIFFUSION_PAIRED_DATA_READY",12,4,digest,flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/zeroscope_lora_smoke.json")
    args=p.parse_args()
    save(args.config)

if __name__=="__main__":
    main()
