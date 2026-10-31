"""Freeze genuinely NEW paired synthetic motion clips with distractor objects.

A single base scene yields forward and reversed 10-frame MP4s. The correct
direction is defined by the target object's actual trajectory; its answer
reverses under temporal reversal. Distractors move independently. All color
centroids are measured from rendered PIXELS, not copied from annotations.
This controlled synthetic dataset is NOT natural-world generalization.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from collections import Counter
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "red": (230, 52, 66), "blue": (48, 94, 220),
    "green": (55, 165, 81), "purple": (157, 72, 190),
    "cyan": (36, 194, 208), "orange": (233, 135, 44),
}
SHAPES = ("sphere", "cube", "triangle")
DIRECTIONS = ("stationary", "up_right", "up_left", "down_right", "down_left")
LABELS = {
    "stationary": "The target object is stationary.",
    "up_right": "Up and to the right.",
    "up_left": "Up and to the left.",
    "down_right": "Down and to the right.",
    "down_left": "Down and to the left.",
}
VECTORS = {"stationary": (0, 0), "up_right": (1,-1), "up_left": (-1,-1),
           "down_right": (1,1), "down_left": (-1,1)}
OPPOSITE = {"stationary":"stationary","up_right":"down_left",
            "up_left":"down_right","down_right":"up_left",
            "down_left":"up_right"}

def sha256(path: Path) -> str:
    d=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(2**20),b""):
            d.update(block)
    return d.hexdigest()

def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    temp.replace(path)

def draw_shape(img,shape,x,y,rgb,size=13):
    bgr=tuple(reversed(rgb))
    x=int(round(x));y=int(round(y))
    if shape=="sphere":
        cv2.circle(img,(x,y),size,bgr,thickness=-1,lineType=cv2.LINE_8)
    elif shape=="cube":
        cv2.rectangle(img,(x-size,y-size),(x+size,y+size),bgr,
                      thickness=-1,lineType=cv2.LINE_8)
    elif shape=="triangle":
        vertices=np.array([(x,y-size-3),(x-size-2,y+size),
                           (x+size+2,y+size)],dtype=np.int32)
        cv2.fillPoly(img,[vertices],bgr,lineType=cv2.LINE_8)
    else:raise ValueError("Unknown shape")

def render_scene(scene,frame_count,height,width):
    frames=[]
    target_color=COLORS[scene["target_color"]]
    distractor_color=COLORS[scene["distractor_color"]]
    dx,dy=VECTORS[scene["direction"]]
    mx,my=VECTORS[scene["distractor_direction"]]
    for t in range(frame_count):
        image=np.full((height,width,3),(238,238,238),dtype=np.uint8)
        # Static neutral landmarks. These do not encode the target direction.
        cv2.line(image,(12,20),(width-12,20),(214,214,214),1)
        cv2.line(image,(12,height-19),(width-12,height-19),(214,214,214),1)
        x=scene["target_center"][0]+dx*scene["speed"]*t
        y=scene["target_center"][1]+dy*scene["speed"]*t
        other_x=scene["distractor_center"][0]+mx*scene["distractor_speed"]*t
        other_y=scene["distractor_center"][1]+my*scene["distractor_speed"]*t
        draw_shape(image,scene["distractor_shape"],other_x,other_y,distractor_color,size=12)
        draw_shape(image,scene["target_shape"],x,y,target_color,size=14)
        frames.append(image)
    return frames

def make_scene(seed,direction,height,width):
    rng=random.Random(seed)
    color=rng.choice(list(COLORS))
    other=rng.choice([c for c in COLORS if c!=color])
    shape=rng.choice(SHAPES)
    other_shape=rng.choice(SHAPES)
    # The target is always clearly visible with large margins, distinct hue
    # and independent distractor motion. No label is drawn on pixels.
    return {
        "seed":seed, "direction":direction,
        "target_color":color, "target_shape":shape,
        "distractor_color":other,"distractor_shape":other_shape,
        "target_center":[rng.randint(92,132),rng.randint(91,128)],
        "distractor_center":[rng.randint(24,55),rng.randint(27,56)],
        "speed":rng.choice([4,5,6]),
        "distractor_speed":rng.choice([0,1,2]),
        "distractor_direction":rng.choice(DIRECTIONS)
    }

def write_mp4(file,frames,fps):
    file.parent.mkdir(parents=True,exist_ok=True)
    h,w=frames[0].shape[:2]
    if file.exists():
        cap=cv2.VideoCapture(str(file))
        decoded=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1
        cap.release()
        if decoded!=len(frames):raise RuntimeError("STALE_VIDEO_FRAME_COUNT")
        return
    writer=cv2.VideoWriter(str(file),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    if not writer.isOpened():raise RuntimeError("VIDEO_WRITER_NOT_OPENED")
    for image in frames:writer.write(image)
    writer.release()
    cap=cv2.VideoCapture(str(file))
    if not cap.isOpened():raise RuntimeError("MP4_CANNOT_BE_DECODED")
    decoded=0
    try:
        while True:
            ok,_=cap.read()
            if not ok:break
            decoded+=1
    finally:cap.release()
    if decoded!=len(frames):raise RuntimeError("MP4_INCOMPLETE_DECODE")

def make_dataset(cfg):
    root=ROOT/cfg["dataset_dir"];root.mkdir(parents=True,exist_ok=True)
    final=root/"manifest.json"
    train_count=cfg["dev_pairs"];test_count=cfg["test_pairs"]
    if train_count%5 or test_count%5:raise ValueError("Must balance five movements")
    width,height=cfg["width"],cfg["height"]
    rows=[];all_hashes=set()
    for split,count in (("dev",train_count),("test",test_count)):
        order=list(DIRECTIONS)*(count//5)
        rng=random.Random(cfg["dev_seed"] if split=="dev" else cfg["test_seed"])
        rng.shuffle(order)
        for i,direction in enumerate(order):
            scene_seed=rng.randrange(1,2**31)
            scene=make_scene(scene_seed,direction,height,width)
            imgs=render_scene(scene,cfg["frames"],height,width)
            # One shuffled option order for BOTH counterfactual variants.
            opts=list(DIRECTIONS)
            random.Random(scene_seed^0x535412).shuffle(opts)
            # When reversed, the source target moves in the opposite direction;
            # labels are computed from the counterfactual generated video.
            for reverse in (False,True):
                order_images=list(reversed(imgs)) if reverse else imgs
                name=f"{split}_{i:04d}"+("_reverse" if reverse else "_forward")
                file=root/"videos"/split/(name+".mp4")
                write_mp4(file,order_images,cfg["fps"])
                unique=sha256(file)
                # A stationary target may still have a moving distractor;
                # video bytes differ except some explicit stationary fixtures.
                if unique in all_hashes:
                    # We preserve a duplicated stationary pair as two distinct
                    # counterfactual conditions, but never use this to train.
                    if direction!="stationary":
                        raise RuntimeError("UNEXPECTED_NONSTATIONARY_VIDEO_DUPLICATE")
                all_hashes.add(unique)
                gold_direction=OPPOSITE[direction] if reverse else direction
                row={"id":name,"pair_id":f"{split}_{i:04d}","split":split,
                    "reversed":reverse,"video":str(file.relative_to(ROOT)),
                    "video_sha256":unique,
                    "question":f"What direction is the {scene['target_color']} {scene['target_shape']} moving in the video?",
                    "target_name":f"{scene['target_color']} {scene['target_shape']}",
                    "options":[LABELS[k] for k in opts],
                    "gold_direction":gold_direction,
                    "gold_letter":"ABCDE"[opts.index(gold_direction)],
                    "direction_source":"generated video trajectory, not output of tracker",
                    "scene_spec":scene if split=="dev" else None}
                rows.append(row)
        print("FROZEN_SCENES",split,count,"PAIRS",flush=True)
    assert len(rows)==2*(train_count+test_count)
    dev={r["video_sha256"] for r in rows if r["split"]=="dev"}
    test={r["video_sha256"] for r in rows if r["split"]=="test"}
    if dev&test:raise RuntimeError("DEV_TEST_EXACT_VIDEO_CONTENT_LEAK")
    digest=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    expected={
      "protocol":"Independent deterministic synthetic counterfactual motion clips",
      "camera":"fixed, no camera-motion robustness claim",
      "source":"new locally generated 2D colored shapes, not camera footage",
      "config_version":cfg["version"],
      "dev_pairs":train_count,"test_pairs":test_count,
      "direction_classes":list(DIRECTIONS),
      "frame_count":cfg["frames"],"width":width,"height":height,
      "video_count":len(rows),"rows_sha256":digest,
      "dev_test_sha256_overlap_count":0,
      "source_metadata_never_used_as_VLM_or_tracker_input":True,
      "rows":rows
    }
    if final.is_file():
        old=json.loads(final.read_text(encoding="utf8"))
        if old["rows_sha256"]!=digest:
            raise RuntimeError("FROZEN_MOTION_MANIFEST_DRIFT")
    else:atomic(final,expected)
    print("NEW_PAIRED_VIDEO_MANIFEST_SHA",digest,"TOTAL_VIDEOS",len(rows),flush=True)
    return expected

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/object_motion_v1.json")
    args=p.parse_args()
    cfg=json.loads((ROOT/args.config).read_text(encoding="utf8"))
    make_dataset(cfg)

if __name__=="__main__":main()
