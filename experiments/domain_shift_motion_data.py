"""New camera/lighting/occlusion domain-shift video benchmark.

Generates forward/reverse pairs with target movement defined RELATIVE TO THE
STATIONARY WORLD BACKGROUND, not raw screen displacement. In camera_pan,
the camera itself translates: screen target displacement = world motion -
camera/background motion. The paired truth uses world-motion vector only.

DEV/TEST fixed before evaluation, split by independent scene seeds, no
pretrained VLM outputs/answer labels inspected during dataset construction.
Procedural synthetic data, not natural-camera benchmark or broad generality.
"""
from __future__ import annotations
import argparse,hashlib,json,random
from collections import Counter
from pathlib import Path
import cv2,numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODES=("clean","illumination","occlusion","camera_pan")
DIRECTIONS=("stationary","up_right","up_left","down_right","down_left")
DXDY={"stationary":(0,0),"up_right":(1,-1),"up_left":(-1,-1),
      "down_right":(1,1),"down_left":(-1,1)}
OPPOSITE={"stationary":"stationary","up_right":"down_left",
          "up_left":"down_right","down_right":"up_left","down_left":"up_right"}
COLOR_RGB={"red":(226,51,65),"blue":(45,91,209),
           "green":(48,159,86),"cyan":(44,186,201),
           "yellow":(221,186,45),"brown":(141,91,46)}
SHAPES=("circle","square","triangle")
LABELS={
 "stationary":"The target stays stationary relative to the background.",
 "up_right":"Up and to the right.",
 "up_left":"Up and to the left.",
 "down_right":"Down and to the right.",
 "down_left":"Down and to the left."
}
def digest(file):
    h=hashlib.sha256()
    with Path(file).open("rb") as f:
        for chunk in iter(lambda:f.read(2**20),b""):h.update(chunk)
    return h.hexdigest()
def save(path,item):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(item,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    tmp.replace(path)

def texture(height,width,seed):
    gen=np.random.default_rng(seed)
    # A fixed in-world background with high-frequency neutral texture.
    noise=gen.normal(0,19,(height,width)).astype(np.float32)
    gray=np.clip(175+noise,70,236).astype(np.uint8)
    img=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    for x in range(15,width,32):
        cv2.line(img,(x,0),(x,height-1),(155,155,155),1)
    for y in range(11,height,28):
        cv2.line(img,(0,y),(width-1,y),(155,155,155),1)
    rng=random.Random(seed)
    for i in range(80):
        x=rng.randint(3,width-4);y=rng.randint(3,height-4)
        col=rng.choice((110,135,200,228))
        cv2.circle(img,(x,y),rng.choice([1,2]),(col,col,col),-1)
    return img

def draw_object(frame,x,y,color,shape,size=13):
    x,y=int(round(x)),int(round(y))
    bgr=tuple(reversed(COLOR_RGB[color]))
    if shape=="circle":cv2.circle(frame,(x,y),size,bgr,-1)
    elif shape=="square":cv2.rectangle(frame,(x-size,y-size),(x+size,y+size),bgr,-1)
    elif shape=="triangle":
        pts=np.array([(x,y-size-2),(x-size-2,y+size),(x+size+2,y+size)],np.int32)
        cv2.fillPoly(frame,[pts],bgr)
    else:raise ValueError("Unknown primitive")

def scene(seed,variant,direction):
    rng=random.Random(seed)
    color=rng.choice(list(COLOR_RGB))
    other=rng.choice([c for c in COLOR_RGB if c!=color])
    velocity=DXDY[direction]
    other_dir=rng.choice(DIRECTIONS)
    camera=[0,0]
    if variant=="camera_pan":
        camera=[rng.choice([-3,-2,2,3]),rng.choice([-3,-2,2,3])]
    return {"seed":seed,"variant":variant,"direction":direction,
            "target_color":color,"target_shape":rng.choice(SHAPES),
            "distractor_color":other,"distractor_shape":rng.choice(SHAPES),
            "target_center":[rng.randint(112,143),rng.randint(112,141)],
            "distractor_center":[rng.choice([rng.randint(29,65),rng.randint(193,224)]),
                                 rng.choice([rng.randint(27,69),rng.randint(183,226)])],
            "target_speed":rng.choice([4,5,6]),
            "distractor_speed":rng.choice([1,2,3]),
            "distractor_velocity":DXDY[other_dir],
            "world_target_velocity":velocity,"camera_velocity":camera,
            "texture_seed":rng.randrange(2**31)}
def render(params,count,height,width):
    base=texture(height,width,params["texture_seed"])
    frames=[]
    tx,ty=params["world_target_velocity"]
    ox,oy=params["distractor_velocity"]
    cx,cy=params["camera_velocity"]
    for t in range(count):
        camx,camy=cx*t,cy*t
        warped=cv2.warpAffine(base,np.float32([[1,0,-camx],[0,1,-camy]]),
                   (width,height),flags=cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REFLECT_101)
        target_x=params["target_center"][0]+tx*params["target_speed"]*t-camx
        target_y=params["target_center"][1]+ty*params["target_speed"]*t-camy
        dist_x=params["distractor_center"][0]+ox*params["distractor_speed"]*t-camx
        dist_y=params["distractor_center"][1]+oy*params["distractor_speed"]*t-camy
        draw_object(warped,dist_x,dist_y,params["distractor_color"],
                    params["distractor_shape"],11)
        draw_object(warped,target_x,target_y,params["target_color"],
                    params["target_shape"],14)
        if params["variant"]=="occlusion" and 3<=t<=6:
            # A partially occluding neutral foreground patch. The target
            # may temporarily disappear; abstention is a valid result.
            x=int(target_x);y=int(target_y)
            cv2.rectangle(warped,(x-9,y-19),(x+11,y+7),(124,126,129),-1)
        if params["variant"]=="illumination":
            factor=.57+.40*(t/max(count-1,1))
            cast=np.array([6,-9,13],dtype=np.float32)
            warped=np.clip(warped.astype(np.float32)*factor+cast,0,255).astype(np.uint8)
        frames.append(warped)
    return frames

def write_video(path,frames,fps):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if video_frames(path)==len(frames):return
        raise RuntimeError("STALE_VIDEO_BYTES_OR_FRAME_COUNT")
    h,w=frames[0].shape[:2]
    out=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    if not out.isOpened():raise RuntimeError("NO_VIDEO_WRITER")
    for arr in frames:out.write(arr)
    out.release()
    if video_frames(path)!=len(frames):raise RuntimeError("DECODED_FRAMES_MISMATCH")
def video_frames(path):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():return -1
    total=0
    try:
        while True:
            ok,_=cap.read()
            if not ok:break
            total+=1
    finally:cap.release()
    return total

def build(cfg):
    root=ROOT/cfg["dataset_dir"]
    root.mkdir(parents=True,exist_ok=True)
    rows=[];pair_summary=[]
    for split,pairs,seed in (("dev",cfg["dev_pairs"],cfg["dev_seed"]),
                             ("test",cfg["test_pairs"],cfg["test_seed"])):
        if pairs%(len(MODES)*len(DIRECTIONS)):
            raise ValueError("Each split must balance nuisance x direction grid")
        conditions=[(mode,direction) for mode in MODES for direction in DIRECTIONS
                     for _ in range(pairs//(len(MODES)*len(DIRECTIONS)))]
        rng=random.Random(seed);rng.shuffle(conditions)
        for i,(mode,direction) in enumerate(conditions):
            params=scene(rng.randrange(2**31),mode,direction)
            forward=render(params,cfg["frames"],cfg["height"],cfg["width"])
            option_dirs=list(DIRECTIONS)
            rng_opts=random.Random(params["seed"]^0x391AE)
            rng_opts.shuffle(option_dirs)
            pid=f"{split}_{i:04d}"
            for reversed_ in (False,True):
                clip=list(reversed(forward)) if reversed_ else forward
                suffix="reverse" if reversed_ else "forward"
                file=root/"clips"/split/(pid+"_"+suffix+".mp4")
                write_video(file,clip,cfg["fps"])
                gold=OPPOSITE[direction] if reversed_ else direction
                row={"id":pid+"_"+suffix,"pair_id":pid,"split":split,
                     "variant":mode,"reversed":reversed_,
                     "video":str(file.relative_to(ROOT)),
                     "content_sha256":digest(file),
                     "query":f"Relative to the stationary textured background, which direction does the {params['target_color']} {params['target_shape']} move?",
                     "choices":[LABELS[d] for d in option_dirs],
                     "gold_direction":gold,
                     "gold_letter":"ABCDE"[option_dirs.index(gold)],
                     "target_color":params["target_color"],
                     "camera_motion_present":mode=="camera_pan",
                     "frames":cfg["frames"],"source_of_gold":"world velocity, not image pixel screen delta"}
                rows.append(row)
            pair_summary.append({"pair_id":pid,"split":split,"variant":mode,
               "original_direction":direction,"source_seed":params["seed"]})
        print("SHIFT_DATA_GENERATED",split,pairs,"PAIRS",flush=True)
    if len(rows)!=2*(cfg["dev_pairs"]+cfg["test_pairs"]):
        raise RuntimeError("UNEXPECTED_DATASET_TOTAL")
    if len(set(x["id"] for x in rows))!=len(rows):raise RuntimeError("DUPLICATED_ROW_ID")
    dev={x["content_sha256"] for x in rows if x["split"]=="dev"}
    test={x["content_sha256"] for x in rows if x["split"]=="test"}
    if dev&test:raise RuntimeError("DEV_TEST_VIDEO_BYTE_LEAK")
    expected=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    doc={"protocol_version":cfg["version"],"description":"Heldout procedural render-shift benchmark, no model-score tuning",
         "synthetic_only":True,"not_natural_video":True,
         "dev_pairs":cfg["dev_pairs"],"test_pairs":cfg["test_pairs"],
         "dev_test_content_overlap":0,"pair_count":len(pair_summary),
         "total_video_clips":len(rows),"modes":list(MODES),
         "directions":list(DIRECTIONS),"source_sha256":expected,
         "rows":rows,"source_pairs":pair_summary}
    dest=root/"manifest.json"
    if dest.exists():
        old=json.loads(dest.read_text())
        if old["source_sha256"]!=expected:raise RuntimeError("FROZEN_DOMAIN_SHIFT_MANIFEST_DRIFT")
    else:save(dest,doc)
    print("SHIFT_TEST_FROZEN",expected,"VIDEO_CLIPS",len(rows),flush=True)
    return doc

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/domain_shift_motion_v1.json")
    args=p.parse_args()
    cfg=json.loads((ROOT/args.config).read_text())
    build(cfg)

if __name__=="__main__":main()
