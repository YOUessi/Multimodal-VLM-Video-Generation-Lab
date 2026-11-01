"""Task-matched point/absence endpoint probe of local SmolVLM2-500M.

The model gets a red-cross query-frame image and the last frame, not gold.
It must return one physical-point (x,y) or ABSENT; malformed replies abstain.
This matches the *output label space* of tracking, NOT the CPU evidence budget.
GPU is admitted exclusively by automation/runner.py's common file lock.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image, ImageDraw

import tapvid_rgb_data as src
import tapvid_rgb_eval as ev

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/tapvid_rgb_stacking_v1/batch38_frozen"
OUT = ROOT / "outputs/tapvid_rgb_stacking_v1"
MODEL = ROOT / "models/SmolVLM2-500M-Video-Instruct"
QUERIES_PER_SCENE = 2
MAX_NEW_TOKENS = 32
PROMPT = (
    "Image A is a video query frame: a red cross marks a single material point. "
    "Image B is a later frame. Find the EXACT SAME point on the same physical "
    "surface, NOT just the center of a similar-looking object. If that exact "
    "point is occluded or outside the frame, output ABSENT. Otherwise output "
    "its x,y integer pixel coordinates in Image B (each 0 through 255). "
    "Your ENTIRE reply must be either ABSENT or two integers as x,y. "
    "No prose, labels, explanation or extra punctuation."
)


def strict_point(text: str, width: int, height: int):
    t = text.strip()
    if t == "ABSENT":
        return None, "explicit_absent"
    m = re.fullmatch(r"([0-9]{1,4})\s*,\s*([0-9]{1,4})", t)
    if m:
        xy = [int(m.group(1)), int(m.group(2))]
        if 0 <= xy[0] < width and 0 <= xy[1] < height:
            return xy, "valid_xy"
    return None, "invalid_format"


def query_image(frame: np.ndarray, xy: list):
    img = Image.fromarray(frame, "RGB")
    draw = ImageDraw.Draw(img)
    x,y = [float(v) for v in xy]
    draw.line((x-7,y,x+7,y), fill=(255,0,0), width=3)
    draw.line((x,y-7,x,y+7), fill=(255,0,0), width=3)
    draw.ellipse((x-2,y-2,x+2,y+2),fill=(255,255,0))
    return img


def infer(model, processor, img0: Image.Image, img1: Image.Image, *, dtype):
    import torch
    messages = [{"role":"user","content":[
        {"type":"text","text":"Image A (marked query point):"},
        {"type":"image","path":str(img0)},
        {"type":"text","text":"Image B (target):"},
        {"type":"image","path":str(img1)},
        {"type":"text","text":PROMPT},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to("cuda",dtype=dtype)
    torch.cuda.synchronize()
    start=time.perf_counter()
    with torch.inference_mode():
        outputs=model.generate(**inputs,max_new_tokens=MAX_NEW_TOKENS,do_sample=False)
    torch.cuda.synchronize()
    return (processor.decode(outputs[0][inputs["input_ids"].shape[-1]:],
                             skip_special_tokens=True).strip(),
            round(time.perf_counter()-start,3))


def prepare_images(frames, q, outdir: Path, scene: str):
    outdir.mkdir(parents=True,exist_ok=True)
    t = int(q["query_time"])
    image_a=query_image(frames[t],q["query_xy"])
    image_b=Image.fromarray(frames[-1],"RGB")
    a=outdir/(scene+"_"+q["id"]+"_query.png")
    b=outdir/(scene+"_"+q["id"]+"_target.png")
    image_a.save(a)
    if not b.is_file():
        image_b.save(b)
    return a,b


def predict(split: str):
    import torch
    from transformers import AutoProcessor,AutoModelForImageTextToText
    man=src.audit()
    qdoc=ev.doc(DATA/"queries-public.json")
    OUT.mkdir(parents=True,exist_ok=True)
    outfile=OUT/f"{split}_vlm_predictions.json"
    if outfile.exists():
        ev.require_pin(outfile)
        saved=ev.doc(outfile)
        if saved.get("source_commit")!=ev.gitsha():
            raise RuntimeError("VLM_STALE_SOURCE_SHA")
        print("VLM_PREEXISTING_PIN_OK",split,flush=True)
        return
    if not (MODEL/"model.safetensors").is_file():
        raise RuntimeError("LOCAL_FROZEN_VLM_CHECKPOINT_MISSING")
    model_digest=src.content_key(MODEL/"model.safetensors")
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor=AutoProcessor.from_pretrained(str(MODEL),local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(
        str(MODEL),local_files_only=True,torch_dtype=dtype).to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    records=[]
    imgdir=OUT/"vlm_private_query_images"
    for s in man["scenes"]:
        if s["split"]!=split:
            continue
        frames=np.load(DATA/s["frames"],mmap_mode="r",allow_pickle=False)
        selected=qdoc[s["name"]][:QUERIES_PER_SCENE]
        for q in selected:
            if q["query_time"]>=len(frames)-1:
                print("VLM_LAST_QUERY_NO_TARGET",s["name"],q["id"],flush=True)
                continue
            a,b=prepare_images(frames,q,imgdir,s["name"])
            answer,seconds=infer(model,processor,a,b,dtype=dtype)
            xy,kind=strict_point(answer,s["size"][1],s["size"][0])
            records.append({
                "name":s["name"],"query_id":q["id"],"target_time":len(frames)-1,
                "reply":answer,"format":kind,"xy":xy,"inference_seconds":seconds,
            })
            print("VLM_ENDPOINT_DONE",split,s["name"],q["id"],"format",kind,
                  "seconds",seconds,flush=True)
    ev.once(outfile,{
        "dataset":"TAP-Vid RGB-Stacking synthetic robotics",
        "split":split,"source_commit":ev.gitsha(),
        "manifest_sha256":ev.sha_bytes((DATA/"manifest.json").read_bytes()),
        "public_query_sha256":ev.sha_bytes((DATA/"queries-public.json").read_bytes()),
        "model_id":"HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "model_weights_sha256":model_digest,
        "prompt_sha256":hashlib.sha256(PROMPT.encode()).hexdigest(),
        "label_space":"point-coordinate-or-ABSENT, final-frame endpoint",
        "visual_input":"query frame with cross and endpoint RGB only",
        "oracle_never_read":True,
        "nqueries_per_scene":QUERIES_PER_SCENE,
        "peak_gpu_gib":round(torch.cuda.max_memory_allocated()/1024**3,3),
        "records":records,
    })
    ev.pin(outfile)
    print("VLM_ENDPOINT_PINNED",split,len(records),ev.sha_bytes(outfile.read_bytes()),flush=True)


def score(split: str):
    man=src.audit()
    gold=ev.doc(DATA/"oracle-private.json")
    queries=ev.doc(DATA/"queries-public.json")
    vfile=OUT/f"{split}_vlm_predictions.json"
    cfile=OUT/f"{split}_predictions.json"
    ev.require_pin(vfile)
    ev.require_pin(cfile)
    vlm=ev.doc(vfile)
    cpu=ev.doc(cfile)
    expected_manifest=ev.sha_bytes((DATA/"manifest.json").read_bytes())
    expected_public=ev.sha_bytes((DATA/"queries-public.json").read_bytes())
    if (vlm.get("source_commit")!=ev.gitsha()
            or cpu.get("source_commit")!=ev.gitsha()
            or vlm.get("manifest_sha256")!=expected_manifest
            or cpu.get("manifest_sha256")!=expected_manifest
            or vlm.get("public_query_sha256")!=expected_public
            or vlm.get("oracle_never_read") is not True
            or vlm.get("prompt_sha256")!=hashlib.sha256(PROMPT.encode()).hexdigest()):
        raise RuntimeError("VLM_SCORE_SOURCE_OR_LABEL_FIREWALL_DRIFT")
    names={s["name"] for s in man["scenes"] if s["split"]==split}
    cpu_by_name={r["name"]:r for r in cpu["records"]}
    if set(cpu_by_name)!=names:
        raise RuntimeError("VLM_CPU_SCENE_MISMATCH")
    counters={m:{"visible":0,"absent":0,"correct8":0,"visible_accepted":0,
                  "absent_false_accepted":0,"format_invalid":0}
              for m in ["vlm_500m_endpoint"]+list(ev.METHODS)}
    touched=set()
    for row in vlm["records"]:
        key=(row["name"],row["query_id"])
        if key in touched or row["name"] not in names:
            raise RuntimeError("VLM_DUPLICATE_OR_UNKNOWN_QUERY")
        touched.add(key)
        q=next(q for q in queries[row["name"]][:QUERIES_PER_SCENE] if q["id"]==row["query_id"])
        track=gold[row["name"]]["tracks"][row["query_id"]]
        target=len(track["occluded"])-1
        if row["target_time"]!=target or q["query_time"]>=target:
            raise RuntimeError("VLM_ENDPOINT_TIME_DRIFT")
        visible=not track["occluded"][target]
        proposals={"vlm_500m_endpoint":row["xy"]}
        for m in ev.METHODS:
            proposals[m]=cpu_by_name[row["name"]]["tracks"][row["query_id"]][m][target]
        for method,xy in proposals.items():
            c=counters[method]
            c["visible" if visible else "absent"]+=1
            c["visible_accepted"]+=int(visible and xy is not None)
            c["absent_false_accepted"]+=int(not visible and xy is not None)
            c["format_invalid"]+=int(method=="vlm_500m_endpoint" and row["format"]=="invalid_format")
            if visible and xy is not None:
                error=float(np.linalg.norm(np.subtract(xy,track["xy"][target])))
                c["correct8"]+=int(error<=8)
    for m,c in counters.items():
        c["pck8_visible"]=round(c["correct8"]/c["visible"],5) if c["visible"] else None
        c["false_acceptance_absent"]=round(c["absent_false_accepted"]/c["absent"],5) if c["absent"] else None
    result={
        "source_commit":ev.gitsha(),"split":split,
        "dataset":"TAP-Vid RGB-Stacking synthetic robotics",
        "endpoint_query_count":len(touched),"nscenes":len(names),
        "source_prediction_sha256":ev.sha_bytes(vfile.read_bytes()),
        "cpu_prediction_sha256":ev.sha_bytes(cfile.read_bytes()),
        "vlm_checkpoint_sha256":vlm["model_weights_sha256"],
        "methods":counters,"limitations":[
            "Same physical point/ABSENT output label space, different temporal evidence budget.",
            "SmolVLM2 500M receives only initial query and endpoint image, CPU tracks all frames.",
            "A cross on initial frame conveys pixel query only, not future identity annotation.",
            "No training, no DEV tuning and no hyperparameter changes using TEST outcomes.",
        ],
    }
    ev.once(OUT/f"{split}_vlm_results.json",result)
    print("VLM_ENDPOINT_SCORED",split,json.dumps(counters,sort_keys=True),flush=True)


def main():
    a=argparse.ArgumentParser()
    a.add_argument("action",choices=("predict-dev","score-dev","predict-test","score-test"))
    cfg=a.parse_args()
    action,split=cfg.action.split("-")
    if action=="predict":
        predict(split)
    else:
        score(split)


if __name__=="__main__":
    main()
