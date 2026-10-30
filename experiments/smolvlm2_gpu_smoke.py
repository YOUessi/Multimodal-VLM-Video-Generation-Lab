"""Actual GPU SmolVLM2 inference. Synthetic visuals are a smoke test, not research metrics."""
import json, time
from datetime import datetime,timezone
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor,AutoModelForImageTextToText
ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models/SmolVLM2-500M-Video-Instruct'
OUT=ROOT/'outputs/gpu_probe'
OUT.mkdir(parents=True,exist_ok=True)
DTYPE=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
def image(name,color):
    p=OUT/name; im=Image.new('RGB',(320,240),(245,244,241))
    ImageDraw.Draw(im).rectangle((50,35,270,210),fill=color);im.save(p)
    return p
def infer(m,p,question,frames):
    content=[{'type':'text','text':'Answer in English, briefly.'}]
    for i,frame in enumerate(frames):
        content += [{'type':'text','text':f'Frame {i+1} at {2*i:.1f} seconds:'},{'type':'image','path':str(frame)}]
    content.append({'type':'text','text':question})
    messages=[{'role':'user','content':content}]
    inputs=p.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt')
    inputs=inputs.to('cuda',dtype=DTYPE)
    torch.cuda.synchronize(); t=time.perf_counter()
    with torch.inference_mode(): out=m.generate(**inputs,max_new_tokens=48,do_sample=False)
    torch.cuda.synchronize()
    tokens=out[0][inputs['input_ids'].shape[-1]:]
    return {'question':question,'images':[x.name for x in frames],'answer':p.decode(tokens,skip_special_tokens=True).strip(),'seconds':round(time.perf_counter()-t,3),'new_tokens':int(tokens.numel())}
def main():
    if not (MODEL/'model.safetensors').exists():raise RuntimeError('Weights not fully downloaded: '+str(MODEL))
    t=time.perf_counter();print('LOADING MODEL',MODEL,flush=True)
    p=AutoProcessor.from_pretrained(str(MODEL),local_files_only=True)
    m=AutoModelForImageTextToText.from_pretrained(str(MODEL),local_files_only=True,torch_dtype=DTYPE).to('cuda').eval()
    print('MODEL READY',round(time.perf_counter()-t,3),flush=True)
    a,b,c=image('red.png','red'),image('green.png','green'),image('blue.png','blue')
    cases=[('What color is the large rectangle?', [a]),('What color is the large rectangle?', [c]),('Which color appears first and which last?', [a,b,c])]
    results=[]
    for q,frames in cases:
        try:row=infer(m,p,q,frames);results.append(row);print('RESULT',json.dumps(row),flush=True)
        except Exception as e: row={'question':q,'error':repr(e)};results.append(row);print('CASE_ERROR',repr(e),flush=True)
    record={'date_utc':datetime.now(timezone.utc).isoformat(),'model':'HuggingFaceTB/SmolVLM2-500M-Video-Instruct','torch':torch.__version__,'transformers':__import__('transformers').__version__,'device':torch.cuda.get_device_name(0),'dtype':str(DTYPE),'gpu_peak_gib':round(torch.cuda.max_memory_allocated()/1024**3,3),'samples':results,'note':'real local GPU model predictions on synthetic images; not a research accuracy benchmark'}
    file=OUT/'results.json';file.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print('SAVED',str(file),'SUCCESS',sum('answer' in r for r in results),'/',len(results),flush=True)
if __name__=='__main__':main()
