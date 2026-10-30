"""First-person video is processed locally on Tang and never uploaded."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, json, time, cv2, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models/SmolVLM2-500M-Video-Instruct'
OUT=ROOT/'outputs/local_video_probe'
OUT.mkdir(parents=True,exist_ok=True)
DTYPE=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
def get_frames(video:Path):
    capture=cv2.VideoCapture(str(video))
    if not capture.isOpened(): raise RuntimeError('Cannot read local video')
    duration=(capture.get(cv2.CAP_PROP_FRAME_COUNT)/max(capture.get(cv2.CAP_PROP_FPS),1))
    frames=[]
    for idx,frac in enumerate([.05,.25,.50,.75,.95]):
        sec=float(duration*frac)
        capture.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        ok,arr=capture.read()
        if not ok:continue
        image=Image.fromarray(cv2.cvtColor(arr,cv2.COLOR_BGR2RGB))
        image.thumbnail((640,640))
        f=OUT/f'frame_{idx:02d}.jpg'; image.save(f,quality=82)
        frames.append((sec,f))
    capture.release()
    if not frames:raise RuntimeError('No valid frame')
    return duration,frames
def predict(model,processor,frames,question):
    content=[{'type':'text','text':'Use the supplied video frames as evidence. Do not invent actions.'}]
    for sec,path in frames:
        content.append({'type':'text','text':f'Sampled time {sec:.1f}s'})
        content.append({'type':'image','path':str(path)})
    content.append({'type':'text','text':question})
    message=[{'role':'user','content':content}]
    batch=processor.apply_chat_template(message,add_generation_prompt=True,tokenize=True,return_tensors='pt',return_dict=True).to('cuda',dtype=DTYPE)
    torch.cuda.synchronize();a=time.perf_counter()
    with torch.inference_mode():out=model.generate(**batch,do_sample=False,max_new_tokens=120)
    torch.cuda.synchronize()
    result=processor.decode(out[0][batch['input_ids'].shape[-1]:],skip_special_tokens=True).strip()
    return dict(answer=result,seconds=round(time.perf_counter()-a,3),frames=len(frames))
def main():
    parser=argparse.ArgumentParser(description='Evaluate local video without uploading any original media')
    parser.add_argument('--video',required=True,type=Path)
    args=parser.parse_args()
    length,all_frames=get_frames(args.video)
    model=AutoModelForImageTextToText.from_pretrained(str(MODEL),local_files_only=True,dtype=DTYPE).to('cuda').eval()
    proc=AutoProcessor.from_pretrained(str(MODEL),local_files_only=True)
    cases=[
        ('Describe what is visible in this video segment.',all_frames[:1]),
        ('Describe the main visible scenes and how they change across time.',all_frames),
    ]
    results=[]
    for question,frames in cases:
        entry=predict(model,proc,frames,question);entry['question']=question
        print(json.dumps(entry,ensure_ascii=False),flush=True);results.append(entry)
    doc={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'private_video_local_only':True,
         'video_duration_s':round(length,2),'samples':[(round(t,2),p.name) for t,p in all_frames],
         'model':'SmolVLM2-500M-Video-Instruct','results':results,
         'evaluation':'Qualitative unlabeled video probe, not accuracy measurement'}
    file=OUT/'report.json';file.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('SAVED',file,flush=True)
if __name__=='__main__':main()
