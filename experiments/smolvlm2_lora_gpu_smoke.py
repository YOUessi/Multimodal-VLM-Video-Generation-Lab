"""Two-step LoRA smoke test on synthetic image QA. NOT an empirical VLM improvement claim."""
import json, math, time
from datetime import datetime, timezone
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, AutoModelForImageTextToText
ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models/SmolVLM2-500M-Video-Instruct'
OUT=ROOT/'outputs/lora_probe'
OUT.mkdir(parents=True, exist_ok=True)
DTYPE=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
torch.manual_seed(42)
def batch_for(processor,image_path:Path,answer:str):
    from PIL import Image
    question='What is the color of the large rectangle? Answer in one word.'
    prompt=[{'role':'user','content':[{'type':'image','path':str(image_path)},{'type':'text','text':question}]}]
    full=prompt+[{'role':'assistant','content':[{'type':'text','text':answer}]}]
    # Compute prompt length separately to avoid training on image and user tokens.
    p=processor.apply_chat_template(prompt,add_generation_prompt=True,tokenize=True,return_tensors='pt',return_dict=True)
    f=processor.apply_chat_template(full,add_generation_prompt=False,tokenize=True,return_tensors='pt',return_dict=True)
    labels=f['input_ids'].clone()
    labels[:, :min(p['input_ids'].shape[-1],labels.shape[-1]-1)] = -100
    batch=f.to('cuda',dtype=DTYPE)
    batch['labels']=labels.to('cuda')
    return batch
def main():
    torch.cuda.empty_cache()
    processor=AutoProcessor.from_pretrained(str(MODEL),local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(str(MODEL),local_files_only=True,torch_dtype=DTYPE).to('cuda')
    config=LoraConfig(r=4,lora_alpha=8,lora_dropout=0.05,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')
    model=get_peft_model(model,config)
    model.train(); model.print_trainable_parameters()
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4)
    sample_root=ROOT/'outputs/gpu_probe'
    images=[(sample_root/'red.png','red'),(sample_root/'blue.png','blue')]
    if not all(p.exists() for p,a in images):raise FileNotFoundError('Run smolvlm2_gpu_smoke.py first')
    records=[]
    for step in range(2):
        image,answer=images[step]
        batch=batch_for(processor,image,answer)
        optimizer.zero_grad(set_to_none=True)
        t=time.perf_counter()
        out=model(**batch)
        loss=out.loss
        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss: {loss.item()}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),max_norm=1.0)
        optimizer.step()
        torch.cuda.synchronize()
        row={'step':step+1,'loss':float(loss.detach().cpu()),'seconds':round(time.perf_counter()-t,3),'image':image.name}
        records.append(row);print('STEP',json.dumps(row),flush=True)
    adapter=OUT/'adapter';adapter.mkdir(exist_ok=True)
    model.save_pretrained(str(adapter))
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'model':'SmolVLM2-500M-Video-Instruct','type':'LoRA SFT pipeline smoke on synthetic samples','steps':records,'adapter_path':str(adapter),'peak_gpu_gib':round(torch.cuda.max_memory_allocated()/1024**3,3),'evaluation':'Not evaluated on real/heldout data; no performance claims'}
    path=OUT/'training_report.json';path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print('TRAINING_REPORT',path,'SUCCESS',flush=True)
if __name__=='__main__':main()
