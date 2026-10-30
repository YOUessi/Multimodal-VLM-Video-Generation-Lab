"""Conservative Wan2.1 1.3B full-weight resource gate. Does NOT download/generate video."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
ROOT=Path(__file__).resolve().parents[1]
def available_ram_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])/1024**2
    raise RuntimeError("RAM info unavailable")
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/round1.json")
    cfg=json.loads((ROOT/ap.parse_args().config).read_text())
    output=ROOT/cfg["output_dir"]
    output.mkdir(parents=True,exist_ok=True)
    p=subprocess.run(["nvidia-smi","--query-gpu=memory.total,memory.free","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=15,check=True)
    total,free=[int(z.strip()) for z in p.stdout.splitlines()[0].split(",")]
    disk=shutil.disk_usage(str(ROOT)).free/1024**3
    ram=available_ram_gib()
    # Actual Hugging Face metadata observed on 2026-10-31, not downloaded.
    weights=26.95
    gates={"disk":disk>=weights+25,"ram_for_fp16_offload":ram>=weights+10,"vram":free>=12000}
    report={"at":datetime.now(timezone.utc).isoformat(),"model":"Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "metadata_based_weights_gib":weights,"free_disk_gib":round(disk,2),
            "available_ram_gib":round(ram,2),"gpu_total_gib":round(total/1024,2),
            "gpu_free_gib":round(free/1024,2),"gates":gates,
            "safe_full_precision_execution":all(gates.values()),
            "status":"PREFLIGHT_ONLY_NO_VIDEO_GENERATED",
            "next":"Design quantized CPU offload or smaller model before full-size Wan download if gates fail"}
    target=output/"wan_preflight.json"
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__":main()
