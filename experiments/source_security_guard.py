"""Source-only publication guard: never include video, model weights or credentials."""
from __future__ import annotations
import json, subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DENIED_EXT={".mp4",".mov",".avi",".jpg",".jpeg",".png",".safetensors",
            ".pt",".pth",".ckpt",".env",".parquet",".arrow",".sqlite",".db"}
REQUIRED_IGNORE=["data/","outputs/","models/",".venv/","*.safetensors","*.mp4"]
def audit_tracked(paths):
    violations=[]
    for raw in paths:
        path=Path(raw)
        if path.suffix.lower() in DENIED_EXT or path.name.startswith(".env"):
            violations.append(raw)
        if path.parts and path.parts[0] in ("data","outputs","models"):
            violations.append(raw)
    return sorted(set(violations))
def main():
    proc=subprocess.run(["git","ls-files","-z"],cwd=ROOT,
         capture_output=True,check=True,timeout=10)
    tracked=[s.decode() for s in proc.stdout.split(b"\0") if s]
    bad=audit_tracked(tracked)
    ignores=(ROOT/".gitignore").read_text(encoding="utf8")
    missing=[i for i in REQUIRED_IGNORE if i not in ignores]
    if bad or missing:
        raise RuntimeError("PUBLICATION_GUARD_FAILED: "+str({"banned":bad,"missing_ignores":missing}))
    output=ROOT/"outputs/release";output.mkdir(parents=True,exist_ok=True)
    data={"checked_utc":datetime.now(timezone.utc).isoformat(),
        "tracked_files":len(tracked),"forbidden_tracked_files":0,
        "required_ignore_patterns_present":True,
        "weights_or_private_video_committed":False,
        "scope":"Frozen local source snapshot only; does not assert future GitHub security"}
    (output/"SOURCE_PUBLICATION_AUDIT.json").write_text(json.dumps(data,indent=2)+"\n")
    print("SOURCE_SECURITY_PASSED",json.dumps(data),flush=True)
if __name__=="__main__":main()
