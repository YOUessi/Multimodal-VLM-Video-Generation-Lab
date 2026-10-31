"""Multi-seed LoRA replication using one FROZEN synthetic video split.

No real MVBench video is used during training. Training seeds affect PEFT
initialization and training example order, NOT the frozen dataset split.
"""
import argparse, hashlib, json, random, shutil, statistics, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SEEDS=(17,101,2027)

def run_config(seed):
    base=json.loads((ROOT/"configs/round2.json").read_text())
    base["train_seed"]=seed
    base["seed"]=20261031
    base["train_steps"]=48
    base["output_dir"]=f"outputs/multi_seed/seed_{seed}"
    out=ROOT/base["output_dir"]
    out.mkdir(parents=True,exist_ok=True)
    path=ROOT/"data"/"multi_seed_configs"/f"seed_{seed}.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    frozen=json.dumps(base,sort_keys=True,indent=2)+"\n"
    if path.exists() and path.read_text()!=frozen:raise RuntimeError("SEED_CONFIG_DRIFT")
    path.write_text(frozen)
    return base,path,out

def run_stage(name,cfgfile):
    command=[sys.executable,str(ROOT/"experiments/temporal_batch.py"),name,
             "--config",str(cfgfile.relative_to(ROOT))]
    print("RUNNING",name,cfgfile.name,flush=True)
    subprocess.run(command,cwd=ROOT,check=True,timeout=1200)

def run():
    ref=ROOT/"outputs/temporal_round2/baseline.json"
    if not ref.is_file():raise RuntimeError("Frozen baseline from batch2 missing")
    baseline=json.loads(ref.read_text())
    fixed=baseline["frozen_dataset_sha"]
    records=[]
    for seed in SEEDS:
        cfg,path,out=run_config(seed)
        base=out/"baseline.json"
        if not base.is_file():
            shutil.copy2(ref,base)
        current=json.loads(base.read_text())
        if current["frozen_dataset_sha"]!=fixed:raise RuntimeError("BASELINE_DATASET_DRIFT")
        train=out/"training.json"
        report=out/"comparison.json"
        if not train.exists() or not (out/"adapter/adapter_model.safetensors").exists():
            run_stage("train",path)
        if not report.exists():
            run_stage("report",path)
        info=json.loads(report.read_text())
        training=json.loads(train.read_text())
        if training["frozen_dataset_sha"]!=fixed:
            raise RuntimeError("TRAINING_DATASET_DRIFT")
        results={}
        for n in cfg["eval_frame_counts"]:
            score=info[str(n)]
            results[str(n)]={"base_correct":score["base"]["correct"],
                             "lora_correct":score["lora"]["correct"],
                             "total":score["base"]["total"],
                             "delta_pp":score["delta_percentage_points"]}
        records.append({"seed":seed,"training_steps":training["steps"],
                        "train_peak_vram_gib":training["peak_gpu_gib"],
                        "metrics":results})
        print("REPLICA_COMPLETE",seed,results,flush=True)
    stats={}
    for n in [1,3,5]:
        vals=[r["metrics"][str(n)]["delta_pp"] for r in records]
        stats[str(n)]={"delta_mean_pp":round(statistics.mean(vals),3),
                       "delta_stddev_pp":round(statistics.stdev(vals),3),
                       "per_seed_deltas_pp":vals}
    summary={"pipeline":"SmolVLM2 synthetic three-scene MP4 LoRA replication",
             "frozen_split_sha256":fixed,"seeds":list(SEEDS),"records":records,
             "stats":stats,"limitation":"Not independent real-world video dataset; seeds only vary model adapter and training order"}
    out=ROOT/"outputs/multi_seed"
    target=out/"REPLICATION.json"
    target.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    lines=["# Synthetic LoRA seed replication","",
           "Do not interpret these values as real-world improvement.",
           "","| Frames | LoRA delta mean (pp) | Stddev | Per-seed changes |",
           "|---|---:|---:|---|"]
    for n,v in stats.items():
        lines.append(f"| {n} | {v['delta_mean_pp']:+.2f} | {v['delta_stddev_pp']:.2f} | {v['per_seed_deltas_pp']} |")
    (out/"REPLICATION.md").write_text("\n".join(lines)+"\n")
    print("REPLICATION_REPORT",target,flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument("action",choices=["run"])
    p.parse_args();run()
if __name__=="__main__":main()
