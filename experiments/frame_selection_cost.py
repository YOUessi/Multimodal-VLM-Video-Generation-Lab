"""Cost-accounting audit of adaptive versus uniform temporal frame selection.

Measures identical frozen STAR 40-video test IDs, 3/5-frame budgets, both
strategies, repeated alternating order. No use of labels in selection. CPU
times include video seeking and frame decoding. GPU inference times are from
already frozen test logs and are joined by video ID; not rerun or re-tuned.
"""
from __future__ import annotations
import argparse,json,random,statistics,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def percentile(values,q):
    if not values: raise ValueError("Cannot compute empty percentile")
    xs=sorted(float(x) for x in values)
    idx=(len(xs)-1)*q
    low=int(idx);hi=min(len(xs)-1,low+1);weight=idx-low
    return round((1-weight)*xs[low]+weight*xs[hi],6)

def verified_config(cfg):
    manifest=json.loads((ROOT/cfg["star_manifest"]).read_text())
    frozen=json.loads((ROOT/cfg["sampling_plan"]).read_text())
    assert frozen["star_manifest_sha256"]==manifest["sha256"],"STAR_MANIFEST_DRIFT"
    rows={row["source_index"]:row for row in manifest["rows"] if row["split"]=="test"}
    assert len(rows)==40
    assert {x["id"] for x in frozen["records"]}==set(rows)
    return rows,frozen

def audit(cfg):
    from adaptive_temporal_sampling import select_video,load_pil
    from star_real_sft import extract_frames
    rows,plan=verified_config(cfg)
    repeats=int(cfg["repeats"])
    assert 2<=repeats<=5
    trials=[]
    order=random.Random(cfg["order_seed"])
    tasks=[(rep,n,rowid) for rep in range(repeats) for n in (3,5) for rowid in rows]
    order.shuffle(tasks)
    for index,(rep,n,rowid) in enumerate(tasks):
        row=rows[rowid]
        video=ROOT/row["video"]
        def uniform():
            return extract_frames(video,n,row["decoded_frames"])
        def adaptive():
            ids,_=select_video(video,row["decoded_frames"],n,cfg["probe_count"])
            assert ids==next(z for z in plan["records"] if z["id"]==rowid)["frames"][str(n)]["frame_indices"],"SAMPLER_REPRODUCIBILITY_DRIFT"
            return load_pil(video,ids)
        # Alternate execution order to mitigate warm-cache method bias.
        methods=[("uniform",uniform),("adaptive",adaptive)]
        if (rep+index)%2:methods.reverse()
        for name,fn in methods:
            t=time.perf_counter()
            images=fn()
            elapsed=time.perf_counter()-t
            assert len(images)==n and all(image.width>0 for image in images)
            trials.append({"repeat":rep,"frames":n,"id":rowid,"method":name,
                           "wall_seconds":round(elapsed,6)})
        if (index+1)%20==0:
            print("COST_AUDIT",index+1,"/",len(tasks),flush=True)
    out=ROOT/cfg["output_dir"]
    out.mkdir(parents=True,exist_ok=True)
    summary={"rules":"Frozen STAR video set; original label-blind adaptive sampling; no changes to model or prompts",
             "sample_count":40,"repeats":repeats,"method":"wall time includes decoding and CPU motion/hue scoring",
             "by_budget":{},"raw_trials":trials}
    for n in (3,5):
        ss={}
        for label in ("uniform","adaptive"):
            sub=[x["wall_seconds"] for x in trials if x["frames"]==n and x["method"]==label]
            assert len(sub)==40*repeats
            ss[label]={"n":len(sub),"mean_s":round(statistics.mean(sub),6),
                       "median_s":percentile(sub,.5),"p95_s":percentile(sub,.95)}
        ss["relative_overhead_ratio"]=round(ss["adaptive"]["mean_s"]/ss["uniform"]["mean_s"],3)
        summary["by_budget"][str(n)]=ss
    path=out/"selection_cost.json"
    path.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print("CPU_SELECTION_COST_DONE",json.dumps(summary["by_budget"]),flush=True)

def report(cfg):
    out=ROOT/cfg["output_dir"]
    cost=json.loads((out/"selection_cost.json").read_text())
    accuracy=json.loads((ROOT/cfg["adaptive_comparison"]).read_text())
    baseline=json.loads((ROOT/cfg["uniform_base"]).read_text())
    lora=json.loads((ROOT/cfg["uniform_lora"]).read_text())
    adaptbase=json.loads((ROOT/cfg["adaptive_base"]).read_text())
    adaptlora=json.loads((ROOT/cfg["adaptive_lora"]).read_text())
    lines=["# Total cost and quality of temporal sampling","",
           "Frozen 40-video STAR test and identical frame budgets. No test-time tuning or model changes.",
           "CPU selection measured with repeated alternating call order; GPU decoding/serving latency is from prior independent runs.",
           "**Addition of CPU and GPU medians is an approximation, not a new synchronous end-to-end service benchmark.**",
           "","| Frames | Uniform median decode ms | Adaptive median select+decode ms | CPU cost ratio |",
           "|---:|---:|---:|---:|"]
    for n in (3,5):
        x=cost["by_budget"][str(n)]
        lines.append(f"| {n} | {1000*x['uniform']['median_s']:.1f} | {1000*x['adaptive']['median_s']:.1f} | {x['relative_overhead_ratio']:.1f}x |")
    lines+=["","| Model | Frames | Uniform correct | Adaptive correct | Delta pp | 95% paired CI pp |",
            "|---|---:|---:|---:|---:|---|"]
    for v in accuracy["comparisons"]:
        lines.append(f"| {v['model']} | {v['frames']} | {v['uniform']['correct']}/40 | {v['adaptive']['correct']}/40 | {v['delta_pp']:+.2f} | {v['bootstrap_95ci_pp']} |")
    lines+=["","## Limitations",
            "- Adaptive estimates 48 candidate visual frames *per video per frame budget*, so overhead is expected.",
            "- In production, decoding cached candidate frames once across n=3/n=5 can reduce duplicated work but was not tested here.",
            "- Optical-flow and hue-change proxies do not guarantee semantic action understanding.",
            "- Accuracy confidence intervals include zero and do not establish superiority.",
            "- Selection has no access to model answers, QA labels or human scoring."]
    (out/"COST_QUALITY.md").write_text("\n".join(lines)+"\n")
    print("COST_QUALITY_REPORT_READY",out/"COST_QUALITY.md",flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["audit","report"])
    p.add_argument("--config",default="configs/frame_selection_cost.json")
    args=p.parse_args()
    cfg=json.loads((ROOT/args.config).read_text(encoding="utf8"))
    if args.action=="audit":audit(cfg)
    else:report(cfg)

if __name__=="__main__":
    main()
