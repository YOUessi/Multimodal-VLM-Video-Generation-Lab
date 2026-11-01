"""Local offline visual audit of new camera/lighting/occlusion motion benchmark.

The review pairs are the first three frozen independent test scene IDs in
each nuisance mode. Not selected based on model or tracker correctness.
All video files remain on Tang. No media or model weights published.
"""
from __future__ import annotations
import argparse,html,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def build(config):
    cfg=json.loads((ROOT/config).read_text(encoding="utf8"))
    manifest=json.loads((ROOT/cfg["dataset_dir"]/"manifest.json").read_text(encoding="utf8"))
    folder=ROOT/cfg["output_dir"]
    cv=json.loads((folder/"cv_predictions.json").read_text(encoding="utf8"))
    byid={r["id"]:r for r in cv["records"]}
    model={}
    for name in cfg["models"]:
        record=json.loads((folder/(name+".json")).read_text(encoding="utf8"))
        if record["source_sha256"]!=manifest["source_sha256"]:
            raise RuntimeError("MODEL_SOURCE_CHANGED")
        model[name]={(r["id"],r["frames"]):r for r in record["records"]}
    entries=[r for r in manifest["rows"] if r["split"]=="test"]
    modes=manifest["modes"]
    cards=[]
    for nuisance in modes:
        originals=[r for r in entries if r["variant"]==nuisance and not r["reversed"]]
        chosen=sorted(originals,key=lambda z:z["pair_id"])[:3]
        cards.append("<h2>"+html.escape(nuisance)+"</h2>")
        for fwd in chosen:
            paired=[r for r in entries if r["pair_id"]==fwd["pair_id"]]
            paired.sort(key=lambda z:z["reversed"])
            pairs=[]
            for row in paired:
                path=ROOT/row["video"]
                if not path.is_file():raise FileNotFoundError(path)
                rel=os.path.relpath(path,folder).replace(os.sep,"/")
                trace=byid[row["id"]]
                opts=" ".join("ABCDE"[i]+". "+value for i,value in enumerate(row["choices"]))
                lines=[]
                for method in ("plain","stabilized"):
                    x=trace[method]
                    lines.append("<tr><td>"+method+"</td><td>"+
                                 html.escape(str(x["direction"]))+"</td><td>"+
                                 html.escape(str(x["delta_xy"] if "delta_xy" in x else x["world_delta_xy"]))+
                                 "</td><td>"+html.escape(str(x["accepted"]))+"</td></tr>")
                for name,budgets in cfg["frame_budgets"].items():
                    for n in budgets:
                        pred=model[name][(row["id"],n)]
                        lines.append("<tr><td>"+name+" "+str(n)+"f</td><td>"+
                                     html.escape(str(pred["prediction"]))+"</td><td colspan='2'>"+
                                     html.escape(pred["raw"][:180])+"</td></tr>")
                pairs.append("<section><h3>"+html.escape(row["id"])+"</h3>"+
                      "<video controls preload='metadata' playsinline loop src='"+
                      html.escape(rel,quote=True)+"'></video>"+
                      "<p>"+html.escape(row["query"])+"</p><p>Options: "+html.escape(opts)+"</p>"+
                      "<p>Reference relative-background motion: "+html.escape(row["gold_direction"])+"</p>"+
                      "<table><thead><tr><th>Method</th><th>Direction/answer</th><th>Delta or raw text</th><th>Gate</th></tr></thead>"+
                      "<tbody>"+"".join(lines)+"</tbody></table></section>")
            cards.append("<article><h3>"+html.escape(fwd["pair_id"])+
                         "</h3><div class='pair'>"+ "".join(pairs)+"</div></article>")
    page="""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Domain-shift motion tracking review</title>
<style>body{font:14px/1.55 system-ui;background:#f5f8fb;margin:0;color:#25394d}
header{padding:30px 5vw;background:#143c5e;color:white}
main{max-width:1360px;margin:0 auto;padding:22px}
article{background:#fff;border:1px solid #e1e8f0;padding:16px;margin:18px 0;border-radius:12px}
.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
.pair section{min-width:0;border:1px solid #e5edf2;padding:14px;border-radius:8px}
video{width:100%;max-width:400px;background:#142434}
table{width:100%;font-size:12px;border-collapse:collapse}td,th{border-bottom:1px solid #e4ebf2;padding:5px;text-align:left}
td{overflow-wrap:anywhere}p{font-size:12px;color:#536f84}
.note{padding:14px;border-radius:8px;background:#fff1d6;color:#765c20}
@media(max-width:900px){.pair{grid-template-columns:1fr}}</style>
</head><body><header><h1>Motion tracking under domain shift</h1><p>Frozen test subset: forward/reverse, lighting, occlusion, camera pan.</p></header>
<main><div class="note">Synthetic 2D backgrounds, not natural camera video. Examples selected by frozen pair ID, not by accuracy. Model raw text and tracker outputs are shown for audit only.</div>"""+''.join(cards)+"""</main></body></html>"""
    folder.mkdir(parents=True,exist_ok=True)
    (folder/"review.html").write_text(page,encoding="utf8")
    print("SHIFT_LOCAL_VIDEO_REVIEW_READY",folder/"review.html","pairs",len(modes)*3,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/domain_shift_motion_v1.json")
    args=p.parse_args()
    build(args.config)
