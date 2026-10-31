"""Build a local, offline forward/reverse counterfactual video evidence review.

All clips and raw model outputs remain on Tang. The example subset is the
first eight frozen pair IDs by source order, NOT chosen by accuracy. The HTML
is intentionally a review artifact and is not pushed with private media.
"""
from __future__ import annotations
import argparse,html,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read(rel):
    return json.loads((ROOT/rel).read_text(encoding="utf8"))

def build(cfg):
    dataset=read(Path(cfg["dataset_dir"])/"manifest.json")
    out=ROOT/cfg["output_dir"]
    models={name:read(Path(cfg["output_dir"])/(name+".json"))
            for name in cfg["models"]}
    for m in models.values():
        if m["dataset_manifest_sha"]!=dataset["rows_sha256"]:
            raise RuntimeError("LOCAL_REVIEW_SOURCE_DATA_DRIFT")
    entries=[r for r in dataset["rows"] if r["split"]=="test"]
    order=sorted({r["pair_id"] for r in entries})[:8]
    byid={r["id"]:r for r in entries}
    lookups={name:{(r["id"],r["mode"]):r for r in m["predictions"]}
             for name,m in models.items()}
    blocks=[]
    for pairid in order:
        rows=sorted((r for r in entries if r["pair_id"]==pairid),
                    key=lambda a:a["reversed"])
        cards=[]
        for row in rows:
            full=ROOT/row["video"]
            if not full.is_file():raise FileNotFoundError(full)
            url=os.path.relpath(full,out).replace(os.sep,"/")
            fields=[]
            for modelname in cfg["models"]:
                v=lookups[modelname]
                for mode in cfg["methods"]:
                    p=v[(row["id"],mode)]
                    letter=p["pred_letter"] or "—"
                    fields.append("<tr><td>"+html.escape(modelname)+"</td><td>"+
                      html.escape(mode)+"</td><td>"+html.escape(letter)+"</td>"+
                      "<td>"+html.escape(p["raw"][:160])+"</td></tr>")
            letters=" ".join(f"{'ABCDE'[i]}. {opt}" for i,opt in enumerate(row["options"]))
            heading="REVERSED" if row["reversed"] else "FORWARD"
            card=("<section class='clip'><h3>"+html.escape(row["id"])+" · "+heading+"</h3>"
                "<video controls preload='metadata' loop muted playsinline src='"+
                html.escape(url,quote=True)+"'></video>"
                "<p><strong>Question:</strong> "+html.escape(row["question"])+"</p>"
                "<p><strong>Options:</strong> "+html.escape(letters)+"</p>"
                "<p><strong>Reference:</strong> "+html.escape(row["gold_letter"])+
                " ("+html.escape(row["gold_direction"])+")</p>"
                "<table><thead><tr><th>Model</th><th>Condition</th><th>Answer</th><th>Raw response</th>"
                "</tr></thead><tbody>"+"\n".join(fields)+"</tbody></table></section>")
            cards.append(card)
        blocks.append("<article><h2>"+html.escape(pairid)+"</h2><div class='pair'>"+
                      "".join(cards)+"</div></article>")
    page="""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Counterfactual Motion — Local Evidence Review</title>
<style>body{margin:0;background:#f5f7fa;color:#243647;font:14px/1.55 system-ui,sans-serif}
header{background:#153d61;color:white;padding:28px 5vw}
main{max-width:1350px;margin:auto;padding:24px}
article{border:1px solid #e2eaf2;background:#fff;border-radius:14px;padding:16px;margin:18px 0}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.clip{min-width:0;border:1px solid #e5ebf2;border-radius:8px;padding:14px}
video{width:100%;max-width:380px;background:#10141a;border-radius:8px}
table{width:100%;font-size:12px;border-collapse:collapse}td,th{border-bottom:1px solid #edf1f5;padding:5px;text-align:left;vertical-align:top}
td:last-child{max-width:250px;overflow-wrap:anywhere}p{color:#566b7b;font-size:12px}
.warning{background:#ffefcd;color:#795d24;border-radius:8px;padding:12px}
@media(max-width:820px){.pair{grid-template-columns:1fr}}</style></head><body>
<header><h1>Object-centric motion / VLM evidence</h1><p>Forward / reverse paired synthetic motion videos. Fixed seed, frozen source frames, five VLM input protocols.</p></header>
<main><p class="warning">Research diagnostic. These clips are generated 2D shapes with an oracle-checkable direction, not natural videos. The sample showcase is the first eight pre-frozen pairs, not selected for favorable model performance. Numerical trajectories are measured from actual pixels; human review is not a substitute for an unseen test.</p>"""+''.join(blocks)+"""</main></body></html>"""
    out.mkdir(parents=True,exist_ok=True)
    target=out/"review.html"
    target.write_text(page,encoding="utf8")
    print("COUNTERFACTUAL_LOCAL_VIDEO_REVIEW_READY",target,"pairs",len(order),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/object_motion_v1.json")
    args=p.parse_args()
    build(read(args.config))
