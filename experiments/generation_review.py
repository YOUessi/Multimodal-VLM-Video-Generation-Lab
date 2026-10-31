"""Blind two-reviewer generation scoring with no fabricated human metrics.

Creates an offline HTML video review tool. Exported CSV can be ingested by the
aggregate subcommand. If no completed independent reviews, no scores are reported.
"""
import argparse, csv, html, json, shutil, subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
ROOT=Path(__file__).resolve().parents[1]

FIELDS=["video_id","reviewer_id","prompt_alignment_1to5","visual_quality_1to5",
        "temporal_coherence_1to5","failure_notes"]

def source_report(path):
    obj=json.loads(path.read_text(encoding="utf8"))
    assert obj["status"]=="REAL_LOCAL_GPU_DIFFUSION_INFERENCE"
    assert obj["samples"],"No generated video"
    return obj

def audit_videos(base,report):
    manifest=[]
    for sample in report["samples"]:
        f=ROOT/sample["output"]
        assert f.is_file(),"Missing generation video: "+str(f)
        p=subprocess.run(["ffprobe","-v","error","-select_streams","v:0",
            "-show_entries","stream=width,height,nb_frames,duration",
            "-of","json",str(f)],capture_output=True,text=True,check=True,timeout=25)
        obj=json.loads(p.stdout)["streams"][0]
        if (int(obj["width"])!=int(sample["width"]) or int(obj["height"])!=int(sample["height"])
            or int(obj["nb_frames"])<int(sample["frames"])):
            raise RuntimeError("Video readback mismatch: "+f.name)
        manifest.append({"id":sample["id"],"path":f,"relative":f.relative_to(base),
                         "prompt":sample["prompt"],"seconds":float(sample["elapsed_sec"]),
                         "memory_gib":float(sample["gpu_peak_gib"]),
                         "frames":int(obj["nb_frames"])})
    return manifest

def prepare(path):
    report=source_report(path)
    base=path.parent
    videos=audit_videos(base,report)
    cards=[]
    for v in videos:
        path=v["relative"].as_posix()
        cards.append("""<article class="card"><h2>Sample """+html.escape(v["id"])+"""</h2>
<video controls playsinline preload="metadata" src='"""+html.escape(path,quote=True)+"""'></video>
<p><strong>Text prompt:</strong> """+html.escape(v["prompt"])+"""</p>
<p><small>"""+str(v["frames"])+""" frames · Diffusion time """+str(v["seconds"])+"""s · peak """+str(v["memory_gib"])+""" GiB</small></p>
<div class="review" data-id='"""+html.escape(v["id"],quote=True)+"""'>
<label>Reviewer ID <input class="reviewer" placeholder="reviewer_1" required></label>
<label>Prompt alignment (1–5) <select class="prompt"><option value="">Not scored</option>"""+''.join(f'<option value="{i}">{i}</option>' for i in range(1,6))+"""</select></label>
<label>Visual quality (1–5) <select class="visual"><option value="">Not scored</option>"""+''.join(f'<option value="{i}">{i}</option>' for i in range(1,6))+"""</select></label>
<label>Temporal coherence (1–5) <select class="temporal"><option value="">Not scored</option>"""+''.join(f'<option value="{i}">{i}</option>' for i in range(1,6))+"""</select></label>
<label>Observed failures <textarea class="notes" rows="2" placeholder="Describe visible defects; do not guess scores"></textarea></label>
</div></article>""")
    page="""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Offline blind video review</title>
<style>body{font:15px/1.6 system-ui;max-width:1000px;margin:32px auto;padding:0 18px;background:#f6f8fa;color:#263544}
h1{font-size:24px}.card{background:#fff;border:1px solid #dfe8f0;border-radius:12px;margin:18px 0;padding:18px}
video{display:block;width:100%;max-width:576px;background:#000}label{display:block;margin:10px 0}input,select,textarea{padding:8px;border:1px solid #cbd7e0;border-radius:6px}input{min-width:200px}
button{background:#235a89;color:#fff;border:0;border-radius:9px;padding:11px 18px;cursor:pointer}
small{color:#687987}.notice{background:#fff4df;padding:14px;border-radius:8px}</style></head><body>
<h1>Blind review of locally generated videos</h1>
<p class="notice">No scores are pre-filled. Each reviewer must score independently.
A downloadable CSV contains only the entered ratings. Please use different reviewer IDs.
Generated local research samples only, not proof of SOTA generation quality.</p>"""+'\n'.join(cards)+"""
<button id="export">Export reviewer CSV</button><p id="msg"></p>
<script>
document.querySelector('#export').addEventListener('click',()=>{
const rows=[['video_id','reviewer_id','prompt_alignment_1to5','visual_quality_1to5','temporal_coherence_1to5','failure_notes']];
for(const panel of document.querySelectorAll('.review')){
const get=(cls)=>panel.querySelector('.'+cls).value.trim();
if(!get('reviewer')||!get('prompt')||!get('visual')||!get('temporal')){
document.querySelector('#msg').textContent='Complete reviewer ID and 3 ratings for every video.';return;}
rows.push([panel.dataset.id,get('reviewer'),get('prompt'),get('visual'),get('temporal'),get('notes')]);}
const esc=(v)=>'"'+String(v).replaceAll('"','""')+'"';
const csv=rows.map(row=>row.map(esc).join(',')).join('\\r\\n');
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
a.download='blind_video_review.csv';a.click();URL.revokeObjectURL(a.href);
document.querySelector('#msg').textContent='CSV exported; obtain a second independent reviewer.';
});</script></body></html>"""
    (base/"review.html").write_text(page,encoding="utf8")
    print("VIDEO_REVIEW_READY",base/"review.html","VIDEO_COUNT",len(videos),flush=True)

def aggregate(path,review_files):
    report=source_report(path)
    expected={x["id"] for x in report["samples"]}
    rows=[];seen=set()
    for f in review_files:
        with f.open(newline="",encoding="utf8") as h:
            reader=csv.DictReader(h)
            if reader.fieldnames!=FIELDS:raise ValueError("Wrong CSV contract: "+str(f))
            for row in reader:
                if row["video_id"] not in expected:raise ValueError("Unrecognized sample")
                key=(row["video_id"],row["reviewer_id"])
                if not row["reviewer_id"] or key in seen:raise ValueError("Reviewer missing or duplicate")
                seen.add(key)
                for field in FIELDS[2:5]:
                    if row[field] not in ("1","2","3","4","5"):raise ValueError("Invalid/unscored "+field)
                rows.append(row)
    grouped=defaultdict(list)
    for r in rows:grouped[r["video_id"]].append(r)
    if any(len(grouped[x])<2 for x in expected):
        raise RuntimeError("NEEDS_TWO_INDEPENDENT_REVIEWERS_PER_VIDEO")
    scores={}
    for vid in sorted(expected):
        group=grouped[vid]
        scores[vid]={field:round(mean(int(z[field]) for z in group),3) for field in FIELDS[2:5]}
        scores[vid]["reviewer_count"]=len(group)
        scores[vid]["max_temporal_disagreement"]=max(int(z[FIELDS[4]]) for z in group)-min(int(z[FIELDS[4]]) for z in group)
    obj={"reviewed":True,"samples":scores,"method":"2+ independent human scores; unscored videos rejected"}
    (path.parent/"human_scores.json").write_text(json.dumps(obj,indent=2)+"\n")
    print("VERIFIED_HUMAN_SCORES",json.dumps(obj),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["prepare","aggregate"])
    p.add_argument("--report",default="outputs/diffusion_round1/generation_report.json")
    p.add_argument("--reviews",nargs="*",default=[])
    args=p.parse_args()
    report=ROOT/args.report
    if args.action=="prepare":prepare(report)
    else:aggregate(report,[Path(x) for x in args.reviews])
if __name__=="__main__":main()
