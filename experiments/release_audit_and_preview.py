"""Evidence-first final project release audit and offline HTML video preview.

Never uploads original videos, generated videos, model weights, or LoRA adapters.
The report uses only the corrected grouped STAR result, external-scene test,
content-parity cache run, and pinned six-video ZeroScope generation.
"""
from __future__ import annotations
import argparse,collections,hashlib,html,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def get(relative):
    file=ROOT/relative
    if not file.is_file():raise FileNotFoundError(str(file))
    return json.loads(file.read_text(encoding="utf-8"))
def full_sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda:source.read(2**20),b""):digest.update(chunk)
    return digest.hexdigest()
def checked_manifest(manifest):
    if len(manifest.get("rows",[]))!=200:raise RuntimeError("GROUPED_STAR_MISSING_ROWS")
    sha=hashlib.sha256(json.dumps(manifest["rows"],sort_keys=True).encode()).hexdigest()
    if sha!=manifest["rows_sha256"]:raise RuntimeError("GROUPED_MANIFEST_DRIFT")
    content=collections.defaultdict(set)
    for row in manifest["rows"]:
        if full_sha(ROOT/row["video"])!=row["content_sha256"]:
            raise RuntimeError("LIVE_VIDEO_CONTENT_DRIFT: "+str(row["video"]))
        content[row["content_sha256"]].add(row["split"])
    if len(content)!=193 or any(len(s)!=1 for s in content.values()):
        raise RuntimeError("STAR_CONTENT_SPLIT_LEAK")
    counts=collections.Counter(x["split"] for x in manifest["rows"])
    if counts!={"train":120,"val":40,"test":40}:
        raise RuntimeError("STAR_COUNTS_DRIFT")
    return {"unique_full_sha256_contents":len(content),
            "duplicate_content_groups":sum(len(c)>1 for c in
                  [ [x for x in manifest["rows"] if x["content_sha256"]==h] for h in content ]),
            "cross_split_sha_overlap":0,"train_val_test":dict(counts)}

def verify_generation(generation):
    import cv2
    if generation.get("status")!="REAL_LOCAL_GPU_DIFFUSION_INFERENCE":
        raise RuntimeError("NO_REAL_GENERATION")
    rows=generation.get("samples",[])
    if len(rows)!=6:raise RuntimeError("EXPECTED_6_GENERATED_VIDEO_RECORDS")
    for x in rows:
        path=ROOT/x["output"]
        if not path.is_file():raise FileNotFoundError(path)
        cap=cv2.VideoCapture(str(path))
        if not cap.isOpened():raise RuntimeError("BAD_GENERATED_MP4")
        dimensions=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if dimensions!=(576,320) or frames<11:
            raise RuntimeError("VIDEO_MISMATCH "+path.name)
        if x.get("quality_score") is not None:
            raise RuntimeError("UNVERIFIED_GENERATED_VIDEO_QUALITY_SCORE")
    if sorted(set(x["diffusion_steps"] for x in rows))!=[8,14,20]:
        raise RuntimeError("GENERATION_STEP_GRID_DRIFT")
    return len(rows)

def video_cards(rows):
    cards=[]
    for r in sorted(rows,key=lambda x:(x.get("prompt_group",x["id"]),x["diffusion_steps"])):
        path="../"+str(Path(r["output"]).relative_to("outputs")).replace("\\","/")
        name=html.escape(r["id"])
        prompt=html.escape(r["prompt"])
        seconds=float(r["elapsed_sec"])
        steps=int(r["diffusion_steps"])
        cards.append(f'<article class="video"><h3>{name}</h3>'
                     f'<video controls playsinline preload="metadata" src="{html.escape(path,quote=True)}"></video>'
                     f'<p>{prompt}</p><small>{steps} diffusion steps · {seconds:.2f}s · '
                     f'{r["gpu_peak_gib"]:.3f} GiB allocated GPU memory · human rating pending</small></article>')
    return "\n".join(cards)

def build():
    star_manifest=get("outputs/star_grouped_sft_v1/manifest.json")
    grouped=get("outputs/star_grouped_sft_v1/AUTHORITATIVE_RESULTS.json")
    cross=get("outputs/mvbench_scene_external_v1/comparison.json")
    external_manifest=get("outputs/mvbench_scene_external_v1/manifest.json")
    cache=get("outputs/shared_cache_grouped_v1/shared_cache.json")
    grid=get("outputs/diffusion_grid/generation_report.json")
    wan=get("outputs/temporal_round2/wan_preflight.json")
    checks=checked_manifest(star_manifest)
    if grouped["frozen_manifest_sha256"]!=star_manifest["rows_sha256"]:
        raise RuntimeError("GROUPED_REPORT_MANIFEST_MISMATCH")
    if grouped["cross_split_exact_content_overlap"]!=0 or grouped["unique_content_hashes"]!=193:
        raise RuntimeError("GROUPED_REPORT_CONTENT_LEAK")
    if cross["manifest_sha256"]!=external_manifest["rows_sha256"] or cross["samples"]!=200:
        raise RuntimeError("EXTERNAL_MANIFEST_DRIFT")
    if cross["star_overlap"]!=0 or cross["training_on_external_test"] is not False:
        raise RuntimeError("EXTERNAL_TEST_LEAK")
    if cache["sample_count"]!=40 or cache["all_cached_and_naive_images_byte_identical"] is not True:
        raise RuntimeError("FRAME_CACHE_NOT_INPUT_EQUIVALENT")
    if wan["status"]!="PREFLIGHT_ONLY_NO_VIDEO_GENERATED":
        raise RuntimeError("WAN_CLAIM_STATUS_UNEXPECTED")
    clips=verify_generation(grid)
    if (ROOT/"outputs/diffusion_grid/human_scores.json").exists():
        raise RuntimeError("RELEASE_NEEDS_INDEPENDENT_REVIEW_VALIDATION_BEFORE_SCORES")
    if not (ROOT/"outputs/diffusion_grid/review.html").is_file():
        raise RuntimeError("HUMAN_REVIEW_UI_MISSING")
    gitsha=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
        capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    out=ROOT/"outputs/release"
    out.mkdir(parents=True,exist_ok=True)
    metadata={
        "release_utc":datetime.now(timezone.utc).isoformat(),
        "frozen_snapshot_sha":gitsha,
        "source":"github.com/YOUessi/Multimodal-VLM-Video-Generation-Lab",
        "grouped_star_data_integrity":checks,
        "grouped_star_heldout":grouped["new_results"],
        "cross_task_scene_transition":cross["results"],
        "shared_cache_speedups":{"cold":cache["metrics"]["cold_speedup_mean"],
                                "warm":cache["metrics"]["warm_speedup_mean"]},
        "exact_selected_pixel_parity":True,
        "diffusion_video_count":clips,
        "diffusion_model":"ZeroScope, not Wan",
        "wan_executed":False,"human_quality_scored":False,
        "limitations":["Public benchmarks may have pretraining contamination",
                       "SHA excludes exact duplicates but not near duplicates",
                       "Small grouped 40 test has wide paired CI",
                       "Warm cache result is only for repeated videos",
                       "Diffusion objective pixel proxies do not measure perceptual video quality"]
    }
    (out/"EVIDENCE_MANIFEST.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+"\n")
    def table_rows(d,labels):
        out=[]
        for n in labels:
            row=d[n]
            lhs=row["base"]["correct"]; rhs=(row.get("lora") or row.get("grouped_lora"))["correct"]
            out.append(f"<tr><td>{html.escape(str(n))} frames</td><td>{lhs}</td><td>{rhs}</td><td>{row['delta_pp']:+.1f} pp</td><td>{html.escape(str(row.get('bootstrap_95_ci_pp',row.get('bootstrap_95ci_pp'))))}</td></tr>")
        return "\n".join(out)
    grouped_body=table_rows(grouped["new_results"],("1","3","5"))
    external_body=table_rows(cross["results"],("3","5"))
    cards=video_cards(grid["samples"])
    page="""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Multimodal VLM & Video Generation Lab — experiment evidence</title>
<style>body{margin:0;background:#f4f7fb;color:#20354a;font:14px/1.65 system-ui,-apple-system,"PingFang SC",sans-serif}
header{background:linear-gradient(120deg,#112d49,#265a83);color:white;padding:50px 25px}
main{max-width:1200px;margin:0 auto;padding:26px}h1{margin:0;font-size:clamp(26px,4vw,40px)}
header p{max-width:900px;color:#ccdfef}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
.stat,.panel,.video{background:white;border:1px solid #dce5ef;border-radius:14px;padding:18px}
.stat strong{font-size:25px;color:#1f669e;display:block}.stat span{color:#718598;font-size:12px}
.panel{margin:18px 0}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:8px;border-bottom:1px solid #e6edf3}
th{color:#60778c;font-size:12px}p,li{color:#526a7e}.notice{background:#fff3dc;padding:14px;border-radius:9px;color:#87622c}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}.video video{width:100%;max-height:300px;background:#091522}
.video small{font-size:12px;color:#6c849a}.video p{font-size:12px}
a{color:#1e6ba1}@media(max-width:800px){.grid,.stats{grid-template-columns:1fr}}@media print{.video video{display:none}body{background:white}}</style></head><body>
<header><h1>Multimodal VLM & Video Generation Lab</h1>
<p>真实 Tang GPU 实验 · 内容 SHA-256 分组防泄漏 · 两随机种子 VLM LoRA · 200 视频外部评测 · 共享缓存 · 视频生成</p></header>
<main><div class="stats"><div class="stat"><strong>193</strong><span>独立 STAR 视频字节内容</span></div>
<div class="stat"><strong>0</strong><span>新分组训练/验证/测试的完全重复文件</span></div>
<div class="stat"><strong>6</strong><span>实际生成的 ZeroScope MP4</span></div></div>
<section class="panel"><h2>STAR 动作序列：完整 SHA 分组后的独立测试</h2>
<p>120 训练 / 40 验证 / 40 测试；两随机种子，验证集选择模型；未使用旧版存在泄漏的 Adapter。</p>
<table><tr><th>输入</th><th>基础模型正确</th><th>新 LoRA 正确</th><th>变化</th><th>配对 Bootstrap CI</th></tr>__STAR_ROWS__</table></section>
<section class="panel"><h2>外部 MVBench 场景转换（200 视频，未用于训练）</h2>
<table><tr><th>输入</th><th>基础模型正确</th><th>STAR LoRA 正确</th><th>变化</th><th>配对 Bootstrap CI</th></tr>__EXTERNAL_ROWS__</table>
<p>以上置信区间包含零，不能证明稳定的跨任务模型收益。</p></section>
<section class="panel"><h2>共享场景特征缓存</h2><p>40 个真实视频、选中帧与输入图片字节完全一致。
平均冷缓存加速 <strong>__COLD__×</strong>，重复视频热缓存加速 <strong>__WARM__×</strong>。
热缓存不代表新视频首次处理速度。</p></section>
<section class="panel"><h2>真实 Diffusion 视频生成（ZeroScope）</h2><p class="notice">
模型授权仅限研究及非商业。下面的视频均已由 Tang GPU 实际生成；Wan2.1 未执行；
视频质量/Prompt 一致性尚无人工盲评分数。模型生成速度不等于视觉质量。</p>
<div class="grid">__VIDEOS__</div></section>
<section class="panel"><h2>科研限制</h2><ul>
<li>此前 STAR 按视频文件名划分的结果存在数据泄漏，不作为独立评测引用。</li>
<li>SHA256 去重只覆盖完全相同的文件；仍需感知近重复审查。</li>
<li>真实动作理解和外部场景转换的 LoRA 收益均未达到统计显著。</li>
<li>重复视频的缓存加速不能外推为新输入性能。</li>
<li>人类质量评价必须由独立评审填写，不接受自动生成的主观分数。</li></ul>
<p><a href="https://github.com/YOUessi/Multimodal-VLM-Video-Generation-Lab">GitHub 源码与实验报告</a></p>
</section></main></body></html>"""
    page=page.replace("__STAR_ROWS__",grouped_body).replace("__EXTERNAL_ROWS__",external_body)
    page=page.replace("__COLD__",str(cache["metrics"]["cold_speedup_mean"]))
    page=page.replace("__WARM__",str(cache["metrics"]["warm_speedup_mean"]))
    page=page.replace("__VIDEOS__",cards)
    (out/"index.html").write_text(page,encoding="utf-8")
    print("PROJECT_RELEASE_READY",out/"index.html",
          "grouped_star_SHA",star_manifest["rows_sha256"],
          "generated_videos",clips,flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--verify",action="store_true")
    ap.parse_args();build()
