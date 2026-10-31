# Multimodal VLM & Video Generation Lab

## Current verified results (2026-10-31; supersedes earlier smoke results)

**The project now contains real GPU VLM training, public video benchmarks, reproducible failure analysis, and six actual text-to-video Diffusion outputs.** The outcome is mixed; no unsupported model accuracy/quality improvement is claimed.

- **Critical integrity correction:** STAR action-sequence had 200 filenames but 193 unique byte contents. Five identical-content pairs crossed the initial video-ID train/val/test split, so the old 40-item heldout results are **INVALID** as independent evidence. See [the leakage audit](docs/STAR_CONTENT_LEAK_AUDIT_2026-10-31.md).
- **Authoritative fresh SHA-disjoint real-video SFT:** 120 train / 40 validation / 40 test, zero exact-content overlap, two new LoRA seeds, validation-only model selection. Accuracy (base → LoRA): 1 frame **17/40 → 15/40**; 3 frames **16/40 → 17/40**; 5 frames **18/40 → 20/40**. All paired confidence intervals include zero. See [corrected grouped results](docs/STAR_GROUPED_SFT_RESULTS_2026-10-31.md).
- **Independent external task, no training:** 200 MVBench scene-transition examples, zero exact-byte overlaps with STAR. Base → grouped LoRA: 3 frames **155/200 → 155/200**; 5 frames **157/200 → 160/200**. Cross-task gain not statistically established. See [external transfer](docs/MVBENCH_EXTERNAL_SCENE_TRANSFER_2026-10-31.md).
- **Video generation is REAL, but not Wan:** ZeroScope v2 576w safetensors on Tang, research-only/noncommercial licensing. Six local 576×320/12-frame clips from two fixed prompts and 8/14/20 diffusion steps. Time: roughly 11–25 seconds per clip at 3.327 GiB allocated VRAM; human prompt/visual quality remains **unscored**. See [generation and cost report](docs/BATCH10_RESULTS_2026-10-31.md).
- **Adaptive sampling engineering finding:** The original 48-probe motion/hue method had roughly **7.5–10.3× higher CPU cost** versus uniform decoding, without statistically proven gain. A shared-content cache benchmark is under development; historic old-split accuracy figures are not independent evidence.

The older paragraphs further down describe initial smoke experiments for provenance, not the latest research conclusions. **Weights, adapters, original real videos, generated MP4s and experiment logs stay on Tang**, not in GitHub.

## 2026-10-31: staged automation and measured VLM work

The complete batch-2 **synthetic** experiment finished real GPU work:
- 48 training, 12 validation and 18 heldout procedural three-scene MP4 samples (unique sequence splits).
- With identical no-image-splitting preprocessing, the base model scored 18/18 for single-frame color classification, 4/18 for 3-frame first/last color, and 4/18 for 5-frame first/last color.
- After 48 actual LoRA steps, the same frozen heldout synthetic test scored 18/18 in all three cases. **This is not proof of real-world temporal reasoning improvement**; the benchmark is simple and uses one seed.
- Full-precision Wan 1.3B preflight failed the conservative RAM gate; no Wan generation claimed.
- ZeroScope Diffusion has since generated six verified MP4s; see the current results section above.

See [detailed measured results](docs/BATCH_2_RESULTS_2026-10-31.md) and [Tang 10-minute local runner with safety gates](automation/README.md). Tang's 10-minute Linux cron cannot proactively post to the ChatGPT conversation, and the approved batch does not autonomously write new source code.


Research-oriented project for **visual question answering, temporal video reasoning, VLM LoRA adaptation, video generation, and evaluation**. The work is split into two independent tracks:

- **Track A: VLM / Video Understanding:** frame sampling, image and multi-image inference, temporal QA, LoRA fine-tuning, split-safe benchmarking.
- **Track B: Video Generation:** Wan/Diffusion pipelines and human-quality assessment, kept separate from VLM understanding and training.

## Actual verified GPU progress

Tang (RTX 4090 Laptop 16GB) has completed:
- SmolVLM2-500M model download through a mirror and **real CUDA inference**.
- 2 successful synthetic color QA answers and 1 incorrect multi-frame order answer (not an accuracy benchmark).
- 2 actual PEFT LoRA optimization steps with finite losses; local adapter saved (not a model improvement claim).
- Local 110.388s video qualitative 1-frame / 5-frame inference (private original video never uploaded).

See [measured results and limitations](docs/GPU_SMOKE_2026-10-31.md).

## Run measured GPU scripts

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install 'transformers==4.57.6' 'pillow>=10.4,<12' 'accelerate>=1.8,<2' 'peft>=0.17,<0.19' 'trl>=0.24,<0.26' 'datasets>=3,<5' 'diffusers>=0.35,<0.38' num2words soxr

# Download once; required for the scripts below.
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'HuggingFaceTB/SmolVLM2-500M-Video-Instruct',
    local_dir='models/SmolVLM2-500M-Video-Instruct'
)
PY

.venv/bin/python experiments/smolvlm2_gpu_smoke.py
.venv/bin/python experiments/smolvlm2_lora_gpu_smoke.py
.venv/bin/python experiments/local_video_gpu_probe.py --video /your/local/video.mp4
```

**Status:** The three GPU experiment scripts and logs summary are committed here. The earlier complete CPU prototype (31 passing tests; annotated data tools and Wan dry-run CLI) is currently provided as a separate source ZIP, and **has not yet been fully pushed into this GitHub repository**. Do not assume that the repository currently contains those other modules.

**Restrictions:** Never commit raw videos, model weights, LoRA adapter files, private paths or credentials. Real-video STAR SFT and independent MVBench scene-transition evaluation have been executed; Wan video generation has NOT been executed.
