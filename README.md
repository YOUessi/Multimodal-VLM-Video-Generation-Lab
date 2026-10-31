# Multimodal VLM & Video Generation Lab

## Latest verified milestones — 2026-11-01

### VLM temporal evidence and model scaling (batch 19)

**2,400 actual GPU inferences**, no task-specific tuning: two frozen SmolVLM2 models × 400 public MVBench video QAs × three conditions (question-only / uniformly sampled 3 frames / 5 frames).

| Task (200 official QAs each) | Model | No video | 3 frames | 5 frames |
|---|---|---:|---:|---:|
| CLEVRER simulated Moving Direction | 500M | 26.0% | 30.5% | 32.5% |
| CLEVRER simulated Moving Direction | 2.2B | 20.5% | 36.0% | 36.5% |
| Public natural-video Scene Transition | 500M | 40.5% | 79.5% | 80.0% |
| Public natural-video Scene Transition | 2.2B | 57.0% | 91.0% | 90.0% |

The 2.2B model **gains +16.0 percentage points on simulated direction with five frames versus no video (paired bootstrap 95% CI [+7.0,+24.5])**. In 3-frame scene transition it outperforms 500M by +11.5pp (CI [+5,+18]). **Increasing three to five uniformly sampled frames shows no reliable improvement** in either task. These are specific public benchmarks, not evidence of all-domain VLM capability.

Detailed [batch19 results, protocol, source hashes and limitations](docs/BATCH19_CAPACITY_TEMPORAL_RESULTS_2026-11-01.md).

### Actual video Diffusion LoRA (batch 21)

A pretrained **ZeroScope UNet3DConditionModel** has now undergone **real PEFT LoRA optimization** on 12 procedurally generated paired motion videos/captions, with four different synthetic validation clips. Rank 4, **830,720 of 1.412B model parameters trainable**, 12 genuine GPU update steps, training peak allocated VRAM **3.483 GiB**. Fixed validation epsilon-denoising MSE decreased from **0.072447 to 0.066950 (-7.59%)**.

Four actual matched pretrained/tuned MP4s (two prompts/seeds; four 192×128 frames, eight Diffusion steps) were successfully generated. A reviewer-blinded offline HTML and private variant mapping exist locally on Tang. **Generated-video human quality and prompt adherence have NOT been rated**. This small synthetic noise-prediction improvement cannot be presented as a real-world video quality improvement. Model license is **CC-BY-NC-4.0, research/noncommercial**.

See [batch21 real video-generation LoRA log](docs/BATCH21_ACTUAL_ZEROSCOPE_VIDEO_LORA_2026-11-01.md).

### Temporal control results (batch 22): 1,600 additional real inferences

A frozen control compared identical visual-frame counts, either **normal chronological frames**, **repeated static center frames**, or **reversed chronology**, on all 400 public videos using SmolVLM2-2.2B. All five stages completed, 78 tests passed.

- CLEVRER moving_direction 3 frames: **72/200 normal vs 75/200 repeated-center** (no demonstrated loss of accuracy when visual progression disappears); 5 frames: **73/200 normal vs 74/200 repeated-center**.
- Natural scene_transition 3 frames: **182/200 normal vs 171/200 repeated-center** (-5.5pp, paired CI [-10.5,-1.0]); 5 frames: **180/200 normal vs 170/200 repeated-center** (-5pp, paired CI [-9.5,-0.5]).
- Reversed frame order changed only **5–6/200** motion-direction selections and **7–15/200** scene-transition selections. **No reversed-video accuracy was computed**; reversing changes the valid answer in some cases.

The results motivate investigation of temporal evidence use but do not pinpoint an Attention or positional-encoding mechanism. Full [batch22 report](docs/BATCH22_TEMPORAL_CONTROLS_2026-11-01.md).

### Current frozen batch23: exploratory motion-salient sampling

Tang's 10-minute cron is now configured to run a label-blind 12-probe motion/hue frame selector on the same frozen 400 public videos, matched 3/5-frame budgets and both VLM sizes; it records **CPU selection overhead** and paired correctness differences. This is an exploratory ablation on **already inspected** public samples, not fresh never-seen evaluation, and will fail closed on any stage errors. See `configs/motion_sampling_v1.json` and `automation/plans/batch23_motion_salient_v1.json`.

### Caution: earlier STAR data leakage

The initial STAR source-ID-only split was **invalid** because SHA-identical MP4 contents crossed splits. An exact SHA grouped study was completed, followed by a retrospective strict perceptual-candidate grouped rerun. The latter yielded 3-frame **19/40 base → 25/40 LoRA** (+15pp), but its paired CI [-5,+35] includes zero, and it was **not a new never-inspected blind benchmark**. Do not mix this result with the independent MVBench capacity test or claim stable natural-video improvement.

All original videos, model weights, LoRA adapters, and generated MP4s remain on Tang; this public repository contains only source code, frozen configurations, tests and evidence summaries.

---

## Prior verified research and data-integrity work — 2026-10-31

**The project now contains real GPU VLM training, public video benchmarks, reproducible failure analysis, and six actual text-to-video Diffusion outputs.** The outcome is mixed; no unsupported model accuracy/quality improvement is claimed.

- **Perceptual QA caveat:** A subsequent strict 3-frame pHash+dHash audit of 19,900 STAR pairs flagged **two potential near-duplicate training/validation pairs**, not across the test boundary under that heuristic. This may bias validation-based seed selection; exact SHA disjointness is **not** a guarantee of independent visual content. See [near-duplicate audit](docs/STAR_NEAR_DUP_AUDIT_2026-10-31.md).
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
