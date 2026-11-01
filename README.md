# Multimodal VLM & Video Generation Lab

## New prospective experiment — Batch 34: independent 2.5D parallax and target identity (2026-11-01)

**Implemented, 135/135 code regressions passed, and 280 new synthetic MP4 bytes audited; heldout answers remain unscored.** The new independent 40-pair DEV/100-pair TEST source uses three depth layers with unequal apparent camera motion, same-color different-shape distractors, visually identical distractors and moving shadows. **DEV-only negative finding:** plain HSV 38/80, old ORB affine 32/80, new query-grounded shape+affine abstention only **7/80**, with 73/80 abstentions and 0 wrong over just 7 accepted. This high selective precision does **not** establish better task accuracy, especially on heldout data. The source and all 10 frozen stages are predeclared; TEST scoring is forbidden until CV and actual 500M/2.2B GPU outputs are saved. **These are procedural 2.5D videos, not real camera recordings or 3D scene reconstruction.** [Prospective protocol and DEV measurements](docs/BATCH34_PARALLAX_IDENTITY_PROSPECTIVE_2026-11-01.md).


## Latest verified experiment — Batch 33: independently frozen 2D camera rotation, zoom and identity ambiguity (2026-11-01)

**10/10 frozen stages completed, 121/121 frozen-source tests passed, 600 new real GPU VLM inferences.** Newly seeded 40 development scene pairs (80 MP4) and 100 heldout test scene pairs (200 MP4) cover world-relative stationary/diagonal motion under camera pan, rotation, zoom and same-color object ambiguity + partial occlusion. The 200-image labels were only scored after pixel-only tracker and pretrained model predictions had been saved under immutable source SHA `a6fbb8dc6e2715304c339ba1012b1ca1d23e3bd2`.

Against heldout clips, **HSV pixel displacement correctly classified 141/200; translation-only phase compensation 55/200; ORB/RANSAC camera-affine compensation 179/200**. Affine vs plain is **+19.0pp** (paired scene-pair bootstrap 95% CI **[+12.0,+27.0]**). Conservative affine gate accepted 144/200, with **143 right and 1 wrong**. **Same-color/occlusion was still only 34/50**, and affine CPU P50/P95 was 36.26/43.45 ms per video (plain 10.79/18.15). Frozen 2.2B 3-frame **VLM-only 41/200** versus affine-gated+VLM **147/200**, but direct affine tracking alone is **179/200**, so this is a *specialized computer-vision tool result, not VLM reasoning enhancement*. This is **new synthetic 2D source scenes, NOT natural camera footage or 3D parallax**; no real-world generalization claim. [Complete test results, negative cases, source manifest and exact frozen SHA](docs/BATCH33_AFFINE_CAMERA_RESULTS_2026-11-01.md); [precommitted protocol](docs/BATCH33_AFFINE_CAMERA_PROSPECTIVE_2026-11-01.md).

## Latest execution — batch 32, new locked camera-translation benchmark (2026-11-01)

**10/10 stages completed on Tang; 112/112 regression tests passed.** After a batch31 source-data directory contract error, batch32 was rebuilt from a new frozen SHA, with the error preserved in its logs.

An independent procedural 2D render-shift set contains **40 dev source scene pairs (80 MP4s) + 100 test scene pairs (200 MP4s)**, balanced across five world-relative motion directions and four conditions: clean, changing illumination, partial occlusion, and camera translation. The test is synthetic, not natural video.

The predeclared camera-stabilized tracker estimates **background motion from image phase correlation** and subtracts it from the target's on-screen displacement, using the SAME HSV object finder. Test camera-pan accuracy: **36/50 plain HSV → 50/50 stabilized HSV**, **+28pp, paired source-scene bootstrap 95% CI [+12,+48]pp**. Other nuisance accuracy was unchanged. Direct tracker accuracy across all test videos: **182/200 raw vs 196/200 stabilized**.

VLM 3-frame accuracy on this specially rendered 5-option synthetic task was **40/200** for both 500M and 2.2B; camera-stabilized conservative tool fallback reached **171/200** and **157/200**, respectively. This is a measured domain-specific deterministic computer-vision engineering gain, **NOT** VLM architecture, real-camera or natural-video generalization. The stabilized tracker was accepted on 157/200 and correct on all 157 accepted under these narrow source conditions; do **not** extrapolate its precision. Tracker median CPU time: raw **14.83 ms**, stabilized **26.64 ms** per video.

All 280 generated clips, raw predictions, local videos and review page remain on Tang, not GitHub. [Full new source hashes, statistical results, limits and executable paths](docs/BATCH32_NEW_CAMERA_SHIFT_HELDOUT_2026-11-01.md).

The Tang cron checks/executes approved stages every **10 minutes**. In addition, an enabled **hourly ChatGPT research-continuation automation** now checks for completed batches, reviews actual results and attempts to implement the next frozen batch. This is an actual separate scheduled agent, not a claim that the cron itself can invent or code new research tasks. Automation failures, permission issues or untested claims must be reported.

---

## New research milestone: object-centric motion tool + VLM (batches 27–30, 2026-11-01)

The project moved beyond heuristic visual-frame selection to ask **whether a VLM can use explicit object motion evidence and whether a gated deterministic tool can improve direction QA**.

**Frozen synthetic counterfactual study — two pretrained VLMs × five conditions:**

- New dataset: 120 frozen source test scenes, each played forward and reversed (**240 test clips**), with target color+shape queries, independently moving distractors, and five balanced motion classes; 20 separate development scene pairs.
- Actual GPU: **2,400 VLM inferences** across question-only, RGB-3, numeric coordinates only, RGB+measured coordinates, and RGB+intentionally contradictory reversed coordinate order. The coordinates are measured from video pixels; no ground-truth scene annotation enters the tracker or model prompt.
- On this simple fixed-camera rendered-color domain, classical CV direction estimation scored **240/240**, while 2.2B video-only scored **40/240**, and video+numeric coordinates **51/240** (+4.6pp, *paired scene-cluster CI [-0.83,+10]*; not proven). Directly appending correct numeric trajectories **did not establish VLM reasoning improvement**. The 2D classical tracker is **domain-specific, not universal**.
- See [full 240-video experiment, negative results and caveats](docs/BATCH27_28_COUNTERFACTUAL_MOTION_2026-11-01.md). A local offline first-eight-pair review HTML is generated on Tang.

**Public CLEVRER domain transfer, full 200 already-inspected simulated videos:**

- The unmodified synthetic **exact-RGB palette** tracker failed to find usable targets on **200/200** external source items; a conservative fallback therefore produced **no gain**. See [batch29 failure analysis](docs/BATCH29_UNTUNED_CLEVRER_TRANSFER_2026-11-01.md).
- A **generic named-color HSV + connected-component tracking + conservative position-confidence gate** (fixed before external scoring, 40/40 on separate development clips, no CLEVRER label calibration) produced movement options for **119/200** videos and accepted **65/200** high-motion predictions. Among the accepted 65, **58 correct and 7 incorrect** (89.2% tool precision). All 200, including abstentions, remain in the denominator.
- With **existing frozen VLM outputs** (no additional VLM inference), 2.2B three-frame CLEVRER accuracy improved from **72/200 (36.0%) to 121/200 (60.5%)**, +24.5pp, paired bootstrap 95% CI [+18,+31.5]. Five-frame results: 73/200 to 121/200. 500M also improved from 61/200 to 106/200 at three frames.
- **Critical qualification:** CLEVRER is rendered simulation, not natural camera video, and this benchmark was **already analyzed in earlier studies**. No blind out-of-domain independent effectiveness claim, generalized tracker reliability, or novel model architecture is made. The gated tool costs CPU time and can make wrong overrides.
- See [the complete batch30 200-video external tracker benchmark and limitations](docs/BATCH30_HSV_TRACKER_GATED_CLEVRER_2026-11-01.md).

**Historical batch30 status:** four approved stages and 103 tests passed. Later batches have since completed; see the latest execution section above for the active state. Raw model outputs, videos, review HTML, weights and LoRA adapters remain local on Tang, not committed to GitHub.

---

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

### Frozen batch23: motion-salient frame selection negative ablation

The 12-probe label-blind motion/hue frame-selection benchmark completed **1,600 real GPU inferences** on the already-inspected 400 public CLEVRER/Scene Transition videos. For the stronger 2.2B model, movement-direction correctness was unchanged at 3 and 5 frames; all eight model/task/frame comparisons had paired 95% intervals including zero. Mean selection overhead was **208ms CPU per video**. The heuristic was not a reliable improvement over uniform sampling. See [negative results and source hashes](docs/BATCH23_MOTION_SALIENT_RESULTS_2026-11-01.md).

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
