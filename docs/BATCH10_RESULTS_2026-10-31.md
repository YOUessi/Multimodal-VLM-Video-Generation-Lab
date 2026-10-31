# Project 4 — batch 10 verified (Tang, 2026-10-31 21:22 CST)

**All seven approved stages B45–B51 passed** under local frozen source SHA `b8e4d9a94f3289692250f5edfeeea3fa0b00e91b`. Offline regression suite: **35/35 tests passed**. No raw private video, checkpoints, adapters, or generated media are uploaded to GitHub.

## 1. CPU cost versus real-video answer accuracy

40 video-ID-disjoint MVBench STAR action-sequence test clips, matched 3 or 5 sampled frames. CPU timings from **3 repeated randomized alternating-order trials** (120 measurements per method per frame budget).

| Frame budget | Uniform median CPU decode | Motion+hue adaptive median CPU select+decode | Mean adaptive / uniform runtime |
|---|---:|---:|---:|
| 3 frames | 49.6 ms | 508.7 ms | **10.34×** |
| 5 frames | 66.9 ms | 527.5 ms | **7.53×** |

Paired answer accuracy (from the unchanged prior frozen heldout evaluation):

| Model | Frames | Uniform | Adaptive |
|---|---:|---:|---:|
| SmolVLM2 base | 3 | 19/40 | 19/40 |
| SmolVLM2 base | 5 | 20/40 | 21/40 |
| STAR-trained LoRA chosen on validation | 3 | 20/40 | 20/40 |
| STAR-trained LoRA chosen on validation | 5 | 20/40 | 22/40 |

**Interpretation:** +2.5/+5.0 percentage-point changes are not statistically established (paired 95% confidence intervals include zero). High CPU overhead makes the current 48-probe adaptive sampler unattractive versus simple uniform selection. No test-set-driven adjustment of weights, candidate counts or prompts was made.

## 2. Six actual generated ZeroScope Diffusion videos

All videos were generated locally on Tang (or copied from its frozen original 14-step outputs), 576×320 resolution, 12 frames at 6 fps, same two text prompts and random seeds, 14/8/20 diffusion steps. Model revision: `cerspense/zeroscope_v2_576w@1f35e3dc3350d649f342bb7618fe34ac8fae5c9e` (CC-BY-NC-4.0, research/noncommercial).

| Prompt group | Inference steps | Generation wall time | Peak allocated GPU RAM |
|---|---:|---:|---:|
| rolling_ball | 8 | 12.99 s | 3.327 GiB |
| rolling_ball | 14 | 18.82 s | 3.327 GiB |
| rolling_ball | 20 | 25.07 s | 3.327 GiB |
| turning_robot | 8 | 11.03 s | 3.327 GiB |
| turning_robot | 14 | 18.19 s | 3.327 GiB |
| turning_robot | 20 | 25.15 s | 3.327 GiB |

All six MP4 files are valid and available only on Tang under `outputs/diffusion_grid/`. Low-level optical-flow, RGB frame differences, flicker, brightness and sharpness diagnostics were calculated; **none establishes quality or prompt adherence**.

The offline six-video blind-review page (`outputs/diffusion_grid/review.html`) is generated; human ratings remain **unscored**, and no subjective score has been imputed.

## Reproducibility

- Source GitHub branch and tested frozen SHA recorded in Tang `repo-batch10-tested/.automation/`.
- Batch manifest: `outputs/batch10/BATCH10_MANIFEST.json`; technical report: `outputs/batch10/BATCH10_RESULTS.md`.
- Key source scripts: `experiments/frame_selection_cost.py`, `experiments/generation_step_grid.py`, `experiments/generation_diagnostics.py`, `experiments/batch10_summary.py`.
- Fixed configs: `configs/frame_selection_cost.json`, `configs/diffusion_steps_grid.json`.
- The local 10-minute cron does not automatically devise new research tasks or post updates in this chat.

## Next rational research directions

(1) Persistent shared feature/decode cache to remove duplicated CPU work across the 3/5-frame requests, preserving identical frozen frame selections. (2) External real-video benchmark on genuinely unseen scenes, rather than retuning to the 40 STAR heldout videos. (3) Human blind review of video generation; do not claim quality improvements without reviewers. (4) Potential Wan quantization only after memory feasibility and quality controls.
