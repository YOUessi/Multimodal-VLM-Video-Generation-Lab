# Project 4 — continuous research and execution contract (2026-11-01)

## Why the previous ten-minute check did not continue

The previous Tang user crontab ran `automation/runner.py tick --max-stages 2` every ten minutes **against a single immutable source snapshot and its `automation/plan.json`**. When all stages were done, the runner repeatedly returned `ALL_APPROVED_STAGES_DONE`. It did **not** formulate new research ideas, write a new source plan, or deploy a new frozen snapshot. Thus an idle GPU did not indicate progress. This was a genuine orchestration gap.

## Restored two-layer execution

1. **Tang execution cron** stays every 10 min; marker `MVVG_PROJECT4_10MIN`. It runs *only frozen approved stages*, requires the pinned local git SHA, respects existing GPU mutual exclusion, resource gates, timeout, errors, and pause file, and preserves all prior artifacts. It must never delete failed batch evidence or silently retune previously evaluated test samples.
2. **Separate enabled ChatGPT hourly research-continuation task**, titled *项目4｜自动续研与验收*, checks Tang and GitHub, reads actual results and code, and is instructed to do substantive next-batch implementation when a batch completes: source changes, tests, frozen config, GitHub commit, new Tang snapshot and execution. This is a scheduled AI continuation, not the CPU cron inventing research. **Once an hour is the fastest ChatGPT automation recurrence** supported; cron remains ten minutes. Failure to connect or obtain permission must be reported rather than fabricated.
3. GitHub `main` is the sole source of code and documents. Each batch uses an independent frozen local directory `repo-batchN-tested` with a pinned upstream commit hash, local snapshot SHA and `.automation/state.json`. No weights, generated videos, original benchmark MP4s, API credentials or raw private material are committed.

## Latest actually verified science (Batch 32)

- Latest complete frozen project: `repo-batch32-tested`. All **10/10 stages passed**, **112/112 tests** passed, source guard passed.
- Independent procedural test: 100 source scene pairs, each forward/reverse, 200 test clips. Four balanced nuisance modes (clean, illumination, partial occlusion, camera pan) × 5 directions. The 280 total dev+test videos are new 2D synthetic renderings.
- Raw HSV camera-pan source-video accuracy **36/50**; background phase-correlation compensated **50/50** (+28pp, paired scene-cluster 95% CI [+12,+48]pp). This is a narrow camera-translation effect, not natural-video generalization.
- Strong 2.2B VLM 3-frame on the synthetic five-option test **40/200**; stabilized gated fallback **157/200**. Domain-specific deterministic tool dominates. Tool accepted 157/200 and all 157 were correct **in this narrow test only**.
- Source-pixel tracker median CPU time raw HSV **14.83 ms**, camera-compensated **26.64 ms**, excluding VLM. 500M also run for 200 items, and 2.2B for 3/5 frames (600 actual new GPU predictions).
- Machine-readable raw data, QA labels and per-example model/tool predictions remain on Tang. The local review HTML exists. [Full report](BATCH32_NEW_CAMERA_SHIFT_HELDOUT_2026-11-01.md).

## Next independent technical batch: rotation, zoom and distractor ambiguity

**Do not tune on the 200 Batch32 heldout videos.** This entire next study must introduce a new frozen dataset with new independent seeds and a separate dev/test split before scoring.

**New science hypothesis:** phase correlation corrects only global translation; rotating/zooming/parallax cameras produce nonuniform flow that a feature-based robust geometric transformation (ORB/SIFT-like keypoint matches + RANSAC affine) may correct. Color-ambiguous distractors or full occlusion may prevent reliable target correspondence and should yield abstentions. More parameters do not automatically imply better accuracy.

Predeclare:
- Four nuisance families: camera translation, camera rotation, camera zoom, and near-same-color distractor/partial occlusion. If parallax is added, label it distinctly because single-plane global affine cannot represent the entire scene.
- Balanced stationary/up-right/up-left/down-right/down-left target trajectories, reversed paired clips, source/world-relative direction truth and documented actual background transforms. **Never supply ground-truth camera transform or target trajectory to models or trackers.**
- Baselines: plain frozen batch30 HSV tracker, frozen batch32 phase-correlation global translation tracker, and new dev-calibrated robust affine geometry tracker with a confidence gate; VLM 500M/2.2B frozen model baselines if GPU resources permit.
- Development only for threshold/protocol selection; test is used exactly once after all code/criteria are frozen.
- Report full-video denominator, abstention/coverage, accepted precision (including wrong overrides), normal/reverse consistency, per-nuisance and whole-test paired scene-cluster bootstrap CIs, CPU time P50/P95 and peak GPU allocations. Produce local side-by-side video review.
- If a method has no robust gain or loses accepted precision, **retain the negative result**. Never suppress difficult samples.
- All source changes require unit tests plus actual end-to-end fixture tests that verify dataset/result root separation and correct camera motion sign.

## Subsequent research backlog

1. **Real-camera out-of-distribution evaluation** using consented or appropriately licensed captured videos, not just previous CLEVRER or newly rendered 2D clips. Need precise ground-truth motion relative to camera, occlusion/camera movement annotations, and independent freeze.
2. **Evidence-to-answer interface:** the prior batch27 two VLMs did not reliably convert correct numeric coordinates into direction choices; compare structured symbolic tool use and model response generation under externally validated confidence. The tool should be separable from the VLM.
3. **Video generation:** finish real independent human blind quality evaluation for ZeroScope LoRA synthetic vs pretrained; 12-step, 12-example training MSE gain -7.59% is not a high-quality video generation claim. Consider licensed natural video-caption training with independent validation and two seeds only after resource/dataset rights checks.
4. **Reproducible release:** stable source/asset manifest, CI and offline video previews, explicit negative results and license notice.

## Next automation decision rule

Read this file and the latest `docs/BATCH*.md`, then inspect live Tang `repo-batch*-tested/.automation/state.json` and the active `MVVG_PROJECT4_10MIN` crontab. If batch32 or later is complete, **actually implement** the next meaningful stage instead of writing a plan-only message. Never assume a next-stage plan exists merely because the cron is enabled. When a new frozen batch is fully complete, record a concise verified outcome and update this queue. If access errors or test blockers prevent execution, preserve logs and describe the blocker and source SHA honestly.

This project is not complete until a method demonstrates effectiveness beyond narrow synthetic/test-influenced settings, while video-generation quality is independently assessed. A series of completed engineering stages is not, by itself, a proof of broad scientific contribution.
