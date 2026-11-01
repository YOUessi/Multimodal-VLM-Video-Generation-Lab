# Batch 32 — new frozen camera-relative motion shift study (2026-11-01)

**Execution status: 10/10 approved Q1–Q10 stages succeeded on Tang.** All **112 offline regression tests passed**, source-security gate found **zero tracked private videos/model weights** among 135 tracked source files. Local frozen source SHA `5ef9f41511a2a825a2bfb2e31ec7138f738ed61e`; pinned upstream GitHub snapshot SHA `1fee445a961301c2b9d1e58f81e18412a6b2913c`.

The first batch31 source snapshot was halted at a directory contract error: video files had been saved to `outputs/` instead of `data/`. Batch32 fixed the generator, added an integration regression test and created a new frozen source and test manifest. The old failure and erroneous files are preserved for audit, not counted as successful outputs.

## Research hypothesis

The earlier exact-palette tracker and HSV+connected-component tracker operated directly on image-plane displacement. When the camera pans, **screen-space target motion = stationary-background-relative target motion + background image shift**. A fixed, label-blind camera compensation step should therefore improve object-direction decisions in globally translating camera footage without modifying color detection.

The new method: the SAME fixed question-grounded HSV target region tracker as before, plus **global background displacement estimated by two-frame phase correlation with a Hanning window**. World-relative target displacement is the tracked screen displacement MINUS the estimated stationary-world background shift. A conservative gate accepts clear diagonal motion only. All thresholds, phase-correlation settings, and model prompts were frozen before test scoring.

## NEW independently frozen video dataset

- **40 distinct procedural development scenes**, each with forward/reversed video (**80 dev clips**).
- **100 distinct procedural test scenes**, each with forward/reversed video (**200 test clips**).
- Four nuisance families: clean, changing illumination, partial foreground occlusion, moving camera. Five directions (stationary, up-right, up-left, down-right, down-left) balanced within each mode.
- Every test nuisance contains 25 source scene pairs, or 50 counterfactual MP4s. Scene pairs are the independent unit of bootstrap resampling.
- 256x256, 10-frame procedural 2D videos with different target and distractor colors, shapes, fixed textured world background. **This is synthetic, not natural video.**
- Reference labels represent **motion relative to the stationary world background**, not raw on-screen motion.
- Frozen manifest SHA: `15700b28255ac736c03ace4265b649021ab9fcbb93bd13b6ac281790dadc7dd0`.
- Full clip bytes re-hashed, 0 exact-source overlap across development and test. All dev/performance tuning completed before test correctness scoring.
- Development (80 clips): plain HSV tracker **74/80** correct; camera-stabilized HSV **79/80** correct. This is development, not heldout test performance.

## NEW heldout direct-tracker result

| Nuisance | Raw HSV correct | Camera-stabilized HSV correct | Paired gain | Scene-cluster bootstrap 95% CI |
|---|---:|---:|---:|---|
| Clean | 49/50 | 49/50 | 0 pp | [0,0] pp |
| Illumination | 47/50 | 47/50 | 0 pp | [0,0] pp |
| Partial occlusion | 50/50 | 50/50 | 0 pp | [0,0] pp |
| **Camera pan** | **36/50** | **50/50** | **+28.0 pp** | **[+12,+48] pp** |

Overall, raw HSV: **182/200 correct** (192 direction attempts); stabilized HSV: **196/200 correct** (196 attempts).

Accepted by conservative nonstationary motion gate:
- Raw HSV: **143/200 accepted, 143 correct, zero accepted errors**.
- Stabilized HSV: **157/200 accepted, 157 correct, zero accepted errors**.

These perfect accepted-case precision values arise from a specially structured synthetic target/color/background distribution; they must **not** be extrapolated to real video.

## Source-pixel tracker CPU overhead

| Method | Median per-video tracking CPU | P95 |
|---|---:|---:|
| Original HSV tracker | **14.83 ms** | **17.31 ms** |
| HSV + background phase correlation | **26.64 ms** | **30.61 ms** |

The additional median tracker overhead is about **11.81 ms** per clip. This excludes VLM inference and any service overhead. Overall serving and energy gains were not measured.

## Three frozen VLM conditions and conservative tool fallback

Two pretrained models received identical original source video questions/answer options and 3 or 5 chronological RGB frames. No model training, no test-time prompt/threshold changes.

| Model | Input | VLM-only | Plain HSV gated fallback | Camera-stabilized HSV gated fallback |
|---|---|---:|---:|---:|
| 500M | 3 frames | 40/200 (20.0%) | 161/200 (80.5%) | **171/200 (85.5%)** |
| 2.2B | 3 frames | 40/200 (20.0%) | 148/200 (74.0%) | **157/200 (78.5%)** |
| 2.2B | 5 frames | 39/200 (19.5%) | 148/200 (74.0%) | **159/200 (79.5%)** |

Paired scene-cluster bootstrap for **stabilized tool vs VLM-only**:
- 500M 3 frames: **+65.5 pp**, CI **[+57.5,+73.0]**.
- 2.2B 3 frames: **+58.5 pp**, CI **[+51.0,+66.0]**.
- 2.2B 5 frames: **+60.0 pp**, CI **[+52.5,+67.5]**.

**Scientific interpretation:** Global background registration has a substantial measured benefit in the camera translation nuisance on this new independent procedural test. The stronger VLM still struggles around five-option chance accuracy, whereas a task-specific deterministic motion tool is extremely accurate in this narrow synthetic setup. This is evidence for an interpretable hybrid systems approach, **not** a newly trained or generally improved vision-language model.

## Limitations

- This source is **synthetic rendered 2D shapes** with an obvious dominant stationary texture, known color vocabulary, controlled camera translation, moderate occlusion and simple diagonal world motion. It is *not* natural human captured video.
- Phase correlation corrects **global translation only**; rotating, zooming or parallax cameras are unresolved.
- All five direction choices have strong symmetrical geometry and constant motion; no difficult real footage, dynamic backgrounds, severe object identity confusion or adversarial distractors.
- The gate can make wrong overrides outside this distribution; the 100% accepted-case precision here does not demonstrate calibration.
- The 200 test clips arise from **100 paired scenes**, not 200 independent observations. Cluster bootstrap samples scene pairs.
- Real-time end-to-end serving, energy cost and human review have not been measured. Frozen VLM inference and direct tracker costs are reported separately.

## Local reproducible artifacts (not uploaded)

- Frozen repo: `/home/you/projects/Multimodal-VLM-Video-Generation-Lab/repo-batch32-tested`.
- Data: `data/motion_render_shift_v1/manifest.json` and 280 local MP4s.
- Raw results: `outputs/motion_render_shift_v1/cv_predictions.json`, `500m.json`, `2b.json`, `results.json`, `RESULTS.md`.
- Interactive offline forward/reverse review: `outputs/motion_render_shift_v1/review.html`, containing the first three frozen test pairs for each nuisance category.
- Source: `experiments/domain_shift_motion_data.py`, `domain_shift_camera_tracker.py`, `domain_shift_motion_eval.py`, `domain_shift_review.py`.
- Config: `configs/domain_shift_motion_v1.json`; tests `tests/test_domain_shift_motion.py`.
- All original/generated videos, model weights and PEFT adapters remain local on Tang.

## Next independent research stage

Predefine a **new** camera rotation/zoom/parallax and identity-ambiguity dataset with independent seeds, and compare the current fixed phase-correlation method against a dev-calibrated feature-based geometric registration baseline (RANSAC affine/homography) using **a new locked test set**. Track coverage, override precision, camera shift residuals, whole-pipeline CPU/VRAM, and effects of distractors/occlusions. Do not retune the new method on the just-scored 200 test clips.

A separate hourly ChatGPT research continuation automation has been installed to review and create the next frozen work batch; Tang's existing 10-minute cron executes only already-approved source stages. The hourly automation is an orchestration mechanism, **not a guarantee** that any future run will have the required remote access or succeed without validation.
