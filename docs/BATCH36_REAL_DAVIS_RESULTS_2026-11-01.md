# Batch 36 — real-camera DAVIS 2017 independent-scene tracking: immutable results (2026-11-01)

**Status: ALL EIGHT FROZEN STAGES PASS, no tuning after TEST scoring.** This is the first real-photographic-video tracking transfer audit after synthetic Batch 34/35. It is a different task from CLEVRER multiple-choice direction QA and is **not** evidence that SmolVLM2 or camera-compensated object/world motion classification improved.

## Study identity and reproducibility

- **Official dataset:** DAVIS 2017 TrainVal, original 480p JPEG images + original palette-index PNG instance annotations, attributed to Pont-Tuset et al., *The 2017 DAVIS Challenge on Video Object Segmentation*. Media remains local; the individual source-license boundary must be reviewed before redistribution. Source: https://davischallenge.org/davis2017/code.html.
- **Frozen source commit:** `056cbbc65bea5a959faad44a73298a5ebb8e5249`; one isolated `repo-batch36-tested` Git checkout, no source changes throughout scoring. This report is a later, documentation-only commit, not a new model revision.
- **Corpus:** 12 distinct original real-camera video scenes, 4 DEV and 8 heldout TEST, deterministically disjoint via SHA-salted scene-name ordering. Each contributes eight frames with 3-frame source stride: 96 sampled photographs in total, with corresponding independent per-frame annotations. TEST = 8 videos × 7 non-query frames = **56 evaluated frames**, not 56 independent samples. Every scored TEST frame had a visible target; absent-target false acceptance is therefore **not tested**.
- **Source manifest SHA-256:** `4bc039d7e96a84f99ce3919f165c6d4261cd7b0413937933041e1be740ca3a4a`; 192 original JPEG/PNG assets checked, only verified byte-range downloads retained locally.
- **TEST prediction JSON SHA-256, pinned before private-label scoring:** `8bcd6a8fbb326527b9a155d3b9952f064e34c1b96abd924cd71e8472504f6bf8`. Independent sidecar `test_predictions.sha256` was validated before scoring.
- **TEST aggregate report SHA-256:** `648c1dae815ebd915146a0c51e3d23c4115d8177ef0ee37b93897ee61ac924b4`. Raw images, masks, per-scene gold, predictions and logs are intentionally excluded from GitHub.

## Predeclared methods and heldout results

All methods receive the initial target instance mask and center, then only the original RGB frames. No future mask or future centroid is provided at inference. The goal is **image-plane instance-mask-center tracking**; the reference centroid can move with mask shape, so it is not necessarily a material point.

| Fixed, pre-TEST method | TEST PCK@8px on all 56 visible frames | Correct / 56 | Accepted visible / 56 | Wrong accepted | Abstained visible | PCK@8px among accepted | CPU method p50 / p95 per scene |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hold initial center in image pixels | 7.14% | 4 | 56 | 52 | 0 | 7.14% | 0.00 / 0.00 ms |
| Forward Lucas–Kanade KLT | 35.71% | 20 | 56 | 36 | 0 | 35.71% | 5.72 / 12.29 ms |
| Forward–backward KLT, 1.5px check, ≥5 retained features | **42.86%** | **24** | **44** | **20** | **12** | **54.55%** | **6.92 / 7.89 ms |

Additional unconditional PCK@4px / PCK@16px: hold center **5.36% / 12.50%**; forward KLT **16.07% / 53.57%**; forward–backward KLT **25.00% / 58.93%**. PCK counts abstentions as failures. CPU method timing **excludes image decode and data loading**, and eight scenes are too few to make stable system-latency claims.

A 600-resample scene-cluster bootstrap (sampling entire TEST videos, not individual frames) gives a forward–backward KLT minus hold-center **+35.71 percentage-point** PCK@8px improvement, with approximate **95% interval [12.50, 57.14] pp**. This is an internal benchmark interval across eight scenes, not a general real-world confidence guarantee. DEV was separate: 4 scenes × 7 = 28 visible frames; PCK@8px hold center **2/28**, forward KLT **16/28**, forward–backward KLT **17/28**. Thus forward–backward tracking falls from 60.71% DEV to 42.86% TEST (−17.85 pp), a **descriptive** domain/scene-composition gap, not evidence for any individual cause.

## Negative findings, abstention and camera-relative interpretation

1. **Abstention is necessary but insufficient.** The forward–backward gate abstains on 12/56 visible frames (21.43%) and improves conditional precision to 24/44 (54.55%), but **20 accepted predictions remain >8px wrong**. No absent-target frames were encountered, so zero absent false accepts cannot be claimed as absence-detection effectiveness.
2. **Real video is considerably harder than the DEV result implied.** The unchanged strict KLT baseline achieves only 42.86% PCK@8 on TEST, despite being better than a static pixel coordinate. The accepted-prediction 95th-percentile pixel error is 38.91px for KLT-FB (and 195.72px for forward KLT), demonstrating severe residual failures.
3. **Global background affine is not physical camera motion.** ORB+RANSAC first/last 2D background registration passes the predeclared quality filter in **6/8 TEST scenes** (1/4 DEV scenes). It is a **dominant image-plane projection diagnostic** that may mix parallax, independently moving objects, zoom and camera motion. No background-relative world-direction gold labels or calibrated 3D scene/camera poses exist here, so no claim of real object motion relative to the world is supported. The affine diagnostic was not a fourth optimized prediction method.
4. **No apples-to-apples VLM result.** Earlier SmolVLM2 500M/2.2B scores concern different video multiple-choice direction questions, with no first-mask point-identification interface or DAVIS centroid gold. Reporting those numbers alongside this PCK task as a VLM baseline would be scientifically invalid. A *new* independent visual grounding/point-tracking VLM interface and task-matched evaluation set are needed to compare them.
5. **Public benchmark limitation.** DAVIS 2017 images are real, but are not necessarily novel to pretrained models; do not claim foundation-model dataset novelty or external open-world generalization. The first 24 source frames selected per video may under-sample late occlusions and disappearances.

## Run verification and security boundary

The committed plan executed **8/8 stages, every exit code 0**, including 192 media SHA checks, 153 source regression tests both before and after private scoring, DEV digest and result reproduction, blind TEST prediction freeze, TEST scoring, and the source-media/weights/secrets publication guard. The code now verifies the first-frame query mask and metadata against the initial, permitted palette-index annotation and refuses any changed or unpinned predictions. The log/state/source checkpoint reside under `repo-batch36-tested/.automation` on Tang, with the terminal event `ALL_APPROVED_STAGES_DONE`. A completion flag prevents the 10-minute cron from endlessly repeating an empty plan.

No source media, private gold labels, raw trajectories or model weights were pushed. The GPU was not used by this CPU tracking benchmark; the RTX 4090 had no active compute-app processes during the completion check. The frozen test root was not rewritten by the report publication.

## Next falsifiable study (do not tune on Batch 36 TEST labels)

**Bottleneck:** scarce robust feature persistence across difficult natural-object deformations and occlusion, and lack of reliable target identity + calibrated camera-vs-object-motion ground truth. The next study must register a **different real-video source and scene-disjoint DEV/TEST**, deliberately measure out-of-view/occlusion and longer trajectories, and freeze alternative trackers (e.g. identity-aware feature survival, long-range re-detection, or reference-mask prompting) using **DEV only**. Compare unconditional PCK, accepted correctness, false accepts on truly absent targets, abstentions, and per-scene CPU/GPU latency. If a world/background-relative claim is made, obtain independent 3D camera/object motion or corresponding calibrated ground truth first. Do not use Batch 36 scored TEST scenes to choose methods or thresholds.

Run-source commands in an authorized frozen snapshot: `python automation/runner.py status`, `python experiments/davis_real_data.py audit`, `python -m pytest -q tests`. Source requires OpenCV, NumPy, Pillow and pytest; private data and media remain on Tang.
