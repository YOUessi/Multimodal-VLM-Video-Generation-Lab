# Batch 61 — Image-only occlusion-risk proxy: blocked DEV cross-fit and privileged negative control (2026-11-03)

## Purpose and strict evaluation boundary

This batch asks whether *available-at-inference* image-only scene descriptors can rank unsafe occlusion scenarios without using ground-truth target motion. It builds and **actually evaluates** a conservative ridge-logistic risk predictor, a training-fold prior baseline, and a deliberately privileged diagnostic negative control.

The input is **only the already-examined Batch 56 frozen DEV cohort** (36 natural videos, 9 unchanged tracker candidates). The original DEV-selected method is `appearance_fb_055`. The source is pinned to Batch 56's frozen commit `0d70b09b2be7287fb2e95be0f9d84a86cf4888b7`; both DEV result and configuration hashes, the pinned method selection, the complete 36×9 method counts and current frozen HEAD are verified. No TEST data, TEST answers, unsealed TEST predictions, model weights or videos are read.

**This is exploratory only, NOT a new blind independent evaluation.** The original tracker selection already used all 36 DEV scenes. Cross-fitting a second-stage risk estimator does **not** remove that selection bias.

## Implemented protocol

The four predeclared image-only covariates are mean luminance, Laplacian variance, adjacent-frame appearance change and dominant 2-D affine translation. They use frames, not target ground-truth trajectories. All continuous values are validated as finite; neither new thresholds nor tracker parameters are tuned.

The label is a *scene-level* event: the frozen method's occlusion false-accept rate exceeds **10%**. A scene with no annotated occluded frame has an **unknown label**, never a safe zero.

Six contiguous scene-ID blocks of six scenes are held out in turn. The **training** portion alone determines feature means, standard deviations, ridge-logistic coefficients and a Laplace-smoothed risk prior. L2 slope penalty is fixed to **5.0**, max Newton iterations to **60**, and no post-hoc hyperparameter search is performed. One highest predicted-risk scene is flagged per held-out six-scene block, with lexical tie breaking; this gives a predeclared **6/36 (16.7%)** scene alert budget. Alerts use predictions only, not held-out labels. This is diagnostic scene ranking, not real-time per-frame abstention.

A separate negative control uses a single **privileged** background-relative displacement residual that depends on DEV GT target trajectories. It is explicitly *not* an inference-accessible feature, not a usable online camera-motion signal and not a candidate for deployment. Its inclusion is a leakage hazard demonstration/ablation, not a method recommendation. The dominant 2-D affine fit itself does not establish true static background, camera pose, parallax or 3-D motion.

A prediction is scored only when its held-out scene has annotated occlusions. Brier loss and pooled scene AUC are descriptive, not a statistically independent estimate or guaranteed classifier calibration. The training-only class prior uses the same held-out partition for comparison.

## Actual results on 36 DEV scenes

For `appearance_fb_055`, **33/36** scenes contain any occlusion evidence, with **4** scenes exceeding the 10% risk budget. The 3 unsupported scenes are unknown. The four unsafe IDs are `kinetics_10`, `kinetics_14`, `kinetics_17`, and `kinetics_26`.

| Diagnostic (6 fixed group holdouts) | Image-only 4-feature model | Privileged 1-feature negative control |
|---|---:|---:|
| Supported occlusion scenes | 33 | 33 |
| Scenes violating 10% budget | 4 | 4 |
| Out-of-fold AUC (descriptive) | **0.2931** | **0.1379** |
| Out-of-fold Brier | **0.114675** | **0.117216** |
| Train-fold prior Brier | **0.114153** | **0.114153** |
| Flagged scenes | 6/36 | 6/36 |
| Violations caught by alerts | **0/4** | **0/4** |
| Alerts with no occlusion evidence | 1 | 0 |

The image-only alert IDs were `kinetics_01`, `kinetics_07`, `kinetics_16`, `kinetics_20`, `kinetics_27`, `kinetics_32`. **None** is among the four known violating scenes. The out-of-fold image-only Brier score is slightly worse than using a simple per-fold risk prior. Neither approach establishes meaningful safety screening.

The privileged residual, despite its prior *descriptive* association with failure in Batch 60's tercile analysis, performed even worse here, showing that correlation, target-dependent features, and out-of-fold predictive discrimination cannot be conflated. It would remain prohibited as an online input even if it had performed better.

The positive labels are highly concentrated: six training-fold positive counts were `[4, 3, 2, 4, 3, 4]`, with only four unsafe scenes in total. Reported model behavior is highly sensitive to those scenes; sharing examples across training folds does not produce six independent validations. No claim of statistical significance or calibrated deployment risk is justified.

## Negative conclusions and consequence

1. **No new risk gate should be deployed** based on these four image-only scene covariates: the fixed alerts missed all four known violations and Brier loss was worse than the training-only prior. This is an actual failed hypothesis, not an improvement.
2. Even a GT-dependent residual with an apparent DEV association did not generalize across the grouped DEV holdouts. It remains strictly forbidden at inference.
3. A full scene descriptor cannot describe short occlusion intervals or verify feature-point identity. The next algorithmic direction should distinguish what can be observed *at decision time*, such as independently verified background anchors or visibility and occlusion cues, from privileged evaluation metrics.
4. Grouped DEV folds from the already-studied cohort cannot replace the currently frozen Batch 59 independent video evaluation; do not change Batch 59 selections, scores, or input data.
5. A scene with no ground-truth occlusion is *unsupported*, not "no false accepts"; the count of scenes, not the count of frames, limits scene-level generalization claims.

## Implementation, evidence, tests, reproduction

Implementation: `experiments/batch61_dev_unlabeled_risk_crossfit.py`. Adversarial regression: `tests/test_batch61_dev_unlabeled_risk_crossfit.py`.

```bash
cd /home/you/projects/Multimodal-VLM-Video-Generation-Lab/repo-batch61-dev
../.venv/bin/python -m pytest -q tests/test_batch61_dev_unlabeled_risk_crossfit.py
../.venv/bin/python experiments/batch61_dev_unlabeled_risk_crossfit.py
../.venv/bin/python -m pytest -q
```

The focused suite covers feature-source boundaries, GT-dependent negative control separation, train-only preprocessing, held-out label isolation, six fixed alerts, unsupported scenes, pathological constant-feature/all-negative/all-positive folds, non-finite covariates, invalid shapes/labels, fixed safety budget, pinned inputs and immutable output. An initial isolated test failed because its synthetic report stub lacked a required printing field; the test was corrected, then **28/28 new focused tests passed**.

**928/928 complete regression tests passed, 0 failures, 3 existing Pillow deprecation warnings.**

Immutable local output (ignored by Git):
`repo-batch61-dev/outputs/research/batch61_dev_unlabeled_risk_crossfit.json`
with SHA-256 sidecar; SHA-256 **`f06dad7f75e92c56a7b527a795a23d1b8900ba289aff5741fc1f10a11f879a3d`**.

The ongoing Batch 59 frozen plan, scheduler, GPU allocation, previous results, data archives, model weights, media and secrets were left untouched. Only Batch 61's source, tests and this document are intended for GitHub publication.
