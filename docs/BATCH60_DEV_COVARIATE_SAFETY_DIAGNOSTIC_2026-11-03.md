# Batch 60 — Frozen DEV covariate-stratified occlusion risk audit (2026-11-03)

## Scope, evidence and boundary

This is a **new implementation and real DEV-only analysis** of the previously frozen
Batch 56 natural-video DEV cohort. It is **not** an independent TEST result,
a replacement for the running Batch 59 freeze, a proven mechanism of camera motion,
a calibrated 3D motion benchmark, or production authorization.

Input: `repo-batch56b-tested/outputs/kinetics_batch56/dev_results.json`,
the pinned `dev_selection.json`, frozen `configs/kinetics_batch56.json`,
and the Git HEAD corresponding to Batch 56's frozen source SHA
`0d70b09b2be7287fb2e95be0f9d84a86cf4888b7`.
Hash and complete nine-method, 36-scene selection invariants are checked.
No `test_results`, private TEST labels, TEST predictions, raw videos or weights
are read. The running Batch 59 frozen SHA/plan/cron are unchanged.

Motivation: Batch 58 found that blocked DEV cohort transfer may violate the
10% occlusion false-accept budget, even when method selection on 30 training
scenes accepts the method. Batch 60 asks whether this fragility co-occurs with
measurable changes in illumination, texture, observed 2D frame change or
imperfect dominant-affine proxies. It does **not** fit a risk classifier or
derive a new gate from the same already-studied DEV labels.

## Analysis protocol

Six predeclared continuous covariates are taken from frozen DEV diagnostics:
mean luminance, Laplacian texture variance, adjacent-frame absolute difference,
dominant-affine median 2D translation, background-relative residual, and
affine-explained fraction. Each covariate ranks 36 DEV scenes from smallest
to largest and divides them into deterministic 12/12/12 scene terciles
(scene ID breaks exact ties). **Scene rankings are formed without consulting
the method's success or occlusion labels.** Covariates are observed from
existing DEV diagnostics, not an environment-domain identity ground truth.

For each of the **nine unchanged frozen CPU candidate methods**, every bin
reports scene-macro occlusion false acceptance (only scenes with occluded
frames), frame-pooled micro false acceptance, unsupported scene count,
per-scene >10% violations, visible coverage, Jaccard, and frame support.
A scene with no occlusion is **unknown**, never counted as zero risk.
Scenes with fewer than 20 occluded frames are explicitly flagged.
A tie-aware Spearman coefficient is reported only descriptively, without
IID or multiple-comparison claims.

**Critical measurement distinction:** the background-relative residual
uses an evaluation-time point/affine diagnostic, including DEV ground-truth
trajectories. It is **not an unlabeled, deployment-time background detector**.
Neither its strong association nor its bins can be used directly as a
production gate. A dominant 2D affine model may still lock onto a foreground;
all camera-related features are proxies, not real 3D camera pose or proof
of a static background.

## Actual frozen DEV analysis

All 36 scenes and 9 methods passed the integrity and selection checks.
The frozen selected method is `appearance_fb_055`.

- Overall supported occlusion scenes: **33/36**; unsupported: **3/36**.
- Occlusion false accepts: **317 / 10,797 = 2.936% pooled frame rate**.
- **Scene-macro** occlusion false accept: **5.123%**, with **4/33**
  scenes individually exceeding 10%.
- Worst supported scene: **57.69%** false acceptance.
- Visible coverage scene-macro: **45.36%**; Jaccard scene-macro: **0.3227**.
- Two scenes had fewer than 20 occluded frames and were flagged as weakly
  supported, not discarded from the frozen metric.

For the selected method only (the JSON contains all nine methods):

| Covariate, low / middle / high tercile | Occlusion false accept, scene-macro | Supported scenes | Per-scene >10% violations | Visible coverage |
|---|---|---|---|---|
| Background-relative residual, low | 0.237% | 10/12 | 0 | 57.72% |
| Background-relative residual, middle | 3.667% | 11/12 | 1 | 46.99% |
| **Background-relative residual, high** | **10.528%** | **12/12** | **3** | **31.36%** |
| Mean luminance, low | 5.780% | 11/12 | 1 | 43.65% |
| Mean luminance, middle | 7.016% | 10/12 | 2 | 49.24% |
| Mean luminance, high | 2.943% | 12/12 | 1 | 43.18% |
| Laplacian texture, low | 3.932% | 12/12 | 1 | 41.36% |
| Laplacian texture, middle | 2.985% | 11/12 | 1 | 35.72% |
| Laplacian texture, high | 8.904% | 10/12 | 2 | 58.98% |
| Dominant-affine 2D translation, low | 3.645% | 10/12 | 1 | 50.64% |
| Dominant-affine 2D translation, middle | 8.444% | 12/12 | 2 | 50.54% |
| Dominant-affine 2D translation, high | 2.843% | 11/12 | 1 | 34.88% |

Selected method, supported-scene Spearman rank correlation with
occlusion error rate: background-relative residual **+0.273**,
mean luminance **+0.312**, texture **-0.382**, frame appearance change
**-0.086**, affine translation **-0.009**, affine-explained fraction **-0.060**.
The sign and strength of a pooled rank coefficient do not establish
causality or reliable predictive generalization.

## Negative results and scientific interpretation

The strongest descriptive contrast is across the **evaluation-time
background-relative residual**: high-residual DEV scenes show
**10.53%** scene-macro occlusion error and **31.36%** coverage, versus
**0.24%** error and **57.72%** coverage at low residual.
Three of four observed >10% scene failures are in the high-residual tercile.
This identifies a useful **diagnostic association**, not a gating algorithm.

Notably, the relation with raw affine translation is non-monotonic;
lower illumination and lower texture do not uniformly predict failures,
and no one-dimensional feature cleanly certifies safety. Moreover, the
support count is **33 natural scenes**, not 10,797 independent statistical
replicates. Additional scene diversity and occlusion coverage are needed.

Because Batch 56 DEV has already been studied, this audit is **exploratory**.
It neither changes Batch 59 method selection nor draws an independent
TEST conclusion. Future prospective work should predeclare an **available
at inference** image-only risk proxy, compare it against a truly independent
class-disjoint TEST, and report coverage lost to abstention alongside
the number of unsafe accepted cases. The residual using DEV ground truth
must **never** silently become such an inference proxy.

## Reproduction, tests and immutable local output

Files:
`experiments/batch60_dev_covariate_safety.py`,
`tests/test_batch60_dev_covariate_safety.py`.

On Tang, in the independent Batch 60 development worktree:

```bash
../.venv/bin/python experiments/batch60_dev_covariate_safety.py
../.venv/bin/python -m pytest -q tests/test_batch60_dev_covariate_safety.py
../.venv/bin/python -m pytest -q
```

Tests cover synthetic stratification, metric denominators, unsupported
occlusion, low support, deterministic ties, malformed covariates,
inconsistent counts, frozen source/selection pins, immutable output
and refusal to use unpinned altered DEV data.

**17/17 new focused tests passed; 900/900 full-project tests passed**;
3 existing Pillow deprecation warnings, 0 test failures.

Local output (not committed):
`repo-batch60-dev/outputs/research/batch60_dev_covariate_safety.json`
plus its SHA-256 sidecar. Digest:
`221e73eb0db0f56ce7931f186a8db464a743f23305b87e44b514a5ffe4c78ae6`.

The analysis did not modify Batch 59's source, frozen plan, cron,
downloads, model processes, prior logs or media. No model weights,
videos, private annotations, secrets or experiment output are committed.
