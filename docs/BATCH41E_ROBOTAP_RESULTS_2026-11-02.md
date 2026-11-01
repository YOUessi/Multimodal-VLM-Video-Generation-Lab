# Batch 41E — RoboTAP repaired evaluation and failure analysis

Date: 2026-11-02

## Source and execution

- Frozen source SHA: `f434d37cdad13c06a50b3da5d379e772b7a5fe38`
- GitHub main at freeze: `f434d37cdad13c06a50b3da5d379e772b7a5fe38`
- Tang execution directory: `repo-batch41e-tested`
- Stages: 17/17 completed, all return codes 0
- Regression: 281 passed, 3 Pillow deprecation warnings
- Final source/data/prediction audit: passed; 20 scenes audited with source-shard verification
- Publication guard: 205 tracked files checked, 0 forbidden tracked files

Batch41D's R06 failure was caused by a public-query path mismatch: the manifest and frames came from `batch41_grouped_frozen`, while the evaluator read queries from the legacy `batch41_frozen` directory. The failed Batch41D directory and logs remain preserved. Batch41D stopped before DEV scoring, TEST prediction, or any TEST-label read.

Batch41E fixes the lookup to use `src.frozen_dir(cfg)`, adds a regression with a conflicting legacy-path decoy, and writes to a new output directory. It reuses the exact untouched episode-disjoint Batch41B RoboTAP freeze; it is a repaired continuation, **not a second independent split**. The prior TEST portion had not been read before this run.

## Frozen split and protocol

- DEV / TEST scenes: 6 / 14
- DEV / TEST point queries: 24 / 56
- The source freeze uses two of five RoboTAP archive shards.
- DEV was the only source for method selection and the VLM interface gate.
- TEST CPU predictions were sealed before the first TEST oracle score.
- The VLM DEV gate failed, so TEST VLM inference was skipped. No TEST VLM prediction or accuracy was produced.

RoboTAP is synthetic robotics video, not natural camera footage. Its explicit point/occlusion labels make it useful for controlled occlusion tests, but it does not establish natural-camera generalization.

## DEV-only selection

The predeclared rule required visible coverage >= 0.40 and occluded false-accept rate <= 0.25. No candidate passed both constraints:

| Method | Jaccard@8 | Visible coverage | Occluded false accept |
|---|---:|---:|---:|
| KLT forward-backward | 0.5280 | 0.5971 | 29.66% |
| Template recovery | 0.4472 | 0.6151 | 30.16% |
| Affine-stabilized KLT | 0.4657 | 0.5420 | 30.04% |

The predeclared fallback, KLT forward-backward, was selected. This decision was recorded before TEST predictions.

## TEST point-tracking results

TEST contains 7,517 visible positions and 5,629 occluded positions across 14 scenes.

| Method | PCK@8 visible | Jaccard@8 | Visible coverage | Accepted precision@8 | Occluded false accept | Occlusion accuracy | CPU P50/P95 ms/query |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stationary point | 50.51% | 22.51% | 100.00% | 28.88% | 100.00% | 57.18% | 0.044 / 0.099 |
| KLT forward | **60.90%** | 34.91% | 85.49% | 44.99% | 66.60% | 63.18% | 28.309 / 72.757 |
| KLT forward-backward | 56.05% | **49.75%** | 60.49% | 81.57% | 10.98% | **72.71%** | 33.862 / 83.149 |
| Template recovery | 51.74% | 39.69% | 63.04% | 63.03% | 25.42% | 67.98% | **8.749 / 29.251** |
| Affine-stabilized KLT | 50.17% | 46.73% | 52.60% | **87.23%** | **6.56%** | 70.09% | 83.134 / 172.208 |

KLT forward has the highest visible-only PCK but also a 66.60% occluded false-accept rate. KLT forward-backward provides the best Jaccard and occlusion accuracy at a moderate CPU cost. Affine stabilization reduces false accepts further but is slow and abstains on more visible positions. No method dominates all operating points.

### Descriptive paired scene bootstrap

The generated report's original interval compared the selected method to `klt_fb`; because the selected method is also `klt_fb`, that interval is tautologically [0, 0] and is not an informative uncertainty estimate.

A separate post-score, descriptive 14-scene paired bootstrap (not used for method selection) gives KLT-FB Jaccard@8 differences:

| Contrast | Mean difference | 95% scene-bootstrap interval |
|---|---:|---:|
| KLT-FB minus KLT forward | +0.0660 | [+0.0111, +0.1191] |
| KLT-FB minus template recovery | +0.0232 | [-0.0707, +0.1013] |
| KLT-FB minus affine-stabilized KLT | +0.0700 | [+0.0246, +0.1236] |
| KLT-FB minus stationary point | +0.1840 | [+0.0680, +0.3151] |

The small number of independent scenes makes these intervals descriptive, not a broad-domain guarantee.

## Background-relative diagnostic

- Adjacent-pair affine acceptance: 97.71%
- Mean per-scene background-relative residual: 2.64 px
- Interpretation: 2-D background-affine proxy only; not camera pose or world-relative 3-D motion

Low background residual did not guarantee safe point identity. For example, TEST scene_010 had 0.24 px mean background residual but a 63.16% KLT-FB occluded false-accept rate; scene_013 had 0.40 px residual and 100% false accepts. Scene_017 had 11.79 px residual and only 19.16% visible coverage. Camera/background alignment and target identity/occlusion are separate failure axes.

## SmolVLM2-500M visibility baseline

- DEV queries: 48
- Invalid replies: 48/48 (100%)
- Coverage: 0%
- DEV gate: blocked
- TEST VLM: skipped by the DEV gate

Reply distribution: 34 `VISIBLE, ABSENT`, 13 `VISIBLE.`, and 1 `VISIBLE, ABSENT,`. Punctuation normalization alone could at most rescue the 13 unambiguous replies; it would not resolve the ambiguous dual-label replies or satisfy the predeclared gate. No VLM TEST accuracy claim is made.

## Next research bottleneck

1. Replace the weak synthetic-only continuation with new, sequence-disjoint DROID recordings that include calibrated multi-view RGB, camera transforms, 3-D point tracks and depth snapshots.
2. Test a pairwise ORB-affine motion prior for KLT, measuring its incremental CPU cost and per-scene coverage/false-accept trade-off. Keep the existing image-plane and background-relative diagnostics distinct from world-motion claims.
3. Strengthen the VLM DEV gate to include a predeclared DEV accuracy floor as well as format validity; a valid coordinate string with zero PCK must not automatically pass to TEST.
4. Correct the report's tautological selected-versus-itself bootstrap comparison.

No TEST result from Batch41E is used to tune the next batch. New DROID episodes and a new frozen split are required.
