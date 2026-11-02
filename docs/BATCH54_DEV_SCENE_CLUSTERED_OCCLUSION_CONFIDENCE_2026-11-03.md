# Batch 54 — DEV-only independent-scene occlusion-risk audit

## Research question

Can six natural-video DEV clips support a credible cross-scene safety claim
under the **unchanged pre-registered 10% occluded-frame false-accept budget**?

Prior frozen Batch48B DEV reported 2.0% average occluded false acceptance,
but an independent TEST later measured 22.2%. This already-opened historical
TEST result motivates **examining the evidence unit and statistical support**,
not modifying or selecting tracker parameters. Batch53B is independently
frozen, and this analysis was designed using only DEV artifacts.

## Method

Treat each natural video/action class as one *approximate* independent
scene cluster. The event for a method is whether that scene's observed
occluded false-accept rate **exceeds** the frozen budget of 10%. Entire videos
with zero occluded annotated frames are marked **unevaluable**; treating
them as zero-error scenes would be data fabrication.

For `k` violating scenes among `n` supported scenes, compute a one-sided
Clopper–Pearson 95% upper bound `p_U` by solving

`Pr[Binomial(n, p_U) <= k] = 0.05`.

For zero violations, `p_U = 1 - (0.05)^(1/n)`. To account for the
already-predeclared multi-method DEV selection, apply a conservative
Bonferroni confidence adjustment with family size `m = 9`:
`alpha_per_method = 0.05 / 9`.

This is a bound on the **probability a future scene violates the rate
budget**, not a bound on the false-accept rate of individual frames. It
requires i.i.d. scene draws, which unconstrained natural Kinetics
categories cannot guarantee. Correlated video frames must never be counted
as separate independent trials. These numbers are therefore descriptive
risk evidence **and never certification of deployed safety**.

## Real frozen DEV-only results

Source artifacts:

- Batch48B frozen source `9d86f354f4cbed039ef8ea47f14c8eb37df374b3`.
- Batch53B frozen source `92da2060fbc4c842546f50ed32a0ac066579e51a`.
- Both input selections and scoring artifacts are pinned SHA-256, split
  `dev`; all six scene identifiers must be `kinetics_00`–`kinetics_05`.
- Neither TEST oracle nor TEST predictions or TEST metrics are opened.

| DEV batch | DEV-selected tracker | Occlusion-supported scenes | Scenes without occlusion support | Scenes with >10% false accept | 95% nominal upper bound | 95% nine-method familywise upper bound |
|---|---|---:|---:|---:|---:|---:|
| 48B | `klt_fb` | 5/6 | 1/6 | 0/5 | 45.07% | 64.60% |
| 53B | `appearance_fb_055` | 5/6 | 1/6 | 0/5 | 45.07% | 64.60% |

Even if **no supported scene has an observed safety-budget violation**,
five independent scene samples allow a very large bound. A zero-violation
study would need at least **29 supported independent scenes** just to
bring the uncorrected 95% upper bound for future scene violation probability
below 10%; across nine predeclared candidates, even more are needed.

These calculations demonstrate that observed low DEV mean error and a
successful frozen DEV gate do **not** establish cross-scene reliability.

## Limits and next steps

1. These are not six guaranteed independent random draws, and the
   scene-level event differs from the primary frame-level safety metric.
2. Batch48B is historical/opened; this audit is not a new blind TEST or
   a basis for changing thresholds. Batch53B evaluation remains frozen.
3. Future study should expand independently sampled natural-video action
   classes and especially occlusion-supported scenes, prespecify tracking
   metrics and capacity, freeze algorithm and sample IDs, and evaluate only
   after sealing predictions.
4. The audit is **diagnostic-only**. It does not change the active tracker,
   edit frozen DEV/TEST artifacts, alter a GPU experiment, or force a
   new selection based on previously opened TEST results.

## Reproducibility

Implementation: `experiments/dev_cluster_occlusion_certificate.py`.
Regression: `tests/test_dev_cluster_occlusion_certificate.py`.
The tool accepts only a documented frozen batch root and fixed
`dev_results.json`, `dev_selection.json`, config, and immutable
automation source SHA. It rejects TEST split inputs, hash mismatches,
invalid counts/rates, and replay over an existing output path.

Locally generated immutable outputs on Tang:

- `outputs/research/batch54_dev_cluster/batch48b_dev_cluster.json`,
  SHA-256 `7394f2f4c27b668f986e17d337d3b52eb7110a03bc8aeb7684703e12ad3a0e59`.
- `outputs/research/batch54_dev_cluster/batch53b_dev_cluster.json`,
  SHA-256 `7eeaaa270316214cf86f05cff8612b570f8af6e295133f5f7819cf5b421fd83e`.

Only code, tests and explanatory documentation are published; raw videos,
point trajectories, model weights, private oracles and predictions remain
local on Tang.
