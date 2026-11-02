# Batch 58 — 36-scene DEV-only grouped selection transfer audit (2026-11-03)

## Purpose and boundary

Evaluate whether the **already frozen Batch56 DEV method-selection rule** remains
stable when a contiguous group of six natural scenes is absent from method
selection. This is **exploratory**, not a new blind TEST, not deployment
authorization, and not a change to Batch56's running 23-stage frozen plan.
The previously studied CLEVRER set is not read.

Source of measurement: Batch56's immutable `dev_results.json` and its pinned
`dev_selection.json`, from freeze SHA
`0d70b09b2be7287fb2e95be0f9d84a86cf4888b7`.
The audit verifies the selection pin, frozen source HEAD, config hash,
all nine candidate methods, 36 scene identities, per-scene count and rate
consistency, and the original full-DEV selection. Only **DEV** results are read.
No files named `test_results`, private oracles, TEST predictions, model weights,
or video content are opened.

## Methods

The original selection rule is reproduced exactly:
eligible methods must reach scene-macro visible coverage >= 30% and
scene-macro invisible false acceptance <= 10% (the latter omits scenes
with no invisible frames). Among eligible methods choose the largest
scene-macro mean TAP-Vid Jaccard, breaking ties with visible coverage.
If none passes, use the previously frozen fallback `appearance_fb_085`
and report the fallback explicitly.

Two descriptive views:

1. **Six blocked folds**: hold out scene IDs 00–05, 06–11, ..., 30–35,
   select a method on the other 30 DEV scenes, and evaluate that method
   on the six held-out DEV scenes.
2. **36 leave-one-scene-out folds**: select on 35 DEV scenes and evaluate
   on the missing DEV scene. This is a sensitivity check, not the primary
   group-shift diagnostic.

These folds were defined as source-code rules and are **not independent
prospective trials**. We do not use either view to tune methods,
thresholds or the active frozen Batch56 plan.

## Actual 36-DEV-scene results

| DEV analysis | Selected method frequency | Mean held-out occlusion false accept | Scenes above 10% | Held-out supported scenes | Mean held-out coverage | Mean held-out Jaccard |
|---|---|---:|---:|---:|---:|---:|
| 6 blocked groups | `appearance_fb_055` 5/6, `klt_fb` 1/6 | **12.2804%** | **6/33** | 33/36 | 48.3437% | 0.324756 |
| Leave-one-scene-out | `appearance_fb_055` 36/36 | 5.1227% | 4/33 | 33/36 | 45.3561% | 0.322706 |
| Original full DEV selection, not a holdout | `appearance_fb_055` | 5.1227% | 4/33 | 33/36 | 45.3561% | 0.322706 |

**Critical blocked fold: 12–17.** Excluding these six scenes caused the
frozen rule to choose `klt_fb`, which had training-set false acceptance
**9.87%**, just within the 10% admission budget. On the held-out group,
the same method's scene-macro occlusion false acceptance rose to
**49.29%**, with **four scene-budget violations** (14–17).
Its held-out visible coverage reached 74.33%, illustrating the
safety-versus-coverage tradeoff rather than a genuine safe gain.

The other blocked folds chose `appearance_fb_055`. Fold 06–11 held out
a budget failure at scene 10, and fold 24–29 found a failure at scene 26.
In contrast, leave-one-scene-out never switched the selected method,
so it failed to reveal the dramatic group-level selection instability.

**Three of the 36 DEV scenes have no annotated invisible frames.**
They are **not** counted as safe successes and are omitted from the
occlusion-rate mean and scene-violation denominator. The held-out
occlusion means are **macro over supported scenes**, not the
frame-weighted pooled rate. This distinction matters: the grouped
folds' pooled raw counts were 661 false accepts out of 10,797
invisible target frames, but these counts do not determine scene-macro
admission.

## Interpretation and limitations

The original all-DEV selected method `appearance_fb_055` still has four
observed >10%-error scenes and a maximum scene false acceptance of
57.69%. Group shift can additionally change which method is selected,
producing an even more severe held-out failure despite the method
passing the training-group average constraint.

This is a **new diagnostic on the 36 DEV videos of Batch56**,
not a new independent TEST outcome, not a statistical proof of
real-world violation probability, and not a calibrated 3D camera-motion
measurement. The source videos may be correlated and action classes
are not guaranteed to sample the target deployment distribution.
Blocked folds are ordered by stable selected scene IDs; results
are sensitive to this grouping. No confidence interval is treated as
valid with overlapping training folds.

The separate Batch56 evaluation already reports 2D screen motion and
background-relative affine-proxy diagnostics; these do **not** certify
actual 3D camera motion, depth parallax or static background identity.
The present audit specifically addresses safety-constrained method
selection instability, not a new camera-compensation algorithm.

## Reproducibility, tests and publication

Implementation: `experiments/batch58_dev_grouped_transfer.py`.
Synthetic adversarial tests:
`tests/test_batch58_dev_grouped_transfer.py`.

Run on Tang:

```bash
cd /home/you/projects/Multimodal-VLM-Video-Generation-Lab/repo-batch58-dev
../.venv/bin/python -m pytest -q tests/test_batch58_dev_grouped_transfer.py
../.venv/bin/python experiments/batch58_dev_grouped_transfer.py
../.venv/bin/python -m pytest -q
```

Immutable generated output (never committed):
`outputs/research/batch58_dev_grouped_transfer.json`, with adjacent
`.json.sha256` pin. Result digest:
`ca284e9eceb8017763cda053adae24d0ece1ddbe965dbd48027bcca9180451e0`.

Adversarial tests exercise missing or forged pins, TEST-split rejection,
method missing, source-method selection inconsistency, falsified count
and rate fields, unsupported occlusion scenes, fallback selection,
and synthetic group-specific shift.

## Next scientific decision

Do **not** incorporate the held-out DEV findings into the ongoing Batch56
frozen method choice. Finish its one-time independent TEST scoring under
the existing SHA and only then compare the original selected method's
TEST safety outcome with these prospective (DEV-derived) diagnostics.
For future separately frozen experiments, define **group-level safety
selection criteria and validation folds before collecting any new TEST
results**. A new freeze would also require a capacity check and
mutual-exclusion checks before GPU use.

No existing video, checkpoint, historical data, failed log or frozen plan
was deleted or modified during Batch58.
