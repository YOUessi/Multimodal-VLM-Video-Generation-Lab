# Batch 57 — cross-cohort DEV-only occlusion safety falsification

## Purpose and non-negotiable boundary

This batch is an **exploratory research audit**, not a new independent TEST.
It uses *only* the already-frozen DEV evaluation results from Batch 48B,
Batch 53B and Batch 55, with six natural-video DEV scenes per cohort
(18 DEV scenes total). Each cohort's methods, selection rule and published
source SHA are unchanged. There are no TEST file paths, TEST oracle reads,
TEST metric inputs, new model training or new candidate thresholds.

Importantly, these three DEV cohorts have **already been studied**. The
cross-cohort holdouts below are DEV holdouts, **not previously unseen blind
test sets**, and cannot support independent generalization claims. They
must not be used to tune the running Batch 56 or to replace its sealed plan.

## Protocol

For each of three folds, pool the two other cohorts' six-scene DEV
evaluations (12 training scenes), apply the original nine-candidate
selection rule with >=30% visible-target coverage, <=10% scene-macro
occlusion false acceptance, then maximize DEV scene-macro Jaccard
(tie-break by visible coverage; preserve original candidate order).
Evaluate the selected method on the **six held-out DEV scenes**. Record
the number of scenes with nonzero occlusion support and the number whose
observed occlusion false-accept rate exceeds 10%. An unsupported scene
is neither safe nor unsafe and cannot inflate the denominator.

Before any fold calculation, independently recompute each cohort's
recorded original DEV choice and eligible candidate list, verify that the
recorded selection pins its DEV-results SHA-256, validate selection
pin and config SHA-256, verify matching source and config provenance,
reject wrong splits and invalid counts, and reject policy drift.

## Real frozen-DEV evidence

| DEV cohort held out | Method selected using the other 12 DEV scenes | Held-out DEV occlusion false-accept, scene-macro | Held-out DEV visible coverage, scene-macro | Scene-level >10% violations | Supported occlusion scenes |
| --- | --- | ---: | ---: | ---: | ---: |
| Batch 48B DEV | klt_fb | 2.00% | 56.70% | 0 | 5/6 |
| **Batch 53B DEV** | **klt_fb** | **12.03%** | **57.90%** | **2** | **5/6** |
| Batch 55 DEV | klt_fb | 1.17% | 72.33% | 0 | 6/6 |

The adverse 53B DEV held-out fold has violations at
`kinetics_00` and `kinetics_03`. The **training** DEV scene-macro
occlusion false-accept rate for the candidate selected in that fold
was **1.55%**, despite **12.03%** on the held-out DEV cohort.

For context, the original in-cohort DEV selection from Batch 53B was
`appearance_fb_055`, with 0.206% DEV scene-macro occlusion false
acceptance. That number is **in-sample** and is not a fair held-out
comparison with the cross-cohort method.

The nominal one-sided 95% Clopper-Pearson upper bounds for the future
scene budget-violation probability are **45.07%** (48B: 0/5),
**81.07%** (53B: 2/5) and **39.30%** (55: 0/6). These bounds are
descriptive only and require unsupported independent/exchangeable
scene sampling assumptions. They do **not** bound individual-frame
failure or establish real-world safety. The small number of clusters
is a major limitation.

## Interpretation

A candidate may meet the frozen average DEV gate across two natural-video
cohorts and nevertheless fail the exact same gate in the third DEV cohort.
The evidence directly challenges reliance on a pooled or small-cohort
DEV scene-macro constraint as a reliable predictor of domain-shift safety.

No changes to tracker hyperparameters, camera compensation, VLM,
dataset assignment, candidate list, gating thresholds or frozen Batch56
evaluation policy are justified by this finding. In particular, there
is **no demonstrated performance improvement** here.

The next worthwhile research step is to evaluate scene-level selection
stability and risk on the genuinely new **36-scene Batch56 DEV** after it
finishes. Preserve its predetermined TEST firewall and use the frozen
Batch56 reporting/audit code; do not make a post-hoc selection change.
Any prospective new algorithm must be selected using a separate
development population before access to a genuinely new independent TEST.

## Reproducible code, tests and hashes

Implementation:
`experiments/dev_cross_cohort_safety_audit.py`

Synthetic tests:
`tests/test_dev_cross_cohort_safety_audit.py`

Reproduce on the authorized Tang machine (from this repository root,
with frozen sibling worktrees present):

```bash
../.venv/bin/python experiments/dev_cross_cohort_safety_audit.py \
  --root .. \
  --output outputs/research/batch57_dev_cross_cohort.json
../.venv/bin/python -m pytest -q
```

DEV sources (only the DEV-results SHA-256 is listed, not media or labels):
- Batch 48B DEV: `917ebf34b9c98dfb339d56ef5474400e9c661688df9c038fb71fe0eed01a6644`.
- Batch 53B DEV: `4a2b2eefb77a8cc585c90d19a3f9085e94aa534f863ff420022a4e322ad498f5`.
- Batch 55 DEV: `e86508ef7a23e7a602cb9889be0f7bbe46651d784cb1b64e9b22aba2f69fb765`.

Immutable DEV-only output:
`outputs/research/batch57_dev_cross_cohort.json` (local only),
SHA-256 `b326d1f0b805e5355699e2201062c6bd047edcb8ecd3a58318746b8d7ad24bd5`.
The code writes it once and refuses an incompatible overwrite.

Checks on Tang: **17/17 new synthetic regression tests passed**;
**817/817 full project regression tests passed** with three existing
Pillow deprecation warnings. No TEST labels or outputs were read,
and no videos, weights, media or secrets are part of this publication.

Batch56 remains a separate frozen 23-stage workflow at its original
`0d70b09b2be7287fb2e95be0f9d84a86cf4888b7` SHA, managed by
the existing sole `MVVG_PROJECT4_10MIN` cron. This Batch57
exploratory audit does **not** deploy a competing runner or cron.
