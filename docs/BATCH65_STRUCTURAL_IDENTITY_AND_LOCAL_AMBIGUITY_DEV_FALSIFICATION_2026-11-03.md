# Batch 65 — DEV structural identity and local ambiguity veto (2026-11-03)

## Status: falsified; do not deploy

**This is exploratory DEV-only analysis on 36 previously studied Batch 56 scenes, NOT a fresh blind test.** It neither updates the tracker nor changes the frozen Batch 59 protocol. It tests whether direct grayscale **normalized cross-correlation (NCC)** of a target's source patch, plus competition from plausible local lookalikes, can reject persistent co-drift without relying on private oracle information at inference.

The observed result is negative: even stringent combinations leave the **same 117-frame continuously false-accepted occlusion episode** in `kinetics_17`. The error-rate reduction relative to template consensus is tiny while correct visible tracks and Jaccard decrease substantially.

## Frozen inputs, access control, and feature provenance

- GitHub sole source: `YOUessi/Multimodal-VLM-Video-Generation-Lab`, parent published main `f2001e91f84e0573f146054d0972f355f30761c9`; Batch 65's local development base has the exact same **tree SHA** as published main, although connector publishing gave the parent commit a different commit SHA.
- Batch 56 frozen, pinned DEV predictions: `repo-batch56b-tested/outputs/kinetics_batch56/dev_predictions.json`; selected tracker `appearance_fb_055`. Source, query and manifest SHA-256 pins validated against prior frozen data.
- Batch 56 frozen, public query manifest and 36 DEV frame arrays (36 videos, 144 queries); every image array is SHA-256 checked against the public manifest.
- **Phase 1 (label blind):** creates NCC/ambiguity feature data using only public queries, pinned frames and frozen *unscored* DEV predictions. It does not read the DEV oracle, DEV result file, DEV method selection, or TEST predictions/oracle. The artifact is exclusive-create and SHA-256 pinned before scoring.
- **Phase 2 (DEV scoring only):** verifies the feature sidecar pin, DEV scoring/selection pins, and uses the tested DEV private oracle streaming reader. It stops at `kinetics_35` without reading/parsing any TEST object. It replays all frozen per-scene baseline counts exactly. Score output is exclusive-create pinned.
- Batch 59 and its 10-minute frozen cron remain untouched. No new GPU experiment is started. **No test-derived thresholds.**

## Fixed method, no oracle-dependent features

For each query in the query frame, extract an 11×11 grayscale patch at the public query coordinate. If patch standard deviation is less than 5 pixel levels, the structural matcher abstains. Otherwise, on each *current* frame with a base-track proposal, compute normalized zero-mean NCC over the 39×39 neighborhood centered on the predicted location using OpenCV `TM_CCOEFF_NORMED`. Only in-bounds patches are accepted for evaluation.

The **candidate NCC** is the correlation score at the proposed coordinate. The **distinctiveness margin** is this score minus the highest score among rivals at least seven pixels away but no more than 13 pixels from the proposal. This margin captures only **local** ambiguity; a rival farther away is invisible to this test. This feature does **not** verify object identity or distinguish camera motion from object/background motion. Query reference is never updated with future frames, and no future frames influence the current decision.

Predeclared variants (veto-only, never change coordinates): baseline; existing 4px template-consensus veto; standalone NCC ≥0.50; template + NCC ≥0.50; template + NCC ≥0.50 + margin ≥0.05; and template + NCC ≥0.70 + margin ≥0.10.

**Important:** all candidate thresholds are *research diagnostics*, not optimized or production-approved. Even this label-blind feature extraction is not evidence of independent validation, because the cohort has already been examined repeatedly.

## Actual results (36 previously studied DEV scenes)

| Variant | Scene-macro visible coverage | Scene-macro occluded false acceptance | Scene-macro Jaccard | Scenes >10% budget (33 supported) | Max consecutive wrong-accept frames |
|---|---:|---:|---:|---:|---:|
| Frozen baseline | 45.36% | 5.123% | 0.3227 | 4 | 117 |
| Template 4px | 34.09% | 1.165% | 0.2695 | 1 | 117 |
| NCC ≥0.50 only | 32.99% | 2.403% | 0.2421 | 3 | 117 |
| Template + NCC ≥0.50 | 26.25% | 1.093% | 0.2084 | 1 | 117 |
| Template + NCC ≥0.50 + margin ≥0.05 | 18.91% | 1.055% | 0.1627 | 1 | 117 |
| Template + NCC ≥0.70 + margin ≥0.10 | 16.88% | 1.055% | 0.1471 | 1 | 117 |

Relative to template-only, template+NCC ≥0.50 eliminates **5** more wrong accepted occluded frames, but sacrifices **1,690** correctly tracked visible frames. Adding a ≥0.05 distinctiveness margin eliminates **8** wrong accepted occluded frames while sacrificing **3,039** correct visible frames. The strictest variant also eliminates just **8**, while sacrificing **3,435** correct frames.

**Failure concentration:** `kinetics_17` remains at 26.43% false acceptance for all template+structural variants and keeps the 117-frame continuous episode. The worst scene is unchanged. This is consistent with multiple methods agreeing on a visually plausible but wrong candidate, but the mechanism is not proven to be a particular distractor without additional independent identity evidence.

The image feature extraction wall-clock time is recorded per DEV scene in the local result. This excludes tracking, decode, hashing, device-to-device transfers, and full deployment; **do not quote as end-to-end CPU latency**. No new 3D camera pose or true object displacement is inferred.

## Outcome, limitation, next independent experiment

**Reject deployment of NCC or local ambiguity veto.** New evidence in this study shows severe loss of useful coverage with failure to eliminate the persistent episode. Continuing to tweak thresholds on these DEV scenes would overfit.

Prioritize (1) finishing Batch 59's frozen prospective class-disjoint independent experiment and safety audit; (2) independent labeled target visibility and identity mechanisms with explicit provenance and adversarial co-drift controls; (3) new, never opened class-disjoint DEV/TEST data and pre-registered performance/safety criteria for future candidates. Do not read TEST labels during development. The existing dataset size and potential correlations also prevent claims of universal 10% scene-level safety certification.

## Reproduction and immutable local evidence

In the independent Batch 65 checkout on Tang, with frozen Batch 56 data present:

```bash
../.venv/bin/python -m pytest -q tests/test_batch65_dev_structural_identity.py
../.venv/bin/python -m experiments.batch65_dev_structural_identity features
../.venv/bin/python -m experiments.batch65_dev_structural_identity score
../.venv/bin/python -m pytest -q
```

Exclusive-create output requires a new isolated research checkout on repeat, not overwriting local artifacts.

- Local, **not committed**: `outputs/research/batch65_unlabeled_dev_structural_identity.json`, SHA-256 `f39b8efea22187bfaa9543ef8a2ccab65e772a479df3b01c51aafefaac597bbc`.
- Local, **not committed**: `outputs/research/batch65_dev_structural_identity_report.json`, SHA-256 `07742df9c3a8ce6395b9a24a37a7dfdb93f3f4670dcbc0fce8dad21859fdbe05`.

Only implementation, unit/regression tests and this report should be published. No private oracle, frame arrays, videos, weights, credentials, feature artifacts or raw outputs are committed.
