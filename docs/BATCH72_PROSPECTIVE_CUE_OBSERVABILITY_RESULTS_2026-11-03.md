# Batch 72 — Prospective cue observability and post-occlusion identity recovery: results

**Date:** 2026-11-03  
**Frozen algorithm/protocol GitHub main commit:** `9690e6fed262d8934ac6bbffa8e66614ae5119ac`  
**Frozen source tree:** `c312c98baa4151caf7fbf165f392b459701fa974`  
**Evaluation:** Tang, Ubuntu 22.04, CPU-only. No concurrent GPU experiment and no changes to Batch 59's freeze, cron, datasets or jobs.

## 1. Research question and prior evidence

Batch 71 demonstrated that if two physical identity assignments yield identical
visible pixel sequences, a visual tracker cannot infer which object is the
original target after complete occlusion. The next scientific question is
**which observable identity information restores post-occlusion reacquisition,
and whether a label-blind matcher can refuse visually ambiguous duplicates**.

We test a fixed **query-template identity memory + runner-up veto** against a
same-template best-match baseline and the already frozen
`appearance_fb_055` and `template_recover4` trackers.

This is a prospective controlled **synthetic image-plane experiment**, not a
natural-video independent transfer result. Do not equate two assignments in a
pair with two independent trials. The hardest 117-frame shared-failure natural
DEV case is **not** represented here.

## 2. Preregistration and seal

The code, 36 new targeted unit tests and protocol were frozen in GitHub
`main` **before** any Batch 72 videos were generated and before TEST
scoring. GitHub's frozen source Tree SHA was verified to match Tang's tree.
The Tang workspace was then reset to the exact GitHub commit, so manifest
`source_sha` equals `9690e6f...`.

The protocol and dataset rule were unchanged after the DEV and TEST results.
The TEST prediction JSON was written and SHA-256 sealed **before** opening
TEST private oracles. TEST labels were scored once; final audit passed.

Dataset: **25 independent counterfactual pairs / 50 newly generated videos**,
including **15 DEV pairs / 30 clips** and **10 TEST pairs / 20 clips**.
Each clip has 96 256×256 grayscale frames, with full occlusion in [18, 64)
and 32 visible post-occlusion frames. Five regimes:

- `persistent_cue`: target texture remains unique and visible after occlusion;
- `pan_gain_persistent`: same, with synthetic planar background translation and changing brightness;
- `identical_twins`: target and distractor have identical appearance;
- `erased_cue`: original identity cue disappears for both objects;
- `cloned_cue`: the distractor acquires the target's exact appearance.

In the three ambiguous regimes, assignments A/B have **pixel-identical**
public videos but different private identity labels. Only the two persistent
cue regimes are observably distinguishable after the occlusion.

Parameters were fixed a priori: 17×17 query patch; normalized correlation
score ≥0.86; identity-margin veto rejects if the best independent runner-up
is within 0.07 (after ±17-pixel suppression). No DEV/TEST parameter fit.

## 3. Full DEV and single-scored TEST results

### TEST: 10 independent pairs, 20 evaluated counterfactual worlds

There are **640 post-occlusion visible frames** in total and **920 hidden
frames** (20 clips × 46), with 256 observable-regime post frames and 384
ambiguous-regime post frames.

| Frozen method | Post-occlusion correct / 640 | Wrong accepted / 640 | Accepted / 640 | Hidden false accepted / 920 |
|---|---:|---:|---:|---:|
| `appearance_fb_055` | 0 (0%) | 0 | 0 | 0 |
| `template_recover4` | 0 (0%) | 0 | 0 | 0 |
| `best_template_score` | 384 (60%) | **128 (20%)** | 512 (80%) | 0 |
| **`identity_margin_veto`** | **256 (40%)** | **0** | 256 (40%) | 0 |

The new identity-margin matcher recaptured **256/256 (100%)** of
post-occlusion ground-truth points in the *observable* cue regimes, including
the programmed brightness and planar-shift regime. It abstained on
**384/384** ambiguous post frames and on every hidden frame.

However, 40% total post-occlusion correct coverage is **not** an overall
accuracy victory: 60% of the evaluation's post frames were deliberately
unidentifiable. The less conservative best-match method gained correct
coverage but accepted 128 incorrect identity-frame predictions.

### TEST by regime (each 2 independent pairs, 4 counterfactual clips, 128 post frames)

| Regime | Best-score correct / wrong | Margin-veto correct / wrong |
|---|---:|---:|
| persistent cue | 128 / 0 | **128 / 0** |
| planar pan + illumination, persistent cue | 128 / 0 | **128 / 0** |
| identical twins | 64 / 64 | **0 / 0**, abstained |
| erased cue | 0 / 0, abstained | **0 / 0**, abstained |
| cloned target cue | 64 / 64 | **0 / 0**, abstained |

For identical twins and cloned cues, the best-matching appearance can look
perfect while referring to the *wrong physical identity*. Identical public
videos under two latent worlds enforce the information bound:
`2 × post_correct ≤ post_accepted` when both worlds are scored together.

### Historical DEV results from this new frozen generator

DEV spans 15 pairs / 30 clips and 960 post frames. The predeclared margin
matcher correctly recovered **384/960** post frames with **0 wrong accepted**;
best-template recovered **576/960** with **192 wrong accepted**. Both legacy
methods remained fully abstaining after occlusion.

Pre-occlusion correct tracking in TEST was **272/340** for the margin method
versus **340/340** for both legacy trackers. Its losses occur in the regime
with truly identical twins, where there is no independent appearance cue;
unqualified coverage increases could therefore conceal identity risk.

## 4. Actual CPU costs and exclusions

TEST wall times per **96-frame clip**:

| Workload | CPU median | p95 |
|---|---:|---:|
| Existing frozen tracker bundle + affine transform (combined) | 688.656 ms | 698.058 ms |
| `best_template_score` alone | 52.652 ms | 53.215 ms |
| `identity_margin_veto` alone | **53.273 ms** | **54.198 ms** |

These are measured in the experiment's Python/OpenCV CPU process. They
exclude video file decode, model/VLM inference, service overhead, network,
and actual production end-to-end scheduling. The legacy combined time is a
different workload and must **not** be interpreted as a head-to-head
per-model speed comparison. No GPU jobs were started.

## 5. Software verification, immutable artifacts and provenance

- Targeted new tests: **36/36 passed** (first run exposed a synthetic
  illumination test-expectation mistake, fixed **before** source freeze).
- Full project regression: **1307 passed, 1 skipped, 3 existing Pillow warnings**.
- New code/protocol/tests: **835 lines across 3 files**, then pinned on GitHub.
- Frozen manifest SHA-256: `108d75c88e7b692133e6861de957d501107c86c5649575757889c75905aa8db0`.
- DEV predictions: `8365f4a15c4c47510df190d827bcf28e6c16db9ec4fe4d6f3e253fcc031939a9`.
- DEV score: `a1b2af65a0cd9b46410bf837bcbf2fe573c0f39d33a1d3bd9d39b20d91b568cf`.
- TEST predictions (sealed before TEST labels): `2457c176da370e95767aaf61cf73677187e2c43fbe3e04db9279b482d17a87bd`.
- TEST result: `3e82b278754232a3c566eb82dd981361d9bf0a850551b9749bf80afb3c484614`.
- Final pin audit: `BATCH72_FINAL_AUDIT_PASS 9690e6fed262d8934ac6bbffa8e66614ae5119ac`.
- Files remain on Tang under `data/batch72_cue_observability_v1`
  and `repo-batch72-dev/outputs/batch72_cue_observability_v1`. No
  videos, private labels, weights or secrets were committed.

## 6. Negative result and next research decision

The new method produces **conditional** post-occlusion recovery in trivial,
stable visual-identity-cue synthetic cases, and abstains in provably
ambiguous cloned/erased/twin cases. The two legacy trackers had no recovery
in any condition. This validates the **information-availability distinction**
but does not establish natural-video robustness, identity persistence under
deformation, robust occlusion detection, or camera ego-motion reasoning.

This study's 46-frame opaque panel did **not** replicate the sustained
117-frame shared false-accept episode observed in Batch 56 natural DEV.
Only **two independent TEST pairs per regime** were evaluated; even all
correct in eight observable-world clips is not enough for a generalization
or deployment claim. “Any first correct frame” is not yet sustained
reacquisition and may overstate practical identity preservation.

**Deployment verdict: NOT APPROVED.** Do not replace existing production
tracking, change threshold on the sealed TEST, or touch Batch 59's currently
running, source-pinned real natural-video experiment.

The next decisive step is Batch 59's independent natural-video OOD result,
with scene/episode-based occlusion risk, correct target coverage,
post-occlusion reacquisition and VLM baseline; then use *new DEV only* to
study more realistic appearance change, identity-preserving cues, geometric
motion compensation and absence-of-evidence abstention.
