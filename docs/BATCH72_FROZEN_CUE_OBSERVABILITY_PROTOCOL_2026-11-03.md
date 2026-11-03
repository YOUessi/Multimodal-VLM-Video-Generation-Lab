# Batch 72 — Frozen prospective cue observability & post-occlusion reacquisition protocol

Date: 2026-11-03. Source of truth: GitHub `YOUessi/Multimodal-VLM-Video-Generation-Lab`, `main`.

## Question and scope

Batch 71 showed an identifiability limit for latent identity swaps when two
physical worlds produce exactly the same visible pixels. This batch studies a
**controlled information intervention**: under which narrowly specified visual
evidence can a tracker reacquire the original identity after a full occlusion,
and when should it abstain instead? This is a small causal *synthetic-video*
study, **not** an independent natural-video transfer test or a production
safety claim. Batch 59's prospective natural-video frozen protocol remains
untouched and owns GPU resources.

## Preregistered material

Fixed dataset generator: `experiments/batch72_cue_observability.py`.

- 5 regimes × (3 DEV pairs + 2 TEST pairs) × 2 counterfactual assignments
  = **25 independent pairs, 50 videos** of 96 grayscale 256×256 frames.
- Identity A is initially the left object, with two post-occlusion latent
  assignments: original target left or original target right.
- Fully opaque occlusion occupies frames [18, 64), length 46.
- Regimes: `persistent_cue`, `pan_gain_persistent`,
  `identical_twins`, `erased_cue`, `cloned_cue`.
- In `persistent_cue` the unique target texture stays attached to the
  target, while the assignments produce **different observable videos**.
- In the three ambiguous regimes, two assignments produce **identical
  public pixels**, including their initial query. Only private latent identity
  differs; reidentification beyond random assignment chance is impossible
  without extra information.
- The remaining regime tests persistent unique appearance under brightness
  change and **synthetic 2D background translation**. It is not 3D camera
  ground truth.

Pair seeds are disjoint across split. The generation rules and fixed
thresholds are committed and frozen **before** opening new TEST labels.

## Frozen methods

1. `appearance_fb_055` (existing frozen tracker, no tuning);
2. `template_recover4` (existing tracker, no tuning);
3. `best_template_score` (query-frame 17×17 gray patch, global normalized
   cross-correlation; score ≥0.86; may accept duplicate identities);
4. `identity_margin_veto` (same top-score threshold; runner-up must be at
   least 0.07 lower after suppressing ±17-pixel neighborhood of first match).

Both new methods use only the query frame as identity memory, do not update
the template, and cannot inspect any oracle, hidden label, post-occlusion
ground truth, or the 2D camera translation supplied to the scorer.

**No changes to thresholds based on DEV or TEST outcomes.** The thresholds
above are prespecified hypotheses, not learned optima.

## Metrics and non-independent worlds

Report by regime and aggregated: post-occlusion correct coverage (PCK@8),
accepted-but-wrong prediction rate, accepted coverage, proportion of scenes
with any correct reacquisition, first-correct delay, pre-occlusion tracking,
hidden false-accept rate, longest hidden false-accept run, CPU wall time for
frozen legacy and new methods. Include the paired-world indistinguishability
bound `2 × post_correct ≤ post_accepted` in the three ambiguous regimes.
Count the **pair**, not its two counterfactual worlds, as an independent
experimental unit. At most two TEST pairs per regime: treat uncertainty as
large; no significance or deployment claims.

Correctness is identity-sensitive. A visually identical duplicate is not
the target merely because the patch matches. An abstention is neither a
correct reidentification nor an accepted identity error.

## Frozen stage order

```bash
python -m pytest tests/test_batch72_cue_observability.py -q
python -m pytest -q
git commit ...  # freeze on GitHub main before producing TEST data
python -m experiments.batch72_cue_observability prepare
python -m experiments.batch72_cue_observability predict-dev
python -m experiments.batch72_cue_observability score-dev
python -m experiments.batch72_cue_observability predict-test
python -m experiments.batch72_cue_observability score-test
python -m experiments.batch72_cue_observability audit
```

All media, private labels and predictions remain in Tang local ignored paths;
each manifest, prediction and result uses write-once files and SHA-256 tags.
Prediction must complete before private scoring. TEST scoring is performed
once. New source commit after TEST is **report only**; algorithms and rules
must remain identical.

## Decision gates and failure interpretation

Reject deployment if either false accept or identity error is non-negligible,
if legitimate reacquisition remains poor, if novel backgrounds or gain/shift
conditions break the template, or if new methods lack independent natural
video verification. Favor truthful abstention where identity is provably
indistinguishable. Do not equate easy synthetic success with real-world
visibility safety or scene-general 3D motion compensation.
