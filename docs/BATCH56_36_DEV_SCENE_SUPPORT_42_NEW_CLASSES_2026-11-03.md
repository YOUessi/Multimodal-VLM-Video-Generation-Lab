# Batch 56 — prospective 36-DEV-scene natural-video risk replication (frozen proposal)

## Why Batch 56 exists

Batch 55 completed a **new, action-class-disjoint sealed TEST** with unchanged
nine CPU tracker methods and the prior DEV-selection rule. Its selected
`klt_fb` was 1.17% occlusion false acceptance by DEV scene-macro, but 15.64%
by independent TEST scene-macro, exceeding the predeclared 10% safety budget.
One TEST scene had 109 false accepts among 125 occluded target frames (87.2%).
The TEST micro average was only 6.91% (116/1678), demonstrating why pooled
frame counts hide scene-level catastrophic errors. No hyperparameters, methods,
candidate thresholds, or samples are changed using the already-opened TEST.

Batch 54's **earlier DEV-only** research had already identified the
insufficiency of five occlusion-supported scenes: even zero observed violating
scenes produce a nominal one-sided 95% upper bound of 45.1% on future
scene-budget violation probability, or 64.6% after conservative correction
for nine predeclared candidate methods.

Therefore the next bottleneck is **independent occlusion-supported natural
scene sample size and risk estimation**, rather than another tuned affine/KLT
threshold or synthetic-motion experiment.

## New frozen design

- **42 additional natural Kinetics videos, 42 previously unused action classes:**
  36 DEV and six sequestered TEST, 4 official TAP-Vid point queries per clip.
  Both action classes and video identifiers are disjoint from all 72 used
  earlier in Batch42,45E,46E,48B,53B and Batch55. Exclusion uses only public
  metadata manifests and explicitly rejects missing/invalid historical input.
- New SHA-256 selector salt `MVVG-BATCH56-36DEV-6TEST-SCENE-SUPPORT-20261103`
  and a separate data root. The first 36 deterministic ranks are DEV,
  and the remaining six are TEST; selection must be published/frozen before
  TEST oracle access. No video replacement based on observed performance.
- Unchanged 11 total CPU comparators, nine predeclared selection candidates,
  same input resolution, coordinate geometry, frame rate, model weights,
  VLM prompt, 10% occlusion budget, >=30% visible-coverage rule, and
  scene-macro Jaccard objective among eligible candidates.
- SmolVLM2-2.2B evaluates DEV only until the unchanged balanced-accuracy and
  other DEV admission gates pass; otherwise TEST VLM inference is skipped.
- An additional **DEV-only independent scene audit** runs strictly after the
  DEV selection is frozen, but **before any TEST predictions/scoring**.
  It computes a one-sided Clopper–Pearson 95% upper bound on the probability
  that a *future scene's observed occluded false-accept rate exceeds 10%*.
  It also reports a nine-method Bonferroni familywise bound and the number
  of scenes with zero occlusion support, which are **never counted as safe**.
  The audit is descriptive only and **must not modify the method selection**.
- TEST point predictions are saved and hashed before TEST scoring; no
  post-hoc TEST method selection or re-use of opened TEST for threshold tuning.
- 23 write-once frozen stages; unique `MVVG_PROJECT4_10MIN` Tang cron,
  global GPU exclusion lock, fail-closed source SHA and disk checks.
  Downloads or stage failures preserve all logs, media and previous batches.
  No private labels, raw media, weights or secrets are committed.

## Statistical interpretation

With zero violating scenes, at least 29 supported independent scenes are
needed for a *nominal* 95% upper bound below 10% on the **scene violation
probability**. The 36 DEV clips are deliberately larger than the 29-scene
minimum because some clips may contain no occluded frames. However, it is
possible that fewer than 29 will be supported; report the actual count,
do not fabricate a bound. With nine-method familywise correction, about
50 zero-violation independent supported scenes would be needed.
A 36-DEV batch **cannot achieve that stronger certificate by sample size
alone**. Also, Kinetics action classes are not proven IID draws from a
deployment population, so **none of these numbers certifies real-world
safety**.

The natural-video CPU timing excludes full decoding and system I/O;
affine compensation is a 2-D image-plane motion proxy, **not calibrated
background identity or 3-D physical camera motion**.

## Code, tests and reproducibility

- `experiments/kinetics_batch56_data.py`: independently isolated data
  selector with 72-ID/class historical rejection and 36/6 split.
- `experiments/kinetics_batch56_eval.py`,
  `experiments/kinetics_batch56_vlm.py`: isolated frozen evaluation wrappers.
- `experiments/batch56_dev_scene_risk.py`: DEV-only scene risk proof and
  pinned immutable aggregate report, tested against zero-support scenes,
  rate inconsistencies, malformed counts, wrong split, and selection hash
  mismatches. Never reads TEST oracles/results.
- `experiments/kinetics_batch56_report.py`: final source, config,
  candidate selection, scene-risk report and prediction-pin re-audit.
- `automation/plans/batch56_kinetics_72_class_disjoint_2026-11-03.json`:
  23-stage ordered protocol, unambiguous TEST evaluation firewall.
- `tests/test_kinetics_batch56.py`,
  `tests/test_batch56_dev_scene_risk.py`: new independent regressions.
- `configs/kinetics_batch56.json`: only population count, source paths,
  deterministic salt, bootstrap seed and provenance changed; no tuned
  tracker/VLM parameter has been modified.

This is a **pre-registered expansion plan**, not yet an assertion of new
TEST performance. Runtime results and all failure logs remain on Tang.
