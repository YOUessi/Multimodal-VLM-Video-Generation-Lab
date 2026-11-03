# Batch 79 — Frozen new-natural-video DEV reacquisition audit

Status: **PROTOCOL / CODE FREEZE BEFORE BATCH79 ORACLE SCORING**.

## Question

Batch 75 found near-zero post-occlusion PCK8 recovery on the already-studied Batch 56 natural DEV; Batch 76 improved DEV recovery with added false accepts. Does the **new independent Batch 59 natural DEV** demonstrate the same coverage-vs-occlusion failure for previously frozen CPU trackers?

This is an **exploratory DEV analysis**, not an independent held-out TEST result; this analysis itself must never select an algorithm on TEST and cannot authorize deployment.

## Frozen inputs

- Only Batch 77's already-pinned `dev_predictions.json`, `dev_results.json`, `dev_selection.json` from the new Batch 59 data.
- Only `kinetics_00` through `kinetics_35`, 4 deterministic public queries per DEV scene (144 total).
- Read the private annotation oracle with the existing streaming `load_dev_only_oracle` routine, which ends at `kinetics_35` **before parsing the six TEST objects**.
- Validate the source commit, pin SHA-256 values, manifest and public-query hashes, 36-scene membership, per-query length and pre-query null convention, and independent DEV scorer's total occlusion count and false accepts per scene and tracker.
- Do not inspect or open `test_predictions`, `test_results`, or TEST annotation objects, even if scheduled Batch77 scores them separately.

## Fixed methods

`klt_fb`, `appearance_fb_055`, `appearance_fb_070`, `appearance_fb_085`,
`gradient_fb_050`, `gradient_fb_065`, `template_recover4`,
`affine_stabilized_klt`. Do **not** fit any of these methods on this DEV.

Read the already-frozen Batch77 DEV-selected method. It is reported, not changed.

## Exact definitions

For every query, ignore all frames through its query time and find contiguous **occluded periods**. For each such period, consider the immediately following contiguous **visible period** until another occlusion or video end. A period ending at the video end without any following visible frame is **right censored**, not a failed reacquisition.

- `reacquired_events`: evaluable periods with at least one accepted predicted point within **8 image pixels** of the DEV ground-truth position.
- `first_accept_wrong_events`: evaluable periods where the first accepted visible position is outside PCK8.
- `never_accepted_events`: no accepted positions in the subsequent visible period.
- `false_accepted_occluded_frames`: accepted positions during ground-truth occlusion.
- `post_visible_correct_pck8`, `post_visible_coverage`, and `worst_wrong_visible_run`: all computed as in the earlier Batch75 event analyzer.
- `scene_summary`: weight each event/frame by its own frozen count, not the average of potentially undefined per-scene fractions.
- Paired scene-level percentile intervals: **2000 seeded resamples**, seed 20261104, sampling the 36 scenes with replacement. Compare DEV selection against the *fixed* `appearance_fb_055` and `appearance_fb_085` controls; these intervals are exploratory, scene clustering is not necessarily independent.

A correct PCK8 location is **not proof of a recovered physical identity**. A wrong location is not proof of a physical identity switch. A scene-level bootstrap interval must not be interpreted as an independent TEST confidence interval.

## Acceptance gates

1. Unit tests cover invalid inputs, DEV/TEST path protection, immutable pins, reproducible bootstrap, 36-scene synthetic replay, query corruption, and comparison against frozen DEV scorer.
2. Full project pytest must pass (existing skip/known warnings may remain).
3. Publish and freeze algorithm, tests and protocol **before evaluating the new Batch 59 DEV oracle**; a later correction requires a new frozen revision and failure log.
4. Output is local, one-write JSON and `.sha256` file; no media/weights/secrets/annotations are committed.
5. Publish only a post-evaluation report once results are computed and verified. The frozen Batch77 runner/cron/selection and sealed TEST are untouched.
6. Do not deploy candidate or update frozen TEST procedure based on this retrospective diagnostic.

## Run invocation

```bash
python -m pytest -q tests/test_batch79_fresh_dev_reacquisition.py
python experiments/batch79_fresh_dev_reacquisition.py \
  --input-dir ../repo-batch77-tested/outputs/kinetics_batch59 \
  --data-dir ../data/tapvid_kinetics_batch59/batch59_frozen \
  --output outputs/research/batch79_fresh_dev_reacquisition.json
```

The output is deliberately not source-controlled; the final report may cite its SHA-256 and summarized non-sensitive DEV metrics only.
