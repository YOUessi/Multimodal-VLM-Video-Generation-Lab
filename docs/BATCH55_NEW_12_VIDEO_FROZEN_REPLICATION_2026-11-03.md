# Batch 55 — prospective action-class-disjoint natural-video replication (frozen)

## Why this experiment is necessary

Batch48B froze its DEV choice (`klt_fb`) with 2.00% occlusion false acceptance
on six DEV clips, yet the sealed TEST measured 22.24% and failed its unchanged
10% safety constraint. Batch53B independently selected `appearance_fb_055`
on its DEV and observed 0.0% TEST occlusion false acceptance, but with only six
new TEST clips; this is not proof of safe generalization. Batch54 established
that each of the two DEV splits has just five occlusion-supported scene
clusters, yielding a 45.1% nominal and 64.6% nine-method familywise
Clopper-Pearson upper bound on the chance a future scene violates the
10% occlusion false-accept budget. These are **not** risk certificates.

## Prospective Batch55 contract

A new deterministic SHA-256 salt selects **12 wholly new TAP-Vid-Kinetics
natural videos**, with **six DEV and six sealed TEST**, four original official
point queries/video. Both the video IDs and action classes must be disjoint
from 60 history classes/IDs across Batch42,45E,46E,48B,53B. Only public
historic manifest identifiers are read for exclusion, not any prior TEST label.

Nine unchanged CPU tracking candidates, the same >=30% visible-coverage and
<=10% invisible false-accept DEV selection criterion, identical VLM
three-image configuration and 55% DEV balanced-accuracy admission gate
are applied without any TEST-driven threshold or model changes. TEST
prediction SHA must be frozen *before* TEST scoring. No dataset/track
replacement after seeing TEST performance is permitted.

The Batch53B geometry-integrity fix excludes complete malformed annotated
tracks under the **same** rule on DEV and TEST, without clipping labels.
The resulting candidate query population differs from pre-fix Batch53:
this limitation must be reported honestly.

This is a new independent replication, **not** a pooled confidence
certificate and not an attempt to calibrate physical camera motion.
A zero-occlusion scene is marked unevaluable for that safety measure.
Frame-level outcomes are correlated within scenes; scene-macro and
per-scene safety must be reported separately with denominators.
CPU tracker timings exclude video decoding; VLM uses only three images.
No video, weights, private labels or predictions may leave Tang.

## Freeze, capacity and failure behavior

- Source: GitHub `YOUessi/Multimodal-VLM-Video-Generation-Lab` main,
  one clean frozen SHA (recorded in `.automation/state.json`).
- Plan: `automation/plans/batch55_kinetics_60_class_disjoint_2026-11-03.json`.
- Runner: `automation/runner_batch55.py`; 22 write-once stages.
- Only one `MVVG_PROJECT4_10MIN` cron; GPU lock
  `/tmp/tang_shared_gpu_experiments.lock` for inference.
- Preflight rejects a download if fewer than 10 GiB remain, without
  deleting any existing media, models or partial downloads.
- Each stage fails closed on one unsuccessful attempt, preserving logs and
  partial data; repair requires a new SHA and separate tested checkout.

**Selection is fixed before new TEST results.** Even successful TEST point
estimates cannot certify scene transfer with only six fresh natural clips.
The next independent sampling expansion should target at least 29
occlusion-supported DEV scenes for a nominal zero-violation 95% bound
below 10%, or 50 such scenes for the frozen nine-method familywise bound,
under the strong and explicitly unverified i.i.d. scene assumption.
