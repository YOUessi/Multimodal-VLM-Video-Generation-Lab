# Batch 70 — Frozen prospective shared-identity stress protocol (2026-11-03)

## Aim and relationship to previous evidence

Batch 56 natural DEV identified a 117-frame shared-tracker false-accept episode. Batch 69's newly generated 40-frame occlusion scenes did **not** reproduce it. This prospective follow-up asks whether a **visible impostor with exactly the same local sprite texture**, located on a layer that entirely occludes the true target, induces long identity-confusion failures. There is **no algorithm tuning** in this study.

This is a procedural **2D** pixel-plane experiment, not independent natural-video evidence, not real 3D camera pose, and not a video-generation quality benchmark. Batch 59 remains the definitive new independent natural-video protocol, and its immutable cron, runner, GPU lock, source SHA, datasets and logs must remain untouched.

## Prespecified setup

- Generator: `experiments/batch70_shared_identity_stress.py`; 256×256 grayscale, **96 frames per scene**, 1 point queried only on frame 0.
- Four prespecified conditions: (1) identical-texture impostor (doppelganger), (2) same with known 2D background translation, (3) same under global brightness oscillation, (4) **distinct, unmatched occluder** as negative control.
- **Occlusion frames 18 through 80, inclusive (63 frames)**; the occluding layer overwrites the target texture before the impostor is added. In matched regimes, the occluder surface contains an exactly identical 21×21 sprite at a fixed 13–15 px horizontal, 4 px vertical offset from the true point. This is a deliberately *constructed* impostor and must never be presented as a naturally occurring event.
- Independent PRNG scene seeds: DEV base 700111 (four scenes per condition, 16 videos), TEST base 700991 (three scenes per condition, 12 videos); both plus group index ×101 and per-video index ×7. The TEST videos have never been generated, scored or used for model selection at the time of this frozen protocol.
- Five fully frozen, already implemented methods using existing `configs/kinetics_batch56.json`: `appearance_fb_055`, `klt_fb`, `template_recover4`, `affine_stabilized_klt`, and `template_consensus_4px` (only veto; distance ≤4 px). No new tracker and no changed threshold.
- Measure visible accepted/visible, full-visible PCK@8, occluded accepted/occluded, longest contiguous false-accept run, per-condition breakdown, 2D background-relative error on correct visible matches, and actual CPU process wall-time for all methods on each scene. Query time is excluded. Numbers will be reported with numerators/denominators, not just percentages.
- Known background translation and true coordinates are used **only for generation and post-prediction evaluation**. No 3D ego-motion or physical world-motion claim.

## One-way immutable sequence

1. Commit source, tests, and this preregistration document to GitHub **main**, verify exact immutable SHA.
2. Create 28 local frames/label files with exclusive-create writes and SHA256 pins. Only public pixel data and frame-0 query go into `manifest.json`. The oracle is isolated under `data/batch70_shared_identity_v1/private`.
3. Run and pin label-blind DEV predictions, then DEV score.
4. Run and pin label-blind TEST predictions; **only after predictions are pinned**, score TEST exactly once.
5. Verify all public video hashes, source SHA, private oracle checksums, and DEV/TEST result linkage. Do not overwrite a failed stage. Preserve partial and failure logs.

The code blocks TEST prediction unless a sealed DEV score exists and fails on altered source SHA or dirty tracked worktree. The outputs are write-once `xb` files. Public media/private labels/experiment outputs are **not committed**, never deleted, and never copied outside Tang. GPU is unused, and the existing Batch 59 runner is neither changed nor interrupted.

## Falsification and decision criteria fixed before opening TEST

- **Failure to replicate** shared-identity false acceptance is an informative negative result: report every method's longest false-accept run and hidden-false-accept support. Do not retune this generator from TEST.
- **Identity/visibility failure** if any method produces a false-accept run ≥10 consecutive hidden frames. An observed 63-frame episode indicates near-complete failure across the occlusion window.
- Compare paired DEV and TEST per-condition metrics, including the distinct-occluder control. Large differences signal procedural sensitivity, not natural generalization.
- **No production deployment** is justified by this synthetic experiment, even when a method passes all synthetic conditions. For deployment, a future independent natural-video one-time TEST must satisfy safety, visible coverage, PCK@8, worst-scene, event-length, CPU and VLM admission criteria **without tuning on TEST**.

## Operational safety

Source GitHub main is the sole authoritative tracked-code state. Tang project root is a parent folder containing separate Git clones. Neither batch59 frozen checkout (`8e079c70...`) nor `MVVG_PROJECT4_10MIN` may be altered by Batch70. Before starting, verify Tang online, GPU compute-process list, live runner/processes and disk headroom. Preserve archives, model weights and secrets locally. No concurrent GPU experiments. An unchanged set of five CPUs-only trackers is run after first freezing code and the public split, so even a negative TEST result closes this batch without algorithm changes.
