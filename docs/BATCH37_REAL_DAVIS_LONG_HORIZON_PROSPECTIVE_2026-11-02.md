# Batch 37 — new independent real-camera long-horizon tracking and abstention (PRE-TEST protocol, 2026-11-02)

## Motivation, boundary and one-way experiment

Batch 36 measured real DAVIS first-frame-mask centroid tracking over 8 sampled frames (4 DEV / 8 TEST scenes). Its TEST scenes and answers are **exposed** and must not be used for parameter selection. Batch 37 draws **12 different natural-video sequences** from the official DAVIS 2017 TrainVal 480p archive by salted SHA256 scene-name ordering, explicitly excluding **all 12 Batch 36 sequences by immutable name**, not just its TEST subset. New scenes are independent at the *sequence level within the same DAVIS archive*; no new dataset or model-unseen domain claim is made. The source archive may be part of VLM pretraining corpora, and clips must remain local absent per-clip license review. Attribution: Pont-Tuset et al., The 2017 DAVIS Challenge on Video Object Segmentation; https://davischallenge.org/davis2017/code.html.

Exactly four sorted sequences are DEV and the next eight are TEST; no label-driven scene inclusion/exclusion, no post-TEST resampling. Each sequence supplies **16 original photographic frames** at a source stride of two (indices 0,2,...30; 192 sampled photographic frames and 192 instance-label assets). A sequence must have at least 32 available corresponding frame/index filenames. The first instance mask identifies one target by its largest nonzero instance label; future masks are private score-only. All original JPEG/PNG bytes are fetched with verified 206 HTTP Range, ZIP CRC, local SHA256 and immutable manifest. Never publish original video, annotated images, weights, or credentials in Git.

## Predeclared models and metrics

Four CPU-only methods are frozen in source **before accessing TEST labels**: (1) static first image-plane centroid; (2) unchanged Batch 36 KLT forward-backward tracker; (3) FB+LK temporal consensus with at least 7 features, FB <=1.5px, max per-step motion 45px, residual <=4px and >=55% consensus; (4) a stricter conservative alternative with >=9 features, FB <=1.25px, per-step motion <=30px, residual <=2.5px, >=70% consensus. No geometric or photometric threshold will be adjusted using this batch's TEST answers. Failure to maintain enough features/consensus causes permanent abstention, **not** unverified re-identification.

Image-plane GT is the independently annotated future-instance-mask **centroid**, *not* a material-point trajectory. Primary metric PCK@8px over **every visible TEST position**, with abstentions counted as failures; also PCK@4/16, accepted coverage, correct/incorrect accepted, absent-target false accepts, per-scene CPU P50/P95 excluding JPEG decoding and camera registration, and scene-cluster bootstrap confidence intervals vs unchanged FB-KLT. To study occlusion-adjacent cases without inventing visibility labels, report (a) **true mask-absent** when no target pixels exist, and separately (b) **mask-area-shrink proxy** when target area is <40% its first-frame area. Shrink can be scale, cropping, deformation or occlusion: do not call it ground-truth occlusion. In the absence of actual disappearing objects, explicitly mark absence metrics as untested.

For each scene, first-to-last 2-D dominant-background affine registration is a diagnostic only. A KLT image-plane displacement and the displacement relative to 2-D background transform are not interchangeable; neither alone proves physical world motion, full camera pose nor depth. No VLM baseline is claimed for mask-conditioned point tracking: existing SmolVLM2 direction-choice answers are a **different label space**. A future benchmark may construct a matched instance-aware VLM/point-track interface, but must be frozen independently.

## Integrity, lifecycle and negative-result policy

- Source code + plan + config + regression tests must be committed to **GitHub main**, with matching exact SHA imported to isolated local tested checkout **before** first-run DEV/TEST scoring.
- Verify 12 new scene IDs disjoint from old 12, all original source digests, 165+ unit regressions and private-mask blindness before model predictions.
- Predict DEV and seal SHA256 first, score DEV for diagnostics; **do not use TEST to tune, rank or choose a policy**. All four methods remain evaluated on TEST regardless of DEV performance.
- Predict TEST without future masks or gold, pin the exact prediction bytes and source code SHA; only then score TEST once, emit full failures/coverage/CPU and keep all private logs.
- Any failed stage blocks the batch, preserves logs, and requires a fresh modified source commit and new frozen batch; never silently modify a completed plan or re-run old scored TEST.
- Tang scheduler executes at most two frozen stages each 10-minute tick and must stop on completion marker. GitHub public repo includes only auditable source, config and aggregated report, **never** source media, weights, private trajectories or secrets.

This predeclared document is a protocol, not an effectiveness claim.
