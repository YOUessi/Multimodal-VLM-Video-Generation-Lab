# Batch42 pre-registered protocol — TAP-Vid-Kinetics natural video

Date: 2026-11-02
Source branch at protocol start: `main` snapshot `47a511ea707f932f7f0f407dc5cc29fecb0fbd2d`
Dataset: TAP-Vid-Kinetics, 12 Kinetics-700 validation clips from 12 distinct action classes.

## Research question

Do simple image-plane point trackers, appearance gates, and dominant-background affine compensation retain useful point identity and visibility decisions on natural YouTube video, beyond the earlier synthetic robotics and DAVIS subsets? The experiment separates image-plane motion from a 2-D background-relative proxy. It does not claim calibrated camera pose or 3-D world motion.

## Data and freeze

- Official TAP-Vid Kinetics CSV annotations; source CSV SHA-256: `d02e8fafa631ec1929c30172a9b0fe0a17dff9306f5ebd35120e9ed4cdf100e6`.
- Four DEV clips and eight TEST clips; one clip per action class; video IDs are unique across splits.
- Four point queries per clip, selected by a fixed hash over video ID and track index from tracks with at least one visible sample.
- Query time is the first visible point for that track, matching the official TAP-Vid first-visible query convention.
- Raw videos are sampled at 25 fps, truncated to 250 frames, and resized to 256×256. Clips shorter than 10 seconds truncate the corresponding annotations to the decoded frame count.
- The official generator applies a half-pixel coordinate correction before the 256×256 reader scaling; the local parser reproduces that transform.
- The full future point tracks and visibility flags are stored in a private oracle file. Prediction scripts only read the public query file and decoded frames. Scoring opens the oracle only after the prediction JSON and SHA-256 sidecar are sealed.

The TAP-Vid annotations are CC BY 4.0. The source Kinetics/YouTube clips retain their individual creator/platform terms; all videos and derived frames remain local and are not published.

## Pre-declared CPU methods

1. Stationary initial point.
2. Forward KLT.
3. Forward–backward KLT with a 1.5-pixel cycle threshold.
4. Static-template local recovery.
5. ORB + RANSAC partial-affine background compensation followed by KLT in the stabilized image plane.
6. Forward–backward KLT with a fixed initial-patch NCC gate at 0.55, 0.70, and 0.85.

The appearance thresholds are a pre-declared DEV candidate set, not TEST-tuned values. ORB affine estimates use video pixels only. They are diagnostic 2-D transforms, not camera calibration.

## DEV-only selection

Candidate methods: KLT-FB, template recovery, affine-stabilized KLT, and the three appearance-gated KLT variants.

Eligibility constraints:
- scene-macro visible-point coverage at least 0.40;
- scene-macro invisible-point false-accept rate at most 0.25.

Among eligible methods, select maximum scene-macro TAP-Vid Jaccard averaged over 1/2/4/8/16-pixel thresholds, breaking ties by visible coverage. If none qualifies, use the pre-declared KLT-FB fallback. The selected method and DEV-results hash are sealed before any TEST prediction.

## Metrics and diagnostics

- Visible-point PCK@8, unconditional (abstentions count as incorrect).
- Visible coverage and accepted precision.
- Invisible-point false-accept rate, visible false-abstention rate, and visibility accuracy.
- TAP-Vid Jaccard at 1, 2, 4, 8, and 16 pixels.
- CPU milliseconds per target frame, P50/P95; per-video shared affine-registration time is reported separately.
- Per-video mean luminance, contrast, Laplacian variance, ORB feature density, affine-pair acceptance and reprojection residual.
- Image-plane ground-truth displacement versus ground-truth displacement after the estimated dominant-background affine transform. This latter value is explicitly a 2-D background-relative proxy; it cannot resolve parallax, depth, rolling shutter, or true 3-D camera/world motion.
- Scene-cluster bootstrap intervals are descriptive only because there are eight TEST scenes.

## VLM baseline and admission gate

SmolVLM2-500M-Video-Instruct receives a marked first-visible query frame and one later RGB frame at fixed 50% and 90% fractions of the remaining clip. It must emit exactly one of `VISIBLE`, `NOT_VISIBLE`, or `UNCERTAIN`.

TEST VLM inference is allowed only if DEV satisfies all of:
- at least eight visible and eight not-visible target observations;
- invalid-format rate ≤ 0.20;
- non-abstaining coverage ≥ 0.50;
- balanced accuracy ≥ 0.55;
- not-visible false-accept rate ≤ 0.25.

If any DEV condition fails, TEST VLM inference is skipped and TEST VLM labels are not opened.

## Leakage and publication rules

- No thresholds, method choice, prompt, parser, or model settings may be changed using TEST labels or TEST outcomes.
- CPU and VLM TEST predictions are pinned before their one-time score stage.
- Failed stage logs and partial outputs are retained; a repair requires a new source SHA and new frozen batch directory.
- No raw videos, extracted frames, private oracle, model weights, generated media, or secrets are committed.
