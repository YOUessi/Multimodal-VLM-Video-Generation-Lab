# Batch43 — TAPVid-MV synchronized multi-view motion, occlusion and 3-D consistency protocol

**Protocol freeze date:** 2026-11-02  
**Dataset:** TAPVid-MV public Rerun viewer recordings from EgoExo4D and PACE  
**Code/config:** GitHub main commit to be pinned before the frozen execution begins  
**Status:** Prospective protocol; no TEST labels have been inspected for method selection.

## Research question

Can RGB-only point trackers remain reliable when the same physical 3-D point is observed by three synchronized cameras with different camera motion, image scale, lighting, texture and occlusion? When they cannot, can calibrated multi-view geometry diagnose the difference between image-plane displacement, camera-induced displacement and world-relative point motion?

This batch extends the prior single-view DAVIS/Kinetics and synthetic parallax studies with released multi-view camera calibration and 3-D point identities. It does not assume that a 2-D affine transform is a physical camera-motion estimate.

## Source and immutable selection

- Public source index: https://storage.googleapis.com/dm-tapnet/tapvidmv/rrd/index.html
- Exact source-index SHA-256: `eb9ff78058eca50479af98eca5bba21716dcec183ce83d8e70349d9cea58f2f5`
- Rerun reader version: 0.35.0.
- Twelve predeclared sequences: six DEV and six TEST; each source subset contributes three sequence-disjoint collection groups to each split.
- DEV: EgoExo4D `cmu_soccer03_1`, `cmu_soccer07_1`, `iiith_soccer_002_2`; PACE `scene00_video05`, `scene02_video04`, `scene05_video09`.
- TEST: EgoExo4D `cmu_soccer06_2`, `cmu_soccer08_1`, `iiith_soccer_004_5`; PACE `scene01_video03`, `scene04_video02`, `scene08_video02`.
- The collection-group key, rather than an individual view filename, is the split unit. No collection group crosses DEV/TEST.
- Each scene selects three deterministic, hash-ranked frame-0 3-D track IDs that pass the exact depth-cloud visibility proxy in all three cameras. Each shared point is queried independently in camera 0, 1 and 2: nine 2-D queries per scene, 108 total.
- The target time is the available exact depth-cloud snapshot nearest 67% of the sequence. It is frozen in the public query file; no target-frame choice is made from TEST labels.

The RRD recordings are the public viewer subset, not the full released TAPVid-MV track arrays. The Rerun recording metadata may declare 500 tracks while the viewer point stream displays fewer IDs. Results must report the rendered ID count and must not claim full-dataset coverage.

## Camera geometry and separation of motion sources

The Rerun transform convention used by the extractor is:

[
X_c = R_t (X_w - C_t)
]

where (X_w) is a world-frame 3-D point, (C_t) is the camera translation/center, (R_t) is the serialized world-to-camera rotation, and (X_c) is the camera-frame point. Pixel projection is:

[
[u',v',w'] = [X_c/z_c,;Y_c/z_c,;1]K,qquad
(u,v)=(u'/w',v'/w')
]

The implementation validates this convention by projecting the same 3-D point into three cameras and triangulating the three projected pixels back to the known point. The synthetic geometry regression requires sub-micrometer reconstruction error for a well-conditioned camera baseline and abstention for a degenerate baseline.

For diagnostics only, each oracle track stores three distinct projections:

1. **Observed screen projection:** the moving world point through the moving camera pose.
2. **Camera-only counterfactual:** the initial world point through the moving camera pose.
3. **World-only counterfactual:** the moving world point through the initial camera pose.

These are counterfactual projections; perspective makes their pixel displacements non-additive in general. Their residual/cosine statistics are descriptive diagnostics, not a claim of exact causal decomposition.

The image-only background baseline estimates a frame-0-to-frame-t partial affine transform with ORB matches and RANSAC. It is a screen-space dominant-background proxy. It is not calibrated camera pose, a full homography, or a 3-D motion estimator.

## Visibility and occlusion labels

The RRD point-cloud snapshots are projected into each camera to form a nearest-depth buffer.

- **VISIBLE:** the track point projects in-frame and its depth agrees with the nearest local depth sample within the frozen 0.05 m tolerance.
- **OCCLUDED:** a nearer local depth surface is more than 0.05 m in front of the point.
- **OUT:** the point projection is outside the camera image or behind the camera.
- **UNKNOWN:** no sufficiently close depth sample can establish visibility, or the depth evidence is ambiguous.
- **MISSING:** the 3-D track ID is absent from that frame's rendered point stream.

Visibility is evaluated only on exact depth-cloud snapshots. It is not interpolated to intermediate frames. This is a conservative depth-cloud-derived proxy, not a separate official per-point binary visibility annotation.

## Frozen RGB-only baselines

All methods receive only RGB frames and the frame-0 query pixel:

1. Stationary initial pixel.
2. Forward KLT.
3. Forward–Backward KLT.
4. Local template recovery with a bounded search radius.
5. ORB/RANSAC background-affine stabilized KLT.
6. Appearance-gated Forward–Backward KLT at NCC thresholds 0.55, 0.70 and 0.85.

The predictor stage reads only the public query file and RGB frame arrays. It must not open private oracle arrays. Prediction JSON and SHA-256 sidecars are pinned before scoring.

## DEV-only selection and TEST policy

The DEV selector considers only the predeclared candidate methods. A candidate is eligible only if it reaches at least 40% depth-visible coverage and both occluded and out-of-frame false-accept rates are at most 25%. Among eligible methods, maximize DEV scene-macro Jaccard mean, then visible coverage. If no candidate qualifies, use the frozen KLT-FB fallback and explicitly record the blocked selector status.

TEST CPU predictions are generated only after the DEV selection artifact and pin exist. All candidate methods are still reported, but TEST metrics cannot alter thresholds, method choice, prompts or parser behavior.

The SmolVLM2-500M baseline receives only a marked frame-0 query image and the exact public target-frame image from the same camera. It must return one of VISIBLE / NOT_VISIBLE / UNCERTAIN. DEV gate requirements are frozen in the config:

- balanced accuracy at least 0.55;
- visible and non-visible support at least 8 each;
- coverage at least 0.50;
- invalid-format rate at most 0.20;
- non-visible false-accept rate at most 0.25.

TEST VLM inference and scoring are skipped unless every DEV gate passes. The VLM predicts visibility only; it does not output a point coordinate or 3-D position. CPU full-sequence tracking and two-image VLM classification have different temporal evidence budgets and must be compared only at the exact common target frames.

## Metrics

- PCK@8 on all in-frame projected track locations, clearly labeled as including unknown-visibility points.
- PCK@8 on exact depth-visible snapshots.
- Visible coverage and accepted precision@8.
- False accepts on occluded and out-of-frame points; unknown and missing remain separate.
- TAP-Vid Jaccard at radii 1, 2, 4, 8 and 16 pixels, excluding unknown/missing labels.
- Per-scene and per-source-subset metrics; scene-cluster bootstrap intervals.
- CPU P50/P95 milliseconds per target frame and separate ORB registration cost.
- Three-view triangulation coverage, 3-D world-coordinate error, and oracle-pixel triangulation error as a calibration sanity check.
- Camera translation/rotation, screen-vs-world projection diagnostics, image-affine residuals, and per-camera image luma/contrast/texture/ORB-keypoint statistics.

The full scene/video denominator and abstentions must remain visible. A method with low false-accept rate but near-zero coverage is not a successful tracker.

## Execution and data safety

- Tang is the only execution host for this frozen GPU/CPU evaluation. The runner uses the shared GPU lock, checks for other compute processes and requires at least 10 GiB free before VLM inference.
- One attempt per frozen stage. Any failure preserves the log and partial artifacts; a repair requires a new GitHub source SHA and a new tested directory.
- Do not delete old videos, RRDs, frame arrays, predictions, model weights or failed logs.
- Do not commit source RRD files, video/image arrays, private oracle NPZs, model weights, raw predictions, generated media, secrets or machine-specific private paths.
- Individual EgoExo4D/PACE source-data terms apply. This experiment is local research evaluation only; no source recording or extracted frame is redistributed.

## Predeclared limitations

This is a 12-sequence exploratory subset with three queried point identities per scene. The point stream is subsampled by the viewer recording. Visibility is inferred from sparse depth snapshots. The camera calibration enables a 3-D diagnostic, but it does not make the RGB trackers 3-D trackers. The selected source sequences are not a broad benchmark-wide estimate. TEST results are descriptive and cannot be used for later tuning; any follow-up needs a new independent source split.
