# Batch45 — fresh Kinetics action-class-disjoint holdout protocol

Date: 2026-11-02  
Source baseline: GitHub main `1c5ec1c12fcd577ebfa5cfb7a3e293b7d9a0a955`  
Research question: can a simple illumination-normalized local-gradient identity gate improve the coverage / false-accept trade-off of image-plane point tracking on fresh natural-camera videos, and can a stronger VLM make a calibrated visibility decision when given an explicit local reference crop?

## Why this batch follows Batch44D

Batch44D found that calibrated three-view consensus improved some geometric consistency but still exceeded the predeclared occlusion false-accept limit on TEST. Its 500M VLM predicted VISIBLE for every DEV query, so TEST VLM inference was correctly blocked. The Kinetics Batch42 study also found that a four-scene DEV split selected a template tracker that failed to generalize, and that appearance gates traded coverage for precision.

Batch45 therefore does not reuse Batch42 video IDs or action classes. It draws a new, deterministic, class-disjoint subset from the official TAP-Vid-Kinetics annotations and Kinetics-700-2020 validation video archives. This is a fresh within-source-family holdout, not a claim of cross-dataset-family generalization.

## Data freeze

- 12 new YouTube video IDs, one per Kinetics action class; 6 DEV / 6 TEST.
- Exclude all video IDs already present in existing project data manifests and all 12 Batch42 action classes.
- Deterministic SHA-256 selection over eligible annotated clips; at least four queryable point tracks per selected video.
- Official TAP-Vid first-visible point query convention; four hash-selected queries per video.
- Decode at 25 fps, first 250 frames, resize to 256×256. Reproduce the official half-pixel coordinate correction.
- Download only the 12 required Kinetics validation class shards from the official CVDF S3 path list. Verify archive members, source video hashes, extracted frame hashes, public query hash and private oracle hash.
- Raw videos, frames, shards, private point tracks, predictions and VLM weights remain local on Tang. The individual source-video creator/platform terms remain applicable.

## Predeclared CPU methods

1. Stationary initial point.
2. KLT forward and KLT forward-backward (1.5 px cycle threshold).
3. Static-template recovery.
4. ORB/RANSAC partial-affine background compensation followed by KLT.
5. Initial-patch normalized cross-correlation gates at 0.55 / 0.70 / 0.85.
6. New gradient-identity gate on the KLT-FB trace at cosine thresholds 0.50 / 0.65 / 0.80. The descriptor concatenates Sobel x/y gradients and L2-normalizes them; positive brightness gain and additive offset should not change the descriptor in the unclipped regime. Low-texture patches abstain.

OpenCV OpenCL is explicitly disabled in CPU stages. CPU P50/P95 is reported as per-query mean milliseconds per target frame; shared affine-registration time is reported separately.

## DEV-only selection

Candidates: KLT-FB, template recovery, affine-stabilized KLT, three NCC gates and three gradient gates.

Eligibility:
- scene-macro visible coverage >= 0.25;
- scene-macro invisible false-accept rate <= 0.25.

Among eligible candidates, maximize scene-macro TAP-Vid Jaccard averaged over 1/2/4/8/16 px, then visible coverage. If none qualifies, use KLT-FB. No TEST metric may change this selection.

## Image-plane and camera-motion diagnostics

Report screen-space ground-truth displacement separately from the residual after mapping points through the estimated dominant-background affine transform. The latter is only a 2-D background-relative proxy. Kinetics provides no calibrated camera pose or depth; no physical camera/world-motion or 3-D parallax claim is allowed.

Also report per-scene mean luminance, luminance spread, Laplacian variance, ORB keypoint count, affine acceptance, affine reprojection residual, and CPU registration time.

## VLM baseline and admission gate

- Model: local SmolVLM2-2.2B-Instruct, loaded in BF16 when supported, otherwise FP16.
- Input: marked first-visible query frame + a 64×64 crop centered on the marked point + one later target frame at fixed 50% or 90% of the remaining clip.
- Output space: exactly one of VISIBLE / NOT_VISIBLE / UNCERTAIN.
- DEV gate requires at least 8 visible and 8 non-visible labels, invalid-format rate <= 0.20, non-abstaining coverage >= 0.50, balanced accuracy >= 0.55, and non-visible false-accept rate <= 0.25.
- If any DEV gate fails, TEST VLM inference is written as SKIPPED and the TEST VLM oracle is never opened.

The VLM sees only the query frame, reference crop and one target frame; CPU trackers see every intermediate frame. This evidence-budget difference is explicit in the report.

## Leakage and failure policy

- Prediction code reads public queries and RGB frames only; it never opens the private oracle.
- DEV CPU scoring precedes tracker selection. DEV VLM scoring precedes the VLM gate.
- TEST predictions are SHA-256 pinned before one-time TEST scoring.
- No threshold, method, prompt, parser or model choice may be changed from TEST answers.
- Every stage is single-attempt and fail-closed. Preserve failed logs and partial downloads. Any repair requires a new GitHub source SHA and a new frozen batch directory.
- The runner's shared GPU lock is mandatory for both 2.2B inference stages; require no other CUDA process and at least 11,000 MiB free.
- No raw video, extracted frame, shard, private oracle, model weight, generated media or secret may be committed.

## Primary metrics

Visible PCK@8 (unconditional), visible coverage, accepted precision@8, visible false abstention, invisible false acceptance, visibility accuracy, TAP-Vid Jaccard at 1/2/4/8/16 px, scene-macro and micro aggregates, per-scene strata, CPU P50/P95, affine registration latency, and VLM balanced accuracy / coverage / invalid format / non-visible false accepts.

A high precision obtained by nearly zero coverage is not a success. A high PCK obtained by accepting all invisible points is not a success. A low affine residual is not proof of physical camera compensation.


## Batch45B preflight correction

The first targeted regression run caught that a regex parser accepted the substring VISIBLE inside an explanatory reply such as "probably visible". The failing run is preserved in `repo-batch45-tested/.automation/preflight-logs/targeted-parser-failure.log`. Batch45B changes the parser to a full-string match, permits only one exact label (with optional terminal period/exclamation), and rejects all explanatory prefixes/suffixes. The failed source snapshot is not reused for the experiment; the corrected source is frozen in a new Batch45B directory.
