# Batch 40 — DAVIS appearance-consensus tracking and VLM visibility protocol

Date: 2026-11-02

## Motivation from Batch 39C

The independent TAPVid-MV DROID TEST showed that KLT forward-backward achieved 80.96% PCK@8 on depth-visible positions but falsely accepted 42/54 depth-occluded positions (77.78%). A stationary-coordinate baseline had higher visible PCK but falsely accepted every occluded and out-of-frame position. Template recovery reduced false acceptance but lost visible accuracy and still accepted many out-of-frame targets. SmolVLM2-500M coordinate prediction collapsed to the bottom-right image boundary on all 36 queries.

The next bottleneck is reliable visibility/abstention rather than another small PCK gain.

## Dataset

- Official DAVIS 2017 trainval 480p real-camera video frames, acquired through bounded HTTP ZIP byte ranges; no full archive extraction.
- 4 DEV and 8 TEST scenes, 16 frames per scene at fixed stride 2.
- New split salt and scene-disjoint selection; explicitly exclude the 12 scene names used in the prior DAVIS batch.
- Initial instance mask only is the inference query. Future annotation masks are private and read only by scoring after predictions are pinned.
- This is a new scene-disjoint draw from the DAVIS corpus, not a new dataset or a pretraining-novelty claim.

## Frozen CPU methods

1. Stationary initial mask centroid.
2. KLT forward-backward.
3. Multi-feature forward-backward consensus.
4. More conservative consensus gate.
5. Appearance-consistency consensus with per-feature patch NCC thresholds 0.45, 0.55 and 0.65. Each candidate compares an 11x11 mean-normalized grayscale patch between adjacent frames after LK correspondence. Mean normalization is intended to tolerate additive illumination shifts; it does not guarantee invariance to severe lighting, blur, scale or viewpoint changes.

The appearance thresholds are predeclared in the config. DEV selection chooses the highest unconditional visible PCK among candidates satisfying both a minimum visible coverage and a maximum false-accept rate on annotated-invisible positions. If none qualify, the predeclared guarded-consensus fallback is selected. TEST does not tune thresholds.

## Frozen VLM visibility baseline

- Local SmolVLM2-500M-Video-Instruct checkpoint, GPU-exclusive inference.
- The model sees only the first RGB frame with the initial object mask highlighted and four fixed later RGB frames (indices 3, 7, 11, 15).
- It must answer VISIBLE, ABSENT or UNCERTAIN. This is a visibility classifier, not a coordinate predictor or a tracker.
- DEV gate checks output format and decision coverage only. No TEST labels or accuracy are used to tune the prompt.
- CPU trackers see every intermediate RGB frame; VLM sees only the four fixed endpoint comparisons. The unequal evidence budget is reported explicitly.

## Metrics

- Visible centroid PCK@4/8/16, unconditional visible PCK, visible coverage and accepted precision.
- False acceptance on annotated-invisible positions; this does not distinguish complete occlusion from leaving the field of view.
- Mask-area shrink acceptance/wrong acceptance is a separate unverified proxy, never called ground-truth occlusion.
- CPU latency P50/P95 per scene and 2D background-affine registration acceptance.
- VLM visible recall, absent recall, balanced accuracy with abstentions counted wrong, decision coverage, selective accuracy and absent false-accept rate.
- Scene-cluster bootstrap, not frame bootstrap.

## Leakage and integrity

- Prediction code refuses to pin outputs if the Git worktree is dirty.
- CPU and VLM TEST predictions are generated and SHA-pinned before any TEST future masks are read.
- TEST scoring occurs once after all label-blind prediction files are sealed.
- DEV selection and VLM interface gate are SHA-bound to the frozen source and DEV result files.
- Model weights, videos, masks, private annotations, raw predictions and secrets remain on Tang; only source, tests, protocol and aggregate report are published.
- GPU inference is protected by the existing shared GPU lock and a minimum free-memory requirement.

## Limitations

- DAVIS instance-mask centroid is an image-plane proxy, not a material point or world-relative trajectory.
- An empty future mask means the annotated instance is absent; it does not distinguish full occlusion from leaving the field of view.
- A dominant 2D affine background fit is not calibrated 6-DoF camera motion or depth-aware parallax compensation.
- The VLM sees only sparse two-image comparisons and is not a video model baseline with full temporal access.
