# Batch 36 — independent real-camera object-centroid tracking (prospective protocol; 2026-11-01)

Batch 34 showed a sharp synthetic parallax/identity coverage–risk tradeoff (shape+affine 14/200 accepted, zero observed gate errors; plain HSV 121/200 accepted with 34 wrong). Both old heldouts are exposed: do **not** tune on their answers. This new study uses **real camera DAVIS 2017 video frames with independent instance-mask annotations**, not another procedurally generated video. It evaluates a distinct, narrower task: tracking the initial target instance-mask center over natural image frames, not classifying physical world-relative direction.

## Source and legal/reproducibility boundary

Official source: https://davischallenge.org/davis2017/code.html (2017 TrainVal 480p images and annotations; archive https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip). DAVIS authors generally release sequences under CC BY 4.0 but externally sourced footage can have its own terms; preserve SOURCES.md and don't republish media until individual provenance is reviewed. Source reference: Pont-Tuset et al. 2017, The 2017 DAVIS Challenge on Video Object Segmentation. TAP-Vid DAVIS is an alternative larger tracking benchmark; it has **not** been used or scored by Batch 36.

The official ZIP archive is queried by verified byte ranges rather than downloading or extracting it wholesale; read ZIP central directory (with SHA256), reject unexpected layouts, path traversal, oversize/uncompressed entries, or HTTP servers that ignore Range, and validate each selected file against ZIP CRC plus a locally frozen SHA256. Media/annotations, local per-scene labels, raw predictions and review materials remain on Tang under ignored data/ and outputs/. No media or weights in GitHub.

## Predeclared independent split, controls and domain limitations

Select 12 distinct real DAVIS sequences by sorting scene names according to SHA256 of a new literal split salt and each scene's name. The first four are DEV and the next eight are heldout TEST. No sequence overlaps splits. Each contributes 8 original 480p frames (source indices 0,3,...21), with corresponding instance annotation masks; **96 real photographic frames** (192 JPEG/PNG source assets), not 96 independent videos. One foreground instance is selected solely from the first-frame mask (largest labeled object); its initial binary mask and center are a legitimate query. Later masks are private scoring-only data and NEVER entered into the predictor.

Predeclared, non-optimized baselines: (a) first-position constant stationary-pixel, (b) forward Lucas–Kanade KLT seeded with initial-object features, (c) KLT with forward/backward 1.5px consistency reject and min-five-features abstention. First-to-last ORB+RANSAC 2D global affine uses only pixels, excluding initial target mask, and is reported as a **background-relative diagnostic**, never world-relative motion ground truth. No threshold or model changes after TEST scoring; only DEV can guide a separate future batch. Point-mask centroid (which changes with object shape) is a PROXY for tracking, not a true physical material-point trajectory. Fast translation, occlusion, deforming objects, depth parallax, and video camera motion may break all baselines. 8 sparsely sampled frames per source limit temporal scope.

Primary metric: visible-frame image-plane centroid PCK@8px over **all** GT-visible non-query frames, where abstention counts incorrect. Also PCK@4/16, conditional precision, accepted coverage, wrong accepted, absent-object false acceptance, loss due to abstention, per-scene CPU P50/P95, background registration coverage, and scene-cluster bootstrap difference vs constant-position. Because future segmentation masks are scored only after frozen predictions, the TEST stage must verify all asset and prediction hashes and reject missing or duplicated output. No VLM point-tracking comparison: the prior VLM multiple-choice world-motion classification is a different task and has no direct benchmark label here.

## Strict sequence

1. Read official central directory, acquire only selected real-camera JPEG/PNG entries, freeze source SHA, future-label boundary, first-frame query and private oracle.
2. Byte SHA/CRC audit, regression tests and source security guard.
3. Score 4 DEV source videos and log failures. **No TEST selection or tuning.**
4. Freeze TEST label-blind trajectories and computer-time traces for 8 independent source videos.
5. Score TEST once with source-scene-cluster confidence intervals; retain negative results and camera-relative/world-relative distinction.
6. Publish aggregate numbers to main only after the immutable run. The existing Batch 35 and its outputs must remain untouched.

Synthetic test footage in unit tests is for code validation ONLY; it is not counted as evidence of real-world transfer. A released academic benchmark's source videos may have appeared in pretrained model corpora and does not prove model-unseen domain generalization.

## Frozen pre-TEST development result and byte-integrity hardening (source commit)

The official archive's 192 selectively downloaded original assets have been verified by CRC and SHA256. The 12 selected source clips and split were sealed into `data/davis_real_v1/manifest.json` (SHA256 `4bc039d7e96a84f99ce3919f165c6d4261cd7b0413937933041e1be740ca3a4a`). The eight TEST scenes have not been scored or used to choose tracking thresholds in this batch.

The four DEV camera sequences contain 28 visible non-query evaluations. The fixed screen-constant position baseline scores PCK@8px **2/28 (7.14%)**, forward KLT **16/28 (57.14%)**, and forward-backward KLT **17/28 (60.71%)**. The DEV PCK gain over constant center is **+53.57 pp** for forward-backward KLT; this is not a heldout effectiveness claim. Both forward KLT approaches accepted 28/28 visible frames, hence no DEV abstentions and the forward-backward filter provides no coverage advantage here. Dominant 2D background-affine registration accepted only **1/4** DEV scenes. CPU per-scene medians for static/forward KLT/forward-backward KLT were **0.00/7.27/8.69 ms** (not GPU or end-to-end video decoding latency). Do not tune using any TEST result.

Before opening TEST gold, predictor now additionally validates first-frame query PNG, query center, and immutable first source palette-index annotation against the source archive digest. Each DEV/TEST predictions JSON is independently pinned to a sidecar SHA256 created by `predict-*`, and the `score-*` action rejects missing or mismatched prediction SHA files. Two additional regression tests cover first-frame-only access and checksum drift. These protect against accidental source/prediction mutation, not against a malicious operator editing both predictions and their checksum. Frozen git SHA plus per-stage execution logs provide the independent audit trail.
