# Batch 41 — RoboTAP point visibility, occlusion abstention and background-affine compensation

**Prospective protocol; source SHA is frozen only after all code, tests, config and this document are committed. No Batch41 TEST result has been inspected or used for selection.**

## Why this experiment follows Batch40B and Batch39C

- Batch40B's new DAVIS scene draw had no invisible future targets. Its guarded KLT operating point accepted only 16/120 visible TEST positions (13.3% coverage); the VLM had 27/32 visible predictions and five abstentions, but absent recall was undefined.
- Batch39C's DROID point-track subset supplied depth-derived visibility, but Forward–Backward KLT still falsely accepted 42/54 depth-occluded positions (77.78%). Its ORB affine camera diagnostic moved in opposite directions on DEV and TEST.
- The next missing evidence is a new source with explicit per-point visibility labels, multiple moving surfaces, and a frozen DEV/TEST split. RoboTAP provides annotated point tracks and occlusion flags in a separate synthetic robotics domain.

## Dataset and legal boundary

- Source: official Google DeepMind TAPNet RoboTAP archive, specifically robotap_split0.pkl and robotap_split1.pkl from https://storage.googleapis.com/dm-tapnet/robotap/robotap.zip.
- The full ZIP is approximately 12.6 GB. The acquisition code uses HTTP byte ranges to extract only the two pinned members; it does not download the full archive.
- Upstream TAP-Vid README states RoboTAP videos and TAP-Vid annotations are CC BY 4.0. Preserve attribution. No source videos, pickle shards, generated frames, private tracks or weights are committed.
- RoboTAP is **synthetic robotics video**, not natural camera footage. This experiment is not a claim of real-world camera generalization.
- The official upstream README describes RoboTAP samples as video frames, point tracks and per-frame occlusion flags. The implementation validates the actual array shapes and coordinate range before freezing.

## Frozen sampling and split

- Use only the two named official shards. Verify the archive ETag, total byte length, ZIP member CRC32, local member SHA-256 and member size.
- Derive a recording/episode ID from the stable source prefix before the camera-view suffix (for example, gear_v2_17). Group all camera views with the same episode ID before sampling.
- Deterministically rank distinct episode IDs by SHA-256 under the fixed Batch41 salt; take 20 distinct episodes. For each episode, choose exactly one camera view using a second independent view-key hash. Do not inspect point trajectories or occlusion fractions to choose episodes or views.
- Assign the first six hash-ranked episode groups to DEV and the remaining fourteen to TEST. The manifest audit rejects any duplicate episode ID, including duplicate views within one split.
- Within each scene, hash-rank point IDs and take at most four. For each selected point, query at its first annotated visible frame, matching TAP-Vid's first-query protocol. A point with no later frame is excluded by a predeclared rule.
- Save full RGB arrays, public first-visible queries and a separate private oracle. The predictor process may read only RGB arrays, the public query file and source hashes; it must never load the private future coordinates or occlusion labels.
- Freeze source-member hashes, frame-array hashes, public-query hash, private-oracle hash and manifest hash before prediction.

## Methods and outputs

### CPU baselines

1. stationary_point: query coordinate held constant.
2. klt_forward: one-point forward Lucas–Kanade.
3. klt_fb: forward/backward KLT with a fixed 1.5 px round-trip gate.
4. template_recover4: local normalized template matching with fixed 21 px patch, 32 px search radius, 0.75 NCC threshold and at most four consecutive misses.
5. affine_stabilized_klt: estimate adjacent-frame background partial-affine transforms from ORB/RANSAC, accumulate a stabilized image plane, track with FB-KLT there, then map the point back to original image coordinates. If the transform is invalid, abstain; do not silently substitute oracle camera motion.

All methods emit both point coordinates and a visible/abstain decision. A failed point is not replaced by a new identity.

### Camera-motion diagnostics

- Report image-plane point displacement separately from displacement after subtracting the estimated 2-D background affine projection.
- This is only a background-relative image-plane proxy. RoboTAP does not provide calibrated camera pose in this experiment; do not call the residual world-relative 3-D motion.
- Report affine estimation coverage and reprojection residuals. A high affine inlier rate is not proof of physical camera calibration or depth-aware parallax correction.

### VLM baseline

- Frozen local SmolVLM2-500M receives a query frame with a small red point marker and a later unmarked RGB frame.
- At two fixed relative target times (50% and 90% of the query-to-end interval), predict exactly one of VISIBLE, ABSENT, or UNCERTAIN.
- ABSENT means the annotated point is not visible, whether due to occlusion or leaving the image; the label does not distinguish those causes.
- The VLM does not receive future coordinates, point IDs beyond the query image, or private visibility labels. It is a visibility classifier, not a coordinate tracker.
- DEV interface gate checks only output-format validity and decision coverage. If the gate fails, TEST VLM inference is recorded as skipped; no prompt repair is made from TEST replies.

## Metrics and DEV-only selection

- Report TAP-Vid-style Jaccard@8, visible PCK@8, visible coverage, accepted precision, occluded false-accept rate, visible false-abstain rate and occlusion accuracy.
- Also report CPU P50/P95 per query, affine coverage, VLM visible/absent recall, balanced accuracy, coverage, selective accuracy, invalid output rate and absent false-accept rate.
- Keep every query/frame in its applicable denominator. Abstentions are not silently removed from unconditional PCK or visibility accuracy.
- DEV selection maximizes Jaccard@8 among candidates meeting visible coverage >= 0.40 and occluded false-accept rate <= 0.25; tie-break by visible coverage. If no method qualifies, use the predeclared klt_fb fallback.
- No TEST threshold, prompt, method, frame or model selection is allowed. TEST predictions must be pinned before the scoring stage opens the private oracle.

## Acceptance and limitations

- Require a clean source tree and exact source SHA before prediction; pin prediction bytes with SHA-256.
- Run the full regression suite before inference and again before final audit.
- Keep original failed logs and partial downloads. Never overwrite a frozen manifest, prediction, oracle or report.
- Do not commit media, pickle shards, extracted frame arrays, private labels, model weights or secrets.
- The dataset is synthetic, only two of five RoboTAP shards are sampled, and the split is a project-local scene-disjoint evaluation rather than an official leaderboard split. Results cannot establish natural-video generalization or calibrated world motion.

## References

- TAP-Vid/RoboTAP official repository and license: https://github.com/google-deepmind/tapnet
- TAP-Vid dataset page and task definition: https://tapvid.github.io/
- RoboTAP paper citation is listed in the official TAPNet README.
