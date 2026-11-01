# Batch 33 — prospective independent affine-camera and same-color identity shift

Protocol committed before any heldout test scoring. No test labels or VLM accuracy have been inspected during algorithm selection.

## Why this is the next experiment

Batch 32's real (but **synthetic**) heldout camera-translation study showed global phase-correlation registration helped when the camera translates, but phase correlation cannot represent rotation or zoom. The stronger claim of robust camera-aware video motion is not supported. The old color-target finder also does not reliably resolve two same-colored objects from a color-only mask. The next experiment therefore attempts to **falsify** a wider generality claim, with three frozen reference paths: ordinary screen-plane HSV target tracking, the same HSV + translation-only phase registration, and HSV + ORB features / RANSAC partial-affine background registration.

## Prospective protocol

- Independently generated new **40 DEV source scene pairs (80 MP4s) and 100 heldout TEST source scene pairs (200 MP4s)**. Sources and video byte hashes live in `data/affine_camera_motion_v1/manifest.json` only on Tang. DEV and TEST are independent generator seeds (10826923, 328120059) and exact byte-disjoint.
- Four balanced nuisance families × five balanced motion classes: camera pan, camera rotation, camera zoom, and identical-color distractor + target occlusion + moving camera; all are textured **2D synthetic** renderings with the same broad shape/color conventions used earlier. This is a **new source-scene test**, not a fully novel natural-visual-domain test.
- Truth is the target movement **relative to the stationary world background**, not its image-plane displacement. Every clip has a reversed mate, and the world-direction answer reverses. Evaluation samples **independent scene pairs** for confidence intervals rather than counting forward/reverse clips independently.
- A fixed source-question selects a target color **and shape**. The legacy HSV tracker finds a target by color only, intentionally exposing identity confusion when a same-color distractor exists. The new feature-based camera alignment can fix *camera* movement, but cannot fix erroneous target identity; this limitation must be visible in gate-accepted wrong overrides.
- The new geometrical estimator uses video-only ORB matches, 0.80 nearest-neighbor ratio filter, partial-affine RANSAC (3 px), at least 18 matches, 14 inliers, and 0.40 inlier fraction; weak/unphysical registration abstains. These are predeclared after independent DEV and are not selected using Batch 32 or Batch 33 heldout correctness.
- Freeze the test clip manifest and source SHA before test inference. The CV tracker and frozen GPU VLMs must predict before the heldout scorer can inspect labels. The report includes full denominator, coverage, abstentions, tool-gate **wrong overrides**, reverse consistency, per-nuisance paired scene bootstrap confidence intervals and CPU P50/P95. Post-hoc camera mapping errors use generator metadata **only after** blind test predictions are saved.
- Pretrained local SmolVLM2 500M (3 frames) and 2.2B (3, 5 frames) remain frozen. This study does not train them, establish architecture improvements, or validate performance on natural camera video.
- The local GitHub `main` snapshot, frozen code SHA, 10 separate logged execution stages, source byte checks, full regression suite, GPU shared lock and privacy guard protect reproducibility. No model weights, MP4s, private question-level outputs or secrets belong in GitHub.

## Development-only preflight (not heldout evidence)

On newly generated 80 DEV MP4s, **75/80** correct with ORB+RANSAC vs **57/80** unregistered, **26/80** translation-only phase. The new estimator accepted 59 DEV clips at its conservative nonstationary gate with zero DEV wrong overrides; this must **not** be assumed on heldout. Same-color/occlusion yielded only **15/20** correct DEV directions: this is an explicit weakness, not a case to silently remove. The registration estimator's median CPU expense on DEV was **26.59 ms/clip**, compared with **14.71 ms/clip** plain and **27.49 ms/clip** phase. These are development measurements, subject to device variance. Nine new unit/MP4 tests passed, and all 121 project regression tests passed in source preflight.

## Interpretation guardrails

A test gain restricted to simple generated shapes would be domain-specific classical CV tool engineering, **not VLM temporal-reasoning advancement**. Strong accepted wrong overrides can negate any accuracy gain; report all gate errors, even if they harm the combined model. No threshold is to be changed after final heldout test scoring. If source data or code changes are required for a failure, preserve the failed batch, create a new source SHA and a new frozen batch rather than retry the changed source under the same evidence ID.

## Next after Batch 33

Prioritize external, consented/licensed **real camera videos** with independently annotated camera-/world-relative target motion and ambiguity, then evaluate whether the geometry and target-ID gates remain reliable. Keep separate any future blind human evaluation for video generation; the current synthetic-only ZeroScope validation MSE does not constitute human-rated video quality.
