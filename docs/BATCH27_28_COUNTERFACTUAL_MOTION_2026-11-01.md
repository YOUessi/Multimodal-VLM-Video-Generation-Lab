# Batch 27–28: new paired counterfactual object motion and explicit trajectories — 2026-11-01

Both frozen batches completed on Tang:
- Batch27 J1–J9: **9/9 stages exited 0**, 91/91 source tests passed. Frozen local git SHA `5a8e08e9777cb467b2e88fa7c094adcbc25072ec`, upstream snapshot `884e7405b421ef266cbbd22714971cf259ad0d64`.
- Batch28 K1–K4: **4/4 stages exited 0**, 93/93 source tests passed; publication guard found no model weights, source videos or private binaries in the frozen checkout. Local SHA `576b31f72569d5d267c52756d681af964c0198ac`, upstream `4fc0859534c0b3d5c57829e303ca86c7ea42ae71`.

## Research question

A previous CLEVRER public movement task revealed that the 2.2B model's performance barely changes when three different frames are replaced by three copies of a single midpoint frame, or when temporal ordering is reversed. A 12-probe motion-salient frame selector did not improve paired accuracy.

The new question is more specific: **does explicitly measured per-object motion help video-language models convert visual evidence into movement decisions?** We test a narrow, interpretable pixel tracker, not a learned universal object tracker.

## Dataset and protocol

- Procedural **2D moving shapes** with a *fixed camera* and an independent distractor. Six target colors and three target shapes, with stationary and four diagonal-direction movement classes.
- **20 new development scene pairs** (40 MP4s) for integration tests; **120 frozen test scene pairs** (240 MP4s). Each pair reverses video order and expected direction. Five direction classes are balanced, 48 test clips per class. Counterfactual pairs are **not independent observations** in the paired bootstrap.
- All 280 source videos and their SHA256 values are frozen; test videos are distinct from development clips by full-byte hashes. Official QA gold labels are computed from the video generation rule, but are **not read by the tracking or model-prompt code**.
- Color-based source-pixel tracker is supplied only with the question string and video frames. It computes the target's measured centroids at three sample times and returns numeric positions or abstains. The tracker was 40/40 correct on development clips before model testing.
- Five predeclared input conditions per VLM: **question-only**, **RGB 3 frames only**, **measured positions only**, **RGB+measured positions**, **RGB+deliberately reversed-order (conflicting) positions**. The last condition is an intentionally incorrect auxiliary measurement and should **not be presented as a legitimate production method**.
- Pretrained frozen `SmolVLM2-500M-Video-Instruct` and `SmolVLM2-2.2B-Instruct`, no fine-tuning in this study. Total **2,400 real GPU model inferences**, all saved with raw model outputs. 120 paired scene IDs are the cluster resampling unit.
- Five-choice random prior = 20%.

## Actual heldout test scores

| Input condition | 500M correct /240 | 2.2B correct /240 |
|---|---:|---:|
| Question and options only | **49** (20.4%) | **45** (18.8%) |
| RGB chronological 3 frames | **46** (19.2%) | **40** (16.7%) |
| Measured trajectory coordinates only | **43** (17.9%) | **46** (19.2%) |
| RGB + measured trajectory coordinates | **40** (16.7%) | **51** (21.3%) |
| RGB + intentionally reversed auxiliary trajectory | **39** (16.3%) | **47** (19.6%) |
| Deterministic question-grounded color-centroid CV + displacement rule | **240** (100%) | N/A (same CV baseline) |

Every VLM row had zero parsing failures. The deterministic CV-only tracker achieved **240/240** in this narrow synthetic domain, independent of the VLM.

Paired cluster bootstrap comparisons against RGB 3-frame VLM:
- 500M + measured positions: **-2.5 percentage points**, 95% CI **[-7.917, +2.917]**.
- 2.2B + measured positions: **+4.583 percentage points**, 95% CI **[-0.833, +10.0]**.
- Neither demonstrates a stable positive gain.

Counterfactual direction consistency over 120 forward/reversed pairs:
- 500M RGB-only **24/120**; RGB+positions **3/120**.
- 2.2B RGB-only **27/120**; RGB+positions **4/120**.
- Thus supplying the coordinate string did not make the models reliably use temporal motion direction.

## Sliced failure evidence

Each test movement direction had 48 examples. For the 2.2B RGB+coordinates condition, correct counts were:
- stationary **9/48**, up-right **15/48**, up-left **3/48**, down-right **22/48**, down-left **2/48**.

This reveals pronounced direction and option-choice biases in this controlled domain, not reliable object-centric reasoning.

## Interpretation

**The external tracker CAN recover the simple motion signal from pixels, but these two pretrained VLMs did not reliably convert even correct numeric trajectories into movement labels under the frozen prompt protocol.** This supports a *pipeline interface/coordinate-reasoning bottleneck* hypothesis. It does **not** prove an architectural attention or temporal-positional failure: a fixed prompt protocol, small model capabilities, target domain, or question/option biases may all contribute.

The 100% CV result is intentionally **domain-specific**: high-chroma synthetic objects with a known color palette, static camera, no difficult occlusion or natural-video appearance variation. A deterministic rule can solve this dataset, so it must not be marketed as a new state-of-the-art multimodal model or broad motion understanding solution.

The negative/neutral numerical-hint results are kept unchanged. No test-driven prompt rewriting or hyperparameter search is used to inflate these previously frozen test scores.

## Reproducible artifacts (Tang only for data/results)

- Source: `experiments/object_motion_counterfactual_data.py`, `target_motion_tracking.py`, `object_motion_counterfactual_eval.py`, `object_motion_error_slices.py`, `object_motion_review.py`.
- Frozen config: `configs/object_motion_v1.json`; tests under `tests/test_object_motion_*.py`.
- Source video manifest: `data/object_motion_counterfactual_v1/manifest.json`, SHA256 `1a15cdbcd4f2a9a2999bbbfc2d740048d1482d5c444fdc864f5b9b99924cf5bb`.
- Local results: `repo-batch28-tested/outputs/object_motion_counterfactual_v1/` with `500m.json`, `2b.json`, `comparison.json`, `COUNTERFACTUAL_MOTION_REPORT.md`, `ERROR_SLICES.md`.
- **Local HTML video demo:** `repo-batch28-tested/outputs/object_motion_counterfactual_v1/review.html`, showing the first eight frozen forward/reversed pairs with each model's original raw answer. These are not selected by favorability.
- All original/generated MP4s and model weights remain on Tang.

## Next hypothesis to test

The missing capability might be **answer selection from numeric evidence**, rather than visual target identification. A transparent agent should either use a validated deterministic motion tool to map a source-pixel trajectory directly to an option (with confidence gate), or abstain and fall back to VLM when tracking confidence is weak. Crucially, validate the unchanged tracker on a different video domain *without calibrating on its labels* and count all abstentions as unhandled. Do not claim synthetic 100% transfers to actual CLEVRER or natural camera videos.
