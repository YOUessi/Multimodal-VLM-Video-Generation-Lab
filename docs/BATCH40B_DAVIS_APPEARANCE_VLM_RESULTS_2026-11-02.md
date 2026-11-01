# Batch 40B — DAVIS scene-disjoint appearance consensus and VLM visibility

- Frozen source commit: `0bdb386c3a6a7fc8b052de1fdcf93233433ac25b`
- Frozen stages: 17/17 completed; final prediction/data/source-SHA audit passed.
- Full project regression: **249 passed** (3 existing Pillow deprecation warnings).
- DEV / TEST scenes: 4 / 8; 16 sampled frames per scene.
- Dataset manifest SHA-256: `2f3898953bd5521b0172039b691d364b73e7e0c14743ca77f38f9252efb93f4d`
- DEV-selected method: `klt_fb_guarded` (fallback because no appearance candidate met the predeclared minimum coverage/occlusion-evidence requirements).
- VLM DEV interface gate: passed (16 queries; invalid=0; coverage=1).
- SmolVLM2-500M weights SHA-256: `b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3`
- DAVIS source audit: 384 assets verified.

## CPU tracking and abstention

| Split | Method | Visible PCK@8 | Visible coverage | Accepted precision@8 | Invisible false accepts | Invisible evaluations | Mask-shrink false accepts (proxy only) | CPU P50/P95 ms/scene |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DEV | klt_fb_appearance_045 | 40.00% | 58.33% | 68.57% | 0 | 0 | 0/0 | 22.73/33.11 |
| DEV | klt_fb_appearance_055 | 40.00% | 58.33% | 68.57% | 0 | 0 | 0/0 | 22.75/32.29 |
| DEV | klt_fb_appearance_065 | 40.00% | 58.33% | 68.57% | 0 | 0 | 0/0 | 23.46/33.86 |
| DEV | klt_fb_consensus | 40.00% | 58.33% | 68.57% | 0 | 0 | 0/0 | 12.33/16.22 |
| DEV | klt_fb_guarded | 21.67% | 23.33% | 92.86% | 0 | 0 | 0/0 | 6.78/14.29 |
| DEV | klt_forward_backward | 48.33% | 83.33% | 58.00% | 0 | 0 | 0/0 | 15.11/15.89 |
| DEV | stationary_pixel | 16.67% | 100.00% | 16.67% | 0 | 0 | 0/0 | 0.0/0.0 |
| TEST | klt_fb_appearance_045 | 20.83% | 33.33% | 62.50% | 0 | 0 | 0/1 | 10.08/30.48 |
| TEST | klt_fb_appearance_055 | 20.00% | 31.67% | 63.16% | 0 | 0 | 0/1 | 10.56/29.09 |
| TEST | klt_fb_appearance_065 | 20.00% | 31.67% | 63.16% | 0 | 0 | 0/1 | 12.25/28.86 |
| TEST | klt_fb_consensus | 20.83% | 34.17% | 60.98% | 0 | 0 | 0/1 | 7.65/15.39 |
| TEST | klt_fb_guarded | 13.33% | 13.33% | **100.00%** | 0 | 0 | 0/1 | 6.41/12.55 |
| TEST | klt_forward_backward | **27.50%** | 70.00% | 39.29% | 0 | 0 | 0/1 | 14.19/16.42 |
| TEST | stationary_pixel | 6.67% | 100.00% | 6.67% | 0 | 0 | 1/1 | 0.0/0.01 |

The selected guarded tracker accepted only 16/120 visible TEST positions. Its 100% accepted precision therefore represents a **13.33% coverage operating point**, not high overall tracking accuracy. Its unconditional PCK@8 is also 13.33%. The less conservative forward-backward baseline had 70% coverage but only 39.29% precision among accepted predictions.

## DEV-only appearance threshold selection

| Candidate | Visible PCK@8 | Visible coverage | Invisible false-accept rate | Eligible |
|---|---:|---:|---:|---:|
| klt_fb_appearance_045 | 40.00% | 58.33% | n/a | no |
| klt_fb_appearance_055 | 40.00% | 58.33% | n/a | no |
| klt_fb_appearance_065 | 40.00% | 58.33% | n/a | no |

No appearance threshold qualified: this DAVIS draw contained no verified invisible future-mask positions in either split, so the predeclared minimum absent-evaluation requirement could not be met. The guarded method was a declared fallback, not evidence that appearance tuning succeeded.

## SmolVLM2-500M two-image visibility classification

| Split | Visible recall | Absent recall | Balanced accuracy (abstentions count wrong) | Coverage | Selective accuracy | Absent false-accept rate | Uncertain / invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| DEV | 100.00% | n/a | n/a | 100.00% | 100.00% | n/a | 0 / 0 |
| TEST | 84.38% | n/a | n/a | 84.38% | 100.00% | n/a | 5 / 0 |

TEST contained 32 VLM visibility queries, all with visible ground truth: 27 were predicted VISIBLE and 5 UNCERTAIN. There were **zero absent targets**, so absent recall, balanced accuracy, and absent false-accept rate are undefined. This batch does not validate VLM occlusion detection.

## Interpretation limits

- This is a new scene-disjoint draw from the DAVIS 2017 corpus, **not a new dataset** and not a pretraining-novelty claim.
- DAVIS mask centroid is an image-plane proxy, not a material point or world-relative trajectory.
- Empty future masks would mean the annotated instance is absent from the frame; they would not distinguish full occlusion from leaving the field of view. This draw had no such absent positions.
- Mask-area shrink is reported only as a proxy, not as verified occlusion.
- Background affine is a 2-D diagnostic, not calibrated 6-DoF camera motion or depth-aware parallax compensation.
- CPU trackers use all intermediate RGB frames; VLM sees the initial highlighted mask image and four fixed later frames.
- VLM visibility is a separate classifier, not a point-coordinate predictor or a tracker.
- DEV selection was fixed before TEST scoring. TEST labels were used only for the final one-time score.

## Execution integrity

- Batch40B was frozen at source SHA `0bdb386c3a6a7fc8b052de1fdcf93233433ac25b`.
- 17/17 frozen automation stages completed with return code 0.
- Initial and final full regression runs each reported 249 passed.
- Final audit verified 384 source assets, prediction pins, source SHA consistency, and data/result integrity.
- Videos, extracted frames, private masks, model weights, raw predictions and secrets remain local on Tang; none are included in this report or committed.
