# Batch45E — fresh Kinetics action-class-disjoint tracking and VLM visibility

- Frozen source SHA: e59aa07c6ab69a40ab2bb35e64cad4645410d10c
- Dataset: TAP-Vid-Kinetics / Kinetics-700-2020 validation; 12 new YouTube video IDs from 12 action classes, all disjoint from Batch42.
- Split: 6 DEV / 6 TEST; 4 first-visible point queries per scene.
- Frozen manifest SHA-256: 5fe568354623a9a122934be17c2c86cfe4b57bbe4f7c41bc8b016ebd5839be08
- Public query SHA-256: 39fbfeaa7c6a0effa47127e9ef9ae5034cbaa982ee48a129fdfa127111f9592c
- TEST CPU prediction SHA-256: ea6a0ba5e8c1743f1c48849c9d41978f76b826e9b237c581eb8cc60c63f63ba5
- VLM DEV gate: **BLOCK**

## DEV-only tracker selection

Selected method: **affine_stabilized_klt**. Eligibility required visible coverage ≥ 0.25 and invisible false-accept rate ≤ 0.25; the objective was scene-macro Jaccard averaged over 1/2/4/8/16 px. TEST was not used for selection.

| Method | PCK@8 visible | Coverage | Accepted precision@8 | Invisible false accept | Jaccard mean | CPU P50 ms/frame | CPU P95 ms/frame |
|---|---:|---:|---:|---:|---:|---:|---:|
| stationary_point | 30.8% | 100.0% | 30.8% | 100.0% | 0.084 | 0.0002 | 0.0003 |
| klt_forward | 49.5% | 88.2% | 54.1% | 58.1% | 0.227 | 0.0863 | 0.1045 |
| klt_fb | 40.7% | 49.7% | 80.5% | 13.6% | 0.241 | 0.0625 | 0.1567 |
| template_recover4 | 21.8% | 39.5% | 69.5% | 15.5% | 0.115 | 0.0367 | 0.0553 |
| affine_stabilized_klt | 38.4% | 45.4% | 85.6% | 4.5% | 0.251 | 0.1435 | 0.2658 |
| appearance_fb_055 | 23.7% | 24.4% | 96.1% | 9.5% | 0.156 | 0.0672 | 0.1693 |
| appearance_fb_070 | 19.7% | 20.4% | 96.1% | 7.0% | 0.130 | 0.0668 | 0.1682 |
| appearance_fb_085 | 16.3% | 16.9% | 95.7% | 2.0% | 0.113 | 0.0668 | 0.1677 |
| gradient_fb_050 | 19.7% | 20.4% | 95.9% | 6.1% | 0.140 | 0.0667 | 0.1672 |
| gradient_fb_065 | 15.1% | 15.7% | 96.3% | 4.9% | 0.111 | 0.0665 | 0.1671 |
| gradient_fb_080 | 11.5% | 11.9% | 96.1% | 0.5% | 0.090 | 0.0665 | 0.1672 |

## Independent TEST — one-time sealed score

| Method | PCK@8 visible | Coverage | Accepted precision@8 | Invisible false accept | Jaccard mean | CPU P50 ms/frame | CPU P95 ms/frame |
|---|---:|---:|---:|---:|---:|---:|---:|
| stationary_point | 35.0% | 100.0% | 35.0% | 100.0% | 0.181 | 0.0002 | 0.0003 |
| klt_forward | 56.6% | 82.1% | 67.5% | 43.4% | 0.350 | 0.1048 | 0.1272 |
| klt_fb | 48.1% | 58.7% | 81.5% | 20.6% | 0.347 | 0.1593 | 0.2382 |
| template_recover4 | 32.7% | 41.2% | 82.1% | 3.2% | 0.258 | 0.0432 | 0.0784 |
| affine_stabilized_klt | 44.2% | 58.1% | 67.8% | 20.8% | 0.316 | 0.2931 | 0.3665 |
| appearance_fb_055 | 42.3% | 43.1% | 94.5% | 2.2% | 0.337 | 0.1715 | 0.2555 |
| appearance_fb_070 | 37.8% | 38.0% | 98.7% | 0.4% | 0.309 | 0.1703 | 0.2545 |
| appearance_fb_085 | 27.8% | 28.0% | 98.7% | 0.0% | 0.244 | 0.1698 | 0.2544 |
| gradient_fb_050 | 35.8% | 36.2% | 96.5% | 0.4% | 0.310 | 0.1694 | 0.2537 |
| gradient_fb_065 | 31.8% | 31.9% | 98.6% | 0.4% | 0.287 | 0.1691 | 0.2531 |
| gradient_fb_080 | 24.5% | 24.7% | 98.3% | 0.0% | 0.224 | 0.1692 | 0.2530 |

Frozen DEV-selected method 'affine_stabilized_klt' on TEST: PCK@8 44.2%, visible coverage 58.1%, accepted precision 67.8%, invisible false accept 20.8%, scene-macro Jaccard 0.316.

## Image-plane versus background-relative diagnostics

The affine transform is estimated from RGB background features only. The residual below is a 2-D dominant-background affine proxy, not calibrated camera pose, depth-aware parallax or 3-D world motion.

| TEST scene | Mean luma | Luma std | Laplacian variance | ORB keypoints | Affine acceptance | Screen displacement mean px | Background-relative residual mean px |
|---|---:|---:|---:|---:|---:|---:|---:|
| kinetics_06 | 180.3 | 67.3 | 408.8 | 458.0 | 99.6% | 20.40 | 9.15 |
| kinetics_07 | 109.8 | 49.8 | 1946.3 | 455.0 | 100.0% | 87.44 | 58.32 |
| kinetics_08 | 90.7 | 52.5 | 2176.7 | 467.0 | 100.0% | 23.38 | 21.29 |
| kinetics_09 | 97.1 | 62.2 | 1771.3 | 468.0 | 96.1% | 1.63 | 1.75 |
| kinetics_10 | 83.9 | 62.0 | 1231.6 | 457.0 | 100.0% | 71.12 | 21.72 |
| kinetics_11 | 116.1 | 62.0 | 616.0 | 472.0 | 99.2% | 10.17 | 12.07 |

## SmolVLM2-2.2B visibility baseline

DEV gate: **BLOCK**; checks: {"balanced_accuracy": false, "class_support": true, "coverage": true, "invalid_rate": true, "not_visible_false_accept_rate": false}.
DEV VLM: 11/48 correct, balanced accuracy 23.9%, coverage 64.6%, non-visible false accept 80.0%, invalid format 0.0%.
TEST VLM inference/scoring status: **SKIPPED_DEV_GATE_BLOCKED**. TEST oracle read: False.

## Interpretation and limitations

- The 12 clips are fresh video IDs and action classes relative to Batch42, but they remain within the TAP-Vid-Kinetics/Kinetics-700-2020 source family. This is an independent within-family holdout, not an external dataset-family generalization claim.
- Kinetics clips are natural YouTube footage; each source video's creator/platform terms remain applicable. Raw clips, extracted frames, private point tracks, predictions and model weights remain local on Tang.
- CPU trackers process every intermediate 25-fps frame. The VLM receives a marked query frame, a local reference crop and one later frame; the temporal evidence budgets differ.
- The affine compensation is only a 2-D dominant-background proxy. No calibrated camera pose, depth or 3-D world-motion claim is made.
- Coverage and abstention remain first-class metrics. A high PCK obtained by accepting almost nothing, or a low false-accept rate obtained by abstaining on nearly everything, is not a complete success.
- Only DEV may select tracker thresholds/methods or admit the VLM. TEST predictions are SHA-pinned before the one-time TEST scoring pass; no TEST result may be used to revise this batch.

## Reproducibility

- DEV CPU prediction SHA-256: 848063dfda93cf4d6929bbce1fbfa0a7eac45754c513722e9622bc845c86e4fd
- TEST CPU prediction SHA-256: ea6a0ba5e8c1743f1c48849c9d41978f76b826e9b237c581eb8cc60c63f63ba5
- DEV VLM prediction SHA-256: 890272dd272f1fa951c673647f87b8f892cd10302c0fe0aab12295bfcd824076
- TEST VLM prediction SHA-256: e8cd2b8131fae4552bd8dde9d01c6636df6f87e9b813d5ce7ed73d6ede82619e
- All private data and media are local; only code, tests, configuration and this aggregated report are published.
