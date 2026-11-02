# Batch46E independent Kinetics frozen TEST: completed, not a model improvement

Source frozen at `df14676e1f52d9e1a0096581a8558cbbb3f8cd67`; all 22 preapproved R00–R21 stages completed successfully on Tang at 2026-11-02 10:32 UTC, including prediction hashes, one-time TEST scoring, final source/security and report audits. Full local aggregated report: `../outputs/kinetics_batch46/BATCH46_INDEPENDENT_TRANSFER_REPORT.md` (private predictions and videos remain local).

## Data firewall

Twelve previously unused TAP-Vid-Kinetics/Kinetics-700 videos and twelve action categories, disjoint by both video ID and action label from Batch42 and Batch45E. Six DEV and six sealed TEST clips, four official initial-visible point queries per video. The Batch45E DEV-selected `affine_stabilized_klt` was transferred unchanged. TEST CPU predictions were pinned to SHA-256 `eb41ee3328422dcb5ee443ff1b06942f0dbf140aab366fe33355414e1b7f1506` **before** the private TEST labels were opened for scoring. Test oracle was not used to select thresholds or the method.

## Core results

| Metric | DEV (6 videos) | TEST (6 videos) |
|---|---:|---:|
| Affine KLT PCK@8, visible unconditional | 40.3% | **60.5%** |
| Affine KLT visible coverage | 46.7% | **61.0%** |
| Affine KLT accepted precision@8 | 85.0% | 95.0% |
| **Affine KLT invisible false-accept rate** | **17.8%** | **36.0%** |
| Affine KLT scene-macro Jaccard | 0.219 | 0.423 |
| CPU P50 ms/target frame (shared registration excluded) | 0.215 | 0.262 |

The previously selected affine tracker was **not** reliably conservative under this class-disjoint shift: TEST invisible/occluded false acceptance was 36.0%, about twice the DEV rate. A larger TEST Jaccard does not cancel this safety failure. These are six scene-macro averages, not an IID large-scale benchmark.

A fixed, pre-existing `appearance_fb_055` comparator on the same TEST scored Jaccard 0.469, coverage 56.7%, and false acceptance 6.8%; `gradient_fb_050` scored Jaccard 0.461 and false acceptance 9.2%. These are retrospective **diagnostics only**: it is forbidden to promote these alternatives based on this opened TEST set. Any new method selection must be made on a different DEV split and evaluated on a new blind TEST.

## SmolVLM2-2.2B visibility gate

The frozen three-image DEV VLM completed 48 classification queries; 21 correct (43.75% all-query accuracy), 24 UNCERTAIN, zero invalid outputs. Balanced accuracy 35% failed the preregistered 55% minimum, despite 50% coverage and acceptable 16.7% NOT_VISIBLE false acceptance. The gate was **BLOCK**; TEST VLM inference and scoring were **skipped**, correctly. Never report a TEST VLM accuracy for this batch.

## Interpretation and limitations

- An affine image-plane residual is **not** ground-truth 3D camera motion or foreground/background identification.
- KLT CPU microbenchmark excludes shared affine registration, decoding, device memory and end-to-end throughput.
- The 6-scene shift has strong category and visibility composition differences (stationary-point baseline alone obtained 60.4% TEST visible PCK@8), limiting naive cross-split improvement claims.
- No inference about deployment safety, physical ego-motion estimation or objective tracking superiority follows from these numbers.

## Research consequence

The next actionable bottleneck is the **conservative acceptance of compensations and tracked targets** when image support is spatially localized, temporally discontinuous, or mixed with foreground motion. Batch47B/C and Batch47D investigate evidence failure on DEV only; NONE may reuse Batch46E TEST scores for threshold tuning. New independently frozen class-disjoint video and fresh TEST are required to claim an improvement.

No source, raw video, oracle, predicted point tracks, secret, or model weight is published in this documentation.
