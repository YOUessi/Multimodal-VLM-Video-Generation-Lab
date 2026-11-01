# Batch 39C — TAPVid-MV DROID multi-view motion decomposition

- Frozen source commit: `4adf6eb72ede652a7b0ea01d690115939607e7b0`
- DEV scenes / CPU queries: 4 / 24
- TEST scenes / CPU queries: 8 / 48
- DEV VLM queries: 12; TEST VLM queries: 24
- DEV format gate: passed (12/12 valid)
- Manifest SHA-256: `6af9a247502fec577022ac53262f7fec29140bca0db032fbbf6281fc573b45dc`
- Manifest errata SHA-256: `23202dfab3aae478fbd2568a6f7e8bc182c1302b47c5a203717d9f85435c3035`
- SmolVLM2-500M weights SHA-256: `b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3`

## CPU point-tracking results

| Split | Method | PCK@8 all in-frame | PCK@8 depth-visible | Visible coverage | Occluded false accept | Out-of-frame false accept | CPU P50/P95 ms per target frame |
|---|---|---:|---:|---:|---:|---:|---:|
| DEV | klt_fb | 45.62% | 46.52% | 50.75% | 0.00% | 19.37% | 0.077/0.309 |
| DEV | klt_forward | 48.07% | 46.77% | 53.48% | 0.00% | 25.93% | 0.072/0.150 |
| DEV | stationary_point | 86.40% | 87.56% | 100.00% | 100.00% | 100.00% | 0.000/0.000 |
| DEV | template_recover4 | 37.91% | 34.58% | 39.80% | 0.00% | 29.66% | 0.001/0.099 |
| TEST | klt_fb | 78.27% | 80.96% | 85.60% | 77.78% | 3.83% | 0.304/0.321 |
| TEST | klt_forward | 78.45% | 80.96% | 87.25% | 81.48% | 3.83% | 0.135/0.154 |
| TEST | stationary_point | 89.71% | 90.89% | 100.00% | 100.00% | 100.00% | 0.000/0.000 |
| TEST | template_recover4 | 65.36% | 67.72% | 70.86% | 48.15% | 40.77% | 0.096/0.107 |

## SmolVLM2-500M endpoint results

| Split | PCK@8 depth-visible | Visible coverage | Visible precision@8 | Occluded false accept | Out false accept | Format invalid |
|---|---:|---:|---:|---:|---:|---:|
| DEV | 0.00% | 100.00% | 0.00% | n/a | 100.00% | 0/12 |
| TEST | 0.00% | 100.00% | 0.00% | 100.00% | 100.00% | 0/24 |

## Camera-motion decomposition

| Split | Screen vs world-only cosine | Affine-compensated vs world-only cosine | Oracle camera-subtracted vs world-only cosine | Camera-motion dominance | ORB affine reprojection error P50/P90 px | Affine coverage |
|---|---:|---:|---:|---:|---:|---:|
| DEV | 0.7949 | 0.6607 | 0.994 | 0.4882 | 0.0585/0.224 | 93.99% |
| TEST | 0.7298 | 0.7828 | 1.0 | 0.5138 | 0.091/0.8817 | 99.56% |

## Interpretation limits

- The Rerun recordings display a subsampled track set; these are not the full metadata-declared tracks.
- Per-view visibility is inferred from exact released depth-cloud snapshots, not a separately published binary visibility label.
- PCK on all in-frame projections includes positions whose visibility is unknown and is not a visible-only benchmark score.
- ORB affine is a background camera-motion diagnostic, not a physical point-identity tracker.
- Camera-only and world-only projections are oracle diagnostics and are never inputs to predictors.
- The VLM endpoint baseline has a different temporal evidence budget and is reported separately.
- Only DEV may be used for method selection; TEST outcomes are descriptive and must not tune thresholds.
- SmolVLM2-500M receives only the query and endpoint RGB images; CPU methods receive every intermediate frame.
- The endpoint is a fixed cloud-snapshot frame chosen from metadata, not selected from TEST outcomes.
- Depth-visible and depth-occluded are derived from the exact released point-cloud snapshot; unknown/missing are not scored as absent.
- The tolerant coordinate parser is fixed before TEST; no TEST reply was used to change it.
- This is a study of the public Rerun viewer subset, not the full TAPVid-MV track arrays.
- Visibility is inferred from exact released depth point-cloud snapshots; it is not an official per-frame per-view binary occlusion label.
- The DEV gate evaluates only parser/interface validity. No TEST metric is used for threshold, prompt, or model selection.
- ORB partial-affine is an image-background diagnostic; it is not a calibrated physical camera-motion estimator and is not a point-identity tracker.
## Research interpretation and next bottleneck

### What the independent TEST actually says

- Forward–Backward KLT reached **80.96% PCK@8 on depth-visible TEST positions**, with 85.60% coverage and 94.58% precision among accepted visible predictions. Its CPU median was 0.304 ms per target frame.
- The same method falsely accepted **42/54 depth-occluded positions (77.78%)**. Forward KLT was slightly worse at 44/54 (81.48%). The FB check therefore improves visible precision only modestly and is not an occlusion detector.
- The stationary-coordinate baseline scored 90.89% visible PCK, but falsely accepted **all 54/54 occluded positions and all 601/601 out-of-frame positions**. Its high PCK is a static-point prior, not tracking success.
- Template recovery reduced occluded false accepts to 26/54 (48.15%), but its visible PCK fell to 67.72%, visible coverage to 70.86%, and out-of-frame false accepts remained 40.77%. It is not a satisfactory safety gate.
- The 4-scene DEV and 8-scene TEST splits have very different visible PCK/coverage. The scene-cluster intervals are wide; do not interpret the higher TEST point score as evidence of a stable domain improvement.

### Camera motion is material, but affine compensation is not a physical solution

The released camera/point metadata supports a diagnostic decomposition. In TEST, the screen-vs-world-only mean direction cosine was 0.7298; subtracting the calibrated camera contribution from the oracle projection yielded 1.0, which checks the decomposition. ORB partial-affine compensation raised the mean cosine to 0.7828, but DEV moved in the opposite direction (0.7949 to 0.6607). Its TEST static-point reprojection error was 0.091 px median / 0.882 px P90, but the mean was 7.97 px, indicating a small number of severe failures. This is an image-plane background diagnostic, not calibrated 6-DoF pose or depth-aware parallax compensation.

### VLM endpoint coordinate baseline: hard negative

SmolVLM2-500M returned the bottom-right boundary coordinate (447, 251) for all 36 DEV+TEST queries. The parser accepted 36/36, but visible PCK@8 was 0/9 on DEV and 0/17 on TEST; it falsely accepted all 3/3 DEV out-of-frame and all 2/2 TEST occluded plus 3/3 out-of-frame endpoints. Format validity did not imply visual grounding. The coordinate-output interface is not suitable for this task in its current form.

### Next research question

The next batch should focus on reliable visibility/abstention, not another small improvement to point-coordinate PCK. On a new scene-disjoint DAVIS video draw, compare multi-feature motion consensus with a per-feature photometric/appearance consistency gate, and evaluate a separate VLM VISIBLE / ABSENT / UNCERTAIN endpoint classifier. Select appearance thresholds using DEV only; seal all TEST predictions before reading future masks. Preserve the distinction between DAVIS mask-centroid tracking (an image-plane proxy) and physical material-point/world motion.

## Execution and integrity

- Frozen source SHA: 4adf6eb72ede652a7b0ea01d690115939607e7b0.
- 15/15 frozen automation stages completed with return code 0.
- Full regression suite: **240 passed**, with 3 existing Pillow deprecation warnings.
- Source/media/weights/secrets publication guard passed. The RRD videos, extracted RGB arrays, private point tracks/depth clouds, model weights and raw prediction JSON remain on Tang.
- The original Batch39B manifest was not rewritten. manifest-errata.json SHA-binds the corrected two-query-per-camera description to the original manifest and public query file.
