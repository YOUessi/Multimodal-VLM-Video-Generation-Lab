# Batch42 — TAP-Vid-Kinetics natural-camera tracking and visibility results

- Frozen algorithm/source SHA: `5c862feeedf37a1035849f1dd815b55d91e64a68`
- Dataset: 12 natural Kinetics/YouTube clips from 12 action classes; 4 DEV / 8 TEST; 4 first-visible point queries per clip (16 DEV / 32 TEST queries).
- TEST support: 4,229 visible point-frame positions and 2,495 invisible positions.
- Frozen execution: **20/20 stages complete**, every return code 0; **299 tests passed** (3 pre-existing Pillow deprecation warnings).
- Publication guard: 213 tracked files scanned; 0 forbidden media, weights, private labels or secrets.
- The source clips, extracted frames, private annotation oracle, prediction JSON and model weights remain local on Tang.

## 1. Protocol and leakage controls

The task was to track a first-visible point through a natural video and jointly predict its location and visibility. The CPU methods saw every intermediate 25-fps frame. The SmolVLM2-500M baseline saw only the marked query image and one later image, so its temporal evidence budget is different and is reported separately.

The 4-scene DEV split alone selected the CPU method. The predeclared constraints were visible coverage >= 0.40 and invisible false-accept rate <= 0.25; among eligible candidates, scene-macro TAP-Vid Jaccard was maximized. `template_recover4` was selected. All TEST CPU predictions were generated and SHA-pinned before the TEST oracle was opened for the single scoring pass. The VLM DEV gate was evaluated before any TEST VLM inference; it blocked, so no TEST VLM prediction or TEST VLM label was produced. No TEST answer was used to tune thresholds or select methods.

## 2. TEST scene-macro results

Each scene receives equal weight. PCK@8 is conditional on ground-truth visibility; visible coverage measures the fraction of visible positions accepted by the tracker; accepted precision is the fraction of accepted visible predictions within 8 px; invisible false accept is the fraction of invisible positions incorrectly accepted.

| Method | PCK@8 visible | Visible coverage | Accepted precision | Invisible false accept | Jaccard mean (r=1,2,4,8,16) | CPU P50/P95 ms/frame |
|---|---:|---:|---:|---:|---:|---:|
| stationary point | 58.2% | 100.0% | 58.2% | 100.0% | 0.267 | 0.0002 / 0.0007 |
| KLT forward | 55.3% | 75.1% | 70.8% | 40.9% | 0.248 | 0.0666 / 0.0939 |
| KLT forward-backward | 54.1% | 59.1% | 89.5% | 13.0% | 0.336 | 0.1007 / 0.1655 |
| template recovery (4 px gate) | 37.9% | 59.4% | 60.3% | 24.2% | 0.216 | 0.0394 / 0.0653 |
| ORB/RANSAC affine + KLT | 51.0% | 59.7% | 85.1% | 15.6% | 0.325 | 0.2338 / 0.3014 |
| appearance + FB (0.55) | 51.2% | 53.2% | 94.5% | 9.8% | 0.336 | 0.1090 / 0.1785 |
| appearance + FB (0.70) | 46.7% | 47.7% | 96.8% | 6.4% | **0.345** | 0.1085 / 0.1777 |
| appearance + FB (0.85) | 35.7% | 36.1% | 98.2% | 4.4% | 0.295 | 0.1083 / 0.1775 |

### Micro counts across all TEST point-frame positions

| Method | PCK@8 visible | Visible coverage | Accepted precision | Invisible false accept | Correct accepted visible / accepted visible | Visible abstentions | Invisible false accepts / invisible |
|---|---:|---:|---:|---:|---:|---:|---:|
| stationary point | 59.16% | 100.00% | 59.16% | 100.00% | 2,502 / 4,229 | 0 | 2,495 / 2,495 |
| KLT forward | 39.82% | 63.80% | 62.42% | 36.23% | 1,684 / 2,698 | 1,531 | 904 / 2,495 |
| KLT forward-backward | 38.50% | 44.93% | 85.68% | 17.80% | 1,628 / 1,900 | 2,329 | 444 / 2,495 |
| template recovery | 24.54% | 48.76% | 50.34% | 19.80% | 1,038 / 2,062 | 2,167 | 494 / 2,495 |
| ORB/RANSAC affine + KLT | 37.05% | 45.57% | 81.32% | 19.40% | 1,567 / 1,927 | 2,302 | 484 / 2,495 |
| appearance + FB (0.55) | 35.92% | 37.83% | 94.94% | 13.79% | 1,519 / 1,600 | 2,629 | 344 / 2,495 |
| appearance + FB (0.70) | 32.89% | 33.70% | 97.61% | 8.70% | 1,391 / 1,425 | 2,804 | 217 / 2,495 |
| appearance + FB (0.85) | 22.32% | 22.61% | 98.74% | 5.93% | 944 / 956 | 3,273 | 148 / 2,495 |

The stationary baseline's 59.16% micro PCK@8 is not tracking success: it falsely accepts all 2,495 invisible positions. Reporting PCK without coverage and visibility errors would be materially misleading.

## 3. DEV selection did not generalize

`template_recover4` was selected from DEV because it met the predeclared coverage / false-accept constraints and had the best DEV scene-macro Jaccard among eligible candidates. On TEST it reached only 0.216 Jaccard and 0.379 visible PCK@8, while KLT-FB reached 0.336 Jaccard and 0.541 visible PCK@8 at nearly identical visible coverage (0.594 vs 0.591). This is a clear small-DEV selection instability, not a reason to retune on TEST.

The 8-scene paired bootstrap was descriptive, not a population guarantee:

| Paired scene-macro difference | Mean | 95% scene-bootstrap interval |
|---|---:|---:|
| KLT-FB minus KLT-forward Jaccard | +0.0886 | [+0.0141, +0.2197] |
| appearance-FB 0.70 minus KLT-FB Jaccard | +0.0086 | [-0.0241, +0.0535] |
| affine-stabilized KLT minus KLT-FB Jaccard | -0.0112 | [-0.0337, +0.0042] |

The appearance 0.70 row is a post-score diagnostic only; it was not selected from TEST and no threshold was changed after seeing it.

## 4. Camera-motion and appearance diagnostics

ORB/RANSAC background-affine registration was accepted for 83.1%–100% of frame pairs, but the mean background-relative residual ranged from 4.7 px to 50.7 px across TEST scenes. Several scenes therefore passed the registration acceptance check while retaining large residuals. Affine-stabilized KLT did not improve the paired scene-macro Jaccard over KLT-FB (mean difference -0.0112; interval includes zero) and increased per-frame P50 from 0.101 ms to 0.234 ms. The separate affine registration step cost 572–1,414 ms per video.

The darkest / lowest-texture TEST scene had mean luma 18.2, Laplacian variance 157.1, and 275 ORB keypoints; affine-pair acceptance fell to 83.1%. Two high-motion scenes had mean screen displacement 119.9 px and background-relative residuals above 24 px. These are useful strata, but no calibrated camera pose, depth-aware parallax or 3-D world motion is available in TAP-Vid-Kinetics. The residual is only a 2-D dominant-background affine proxy and must not be called physical camera compensation.

## 5. SmolVLM2-500M visibility baseline — blocked on DEV

The model returned a valid `VISIBLE` label for all 32 DEV queries. It correctly labeled all 22 visible positions but falsely accepted all 10 non-visible positions: accuracy 68.75%, balanced accuracy 0.50, coverage 100%, invalid format rate 0%, non-visible false-accept rate 100%. The predeclared DEV gate therefore blocked on balanced accuracy and non-visible false accepts. TEST VLM inference was skipped. This is now a semantic visibility failure, not a parser-format failure.

## 6. Conclusions and next bottleneck

1. KLT forward-backward materially reduces false acceptance relative to forward-only KLT, but still accepts 13.0% of invisible positions at scene-macro level.
2. Appearance gating trades coverage for precision; the strictest threshold reduces false accepts to 4.4% but covers only 36.1% of visible points.
3. The 4-scene DEV split was too small to select a stable tracker; the selected template method lost badly to KLT-FB on the independent TEST.
4. Background-affine registration can be accepted while its residual is large; it is not a reliable substitute for calibrated multi-view camera geometry.
5. SmolVLM2-500M collapses to `VISIBLE` on this task and fails the DEV admission gate.

The next batch should move to a new, sequence-disjoint multi-view set with calibrated cameras and released 3-D point tracks. It should evaluate image-plane tracks separately from camera-relative / world-frame motion, use a larger DEV split for method selection, and test a stronger 2.2B VLM with a 3-frame visibility prompt under the same DEV-only admission rule. The current TEST is closed: do not adjust thresholds, prompts, or model selection from these results.

## 7. Reproducibility and integrity

- Frozen source SHA: `5c862feeedf37a1035849f1dd815b55d91e64a68`; prediction and config hashes are recorded in local JSON artifacts.
- Frozen plan: `batch42-kinetics-real-camera-visibility-20261102-v1`; all 20 stage statuses are `done`, return codes 0, and `ALL_APPROVED_STAGES_DONE` contains the same frozen SHA.
- Full regression: 299 passed, 3 deprecation warnings; source publication guard: 213 tracked files, 0 forbidden files.
- Final data hash audit and CPU / VLM pin audits passed. No video, frame array, private annotation, weight or secret was committed.

## References and data terms

- Google DeepMind TAP-Vid documentation: https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid
- TAP-Vid dataset page: https://tapvid.github.io/
- Kinetics source clips are YouTube videos; annotation licensing does not replace each source video's individual terms. Do not redistribute source videos.
