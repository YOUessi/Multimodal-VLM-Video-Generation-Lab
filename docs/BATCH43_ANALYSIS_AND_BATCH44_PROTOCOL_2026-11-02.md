# Batch43 findings and Batch44 protocol — 2026-11-02

## Batch43 sealed findings

Source commit: `41049d9edbc79f6c4b35ca48f743af7f77d5c057`. The frozen Batch43 run used 12 TAPVid-MV Rerun viewer sequences (6 DEV / 6 TEST), 108 query instances and sequence/collection-group-disjoint splits.

### What the measurements say

- TEST KLT-FB: PCK@8 depth-visible **63.6%**, visible coverage **72.1%**, occluded false accepts **74.7%**, out-of-frame false accepts **1.7%**, scene-macro Jaccard **35.6%**, CPU P50/P95 **0.296/0.341 ms per target frame**.
- TEST KLT-Forward: PCK@8 depth-visible **66.5%**, coverage **84.2%**, occluded false accepts **80.0%**, Jaccard **34.4%**, CPU P50/P95 **0.153/0.171 ms**.
- Appearance gating trades coverage for safer abstention: NCC 0.55 gives **33.9%** coverage and **12.0%** occluded false accepts; NCC 0.70 gives **24.5%** coverage and **7.1%** false accepts; NCC 0.85 gives **12.4%** coverage and **3.6%** false accepts. None is a high-coverage solution.
- ORB affine stabilization did not solve camera motion: TEST PCK@8 depth-visible was **46.9%**, coverage **55.2%**, and occluded false accepts **73.3%**. It remains an image-plane background proxy, not a calibrated pose estimate.
- Three-view reconstruction from independent trackers was sparse: KLT-FB triangulated **2/9** groups; KLT-Forward **4/9**. Forward's world error was median **1.8 cm**, P90 **6.6 cm** among those accepted groups. Stationary coordinates triangulated 9/9 but had median world error **12.3 cm**, showing that coverage alone is not enough.
- SmolVLM2-500M predicted VISIBLE for every DEV query: balanced accuracy **50.0%**, non-visible false-accept rate **100%**, not-visible recall **0%**. The frozen DEV gate correctly blocked TEST VLM inference and scoring.

### Research bottleneck

The dominant failure is not raw 2-D point displacement alone. Independent per-camera trackers drift differently, and there is no joint mechanism to test whether three predicted pixels can belong to one calibrated 3-D point. At the same time, appearance thresholds reduce false accepts only by abandoning most visible points. The VLM baseline is currently overconfident on non-visible points.

## Batch44 — fresh collection groups and calibrated three-view consensus

### New independent video selection

Batch44 uses all 21 previously unused RRD sequences from the public TAPVid-MV index:

- DEV: 2 EgoExo4D groups (`cmu_soccer09`, `cmu_soccer14`) plus all 9 PACE sequences from `scene06`.
- TEST: 2 different EgoExo4D groups (`cmu_soccer15`, `iiith_soccer_006`) plus all 8 PACE sequences from `scene07`.
- Total: **11 DEV / 10 TEST sequences**, 3 cameras per sequence, 3 shared point identities per scene. The PACE collection group is assigned wholly to one split; no collection group crosses the split boundary.
- These are new sequence groups relative to Batch43. They are not a different dataset family, so the result remains a fresh within-TAPVid-MV holdout rather than a broad external benchmark claim.

### Algorithm under test

For each shared point and frame, Batch44 starts from the frozen RGB-only KLT-FB and KLT-Forward predictions in all three cameras. It intersects the three calibrated world-frame camera rays, projects the estimated 3-D point back into each camera, and accepts the three-view result only when **all three** reprojections are in-frame and each predicted pixel is within a frozen maximum residual threshold (4, 8 or 12 px). Otherwise it abstains in all views for that track/frame.

The predictor may read only RGB frames, public initial 2-D queries, and a separately hashed public calibration artifact containing rotations, translations, intrinsics and image resolutions. It must not read 3-D target tracks, track IDs, depth clouds or future visibility labels. The private oracle remains isolated to scoring.

### Selection and reporting

- Choose among the predeclared single-view and three-view methods on DEV only.
- Report PCK@8, visible coverage, accepted precision, occluded/out false accepts, Jaccard at multiple radii, 3-D reconstruction coverage/error, per-source subset results, and CPU P50/P95.
- Keep all abstentions in denominators. A low false-accept method with near-zero coverage is not a success.
- Re-run SmolVLM2-500M on fresh DEV queries with the unchanged strict visibility task; skip TEST VLM inference if any predeclared gate fails.
- TEST predictions are pinned before TEST scoring. No TEST-derived threshold, prompt, parser or method changes are permitted.

## Safety

Only source code, tests, configs, plans and this process note are committed. RRDs, extracted frames, public calibration arrays, private oracles, raw predictions, model weights, generated media and secrets stay local. Batch43 files and failed logs are preserved.
