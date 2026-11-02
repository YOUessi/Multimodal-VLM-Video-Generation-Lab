# Batch 47B — spatial out-of-region affine evidence (DEV only)

**Date:** 2026-11-02 UTC. **Research status:** exploratory, **not** an update to the running frozen Batch46E, **not** a new independent TEST result. New code and tests were developed on the separate source worktree whose parent commit is `569810184829bf8a018c88665f99e70038dca930`.

## Question and controlled design

Previous DEV-only analysis showed that a tiny RANSAC inlier residual can coexist with poor global feature support. To test whether the affine estimate transfers spatially rather than merely explaining the correspondences used to fit it, we introduced **leave-one-quadrant-out cross-fitting**.

At the original query interval (11 frames uniformly sampled per video, width 512 px), detect up to 500 Shi–Tomasi corners and use pyramidal KLT to obtain 2D image-plane pairs. Divide the source image into four nonoverlapping quadrants. For each quadrant, fit an OpenCV full-affine 2D RANSAC model on matches **outside** that quadrant and measure the held-out matches' Euclidean reprojection residual. A fold is not scored when fewer than 24 training matches or 12 held-out matches exist. If fewer than two folds can be scored, return an explicit insufficient-support status.

The protocol records per-fold held-out median and P90 error, the proportion of held-out points within 3 resized pixels, spatial fold coverage, and elapsed CPU time. **The 3 px tolerance and coverage minimums are descriptive predeclared instrumentation constants; they do not select a tracker or authorize any test inference.** Report fields never identify motion as a physical camera pose or true world-space object velocity.

## Data isolation

Only six preselected **DEV** video members were read from existing complete Kinetics shards (playing controller; skydiving; throwing axe; doing aerobics; gospel singing in church; making balloon shapes). The DEV selection metadata SHA-256 is `0d49c3096e8d302596c6044eefd313508d072e113d880b50ec74cc237f84384c`, the same frozen selection used by Batch46E. Code enforces six DEV rows, refuses missing DEV shards before processing, never accesses TEST archives, and never opens the private oracle. Raw video, frames, label tracks, large pair-level diagnostics and model weights are not published.

Code: `experiments/dev_spatial_affine_crosscheck.py`; regression: `tests/test_dev_spatial_affine_crosscheck.py`. Detailed local-only output: `outputs/research/batch47_dev_spatial_crossfit_v1.json`; output SHA-256 `65220feb6c26064466245eb7278d7d919bfa4e641eaaea3448782fe574b15647`. Report is write-once (re-running into an existing different report refuses to overwrite it); elapsed CPU numbers are inherently environment-dependent.

## Observations, median across the valid consecutive sampled-frame pairs

| DEV video category | Valid cross-fitted pairs / 10 | Inlier-only median residual in prior diagnostic (px) | Cross-fitted held-out median residual (px) | Worst-quadrant median residual (px) | Median KLT CPU (ms/pair) | Median four-fold affine CPU (ms/pair) |
|---|---:|---:|---:|---:|---:|---:|
| playing controller | 10 | 0.71 | 2.34 | 7.20 | 1.87 | 0.68 |
| skydiving | 7 | 0.72 | 46.77 | 63.33 | 1.28 | 0.58 |
| throwing axe | 10 | 0.06 | 18.32 | 50.33 | 1.98 | 0.72 |
| doing aerobics | 10 | 0.72 | 34.35 | 50.68 | 1.62 | 1.66 |
| gospel singing in church | 10 | 0.59 | 75.18 | 145.04 | 1.35 | 1.42 |
| making balloon shapes | 2 | 0.62 | 5.21 | 7.10 | 1.14 | 0.18 |

Across 60 examined DEV pairs, **49/60** produced at least two valid spatial folds. The other 11 had insufficient out-of-region evidence, mostly in `making balloon shapes` (only 2/10 valid).

**Negative result:** In several natural scenes, excellent *in-sample* affine fit error fails to transfer to another region. For example, `throwing axe` had an inlier-only median residual of about 0.06 px, whereas its spatially held-out median was about 18.32 px. This does not establish a camera-motion error rate or explain which moving layer is background: foreground motion, parallax, scene cuts, temporal undersampling, tracking failures, and low texture can all contribute. Inlier-only and spatial-held-out residual distributions also use different subsets, so their numeric ratio is **not** a tracker-performance gain or loss.

The CPU measurements above are local DEV diagnostic microbenchmarks with concurrent unrelated processes. They are **not** end-to-end tracker inference latency and cannot replace Batch46E's sealed CPU latency report.

## Regression and falsifiability

- The pure spatial cross-fit passes a known 2D translation control with all four held-out quadrants under 0.05 px residual.
- Introducing a spatially concentrated independently moving foreground breaks held-out transfer (worst held-out median >15 px) even when the background shift remains known.
- A single-quadrant correspondence cluster returns `insufficient_valid_spatial_folds` rather than falsely asserting global support.
- Shape errors, nonfinite coordinates, out-of-image points, and missing DEV shards fail closed.
- A TEST-only video entry and archive are never opened. A write-once report cannot be silently changed.
- Synthetic image KLT translation is recovered without treating fitted translation as 3D camera motion.

**Targeted new tests: 11/11 pass. Full source worktree regression: 498/498 pass**, with three preexisting Pillow deprecation warnings. Running Batch46E frozen `df14676e1f52d9e1a0096581a8558cbbb3f8cd67` remains unchanged; no new TEST metrics were produced in this study.

## Research decision and next experiment

This is **evidence for investigating** an explicit low-global-support abstention mechanism, **not evidence that the mechanism improves tracking**. The next prospective candidate must be specified and selected on a new DEV set or controlled synthetic study and then frozen *before* opening another independent TEST. The current Batch46E TEST must not be used to adjust thresholds, features, or model selection.

## Reproduce on authorized Tang (DEV only)

```bash
cd /home/you/projects/Multimodal-VLM-Video-Generation-Lab/repo-batch47-dev
../.venv/bin/python -m pytest -q tests/test_dev_spatial_affine_crosscheck.py
../.venv/bin/python -m pytest -q tests
nice -n 10 ../.venv/bin/python experiments/dev_spatial_affine_crosscheck.py \
  --selection ../data/tapvid_kinetics_batch46/preflight/selection.json \
  --shards ../data/tapvid_kinetics_batch46/shards \
  --output outputs/research/batch47_dev_spatial_crossfit_replica.json \
  --sample-frames 11 --width 512
```
