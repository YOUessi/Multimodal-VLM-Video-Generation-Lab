# Batch 78 — Label-free natural DEV domain characterization (2026-11-04)

## Scope and isolation

This batch supplies a **new repeatable, non-oracle analysis** for future fresh
TAP-Vid-Kinetics evaluations. The running Batch 77 recovery worktree is
unchanged. The input in this run is the **previously studied Batch 56 natural
DEV** (36 videos, public track predictions only); it is **not a fresh blind
TEST**, not model selection and not permission to deploy a new tracker.

Code: `experiments/batch78_unlabeled_motion_audit.py`; regression tests:
`tests/test_batch78_unlabeled_motion_audit.py`.

Source baseline at development start: GitHub `main`
`f7146febabcd852d6c941007ce3bdd937f0a081a` (Batch 77).
No private DEV oracle, TEST video, TEST predictions or TEST oracle is opened.
The public manifest *contains hash references* to the private oracle, but this
batch reads only the manifest bytes, DEV frame arrays and pinned DEV predictions.
The CLI refuses `--split test`, blocks oracle input paths, verifies every
frame SHA-256, validates the existing public-prediction SHA pin, and creates
immutable pin-protected JSON output.

## Rationale and algorithm

Prior gates were being compared using raw image-plane target motion as if it
were evidence of physical identity. This audit explicitly separates:

- **Image-plane reported target displacement**: difference of publicly
  predicted accepted tracker points. These points may be incorrect.
- **Dominant 2-D image-flow translation proxy**: median forward/backward
  Lucas–Kanade tracked features between frames, after roundtrip rejection and
  inlier/4×4 spatial-cell checks. This is a statistic on image pixels, **not
  ground-truth camera motion or background motion**. If the foreground dominates
  the visible features it can dominate this estimate. It must not directly
  justify camera compensation or physical object identity.
- **Scene-relative image-flow residual proxy**: the difference between a
  reported tracker displacement and the estimated dominant image displacement,
  only when the latter passes its own support gate. It is **not true 3-D/world
  velocity**, and must not be interpreted as evidence that a tracked ID is right.
- **Texture/lighting proxies**: Laplacian texture variance, saturation
  fraction, frame-median luminance delta, and median absolute frame change.
- **Trackability proxy**: forward/backward LK retention and spatial-consensus
  support. This does **not** measure actual visibility or occlusion without
  ground-truth annotations.

All choices were fixed without consulting this run's oracle: 10-frame pair
stride, 256-pixel maximum analysis raster size, maximum 320 corners, 1.5-pixel
forward/backward tolerance, 25 minimum inlier features, 60% inlier consensus
and 6 spatial cells out of 16. Flow displacement is mapped back to **native
frame pixel units** before it is compared to public tracker coordinates.

## Actual DEV run (pinned)

Command (run on Tang in the isolated `repo-batch78-dev` clone):

```bash
R=/home/you/projects/Multimodal-VLM-Video-Generation-Lab
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 \
  "$R/.venv/bin/python" -m experiments.batch78_unlabeled_motion_audit \
  --manifest "$R/data/tapvid_kinetics_batch56/batch56_frozen/manifest.json" \
  --data-root "$R" \
  --source-predictions "$R/repo-batch56b-tested/outputs/kinetics_batch56/dev_predictions.json" \
  --output "$R/repo-batch78-dev/outputs/research/batch78_dev_unlabeled_image_flow.json" \
  --stride 10
```

Pinned result SHA-256:
`75c339c368cd109e80c71478cdf2339ab5e7a6f0c97a8ffbb747333bc1b34f07`.

| Measurement | Observed |
| --- | ---: |
| Historical natural DEV scenes | 36 |
| Time-separated image pairs sampled | 845 |
| Pairs passing 2-D flow support checks | 736 / 845 (87.10%) |
| Pairs failing support checks | 109 / 845 (12.90%) |
| Median scene support fraction | 95.74% |
| Median per-scene dominant image displacement | 2.319 native pixels per 10 frames |
| Median per-scene Laplacian texture variance | 906.84 |
| Public track query-pair comparisons with support and two accepted positions | 875 |
| Sum of per-scene feature-computation time | 2,133 ms |
| Whole audit wall time, including file reads and SHA-256 verification | 3,403 ms |
| GPU utilization required by this experiment | None |

Seven DEV scenes had <75% trusted-pair coverage: `kinetics_04`
(16/24), `kinetics_05` (15/24), `kinetics_10` (13/24),
`kinetics_24` (15/24), `kinetics_28` (14/24),
`kinetics_30` (8/24), `kinetics_33` (7/24).

Large median dominant-image displacements included `kinetics_28`
(18.98 px), `kinetics_35` (17.50 px) and `kinetics_30`
(17.06 px). **These cannot be labeled true camera motion.**
`kinetics_35` showed 36.5 gray levels of median absolute frame
change even though the median difference of global luminance medians was 0.
Hence global brightness-median change alone is an insufficient indicator
of photometric/scene change.

A baseline track displacement minus dominant scene-flow translation can be
reported on 875 sampled pairs, but its correctness is unknown. In particular,
the audit cannot identify target visibility, distinguish moving foreground from
moving camera, validate physical identity, or measure real occlusion risk.

## Important limitations and negative findings

1. The earlier natural DEV videos have already been examined in research.
   The metrics above are **exploratory diagnostics**, not independent
   generalization claims, and no thresholds were tuned from them.
2. The apparent spatial flow majority may track a large moving foreground.
   There is no calibrated camera pose, parallax estimator or 3-D depth.
3. If a scene fails its minimum support gate, `dx_px` and `dy_px` are null:
   **do not fill them with zero** and do not infer static camera motion.
4. Comparing public tracker tracks to flow is observational and *not*
   a ground-truth motion metric. No correctness/visibility label is read.
5. Per-scene CPU timings are feature-only timings for 10-frame subsampling,
   not full tracking, decoding, VLM inference or serving latency.

## Next independent stage and research gate

Batch 77 remains the **single active frozen** natural-video pipeline. Its
36 DEV / 6 TEST protocol, VLM settings, R00 retry budget and source SHA
must not be modified by Batch 78. When the new video samples and their
public frame hashes are frozen, this exact Batch 78 code can be run against
the **new DEV only** to measure label-free scene drift. Any method selection
or parameter freeze must be completed before opening the new TEST oracle.
New TEST measurements can be produced only by a separate frozen,
single-use scoring protocol, never by relaxing this CLI's TEST firewall
in place. Compare the DEV risk groups to actual ground-truth occlusion
metrics only in a separately identified, explicitly DEV-scoped analysis.

No media, annotations, model weights, secrets or experimental output JSON
are committed. Only source, tests and this report are public.
