# Batch 47 research preparation — DEV-only evidence for affine stabilization

Date: 2026-11-02 (UTC). This is a **diagnostic preparation** during Batch 46E's ongoing frozen R00 download, **not a new frozen batch, not a tracker change, and not a TEST score**.

## Strict input / reproducibility contract

- The upstream active Batch46E frozen runtime stays at `df14676e1f52d9e1a0096581a8558cbbb3f8cd67`, entirely untouched.
- Only six public **DEV** video members were opened from their fully downloaded Kinetics shard archives. No TEST video, TEST prediction, TEST oracle, or private ground truth was read.
- Input selection metadata SHA-256: `0d49c3096e8d302596c6044eefd313508d072e113d880b50ec74cc237f84384c`.
- The six source shard SHA-256 values, current analysis code SHA-256 and detailed per-pair measurements are recorded in local ignored output `outputs/research/batch47_dev_affine_motion_evidence_v2.json`, and **are not published as media**.
- Method: resize to width 512 px; sample 11 uniformly spaced frames/video; for each of ten consecutive sampled-frame pairs run Shi–Tomasi + KLT correspondences, robust 2D full-affine RANSAC, and report image-plane displacements / fit support. Measurements are normalized to this sampled-frame interval, **not** to every original video frame.
- This differs from the Batch45E tracker evaluation pipeline and **does not replace its performance or CPU-latency measurement**.

## Observation: median over ten sampled DEV frame pairs

| DEV action class | RANSAC inliers | Spatial support (occupied 4×4 cells) | Screen displacement (512 px width) | All-match residual after affine | Mean brightness change |
|---|---:|---:|---:|---:|---:|
| playing controller | 61.8% | 78.1% | 20.94 px | 1.27 px | 2.17 |
| skydiving | 13.3% | 18.8% | 71.11 px | 27.23 px | 5.61 |
| throwing axe | 65.7% | 78.1% | 0.10 px | 0.11 px | 1.15 |
| doing aerobics | 12.6% | 21.9% | 42.52 px | 23.92 px | 6.60 |
| gospel singing in church | 13.6% | 34.4% | 78.11 px | 83.45 px | 2.05 |
| making balloon shapes | 44.1% | 28.1% | 13.37 px | 2.87 px | 0.80 |

Brightness is the absolute difference in pairwise mean 8-bit grayscale values. These are image-wide unsupervised video statistics, not annotated target visibility or movement accuracy. Sampling, camera cuts, parallax, scene motion and bad optical flow can all alter these statistics.

**Key negative finding:** A small median residual **among RANSAC inliers** is not evidence of correct global camera compensation. For instance, `gospel singing in church` has an inlier residual below one pixel but only 13.6% correspondence support, with a median 83.45 px all-match residual. A coherent moving foreground or locally concentrated correspondences can produce a convincing yet misleading affine estimate. Grid-cell spatial support measures the spatial *distribution* of inliers without labeling background regions.

## Screen vs affine-relative motion

For matched pixels `p_t` and `p_(t+1)`, screen displacement is
`d_screen = p_(t+1) - p_t`. Given a robustly estimated 2D affine map `A`,
the residual is `d_relative = p_(t+1) - A(p_t)`.

**Neither** `A` nor `d_relative` measures calibrated 3D camera translation, object velocity in the world, or physical motion. Dominant layers need not be background, and a single affine model cannot explain depth-dependent parallax or cuts.

The DEV-only evidence motivates a **future preregistered candidate**: abstain from applying a global affine correction when correspondence support is too low or too spatially concentrated. Candidate acceptance thresholds must be selected and frozen **using DEV only**, *before* opening the next independent TEST set. The running Batch46E test and model parameters must not be changed based on this diagnostic.

## Execution and tests

```bash
cd /home/you/projects/Multimodal-VLM-Video-Generation-Lab/repo-batch47-dev
../.venv/bin/python -m pytest -q tests/test_dev_affine_evidence_audit.py
nice -n 10 ../.venv/bin/python experiments/dev_affine_evidence_audit.py \
  --selection ../repo-batch46e-tested/data/tapvid_kinetics_batch46/preflight/selection.json \
  --shards ../repo-batch46e-tested/data/tapvid_kinetics_batch46/shards \
  --output outputs/research/batch47_dev_affine_motion_evidence_v2.json \
  --sample-frames 11 --width 512
```

Eleven targeted regression tests check known image-plane affine motion, locally concentrated correspondences, corrupted/missing input rejection, finite coordinates, strict six-DEV-only selection, refusal to open TEST shards, idempotent output pinning, and synthetic moving-camera KLT registration.

## Freeze boundary and next step

The ongoing Batch46E remains locked to its original SHA, immutable plan, existing R00 process, and unique `MVVG_PROJECT4_10MIN` cron. This new research code is **not** installed in that running workspace, does not start another batch, and reads only already completed DEV shards. Once Batch46E finishes and its TEST predictions are sealed and scored, evaluate whether this DEV-motivated support/abstention diagnostic can be transformed into a *separately frozen* candidate on a new development set. Avoid using Batch46E TEST to select thresholds.

## Additional regression issue found during independent-worktree verification

The first full regression on this source-only worktree reported **485 passed / 1 failed**, despite the new 11 targeted tests passing. Root cause: importing `kinetics_batch46_data.py` monkeypatched `kinetics_batch45_data.deterministic_class_selection` and other globals in the shared Python module namespace. A genuine fresh clone (without historical private manifest files under `data/`) then caused the Batch45 unit test to fail by accidentally calling the Batch46 selector. Historical workspaces with private data had masked this isolation defect.

Fixed in the **future source worktree only** by loading a private instance of the Batch45 data module for Batch46, while preserving Batch46's explicit path and selector overrides in that private instance. Added a cross-batch module-isolation regression. **Full fresh-worktree regression now 487 passed / 0 failed; 3 existing Pillow deprecation warnings**. This repair has **not been injected into frozen Batch46E**, so its experiment results remain reproducible at the originally approved SHA.

No GPU or new TEST evaluation was started by these DEV-only checks.
