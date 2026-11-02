# Batch 47C — DEV-only temporal spacing and KLT cycle consistency

**Date:** 2026-11-02 UTC. **Status:** Completed exploratory DEV diagnostic, not a new independent blind TEST. Parent GitHub main SHA: d5999455b307d73e4766c07369c240f9e1e19fae. Concurrent frozen Batch46E remains at df14676e1f52d9e1a0096581a8558cbbb3f8cd67 and was not modified by this experiment.

## Research problem

Batch47B sampled 11 frames over ~10s of video; adjacent sampled frames were roughly 0.8–1.2 seconds apart. Large spatially held-out affine residual could arise from **long-baseline optical-flow correspondence failures**, rather than a genuinely different global motion layer. Batch47C isolates this potential confound using equal source anchors at several shorter temporal spacings on the same six preselected DEV videos. No TEST media or private oracle are opened.

## Predeclared DEV protocol

- Hard-pin the existing class-disjoint Batch46 DEV/TEST preflight selection SHA-256: **0d49c3096e8d302596c6044eefd313508d072e113d880b50ec74cc237f84384c**. Process only six DEV rows, abort on missing DEV shards or selection drift.
- In each DEV clip, choose **eight identical anchor frames** shared across four source-frame strides of **1, 3, 10, 30**. Read actual source FPS: 25 or ~30 frames/s. All images are resized to width 512 px.
- Detect up to 500 Shi–Tomasi corners at each anchor, perform forward and backward pyramidal KLT with the same parameters, and reject cycles with >1.5 resized px round-trip error, nonfinite values, invalid KLT status or out-of-bounds endpoints. The 1.5 px cutoff is an instrumentation choice, not a production tracker threshold.
- On all forward-KLT points and the cycle-checked subset independently, fit four leave-one-quadrant-out affine RANSAC models using the existing Batch47B diagnostic. Track successful fold coverage and held-out errors separately. Report **paired-only** raw-vs-cycle medians on identical anchor pairs, avoiding a purely conditional comparison across unequal sets.
- Measurements are 2D image-plane diagnostics only. They do **not** identify true background, physical camera pose, 3D world-relative movement, target identity, ground-truth occlusion, or production tracker quality.

New implementation: experiments/dev_temporal_affine_stability.py. New regression tests: tests/test_dev_temporal_affine_stability.py. Full pair-level report was written locally to outputs/research/batch47c_dev_temporal_stability_v1.json with SHA-256 **a94d9bb18894efc34683574f09493fcea6bfe78556a4b94b73d45a54d37eabc7**. No media, weight files, private labels, or secrets are committed.

## Actual six-clip DEV results: 192 frame pairs

Exactly six clips × four strides × eight anchors = **192** pairs, **48** per stride. Numbers pool correlated observations within six videos; they are descriptive, not independent-sample statistical significance claims.

| Stride | Raw spatial-CV valid /48 | Cycle spatial-CV valid /48 | Jointly valid | Raw heldout median on jointly valid pairs | Cycle heldout median on jointly valid pairs | Median cycle retention | Median KLT forward/back CPU ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 frame | 40 | 39 | 39 | 0.423 px | 0.401 px | 98.7% | 1.839 |
| 3 frames | 40 | 38 | 38 | 1.450 px | 1.269 px | 92.2% | 1.605 |
| 10 frames | 37 | 31 | 31 | 4.891 px | 1.834 px | 74.1% | 1.742 |
| 30 frames | 36 | 24 | 24 | 11.161 px | 2.362 px | 47.1% | 1.848 |

Retention is computed per pair as cycle-valid / max(1, forward-valid). CPU time excludes video decoding, spatial CV, report writing and full tracker inference. The forward-backward pass roughly doubles LK direction work relative to forward-only matching.

Per-scene status at **30 source-frame stride**:

| DEV scene | Raw valid /8 | Cycle valid /8 | Raw conditional median (px) | Cycle conditional median (px) |
| --- | ---: | ---: | ---: | ---: |
| playing controller | 8 | 8 | 1.845 | 1.711 |
| skydiving | 4 | **0** | 47.405 | abstain |
| throwing axe | 8 | 7 | 11.969 | 0.097 |
| doing aerobics | 8 | 6 | 26.669 | 18.955 |
| gospel singing in church | 8 | 3 | 48.817 | 19.249 |
| making balloon shapes | **0** | **0** | abstain | abstain |

**Research finding:** The temporal baseline strongly changes the spatial-affine reliability diagnostic. A large residual at sparse time sampling cannot by itself establish that global camera compensation fails: long-baseline KLT errors are an important competing explanation.

**Negative result:** Cycle filtering reduces the held-out error conditional on sufficient evidence, but coverage drops markedly. At 30 frames, usable pairs fall from **36/48 to 24/48**. Skydiving and making balloon shapes show fail-closed abstention, and doing aerobics remains difficult. Do **not** interpret the smaller conditional residual as improved true tracking correctness or occlusion rejection. Repeated textures and cross-layer correspondences can still pass a cycle check.

## Validation and reproducibility

Targeted new tests: **15/15 passed**. Full project regression: **513/513 passed**, three existing Pillow deprecation warnings. Tests cover exact image translation, an occluded/replaced patch, low-texture abstention, malformed images, same temporal anchors across strides, selection hash mismatch before media access, six DEV-only rows, absent DEV data, immutable report, and invalid sampling protocol.

Reproduce only on authorized Tang, from a checkout containing Batch47C, writing to a new report location:

    ../.venv/bin/python -m pytest -q tests/test_dev_temporal_affine_stability.py
    ../.venv/bin/python -m pytest -q tests
    nice -n 10 ../.venv/bin/python experiments/dev_temporal_affine_stability.py \
      --selection ../data/tapvid_kinetics_batch46/preflight/selection.json \
      --shards ../data/tapvid_kinetics_batch46/shards \
      --output outputs/research/batch47c_dev_temporal_REPLICA.json --width 512

Reports are write-once. Repeated elapsed CPU times vary, so compare semantic results instead of report bytes. Since the same six videos have already been inspected in Batch47B, these results are exploratory DEV analysis, not new blind validation.

## Follow-up research gate

Test a new short-step temporal-bridging KLT candidate with explicit forward-backward and spatial-support abstention, using a **new development split or independently generated controlled evaluation** for algorithm selection. Freeze that choice and the new TEST selection before scoring once. Existing Batch46E sealed TEST answers must never be used for model selection, threshold adjustment, or deciding which variant to promote. No active batch plan, cron, or frozen source was altered by Batch47C.
