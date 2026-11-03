# Batch 69 — Prospective synthetic point-occlusion stress test

## Aim and validity boundaries

Batch 59 remains the **priority** new, independent, action-class-disjoint
**natural-video** evaluation (36 DEV + 6 TEST). It is running from an immutable
Batch 59 commit. Batch 69 is separate, CPU-only, and intentionally does not
modify the active Batch 59 runner, .automation, cron, GPU usage or media.

The previous Batch 62–68 analyses repeatedly found an exceptionally persistent
117-frame false accept on the *already opened* Batch 56 DEV. Rather than
retuning to that video or touching Batch 59's sealed TEST, Batch 69 creates
brand-new **procedural synthetic grayscale videos**, with known per-frame 2D
point trajectories, 2D background pan, and pixel-point occlusion. They are
**not** independently sourced natural footage and have **no 3D camera pose**.
The 2D pan does not imply physical scene-relative motion with parallax.

## Pre-registered design (must be committed before preparing or scoring)

- **Six regimes**, each with **four DEV scenes and three TEST scenes**:
  clean, camera pan, illumination variation, static crossing occluder,
  occluder deliberately moving in synchrony with target, and pan plus the
  synchronized occluder. Total: **42 disjoint seeded 40-frame scenes**,
  24 DEV + 18 TEST.
- Explicit split-unique seeds are in the committed `specifications()`.
  All procedures and parameters are frozen before any scored output is read.
  No experiments on old Batch 56 outcomes select thresholds or families here.
- **Five unchanged methods:** KLT forward/backward, template recovery,
  affine-stabilized KLT, Batch 56 fixed `appearance_fb_055`,
  and a fixed 4-pixel template agreement veto on that same baseline.
- **No selection or optimization after DEV:** The candidate set, exact
  thresholds and reports are identical on both splits. There is no adaptive
  choice based on TEST, and this experiment does not propose a production gate.
- Unit of independence is a **generated scene**, not an individual frame.
  Point visibility is whether the occluding layer covers the ground-truth
  query **point center**, not whether any part of the object remains visible.
- Report raw visible coverage, visible PCK@8 over the full visible
  denominator, precision among accepted visible points, occluded false
  accept rate (accepted prediction at an occluded point), maximum contiguous
  false-accepted occlusion frames, and the observed per-scene CPU tracking
  execution time excluding decoding/hashing/IO and any GPU/model inference.
- Report the known **synthetic 2D background** displacement separately from
  the point's image-plane displacement. Ground truth is used only to score
  after predictions have been pinned; never to compensate tracker output.

## Anti-leakage and immutable experiment execution

1. Commit code, tests and this protocol. Obtain the exact GitHub main SHA.
2. Prepare public `.npy` frames, public query/split manifest and privately
   pinned ground-truth traces in the ignored shared local `data/` directory.
3. `predict-dev` loads public frames and public queries, *never* opens a
   private oracle. Persist and SHA-pin the prediction JSON without overwrite.
4. `score-dev` authenticates the DEV predictions, then reads only private
   DEV traces. Persist and SHA-pin.
5. `predict-test` requires the pinned DEV score, reads only public TEST
   frames, and commits unchanged methods/predictions before scoring.
6. `score-test` reads TEST oracle **once** after the pinned predictions
   exist. Final audit validates every public frame hash and every output
   SHA-256, source SHA and prediction/result linkage.
7. All outputs/media/oracles remain on Tang and are **not** Git-tracked.
   Retain evidence and failure logs; never repair a frozen run in place.

## Falsification hypothesis and interpretation guard

The synchronized occluder is an intentionally adversarial 2D construction.
A tracker can follow the occluder's pixels for many frames even while its
internal motion or template consistency appears credible. The protocol asks
whether fixed methods abstain rather than simply producing smoother wrong
tracks. A successful result on clean camera pan is not evidence of physical
camera pose estimation. Any apparent benefit or failure is limited to the
synthetic generator distribution; neither score justifies deployment without
Batch 59's held-out natural-video evidence.

## Source and reproducibility

- `experiments/batch69_prospective_occlusion.py`
- `tests/test_batch69_prospective_occlusion.py`
- `configs/kinetics_batch56.json` is read unchanged for the frozen tracker.
- Frozen output root: `outputs/batch69_prospective_occlusion_v1/`
- New data root: `../data/batch69_prospective_occlusion_v1/`.
- Command order:
  `python experiments/batch69_prospective_occlusion.py prepare`
  then `predict-dev`, `score-dev`, `predict-test`, `score-test`, `audit`.
- Results are to be published **after** the single scored TEST; do not write
  expected held-out values into this preregistration.
