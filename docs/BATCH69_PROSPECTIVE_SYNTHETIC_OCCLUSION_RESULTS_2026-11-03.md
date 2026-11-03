# Batch 69 — Prospective synthetic point-occlusion results (2026-11-03)

## Evidence status, frozen design and purpose

**All six protocol stages completed successfully on Tang**: prepare, DEV prediction,
DEV score, TEST prediction, TEST score, and final immutability audit.
The preregistered source commit was `2f69fa80cf01ce67b13cc09e439e054b6223469a`.
It was published to GitHub `main` **before preparation and before opening
any TEST oracle**. All decisions and thresholds remained unchanged between
DEV and TEST; TEST was scored exactly once.

This study comprises **42 newly generated, disjoint procedural 2D scenes**
(24 DEV, 18 TEST), each with **40 grayscale video frames**, spanning six
specified synthetic nuisance regimes. It is **not** an independent natural-video
benchmark, a VLM benchmark, or calibrated 3D camera/object-ground-truth
evaluation. Batch 59's new natural-video protocol and GPU lock were not touched.

The *new* hypothesis concerns long, correlated occlusions and tracker
abstention. The dataset does produce persistent false tracks for basic
KLT and affine methods, but it does **not** reproduce the previously observed
117-frame natural-DEV shared-failure episode.

## Frozen scores — DEV vs untouched TEST

For each scene one point query starts at t=0. All denominators include the
39 subsequent frames, and rates pool frames across the stated scenes.
An accepted prediction at an occluded point counts as false acceptance even
when its coordinates match a plausible appearance of the occluder.
PCK@8 counts *correct visible predictions over all visible frames*.

| Method (unchanged from preregistration) | DEV visible accepted / visible | DEV hidden false accepts / hidden | DEV longest false run | TEST visible accepted / visible | TEST PCK@8 | TEST hidden false accepts / hidden | TEST longest false run |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen `appearance_fb_055` | 565/683 (82.72%) | 0/253 (0%) | 0 | 355/513 (69.20%) | 355/513 (69.20%) | 2/189 (1.06%) | 2 |
| Appearance + fixed template-consensus veto | 565/683 (82.72%) | 0/253 (0%) | 0 | 355/513 (69.20%) | 355/513 (69.20%) | 0/189 (0%) | 0 |
| KLT forward/backward | 627/683 (91.80%) | 92/253 (36.36%) | 23 | 390/513 (76.02%) | 390/513 (76.02%) | 74/189 (39.15%) | 23 |
| Template recovery | 619/683 (90.63%) | 0/253 (0%) | 0 | 469/513 (91.42%) | 469/513 (91.42%) | 0/189 (0%) | 0 |
| Affine-stabilized KLT | 627/683 (91.80%) | 161/253 (63.64%) | 23 | 478/513 (93.18%) | 476/513 (92.79%) | 120/189 (63.49%) | 23 |

**Important negative finding:** On the synthetic DEV, the frozen appearance
tracker's strict abstention was already sufficient to eliminate every hidden
false accept, leaving no DEV safety improvement for the consensus gate.
On TEST, the gate eliminated just **two** hidden false accepts and discarded
**zero** additional visible accepted predictions. This is not evidence that
the gate will behave safely in natural videos; the controlled texture and
perfectly specified synthetic point geometry make these examples much simpler
than the difficult natural DEV case.

**Positive failure-detection finding:** KLT and affine-stabilized KLT still
accepted a **23-consecutive-frame** false track under synchronized occlusion,
across both splits. Those numbers demonstrate that tracking motion smoothly
is not equivalent to establishing target visibility or identity. In contrast,
template recovery avoided hidden accepts in this generator, but this does
not overcome its much weaker natural-DEV performance from Batch 62.

## Domain-stratified TEST outcomes for frozen appearance baseline

| Synthetic regime (3 scenes each) | Accepted visible / visible | False accepted hidden / hidden | Worst contiguous false-accept |
|---|---:|---:|---:|
| Clean | 117/117 | 0/0 (no hidden denominator) | 0 |
| 2D camera pan | 42/117 | 0/0 (no hidden denominator) | 0 |
| Illumination shift | 87/117 | 0/0 (no hidden denominator) | 0 |
| Static crossing occluder | 62/66 | 2/51 | 2 |
| Correlated moving occluder | 27/48 | 0/69 | 0 |
| 2D pan + correlated occluder | 20/48 | 0/69 | 0 |

The two occluded false accepts both came from `test_03_00` and lasted
**two** consecutive frames. The fixed template-consensus veto rejected both.
The camera-pan domain was notably unstable across the seed-disjoint split:
DEV visible acceptance was **156/156** (four scenes), while TEST was only
**42/117** (three scenes). There was no re-selection or threshold change
after observing the TEST shortfall. This is an unusually large sensitivity
to procedural scene variation and argues against extrapolating average
synthetic pan performance even within this generator.

The affine/KLT 23-frame mistake is a **separate** failure from the old
117-frame natural-DEV failure. They should not be counted together or
interpreted as independent confirmation of a physical mechanism.

## Cost, reproducibility, and information controls

- Single-thread OpenCV CPU pipeline including 2D affine estimation and five
  tracking traces: DEV scene-time median **353.8045 ms**, p95 **364.1792 ms**;
  TEST median **350.0130 ms**, p95 **374.6182 ms**, over 40 frames per scene.
  These are per-scene algorithm wall-times, not measured serving latency.
  Generation, public-frame reads, hash validation, IO, VLM GPU inference,
  rendering and service overhead are excluded.
- 24 new Batch 69 synthetic/contract tests pass. Complete project regression
  on this source: **1,226 passed, 1 skipped, 3 existing Pillow warnings**.
- Frozen public manifest SHA-256:
  `1ced62f5a7526fcc23aa42dfd41926f5bf7f7a884804d6586c43090c718d86d5`.
- Frozen DEV prediction SHA-256:
  `e631da9d2d09a3b95227df0be23c9a4389fc99dfa459f25ac073dc1258610db4`.
- Frozen DEV score SHA-256:
  `e752b09a12bc7a59c1ca4693c56c7ae22dc1a4278f73b1b8f5c9c45b4b4da65d`.
- Frozen TEST prediction SHA-256:
  `b708014f1c2fd5b42c386127a3eca627efaeef5d52cf25863a06a4e8168d6113`.
- One-time TEST score SHA-256:
  `2177299bee2c586853e6af19d179a57f0fa71906e02f6733fed874c0f1b29b18`.
- Final audit: `BATCH69_FINAL_AUDIT_PASS 2f69fa80...`.
  Source tree and public frame files were checked without changing any
  frozen media or scores.

Full frozen JSON reports, private oracle and public synthetic frames are
local to Tang; only code/tests/docs are committed. The Batch 59 downloader,
the `MVVG_PROJECT4_10MIN` cron, old data, and all prior logs/media remain
unchanged. No new GPU workload or VLM inference was started.

## Scientific conclusion and deployment decision

**No deployment.** Batch 69 offers a distinct and reproducible
procedural occlusion stress test, exposing 23-frame KLT/affine persistence
and a major synthetic camera-pan generalization gap. But the texture-based
appearance tracker largely abstains during hidden intervals in this generator,
whereas the previously studied natural-video DEV contains long joint errors.
The synthetic benchmark is not hard or representative enough to explain
or resolve the real identity/visibility bottleneck.

A new natural-video method cannot be selected or validated from this
controlled synthetic result. Finish the already frozen Batch 59 independent
natural DEV/TEST assessment, then develop genuinely external visibility and
identity cues **on DEV only**, with preregistered event-level safety, visible
coverage, CPU cost and one-time TEST gates.
