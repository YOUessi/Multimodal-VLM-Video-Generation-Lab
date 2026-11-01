# Batch 34 — prospectively frozen 2.5D parallax, query-grounded target identity, and abstention (2026-11-01)

## Scientific question and adversarial falsification

Batch 33 demonstrated that a video-only ORB/RANSAC 2D partial-affine camera correction achieved 179/200 on a *new synthetic* single-camera-plane heldout, but its same-color/partial-occlusion slice was 34/50. It cannot infer 3D depth and it inherits a color-only target finder. That dataset is **already scored and MUST NOT be reused for threshold tuning**. This new batch asks whether adding a shape-grounded identity check and refusing weak, mixed-depth background registrations improves conservative gate precision without destroying answer coverage.

The comparator is **frozen unchanged** from the preceding batch: (1) plain HSV image-plane motion, (2) ORB+RANSAC dominant affine camera registration with HSV target, and (3) new independent query-matched color+contour-shape selector followed by the same ORB partial affine, requiring an inlier fraction at least **0.72** and abstaining on ambiguous same-color same-shape identity. There is no depth oracle and no fake camera model supplied to the tracker.

## Independent media, labels and scoring boundary

- **Procedurally synthesized 2.5D videos, NOT natural footage and NOT recovered 3D scenes.** Three depth layers (far texture, moving target/distractor, near texture) undergo distinct image-plane camera-parallax factors, so one global affine camera transform is not universally correct. Geometry is translational with depth-varying factors; it does **not** test arbitrary perspective, rolling shutter or actual camera intrinsics.
- Four balanced modes: (A) dual-depth camera pan, (B) same-color *different-shape* distractor with temporary occlusion, (C) visually indistinguishable same-color same-shape distractor with temporary occlusion, (D) low-texture scene with moving shadows and depth parallax. Five balanced target world-relative directions: stationary and the four diagonals. The direction label is world-relative target velocity, **not** raw image-plane displacement.
- Exactly **40 development source pairs/80 clips** and **100 heldout test source pairs/200 clips**, with forward/reverse members of each source pair dependent. Independent seeds: DEV **261740392**; TEST **496317203**; 280 MP4s generated locally at 256×256/10 frames/6 fps. The world labels derive from generator scene parameters, and reversal switches the direction. All 280 hashes are retained in a private manifest; SHA-256 of the complete source-row manifest: **15a85322176a81d605cc86033acf4bf861eb42ef7d1c09c730bb1b61ae9ce0a5**.
- The 3 comparators consume **only decoded video pixels, the query string, and frozen generic thresholds**, never true world velocity, camera displacement/depth layer, gold direction or correct choice. TEST CV predictions must be persisted before any TEST scoring. Frozen pretrained SmolVLM2-500M 3-frame and 2.2B 3-/5-frame infer 600 outputs under an exclusive shared GPU lock and are saved before scoring. The program fails closed if incomplete.
- Report full-denominator direct accuracy with abstentions counted as incorrect, source-pair-cluster paired bootstrap CI, per-mode correctness/coverage, gate accepted/right/**wrong overrides**, inverse-direction pair consistency, CPU P50/P95 and VLM-only versus tool-gated fallback. True scene metadata may only be used for source hashing and final audit, NEVER to estimate camera shift.

## Development-only preflight — not TEST evidence

The first complete 80-clip DEV run revealed a **substantial negative result**:

| Pixel-only method | Direct correct / 80 | Predicted direction | Gate accepted | Gate right | Gate wrong | CPU median |
|---|---:|---:|---:|---:|---:|---:|
| Plain HSV | **38/80** | 49 | 40 | 29 | **11** | 15.91 ms |
| HSV + prior ORB/RANSAC affine | **32/80** | 39 | 34 | 32 | **2** | 22.15 ms |
| New shape + ORB with strict mixed-depth abstention | **7/80** | 7 | 7 | 7 | **0** | 12.38 ms |

There is **no demonstrated general accuracy gain** from the new method. It abstains in **73/80 DEV clips** (91.25%), frequently on missing contour-shaped target frames, mixed-depth inlier fractions and two visually identical objects. The apparent perfect 7/7 DEV accepted precision has **very weak coverage**, so it is not reliable evidence of system-wide safety. This negative result was measured **before any heldout TEST answer inspection**, and thresholds remain fixed for the prospective test. The 2D affine baseline's decline relative to prior Batch 33 is an expected possible parallax/ambiguity domain failure, not evidence about real video.

Per-mode DEV direct correctness for plain/affine/shape-aware: dual-depth parallax **13/20, 15/20, 1/20**; different-shape same-color distractor **8/20, 5/20, 1/20**; truly indistinguishable same-shape distractor **1/20, 2/20, 0/20**; moving-shadow low-texture **16/20, 10/20, 5/20**. These are development findings and do not justify modifying frozen heldout data or claiming a favorable TEST outcome.

## Implementation and reproducibility boundary

Code: experiments/parallax_identity_data.py (depth-layer renderer with private deterministic world labels), experiments/shape_aware_tracker.py (video pixels and query only, conservative identity gating), experiments/parallax_identity_eval.py (staged blind inference, paired confidence intervals, wrong gate overrides), configs/parallax_identity_v1.json, and tests/test_parallax_identity.py (unit/real-MP4 checks including identical-object abstention). New prospective source regression suite: **135/135 passed**. Source-publication security guard passed during development.

The frozen plan automation/plan.json has 10 strictly ordered stages, each max 1 attempt. No TEST answer scoring may execute until tracker + **all** frozen model outputs are present. The active test directory must be a separate immutable GitHub-main snapshot (repo-batch34-tested) pinned to the new source commit. If code or configuration changes after freezing, preserve logs and create a **new snapshot/batch**; do not alter this scored source. Videos, test manifest, raw predictions, weights, reviewer HTML and keys stay on Tang; only source, tests, prospective protocol and aggregate findings go to GitHub.

## Interpretation rules

Strong accepted wrong overrides invalidate naive coverage claims. A selective classifier can look deceptively precise by abstaining on nearly everything; both selective accuracy and unconditional correctness are mandatory. A pure CV tracker performing better than a pretrained VLM hybrid is **not** a learned multimodal reasoning improvement. A synthetic 2.5D depth-plane experiment cannot establish natural-camera transfer. The next truly independent validation requires consented/licensed **real camera footage**, with separately annotated target identity and true world-relative/camera-relative motion, independent of any previously scored development or test set.
