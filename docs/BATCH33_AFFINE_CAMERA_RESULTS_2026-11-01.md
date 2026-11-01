# Batch 33 — independent synthetic affine-camera heldout: measured results (2026-11-01)

**Status: completed on Tang; source frozen at `a6fbb8dc6e2715304c339ba1012b1ca1d23e3bd2`.** All 10/10 stages returned exit 0, frozen-source regression suite **121/121 passed**, publication guard passed. No thresholds were tuned using heldout answer labels. This is a **new** independently seeded procedurally rendered 2D scene holdout, **NOT** human-captured camera video; positive results do not prove natural-video transfer, VLM reasoning or 3D parallax robustness.

## Immutable source and what actually ran

- New dataset: **40 DEV scene pairs (80 forward/reverse MP4 clips)** and **100 TEST scene pairs (200 MP4 clips)**, four balanced nuisance families × five balanced world-relative target-direction classes. World-relative ground truth differs from image-plane apparent motion, and reversed video pairs are sampled as **one** independent source scene when bootstrapping.
- Modes: camera translation, camera rotation, camera zoom and near-same-color target/distractor with partial target occlusion and moving camera. Ground-truth camera affine transforms are private *scoring-only* metadata; all three estimators received only decoded video pixels, the question, and frozen thresholds.
- Dataset manifest SHA256: `5303647d2a910fba29b91691cfc1fff40c1b9ad58b4217ac4c6c1be8eccdc7ce`; all 280 MP4 bytes were verified on Tang. The tracker saved all 200 predictions before the test answer scorer ran.
- Real GPU workload: 200 SmolVLM2-500M 3-frame, 200 SmolVLM2-2.2B 3-frame and 200 2.2B 5-frame inference outputs (**600 new VLM inferences**). Models were pretrained/frozen, not fine-tuned. Maximum allocated VRAM 1.054 GiB (500M), 4.357 GiB (2.2B). Model-generation time P50/P95: 0.097/0.124 seconds (500M), 0.139/0.199 seconds (2.2B), excluding CPU decode/tool.
- Local results, predictions, 280 MP4s, reviewer HTML, logs, original metadata and model files remain in Tang's `repo-batch33-tested`, not committed to GitHub. Published documentation contains only aggregate evidence.

## Blind test: direct world-relative target-motion classification

All fractions use the full 200 TEST MP4 denominator; 50 clips per nuisance family. **Abstention counts as not correct**, rather than being excluded.

| Frozen method | Overall correct | Pan | Rotation | Zoom | Same-color/occlusion | Emits direction | Abstains |
|---|---:|---:|---:|---:|---:|---:|---:|
| HSV source-pixel (plain) | 141/200 | 36/50 | 41/50 | 40/50 | 24/50 | 168 | 32 |
| HSV + global translation phase correlation | 55/200 | 48/50 | 4/50 | 3/50 | 0/50 | 65 | 135 |
| **HSV + ORB/RANSAC partial affine** | **179/200** | **48/50** | **49/50** | **48/50** | **34/50** | **180** | **20** |

The **affine vs plain** improvement is **+19.0 percentage points**, paired **scene-cluster bootstrap 95% CI [+12.0,+27.0]pp**. Affine vs phase is +62.0pp (CI [+52.5,+71.0]pp). Paired improvements versus plain in each 50-clip slice are pan +24pp [8,40], rotation +16pp [4,32], zoom +16pp [4,32], same-color/occlusion +20pp [4,36]. These are one heldout procedural distribution, not reproducibility across independently generated domains/seeds.

On the post-hoc camera registration audit, median 2D background-point mapping error for accepted affine registrations was 0.059px (pan, 48/50 registered), 0.183px (rotation, 49/50), 0.253px (zoom, 48/50), 0.156px (color ambiguity, 35/50). The generator's actual affine transform was used **only for this analysis after predictions were frozen**, never as tracker input. The old phase method's camera mapping residual was ~34px in the accepted rotation subset and ~58px in the accepted zoom subset.

## Conservative gate errors, accuracy and costs

| Method | High-confidence gate accepted / 200 | Right / accepted | Wrong overrides | Accepted precision | CPU P50/P95 per video |
|---|---:|---:|---:|---:|---:|
| Plain | 105 | 98 | **7** | 93.3% | 10.79/18.15 ms |
| Phase | 39 | 39 | 0 | 100% | 21.74/30.52 ms |
| **Affine** | **144** | **143** | **1** | 99.3% | **36.26/43.45 ms** |

The affine override error occurs in the **same-color/occlusion** mode. Its direct correctness there is only **34/50** (versus 49/50 for rotation and 48/50 for zoom): **correct camera geometry cannot repair a color-only target-identity switch.** Affine is ~3.4× the plain method's median CPU time and should not be deployed without latency/precision constraints. Phase's zero gate errors apply to its very low coverage (39/200), not a demonstration of universal precision.

Forward/reverse consistency: affine produced both directions in 89/100 scene pairs, of which 88 were direction-inversion-consistent; plain 82/83; phase 28/28. Missing a direction in a pair is not an extra success.

## Pretrained VLM baseline and tool fallback

| Frozen model | Frames | VLM-only | Plain gate + VLM | Phase gate + VLM | **Affine gate + VLM** |
|---|---:|---:|---:|---:|---:|
| SmolVLM2-500M | 3 | 40/200 | 118/200 | 69/200 | **156/200** |
| SmolVLM2-2.2B | 3 | 41/200 | 117/200 | 67/200 | **147/200** |
| SmolVLM2-2.2B | 5 | 45/200 | 120/200 | 71/200 | **152/200** |

Compared with model-only, affine tool fallback improved 500M 3-frame by +58.0pp (paired scene-cluster CI [+49.5,+66.5]), 2.2B 3-frame by +53.0pp (CI [+45.0,+60.5]), and 2.2B 5-frame by +53.5pp (CI [+45.5,+61.5]). **A direct deterministic affine tool (179/200) actually outperforms these gate+VLM hybrids** because the fallback VLM often fails on samples the gate declines. Thus this is an object-track tool engineering result, **not** evidence that the pretrained VLM learned better temporal reasoning.

## Research limits and next bottleneck

This is synthetic 2D colored-shape video using the same broad visual conventions as earlier work, although with new independent source scenes and genuinely new geometric nuisances. The tracker remains dependent on an HSV color-based object finder, fails on some same-colored distractors, never estimates depth, and its partial affine motion model assumes a single dominant planar textured world. Natural videos, real parallax, strong shadows and true near-lookalike targets remain unvalidated. The next study should freeze **licensed/consented real-camera video and independent camera-/world-motion annotations**, or another genuinely new data domain, before selecting any model on that test. If researching the identity problem, build an independent development split for shape/appearance association and uncertainty gating; do **not** tune against this now-scored 200-video heldout.

## Reproduction/audit

Authoritative scored code SHA: `a6fbb8dc6e2715304c339ba1012b1ca1d23e3bd2`. Tang frozen directory: `/home/you/projects/Multimodal-VLM-Video-Generation-Lab/repo-batch33-tested`. Frozen plan: `automation/plan.json`; state: `.automation/state.json` (`ALL_APPROVED_STAGES_DONE`); stage logs: `.automation/logs/R*.log`. Machine-readable report: `outputs/affine_camera_motion_v1/results.json`, local video viewer: `outputs/affine_camera_motion_v1/review.html`. Only source/config/tests/aggregate reporting belongs on GitHub. A later **presentation-only** VLM Markdown separator correction and extra regression test are tracked on GitHub `main` separately; the completed experimental source SHA and results are unchanged.
