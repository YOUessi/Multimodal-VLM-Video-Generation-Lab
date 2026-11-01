# Batch 37B — newly scene-disjoint real DAVIS long-horizon tracking: sealed heldout results (2026-11-02)

**Status: all 8 new SHA-frozen execution stages PASS (165/165 project regressions; 2 existing Pillow deprecation warnings). No TEST-informed parameter tuning.** This report was written **after** the one-time TEST score, in a documentation-only commit. The exact algorithm, plan and config come from frozen GitHub source commit `42de7da590657f1c311e6995ee56f0e2464b4d08`, not from this later report commit.

## Why a B37B recovery exists

The first Batch 37 frozen source `512840d056e289a57606ab3ceeed007e8f83deee` successfully acquired all original assets, passed 165 tests and sealed DEV predictions, but **failed before completing DEV scoring** when NumPy comparison accumulators became `numpy.int64` and could not be encoded with Python `json.dumps`. It generated **no TEST predictions or TEST scores**. Failed stage logs and DEV predictions were preserved, and the original cron batch was paused. Batch 37B created a new exact GitHub code SHA and a regression asserting JSON serialization of all counters. **No model thresholds, sequence selection or frozen media bytes were changed.** This failure must not be counted as tracking evidence or erased from the study.

## Data provenance and frozen evaluation boundary

- Official DAVIS 2017 TrainVal 480p original JPEG video frames and palette-index instance masks, attributed to Pont-Tuset et al. (2017): https://davischallenge.org/davis2017/code.html. Media remain local on Tang, subject to individual clip provenance before any redistribution.
- **12 independently sampled real source videos** within the same archive, completely disjoint in sequence identity from all 12 previously scored Batch 36 scenes. Deterministic DEV scenes: `drift-turn`, `car-shadow`, `motocross-jump`, `kid-football`. Deterministic TEST scenes: `skate-park`, `goat`, `color-run`, `boat`, `tuk-tuk`, `drift-chicane`, `bike-packing`, `dance-twirl`. DEV/TEST are also scene-disjoint.
- Each scene has **16 original camera frames** (0,2,...,30), with the corresponding 16 original instance masks: **192 photographic frames + 192 private masks = 384 SHA/CRC-verified official assets**. These are **not 192 independent videos** and do not guarantee models never saw DAVIS during pretraining.
- Immutable data manifest SHA256: `9399d2d0bc6af96a2390940fc81f8049060b37e1309a3254b838ce21892a9a95`. DEV prediction SHA256: `33be1274b969f9108cdef045daf2f3d3df35e1065e33f8b0e64d360d72bd5821`. TEST prediction SHA256: `ee58d056552cc533a8ed5f4ed969fb00a064c1148ad9fc9e976cda7ad7e62f92`. Both predictions were sealed by SHA256 before opening the corresponding private labels.
- Predictor sees only the **first-frame instance query mask** and RGB images. Future masks and private gold are score-only. No GPU was used. The studied target is **future mask centroid in image-plane pixels**, not a true material point, full 3D motion, camera pose, world-relative direction or VLM multiple-choice QA.

## Predeclared methods: new TEST, 8 videos × 15 evaluated frames

All **120** non-query TEST target positions remained GT-visible. The predeclared strict consensus gate improves **accepted precision**, but **reduces** unconditional PCK@8px and abandons most valid positions.

| Fixed method | PCK@8 on all 120 visible frames | Accepted / 120 | Correct accepted | Wrong accepted | Abstained visible | Accepted precision@8 | CPU P50 / scene |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stationary first-pixel center | 4/120 = **3.33%** | 120 | 4 | 116 | 0 | 3.33% | 0.00 ms |
| Unchanged KLT forward-backward | 43/120 = **35.83%** | 88 | 43 | 45 | 32 | 48.86% | 12.60 ms |
| FB + predeclared temporal consensus | 40/120 = **33.33%** | 67 | 40 | 27 | 53 | 59.70% | 10.59 ms |
| Stricter FB + consensus guard | 28/120 = **23.33%** | 29 | 28 | 1 | 91 | 96.55% | 6.74 ms |

Difference in unconditional PCK@8 vs unchanged FB-KLT: consensus **−2.50 percentage points**, approximately 95% scene-cluster bootstrap interval **[−5.83, 0.00] pp**; stricter guard **−12.50 pp**, interval **[−23.33, −3.33] pp**. Eight source-video clusters are far too few for a broad population-level generalization claim. CPU figures are whole-method-per-16-frame-scene timings, **excluding image decoding, preprocessing outside each tracker, and separate background-affine registration**. Early tracking failure can trivially reduce measured CPU time, so lower latency alone is not a win.

## DEV diagnostic (not a new blind claim)

Four independent DEV videos gave **60 visible / zero absent** non-query frames. DEV unconditional PCK@8: fixed center **0/60**, unchanged FB-KLT **5/60 = 8.33%**, consensus **4/60 = 6.67%**, strict **4/60 = 6.67%**. DEV accepted coverage for unchanged FB/consensus/strict was **46/60**, **15/60**, **9/60**, respectively; the new gates were not selected or tuned on this batch's TEST. DEV's much lower absolute accuracy than TEST highlights uncontrolled scene-composition effects, not a validated generalization effect.

## Absence, apparent occlusion, background movement and negative claims

- **Zero of the 120 TEST evaluation positions were actually annotated absent.** Therefore absent-target false acceptance and occlusion-reappearance performance **cannot be estimated**, even though all algorithms report zero absent false acceptances as a *vacuous count*. It would be invalid to call this zero-error absence handling.
- Only **2/120** TEST instances had mask area below 40% of first-frame target area. This is a **shrink/scale/crop/occlusion proxy** (not validated occlusion ground truth), and both consensus gates abstained on those two positions; no conclusion on occlusion recognition is possible.
- First-to-last dominant 2-D **background affine registration passed for 6/8 TEST and 1/4 DEV sequences**. This is diagnostic only; with no extrinsics, depth or true camera motion labels it cannot separate physical target motion from camera-induced apparent image motion.
- Neither model is a task-matched VLM baseline. No GPU/VLM evidence was generated in this batch, and previous SmolVLM2 direction-choice QA scores **must not** be compared numerically to mask-conditioned centroid PCK.
- Negative outcome: adding stricter motion-consensus gating **did not increase the unconditional target-centroid tracking success rate**. It exchanged coverage for reduced wrong acceptance, with the strict method accepting only **24.17%** of target-visible TEST positions. A high conditional precision cannot be reported as overall tracking accuracy.

## Frozen execution proof and source safety

On Tang, the isolated tested source directory `repo-batch37b-tested` has exact Git SHA `42de7da590657f1c311e6995ee56f0e2464b4d08`; local `.automation/state.json` records eight stages `done` with `returncode=0` and `ALL_APPROVED_STAGES_DONE`. The original 384 official bytes were re-audited; DEV and TEST predictions were generated in strict order and pinned before score, and all project regressions passed twice. Source publication guard reported no tracked media/weights/private data/secrets. All private assets and failed-batch logs remain on Tang, not GitHub. The `MVVG_PROJECT4_10MIN` cron points at Batch 37B but terminates without running when its completion marker exists, preventing repeated empty work.

**Next non-test-overfitting bottleneck:** use a genuinely new, independently frozen corpus with **explicit visibility/occlusion labels and target identity through long occlusions**, rather than continuing to tune DAVIS TEST results. A task-matched VLM baseline must receive the same target identity/query and be scored on the same label space. Distinguish genuine object/world movement from merely background-relative image displacement only when the dataset offers appropriate motion or camera-pose truth. Analyze risk-coverage/calibration on DEV and freeze a future TEST once; preserve low-coverage and identity-switch failures.
