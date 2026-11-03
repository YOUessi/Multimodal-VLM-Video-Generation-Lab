# Batch 68 — Causal Optical-Flow Agreement: DEV Falsification (2026-11-03)

## Research question and scope

Can one-step, image-observed, forward/backward Lucas–Kanade (LK) flow offer additional *causal* evidence to reject wrong tracks that two geometrically consistent trackers jointly accept? This is motivated by the 117-frame false-acceptance run established in Batches 63–67, not by an assumption that another tracker automatically provides independent evidence.

**Strict scope:** This is an exploratory analysis of the **previously studied** Batch 56 Kinetics **36 DEV natural videos, 144 official point queries**, using the immutable Batch 56 prediction and public frame/query hashes. It is **not** a fresh blind test, not a class-disjoint TEST result, and not a deployment candidate. The frozen Batch 59 (36 DEV + 6 TEST) protocol, cron, SHA and files are unchanged.

## Method and firewall

The label-blind extraction phase reads only frozen unscored DEV predictions, public manifest, public queries and the 36 DEV RGB frame arrays, each verified against its SHA-256 manifest entry. For each predicted point, a one-step LK forward track is estimated from previous to current RGB frame, then reverse-tracked to measure cycle error. A point is rejected when missing or outside safe bounds; no future frame or label is accessed.

Two checks, fixed *before* score inspection, are applied either to the original method or in conjunction with the previously defined 4-pixel template-agreement gate:

| Variant | Candidate versus 1-step LK displacement | Forward/back cycle error | Template prerequisite |
|---|---:|---:|---|
| LK strict | ≤2 px | ≤1 px | No |
| LK loose | ≤4 px | ≤2 px | No |
| Template+LK strict | ≤2 px | ≤1 px | Yes |
| Template+LK loose | ≤4 px | ≤2 px | Yes |

The methods only veto an existing prediction; they neither alter coordinates nor create new detections. Whole-image phase correlation at half resolution yields an *auxiliary two-dimensional dominant-image translation proxy*. It is **not** calibrated camera motion, proof of static background, object/world velocity or depth-aware compensation. A regression test discovered that Hanning-windowed phase correlation could assign high confidence to completely flat images; extraction now rejects low-texture frames before phase estimation.

The feature JSON is written once and SHA-256-pinned **before** the private DEV oracle may be opened. The score phase authenticates the pinned feature JSON, existing DEV result SHA and the original selection SHA, streams only the 36 DEV oracle records, and fails if baseline counters differ from the original frozen results. No TEST oracle or predictions are parsed.

## Real DEV results

| Method | Scene-macro occluded false accepts | Visible coverage | Scene-macro Jaccard | Scenes exceeding 10% | Longest wrong run |
|---|---:|---:|---:|---:|---:|
| Original tracker | 5.123% | 45.356% | 0.3227 | 4/33 | 117 |
| Template 4 px | 1.165% | 34.087% | 0.2695 | 1/33 | 117 |
| LK strict | 4.826% | 44.376% | 0.3188 | 4/33 | 117 |
| LK loose | 4.853% | 44.496% | 0.3188 | 4/33 | 117 |
| Template + LK strict | 1.129% | 33.933% | 0.2686 | 1/33 | 117 |
| Template + LK loose | 1.129% | 33.956% | 0.2687 | 1/33 | 117 |

Relative to template agreement alone, the strict LK veto eliminates **3 additional occluded false-accept frames** at the cost of **20 previously correct visible predictions**; the loose veto eliminates **3** at the cost of **17**. This is not a favorable resolution of the main failure mode. These paired quantities are **frame counts**, not independent video-level effects or causal effect estimates.

The dominant failure, **kinetics_17**, remains unchanged by either template+LK gate: **139/526 occluded frames falsely accepted**, including the same **117 consecutive** wrong frames. The one-step LK estimator is able to follow the same incorrect motion; local flow-cycle consistency does not establish identity, visibility or object permanence.

Across all 36 DEV scenes, the offline extraction phase took **3,387.5 ms total** and **77.94 ms median per video** on Tang. These are partial CPU measurements **excluding** tracker evaluation, decoding, hashing and orchestration. They are neither real-time online latency nor a GPU benchmark.

## Engineering artifacts and reproducibility

- Implementation: `experiments/batch68_dev_causal_flow.py`.
- Targeted synthetic/contract tests: `tests/test_batch68_dev_causal_flow.py`.
- Label-blind feature output on Tang: `outputs/research/batch68_unlabeled_dev_flow.json` and `.sha256`.
- DEV-only diagnostic result on Tang: `outputs/research/batch68_dev_flow_consistency.json` and `.sha256`.
- Immutable feature SHA-256: `51a97b7e67ef22fb16a85dd45251e17b29b74fbed537218c80cf7a3552db3af3`.
- Immutable result SHA-256: `ec2075957c5d80a971d38542671bb0f19bf32021d2c7732b8ed9316724ae4f4a`.
- Targeted tests: **41 passed**.
- Entire project suite: **1,202 passed, 1 skipped, 3 pre-existing Pillow deprecation warnings**.

The first full-suite attempt found 45 unrelated history-manifest failures due to a missing **development-workspace shared-data symlink**, not changed project code or frozen data. The workspace-only `data -> ../data` symlink was restored, and the full suite passed. This failure log remains on Tang as `outputs/research/batch68_full_pytest.log`; the passing rerun remains as `outputs/research/batch68_full_pytest_fixed.log`.

## Research decision

**Reject deployment.** The newly implemented local flow agreement is useful as a low-cost independent computation, but is *not independent identity evidence*: the long occluded false-track episode survives all gates, and correct visible predictions are discarded for negligible safety gain.

Future work should prioritize explicit target visibility / re-identification or genuinely independent scene/occluder evidence, prospectively pre-registered on **new independent videos**. Finish Batch 59's untouched class-disjoint TEST once its data are available. Do not select LK thresholds using either its TEST labels or the previously opened DEV outcomes.
