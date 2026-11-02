# Batch53 — frozen replication on new action-class-disjoint Kinetics videos

**Prepared:** 2026-11-02 UTC / 2026-11-03 JST. **Protocol:** no
selection on any previously opened TEST scores; new DEV/TEST labels remain
untouched until their respective frozen stages. GitHub `main` is the source
of truth, Tang the execution host.

## Trigger: completed independent Batch48B and observed safety failure

Batch48B completed **22/22 approved stages** with a frozen Git SHA of
`9d86f354f4cbed039ef8ea47f14c8eb37df374b3`. Six genuinely new
DEV and six new sealed TEST natural Kinetics videos were selected without
reusing action classes or IDs from earlier Batch42, Batch45E and Batch46E.

The Batch48B candidate `klt_fb` was selected using the original DEV-only
criterion: >=30% scene-macro visible coverage, <=10% scene-macro
invisible false acceptance, then largest DEV scene-macro Jaccard.

| Frozen selected-method metric | New DEV | Independently scored TEST |
|---|---:|---:|
| PCK@8 unconditional visible | 54.26% | 56.69% |
| Visible coverage | 56.70% | 60.51% |
| Accepted-point precision@8 | 94.07% | 94.58% |
| **Invisible false acceptance** | **2.00%** | **22.24% — FAIL** |
| Scene-macro Jaccard (radii 1/2/4/8/16 px) | 0.452 | 0.439 |
| CPU p50 ms per target-frame | 0.091 | 0.119 |

The latency explicitly excludes video decoding, shared affine estimation
and end-to-end deployment time. The TEST method was not reselected.

New TEST failures include `kinetics_08` with 81.05% false acceptance
(95 invisible samples), and `kinetics_06` with 27.63% false acceptance
(503 invisible samples); other scenes have small or zero invisible support.
These are **post-hoc error diagnostics only**, not grounds for modifying
Batch53 thresholds, selected method, or candidate ranking.

Frozen SmolVLM2-2.2B three-image DEV gate **BLOCK**: balanced accuracy
31.03%, not-visible false acceptance 57.89% (11/19). TEST VLM
inference was correctly skipped. This is not a direct same-frame-budget
comparison against dense CPU KLT trackers.

## Batch52 independent DEV evidence

Batch52 created a pinned, write-once six-scene leave-one-natural-video-out
DEV reliability audit. Every five-scene subset independently selected
`klt_fb`; five held-out videos with invisible support satisfied both
point-estimate constraints, one had zero invisible support and is **not
evaluable** for occlusion safety. In the lowest-light DEV scene
(mean luminance 24.53), coverage was 32.2%, invisible false acceptance
8.59%, and Jaccard 0.176. These data suggested a vulnerability but do
not establish causality or justify retuning on prior TEST.

## New Batch53 registered experiment, BEFORE new TEST scoring

- **Exactly unchanged from Batch48B:** KLT, template, affine, Sobel
  appearance-gate candidate algorithms and thresholds; DEV choice criterion;
  6 DEV + 6 TEST natural videos, 4 annotated point queries/video, 25 FPS
  256x256, 250-frame cap, all CPU metrics, VLM model and DEV gate, 22
  stage order, GPU-exclusive and single-writer fail-closed protection.
- **Changed for independence only:** new deterministic salt
  `MVVG-BATCH53-NEW-CLASS-DISJOINT-REPLICATION-20261103`,
  new Batch53 video selection/data/output namespace and commit/pins;
  required historic metadata exclusion now includes 12 Batch48B
  classes and video IDs *in addition to* 36 from Batch42/45E/46E.
  This totals **48 historically excluded IDs and action classes**.
- **No old TEST scores enter algorithm, candidate selection, thresholds or
  data selection.** Old manifests are used for public class/ID exclusion
  only; no old oracle or media is read by the selector.
- The newly selected videos may vary in illumination, texture, foreground
  dominance and camera motion. Report 2D affine proxies separately from
  physical background-relative motion, which these signals cannot identify.
  No claim of calibrated camera motion or 3D ego-motion.
- The original safety constraints are explicitly tested on both new DEV
  and sealed TEST, with failure retained as a negative result. No TEST
  adaptation even if the method underperforms.

## Automated execution and immutable artifacts

Code files: `experiments/kinetics_batch53_{data,eval,vlm,report}.py`.
Config: `configs/kinetics_batch53.json`.
Plan: `automation/plans/batch53_fresh_independent_safety_dev_select_2026-11-03.json`.
Entrypoint: `automation/runner_batch53.py`.
New regression: `tests/test_kinetics_batch53.py`.

The frozen, clean GitHub checkout is `repo-batch53-tested` and runner
writes `.automation/state.json` plus per-stage logs. Cron marker
`MVVG_PROJECT4_10MIN` may point to Batch53 only after previous
Batch48B has `ALL_APPROVED_STAGES_DONE` and GitHub/checkout SHA,
protocol-integrity checks and full regression have passed.

Frozen plan stages: download 12 fresh action classes, targeted+full
regressions, prepare sealed manifest/oracle, public-only audit, DEV CPU
predictions and scoring, select DEV method, GPU-exclusive DEV VLM,
DEV gate, pinned TEST CPU predictions, conditional TEST VLM,
one-time TEST scoring, immutable report, final security and provenance
audits. No stage may be silently retried after a failure.

Pre-publish development regression **694/694 passed** (3 existing Pillow
warnings). After publishing the batch source SHA, run the full regression
again on the final frozen checkout before arming cron.

Only aggregate reports, code, tests and plans can be in GitHub; no
video, image, annotation oracle, individual prediction, model weights,
access tokens or other secrets should be committed.

**Decision:** this is a new *replication* of the original method selection
under domain shift, not an optimized tracker. If TEST false acceptance
remains high, treat as a generalization failure and design separately
frozen new DEV algorithms without using this TEST to select thresholds.
