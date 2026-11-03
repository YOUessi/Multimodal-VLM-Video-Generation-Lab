# Batch 62 — DEV-only per-frame tracker consensus veto falsification

Date: 2026-11-03. Research status: **exploratory / not deployable**.

## Question and strict data boundary

Batch 61 showed image-only scene-level risk proxies were worse than a train-fold
prior on the previously inspected 36-scene Batch 56 DEV cohort. Here we test
a different, target-specific *observable* signal: can a second tracker veto an
accepted prediction when its proposed location disagrees?

We reuse only the already frozen Batch 56 **DEV** predictions, public queries
and the first 36 DEV truth objects for scoring. All prediction records and the
DEV selection/report digests are checked before the DEV oracle is accessed.
The oracle-private.json file also contains 6 TEST objects; a format-locked,
one-scene-at-a-time parser **stops at the end of kinetics_35**, never parsing
TEST truth. The TEST predictions, results and videos are not opened.

The chosen baseline is the previously pinned appearance_fb_055. This experiment
cannot choose a production threshold or claim a new independent holdout.

## Precisely defined interventions (fixed before scoring)

Every intervention **only abstains**; it never moves a point, makes an
existing abstention visible, or reads truth at inference time.

- **Baseline**: original appearance_fb_055 trace.
- **Template 4 px**: accept a baseline point only if template_recover4
  produces a point within Euclidean distance at most 4 px.
- **Affine 3 px**: accept a baseline point only if affine_stabilized_klt
  agrees within 3 px.
- **Both**: both of the previous supports must be present and agree.

These secondary methods are not independent object-identity sensors. The
affine solver fits a **2D image-plane dominant motion proxy**, not 3D camera pose
or true static-background motion. Shared optical flow can cause correlated
failure, and conservative consensus can incorrectly reject a visible target.

## Replayed real DEV results

The original selected tracker's per-scene visible/occluded, accepted, correct@8
and false-accept frame counts were reproduced **exactly** before comparison.

| Frozen DEV measure | Baseline | Template 4 px | Affine 3 px | Both |
|---|---:|---:|---:|---:|
| Scene-macro visible coverage | 45.36% | 34.09% | 41.14% | 31.32% |
| Scene-macro invisible false-accept | 5.12% | **1.16%** | 4.77% | **1.11%** |
| Scene-macro Jaccard (1/2/4/8/16) | **0.3227** | 0.2695 | 0.2934 | 0.2466 |
| Occlusion-supported scenes | 33 | 33 | 33 | 33 |
| Scenes over 10% risk budget | 4/33 | **1/33** | 4/33 | **1/33** |
| Occluded false-accept frames (micro) | 317/10797 | 162/10797 | 294/10797 | 159/10797 |
| Accepted visible frames (micro) | 9915/21166 | 7479/21166 | 9104/21166 | 6922/21166 |

**Tradeoff**: template agreement reduced scene-macro false accepts by about
3.96 percentage points, but cost 11.27 percentage points of visible coverage
and 0.0532 of scene-macro Jaccard. The combined gate barely improved over the
template-only gate on safety and further reduced coverage and Jaccard.

**Failure persists**: kinetics_17 still exceeds the 10% budget under
template and combined gates, reaching **26.43%** occluded false acceptance.
The six fixed-index DEV scene blocks also vary strongly; this is not evidence
that the threshold would pass independent TEST or a new scene distribution.

No new model parameters were trained and no already unsealed TEST score was
used to change these four preset comparisons.

## CPU costs: component-only proxies, not online benchmark

Using previously measured per-query CPU timings from 144 DEV queries,
we sum measured timings for the required tracker components and amortize
scene-level affine registration when applicable. This is a **cost proxy**;
shared optical flow may be counted twice, and orchestration, decoding,
cache conditions and real fusion/veto overhead are unmeasured.

| Variant | Summed components p50 ms / target-frame | p95 ms / target-frame |
|---|---:|---:|
| Baseline | 0.0826 | 0.2881 |
| Template 4 px | 0.1127 | 0.3719 |
| Affine 3 px | 1.6372 | 2.2413 |
| Both | 1.6717 | 2.3011 |

Affine registration alone has a median amortized proxy of ~1.2406 ms /
target-frame in the affine variants. These numbers **are not** the new
combined method's end-to-end latency.

## Interpretation and decision

The cheap template-consensus veto is a plausible DEV research lead: unlike
the image-only scene-classifier from Batch 61, it responds to the candidate
trajectory. Yet its safety benefit comes with a sizable visible-recall penalty,
one remaining severely unsafe scene and correlated-tracker failure modes.

**Decision**: keep **DEV-only, not promoted**. The already running Batch 59
frozen plan must remain untouched. A future separately preregistered
class-disjoint evaluation may test this veto, with fixed thresholds, actual
CPU p50/p95 including video decode and end-to-end latency, and no reuse of
the already opened TEST cohorts for tuning.

## Artifacts and verification

- Analyzer: experiments/batch62_dev_consensus_veto.py
- Tests: tests/test_batch62_dev_consensus_veto.py
- Source: frozen Batch56 SHA 0d70b09b2be7287fb2e95be0f9d84a86cf4888b7.
- DEV predictions SHA256: 1a3ba13d3408d5b0305efe210c7732fa38fdb4768b04c22fb68ac52d54ef3d51.
- No media, model weights, private oracle, derived private per-frame labels
  or secrets are committed.
- On Tang run: ../.venv/bin/python -m pytest -q tests/test_batch62_dev_consensus_veto.py.
- Reproduction (using already sealed local DEV assets):
  ../.venv/bin/python experiments/batch62_dev_consensus_veto.py --output outputs/research/NEW_NAME.json.
  The tool forbids overwrites and writes a matching .sha256 pin.
- Actual analysis output (private/local only):
  repo-batch62-dev/outputs/research/batch62_dev_consensus_veto_v2.json
  SHA256 ecb170f95031a43f3e31fb2ab249d4055bb1103c2a1ac04167433c1105361f4f.
- Initial output version 1 is preserved, not overwritten, separately pinned.

The absence of statistically independent new TEST evidence and unmeasured
online fusion runtime prevents any deployment or generalized safety claim.
