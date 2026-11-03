# Batch 63 — Causal temporal-consensus veto falsification (DEV-only)

**Date:** 2026-11-03 UTC. **Status:** completed exploratory negative result; **no deployment**.
**Repo:** YOUessi/Multimodal-VLM-Video-Generation-Lab.
**Source:** frozen Batch 56B (SHA 0d70b09b2be7287fb2e95be0f9d84a86cf4888b7), 36 DEV scenes, 144 official point queries.
**Method:** appearance_fb_055. **Safety budget:** scene occluded false-accept rate at most 10%.
**Previous experiment:** Batch 62 single-frame agreement with template_recover4 (4 px) or affine_stabilized_klt (3 px).

## Question and hypothesis

Batch 62 found that template agreement reduces DEV false accepts, but failures may
persist if both trackers follow the same incorrect foreground. Can a *causal
confirmation streak* suppress transient false accepts sufficiently to justify
the cost in visible-target coverage?

Streak lengths 1, 2, and 3 were fixed before recomputing audited metrics;
no ground-truth-driven tuning occurred in this run. They were **not**
pre-registered on unseen data. This is the previously studied Batch 56 DEV,
so this analysis cannot support independent generalization claims.

## Methods

For query time \(t_0\), base location \(p_t\), support location \(q_t\),
and fixed radius \(r\), define:

\[
e_t=\mathbf{1}[p_t\ne\varnothing \land q_t\ne\varnothing
                     \land \lVert p_t-q_t\rVert_2 \le r].
\]
\[
s_{t_0}=0,\quad
s_t=\begin{cases}s_{t-1}+1 & e_t=1 \\ 0 & e_t=0\end{cases},
\qquad a_t=\mathbf{1}[s_t\ge k].
\]

Both \(e_t\) and \(s_t\) use *current and past* frames only. There is no
query-frame credit, future-frame access, or coordinate correction.
Missing predictions or disagreement reset the streak.

Six fixed comparisons: original no-veto baseline, one-frame template gate,
two-/three-frame template confirmation, two-frame affine confirmation,
and two-frame dual template-and-affine confirmation.

The mixed-split private oracle is consumed **only for the 36 DEV records**
kinetics_00 through kinetics_35. Its guarded reader stops before any TEST
record. Frozen SHA-256 and the existing Batch 62 baseline are replay-verified
before the audit. No running experiment was modified.

Scene-macro results are averages across scenes. Three scenes have no
occlusion support; they are never counted as safe-zero observations.
Persistent false-accept episodes are consecutive erroneously accepted
frames within annotated occlusion segments.

## Results

| Gate | DEV visible coverage (scene macro) | Occluded false accept (scene macro) | Jaccard | Scenes >10% / 33 | Longest false-accept run |
| --- | ---: | ---: | ---: | ---: | ---: |
| No veto | 45.36% | 5.12% | 0.3227 | 4 | 117 |
| Template 4px, 1 frame | 34.09% | 1.165% | 0.2695 | 1 | 117 |
| **Template 4px, 2 frames** | **32.89%** | **1.138%** | **0.2601** | **1** | **117** |
| Template 4px, 3 frames | 31.86% | 1.069% | 0.2525 | 1 | 117 |
| Affine 3px, 2 frames | 39.88% | 4.63% | 0.2843 | 4 | 117 |
| Dual template + affine, 2 frames | 30.15% | 1.082% | 0.2373 | 1 | 117 |

The persistent high-risk scene **kinetics_17** retains **26.43%**
occluded false acceptance for the one-, two- and three-frame template
gates and the two-frame dual gate. The longest false-accept episode
remains **117 consecutive occluded frames** even after temporal gating.

### Paired effect relative to the one-frame template gate

| Added confirmation | Occluded false-accept frames removed | Correct visible frames lost | Scene-macro coverage change | Scene-macro false-accept change |
| --- | ---: | ---: | ---: | ---: |
| Template streak 2 | 2 | 178 | −1.196 percentage points | −0.0274 percentage points |
| Template streak 3 | 5 | 328 | −2.227 percentage points | −0.0955 percentage points |
| Dual streak 2 | 5 | 725 | −3.935 percentage points | −0.0831 percentage points |

The incremental reduction of false acceptance is small, while correct
visible-frame loss is much larger. The scene-level 10% safety failure
persists. A prolonged streak can confirm a **shared incorrect track**;
temporal agreement is not independent evidence of target identity.

### Exploratory scene-cluster sensitivity

The deterministic 2,000-resample paired **scene-level** bootstrap (seed
6303) gives the following descriptive percentile intervals for the
two-frame template streak versus the one-frame template:

- Scene-macro coverage difference: **−1.497 to −0.927 percentage points**.
- Scene-macro false-accept difference: **−0.0699 to 0.0000 percentage points**.

These are not prospective confidence bounds or independent deployment
certification; scene exchangeability is not established.

## Negative findings and safety boundary

The 36 DEV scenes were previously studied (Batches 58, 60, 61, 62).
No new independent TEST was accessed; Batch 59 frozen method selection,
cron and files remain unchanged. Additional supporters still require
extra CPU tracker computation. Frame-streak arithmetic alone is cheap,
but new end-to-end CPU latency was **not** measured. Shared tracker
bias can persist even when optical agreement is high; global affine
movement is not proof of static background or 3D camera motion.

**Decision: REJECT temporal streak variants for production.**
Seek stronger *observable, independent* visibility/background evidence
and evaluate it on new frozen independent natural videos without
using TEST labels to tune method selection.

## Reproduction / evidence

From the isolated DEV working copy with original immutable Batch 56
inputs available as parent-directory mounts:

    ../.venv/bin/python -m pytest -q tests/test_batch63_dev_temporal_consensus.py
    ../.venv/bin/python -m experiments.batch63_dev_temporal_consensus
    ../.venv/bin/python -m pytest -q

Immutable output: outputs/research/batch63_dev_temporal_consensus.json

Output SHA-256:
8bffd75de614f49b52c5394f9ad10d477bb5970f7db06f18ee89f2eaef20269f

The generator creates a sidecar hash and refuses overwrite.
No private oracle data, media, model weights or secrets belong in Git.
