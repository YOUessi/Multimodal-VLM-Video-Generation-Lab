# Batch47E — Controlled synthetic image-plane motion on real DEV frames

Date: 2026-11-02 UTC. Batch47D completed 4/4 stages and 539 regressions. Batch46E already opened its independent TEST: the frozen affine tracker reached 36.0% invisible false acceptance. That TEST is prohibited for any new model or threshold selection.

## Purpose

Batch47D direct 30-frame matching had valid spatial crossfit on 24/48 DEV anchors; 30x1 bridging achieved 32/48, but required greater CPU time and had higher conditional spatial residual. Those are unlabeled image-plane diagnostics, not evidence of correct background compensation. Batch47E introduces known synthetic global translation and independent synthetic foreground to test whether apparently confident affine estimates truly follow the background.

## Data and preregistration

Read only six SHA-pinned Batch46 DEV clips; no TEST video or target oracle. Selection SHA256: 0d49c3096e8d302596c6044eefd313508d072e113d880b50ec74cc237f84384c. Download manifest SHA256: f4f116c84092ef8c99d8f9c423226467a3475c868943bc460994a816175658e1. Verify each DEV shard hash and extract only its whitelisted member.

Six clips x eight fixed source-frame anchors x five predetermined conditions = 240 generated pairs, at width 512 px. All share known image-plane shift (dx,dy)=(6,-4) px: clean affine, lighting shift, local independently moving textured foreground, large moving foreground spanning many quadrants, and independently moving foreground plus a newly occluded region. Synthetic random layers use a seed derived from DEV video ID and anchor.

The same input frame underlies the five paired conditions. Keep separate synthetic known-background grid points that are untouched by moving foreground, its destination footprint, occlusion, or invalid borders. Only use this ground-truth grid AFTER fitting and the gate decision.

## Frozen algorithms and comparisons

Existing forward-backward KLT with 1.5 px cycle threshold; global affine RANSAC; leave-one-quadrant-out crossfit. The predeclared hypothetical gate accepts only if at least three folds are valid, worst heldout-quadrant median error <=3 px, and heldout fraction within 3 px >=0.65. Wrong synthetic global motion is defined as median error >2 px against known background grid.

Report fit coverage, gate coverage, baseline wrong fits, wrong accepted, errors conditioned on successful fit and acceptance, and per-pair CPU timing for matching+fitting+crossfit. A missing estimate is not a zero-error success. An accepted incorrect estimate counts as a gate failure.

## Scope and limitations

This is already-used DEV footage with generated motion, NOT a fresh independent TEST, NOT 3D camera pose, NOT actual video-object occlusion tracking, and NOT evidence of target visibility accuracy. Artificial patches and brightness changes may not resemble real domains. Decoding cost excluded. No production tracker updates, TEST tuning, gate promotion or GPU work.

The next real validation requires a newly frozen DEV split with real target annotations, then a new class-disjoint sealed TEST. Old Batch46E TEST must never choose a candidate. Maintain both baseline and gate precision/coverage and per-scene breakdown.

## Execution and artifacts

Freeze the GitHub main SHA in an isolated clean repo-batch47e-tested. Execute plan automation/plans/batch47e_dev_counterfactual_layers_2026-11-02.json using automation/runner_batch47e.py. Stage sequence: focused tests; all tests; deterministic six-clip counterfactual experiment; evidence and source pin audit. Fail closed, write-once output, preserve logs, no duplicate tasks. Private media, ground-truth grid, generated pairs, weights and secrets stay on Tang under outputs/research; do not commit them.

## Executed DEV preflight observations (exploratory, not TEST)

Before freezing the run, the fully implemented protocol was executed on all 240 generated pairs and the evidence audit passed (6 scenes, 240 cases; no sealed TEST access). Each condition has 48 cases, with 48/48 affine fits valid, so these are defined coverage comparisons on this synthetic design, **not** independent video-domain evaluation.

| Condition | Raw affine estimates wrong (>2px) | Proposed spatial gate accepted | Wrong among accepted | Accepted coverage |
|---|---:|---:|---:|---:|
| clean_affine | 0/48 | 36/48 | 0/36 | 75.0% |
| lighting_shift | 0/48 | 35/48 | 0/35 | 72.9% |
| local_foreground | 41/48 | 0/48 | 0/0 (undefined) | 0.0% |
| dominant_foreground | **48/48** | **26/48** | **26/26** | 54.2% |
| foreground_and_occlusion | 47/48 | 0/48 | 0/0 (undefined) | 0.0% |

The failure is more important than the apparent successes: a large coherently moving foreground spans multiple spatial regions and can pass quadrant holdout validation even while the estimated warp is wrong for the background. The gate's **conditional wrong-accept fraction is 100%** on accepted dominant-foreground examples. For localized foreground and foreground+occlusion, it abstains entirely, so it is not a usable tracker without separate target coverage evaluation.

This disproves the assumption that quadrant crossfit alone certifies *background* motion. A next candidate needs an independent cue that separates background support from dominant coherent foreground, rather than merely decreasing a spatial residual threshold. Synthetic truth may guide controlled analysis on DEV only; a new independent DEV plus fresh sealed TEST are still required for any performance claim.
