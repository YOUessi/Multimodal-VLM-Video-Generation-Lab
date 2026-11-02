# Batch 47D — Frozen DEV-only multi-hop temporal bridge experiment

Date: 2026-11-02 UTC. Parent baseline: Batch46E completed all 22 frozen stages, including a single independent TEST evaluation at source SHA `df14676e1f52d9e1a0096581a8558cbbb3f8cd67`. This report documents a **new DEV-only diagnostic**, not a new TEST or a proposed production tracker improvement.

## Why this experiment exists

The prior unchanged affine-KLT tracker reached 60.5% visible PCK@8 and 61.0% visible coverage but incorrectly accepted 36.0% of invisible target frames on the sealed Batch46E TEST. It is not safe to optimize on that opened TEST. Batch47B found low spatial support of some two-dimensional affine fits; Batch47C found large direct KLT spacing can corrupt correspondences. Batch47D isolates whether short-step **temporal bridging** preserves more reliable correspondences than directly matching 30 source frames apart, without any private target labels.

## Preregistered protocol and input pins

Use exactly six existing Batch46 DEV action clips, zero TEST media and zero private target oracle reads. Hard-pin the DEV/TEST selection SHA-256 `0d49c3096e8d302596c6044eefd313508d072e113d880b50ec74cc237f84384c` and downloaded shard manifest SHA-256 `f4f116c84092ef8c99d8f9c423226467a3475c868943bc460994a816175658e1`. Verify each DEV archive SHA-256 from the pinned manifest before extraction.

Compare identical anchor frames and initial Shi–Tomasi detected corners, at width 512 px and an identical 30-original-frame total separation. Use eight anchors per clip (48 DEV anchor pairs). Methods are frozen as:

- `direct_30`: one forward/backward KLT hop of 30 original frames.
- `bridge_3x10`: three hops of ten frames.
- `bridge_10x3`: ten hops of three frames.
- `bridge_30x1`: thirty hops of one frame.

Each hop uses the same pyramidal KLT parameters and a **1.5 px round-trip consistency threshold** at resized resolution; points failing at any hop are permanently dropped and never reacquired. For each method, fit an **image-plane** affine map using the earlier leave-one-quadrant-out cross-fit diagnostic. Record fold support, coverage, survivor fractions, held-out residuals, and per-method matching wall time. Use *paired-only* direct-versus-bridged residuals on the same valid anchors, explicitly reporting when no anchor passes.

Cross-fitting examines spatial generalization of a model of optical-flow correspondences, not known target coordinates, physical camera translation, or occlusion correctness. Large error may have several causes: independently moving foreground, parallax, long-baseline LK failure or bad spatial support. Missing folds are **abstentions**, not zero-error successes.

## Automated execution and non-interference

Plan: `automation/plans/batch47d_temporal_bridge_dev_only_2026-11-02.json`. Entrypoint: `automation/runner_batch47d.py`. Four stages: targeted tests, full project regression, DEV evidence generation, DEV report pin audit. The generic runner enforces Git clean checkout, frozen SHA, one-time stage state, one-worker lock and stop-on-failure. This is a CPU-only study and does not take the RTX4090. Existing Batch46E source, TEST predictions, scores and historical logs stay immutable. New outputs are write-once under the shared `outputs/research/` directory and never committed.

After the DEV audit, any algorithm changes or threshold selection must use a **new development split**; the already-opened Batch46E TEST cannot be used for candidate promotion. A future *fresh, class-disjoint* TEST is required for an actual improvement claim.

## Files and verification

- `experiments/dev_temporal_bridge_affine.py`: matching, frame controls and report generator.
- `experiments/dev_temporal_bridge_audit.py`: data/source/coverage/evidence pin verification.
- `tests/test_dev_temporal_bridge_affine.py` and `tests/test_dev_temporal_bridge_audit.py`: fail-closed regression tests.
- `docs/BATCH46E_FROZEN_INDEPENDENT_KINETICS_RESULTS_2026-11-02.md`: factual prior TEST result, not parameter-selection instructions.

No raw media, image frames, embeddings, weights, predictions, private oracle, API keys or credentials should be pushed to GitHub.
