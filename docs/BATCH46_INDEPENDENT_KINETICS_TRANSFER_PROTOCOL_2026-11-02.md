# Batch46 — Independent Kinetics Action-Class Holdout / Pre-Registered Tracker Transfer

Date: 2026-11-02. Baseline GitHub main: `d0e76bf409bde7d91a42d2af09bf72a7e932af7f`.

## Hypothesis and experimental boundary

Batch45E chose affine-stabilized KLT with its DEV-only scene-macro Jaccard objective; subsequent sealed TEST had weaker invisible-point protection than its DEV result (4.5% -> 20.8% scene-macro), and the SmolVLM2-2.2B visibility gate was blocked. This study tests whether the **unchanged** Batch45E choice and frozen parameters generalize to unseen natural footage.

This is an independent **within-source-family** replication, not cross-dataset or cross-camera-geometry validation. Twelve new TAP-Vid-Kinetics video IDs from twelve action classes unused by both Batch42 and Batch45E, six DEV and six sealed TEST, four query tracks each. Old TEST accuracy is not used to choose a new model.

## Immutable protocol

- History exclusion requires both old public metadata manifests. Fail closed if either is missing, malformed, or overlaps by action class or video ID.
- Source Kinetics validation metadata, TAP-Vid CSV, class archive URLs and upstream coordinate conversion are SHA-256 checked. New video media and extracted frames stay local.
- Frozen method: `affine_stabilized_klt`, inherited strictly from Batch45E `dev_selection.json` SHA-256 `c7a7804918af64ff5bd8cc817db22ab16745f7dcd910e6e26c1601bf85133dc2` at source `e59aa07c6ab69a40ab2bb35e64cad4645410d10c`. All baseline methods remain unchanged; no Batch46 DEV or TEST label may select another method.
- Compare fixed tracker with KLT forward, KLT forward-backward, template, NCC gates, and gradient gates only **descriptively**. No hyperparameter or parser tuning from TEST.
- SmolVLM2-2.2B retains its three-image reference crop zero-shot prompt and DEV-only admission gate; TEST inference must be skipped unless all predeclared DEV thresholds pass.
- Commit code and tests before freezing. All media, weights and private oracles remain untracked and off GitHub.
- Each test prediction artifact must be SHA-256 pinned before its oracle can be opened. CPU and VLM TEST predictions both precede either TEST score. On stage failure stop and preserve logs; only a new frozen SHA and new tested directory can retry.
- Distinguish image-plane displacement, background-relative 2D affine residual, and calibrated physical 3D camera/world motion (not measured). Track visible coverage, accepted precision, occluded false accept, Jaccard and CPU cost jointly.
- This study is a replication. Post-hoc per-scene differences are diagnostic only; treat small scene samples as exploratory.

## Deliverables

Frozen public/private manifest hashes; DEV and TEST predictions + pins; scene-macro/micro tracking metrics and occlusion; DEV VLM results and admission decision; TEST VLM result or documented skip; 2D relative-motion diagnostics; 95% scene bootstrap differences; full regressions; aggregated SHA-pinned report; GitHub code+protocol+public summary.
