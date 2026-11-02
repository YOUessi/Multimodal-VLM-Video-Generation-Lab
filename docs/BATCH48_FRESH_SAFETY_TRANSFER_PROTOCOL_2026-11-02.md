# Batch48 — independent natural-video safety-constrained DEV selection

Date: 2026-11-02 UTC. Source of truth is GitHub `YOUessi/Multimodal-VLM-Video-Generation-Lab` main. Execute only a clean frozen SHA, never the mutable working tree.

## Decision boundary and why a new TEST is necessary

The earlier Batch46E experiment **completed** all 22 frozen stages with valid one-time TEST scoring. Its unchanged affine-KLT tracker obtained 60.5% visible PCK@8, 61.0% visible coverage, 95.0% accepted visible precision@8, 0.423 scene-macro Jaccard and, critically, **36.0% incorrect acceptance while a queried point was invisible**. SmolVLM2-2.2B DEV balanced accuracy was 35%, below the predeclared 55% bar, so the TEST VLM was correctly skipped.

Batch46E TEST is now **open**. No method, threshold, fallback, sample, model or admission decision for Batch48 may be selected by looking at its TEST answers or its post-hoc ranking. Batch47D/E's motion evidence came from six already-used DEV clips and synthetic image-plane warps; that evidence is a diagnosis, not a replacement for a new sealed evaluation.

## Frozen research protocol

1. Identify 12 Kinetics-700/TAP-Vid-Kinetics natural videos with **unique video IDs and unique action classes**, six DEV then six TEST. Exclude the identities and classes in the audited Batch42, Batch45E and Batch46E manifests. Missing/malformed history or cached selection overlap is fatal. Read historical *metadata* for exclusion only; do not access any historical private answer file.
2. Select exactly four official first-visible point queries per clip through the reproducible hash-based query selection algorithm. Preserve the Kinetics 25-fps, 256×256, up-to-250-frame protocol and auditable half-pixel coordinate mapping.
3. Evaluate the fixed CPU candidate set on **only the new Batch48 DEV**: KLT FB, template, affine KLT and pre-existing appearance/gradient gates. The risk budget is predeclared as **visible coverage ≥30% AND invisible false-accept ≤10%**. Among eligible methods choose highest scene-macro Jaccard with coverage as tie-breaker. If none qualifies, use the fixed `appearance_fb_085` diagnostic fallback, explicitly marked **NOT DEV ADMITTED**; it must not be called safe. This fallback is a fail-closed reporting policy, not promotion by old TEST ranking.
4. Run the unchanged three-image SmolVLM2-2.2B DEV visibility reference under GPU exclusivity. Allow TEST VLM inference only if all existing DEV admission checks pass, including sufficient per-class support, coverage, validity, balanced accuracy and NOT_VISIBLE false acceptance.
5. **Pin SHA-256 of all TEST CPU predictions before opening TEST point/occlusion oracle**. If VLM is admitted, pin its TEST predictions before opening its TEST labels. Score the TEST once. Never change thresholds or select another method based on these TEST scores. Publish only aggregate numbers and caveats.
6. Inspect per-scene lighting and texture diagnostics, global affine *image-plane* registration and local track residual separately. These are not metric-scale camera pose, 3D ego-motion or calibrated background/foreground motion. Report false accept, coverage, accepted precision, unconditional visible PCK@8, scene-macro Jaccard, and measured CPU latency; CPU per-target timing excludes shared preprocessing/decode.

The DEV risk budget is a prospective **operating criterion**, not a data-derived cutoff calibrated on any previous TEST. Its numerical value is not claimed to be statistically proven safe. Six TEST videos are a small shift slice: aggregate averages are descriptive, not a deployment safety guarantee.

## Frozen automation and release

The 22 stages are in `automation/plans/batch48_fresh_independent_safety_dev_select_2026-11-02.json`, with entrypoint `automation/runner_batch48.py`. R00 requires verified resumable shard downloads. R01/R02 run targeted/full tests; R03/R04 freeze and audit data; R05–R07 complete DEV tracker selection; R08–R10 make GPU-gated VLM DEV decisions; R11/R12 seal TEST predictions; R13/R14 perform one-time scoring; R15–R21 build report, revalidate pinned evidence, rerun regressions and security audit.

`experiments/kinetics_batch48_data.py` uses an isolated copy of the audited Batch45 data module rather than changing its global state. A new preflight guard verifies the 36 historical video IDs/action labels before any shard transfer, and again after dataset preparation. New `tests/test_kinetics_batch48.py` regressions cover missing history, old-video/class leakage, duplicate/split defects, isolation, DEV-only constrained method choice, unsafe fallback, frozen order and independent report recomputation.

All raw video shards, frames, tracks, oracle labels, predictions, weights and credentials remain on Tang in ignored local directories. GitHub stores **only** protocol, configuration, source code and tests. The dataset download is network-bound and can take many hours; do not report a TEST result before the predicted and scored artifacts actually exist. On any failure preserve partial data and logs and change to a **new** frozen SHA/plan before restarting, never force-override state or silently retry a failed stage.
