# Batch 70 — Prospective long-occlusion identity-stress results (2026-11-03)

## Status, provenance and limits

**All six sealed stages ran successfully on Tang** (prepare → label-blind DEV predict → DEV score → label-blind TEST predict → one-time TEST score → final immutable audit). The preregistered source is the local frozen commit `ec183c7c620537115393d7da49d3280f30b2211f`, with source **Tree SHA** `057fd8e78b5201598329d88538f080df062c0646`. The **same exact tree** was independently committed and verified on GitHub `main` as `f782661b200ed3ed2c394df19196e82f2a7f71a9` **before any data generation or evaluation**. Distinct commit IDs are due to the GitHub publication path; byte-identical tree was verified. This results-only report was published after the complete sealed experiment and changes no source, tests or parameter.

This is a new, seed-disjoint **procedural 2D grayscale** benchmark, **not independent natural-video evidence** and not a 3D camera-motion experiment. We generated 28 scenes of 96 frames: 16 DEV and 12 previously unexamined TEST scenes, across four frozen conditions. Every scene has exactly 63 occluded evaluation frames (frames 18–80), 32 visible evaluation frames (frame 0 is the query and excluded). Public video frames and private oracle were immutably pinned before any prediction. TEST prediction was saved and SHA-fixed before the TEST oracle was first opened for its single score.

The question was whether an exact pixel-texture impostor *placed on an occluding layer* could reproduce the natural-DEV shared-identity failure. **It does trigger long accepted hidden episodes in template recovery, KLT and affine-KLT, but does not fool the conservative appearance tracker or appearance/template consensus on this generator.** Hence it does **not** reproduce the natural-DEV shared failure observed previously.

## DEV and prospectively frozen TEST scores

Counts are pooled over independent scenes. “Hidden accepted” is an error even if the coordinate numerically overlaps the true point: the oracle point is not visible. “Visible accepted” is coverage, while PCK@8 is accurate visible predictions divided by all visible frames (not just accepted). Full-visible PCK@8 equals visible coverage for most methods here, except the DEV affine case.

| Frozen method | DEV visible accepted/512 | DEV PCK@8/512 | DEV hidden accepted/1008 | TEST visible accepted/384 | TEST PCK@8/384 | TEST hidden accepted/756 | TEST worst consecutive hidden accepts |
|---|---:|---:|---:|---:|---:|---:|---:|
| `appearance_fb_055` | 302 (58.98%) | 302 (58.98%) | **0 (0%)** | 204 (53.13%) | 204 (53.13%) | **0 (0%)** | **0** |
| `template_consensus_4px` | 287 (56.05%) | 287 (56.05%) | **0 (0%)** | 204 (53.13%) | 204 (53.13%) | **0 (0%)** | **0** |
| `template_recover4` | 452 (88.28%) | 452 (88.28%) | **756 (75.00%)** | 339 (88.28%) | 339 (88.28%) | **567 (75.00%)** | **63** |
| `klt_fb` | 332 (64.84%) | 332 (64.84%) | 630 (62.50%) | 219 (57.03%) | 219 (57.03%) | 441 (58.33%) | **63** |
| `affine_stabilized_klt` | 362 (70.70%) | 347 (67.77%) | 567 (56.25%) | 294 (76.56%) | 294 (76.56%) | **630 (83.33%)** | **63** |

### Crucial paired observations

- Template recovery on TEST accepts **all 189/189 hidden frames in each of the three matched-identity conditions**, and **0/189 hidden frames** for the unmatched-occluder control. Its 567/756 aggregate hidden mistakes are entirely from the matched-identity regimes. This deliberately constructed contrast provides a controlled stress test of duplicated local appearance, **not causal proof about natural-world identity failures**.
- The conservative appearance tracker rejects **all 756/756 hidden TEST frames**, but covers only **204/384 (53.13%)** visible TEST frames. The two-tracker template consensus makes no additional TEST difference (same coverage and hidden errors), while losing 15 correct visible frames on DEV for no further hidden-error reduction. Thus no gain from deploying an extra gate is demonstrated.
- KLT and affine KLT still produce many 63-frame continuous mistakes, **including some unmatched-occluder scenes**. Their errors therefore cannot all be attributed to texture duplication alone; persistence and motion/occlusion confusion remain competing explanations.
- The **natural Batch 56 DEV 117-frame shared failure remains unreproduced** in this prospective synthetic generator. It is incorrect to interpret a 63-frame template-only failure as independent confirmation of that natural event.
- All parameters were inherited from Batch 56, fixed in the preregistration. No TEST-derived change to algorithms, scene generator, thresholds, data split or scoring was performed.

### Domain-stratified TEST counts

| Synthetic condition | Appearance visible/96 | Appearance hidden/189 | Template recovery visible/96 | Template recovery hidden/189 | Affine hidden/189 |
|---|---:|---:|---:|---:|---:|
| Exact impostor | 51 | 0 | 96 | **189** | **189** |
| 2D pan + impostor | 51 | 0 | 96 | **189** | **189** |
| Brightness shift + impostor | 51 | 0 | 96 | **189** | 126 |
| Unmatched blocker control | 51 | 0 | 51 | **0** | 126 |

These labels describe programmatically imposed image conditions. Synthetic “camera pan” is exactly a known 2D background translation; it is **not calibrated camera pose or true 3D ego-motion**.

## Execution cost, tests and integrity

- Five frozen CPU methods including feature extraction and 2D affine estimation: TEST median **801.921 ms / 96-frame scene**, p95 **841.4364 ms**; DEV median **793.5855 ms**, p95 **822.0245 ms**. This is algorithm execution time on Tang, excluding loading frames, orchestration, serving, video decoding and VLM inference, **not end-to-end latency**.
- New Batch 70 targeted tests: **24 passed**. Complete project regression: **1,250 passed, 1 historical Batch64 artifact-dependent test skipped, 3 existing Pillow warnings**; no test failures.
- Frozen public manifest: `0ff6236a98d3224e7b22ea86bb711a6face315679929a6a9497cd09240a9b168`.
- DEV predictions: `9a2a12275110fd5590f795e1ee0acbb2b73676f19b386f268900ed3f380d1816`.
- DEV score: `fb166164f63b58ad734f0c42f8ec0d3e9e2f8b8ecc22e755d4895ea002e3818f`.
- TEST predictions pinned before scoring: `35fe2badcb52d345b4f7192bc12f9797210892be0c9dac53625aab94eda3b1c9`.
- One-time TEST score: `e5268a3a72a3fbb70a711196e9e8e945010d217fd3f01ced3f539b34617043c6`.
- Final offline audit: `BATCH70_FINAL_AUDIT_PASS ec183c7c620537115393d7da49d3280f30b2211f`.
- Complete logs: Tang `repo-batch70-dev/outputs/batch70_shared_identity_v1/stage-run.log`. Media and oracle: Tang `data/batch70_shared_identity_v1/`. Immutable prediction/score artifacts and their `.sha256` tags: local `repo-batch70-dev/outputs/batch70_shared_identity_v1/`. No weights, private labels, videos or secrets are committed.

## Scientific conclusion, deployment and next bottleneck

**No deployment.** Batch 70 falsifies the claim that stable template recovery necessarily guarantees visible identity under long occlusion: an exact cloned appearance on an occluder yields a **63-frame continuous wrong acceptance**. However, the deliberately controlled synthetic scene does **not** break both conservative trackers together and is not representative enough for natural safety generalization. Additional synthetic TEST inspection must not be used to retune the generator as though TEST remained untouched.

The critical independent evidence is still Batch 59's new, category-disjoint natural-video evaluation, which was left unchanged. Pending that protocol, prioritize DEV-only investigation of external segmentation/point visibility evidence and real temporal identity reacquisition; admission criteria must jointly cover worst-scene/event errors, visible coverage, PCK, CPU costs and VLM baseline. Batch 59's GPU lock, frozen SHA `8e079c70...`, 10-minute marked cron and all earlier downloaded archives/logs remain unmodified.
