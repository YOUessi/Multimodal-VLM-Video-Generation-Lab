# Project 4 — verified evidence release (2026-10-31)

## Repository and local release
- Repository: `YOUessi/Multimodal-VLM-Video-Generation-Lab`.
- Frozen Tang local source snapshot: `54e472b288776058206aab063027048c4c45a231`; upstream freeze `b4425bb1ba0813344e5c30ab555fd282b3968357`.
- Status: batch15 stages B70–B73 **4/4 completed with exit code 0**. Tang source tests: **51/51 passed**.
- Local offline project showcase: `~/projects/Multimodal-VLM-Video-Generation-Lab/repo-batch15-tested/outputs/release/index.html`.
- Machine-readable audit: `outputs/release/EVIDENCE_MANIFEST.json` and `outputs/release/SOURCE_PUBLICATION_AUDIT.json`.
- Repository source audit: 74 tracked paths, **zero model/media/secret binaries**, required ignores in place.
- Source compilation succeeded for `experiments/`, `automation/`, and `tests/`.

## Verified experiments (not model advertising)
- STAR action-sequence exact SHA-group split: **200 filenames, 193 unique byte contents, 7 duplicate groups, 0 cross-split duplicates**, fresh 120/40/40 SFT results. LoRA vs base: 1 frame 15/40 vs 17/40; 3 frames 17/40 vs 16/40; 5 frames 20/40 vs 18/40. All paired confidence intervals include zero.
- External MVBench `scene_transition`: all 200 publicly labeled clips, no full-byte overlap with STAR, no training; LoRA vs base: 3 frames 155/200 vs 155/200; 5 frames 160/200 vs 157/200. No statistically established external gain.
- Repeated-video frame selection feature cache: on the grouped 40-video split, **2.00× cold cache and 14.95× warm cache** mean speedups versus the uncached pair of 3-frame and 5-frame requests; all selected RGB bytes identical.
- ZeroScope v2 576w: **six actual 576x320, 12-frame generated MP4s** at 8/14/20 steps, all verified locally. Wall-time approximately 11–25 seconds. Video quality has **not** been subjectively scored; no Wan generation occurred.
- The initial STAR 40-sample ID-only test suffered cross-split exact duplicate byte contents. Those historical scores are **invalid** as independent evidence and excluded from this release's authoritative evaluation.

## Source code and scope restrictions
- The release preview on Tang loads local generated MP4s from `outputs/diffusion_grid`; those MP4s are not published in GitHub.
- Original real video datasets, weights and saved PEFT adapters remain on Tang.
- Model-asset licensing differs from the source-code repository (ZeroScope is research/noncommercial).
- This repository currently contains the tested GPU experimental scripts, orchestration and regression suite. The earlier stand-alone annotated-CPU prototype archive (31 tests) is still a distinct artifact and has **not** been imported as `src/mvvg`.
- Perceptual near-duplicates and possible foundation-model pretraining contamination remain open; exact content hashes alone do not solve these.

## GitHub CI
A new public-source GitHub Actions job was added to compile sources, install CPU-only test dependencies, run the regression suite and reject model/media binary commits. The initial run failed because its clean runner lacked `ffprobe`, which Tang already had. The workflow has been corrected to install FFmpeg; CI completion must be verified independently (do not infer green status from Tang tests).

## Important next work
1. Import the separate CPU prototype into the single GitHub repository and run its full smoke/annotation/test pipeline alongside current experiments.
2. Obtain independent human blind quality reviews of the generated clips, including prompt alignment, visual quality and temporal coherence. Never invent scores.
3. Add perceptual near-duplicate checks (beyond SHA256 exact duplication), and larger independently verified real-video evaluation without tuning on the heldout tests.
4. Consider a low-VRAM licensed Wan path only after resource preflight; the current generation result is ZeroScope, not Wan.

GitHub evidence: [STAR SHA-grouped evaluation](STAR_GROUPED_SFT_RESULTS_2026-10-31.md), [external real-video transfer](MVBENCH_EXTERNAL_SCENE_TRANSFER_2026-10-31.md), [cache speedup](SHARED_CACHE_GROUPED_RESULTS_2026-10-31.md), [diffusion step grid](BATCH10_RESULTS_2026-10-31.md).
