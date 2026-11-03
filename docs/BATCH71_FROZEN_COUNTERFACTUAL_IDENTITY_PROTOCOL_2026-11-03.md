# Batch 71 — Frozen counterfactual identity observability protocol (2026-11-03)

## Research question and scientific boundary

Does consistency in optical flow, template appearance, or trajectory recovery provide identifiable object identity when two physical identity assignments are compatible with precisely the same observed images? This is a deliberately adversarial, synthetic identifiability test. It is NOT new natural-video evidence and does NOT estimate real 3D ego-motion. Its positive result can show a fundamental information limit; it cannot quantify how often the limit occurs in natural video.

This batch does NOT modify the running Batch59 natural-video protocol, its GPU/cron/data, or any production tracker. It diagnoses the limitation of identity-only evidence exposed in Batch56 DEV and partially reproduced in Batch70 synthetic trials.

## Frozen paired-world generator and evaluation

Every independent scene is **one public 96-frame video**, one query coordinate at frame 0, and **two distinct private oracle identities**. Two indistinguishable 17×17 sprites are visible until frame 17. A fully opaque planar obstruction covers both sprites for frames 18–63. Upon reappearance at frame 64, both sprites remain pixel-identical and spatially separate, but the unseen identity assignment can be either (A) query identity reappears on the left or (B) query identity reappears on the right. The image sequence, query and occlusion flags are literally the same in both worlds; after frame 64 the two oracle point positions are more than 16 pixels apart. Both are physically admissible because the hidden sprites may exchange paths under the occluder; the video provides no visible evidence resolving the exchange.

There are 15 independent scene **pairs** (9 DEV, 6 TEST) in three frozen conditions: static background, known 2D planar background translation, and image brightness variation. Per pair, 17 pre-occlusion evaluated frames, 46 hidden frames and 32 post-reappearance evaluated frames (frame 0 is a query, not scored). Seeding and all scene geometry are defined in source. Pair identity worlds are mutually exclusive counterfactual hypotheses, **not** 30 independent videos or observations. Do not inflate sample size.

The 2D shift is known to the renderer and oracle only. It is not passed to the tracker, and it must never be described as calibrated 3D camera motion or natural-world ground truth.

## Frozen baselines and registration

Use existing Batch56 tracker configuration unchanged, with existing Batch70 frozen five CPU methods: appearance_fb_055, template_consensus_4px, template_recover4, klt_fb and affine_stabilized_klt. No tuning or variant choice from DEV or TEST; every configured baseline is reported. No VLM is run: this experiment is about missing information in pixels; Batch59 retains the independent natural-video SmolVLM2 baseline. There is no GPU competition.

Run strictly in this order:
1. GitHub-publish this source, tests, and protocol; record exact main SHA and source tree before data generation.
2. PREPARE: generate a unique public frame file per pair and two sealed private oracles. Pin each file and manifest by SHA-256, no overwrites.
3. PREDICT-DEV: only public frames and query; pin predictions before any DEV oracle read.
4. SCORE-DEV: read sealed DEV world-A and world-B truths and compute per-pair scores.
5. PREDICT-TEST: require a pinned DEV score; read only public TEST frames; pin once.
6. SCORE-TEST: read both TEST oracles **once** after TEST predictions have been sealed. NEVER rerun this stage to select a variant.
7. AUDIT: validate source SHA, frame hashes, both private world seals, and prediction/score provenance for both splits. Publish results-only documentation after full audit.

Prediction code must never open private oracle paths. The score rejects nonfinite predictions, altered source/config, split or scene drift, incorrect visibility, non-separated world identities, and any prediction claimed correct within PCK@8 of both post-occlusion identities. Keep failure logs and every pinned artifact.

## Primary registered metrics

- Pre-occlusion full-visible PCK@8 and correct accepted count.
- During-occlusion accepted fraction and longest continuous erroneous acceptance.
- **Post-reappearance acceptance/coverage**, PCK@8 under each counterfactual world, equal-world average and accepted-but-wrong fraction.
- The **theoretical bound**: for an accepted post-occlusion point, at most one world can count it PCK@8 correct because the two oracles are separated by more than 16 pixels. Thus with equally weighted admissible worlds, post-occlusion PCK@8 is at most half of post-occlusion coverage, even for a perfect local-template matcher. This is a hard identifiability bound, **not a fitted model result**. If every point is accepted, the accepted-but-wrong fraction is at least 50% under the two-world symmetry. Actual error can be worse.
- CPU p50/p95 per pair for all five methods, excluding file I/O, data generation, VLM, serving, and decoding.
- Results by regime and worst contiguous hidden errors. No production admission on synthetic results alone.

## No TEST tuning and deployment gate

The DEV split is only a completeness/sanity check. TEST is used once for final sealed evaluation. This benchmark does not provide a method for recovering unobservable latent identity; it tests the *need for abstention, external identity cues, or calibrated uncertainty* where physical worlds cannot be distinguished from pixels.

Even confirmation of the bound cannot establish typicality in natural video. Highest priority remains Batch59 fresh natural action-class-disjoint videos, with event-level error, worst-scene risk, coverage, relative motion and admissible VLM reference. No Batch59 runner code, freeze SHA, plan, GPU lock or cron may be changed by this experiment.
