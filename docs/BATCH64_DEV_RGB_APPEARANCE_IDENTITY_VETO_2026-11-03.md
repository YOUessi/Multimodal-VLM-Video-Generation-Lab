# Batch 64 — DEV RGB appearance identity as an independent occlusion veto (2026-11-03)

## Status and provenance

**Outcome: FAIL to solve persistent occlusion; do not deploy.** This is a retrospective, explicitly exploratory analysis on the **previously studied** 36 DEV videos from Batch 56, not a new blind TEST or an independent statistical confirmation. The frozen Batch 59 experiment continues unchanged and is not read by this evaluation.

* Source: GitHub `YOUessi/Multimodal-VLM-Video-Generation-Lab`, parent SHA `d9e9f814aa665dece15e3b508f0290d861474baf`.
* Input DEV predictions: Batch 56 frozen `dev_predictions.json` with its SHA-256 sidecar; DEV evaluation/selection pins rechecked, 36 DEV scenes and four queries each.
* Input image evidence: Batch 56 frozen RGB frame arrays from `frames/kinetics_00.npy` through `frames/kinetics_35.npy`; every file SHA-256 verified against the **public** frozen manifest.
* Query data: `queries-public.json`; only the DEV prefixes are consumed.
* **Phase 1** extracts query-reference RGB and predicted-location RGB histograms without opening any oracle labels, DEV scoring results or DEV method-selection artifact. The feature object is written in exclusive-create mode and SHA-256 pinned.
* **Phase 2** verifies the feature pin, then loads the DEV-only prefix of the oracle through the existing tested DEV firewall. The private oracle reader stops at `kinetics_35` without reading or parsing a TEST object. The report is also exclusive-create pinned.
* Neither phase reads Batch 56 TEST predictions/labels; neither phase accesses Batch 59 TEST or alters any prior frozen tracker or cron.

## Hypothesis and offline procedure

Batch 62 demonstrated that spatial tracker agreement reduces average occluded false acceptance but misses a persistent 117-frame shared-drift episode. Batch 63 showed requiring two or three successive agreements has little additional safety benefit. Batch 64 asks whether **additional, visually observable, source-image appearance evidence** can reject that shared error without oracle-dependent motion residuals.

At the query frame, take a 15×15-pixel patch centered on the published `query_xy`. At each following frame where the frozen `appearance_fb_055` tracker proposes a coordinate, sample the corresponding RGB patch. Compute a normalized 8-bin-per-channel color histogram for each patch and its histogram intersection (mean of channel-wise overlaps, within [0,1]). No image information from future frames is used in an earlier decision; the reference is never adapted with future information. An unavailable or out-of-bounds patch causes conservative abstention.

Three **predeclared, descriptive** thresholds are 0.55, 0.70 and 0.85. Compare baseline, the established one-frame template-consensus veto, RGB veto alone, and template+RGB veto. All are strictly rejection-only; none changes an estimated coordinate. Thresholds are not selected by private TEST; this is a diagnostic threshold grid, not evidence of an optimal production setting.

Image-plane pixel resemblance can correlate with the source point's appearance, but **does not prove object identity, visibility, 3-D motion, background identity or true camera motion**. Large flat surfaces or background patches of similar color can fool this signal; occlusion and target pose or illumination changes can also invalidate a correct track.

## Actual results — scene macro metrics, 36 DEV scenes

| Variant | Visible coverage | Occluded false-accept rate | Scene-macro Jaccard | Scenes exceeding 10% (of 33 with occlusion) | Longest false-accept episode |
|---|---:|---:|---:|---:|---:|
| Original frozen `appearance_fb_055` | 45.36% | 5.12% | 0.3227 | 4/33 | 117 frames |
| Template agreement, 4 px | 34.09% | 1.165% | 0.2695 | 1/33 | 117 frames |
| RGB only, ≥0.55 | 44.80% | 4.63% | 0.3210 | 4/33 | 117 frames |
| RGB only, ≥0.70 | 40.98% | 2.72% | 0.3016 | 3/33 | 117 frames |
| RGB only, ≥0.85 | 28.29% | 1.17% | 0.2221 | 1/33 | 117 frames |
| Template + RGB ≥0.55 | 33.70% | 1.039% | 0.2671 | 1/33 | 117 frames |
| Template + RGB ≥0.70 | 31.33% | 0.946% | 0.2501 | 1/33 | 117 frames |
| Template + RGB ≥0.85 | 23.86% | 0.895% | 0.1966 | 1/33 | 117 frames |

These are scene-macro averages, not pooled-frame rates. Macro occlusion denominators exclude three DEV scenes that have no annotated occluded frames; frames are not treated as independent statistical samples.

Compared with template-only, template+RGB 0.55 removes **eight** additional wrongly accepted occluded frames but loses **94** previously correct visible frames. Tightening to 0.85 removes **18** additional occluded false accepts while losing **2,051** correct visible predictions. As a result, scene-macro Jaccard worsens. The original problematic scene `kinetics_17` remains at **26.43%** false acceptance under every template+RGB threshold, with an unchanged **117-frame** continuous wrong-accept episode. Even RGB alone at 0.85 leaves its false-accept rate at approximately **27.57%**.

RGB feature extraction took approximately **195 ms summed across all 36 DEV scenes**, with a median of about **3.90 ms per scene** on the research workstation. This is a **new offline feature-only wall-clock measurement**, excluding decoding, tracker execution, process setup, artifact hashing, system orchestration and full end-to-end CPU latency. Do **not** equate it to deployed per-frame latency; do not compare directly against the earlier additive per-target-frame CPU proxies.

## Interpretation, falsification, and next research action

A static reference-patch histogram is insufficient to detect persistent wrong tracking. The failure is consistent with visually similar distractors or spatially broad homogeneous colors; those are **hypotheses**, not demonstrated causal mechanisms from this data alone. The residual high-risk scene requires new independent visibility/identity information, such as multiple target/background descriptors and visibility-specific cues, with safeguards against co-drifting on the same foreground surface.

Do not select a production threshold on the already opened DEV/TEST metrics. For a future fresh, category-disjoint, pre-registered experiment, freeze the algorithm, CPU/coverage/occlusion criteria and data split before any new TEST scoring. Include per-scene worst-case errors and long-error episodes, not just mean false acceptance.

## Reproduction

From the repository checkout on Tang, with existing pinned Batch 56 DEV data available:

```bash
../.venv/bin/python -m pytest -q tests/test_batch64_dev_rgb_identity.py
../.venv/bin/python -m experiments.batch64_dev_rgb_identity features
../.venv/bin/python -m experiments.batch64_dev_rgb_identity score
../.venv/bin/python -m pytest -q
```

An earlier v1 exploratory artifact remains preserved locally. The v2 feature and scored report are **immutable on first write**; reruns should use a new independent research checkout/output directory, rather than overwriting pinned results. Only three source/test/doc files are committed; no videos, model weights, private oracles, feature result JSON, tokens, or private media are committed.

Pinned local (uncommitted) artifacts:

* `outputs/research/batch64b_unlabeled_dev_rgb_identity.json` SHA-256 `0c291760de84fe09b8f31758954c818000b36487ec70120bc58f2f372b714929`.
* `outputs/research/batch64b_dev_rgb_identity_report.json` SHA-256 `dce060a9ca4a069a84c44afc3441a703951d9feac11d936cff0c50829de1ee96`.
