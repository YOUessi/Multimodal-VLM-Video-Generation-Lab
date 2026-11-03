# Batch 73 — Prospective singleton-impostor identity falsification results

Date: 2026-11-03. GitHub source of truth: `YOUessi/Multimodal-VLM-Video-Generation-Lab`, `main`.
**Frozen algorithm + generator + tests SHA:** `d0432fc351879a6d6887d86f2676bb8082915987`.
**Release decision: FAIL — no deployment of identity-margin-only reacquisition.**

## Research question

Batch 72 reported conditional success with a unique query-patch appearance cue. Batch 73 intentionally separates **appearance uniqueness** from **persistent physical identity**: a single object appears after an opaque interval, and is either the original target or an indistinguishable impostor. The two world assignments produce the same bytes in all 96 frames. A best-vs-runner-up visual similarity margin is therefore high even when **there is only one wrong candidate**.

This is newly frozen independent-seed **procedural synthetic video**, **not** new natural video, a 3D camera dataset, or natural safety certification. The already running Batch 59 natural experiment was untouched.

## Audited provenance

- Initial GitHub `main`: `2628d5af68b62d0fab6f797095f64594b460d85e`.
- New source, tests and written protocol committed and pushed **before generation / TEST scoring** as `d0432fc351879a6d6887d86f2676bb8082915987`. Upstream `main` confirmed with `git ls-remote`.
- Regimes: exact singleton, synthetic planar pan + gain, Gaussian blur, and local retexturing. Data: **4 regimes × (3 DEV + 2 TEST) independent pairs × 2 latent worlds = 40 videos**. Each 96 × 256 × 256 grayscale.
- Uniqueness and non-overlap checked: DEV **12 pairs / 24 videos**; TEST **8 pairs / 16 videos**; worlds are correlated and must **not** be counted as independent replicates.
- Occlusion: frames 18–63 inclusive (46 frames). Post interval: 64–95 (32 frames). Query target visible in frames 0–17. One post-occlusion visible candidate in each public video.
- Label-blind inference and scoring are isolated: public frame hashes and query positions pinned before prediction, private latent identity is never read during predictions. Predictions are pinned before any corresponding scoring. TEST scoring was executed **once**, after pinned TEST predictions.
- Source config and artifact SHA-256 audited at every step. Final audit: `BATCH73_FINAL_AUDIT_PASS d0432fc351879a6d6887d86f2676bb8082915987`.
- Manifest SHA-256: `88f91fbf0b6c1e6b0356cc1bbe45596c3143790c0e850b36ff7feacfe9446dbb`.
- DEV predictions: `01442b3c3343b3da3602f2f6cffb8f7e6a6690965e45d1f045d004ab3361e865`.
- TEST predictions: `5409f782a7ead2c2d1a7af18491c975e13cfcf3656140e4374a255d2606c9c76`.
- TEST results: `86eaf4d8a45f17495c5441660aa5bfcacf4b732ffa46c5103b713603c32c9329`.
- Source modified after frozen TEST? **No**. The report-only commit must not alter code or rules.

## New independent TEST result

All four existing methods and their parameters were fixed before any TEST ground truth was scored.

| Method | Genuine target correctly recovered (visible world) | False identity accepted (absent world) | Opaque-interval false accepts | Longest post-absence false-accept run |
| --- | ---: | ---: | ---: | ---: |
| Conservative `appearance_fb_055` | 0 / 256 | **0 / 256** | 0 / 736 | 0 |
| `template_recover4` | 0 / 256 | **0 / 256** | 0 / 736 | 0 |
| `best_template_score` | **256 / 256** | **256 / 256** | 0 / 736 | **32 frames** |
| `identity_margin_veto` | **256 / 256** | **256 / 256** | 0 / 736 | **32 frames** |

Each genuine-positive frame has a paired, bit-identical absent-world frame. Thus, the new margin gate succeeds in the present world **and fails on the physically absent target in every paired scene**. Its 256/256 genuine recovery is **not** identity verification. It is exactly balanced by 256/256 physically wrong acceptances; it cannot distinguish whether a unique-looking candidate is truly the original target.

All four TEST regimes contribute **64 correct-present / 64 wrong-absent** frames per patch-matching method. Both older conservative methods abstain throughout the post interval, preserving safety but missing all possible reacquisition. Neither wins an overall success criterion.

All methods accepted 0/736 frames of the opaque interval. **This metric is insufficient:** it misses the post-occlusion impostor entirely.

## DEV results (not used to tune any parameters)

DEV totals: for both patch methods, **352/384** genuine post-visible correct and **352/384** post-absent false acceptances. In the retexturing regime they accepted 64/96 on each side; the other regimes each achieved 96/96 on both sides. Legacy trackers scored zero on both measures.

These historical-protocol comparisons do not turn the TEST into a parameter selection exercise. There was no method alteration between DEV and TEST, and only one TEST scoring pass.

## Non-identifiability explanation and bound

For each pair and time step after occlusion, the public input (I_{0:t}) and query (q) are identical in the two worlds. A deterministic prediction (\hat{x}_t = f(I_{0:t}, q)) must therefore be identical. If it accepts the sole candidate, it may be correct in the `present` world and wrong in the `absent` world; if it abstains, it abandons recovery in both. With equal weighting of both constructed worlds and the physically distinct target, a pixel-only algorithm cannot guarantee identity correctness beyond random latent-world assignment.

The audit enforces the paired upper bound

```
2 * correct_post_present <= accepted_post_present + accepted_post_absent
```

and verifies byte-level equality of the paired videos and equality of all prediction traces. This is a **conditional information bound for these engineered worlds**, not a natural-video prevalence estimate.

## CPU cost

Measured per synthetic 96-frame video, single-thread OpenCV, from the pinned TEST predictions; video serving and VLM inference excluded:

| Component | p50 (ms) | p95 (ms) |
| --- | ---: | ---: |
| Legacy tracker/affine computation (combined) | 684.004 | 711.139 |
| Global best-template scan | 53.261 | 53.469 |
| Identity-margin scan | 54.069 | 54.221 |

The legacy and new columns cover different work; these are not equal-workload speed comparisons. In particular the VLM baseline was not run in Batch 73.

## Engineering verification and artifacts

- New targeted regression: **23 / 23 passed**.
- Entire project regression: **1330 passed, 1 skipped, 3 existing Pillow warnings**, no failures, completed before source freeze.
- Frozen stages actually executed successfully in order: `prepare`, `predict-dev`, `score-dev`, `predict-test`, `score-test`, `audit`.
- Logs: `repo-batch73-dev/outputs/batch73_singleton_impostor_v1/stage_*.log`.
- Generated frames / private truth: `data/batch73_singleton_impostor_v1` on Tang.
- Score artifacts: `repo-batch73-dev/outputs/batch73_singleton_impostor_v1` on Tang.
- No raw video, private oracle, weight, access token or secret is committed.

## Explicit negative conclusion and next research task

**Batch 72's positive appearance-margin result does not establish true physical identity.** A single plausible impostor defeats the uniqueness test even when there are no secondary candidate patches. Increasing a similarity threshold cannot distinguish these two observationally identical worlds; only independent identity-bearing information or principled unverified/abstention semantics could change the decision.

The stronger natural-video experiment, frozen Batch 59, remains the priority. It must measure scene-level and episode-level false acceptance, post-occlusion identity recovery/abstention, confidence, CPU time, distinct image-plane vs background-relative motion, and the frozen VLM reference. Its ongoing 42-shard download is not altered by this report. **No production algorithm change is authorized on this evidence.**
