# Batch 76 — Historical natural DEV causal singleton re-detection: results

Date: 2026-11-04 (Asia/Shanghai); repository `YOUessi/Multimodal-VLM-Video-Generation-Lab`.

**Decision: exploratory analysis completed; DO NOT deploy.** New candidate inference works causally and improves post-occlusion *spatial reacquisition* on previously studied natural DEV, but its added acceptance also introduces new false accepts while the target is occluded. It is **not** a natural held-out TEST improvement or proof of correct physical identity.

## 1. Immutable source and prediction-first protocol

GitHub `main` freeze commit **`432bcf7bb67b40259cee53f9a69737aa23e82389`** was published **before** generating candidate predictions. Its Git Tree SHA is `19e8f934eb04703918e2cb9edc2d774e07005a2b`, independently matched in the isolated Tang worktree. The freeze consists of the algorithm, 31 tests and the protocol `docs/BATCH76_FROZEN_CAUSAL_SINGLETON_PROTOCOL_2026-11-04.md`.

The prediction stage read only the previously pinned 36 DEV RGB videos, their public queries and frozen public Batch56b baseline trajectories. All frame SHA pins were checked; candidate predictions were serialized and hash-pinned *before* any DEV oracle scoring. The score stage verified this prediction pin and streamed only the 36 named DEV oracle records, stopping before the hidden TEST entries. No TEST oracle/prediction was loaded or scored, and there was no model selection on TEST.

Immutable Tang-only outputs:

- `outputs/research/batch76/dev_public_predictions.json`: SHA-256 `c75501c6f70e627104297a8a40a5df5e46bc6d5d0dfab63368ddea87eda25189`.
- `outputs/research/batch76/dev_exploratory_results.json`: SHA-256 `e9194488db07a54ae8d25032f6c748be9f36e9d8ba9ba34de5fc28eaa7a32d41`.
- Unit / pre-freeze full / post-score full logs in `outputs/research/batch76/logs/` (not committed).

This experiment is **retrospective DEV exploration** on the same 36 natural scenes and 144 point queries already investigated in Batch56–75. There are 214 occlusion episodes, including 64 right-censored episodes, and 150 evaluable post-occlusion windows. No new independent natural videos were created here; the separately frozen Batch59 is still responsible for independent natural-video generalization.

## 2. Frozen methods

- **Baseline:** frozen `appearance_fb_055` predictions without modification.
- **Singleton raw:** only after 12 consecutive abstentions, search every third missing frame for a unique 17×17 query patch using normalized grayscale template matching (minimum score 0.86, margin 0.07 to the next nonoverlapping candidate). Accept only when the baseline abstains.
- **Singleton 3-confirm:** same features and thresholds, but require three consecutive sampled matches with <=15 pixels displacement between sampled predictions. No future-frame confirmation or dense backfilling.

All parameters were fixed before the candidate prediction stage and not changed after observing scores. A unique appearance match is not verified physical identity, as the controlled singleton-impostor test demonstrates.

## 3. Observed historical DEV results

| Metric | Frozen baseline | Singleton raw | Singleton 3-confirm |
|---|---:|---:|---:|
| Post-occlusion events reacquired (PCK@8) | 5/150 (3.33%) | 17/150 (11.33%) | **11/150 (7.33%)** |
| Occluded-frame false accepts | 317/10,797 (2.94%) | **357/10,797 (3.31%)** | **335/10,797 (3.10%)** |
| All visible frames accepted | 9,915/21,166 (46.84%) | 10,246/21,166 (48.41%) | 10,135/21,166 (47.88%) |
| All visible PCK@8-correct frames | 9,230 | 9,536 | **9,446** |
| Post-visible correct frames | 214/4,753 | 379/4,753 | **323/4,753** |
| Post-visible accepted frames | 287/4,753 | 465/4,753 | 396/4,753 |
| Post-visible spatially wrong accepted frames | 73 | **86** | **73** |
| Events with first accepted post-visible prediction wrong | 4/150 | **11/150** | **4/150** |
| Longest consecutive wrong post-visible predictions | 24 | 24 | 24 |

**Paired difference versus baseline:** raw adds 306 correct visible frames and 40 false accepts during occlusion, while 3-confirm adds **216 correct visible frames and 18 occlusion false accepts**. The 3-confirm method increases reacquired episodes by **6**, but fails the strict no-new-occlusion-false-accepts condition. It is therefore not safety preserving and must not be deployed.

The 3-confirm new occlusion false accepts are concentrated in just three DEV scenes: `kinetics_04` (+14), `kinetics_12` (+2), `kinetics_26` (+2). Conversely, correct visible-frame gains occur in nine scenes, dominated by `kinetics_27` (+92), `kinetics_13` (+45) and `kinetics_03` (+33). Frame counts are highly correlated within each scene; this is *not* nine independent validations of generalization.

The unchanged 24-frame maximum wrong-visible streak and unchanged 117-frame historical shared tracking-failure context show that a sparse one-sided recovery supplement cannot solve persistent accepted wrong identities. The identity-margin condition also does not disambiguate singleton impostors, as established by the synthetic counterfactual control.

## 4. CPU and validation

There were **5,765** actual OpenCV `matchTemplate` calls, totaling **3,288.3 ms** measured wall time, averaging about 0.57 ms per search call. This excludes frame hashing and loading, color conversion, existing baseline tracker cost, VLM inference and orchestration. It is not a total pipeline latency or real-time guarantee.

- Dedicated tests: **31/31 passed**.
- Full project regression after the frozen DEV scoring: **1,448 passed, 1 skipped, 3 existing Pillow deprecation warnings**, exit 0.
- Historical baseline replay: **317** occlusion false-accept frames exactly, and **5/150** successful recovery events, matching Batch75.
- All new predictions were computed and hash-pinned before private DEV scoring; hidden TEST score executions **0**.
- All results, logs, original media and oracle stayed on Tang; no weights, videos, private labels or secrets were committed.
- GPU was not used. Batch59's `8e079c70` SHA, cron, state, runner and data were not modified.

## 5. Research decision

**This is a useful but negative safety result**: a frozen, causal, public-only redetector can recover some additional historical natural DEV trajectories without changing existing accepted tracks, but its appearance-based singleton recovery still produces occlusion false accepts. Three confirmations reduce the added risk versus one confirmation, but do not eliminate it. Increasing coverage alone is not evidence of a safe identity-continuity mechanism.

Do not deploy, do not tune on already studied TEST, and do not fold this experimental method into the frozen Batch59 plan. The next independent natural test should examine optical/semantic target uniqueness, background-relative motion, scene-level worst-case occlusion failures, and reacquisition/abstention *jointly*, only according to its frozen evaluation protocol. If a future DEV-only candidate is proposed, require an explicit unsafe-impostor veto and a non-inferiority margin on occlusion false accepts before authorizing an entirely new sealed natural TEST.

Batch59 remains in `R00_SELECT_AND_DOWNLOAD_42_FRESH_CLASS_SHARDS`, 26/42 shards ready as of the closing audit (2026-11-03 UTC), with resume bytes progressing, no competing CUDA task, exactly one `MVVG_PROJECT4_10MIN` cron and approximately 69 GiB disk space free. Do not misinterpret an active long-running downloader as an `ALL_APPROVED_STAGES_DONE` batch or create a replacement frozen plan while it is active.