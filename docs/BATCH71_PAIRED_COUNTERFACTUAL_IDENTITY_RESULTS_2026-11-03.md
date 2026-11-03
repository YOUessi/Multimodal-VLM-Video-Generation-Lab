# Batch 71 — Prospective paired-world identity observability results (2026-11-03)

## Execution and provenance

**All six sealed actions completed successfully on Tang:** prepare, label-blind DEV prediction, DEV scoring, label-blind TEST prediction, **one-time** TEST scoring, final SHA audit. The source was frozen **before any data generation** in local commit 6d59a56684fa386f427d3625ac35c6cdfe81b2db, **Tree SHA 28e1b1713282d818c9b85a79e9e8f84d3da6841b**, and byte-identically published to GitHub main as e93ab4373e237824b3013c7a60f1066c85664a96, with all three new Git blob SHAs and the complete Tree SHA verified. Different commit identifiers stem from the publication path.

The study has 15 fresh, independent, paired **synthetic** video scenes: nine DEV pairs and six newly unseen TEST pairs. Each 96-frame clip has two possible, **pixel-identical** latent identity assignments; the same public frame sequence and exact same query are passed to the tracker **only once per pair**. A two-sprite opaque occlusion occupies frames 18–63, and the two physically possible query identities reappear at widely separated points from frame 64 onward. Both worlds were hashed separately before any prediction. The renderer includes static, 2D planar background translation and lighting variants. There is **no real-world natural video, no calibrated camera-ego-motion ground truth, no VLM run, and no GPU training** in this experiment.

## Actual sealed DEV and TEST performance

The table counts independent scene pairs, not latent worlds as independent samples. Every method uses the original frozen Batch56 tracker parameters without threshold tuning, and the same 5 methods as Batch70.

| Frozen method | DEV pre-visible correct / 153 | DEV hidden accepted / 414 | DEV post-visible accepted / 288 | TEST pre-visible correct / 102 | TEST hidden accepted / 276 | TEST post-visible accepted / 192 | TEST post-reappearance random-world PCK@8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| appearance_fb_055 | 153 | 0 | **0** | 102 | 0 | **0** | **0%** |
| template_consensus_4px | 153 | 0 | **0** | 102 | 0 | **0** | **0%** |
| template_recover4 | 153 | 0 | **0** | 102 | 0 | **0** | **0%** |
| klt_fb | 153 | 0 | **0** | 102 | 0 | **0** | **0%** |
| affine_stabilized_klt | 153 | 0 | **0** | 102 | 0 | **0** | **0%** |

All five methods **accepted every pre-occlusion visible point** and correctly localized it at PCK@8; all five **abstained during every hidden frame**. But **all five also abstained during every post-occlusion visible frame**, in every one of the three frozen regimes. Their hidden-error rates and longest hidden-error runs are both zero here, but their post-reappearance coverage is also zero. This is a **complete reacquisition failure**, not evidence that they reliably distinguish the two identity worlds or safely recover after occlusion.

**Important negative result:** the frozen trackers produce **no post-occlusion accepted predictions at all**, so this particular generator does **not empirically demonstrate a wrong-world identity switch**. It instead exposes a distinct usability failure: complete, persistent abstention after ambiguity. The synthetically challenging hidden period is sufficient to lose the tracked point; current methods have no mechanism to restore both useful coverage and reliable identity. It is not legitimate to retune this sealed TEST generator or tracker thresholds to force more post-reappearance accepts.

## Independent structural identifiability result

For every t in 64..95, two different latent histories yield **the exact same RGB/grayscale video and query**, but assign different positions to the queried object. The positions are separated by more than 16 px (2 × the 8-pixel PCK radius). For any accepted point predicted from those identical pixels, its PCK@8 can be correct under **at most one** of the two worlds. Consequently:

PCK@8_random_world_post ≤ 0.5 × post_coverage.

Equivalently, if the two hidden identity assignments are equally plausible, among accepted post-reappearance predictions the PCK@8-incorrect fraction must be **at least 50%** in the best case when any point is accepted. This can be an identity mismatch or a localization error. This is an identifiability bound for the deliberately symmetric construction. It is not a measurement of 50% error in nature, and the conditional bound is vacuous when post_coverage = 0, as occurred for all measured baselines.

This establishes why 2D pixel correspondence, appearance similarity, locally consistent optical flow or a long trajectory alone cannot certify object identity in perfectly ambiguous scenes. Correct behavior requires calibrated abstention, additional disambiguating information, or a well-defined new query/identity-reacquisition contract.

## CPU, tests and integrity

- Combined CPU wall time for **all five methods per 96-frame independent paired scene**, excluding data loading and generation, decoding, serving and VLM: DEV p50 **702.765 ms**, p95 **750.4578 ms**; TEST p50 **707.8705 ms**, p95 **764.3885 ms**. Both latent worlds reuse **one** prediction; they do not double runtime.
- Batch71 targeted regression: **21 passed** after fixing an in-memory private-oracle aliasing bug found during initial tests (initial result 20 passed, 1 failed). The corrected source was committed only after the fix and a clean retest.
- Full project regression before freeze: **1271 passed, 1 skipped, 3 existing Pillow warnings; exit 0**. Skipped test depends on a historical Batch64 experiment artifact not present in this workspace.
- Frozen public manifest SHA-256: b655be5b461eef5d33add3cf6570845770af182b1b2bf57e79b00233079fbb0b.
- DEV prediction SHA-256: 90b5ad57d49bdf66f35429201166a1fa5db5a28b2c8424292c95a7a4b861e244.
- DEV score SHA-256: 7c8b3535e57159b6c134f115e0e6728cb6a6057a07cfcc411799b995c38eecc9.
- TEST prediction SHA-256 pinned **before scoring**: 4d7969e3359bfc1b41b2099c85c211f5a2d61c893d04df7d7bae65e46e368e09.
- TEST one-time score SHA-256: cf7aa5e3e032c0343a2e82f72c4f9a5866921cc0ea8ef7c713da67ec24b81bd0.
- Offline final audit output: BATCH71_FINAL_AUDIT_PASS 6d59a56684fa386f427d3625ac35c6cdfe81b2db.

Frozen logs are retained at Tang repo-batch71-dev/.batch71-stages.log and .batch71-full-regression.log. The ~91 MB private/oracle and public-generated media stay on Tang under data/batch71_counterfactual_identity_v1; frozen scores and their hash tags stay in repo-batch71-dev/outputs/batch71_counterfactual_identity_v1. **No videos, weights, private annotation files, personal data or secrets are committed.**

## Scope and next research bottleneck

**No deployment.** This synthetic TEST was evaluated only once and is closed to method selection. The hard information-theoretic ambiguity bound is real for this construction, but the particular evaluated methods all abstain after occlusion and cannot be credited with solving identity reacquisition. Future DEV-only algorithm work should focus on distinguishing **scene-level visibility** from **recoverable object identity** and explicitly distinguish safe abstention from usefully correct tracking. An external identity cue, a re-anchoring interaction or an informative sensor may be necessary in genuinely symmetric scenarios.

The highest-priority independent natural-video evidence remains Batch59, which is still running its frozen R00 download; this batch does not touch its SHA, data, plan, cron, GPU lock or scoring protocol. A subsequent prospective natural-video evaluation needs post-occlusion reacquisition coverage, PCK, occlusion-event errors, longest hidden errors, actual CPU cost, and VLM admission simultaneously. A new frozen experiment is warranted only after isolating a DEV-valid candidate, not by searching the now-exposed TEST answers.
