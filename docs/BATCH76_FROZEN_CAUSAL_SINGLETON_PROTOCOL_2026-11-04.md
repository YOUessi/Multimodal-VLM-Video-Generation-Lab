# Batch 76 — Causal singleton re-detection on historical natural DEV (frozen protocol)

Date: 2026-11-04. Repository: `YOUessi/Multimodal-VLM-Video-Generation-Lab`.

## Hypothesis and permitted interpretation

Batch75 found that the selected conservative baseline reacquired a spatially correct target after only 5 of 150 evaluable occlusions; combined veto recovered only 1/150. An apparently unique local patch match after a long abstention is a tempting rescue mechanism. Batch73 proved that a visually identical singleton impostor can defeat an identity-margin gate, so this experiment explicitly measures **both recovered points and newly introduced false accepts**. It is a *falsification attempt*, not a production candidate. No claim of physical identity certainty is permitted.

All 36 previously studied natural Batch56 DEV scenes are exploratory. They are **not a new held-out natural TEST**. The entire independent Batch59 TEST run remains frozen and untouched. There is no tuning on Batch56 TEST or Batch59 labels, no GPU job and no video/weight upload.

## Inference-only preregistration

Use the existing SHA-256-pinned Batch56b public `dev_predictions.json`, public DEV queries and 36 named DEV RGB frame arrays. The inference entry point `--predict` never opens the private oracle.

For each of four point queries per video, use a fixed 17×17 grayscale query-frame template. Starting only after at least 12 consecutive baseline abstentions, search **every 3rd missing frame** using normalized cross-correlation, accepting a candidate only when its best match score is at least **0.86** and exceeds any spatially non-overlapping runner-up by at least **0.07**. A patch with grayscale standard deviation <4 is not searchable. No template update, oracle visibility, future-frame lookahead, or retrospectively accepted frame is allowed.

Compare exactly three methods: frozen `appearance_fb_055` baseline; baseline plus each accepted sampled singleton candidate (`singleton_raw`); baseline plus a candidate accepted after **3 consecutive sampled matches** with <=15 pixels movement per checkpoint (`singleton_3confirm`). Rescue can only add predictions where the baseline had abstained. If the baseline makes a prediction, it is preserved, even if it is wrong. This is a deliberately restrictive one-sided experiment; it cannot fix already-wrong accepted identities.

Output is sparse sampled positions, not dense tracking. The CPU number measures only `cv2.matchTemplate` wall time and excludes frame I/O, grayscale conversion, SHA verification, existing baseline inference and VLM.

## Freeze–predict–score sequence

1. Freeze this code, its regression tests and protocol on GitHub `main` before generating DEV redetection predictions.
2. Run `--predict` using only 36 DEV public frame arrays and public query locations. Write an immutable JSON and SHA-256 pin. Verify per-video frame SHA and manifest/query pins.
3. In a separate `--score` process, verify the prediction pin, then open exactly 36 DEV entries of the existing mixed oracle via the strict Batch62 parser, stopping before TEST. Score three methods without changing thresholds.
4. Compare baseline replay to Batch75 occlusion false accepts (317), event-level PCK@8 reacquisition and wrong-first acceptance, total visible coverage, false accepts during occlusion, longest post-occlusion wrong spatial run, and per-scene worst cases. Report negative results fully.
5. Run dedicated and full regression; publish results as a separate commit. Never replace frozen Batch59 or deploy a method from this exploratory analysis.

## Mandatory controls

- No private oracle during inference, no hidden TEST objects in the evaluator, no TEST scoring, no GPU contention.
- Pixel-level singleton uniqueness is **not** physical identity evidence; include a unit-test with identical *observable* pixels and an unobserved physical impostor truth.
- Matched frames and 144 queries are nested within 36 scenes, not 144 independent experiments.
- An accepted but spatially wrong visible point is *not* proof of a physical ID switch.
- Sparse rescue cannot be compared directly to dense temporal recovery as an operational tracker.
- Reject changed SHA pins, source/split drift, duplicate output files and any attempt to overwrite existing experimental artifacts.

**Deployment gate:** none. No results on these historical DEV videos are sufficient for deployment; new independent natural TEST must remain untouched until the original Batch59 stage arrives.