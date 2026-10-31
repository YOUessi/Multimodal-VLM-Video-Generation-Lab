# Batch 30 — appearance-robust HSV object tracker and conservative VLM fallback (2026-11-01)

**All four approved stages M1–M4 succeeded** under frozen Tang source `a83fdcddd7baaba104b013b163760ec585e79c7a` (pinned GitHub upstream `7d262347e640c6f38a2ae0e7b36f2c038b10e0d0`). **103/103 tests passed**, source-security gate passed, no model weights or video binary uploaded.

## Controlled question and protocol

A prior fixed-RGB exact-palette tracker correctly solved 240/240 newly generated 2D synthetic direction clips, but on 200 CLEVRER simulated videos it could not produce a single valid target track: 82 unsupported color names and 118 insufficiently matched video pixels. Strict fallback did not degrade model accuracy but added CPU overhead and zero value.

This batch changes the tracking front-end **without using CLEVRER correctness labels for calibration**:

- Generic named color HSV templates with hue tolerance 15, minimum saturation/value bounds, brown/gray handling, background and component-size filters.
- Query-grounded target identification, component association across 3 chronological sampled frames, geometry-only displacement.
- A fixed predeclared gate accepts only confidently located **nonstationary diagonal** motions with |dx| and |dy| each >=20 pixels. All other cases are explicit abstentions; the system reuses frozen VLM output.
- The same tracker was **40/40 correct on the separate procedural development split** before this CLEVRER evaluation; no question/answer labels from the external source are read by segmentation.
- **All 200 original public CLEVRER moving-direction questions** are evaluated. No sample filtering or extra model inference. Existing SmolVLM2-500M/2.2B 3/5-frame answers are those frozen in batch19.
- This CLEVRER dataset **had been analyzed in earlier studies**. The experiment is exploratory, not a new blind evaluation. CLEVRER is a rendered scene benchmark, not natural camera video.

## Actual tracker coverage and accuracy

| Indicator on all 200 source questions | Result |
|---|---:|
| Frozen development source, before external scoring | 40/40 correct |
| External CLEVRER source questions | 200 |
| Tracker produced a direction answer | **119/200** |
| Tool answer correct over all 200, abstentions treated as unhandled | **103/200** |
| Accuracy among 119 attempted outputs | **103/119 = 86.6%** |
| Conservative gate accepted | **65/200 = 32.5%** |
| Correct accepted decisions | **58/65 = 89.2%** |
| Incorrect accepted decisions | **7/65 = 10.8%** |
| Median per-video CPU tracking time | **116.3 ms** |

Thus the tool can be useful but is **not** an infallible motion oracle. The 7 incorrect accepted predictions must not be erased from results.

## Gated tool + frozen VLM (no additional model inference)

| Frozen model | Sampled frames | Original VLM | HSV tool-gated VLM | Absolute gain | Paired bootstrap 95% CI pp |
|---|---:|---:|---:|---:|---|
| SmolVLM2-500M | 3 | 61/200 (30.5%) | **106/200 (53.0%)** | **+22.5 pp** | [+16.0,+29.5] |
| SmolVLM2-500M | 5 | 65/200 (32.5%) | **110/200 (55.0%)** | **+22.5 pp** | [+16.0,+29.0] |
| SmolVLM2-2.2B | 3 | 72/200 (36.0%) | **121/200 (60.5%)** | **+24.5 pp** | [+18.0,+31.5] |
| SmolVLM2-2.2B | 5 | 73/200 (36.5%) | **121/200 (60.5%)** | **+24.0 pp** | [+17.5,+30.5] |

Of the 65 gate-accepted items, the new tracker correctly identified 58. For the same accepted items, the frozen 2.2B VLM had correctly answered only 9 with three frames and 10 with five frames. Replacing only those decisions yields the net increases of 49 and 48 items, respectively. For the remaining 135 questions, the baseline VLM result is preserved.

This result is consistent with a *tool-augmented motion decision* being more useful than merely appending numerical trajectories to the VLM prompt. In batch27, the same 2.2B model's RGB-only result was 40/240 and RGB+positions was 51/240 on a separate newly generated five-option test (difference CI included zero), whereas a domain-specific classical color tracker was 240/240. These are different benchmark domains and cannot be directly compared as effect sizes.

## Interpretation and limitations

1. **Predeclared heuristic, public benchmark explored earlier.** The HSV hue thresholds were not optimized against CLEVRER labels in this run, but earlier experiments had already exposed model performance on this public task, so this is *not* a fresh independent blind evaluation.
2. The source videos are CLEVRER simulated scenes, not natural video; the tracker still relies on text-named target colors and sufficiently separated color components. It can fail with shading, camera motion, occlusion or similarly colored objects.
3. The gate accepted 32.5% of examples and had 89.2% precision there; it **can make erroneous high-confidence overrides** (7 errors), and its probability is not externally calibrated.
4. Median ~116 ms CPU overhead is on top of frozen video decoding/model GPU latency and may erode serving SLOs.
5. Public benchmark contamination in original VLM pretraining cannot be ruled out. The paired CI resamples 200 source questions; it is not repeated across multiple fresh model-training seeds.
6. This constitutes an **interpretable engineering pipeline** with genuine measured benefit on the scored public task, not a novel VLM architecture or proven broadly general tracking algorithm.

## Reproducibility

- New code: `experiments/appearance_hsv_tracker.py`, `experiments/appearance_hsv_transfer.py`.
- Config: `configs/appearance_hsv_transfer_v1.json`. Tests: `tests/test_appearance_hsv_transfer.py`.
- Source data and frozen reference VLM outputs: same as batch19 (all 200 CLEVRER Moving Direction videos and the 3/5-frame SmolVLM2-500M/2.2B outputs).
- Local raw per-question tracking, abstention, gate, and model comparison: `repo-batch30-tested/outputs/appearance_hsv_transfer_v1/HSV_TRANSFER.json`; human-readable results: `HSV_TRANSFER.md`.
- Source videos, saved model weights and LoRA adapters remain on Tang; never committed.

## Next experiment of value

Freeze a truly separate evaluation corpus with different video rendering/lighting or actual camera motion. Compare a pretrained object detector / tracker with the HSV heuristic, explicitly report tracking coverage, calibration, confidence-conditioned correctness and total end-to-end latency. Test whether a stricter learned/validated confidence gate reduces the seven incorrect overrides without losing most useful predictions. Any development must happen on separate development data, **not by tuning the same scored 200 CLEVRER items until metrics improve**.
