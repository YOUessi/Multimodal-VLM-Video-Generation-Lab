# Batch 29 — unchanged synthetic-palette tracker on public CLEVRER video (2026-11-01)

**Status:** All three L1–L3 approved stages completed on Tang, return code 0. **98/98 source tests passed** and source publication guard passed. Frozen Tang local SHA `ae2af3cb6108dafe26d28c57681680bdae4c8c69`, upstream GitHub SHA `99813dee80089ba0453f23c44e9175a86415fe8d`.

## Scientific motivation

Batch27 measured **240/240** correct directions for a deterministic query-grounded pixel-color tracker on newly generated bright 2D shapes; 500M and 2.2B video-language models mostly remained near 20% on five-option direction QA, with/without its numeric output. Such a simple tracker may be strongly synthetic-domain-specific.

This batch tests the **exact same color names, palette thresholds, segmentation and displacement rules**, WITHOUT recalibrating from CLEVRER labels or test outputs, on **all 200 official MVBench moving_direction questions**. CLEVRER is simulated video rather than camera footage. It was previously inspected in batch19, so this is **not a new never-seen blind evaluation**.

A predeclared conservative gate accepts the tracker only when a target is identified on all three samples and it predicts a nonstationary diagonal with absolute pixel displacement >=20px on BOTH axes; otherwise the system **abstains** and uses frozen VLM predictions previously saved in batch19. No extra GPU model inference is performed.

## Actual result

| Metric over all 200 CLEVRER videos | Result |
|---|---:|
| Original-source public questions evaluated | **200/200** |
| Recognized color name but missing reliable target in sampled frames | **118** |
| Unsupported target-color query (no palette entry) | **82** |
| Tracker provided an answer option | **0/200** |
| Confidence-gate accepted motion predictions | **0/200** |
| CV-only correct (all abstentions count as unhandled) | **0/200** |
| Median CPU tracking overhead | **0.08883 seconds per video** |

The 82 'RuntimeError' cases were manually checked at the ERROR METADATA level (not the gold labels) and all were `AMBIGUOUS_TARGET_COLOR_QUERY:[]`, i.e. the question names a color absent from this unchanged synthetic palette. The other 118 use a recognized color string but the video pixels did not yield reliable color-matched components. These are **coverage failures, not incorrect motor-direction predictions**; they must not be silently removed from the denominator.

The conservative fallback therefore always used the frozen VLM output:

| Frozen model | 3-frame VLM | 3-frame gated fallback | 5-frame VLM | 5-frame gated fallback |
|---|---:|---:|---:|---:|
| SmolVLM2-500M | 61/200 | **61/200** | 65/200 | **65/200** |
| SmolVLM2-2.2B | 72/200 | **72/200** | 73/200 | **73/200** |

All paired differences are **0 percentage points**, bootstrap CI [0,0], because the motion tool declined to answer on every external source item. The gate prevented false confident overrides, but **did not increase accuracy or add utility**, and still imposed measurable CPU cost.

## Conclusion

**Synthetic 100% did not transfer even to another simulated-video domain.** This is evidence of brittle exact-RGB palette assumptions and lack of robust object localization. It reinforces that a hand-built source-pixel tracker is not a general motion understanding solution.

The next defensible improvement should replace fixed RGB-color lookup with category-grounded object detection / robust color segmentation or a pretrained visual tracker, add tracking-confidence checks and compare under new independently frozen data. If a tracker abstains on most examples, a gated agent does not improve over the VLM and may waste compute.

## Frozen artifacts

- Source: `experiments/clevrer_untuned_tracker_transfer.py`, `configs/clevrer_untuned_tracker_v1.json`, `tests/test_clevrer_untuned_tracker_transfer.py`.
- Tang local report and per-video observations: `repo-batch29-tested/outputs/clevrer_untuned_tracker_v1/UNTUNED_CLEVRER_TRANSFER.md` and `.json`.
- Source public video and QA dataset, model weights and adapters remain on Tang; no raw video or weights published.
- Tang local cron is enabled every 10 minutes, but batch29 has **no remaining approved stages**. It will report `ALL_APPROVED_STAGES_DONE` until a new tested/frozen batch is installed.
