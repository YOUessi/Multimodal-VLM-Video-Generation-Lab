# Batch 23 — motion-salient 12-probe frame selection (2026-11-01)

**Completed:** all 6 approved stages F1–F6 on Tang, return code 0. **82/82 source tests passed**; security guard found zero model/video/secret binaries among tracked files. Frozen Tang source SHA: 6732c87fdb57ba3eaf39c2c2c04389b282dad984. Pinned upstream SHA: 623ce373be530acdd6a47e5eca7082f10f39e1d7.

## Protocol

Public MVBench: 200 simulated CLEVRER moving-direction videos and 200 camera-footage scene-transition videos. Frozen SmolVLM2-500M-Video-Instruct and SmolVLM2-2.2B-Instruct, 3/5 selected frames, no task-specific training. The selector first decodes 12 low-resolution candidate frames and uses 0.7 normalized grayscale motion + 0.3 normalized hue change, with 5%-95% endpoint and diversity guards. No question labels, answers, or model predictions inform frame selection.

Both models ran on each of the 400 videos with 3 and 5 frames: **1,600 actual GPU inferences**. These public test examples had been analyzed previously; this is exploratory, not a new blind-heldout experiment.

## Paired correctness

| Model | Task (n=200) | Frames | Uniform | Motion-selected | Delta pp | Paired bootstrap 95% CI pp |
|---|---|---:|---:|---:|---:|---|
| 500M | Moving Direction | 3 | 61/200 | 64/200 | +1.5 | [-0.5,4.0] |
| 500M | Moving Direction | 5 | 65/200 | 65/200 | +0.0 | [-2.0,2.0] |
| 500M | Scene Transition | 3 | 159/200 | 163/200 | +2.0 | [-0.5,5.0] |
| 500M | Scene Transition | 5 | 160/200 | 165/200 | +2.5 | [-1.0,6.0] |
| 2.2B | Moving Direction | 3 | 72/200 | 72/200 | +0.0 | [-2.0,2.0] |
| 2.2B | Moving Direction | 5 | 73/200 | 73/200 | +0.0 | [-1.5,1.5] |
| 2.2B | Scene Transition | 3 | 182/200 | 180/200 | -1.0 | [-3.5,1.5] |
| 2.2B | Scene Transition | 5 | 180/200 | 182/200 | +1.0 | [-1.5,3.5] |

**Every 95% confidence interval contains zero.** No reliably positive gain from this frame-selection algorithm. In particular 2.2B Moving Direction results are identical to uniform sampling.

Mean per-video 12-probe CPU selection overhead: **0.208 s**; median **0.113 s**, on top of selected-frame decoding and model inference. This method cannot be recommended as a better video-sampling approach on these tests.

## Interpretation

Comparing this result with the frozen batch22 repeated-mid-frame and reversed-order tests suggests that motion-salient visual selection is not sufficient to address temporal evidence utilization. The precise cause (visual token aggregation, positional encoding, attention, object tracking, language priors) is **not established**. A new algorithm should test temporally tracked per-object positions and independent video datasets, without tailoring this same inspected test set.

## Artifacts

Tang: repo-batch23-tested/outputs/motion_sampling_v1 contains the frozen 400-video frame-plan SHA, 500M and 2.2B per-example results, comparison JSON and human-readable report. Logs and state are under repo-batch23-tested/.automation. The local Tang 10-minute cron is enabled but has **no remaining approved stages** after completion. No video, model weights, private data or generated MP4s were published.

Code: experiments/motion_salient_sampling.py, configs/motion_sampling_v1.json and tests/test_motion_salient_sampling.py.
