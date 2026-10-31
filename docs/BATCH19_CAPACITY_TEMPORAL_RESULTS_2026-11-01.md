# Batch 19 — Frozen SmolVLM2 500M vs 2.2B video evidence ablation

Verified on Tang on **2026-11-01** (Beijing time). All **eight approved stages B89–B96 returned exit code 0**, with **64/64 source tests passed** under frozen local source SHA `51f065c669620f70a3ade98ed60a46ae957be5ab` (GitHub pinned SHA `5a2aec8d91f690486ba373fb9eec1dbb937b5b78`).

## Experimental protocol

Two frozen pretrained models, with **no task-specific finetuning**:
- `HuggingFaceTB/SmolVLM2-500M-Video-Instruct@7b375e1b73b11138ff12fe22c8f2822d8fe03467`
- `HuggingFaceTB/SmolVLM2-2.2B-Instruct@482adb537c021c86670beed01cd58990d01e72e4`

Two full public MVBench tasks with 200 official multiple-choice examples each:
- **moving_direction**, CLEVRER simulated object motion videos, **NOT real camera footage**.
- **scene_transition**, public video scene transition task with camera footage.

All 400 source video files have unique full SHA256 contents and zero exact-content overlap with the 193 unique byte contents in the preceding STAR training corpus. Provenance archives, annotation JSONs and their hashes are pinned in `configs/capacity_temporal_v1.json`.

For each model/sample, evaluate **0 frames (question and answer text only), uniform 3 frames, uniform 5 frames** with identical prompts and 5%–95% actual decoded-frame selection. All generated answers saved verbatim, with parsed letter, gold choice, correctness, image decoding time and GPU generation time. Unparseable answers count incorrect; no test-time sample filtering or model choice tuning. Total successful GPU model requests = **2,400**.

## Task accuracy (correct / 200)

| Task | Model | Text only | 3 frames | 5 frames | PyTorch peak allocated VRAM |
|---|---|---:|---:|---:|---:|
| CLEVRER Moving Direction | 500M | 52/200 (26.0%) | 61/200 (30.5%) | 65/200 (32.5%) | 1.107 GiB |
| CLEVRER Moving Direction | 2.2B | 41/200 (20.5%) | 72/200 (36.0%) | 73/200 (36.5%) | 4.361 GiB |
| MVBench Scene Transition | 500M | 81/200 (40.5%) | 159/200 (79.5%) | 160/200 (80.0%) | 1.107 GiB |
| MVBench Scene Transition | 2.2B | 114/200 (57.0%) | 182/200 (91.0%) | 180/200 (90.0%) | 4.361 GiB |

## Paired bootstrap 95% confidence intervals

| Main contrast | Task | Delta percentage points | 95% CI pp |
|---|---|---:|---|
| 2.2B, 3 frames vs text-only | Moving Direction | +15.5 | [+6.5,+24.5] |
| 2.2B, 5 frames vs text-only | Moving Direction | +16.0 | [+7.0,+24.5] |
| 2.2B, 3 frames vs text-only | Scene Transition | +34.0 | [+27.0,+41.0] |
| 500M, 3 frames vs text-only | Scene Transition | +39.0 | [+31.0,+47.0] |
| 2.2B vs 500M, both 3 frames | Scene Transition | +11.5 | [+5.0,+18.0] |
| 2.2B vs 500M, both 5 frames | Scene Transition | +10.0 | [+4.0,+16.0] |
| 2.2B vs 500M, both 3 frames | Moving Direction | +5.5 | [-1.5,+12.5] |
| 2.2B vs 500M, both 5 frames | Moving Direction | +4.0 | [-2.0,+10.0] |
| 2.2B, 5 vs 3 frames | Moving Direction | +0.5 | [-2.0,+2.5] |
| 2.2B, 5 vs 3 frames | Scene Transition | -1.0 | [-3.5,+1.5] |

## Findings

1. Video frames **materially improve accuracy** over a question-only prior, especially on scene transitions. On Moving Direction, 2.2B yields a paired +16pp (95% CI [+7,+24.5]) with 5 frames over no video.
2. Larger model capacity helps substantially on scene transitions (+11.5pp at 3 frames, CI [+5,+18]), but evidence of a capacity gain on CLEVRER moving directions is **not conclusive**.
3. **Five uniformly sampled frames are not reliably better than three** at either model size on either task. More visual tokens are not automatically more effective temporal evidence.
4. Moving-direction remains difficult: 2.2B with 5 frames reaches **36.5%** on four-option simulated motion, far from robust motion reasoning.

## Interpretation caveats

- A zero-frame baseline measures question/option priors, but does not prove all forms of dataset leakage or model pretraining contamination are eliminated.
- CLEVRER videos are simulated and cannot be described as natural human activity videos.
- Public MVBench may have been included in the foundation model's original training.
- Single inference run per deterministic model/configuration. The confidence intervals resample evaluation examples, not random training seeds.
- Total VRAM values shown are **peak allocated PyTorch memory**, not total NVIDIA reserved memory or overall process usage.
- The model comparison is a measured baseline, **not** a new VLM training algorithm or proven architecture contribution.

## Reproducibility

- Code: `experiments/mvbench_capacity_temporal.py`
- Config: `configs/capacity_temporal_v1.json`
- Tests: `tests/test_capacity_temporal.py`
- Local manifest source SHA256: `cbbcf19e083fe27832bb4a36b1a7f67145fe000a81c7cce8b7694a18a4e0bf32`
- Local raw results: `repo-batch19-tested/outputs/capacity_temporal_v1/{manifest.json,500m.json,2b.json,comparison.json,CAPACITY_TEMPORAL_REPORT.md}`
- Original public videos, model weights and any private artifacts stay on Tang; source and metric summaries are published, raw copyrighted videos are not.
