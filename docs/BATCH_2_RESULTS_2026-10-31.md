# Batch 2 results (Tang, 2026-10-31)

**Status:** GPU VLM benchmark, LoRA and heldout comparison completed. Full precision Wan only preflighted; ZeroScope safetensors model download is in progress and video generation has **not** completed.

## Controlled benchmark

Hardware: RTX 4090 Laptop (16 GiB). SmolVLM2-500M-Video-Instruct.
Dataset: 48 procedurally generated training MP4 clips, 12 validation MP4 clips, 18 separate heldout MP4 clips. The color triples are distinct between splits, although all splits use the same limited six-color vocabulary and synthetic visual style.

Dataset SHA256: `31cba8fa90163c0ede5eb464208656e2fab36ce868063971a6d35fc3e51c779b`.

Important change: use `processor.image_processor.do_image_splitting=False` for BOTH baseline and LoRA. Before correction, 3 frames expanded to **39 crops / 2601 tokens**, causing LoRA OOM. After correction, **3 images / 216 tokens**; batch 2 completed with approximately 2.20 GiB peak allocated VRAM in training.

| Task | Baseline | LoRA | Delta |
|---|---:|---:|---:|
| 1 frame, identify visible rectangle color | 18/18 | 18/18 | 0.0 percentage points |
| 3 frames, first/last color | 4/18 | 18/18 | +77.8 pp |
| 5 frames, first/last color | 4/18 | 18/18 | +77.8 pp |

48 real LoRA training steps; adapter saved locally and not committed. Frozen test split not used during training.

**Interpretation limit:** Small procedural clips; repeated color vocabulary, simple templates and only one training seed. The strong post-LoRA test gain is *only* on this synthetic heldout split. It does not demonstrate real-world video understanding, generalized temporal reasoning, or production reliability. The 1-frame task is different from the 3/5-frame task and must **not** be compared as a controlled effect of frame count.

## Wan full precision preflight

Actual Tang readings at preflight: 82.99 GiB free disk; 25.89 GiB available RAM; 15.99 GiB total GPU VRAM. Metadata-estimated Wan2.1-T2V-1.3B Diffusers full weights: 26.95 GiB. Conservative full-precision CPU-offload RAM gate failed. **No Wan download or generation** was attempted.

## Generation stage

A smaller text-to-video Diffusion baseline (`cerspense/zeroscope_v2_576w`, safetensors revision `1f35e3dc3350d649f342bb7618fe34ac8fae5c9e`, CC-BY-NC-4.0) has been staged. The three safetensors tensors total approximately 3.419 GiB. The source is version-pinned and no binary pickle checkpoints are requested. If the download is incomplete, `automation/runner.py` reports WAITING_FOR_REQUIRED_FILES and checks again on the next local cron tick without burning an attempt.

## Frozen scheduler

Batch 2 local frozen commit: `887b7ca37518fe26f81e17ec53cfd123d5ed65a6`.
Upstream GitHub code commit at snapshot time: `cd0ae5351c6c4adb2270d5062f7f562c7cfc30a7`.
Scheduler: Tang cron `*/10 * * * *`, GPU lock, no blind retry, failure fail-closed.
Artifacts: on Tang under `repo-batch2/outputs/temporal_round2/` and `repo-batch2/.automation/logs/`. They are not uploaded raw because model weights, private inputs and generated video data are excluded.

## Remaining work

- Real-video data with human verified temporal labels and heldout conditions.
- Multi-seed LoRA and more diverse tasks with independently frozen splits and ablations.
- Actual safe Diffusion video generation and blind human evaluation; Wan quantized/offloaded feasibility before running larger model.
- Merge prior CPU module suite to GitHub and run full end-to-end CI.
