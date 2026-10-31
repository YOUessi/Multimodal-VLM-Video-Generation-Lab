# Batch 21 — actual ZeroScope video Diffusion UNet3D LoRA (2026-11-01)

**Tang frozen source snapshot `91df88ae98fab4a54240688eaf675349fee11a06`. All seven approved batch21 stages D0–D6 completed with return code 0; 74/74 local regression tests passed; no private media, model weights or trained adapter files committed.**

## Why this is a new experiment

Previous work demonstrated only video Diffusion **inference** from an existing pretrained ZeroScope model. This batch trains a genuine **PEFT LoRA adapter on the actual pretrained `UNet3DConditionModel`** for paired synthetic motion video + captions, saves adapter weights, and generates matched pretrained vs newly tuned MP4s.

The initial batch20 run was stopped **before any optimization** because Diffusers v0.37.1 exposes but does not support `UNet3DConditionModel.enable_gradient_checkpointing()`. The failure log was preserved. Batch21 uses a new source freeze, isolated output directory, and a known compatible configuration with gradient checkpointing disabled.

## Dataset and training

- Source: procedural red/blue/green moving squares on a plain light-gray background, fixed descriptive captions.
- Training **12** generated four-frame clips; validation **4** distinct generated four-frame clips (content-SHA disjoint).
- Target: ZeroScope v2 576w 3D UNet, pinned local safe weights, **CC-BY-NC-4.0, research/noncommercial**.
- VAE and CLIP text encoder **frozen**, with cached CPU-computed latents and text embeddings.
- PEFT LoRA: rank 4, alpha 4, target denoiser attention `to_q` and `to_v`; **830,720 trainable** of **1,412,064,580** total parameters (~0.0588%).
- Actual GPU optimizer updates: **12**, finite epsilon-prediction noise MSE throughout.
- Tang NVIDIA RTX 4090 Laptop, peak PyTorch allocated GPU memory during training: **3.483 GiB**.
- Validation evaluates **the same four heldout synthetic clips, fixed noise and fixed timestep** both before and after training.

| Validation outcome | Epsilon-denoising MSE |
|---|---:|
| Before LoRA | **0.072447** |
| After 12 optimizer updates | **0.066950** |
| Relative change | **-7.59%** |

This controlled reduction is genuine, but **not a video visual-quality score**, and the validation set is tiny and synthetic. It does not show generalized natural-video or high-fidelity video-generation improvement.

## Four actual paired model generations

All four MP4s were produced on Tang with **the same two text prompts and seeds**, 4 frames at 192×128, eight Diffusion inference steps.

| Prompt | Pretrained | LoRA | Peak PyTorch allocated VRAM | Color-centroid movement proxy |
|---|---:|---:|---:|---|
| Red square moving left to right | 3.3188 s | 2.9319 s | 2.727–2.730 GiB | **Not measurable for either model** |
| Blue square moving right to left | 2.2549 s | 2.7269 s | 2.727–2.730 GiB | Left movement recognized in both outputs |

These small per-run timing differences do not establish a speedup. The color proxy is a heuristic with missing/uncertain observations. **No human prompt alignment, visual quality or temporal-coherence scores have been collected**; no claim of better generated-video quality is made.

## Blinded human review and provenance

A four-video offline reviewer HTML has been generated under:
`repo-batch21-tested/outputs/zeroscope_video_lora_v2/blind_review/review.html`.

The review interface shows anonymous sample names and matching text prompts. The base/LoRA-to-sample mapping is stored separately on Tang in `outputs/zeroscope_video_lora_v2/blind_private/variant_mapping.json`. Reviewers enter their own scores; the application cannot invent them. All synthetic generated videos and adapters remain on Tang.

Primary local files:
- `outputs/zeroscope_video_lora_v2/training_report.json`
- `outputs/zeroscope_video_lora_v2/adapter/adapter_model.safetensors`
- `outputs/zeroscope_video_lora_v2/generated_comparison/generation_report.json`
- `outputs/zeroscope_video_lora_v2/EVIDENCE_REPORT.md`

Reproducible source and tests:
- `experiments/diffusion_paired_motion_data.py`
- `experiments/zeroscope_video_lora.py`
- `experiments/zeroscope_lora_generate.py`
- `experiments/zeroscope_lora_report.py`
- `configs/zeroscope_lora_smoke_nockpt.json`
- `tests/test_zeroscope_video_lora.py`

## Next valid research milestones

1. Independent human (two raters) review of the generated clips; use independent ratings rather than inferring quality from a denoising MSE reduction.
2. Larger natural video-caption training with verified usage rights, multiple training seeds, and heldout prompts. Twelve synthetic clips do not substantiate performance claims.
3. Improve video temporal evidence utilization in the separate VLM track. Batch19 finds that adding 3 visual frames increases 2.2B accuracy on CLEVRER motion by +15.5pp versus text-only (95% CI [+6.5,+24.5]), while adding another two frames barely changes predictions.
4. Wan2.1 was not trained or run; the actual training pipeline here is **ZeroScope**, with its noncommercial model license.
