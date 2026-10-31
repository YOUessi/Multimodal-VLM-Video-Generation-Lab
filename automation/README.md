# Tang staged execution: 10-minute local checks

## Batch 2 status (2026-10-31)

The active cron points at the frozen Tang snapshot `repo-batch2`, not the earlier OOM-failed `repo-source`. On this snapshot, B1 baseline, B2 48-step LoRA and B3 frozen heldout comparison completed successfully; B4 Wan full precision RAM gate ran and reported unsafe (preflight only). B5 requires three locally complete ZeroScope safetensors checkpoints and will remain pending (no attempt charged) while those weights download.

Batch 2 synthetic heldout metrics: single frame base / LoRA 18/18 vs 18/18; three and five frames 4/18 vs 18/18. These results are not real-world video understanding. Measured results: [batch2 report](../docs/BATCH_2_RESULTS_2026-10-31.md).

The current frozen Tang snapshot was created from GitHub main at upstream commit `cd0ae5351c6c4adb2270d5062f7f562c7cfc30a7`. Later **documentation-only GitHub commits** do not alter the running frozen local code.

## After B5: next research batches to approve and freeze

B6: Import and test the remaining CPU prototype; create a real-video annotation pipeline with verified labels and an independently frozen test split. B7: repeat LoRA across random seeds, training-data difficulty and frame-sampling ablations. B8: evaluate local diffusion videos using a blinded human rating sheet; investigate quantized or group-offloaded Wan only if RAM/VRAM/disk gates are satisfied. Do not mark these stages as executed until code, tests, real GPU logs and source hashes are present.


**Where**: Tang (Linux) only. This is a local cron job, NOT ChatGPT's scheduled-task feature (its minimum frequency is hourly). It does not send messages to this ChatGPT conversation.

## Planned fixed batch
- B1_temporal_baseline: Generate distinct train/val/test synthetic MP4 clips; actual SmolVLM2 CUDA baseline on 1/3/5 sampled frames and save all per-sample predictions.
- B2_lora_training: Genuine PEFT LoRA optimization on training split only, log loss, save adapter locally.
- B3_heldout_comparison: Baseline vs LoRA on the *same frozen test set*, annotate failure patterns and report deltas, **not real-world performance**.
- B4_wan_feasibility: Check CPU RAM, VRAM and disk before any download (full Wan 1.3B metadata estimate ~26.95 GiB). Preflight is **not** generation.
- P3 actual video generation and human scores require a subsequent explicit executable batch with tested offloading/quantization; no quality scores will be invented.

## Safety gates
1. Only execute when Git HEAD equals frozen SHA recorded under .automation/state.json.
2. No concurrent check runner (flock); GPU stages acquire global /tmp/tang_shared_gpu_experiments.lock.
3. Check 11,000 MiB minimum available VRAM, no other GPU compute process.
4. Hard timeout per stage; 1 attempt; **failure stops progression**, no blind retry.
5. Persist logs for each stage plus machine-readable state and checkpoints. Never delete prior outputs.
6. Frozen source drift, missing weights, low GPU memory or other running job all cause fail-closed checks.
7. All source, configs and docs may go to GitHub; private videos, model weights, adapter weights, and train/test generated media remain on Tang.

## Commands

From the Tang checkout:
```bash
.venv/bin/python automation/runner.py initialize
.venv/bin/python automation/runner.py status
.venv/bin/python automation/runner.py tick --max-stages 2
tail -100 .automation/logs/B1_temporal_baseline.log
crontab -l | grep mvvg
```

Pause locally (does not alter other user's cron entries):
```bash
touch .automation/PAUSE
```
Resume:
```bash
rm .automation/PAUSE
```
Disable checks (remove only the # MVVG_STAGED_EXECUTION line from crontab). Do not edit or stop unrelated jobs.

## What can and cannot advance automatically

The scheduler continues **approved deterministic scripts** under the frozen source SHA.
It cannot research new architectures, patch a broken training pipeline, or fabricate missing labels by itself. A failed or blocked stage pauses until diagnosed and a new immutable batch SHA is approved and frozen. ChatGPT cannot report every 10-minute local check in this chat; see .automation/state.json for actual progress.
