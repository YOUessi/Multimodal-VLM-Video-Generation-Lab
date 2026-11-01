# Tang staged execution: 10-minute local checks

## Current approved research batch — Batch44B TAPVid-MV query-frame and three-view consensus (2026-11-02)

Batch44B repairs the Batch44 initialization failure without changing the selected collection groups or scoring rules. The 21 sequence-group-disjoint EgoExo4D/PACE RRDs are reused by hard link from the preserved Batch44 download; no media is duplicated or deleted. Each scene starts at the earliest exact depth snapshot with at least three shared points visible in all three synchronized views and at least 30 future frames. This is a fixed query-definition rule applied identically to DEV and TEST; only DEV may select RGB/cross-view methods or admit the SmolVLM2 visibility baseline. Public calibration contains camera poses/intrinsics/resolution only; point tracks, depth clouds, future labels, predictions and weights remain local. The failed Batch44 state and logs remain untouched. See [Batch44B query-frame protocol](../docs/BATCH44B_QUERY_FRAME_INITIALIZATION_2026-11-02.md) and [frozen plan](plan.json).

The Tang cron runs at most two approved stages per tick. Failed stages stop the batch; repairs require a new source SHA and new tested directory. GPU stages acquire `/tmp/tang_shared_gpu_experiments.lock`, require at least 11,000 MiB free, and reject any competing CUDA compute process.

## Historical Batch42 Kinetics visibility plan (2026-11-02)

The current frozen plan is `batch42-kinetics-real-camera-visibility-20261102-v1` with 20 fail-closed stages. It starts by verifying the official TAP-Vid-Kinetics source and local video/frame hashes, runs targeted and full regression, generates DEV CPU predictions, scores DEV once, seals DEV-only method selection, then runs the SmolVLM2-500M DEV baseline and a strict admission gate. Only after those steps does it produce TEST CPU predictions and score them once; TEST VLM inference is skipped unless every DEV gate passes. It finishes with report, source-security, data-hash and prediction-pin audits. The protocol is [BATCH42_KINETICS_PROTOCOL_2026-11-02.md](../docs/BATCH42_KINETICS_PROTOCOL_2026-11-02.md), and the machine plan is [batch42 JSON](plans/batch42_kinetics_real_camera_visibility_2026-11-02.json).

The scheduler runs at most four stages per tick. A failed stage has one attempt and stops the batch; retain its log and partial outputs, repair on a new source SHA and new tested directory. GPU stages require the global lock, at least 11,000 MiB free and no competing compute process. Raw clips, frames, annotations, weights and secrets remain local.

## Historical Batch 2 status (2026-10-31)

At that time, the active cron pointed at the frozen Tang snapshot `repo-batch2`, not the earlier OOM-failed `repo-source`. On that snapshot, B1 baseline, B2 48-step LoRA and B3 frozen heldout comparison completed successfully; B4 Wan full precision RAM gate ran and reported unsafe (preflight only). B5 required three locally complete ZeroScope safetensors checkpoints and remained pending. This is historical context, not the current cron target.

Batch 2 synthetic heldout metrics: single frame base / LoRA 18/18 vs 18/18; three and five frames 4/18 vs 18/18. These results are not real-world video understanding. Measured results: [batch2 report](../docs/BATCH_2_RESULTS_2026-10-31.md).

The historical Batch2 Tang snapshot was created from GitHub main at upstream commit `cd0ae5351c6c4adb2270d5062f7f562c7cfc30a7`. Later **documentation-only GitHub commits** did not alter that frozen local code.

## Historical post-B5 research ideas

B6: Import and test the remaining CPU prototype; create a real-video annotation pipeline with verified labels and an independently frozen test split. B7: repeat LoRA across random seeds, training-data difficulty and frame-sampling ablations. B8: evaluate local diffusion videos using a blinded human rating sheet; investigate quantized or group-offloaded Wan only if RAM/VRAM/disk gates are satisfied. Do not mark these stages as executed until code, tests, real GPU logs and source hashes are present.


**Where**: Tang (Linux) only. This is a local cron job, NOT ChatGPT's scheduled-task feature (its minimum frequency is hourly). It does not send messages to this ChatGPT conversation.

## Historical fixed batch (Batch2)
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

The local Tang cron continues **approved deterministic scripts** under the frozen source SHA; it cannot invent research or edit code. The separate ChatGPT research-continuation workflow reviews completed metrics, develops and tests the next source batch, publishes it to GitHub main, then freezes a new Tang checkout. A failed or blocked local stage pauses until diagnosed and repaired on a new immutable SHA and new tested directory. The local cron itself does not post chat messages; `.automation/state.json` and `.automation/logs/` are the execution record.
