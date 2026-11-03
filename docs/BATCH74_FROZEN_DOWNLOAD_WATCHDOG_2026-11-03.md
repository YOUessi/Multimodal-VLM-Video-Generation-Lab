# Batch 74 — Read-only frozen-run observability and storage risk control

Date: 2026-11-03. Repository: `YOUessi/Multimodal-VLM-Video-Generation-Lab`, `main`.

**Status: ENGINEERING DIAGNOSTIC COMPLETE; NO ALGORITHM RELEASE OR TEST RETUNING.**

## Why this batch

The strongest independent natural-video generalization study is frozen Batch59 (36 DEV + 6 TEST new action classes, disjoint from 114 historical classes). As of this study, its R00 acquisition stage has been active for more than 14 hours; TEST evidence is **not yet available**. Prior Batches 62–73 conducted several exploratory/synthetic investigations; producing yet another synthetic positive result would not replace the blocked natural evidence. The highest-leverage next work is an independent, read-only source/runner/GPU/download/storage safety auditor that **does not edit** the ongoing frozen experiment.

## New implemented code

- `experiments/batch74_frozen_run_watchdog.py`: deterministic audit; separate pure `analyze` and read-only `snapshot` entry point.
- `tests/test_batch74_frozen_run_watchdog.py`: fault-injection regressions for frozen-SHA drift, tracked-file mutation, duplicate cron, duplicate runner/downloader, stalled partial, GPU occupation, 10-GiB reserve, pessimistic p95 archive forecast, malformed/contradictory states, repeated curl retry events, missing artifacts and hidden TEST oracle protection.
- No changes to Batch59's source tree, plan, cron, state, downloads, raw media, model weights, predictions, private labels or GPU lock. The code never reads frames, media or private labels and never launches a stage. No second cron is created.

## Real read-only assessment of Batch59 (Asia/Shanghai, 2026-11-03 approx. 22:19)

| Evidence | Observation |
| --- | --- |
| Tang device | online; Ubuntu RTX4090 laptop |
| Frozen checkout SHA | `8e079c70d0d5e687db54fb3a54da9c85a0f86831` |
| GitHub main at pre-development verification | `fe7c0381fe4aa1b3e5c1f0cc32679d14ab25cb8c`; intentionally newer than frozen checkout |
| Frozen tracked checkout | clean (read-only `git status --porcelain --untracked-files=no`) |
| `MVVG_PROJECT4_10MIN` | exactly one entry; 10-minute tick still on Batch59; `ALL_APPROVED_STAGES_DONE` guard present |
| Runner / downloader | one matching PID each, 714382 / 714387; no duplicate batch59 process |
| CUDA compute consumers | none in `nvidia-smi --query-compute-apps` |
| Stages | 1 R00 running, 23 pending, 0 complete or failed; 24 stages total |
| Completed archives | 26 / 42 |
| Current partial | `k700_val_594.tar.tar.gz.partial`; byte count recently increasing |
| Last logged retry | attempt 11, curl rc=28, 33,538,508 / 174,045,245 bytes; recoverable timeout, **not** proof of completion |
| R00 log archive-ready events | 26 unique shards |
| Disk free | approximately 12.08 GB decimal (filesystem reported 99% use) |
| Completed archive size p50 / p95 | 67,146,232 / 119,027,209 bytes |
| Pessimistic remaining archive estimate | 16 × 119,027,209 = 1,904,435,344 bytes, excluding future frame extraction/temp amplification |
| Reserve | 10 GiB = 10,737,418,240 bytes |
| Forecast reserve gap | approximately **-0.56 GB** after the p95 shard estimate |
| Audit finding | **WARN: DISK_FORECAST_TIGHT**; not a confirmed out-of-space failure |

P95 extrapolation from completed shards is a *heuristic*, not a prediction interval or bound for the unknown remaining shards. The active 174-MB shard already exceeds the observed historical p95, so the forecast is likely optimistic for a tail-heavy download. There is no guarantee sufficient space remains for video extraction or GPU inference artifacts. Conversely, the active downloader has made **real byte progress**, so the repeated `RUNNER_ALREADY_ACTIVE` and recoverable curl rc=28 lines do **not** justify killing or replacing it.

## Exact command (safe to repeat, no writes)

```bash
cd /home/you/projects/Multimodal-VLM-Video-Generation-Lab/repo-batch74-dev
../.venv/bin/python experiments/batch74_frozen_run_watchdog.py \
  --root /home/you/projects/Multimodal-VLM-Video-Generation-Lab/repo-batch59-tested
```

The JSON stdout contains `status`, structured alerts, actual process IDs, stage counts, shard counts, last curl retry, disk forecast and source SHA. It does not contain credentials, raw videos or labels. `remote_main_sha` may be null on network timeout; this is unknown, not evidence of source drift. The frozen checkout's *own* SHA and clean tracked file status are separately checked. The script deliberately does not launch, repair, stop or reschedule anything.

## Safety decisions and negative results

1. **No new algorithm is claimed.** This batch is engineering observability; no DEV/TEST tracking or VLM scores were generated and no production tracker was promoted.
2. **No source change in an active freeze.** The current 24-stage frozen Batch59 process remains authoritative. A later `main` release is not a reason to hot-swap source mid-run.
3. **No GPU contention.** The watchdog uses CPU-only read operations; it does not reserve or run the GPU.
4. **No storage cleanup without review.** Disk is too tight for casual new downloads. Preserve shards/media and failure logs. If the free-space guard actually trips, collect evidence and design a new frozen batch rather than deleting prior artifacts or restarting the current run.
5. **No test leakage.** The watchdog reads only `.automation/state.json`, its R00 progress log, archive filename/stat metadata, process/GPU/cron status and git status. It never opens the TEST oracle, prior TEST scoring or media frames.
6. **Expected progress.** Once R00 completes, its *existing frozen runner* should advance R01 onwards. When `ALL_APPROVED_STAGES_DONE` is genuinely set, audit real scene/episode/identity-reacquisition errors, frame vs background-relative motion, CPU and VLM metrics before any new algorithm proposal. If the runner fails, keep failure logs; repair only in an independent freeze.

## Validation

After first targeted test run, 3 issues were found and fixed *before release*: counting a shell command as the actual Python runner, incorrectly constructed test expected-count fixture, and a test assertion that confused descriptive documentation with actual oracle access. After repair, the targeted suite passed. A second improvement added detection of tracked-file mutation despite unchanged HEAD SHA, with independent fault-injection tests.

- New targeted suite: **47 passed**.
- Complete project suite: **1377 passed, 1 skipped, 3 existing Pillow deprecation warnings, zero failures**, confirmed exit code 0 (12.52 seconds).
- Real read-only Batch59 audit: ran successfully and returned `WARN` only for disk headroom.
- No changes to frozen Batch59 or raw datasets.

**Release decision:** Approve the read-only diagnostic utility and tests; do not call this a new tracker, independent natural validation, successful VLM run, or authorization to deploy the algorithm.
