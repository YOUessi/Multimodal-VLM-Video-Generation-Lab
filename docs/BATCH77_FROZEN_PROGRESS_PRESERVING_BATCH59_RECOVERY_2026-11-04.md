# Batch 77 — Frozen progress-preserving recovery of Batch 59 natural-video run

Date: 2026-11-04. Repository: `YOUessi/Multimodal-VLM-Video-Generation-Lab`.

## Failure motivating the repair

Batch59 `8e079c70d0d5e687db54fb3a54da9c85a0f86831` failed stage
`R00_SELECT_AND_DOWNLOAD_42_FRESH_CLASS_SHARDS` on 2026-11-03 17:02:29 UTC.
The original Kinetics shard downloader successfully pinned **26 of 42 archives**,
but `k700_val_594.tar.gz` exhausted its per-shard 4-hour retry budget with
**148,475,765 / 174,045,245 bytes** of persistent partial content. The log
records continuing positive progress throughout attempts; this was not a
source-identity drift, corrupted verified gzip, or stalled connection.

The original `repo-batch59-tested` checkout, `.automation/state.json`,
failed log, and frozen plan must remain unchanged, including their failure.
A runner with a failed stage cannot be revived by editing its state in place.

## Safe continuation

Batch77 clones the current published source tree into a clean, independently
frozen `repo-batch77-tested` checkout, reusing only immutable video selection
metadata, verified archives and persistent partial bytes through the already
approved shared `data -> ../data` directory.

The 24-stage Batch59 evaluation protocol is preserved: 36 DEV and six
untouched TEST videos with **action classes and video IDs disjoint from the
114 historical items**; TEST predictions are fixed before reading TEST labels.
There are no altered motion methods, thresholds, scene scores, frozen test
inputs, VLM settings or GPU policies in this batch.

The new entry point is `experiments/batch77_resumable_download.py download`.
It delegates all downloading, remote HEAD/ETag verification, gzip/CRC checking,
manifest hash pinning, class-disjoint exclusion and disk preflight to the
**unchanged original Batch59 downloader**. Only when that downloader raises
the exact `KINETICS_SHARD_RETRY_EXHAUSTED` exception AND the corresponding
persistent `.partial` file grew during that invocation, the wrapper returns
dedicated exit code 75.

The separate runner `automation/runner_batch77.py` permits exit 75 to move
R00 back from `failed` to `pending` *only if fewer than 96 bounded sessions
have been attempted*. It records each deferral in independent automation state.
A nonprogressing transfer, remote-identity drift, unexpected network error,
short/corrupt complete archive, exhausted stage-attempt budget or failure
in **any** other stage remains fatal. Only a true downloader exit code 0
marks R00 complete; the runner must not manufacture evaluation results.

The existing GPU isolation via `/tmp/tang_shared_gpu_experiments.lock` and
11,000 MiB minimum free-memory condition are unchanged. Never run parallel
VLM jobs. Media, metadata, oracle, existing weights, secrets and failed logs
are Tang-only and must never be published.

## Verified before freezing

- 26 dedicated and inherited shard-retry regression tests passed.
- Full repository test suite: **1458 passed, one skipped**, three known
  Pillow deprecation warnings, exit code zero.
- Batch59 frozen original SHA and failed R00 log inspected on Tang.
- Current GitHub main before this recovery was
  `f6d990c8943fd627907860464a7fef63e465d0ed`;
  local starting source tree was `bd31e4fedd7ca3a38dc6a29f9530a4535302cd6f`.
- Recovery produces no natural-video results until the prospective frozen
  Batch59 evaluation completes. No TEST oracle was read during this repair.

## Required deployment audit

1. Verify source file hashes, clean checkout, GitHub main tree parity and
   newly frozen commit SHA.
2. Confirm old Batch59 R00 remains FAILED and old cron target is removed
   from the single `MVVG_PROJECT4_10MIN` entry.
3. Initialize new state, then install exactly one 10-minute cron entry
   targeting new frozen runner.
4. Start the new runner only when no duplicate runner, curl downloader or
   CUDA compute job exists. Preserve prior partials.
5. Audit progress from actual shard sizes, `.automation/state.json`, cron
   and stage logs; record negative outcomes including download failures.

**Deployment authorization:** operational recovery of an already frozen
natural-video experiment only; no approval for tracking algorithm deployment.
