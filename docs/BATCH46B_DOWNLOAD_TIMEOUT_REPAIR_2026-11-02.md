# Batch46B — resumable Kinetics shard-download timeout repair

Date: 2026-11-02

## Why Batch46 was preempted

The original Batch46 freeze (`6ab87777140ec54f00c6e346178cebdab2e34b56`) assigned 5,400 seconds to R00 while downloading twelve Kinetics class shards sequentially. Each curl subprocess can consume up to 2,460 seconds. At the observed Tang-to-S3 throughput, the stage budget could not cover the complete sequential workload.

R00 had verified shards 389 and 551. Shard 620 was still downloading when the operator stopped the transfer deliberately, before the stage timeout, to avoid spending the remaining budget on a run that could not finish. The failed-stage log is retained under the old Batch46 worktree. The 62 MB partial archive was preserved; no media or experiment data were deleted. No GPU inference stage had started.

## Repair committed to GitHub main

- Plan timeout for `R00_SELECT_AND_DOWNLOAD_12_FRESH_CLASS_SHARDS`: 5,400 s → 36,000 s.
- Added a regression assertion that the stage budget covers twelve sequential 2,460 s shard attempts plus a 600 s margin.
- The existing downloader already resumes a preserved partial archive with `curl -C -`; the replacement run was verified to invoke that resume path.

Commits:
- `bf14034a74436e26deb7caaac275c11654180665` — increase the R00 budget.
- `82a0bda1a542d4d0219abb1d4d0d81af2635f806` — add the timeout-budget regression.

## Replacement freeze and validation

- Replacement worktree: `repo-batch46b-tested`.
- Frozen source SHA: `82a0bda1a542d4d0219abb1d4d0d81af2635f806`.
- Targeted Batch46 tests: 8/8 passed.
- Full project suite: 444/444 passed; three existing Pillow deprecation warnings.
- The 10-minute cron marker now points only to the replacement worktree.
- R00 was restarted and resumed the existing shard-620 partial. Shards 389 and 551 remain verified in the download manifest.

## Current status

The replacement R00 download is still running. There are no Batch46 DEV or TEST metrics yet. The new freeze retains the original twelve-class, six-DEV/six-TEST selection and the pre-registered Batch45E DEV-selected `affine_stabilized_klt` method. No TEST-based tuning was introduced. GPU inference remains queued behind the existing shared-GPU admission lock; at the last check the RTX 4090 had no compute workload.

All raw videos, shards, frames, private point tracks, predictions and model weights remain local on Tang. Only source code, tests, protocol and this process note are published.
