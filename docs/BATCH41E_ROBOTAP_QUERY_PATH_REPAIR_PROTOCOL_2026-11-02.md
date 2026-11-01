# Batch 41E — RoboTAP query-path repair and isolated continuation

Date: 2026-11-02

## Why this batch exists

Batch41D preserved a failure at `R06_DEV_CPU_private_oracle_score`. Its DEV prediction file was created and SHA-pinned, but scoring stopped because prediction generation read public queries from the legacy `batch41_frozen` directory while the manifest and frames came from the configured `batch41_grouped_frozen` directory. The query IDs therefore did not match the private oracle.

The failure record explicitly confirms that no predictions were scored, no TEST predictions were created, and TEST labels were not opened. The failed directory and logs remain preserved.

GitHub main now resolves public queries through `src.frozen_dir(cfg)`. Batch41E adds a regression test with a deliberately conflicting legacy-path decoy, so a future fallback to the wrong directory fails deterministically.

## Data and split

This is a repair/continuation using the already-frozen, episode-disjoint RoboTAP manifest from Batch41B: 6 DEV recording episodes and 14 TEST recording episodes, one selected camera view per recording episode. It is **not** claimed to be a second independent split relative to Batch41D. The prior TEST portion remains untouched because Batch41D failed before TEST prediction or scoring.

RoboTAP is synthetic robotics video, not natural camera footage. It is used here because it provides point tracks and explicit occlusion labels. No claim of natural-camera generalization is made.

## Isolation and protocol

- Batch41E writes to `outputs/robotap_batch41e_grouped`; no Batch41D output is reused.
- The frozen data remains at `data/robotap_batch41/batch41_grouped_frozen`; source shards, frame arrays, public queries, and private oracle remain local and are hash-audited.
- CPU prediction is label-blind. Private labels are opened only by the DEV score stage, then the DEV-only selection stage.
- The VLM interface gate is based only on DEV output validity and decision coverage, not TEST accuracy.
- TEST predictions are pinned before the first TEST score stage. TEST is scored once, after DEV selection and prediction sealing.
- The VLM baseline is visibility classification and is reported separately from point-coordinate tracking.
- No video, pickle shard, extracted frame, private label, model weight, or secret is committed.

## Source and execution evidence

The exact source SHA is recorded in `.automation/state.json` as `frozen_sha`. The stage logs and state file are the execution record. The final report must distinguish DEV-selected behavior from TEST-only evaluation and disclose the synthetic-domain limitation.
