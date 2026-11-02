# Batch 46D — slow-but-progressing shard download budget repair (2026-11-02)

## Observed failure (frozen Batch 46C)

Source: `9555319c0ab5bd833a0508d01b0c4e425f4f55b9`.
The frozen `repo-batch46c-tested/.automation/state.json` reported
`STOPPED_AT_R00_SELECT_AND_DOWNLOAD_12_FRESH_CLASS_SHARDS:failed`,
return code 1, at **2026-11-02 04:24:44 UTC**.
The terminal failure was `KINETICS_SHARD_RETRY_EXHAUSTED` on
`k700_val_292.tar.gz`: **71,434,751 / 74,184,521 bytes (96.29%)**.
The six preceding DEV shards had completed and were SHA-256-pinned.
The log shows successive *increases* in partial bytes. Thus this was the
per-shard 2,460-second wall-clock budget, not zero-progress or CRC failure.

All failed state, logs, and partial shard bytes must be retained in Batch 46C.
Never alter Batch 46C's code or restart its failed state.

## Repair (new frozen source)

- Expand the per-shard retry budget from **2,460 to 7,200 seconds**,
  while retaining 420-second curl slices, a maximum of 16 attempts,
  75-second low-speed detection, and failure on three consecutive
  zero-byte-progress attempts.
- Expand the overall R00 hard stage timeout from **36,000 to 87,000 seconds**.
  A regression test verifies stage timeout >= 12 * shard budget + 600 seconds;
  stage and per-shard execution remain bounded.
- Keep `curl -C -` resume, Content-Length/ETag drift rejection, whole-gzip
  CRC verification, archive member verification, and SHA-256 manifests.
- Add an offline regression simulation that progresses slowly for over
  2,460 seconds, then finishes, without dropping any bytes.

## Regression and evaluation firewall

Development worktree: `repo-batch46d-dev`.
Targeted regression: **20/20 passed**.
Full project regression: **456/456 passed, 3 pre-existing Pillow warnings**.
No algorithm, video selection, prior DEV method, TEST answer, or gate threshold
was changed. The original data-selection SHA remains unchanged.

After GitHub main fast-forward and SHA verification, Batch 46D must be
initialized in a **new** `repo-batch46d-tested` worktree at the committed
SHA, using the existing read-only shared media links. The Tang cron must
point to only one frozen worker. Data and TEST results remain private.
