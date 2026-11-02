# Batch49B — Prospective bounded slow-shard continuation repair

Date: 2026-11-02 UTC. **This is source preparation only; the running Batch48 frozen checkout is NOT modified or replaced.**

## Evidence and diagnosis

On Tang, the isolated Batch48 frozen SHA `6120406c5ef9931d93f95a2a7b3c294ba0ce78d3` continued to run R00. It downloaded `k700_val_635.tar.gz` (90,318,464 bytes). At 2026-11-02 14:33 UTC, log attempt 13 reported 61,327,716 bytes transferred. The curl attempts were returning timeouts (rc 28) **with continuing byte progress**. The current downloader source in that frozen checkout has a 16-attempt and 7,200-second total per-shard ceiling. This is a **risk of premature exhaustion**, not a claim that Batch48 has already failed. Current state and logs must remain authoritative.

A download taking >16 successful-progress slices can be stopped by the attempt limit even when it is not stalled. A longer total budget alone does not fix this if the attempt cap is left at 16. Batch46C/46D had analogous earlier stop conditions.

## Forward-only changes in GitHub main

In the common, shared, SHA-pinned Batch45 downloader (reused under isolated Batch48 module namespace):

- Increase per-shard total elapsed cap to **14,400 seconds** (4 hours).
- Increase finite maximum curl attempts to **48**.
- Extend both forward-only R00 plan definitions to **173,400 seconds** (12 × 14,400 s + 600 s allowance), preserving a finite independent stage limit consistent with the per-shard cap. The live frozen Batch48 plan remains at its original 87,000-second limit.
- Retain each curl slice's hard maximum of **420 seconds**, speed floor 2,048 bytes/s after 75 seconds, at most **three consecutive zero-progress attempts**, immutable ETag and Content-Length checks, gzip CRC, and atomic final publication.
- On failure, preserve partial bytes, original logs and all prior completed files; no manual intervention in the currently running frozen SHA.
- No video selection, DEV/TEST identity, annotations, tracker thresholds, VLM prompts, or scoring changes.

Offline regression simulates **21** consecutive progressive download attempts lasting **7,560 seconds**; it must succeed under the new bounded policy and would have failed under the prior 16-attempt / 7,200-second limits. Additional regressions confirm the no-progress abort and truncated-success rejection remain enforced.

## Deployment policy

Keep the single `MVVG_PROJECT4_10MIN` cron bound to the unchanged Batch48 freeze while it runs. Only if Batch48 really fails or finishes should a separate, clean source-frozen batch be initialized after verifying GitHub main and retaining all logs. Do NOT alter a running `repo-batch48-tested` or restart its R00 process. A future new batch can reuse byte-identical `.partial` files under existing ETag/size/gzip gates.

Limitations: the patch does not improve download speed, guarantee the remote object remains available, or ensure that all twelve shards fit within the *stage-level* execution budget. It is a finite engineering retry-policy extension, not a model performance result.
