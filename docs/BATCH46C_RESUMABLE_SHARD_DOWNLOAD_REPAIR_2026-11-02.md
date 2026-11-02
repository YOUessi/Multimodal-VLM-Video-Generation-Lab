# Batch 46C — Kinetics S3 stall and integrity-safe resume repair (2026-11-02)

## Observed incident

At 2026-11-02 02:15:55 UTC, the previously frozen Batch46B stage `R00_SELECT_AND_DOWNLOAD_12_FRESH_CLASS_SHARDS` failed. Frozen SHA: `82a0bda1a542d4d0219abb1d4d0d81af2635f806`. The stage logged `subprocess.TimeoutExpired` for a curl download of official S3 shard `k700_val_163.tar.gz` after 2,460 seconds. The process had not finished and its `.partial` file was retained. The three preceding shards (389, 551, 620) were complete and SHA-pinned in the existing download manifest; no TEST predictions were opened or scored.

The failure was **not** evidence of a tracking or VLM performance regression. The existing downloader had one long per-shard curl attempt with a timeout but lacked an internal retry loop and an explicit no-progress guard. After the failure, the project cron stopped at the failed frozen stage as intended. The GPU was idle; there was one cron entry with marker `MVVG_PROJECT4_10MIN`; 25 GB free disk space remained.

Live transport diagnostics confirmed official S3 shard 163 HEAD status 200, `Accept-Ranges: bytes`, object length 85,459,061 bytes and an ETag. A bounded manual byte-range test against the existing 1,843,200-byte partial returned HTTP 206 and extended the **same** partial to 3,252,815 bytes without starting over (curl itself returned timeout code 28 after 18 seconds). No media or failure logs were deleted.

## Code change

The shared Kinetics downloader now uses bounded retriable 420-second curl slices with `-C -` when partial bytes exist. The per-shard wall-clock budget remains 2,460 seconds, with a maximum of 16 attempts and early failure after three **consecutive** zero-progress attempts. Curl checks low-transfer speed (2 KiB/s for 75 s), instead of allowing prolonged silent stalls.

A frozen HEAD identity (Content-Length + ETag + byte-range support) is checked before any download and revalidated before final publication. Oversized or shrinking partials, short successful downloads, object identity changes, nonretryable curl codes, and invalid gzip CRC/trailer fail closed. All unsuccessful partial bytes are preserved; the final file is published only after completion and gzip verification. The manifest retains its SHA-256 pin and now records Content-Length and ETag. Existing complete shard files are reused and verified, never replaced simply because a retry is underway.

The 12-video, 6 DEV / 6 TEST class-disjoint selection and its SHA pin are unchanged. No dataset or TEST-answer-driven algorithm changes were made.

## Regression acceptance and continuation

Offline deterministic tests cover successful byte-range resume, timeout then resume, repeated stalls, resumed progress following stalls, completed-partial promotion, overflow, truncated success, remote ETag drift, nonretryable HTTP failures, existing final-file size mismatch, and gzip failure. Full repository regression results and frozen SHA are recorded by the Batch46C runner after publishing the source.

The next frozen run uses exactly the same selection and shared previously downloaded data, in a **new** `repo-batch46c-tested` checkout. The old failed Batch46/46B state and logs remain unmodified. Later stages must still pin DEV predictions, apply the pre-registered Batch45E tracker choice, gate VLM inference using DEV only, pin TEST predictions before opening TEST labels, and report coverage, PCK, incorrect acceptance, occlusion risk, latency, and error cases. If any stage fails again, stop at that stage and preserve all output.
