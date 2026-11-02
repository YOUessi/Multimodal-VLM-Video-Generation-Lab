# Batch 46E — transient Kinetics HEAD failure repair (2026-11-02)

## Frozen prior run and incident

The previous Batch46D source commit was `fd70adf39eef8a65e2c643d013138ef953b31777`. Its R00 download failed on 2026-11-02 at 05:25:58 UTC (23 seconds after starting). `urllib.request.urlopen(Request(url, method="HEAD"), timeout=25)` raised `http.client.RemoteDisconnected: Remote end closed connection without response`. The failure occurred at the next shard metadata HEAD after two previously verified shards. It was not a tracker failure, test oracle result, or corrupted video, and must not be interpreted as a generalization score.

The existing `repo-batch46d-tested/.automation/state.json`, R00 failure log, downloaded media, and partial shards remain untouched. Its 21 downstream stages were never executed. A matching single project cron previously pointed to this failed frozen checkout; repeated ticks only reported the same failed stage.

## Change — strictly transient metadata retries only

The shared Kinetics downloader now gives HEAD up to five attempts, with backoffs of 1, 2, 4, and 8 seconds. Retryable cases are transport disconnects, timeouts, reset/broken connections, urllib URL errors, and HTTP 408/429/500/502/503/504. Permanent 4xx, missing/invalid Content-Length, missing ETag, and absent byte range support remain fail-closed. The original ETag and Content-Length must match every later successful HEAD, even after the download has reached 100%. HEAD failure writes no bytes, never removes partial media, and never publishes a shard.

The frozen R01 targeted regression now includes HEAD retry and shard resume tests. The source selector, selected Kinetics videos/classes, DEV/TEST split, tracking parameters, VLM admission criteria, scoring firewall, and existing download-retry/time-budget contract are unchanged.

## Validation and controls

Development checkout `repo-batch46e-dev` from the verified Batch46D GitHub commit. Initial targeted run: 39/39 passing. Initial full suite before the plan modification: 475/475 passing, 3 pre-existing Pillow deprecation warnings. After staging the plan change, rerun full regression and record the frozen SHA separately. No raw video, model weights, hidden TEST annotations, or credentials may be published.

Previous experiment data paths, including `data/tapvid_kinetics_batch46/shards`, are shared and must remain intact. Only a newly committed GitHub main SHA can be used to initialize an independent `repo-batch46e-tested` frozen checkout and switch the sole `MVVG_PROJECT4_10MIN` cron. No frozen TEST metric is available from Batch46D.
