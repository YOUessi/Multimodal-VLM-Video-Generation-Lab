# Batch44B — query-frame initialization and calibrated three-view consensus (2026-11-02)

## Preserved failure

The first Batch44 freeze downloaded and hash-verified all 21 previously unused TAPVid-MV RRD sequences. R00 succeeded for all 21. R01 then failed on the first sequence, `cmu_soccer09_1`, with `NOT_ENOUGH_COMMON_THREE_VIEW_VISIBLE_TRACKS:...:0`. The cause was a hidden protocol assumption: the extractor required three shared points to be visible in all cameras at frame 0. That assumption is false for some synchronized multi-view recordings.

The failed Batch44 state and R01 log are retained in `repo-batch44-tested/.automation`; the RRDs and source manifest are preserved. No test scoring occurred.

## Query feasibility preflight

A read-only preflight examined only the initial-query feasibility condition at exact released depth snapshots (not tracker predictions or test performance). All 21 sequences have at least three shared points visible in all three cameras at a snapshot with at least 30 future frames.

- `cmu_soccer09_1`: first feasible query frame **18** (frame 0 had zero shared visible points).
- `cmu_soccer15_5`: first feasible query frame **30** (frame 0 had only one shared visible point).
- The remaining 19 sequences first satisfy the same rule at frame **0**.

This is a fixed benchmark-query construction rule applied identically to DEV and TEST. It does not select an algorithm, threshold, prompt, parser or model.

## Batch44B protocol

- DEV remains 11 sequences: two EgoExo4D collection groups and all nine PACE scene06 sequences.
- TEST remains 10 sequences: two different EgoExo4D collection groups and all eight PACE scene07 sequences.
- For each scene, choose the earliest exact depth-cloud snapshot with at least `queries_per_camera=3` shared 3-D track IDs visible in all three synchronized camera views, while leaving at least 30 subsequent frames.
- Rank those common IDs with the frozen per-sequence hash salt. Query each shared point once in each camera.
- Choose the evaluation target as the exact depth snapshot after the query frame nearest the frozen target fraction.
- Trackers start from the selected query frame and only future frames are scored. Pre-query trace entries are null and excluded from every metric denominator.
- The joint method triangulates the three RGB-only tracker predictions using the separately hashed public calibration file, reprojects into all three views, and accepts only if every reprojection is in-frame and within the predeclared 4/8/12 px residual threshold. Otherwise it abstains in all views.

## Leakage and safety

Public calibration contains only camera poses, intrinsics and image resolutions. 3-D point tracks, track IDs, depth clouds and future visibility labels remain in the private oracle. The private audit verifies that each query is visible and in-frame at its declared query frame. The 11 DEV / 10 TEST collection-group split is unchanged. TEST is not used to tune any tracker or VLM choice.

Only source code, tests, configs, plans and this protocol note are committed. RRDs, frames, calibration arrays, private oracles, predictions, model weights and secrets remain local. Batch44 failure logs and the original source RRD set are preserved.
