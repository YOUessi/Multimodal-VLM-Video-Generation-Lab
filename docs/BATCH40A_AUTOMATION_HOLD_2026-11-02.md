# Batch40A automation hold — SHA marker path mismatch

Date: 2026-11-02

## Observed failure

Batch40A froze source SHA a871dc2349c6a9dbc8bff9455f4a23a4d3ccc085 and completed:

- D01: acquired 12 scene-disjoint DAVIS clips, 16 frames each, 384 RGB/mask assets.
- D02: full manifest, frame/mask hashes and private gold audit passed.
- D03: 249 regression tests passed.
- D04: four DEV-scene CPU predictions completed and were SHA-pinned. The predictor did not read future masks.

The runner then stopped before D05 scoring with WAITING_FOR_REQUIRED_FILES for outputs/davis_occlusion_v3/dev_predictions.json.sha256.

## Root cause

The shared pin helper uses pathlib.Path.with_suffix(".sha256"). For dev_predictions.json this creates dev_predictions.sha256, not dev_predictions.json.sha256. The plan's four prediction-marker requirements used the wrong filename convention. No prediction, dataset, or scoring code failed.

## Recovery

- Preserve Batch40A state.json, logs, and outputs in repo-batch40-dev; do not overwrite or delete them.
- Preserve the exact 12-scene data freeze and its manifest/hash records; copy the 13 MB frozen image/mask dataset into the Batch40B checkout.
- Correct all CPU/VLM DEV/TEST marker paths to Path.with_suffix(".sha256") names.
- Freeze a new source SHA, re-run predictions under that SHA, and only then continue DEV scoring, DEV-only selection, TEST prediction sealing, and one-time TEST scoring.
- Do not reuse the Batch40A DEV prediction JSON because its source SHA differs from Batch40B. No TEST labels were read in Batch40A.

This is a runner-plan path correction, not an algorithm result. The Batch40A failure log remains available on Tang.
