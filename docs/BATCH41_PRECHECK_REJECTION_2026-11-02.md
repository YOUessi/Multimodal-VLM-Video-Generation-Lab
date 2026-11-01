# Batch41 preflight rejection — view-level split leakage (2026-11-02)

The first Batch41 freeze at source SHA e86a7e7296e9e004022c89f6e597b7e34e59eeb4 completed data acquisition, manifest preparation and public frame/query integrity checks. It was rejected **before any CPU/VLM predictions and before any private-oracle scoring**.

## Failure found

The initial deterministic split ranked camera-view keys independently. A metadata-only audit of the 20 selected keys found:

- gear_v2_17 had one camera view in DEV and another in TEST.
- gear_v2_8 had one camera view in DEV and another in TEST.
- gear_v2_11 appeared twice within TEST.

These are distinct view keys but share a recording/episode prefix. A view-level split therefore did not establish episode-level independence and could leak scene/task context across DEV and TEST.

## Preserved evidence

- Tang snapshot: repo-batch41c-tested, source SHA e86a7e7296e9e004022c89f6e597b7e34e59eeb4.
- Frozen data: data/robotap_batch41/batch41_frozen/.
- Runner state and logs: repo-batch41c-tested/.automation/.
- Local rejection record: repo-batch41c-tested/.automation/preflight_rejection.json.
- No predictions, VLM generations, thresholds, or TEST scores were produced from this rejected freeze.

The old snapshot is paused and retained. No source data, media or logs were deleted.

## Corrective protocol

Batch41B now groups all view keys by the stable recording prefix before any split, hash-ranks distinct episode IDs, chooses one camera view per episode using a separate hash, and assigns six DEV plus fourteen TEST episodes. The new manifest records episode IDs and fails closed if any episode appears more than once. It writes to a new frozen subdirectory so the rejected freeze remains intact.

This is a methodological correction, not a model-performance result.
