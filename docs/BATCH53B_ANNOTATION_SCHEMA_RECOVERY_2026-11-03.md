# Batch 53B — frozen recovery from a natural-video annotation schema failure

## Failure and evidence preservation

On **2026-11-02 at 19:15:54 UTC**, frozen Batch 53 source
`51422c4454c67e7a1a393a82f503b131e92b2054` stopped in R03 with
`KINETICS_VISIBLE_COORDINATE_NOT_NORMALIZED` after completing R00–R02.
The original `repo-batch53-tested/.automation/state.json` and R03 log are
preserved, including the exact failed source SHA. Neither this new source nor
the old frozen source was modified in place.

This is an **annotation geometry integrity failure**, not a tracker or VLM
result. No TEST predictions or TEST scores existed when R03 failed. Structural
validation does not justify using TEST annotation coordinates to select
hyperparameters, models, or thresholds.

## Repair contract (identical for DEV and TEST)

The previously SHA-selected 12 Kinetics video IDs, 12 action classes, 6/6
DEV/TEST split, per-video 4 deterministic point queries, downloaded original
shards, original video data and official annotation CSV are *unchanged*.
No clip substitution or new data selection is permitted.

`parse_track_row` remains strict and rejects nonfinite values and invalid
occlusion flags. A new `usable_track_for_freeze` integrity filter excludes
an **entire original track**, without clipping or rewriting labels, when
a supposedly visible coordinate falls outside normalized [0,1] or the decoded
256×256 raster, allowing the documented upstream half-pixel correction.
Offscreen coordinates that are explicitly marked *occluded* are retained.
No other parse errors are suppressed.

For every scene, candidate queries are again sorted by the original
`SHA256(selection_salt, video_id, original_track_index)` ordering. The
original track index is preserved. The requested four queries must still be
available; otherwise the new batch also fails closed. The manifest records
raw track count, valid track count, and total geometry-excluded track count
for auditing. The same rule is applied before drawing queries on both splits.

## Frozen continuation, safeguards and limits

The 22 stage commands, CPU methods, model weights, prompt, GPU lock, DEV
method selection constraints (visible coverage ≥30%, occluded false accept
≤10%), TEST prediction-before-oracle-scoring firewall and VLM DEV admission
gate are byte-for-byte identical to Batch 53. Batch 53B has its own version,
source SHA, clean worktree, automation state and single scheduled cron.
GPU stages require the shared exclusive lock and no other CUDA processes.

This repair changes **which original annotation tracks can be queried**
when TAP-Vid rows are geometrically inconsistent. That can change the
evaluation population and may favor in-frame tracks. Consequently, report
Batch 53B as a distinct evaluation protocol, **not** as an unchanged
continuation of Batch 53 results. Do not compare it with Batch 53 as a
method-performance gain. Report any discarded-track counts and reasons,
with one immutable selection/public data hash.

The fix is tested solely on synthetic tracks before frozen evaluation.
It is not fit using TEST answers. Prior already-opened batches cannot be
reused to optimize acceptance thresholds. The new independent Batch 53B
TEST must remain sealed until its prediction artifact is pinned.
