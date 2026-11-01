# Batch 39 — TAPVid-MV DROID multi-view motion decomposition

## Research question

When the camera moves, how much of a point's image-plane displacement is caused by camera motion versus motion of the physical point in the world? Does an image-only background affine estimate recover the camera-induced component at the point's depth, and how does that relate to point-tracking accuracy?

## Frozen data design

- 12 new DROID recordings from 12 distinct collection-group prefixes; 4 DEV groups and 8 TEST groups.
- 3 synchronized RGB views per recording; 53–196 frames per recording.
- Rerun 0.35.0 viewer recordings provide RGB, camera transforms/intrinsics, a depth-derived point cloud at selected snapshots, and a display subset of 3D point tracks.
- Group split and within-group sequence choice are SHA-256 ranked from a fixed salt. The preflight AUTOLab sequence is excluded.
- Three first-frame queries per camera are chosen only from query-frame points whose projection is in-frame and agrees with the exact frame-0 depth snapshot.
- TEST predictions are generated from RGB and public first-frame query coordinates only. Future 3D tracks, camera poses and depth snapshots remain in the private oracle until predictions are SHA-pinned.

## Coordinate and visibility contract

The released camera transform is applied as camera_xyz = R @ (world_xyz - camera_translation). The serialized pinhole matrix is applied to [x/z, y/z, 1] as a row vector. The implementation is tested with synthetic projections and color-consistency preflight.

Depth visibility is computed only on frames with an exact released point-cloud snapshot; no stale-cloud carry-forward is allowed. A 3 px neighborhood and 5 cm depth margin produce visible, occluded, out, or unknown. These are depth-cloud-derived visibility labels, not a separately published per-view binary occlusion annotation.

## Methods

- Stationary image coordinate.
- Forward KLT.
- Forward–Backward KLT with 1.5 px consistency gate.
- Fixed-template recovery with 21 px patch, 32 px search radius, NCC 0.75 and four-miss abstention.
- ORB + RANSAC partial-affine background transform. This is a camera-motion diagnostic, not a point-identity tracker.
- SmolVLM2-500M two-image endpoint point/ABSENT baseline, one query per camera per scene; tolerant parser accepts x y and x,y coordinate forms, based on DEV validation only.

## Primary metrics

- PCK@8 over all in-frame projected 3D track positions (explicitly labeled projection metric where visibility is unknown).
- PCK@8 on exact-snapshot depth-visible positions.
- Accepted coverage, accepted precision, abstention, false acceptance on depth-occluded positions and out-of-frame positions.
- ORB-affine reprojection error against the release-pose camera-only projection of the initial 3D point.
- Image-plane versus world-only projected motion direction, camera-motion dominance, per-camera pose translation/rotation.
- CPU P50/P95 per target frame; VLM endpoint PCK/ABSENT and output-format rate.
- Scene-cluster bootstrap intervals; no TEST-based parameter selection.

## Known limitations

The public Rerun page explicitly says viewer tracks are subsampled when a sequence has more than a few hundred tracks; this batch therefore evaluates the RRD display subset, not all metadata-declared tracks. The official full arrays were not available in the RRD index at preparation time. The exact-snapshot point-cloud visibility is a depth-derived proxy and does not replace full per-frame per-view visibility labels. Results must be reported as a public RRD-viewer-subset study, not as a full TAPVid-MV benchmark result.

## Provenance

- Source index SHA-256: eb9ff78058eca50479af98eca5bba21716dcec183ce83d8e70349d9cea58f2f5
- Source dataset: DROID; CC BY 4.0 attribution required.
- TAPVid-MV RRD format: Rerun SDK 0.35.0.
