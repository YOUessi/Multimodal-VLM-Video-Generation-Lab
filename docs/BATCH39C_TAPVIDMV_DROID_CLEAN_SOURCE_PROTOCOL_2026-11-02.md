# Batch 39C — TAPVid-MV DROID clean-source multi-view evaluation protocol

Date: 2026-11-02

## Research question

Separate image-plane point displacement into (a) camera-induced displacement and (b) point motion relative to a fixed world camera. Compare an RGB-only background affine diagnostic with the released calibrated camera pose and 3D point track, and quantify when 2D trackers preserve point identity under camera motion.

## Dataset and split

- Use the already downloaded, SHA-256 verified TAPVid-MV DROID Rerun 0.35.0 viewer recordings.
- 12 recordings from 12 distinct collection-group prefixes: 4 DEV and 8 TEST, 3 synchronized RGB cameras each.
- 72 public queries total: exactly 2 frame-0 depth-visible points per camera per scene. The first attempted three-per-camera design was not feasible because one TEST camera had only two frame-0 depth-visible candidates. The corrected split is fixed by metadata and frame-0 visibility only.
- The public Rerun viewer recordings display a subsampled track set (up to 400 tracks per frame), not the full TAPVid-MV track arrays.
- Exact-snapshot depth point clouds provide a depth-derived visible/occluded/out/unknown proxy. Unknown visibility is not silently treated as absence.

## Frozen methods

1. Stationary initial coordinate.
2. KLT forward.
3. KLT forward-backward with 1.5 px consistency gate.
4. Fixed-template recovery with 21 px patch, 32 px search radius, NCC 0.75, four-miss abstention.
5. ORB + BFMatcher ratio test + RANSAC partial affine from frame 0 to each frame; diagnostic only, not a point tracker.
6. SmolVLM2-500M two-image endpoint baseline, one query per camera per scene. The tolerant parser accepts coordinate forms and explicit ABSENT. No intermediate frames are shown to the VLM.

## Metrics

- PCK@8 over in-frame projected 3D point positions, clearly labeled as a projection metric when depth visibility is unknown.
- PCK@8, accepted coverage and precision on exact-snapshot depth-visible positions.
- False acceptance on depth-occluded and out-of-frame positions; abstention; unknown/missing counts.
- CPU P50/P95 milliseconds per target frame.
- ORB-affine reprojection error against the calibrated camera-only projection of the initial 3D point.
- Screen/world-only direction cosine, affine-compensated/world-only direction cosine, oracle camera-subtracted/world-only direction cosine, and a clearly labeled camera-motion dominance diagnostic.
- Scene-cluster bootstrap intervals; scene, not frame, is the resampling unit.

## Protocol and leakage controls

- First run the complete source/data audit and project regression suite.
- Run CPU predictions on DEV without opening the private oracle, pin prediction bytes and source SHA, then score DEV.
- Run the VLM on DEV only. A predeclared interface gate requires at least 80% syntactically valid/explicit-ABSENT responses; this gate does not use coordinate accuracy.
- Only after the DEV gate passes, generate CPU and VLM TEST predictions without reading future tracks, future camera poses, depth clouds or private oracle files. Pin both prediction files and their SHA-256 sidecars before any TEST scoring.
- Read the private TEST oracle exactly once for CPU/VLM scoring after all label-blind predictions are sealed. No TEST metric is used for model, prompt, threshold or method selection.
- Prediction code refuses to stamp a source SHA when the Git worktree is dirty. The automation runner applies the same clean-tree rule before freezing a plan.
- The earlier Batch39B DEV prediction files were generated from an uncommitted worktree and are retained as exploratory artifacts only; they are not reused or scored in Batch39C.

## Metadata erratum

The existing frozen Batch39B manifest's human-readable query_contract string says three points per camera, while the frozen config and public query payload contain two. The original manifest, RGB arrays, query file and private oracle are not rewritten. A separately SHA-bound manifest-errata sidecar records the correction and verifies exactly two queries for each of the three cameras in every scene.

## Known limitations

- This is a public Rerun viewer-subset study, not a full TAPVid-MV benchmark reproduction.
- Depth-cloud visibility is sparse and derived; it is not an official per-frame, per-view binary occlusion annotation.
- ORB affine is an image-plane background diagnostic. It cannot recover depth-dependent parallax or metric 6-DoF camera motion by itself.
- The VLM sees only query and endpoint images, while CPU trackers see intermediate frames; this is an intentionally unequal-information baseline and must be stated.
- DROID source recordings are CC BY 4.0 and require attribution. TAPVid-MV annotation terms remain subject to their own release terms.
