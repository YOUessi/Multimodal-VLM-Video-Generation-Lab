# Batch49 — Two-hypothesis 2D motion falsification on DEV frames

Date: 2026-11-02 UTC. Scope: **reused six Batch46 DEV natural-video clips, deterministic synthetic image-plane transformations only**. This is NOT an independent TEST and NOT physical camera pose estimation. No Batch46E or Batch48 TEST frames/answers used for model selection or parameter tuning.

## Motivation

Batch46E completed 22/22 frozen stages. On its already-opened independent TEST, affine-stabilized KLT produced 427 incorrect visible predictions on 1,346 actually invisible target frames (31.72% micro invisible false-accept). Its DEV number was 343/1,833 (18.71%). These are archival evaluation observations, not permitted for tuning new gates. The SmolVLM2 DEV admission was blocked (balanced accuracy 0.35 vs a frozen minimum 0.55), so TEST VLM inference was correctly skipped.

Batch47E's synthetic counterexample demonstrated that a dominant moving foreground can pass spatial quadrant cross-fitting even if the fitted warp is wrong for the known synthetic background: 26 accepted / 26 incorrect. This invalidates any interpretation of low cross-fit residual as a sufficient *background motion* certificate.

## Hypothesis and implementation

The new **competing-motion diagnostic** fits an alternative affine model ONLY on image correspondences outside the initial affine model's 3-pixel residual tolerance. A second hypothesis is considered material if it has at least 12 inliers, at least 10% of all matches, spatial evidence from at least two quadrants, and median disagreement at nine sample image locations of at least 4 pixels. Alternative fit uses 2-pixel RANSAC tolerance. These are prior exploratory thresholds, not selected from TEST.

If a strong incompatible second model is present, the diagnostic rejects an otherwise-accepted original spatial gate. It does **not** choose which model is the background. The experiment preserves the original 47E synthetic conditions and the old gate exactly, reusing the SHA-pinned selection and manifest. Known synthetic background displacement is scored only AFTER both decisions. Media and generated row-level outputs remain local.

Execution:

```bash
cd /home/you/projects/Multimodal-VLM-Video-Generation-Lab/repo-batch49-dev
../.venv/bin/python -m pytest -q tests/test_dev_competing_motion_audit.py
../.venv/bin/python experiments/dev_competing_motion_audit.py \
  --selection ../data/tapvid_kinetics_batch46/preflight/selection.json \
  --shards ../data/tapvid_kinetics_batch46/shards \
  --output ../outputs/research/batch49_competing_motion_dev_v2.json
```

The run uses six DEV videos × eight anchors × five paired conditions = **240 pairs**. The output is write-once, source- and selection-hash pinned, and excludes TEST IDs and private target labels.

## Real executed DEV-only diagnostic results

| Synthetic condition | Old gate accepted / 48 | Newly accepted / 48 | New wrong accepted / new accepted | Detected competing layer / 48 | Median diagnostic extra CPU ms |
|---|---:|---:|---:|---:|---:|
| clean affine | 36 | 36 | 0/36 | 0 | 1.56 |
| lighting shift | 35 | 35 | 0/35 | 0 | 1.66 |
| local foreground | 0 | 0 | undefined (0 accepted) | 34 | 2.63 |
| **dominant foreground** | **26** | **25** | **25/25 (100% wrong)** | 11 | 2.98 |
| foreground + occlusion | 0 | 0 | undefined (0 accepted) | 31 | 2.75 |

Full result: local `outputs/research/batch49_competing_motion_dev_v2.json`, not committed. CPU timing is an **incremental diagnostic** including a repeated LK correspondence pass, not the full real-time tracker latency. Input media decoding is excluded.

**Negative result:** This two-hypothesis gate did NOT solve the dominant-foreground ambiguity. In fact, 25/25 accepted dominant-foreground pairs remain wrong relative to known synthetic background motion. Rejecting one additional pair is not a credible algorithm improvement. While a second motion layer can falsify uniqueness, failing to detect one cannot certify a semantic background.

The reason is structural: two-image optical flow alone is invariant to relabeling a coherently moving full-frame foreground as the background. Unsupervised correspondence fitting cannot identify the true background layer under that ambiguity without additional independent evidence such as trusted background anchors, depth/geometry, or reliable object/background segmentation.

## Restrictions and decision

Do **not** ship, benchmark on opened Batch46E TEST, or promote the exploratory diagnostic to Batch48's frozen tracker. Batch48 has an independent frozen 12-video run with its own DEV-only candidate-selection and occlusion safety constraint. That experiment must finish, seal TEST predictions and only then open its private TEST labels.

Any next proposal must first quantify its real annotated DEV target coverage, wrong accepts, invisible false accepts, and CPU latency, including foreground-dominated scenes and missing-background cases; select solely on fresh DEV and evaluate once on a genuinely new class-disjoint TEST. This exploratory negative result is evidence against falsely confident camera compensation, not proof of a better motion estimator.

All historical shards, media, weights, oracles, and frozen states are preserved. No GPU contention, no secrets, no private media published.
