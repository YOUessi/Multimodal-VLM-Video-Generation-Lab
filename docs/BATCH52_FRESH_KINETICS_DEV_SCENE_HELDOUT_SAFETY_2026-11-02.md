# Batch52 — Fresh class-disjoint natural DEV: scene-held-out safety stability

**Date:** 2026-11-02 UTC. **Scope:** retrospective DEV-only reliability audit.
**Status:** evaluation tooling and tests completed, not a tracker/VLM improvement and
not a newly opened TEST. No change to running Batch48B freeze or its selection.

## Why

Batch46E's transferred tracker suffered 36% invisible false acceptance on its
independent TEST. Earlier Batch47–51 counterexamples used six repeatedly studied
natural DEV clips with synthetic motion layers. The frozen Batch48B now contains
**six NEW, action-class-disjoint DEV natural videos** and six reserved TEST videos.
Its original evaluation selected a tracker using only DEV, before any TEST score.
We inspect whether the DEV safety decision survives leaving out an entire scene.

A frame-wise random split would be misleading: a single video's neighboring
frames share camera motion, lighting, actors, and occlusions. Therefore the unit
of holdout here is a complete video. We recompute the frozen candidate chooser
on five scenes and assess the selected method on the excluded sixth, six times.

## Protocol and anti-leakage controls

- Source code: `experiments/dev_scene_heldout_safety_audit.py`. Synthetic regressions:
  `tests/test_dev_scene_heldout_safety_audit.py`.
- Inputs exclusively from **the untouched Batch48B frozen workspace**:
  `outputs/kinetics_batch48/dev_results.json`,
  `outputs/kinetics_batch48/dev_predictions.json`, and
  `configs/kinetics_batch48.json`. Never read TEST frames, predictions,
  or TEST oracle; no dependence on raw videos.
- The DEV prediction SHA sidecar must match its content, the scorer's prediction
  SHA must match it, and source/config/manifest hashes must agree. Reject
  any `split=test`, non-DEV scene, missing method, malformed rate, or changed config.
- Candidate methods, order, constraints and fallback are from the **pre-existing**
  frozen Batch48B config: visible coverage ≥30%, invisible false accept ≤10%,
  rank eligible candidates by scene-macro Jaccard. No post-hoc threshold tuning.
- Report and SHA-256 sidecar are **write-once** in Tang's local outputs.
  New methods from Batch49–51 are not included as candidates in this audit.
- The six folds are correlated and do not constitute six independent TEST sets.
  Held-out results are a DEV robustness diagnostic only.

## Executed six-scene results

Source Batch48B freeze SHA:
`9d86f354f4cbed039ef8ea47f14c8eb37df374b3`.

Complete six-scene DEV selected `klt_fb`; the exact same method was selected
on **all six five-scene training subsets**. Eight methods passed full-DEV
safety constraints. In all five folds with ground-truth occlusions, the
held-out method met the point-estimate coverage/safety criteria; one video
had **zero invisible frames**, so its occlusion-safety verdict is **unevaluable**.

| Held-out natural DEV scene | Visible coverage | Invisible false accept | Invisible support | Scene Jaccard | Mean luma | 2D affine translation median |
|---|---:|---:|---:|---:|---:|---:|
| 00 / trapezing | 39.9% | 0.00% | 374 | 0.315 | 131.3 | 1.46 px |
| 01 / carving pumpkin | 91.9% | 0.00% | 57 | 0.819 | 112.2 | 0.05 px |
| 02 / triple jump | 70.2% | 1.19% | 419 | 0.576 | 122.6 | 1.67 px |
| 03 / burping | **32.2%** | **8.59%** | 291 | **0.176** | **24.5** | 2.27 px |
| 04 / playing volleyball | 46.6% | 0.23% | 435 | 0.416 | 104.3 | 1.83 px |
| 05 / sailing | 59.4% | **not measurable** | **0** | 0.411 | 139.4 | 0.43 px |

Batch48B six-scene macro `klt_fb`: visible coverage **56.7%**, invisible
false accept **2.00%** (macro over scenes with invisible support), scene-macro
Jaccard **0.452**. Source's tracker-only CPU p50 is **0.091 ms/target-frame**,
excluding video decoding, shared affine computation, and end-to-end application.

This reveals a concrete **domain-shift concern even before opening TEST**:
in the low-light scene 03 (mean luma 24.53 and Laplacian texture variance
138.4), coverage barely clears 30%, false acceptance approaches 10%, and
Jaccard falls to 0.176. This is descriptive association, not proof darkness
causes the failure. The bright scene 05 has no occluded support and cannot
validate the main safety requirement, despite contributing visible coverage.

SmolVLM2-2.2B remains a **distinct three-image reference-crop baseline**.
In the frozen Batch48B DEV gate its balanced accuracy was **31.0%** and
not-visible false-accept was **57.9%** (11/19), so the frozen TEST VLM gate
should block further VLM inference; do not compare its 48 temporal query
judgments directly with dense-frame PCK/Jaccard as if budgets matched.

## Interpretation and decision

**No production promotion.** A stable LOOCV method choice is not sufficient
evidence of safe transfer: one low-light scene nearly violates the safety
budget and another scene provides no occluded evidence. With only six scenes,
neither micro-frame counts nor independence assumptions justify narrow
generalization confidence intervals. The 2D affine diagnostic is a dominant
motion-layer proxy, **not a physical camera pose or identified static background**.

The next algorithm study should use additional NEW DEV clips intentionally
spanning low light, low texture, camera translation/parallax and partial/full
occlusion; distinguish observed image-plane motion from real background
motion with explicit semantic/static-scene evidence. Independently freeze
candidates and evaluate on future class-disjoint TEST, not Batch48B's current
sealed TEST answers. The current Batch48B 22-stage pipeline is left unchanged.

## Reproduce and provenance

```bash
cd /home/you/projects/Multimodal-VLM-Video-Generation-Lab/repo-batch52-dev
../.venv/bin/python -m pytest -q tests/test_dev_scene_heldout_safety_audit.py
../.venv/bin/python experiments/dev_scene_heldout_safety_audit.py \
  --batch-root ../repo-batch48b-tested \
  --output ../outputs/research/batch52_dev_heldout_safety_v1.json
../.venv/bin/python -m pytest -q
```

The report was already generated once; reproductions should use a *new*
output basename since the original is write-once. Result:
`outputs/research/batch52_dev_heldout_safety_v1.json`, SHA-256:
`506522a1e62b69cf7b55e35d4b7e392b07132d6160c9e17ad11f4b50bde6e6ea`.

Tests: **23/23 added; 676/676 entire repository passed**, 0 failures,
three pre-existing Pillow deprecation warnings. The complete local test log
is `outputs/research/batch52_full_pytest.log`. No raw media, model
weights, private labels, serialized point tracks, or secrets are published.
