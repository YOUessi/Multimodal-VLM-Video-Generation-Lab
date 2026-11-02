# Batch51 — DEV-only temporal anchor provenance, identity limits, and abstention controls

Date: **2026-11-02 UTC**. Development-only exploration; **not** a new independent natural TEST, a production tracker replacement, physical camera-pose estimate, or an accuracy gain proven in the target-tracking benchmark.

## Research question and controls

Batch49's secondary affine and Batch50's edge-based prior both failed at determining *which motion layer actually belongs to static background*. In particular, peripheral agreement can wrongly certify a dominant moving layer whenever foreground occupies the image boundary.

This batch tests an alternative **conditional** assumption: the first frame of a three-frame sequence is known clean in the *synthetic control*, so features observed at t0 can be tracked through t1 and t2 without immediately reselecting potentially foreground-dominated t1 features. Forward/backward LK tracks are preserved across both hops; the last hop estimates an independent affine using >=16 surviving original features, >=12 RANSAC inliers, >=60% inlier fraction, >=3 image quadrants, and median model-disagreement <=3px. A result is accepted only when the unchanged Batch47E original crossfit gate also accepts.

Crucially, **historical pixel tracking does not prove static-scene identity**. We explicitly include failure controls, and the function always reports `background_identified=false`. All thresholds are fixed as exploratory hypotheses, not tuned using opened TEST outcomes.

## Dataset and replay

Input: the same six pre-existing SHA-pinned Kinetics DEV clips and manifest used in the earlier Batch47E–50 diagnostics. Six scenes × eight sampled anchors × seven deterministic 2D synthetic conditions = **336 generated three-frame trials (48 per condition)**. There is no access to original private TEST annotations, and generated motion does not establish actual 3D camera pose. All decisions are computed **before** scoring against the synthetic known translation.

```bash
cd /home/you/projects/Multimodal-VLM-Video-Generation-Lab/repo-batch51-dev
../.venv/bin/python -m pytest -q tests/test_dev_temporal_anchor_provenance.py
../.venv/bin/python experiments/dev_temporal_anchor_provenance.py \
  --selection ../data/tapvid_kinetics_batch46/preflight/selection.json \
  --shards ../data/tapvid_kinetics_batch46/shards \
  --output ../outputs/research/batch51_dev_temporal_provenance_v1.json
../.venv/bin/python -m pytest -q
```

The generated JSON is **write-once** and includes SHA pins, code hash, per-class counts, conditional wrong-accepted rates, CPU diagnostics, and explicit limitations. The file remains on Tang; no video, weights, private oracle, or secrets are committed.

## Executed results

| Synthetic condition | Baseline accepted /48 | Peripheral accepted /48 | Temporal-provenance accepted /48 | Wrong baseline / peripheral / provenance accepted | Median **extra** CPU ms |
|---|---:|---:|---:|---:|---:|
| Clean global translation | 36 | 34 | 36 | 0 / 0 / 0 | 2.32 |
| Lighting shift | 34 | 33 | 34 | 0 / 0 / 0 | 2.55 |
| Center moving foreground | 1 | 0 | 0 | 1 / 0 / 0 | 2.62 |
| **Moving foreground dominates image periphery; visible background island** | **29** | **29** | **0** | **29 / 29 / 0** | 2.48 |
| Background only in one corner | 2 | 2 | 0 | 2 / 2 / 0 | 2.30 |
| Full-screen unrelated foreground (latent background hidden) | 0 | 0 | 0 | 0 / 0 / 0 | 0.006 |
| **Full-screen visually equivalent moving foreground (latent background hidden)** | **36** | **34** | **36** | **36 / 34 / 36 latent mismatches** | 2.36 |

**Conditional positive:** on the previously researched six DEV clips with *synthetically constructed* clean t0 and majority foreground at image boundaries, provenance refuses all 29 baseline false accepts. Unlike Batch50's ring, this controlled history protects against some edge-dominated distractors. There is no accepted prediction in this failure category, so precision is **undefined**, not 100%. Clean-condition coverage is retained (36/48), which is also specific to these generated controls.

**Important negative #1 — unidentifiable motion:** even with a clean historical t0, a full-screen foreground that is pixelwise equivalent to a coherent image translation passes the gate in 36/48 cases, while contradicting the arbitrarily designated *hidden* background translation. No method based only on the observed images can infer that unobserved latent motion. This is not an observed visible-background error.

**Important negative #2 — provenance identity hallucination:** an additional **deterministic random-texture falsification** (unit test, outside the 336 DEV trials) constructs an edge-dominant foreground with an actually visible central background. The baseline, peripheral method, **and new provenance method all accept a wrong foreground affine**. Two-hop forward/backward consistency is not proof that pixels retain the same object identity under occlusion and texture aliasing. This test is kept permanently to prevent claims of a generally safe gate.

**Selectivity and latency:** reported timings are extra provenance matching and fitting on CPU (median ~2–3 ms for conditions that supply a primary model); they exclude original primary estimation, decoding, video inference, memory, and end-to-end CPU tracking latency. Abstention is not counted as a correct target-tracking prediction.

## What was and was not validated

- The **logic of conditional temporal evidence** and its limits were stress-tested in 336 synthetic triplets derived from reused DEV videos, including lighting, foreground dominance, spatial collapse, fully hidden background, and identifiable/indistinguishable motion controls.
- Tests enforce input validation, fail-closed behavior, deterministic construction, write-once pins, visible-only gate decisions before synthetic oracle scoring, and failure on known adversarial texture.
- There is **no evidence of improved natural-domain Jaccard, invisible-target false-acceptance, VLM accuracy, or real-camera ego-motion**. Those require fresh semantic background/foreground annotations and an independently frozen TEST, not these reused DEV clips.
- Batch48B keeps its **original frozen SHA** and 22-stage independent evaluation. This new code is never injected into its running process. No new evaluation batch may contend for GPU.

## Decision

**Do not promote to the tracking production path.** The useful result is a clear set of testable observability conditions and a failure-producing counterexample. Future research should add independently verified static-scene identity (e.g., visible landmark semantics / depth and geometry) and an explicit *unknown* state for completely occluded backgrounds, test on newly collected DEV scenes, then freeze new algorithms for a distinct class-disjoint TEST. The highest priority remains completing Batch48B's real-world visible-coverage, invisible false-acceptance, Jaccard, full CPU latency, and VLM-baseline gates.

## Evidence artifacts

- On-device immutable JSON: `outputs/research/batch51_dev_temporal_provenance_v1.json`
- JSON SHA-256: `eddf8f3daddb78b7c49833f7eafaba0e0c3553dd4eb1381d3025a50ed4ffa55d`
- Full test log (untracked evidence): `outputs/research/batch51_full_pytest.log`
- All recordings and original archives remain untouched.
