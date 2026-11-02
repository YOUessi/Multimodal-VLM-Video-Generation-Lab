# Batch50 — DEV-only peripheral motion assumption and identifiability counterexample

Date: 2026-11-02 UTC. **Not** a camera-pose estimator, production tracking update, model promotion, or a new independent TEST. Six previously used SHA-pinned Kinetics **DEV** videos (8 frame anchors per video), modified with deterministic 2D image-plane transforms only. Input media and generated JSON stay on Tang. No Batch46E opened TEST or Batch48B frozen TEST is used for method selection.

## Scientific bottleneck

Batch46E's already-opened TEST showed a large invisible-target false-accept burden (427/1346 = 31.72% micro). This observation motivates a *new* DEV safety investigation, **not** threshold tuning on TEST. Batch49's competing-affine experiment failed on 25/25 accepted dominant-foreground cases, despite ordinary spatial residual checks. It cannot identify whether a global-looking coherent motion is background or foreground.

The predeclared hypothesis tested here is modest: **under the additional assumption that some frame-edge features actually belong to static background**, disagreement between a feature-supported boundary affine and the dominant affine can **falsify** the claim that one affine explains the whole image. Agreement of the two affines **does not identify camera motion or prove background**.

## Implementation and protocol

New module: `experiments/dev_peripheral_anchor_motion.py`. New tests: `tests/test_dev_peripheral_anchor_motion.py`. All decisions are input-label-free. Correspondences are forward-backward LK matches. A peripheral ring is defined as any point within 10% of any image edge (fixed before this run, no search). A peripheral affine is fitted only if there are >=16 ring matches across >=3 image quadrants, with >=12 RANSAC inliers and >=60% inlier fraction. A global hypothesis is accepted only if the **unchanged Batch47E original crossfit gate** accepted AND the independently fitted ring affine is supported AND its median disagreement on nine predetermined image locations is at most 3px. Failure to fit or to acquire distributed support leads to abstention. These thresholds are exploratory and **not** optimized against previously viewed TEST.

Controls are the five existing Batch47E deterministic synthetic image-plane conditions plus an explicit **completely occluded background** negative control. In the latter, every visible pixel is synthetic foreground with displacement (-11,+9) and the latent, unobserved background has (+6,-4). The latent mismatch is purely a *counterexample to identifiability*, not an empirical background error measured from pixels.

An oracle synthetic translation is used **only after** both gate decisions. Evaluation never accesses TEST IDs or private tracking annotations. All six input members and manifest are SHA-pinned; write-once result includes source SHA. The output stores per-scene and overall aggregates but no raw frames.

Run:

```bash
cd /home/you/projects/Multimodal-VLM-Video-Generation-Lab/repo-batch50-dev
../.venv/bin/python -m pytest -q tests/test_dev_peripheral_anchor_motion.py
../.venv/bin/python experiments/dev_peripheral_anchor_motion.py \
  --selection ../data/tapvid_kinetics_batch46/preflight/selection.json \
  --shards ../data/tapvid_kinetics_batch46/shards \
  --output ../outputs/research/batch50_dev_peripheral_anchor_v1.json
../.venv/bin/python -m pytest -q
```

## Actual executed results: 6 DEV scenes × 8 anchors × 6 conditions = 288 pairs

| Condition (synthetic 2D) | Prior crossfit accepts /48 | Peripheral gate accepts /48 | Wrong accepted / new accepted | Ring-supported pairs /48 | Median **extra** CPU ms |
|---|---:|---:|---:|---:|---:|
| Clean translation | 36 | 34 | 0/34 | 36 | 1.76 |
| Lighting shift | 35 | 34 | 0/34 | 35 | 1.92 |
| Small independent foreground | 0 | 0 | n/a | 14 | 2.65 |
| **Dominant foreground** | **26** | **0** | n/a | 11 | 2.71 |
| Foreground and occlusion | 0 | 0 | n/a | 3 | 2.68 |
| **Whole-frame foreground, no visible bg** | **48** | **48** | **48/48 latent mismatches** | 48 | 2.75 |

### Interpretation and negative result

**Conditional success under a convenient visible-border assumption:** on synthetic dominant foreground with corners/edges left as true background, the new *reject-only* gate abstained on all 26 otherwise-wrong accepted cases. It also lost 2/36 clean and 1/35 lighting accepted cases. This is a selective-prediction tradeoff; zero accepted dominant-foreground cases means **no measured precision or improvement in target tracking**, not 100% tracking accuracy.

**Critical falsification:** when the foreground occupies the entire frame, all 48 pairs pass BOTH gates while all 48 disagree with the *unobservable* latent background displacement. This is not a violation of image measurement—there was **no visible background to identify**. A peripheral prior cannot certify that boundary pixels are background. We therefore **reject** any scientific claim that this method solves camera-motion / foreground ambiguity in general. Without trusted semantics, depth, independently anchored landmarks, or multi-view geometry with identifiable static scene structure, the ambiguity persists.

This is a reused DEV-set synthetic ablation rather than fresh class-disjoint natural TEST. Fixed synthetic foreground shapes, textured patch distribution and hand-chosen ring prior are favorable to the new diagnostic. The result cannot establish real natural-domain utility or occlusion safety. The reported CPU time is the extra ring audit including redundant LK matching; not full tracker CPU end-to-end latency.

## Tests, evidence and deployment safety

- Targeted tests: **21/21 passed**. Verify spatially distributed support, rejection of foreground-majority motion with background edges, fail-closed insufficient support, invalid input handling, full-frame nonidentifiability, deterministic synthetic generation, DEV-only selection pin and oracle-after-gate ordering.
- Full suite: **623/623 passed**, three existing Pillow deprecation warnings.
- Local write-once report: `outputs/research/batch50_dev_peripheral_anchor_v1.json`; SHA256 `98cf9f4c71b0a567c41c86636e8097d92d1cc0132dc311a764861552705339f5`.
- GPU not used. Media, weights, private labels, credentials and raw experiment outputs are **not** committed.
- Batch48B remains isolated on its original frozen SHA `9d86f354f4cbed039ef8ea47f14c8eb37df374b3` and original 22-stage evaluation. Neither the running `R00` nor unique `MVVG_PROJECT4_10MIN` cron is modified.

**Decision:** keep this method as a DEV-only diagnostic and documented counterexample. Do NOT ship into Batch48B, re-open Batch46E TEST for adjustment, or claim improved real camera compensation. Prioritize finishing Batch48B independent frozen tracking / invisible-target acceptance / CPU / VLM evaluation. Any further work on reliable background anchors must be evaluated on fresh DEV with annotated visible foreground/background semantics and then a new class-disjoint frozen TEST.
