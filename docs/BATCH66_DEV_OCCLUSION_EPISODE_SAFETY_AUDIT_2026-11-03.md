# Batch 66 — Occlusion-episode safety audit on pinned DEV (2026-11-03)

## Decision: safety not established; do not deploy new gates

**Retrospective exploratory DEV diagnostic, NOT a fresh independent blind TEST.** This audit reuses 36 previously studied Batch 56 natural-video DEV scenes and 144 official point queries. It does not tune trackers, use TEST labels, introduce a new production method, claim camera 3D motion recovery or interfere with frozen Batch 59.

**Research question:** does a low *scene-macro per-frame occlusion false-acceptance rate* hide an unsafe, persistent block of accepted predictions while the target remains invisible? Does independent tracker consensus reduce the **number of affected occlusion events**, or merely reduce some false-accepted frames while failing on a catastrophic episode?

## Frozen provenance and definitions

- Source of truth before this research: GitHub main `7fa0262dd7fc813e4a90bd82cb8074dc1abc51a3`. The Batch 66 development checkout was cloned from GitHub main (not copied from a stale local tree).
- Frozen Batch 56 DEV predictions, selection and scoring are validated with their existing SHA-256 pins, public query manifest and source commit. Selected base tracker: `appearance_fb_055`.
- Only 36 named DEV scenes `kinetics_00..35` are scored. The existing tested oracle streaming reader parses exactly these scenes and **stops before the first TEST key**. TEST predictions and TEST oracle records are neither opened nor parsed.
- Each point query defines a start frame `t0`. Scoring begins at `t0 + 1`. An **occlusion event** is a maximal contiguous run of oracle `occluded=True` frames *after* that query time, independently for each query. This avoids treating successive frames as independent natural scenes.
- A **false-accepted frame** is a frame where the oracle marks the target occluded but the method still emits an accepted prediction. The event is **affected** if at least one such frame occurs.
- A **false-accept run** is a contiguous succession of such frames within one occlusion event. Fixed, diagnostic run thresholds: **1, 3, 10, 30 and 60 frames**. False-accept events separated by rejected frames are not merged into a longer run.
- Four **pre-existing frozen** methods only: base; template-consensus `4px`; affine-consensus `3px`; and both. All vetoes use precomputed unscored predictions and only reject coordinates. We do not optimize thresholds, use future frames in the inference gate, or propose an oracle-derived runtime feature.
- The **10% scene-level false-accept budget** is the existing research threshold. Supported scenes have at least one occlusion frame. Scenes without such frames are excluded from the rate denominator and not counted as zero-risk passes.
- The pinned DEV scorer's exact per-scene baseline counts are replayed before report creation. All reported analysis artifacts remain local and are exclusive-create with SHA-256 sidecars.

## Observed results (old DEV; no new blind generalization claim)

There are **214 occlusion events** and **10,797 occluded frames** among 36 DEV scenes (33 scenes have occlusion support).

| Variant | Events with any false acceptance | Events with run >=10 | Events with run >=30 | Longest run | False-accepted frames | Scene-macro false acceptance | Scenes >10% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen base | 28/214 | 6/214 | 2/214 | **117** | 317 | 5.123% | 4/33 |
| Template 4px | **11/214** | 3/214 | 1/214 | **117** | 162 | 1.165% | 1/33 |
| Affine 3px | 22/214 | 6/214 | 2/214 | **117** | 294 | 4.765% | 4/33 |
| Both | 9/214 | 3/214 | 1/214 | **117** | 159 | 1.109% | 1/33 |

The base has 16 events with a run >=3, 6 with >=10, 2 with >=30, and 1 with >=60. Template has 3 with >=3, 3 with >=10, 1 with >=30, and 1 with >=60. The long episode does not disappear even when both independent geometric vetoes agree.

**Temporal concentration:** on `kinetics_17`, the template veto still accepts **139/526 occluded frames (26.43%)** and preserves a **117-consecutive-frame** failure. This one scene accounts for **139/162 (85.8%)** of template's false-accepted occlusion frames across all 36 scenes; the 117-frame single run accounts for **72.2%** of template's total. Neither the image-plane trajectory nor agreement between trackers proves target identity or correct visibility. Whether the trackers followed a particular distractor has not been causally established by this audit.

On `kinetics_26`, by contrast, template reduces the previous worst run **42 to 13 frames** and the scene's false-accepted count **45 to 14**. The safety outcome is *scene-dependent* rather than uniformly effective.

**Cost:** template reduces false-accepted frames 317 -> 162 (155 fewer) but loses **2,136 correctly accepted visible frames** (9,230 -> 7,094) and decreases micro visible coverage from **46.84% to 35.33%**. Adding the affine veto on top reduces only 3 more false-accepted frames (162 -> 159) while losing another **552 correctly accepted visible frames** (7,094 -> 6,542). The control cannot claim an improved safety–utility frontier without a fresh, separately frozen test.

**Different denominators:** event micro affected rates (base 28/214 = 13.08%; template 11/214 = 5.14%) are not scene-macro false-accept rates (5.123% and 1.165%). The 214 episodes can be correlated within the same query and scene; they cannot be treated as 214 independent trials or used to certify a probability of future-scene safety.

## Interpretation and research consequence

Batch 66 explicitly falsifies the assertion that a low *average* frame false-acceptance rate means the absence of sustained invisibility violations. Template consensus helps many episodes, but leaves the severe 117-frame failure intact. Further tightening consensus without independent target-visible/identity evidence risks exchanging useful visible coverage for marginal safety gains.

The next prospective design should **pre-register both** frame-level false acceptance and **event-level** affected-event rate, longest contiguous wrong-accept run, and unsupported-scene counts *before* opening any TEST labels. Events must be nested under distinct scenes for uncertainty analysis. A tracking coordinate consistent with background motion is not proof of true relative object motion; affine image motion is not 3D camera motion.

This is an **offline annotation-dependent evaluator**, not an inference gate. It never reads TEST data and **does not modify frozen Batch 59**. The currently running Batch 59 should complete its established stage sequence and safety audit unchanged; new event-level metrics can be incorporated into a future plan after it completes, with an explicit new frozen SHA and independently selected videos.

## Reproduce (authorized Tang only)

```bash
cd /home/you/projects/Multimodal-VLM-Video-Generation-Lab/repo-batch66-dev-clean
../.venv/bin/python -m pytest -q tests/test_batch66_dev_occlusion_episode_audit.py
../.venv/bin/python -m experiments.batch66_dev_occlusion_episode_audit
../.venv/bin/python -m pytest -q
```

The output command is **exclusive create**: reruns must use a *different* output path or fresh checkout. Do not delete or overwrite the prior artifact to rerun.

Local output only: `outputs/research/batch66_dev_episode_safety.json` and sidecar SHA-256 **`6a631534f8adb2dce11e89764994d38dcfefaa299be5ec31f8c78d97e7467e37`**. This contains per-scene aggregate label-dependent diagnostics and must **not** be committed or disclosed as raw private oracle traces. Commit source, tests and this aggregated report only. No video, frame array, checkpoint, token or private labeling is published.
