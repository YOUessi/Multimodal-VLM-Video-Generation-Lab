# Batch 67 — Scene-cluster uncertainty and influence audit (2026-11-03)

## Decision: no deployment approval; coverage–safety trade-off remains unresolved

**Retrospective exploratory analysis of the same previously studied Batch 56 DEV, not new data, not a new TEST, and not a performance improvement of any tracker.** Only a separately pinned **aggregate** Batch 66 report was opened; this analysis does **not** open private per-frame oracle, image/video, model checkpoints, VLM outputs, or TEST labels/predictions.

Batch 66 showed that the template-consensus veto reduces per-frame occlusion errors but leaves a 117-frame invisible-target acceptance episode. Batch 67 asks a different and necessary reliability question: how do the apparently favorable aggregate changes vary when the **unit of resampling is a scene**, and how dependent are these estimates on individual scenes or a six-scene block? We do not use the bootstrap to select a gate or change any existing threshold.

## Provenance and sealed inputs

- Starting GitHub source: `main` commit `e755d804fbfe8c163944d03a550dff000b393c81`.
- Only input: frozen aggregate `batch66_dev_episode_safety.json` whose bytes, SHA-256 sidecar and hard-coded expected SHA must all agree: `6a631534f8adb2dce11e89764994d38dcfefaa299be5ec31f8c78d97e7467e37`.
- Strict evidence checks require exactly `kinetics_00..35`, 36 DEV scenes and 144 queries, four pre-existing gate variants, DEV-only metadata, no TEST access, unchanged Batch 59, consistent per-gate denominators, integer support counts, and exact agreement with Batch 66 aggregate replay.
- A supported occlusion scene has at least one occluded frame; **unsupported scenes are not scored as zero risk**.
- Four variants are the existing, fixed `baseline`, `template_4px`, `affine_3px`, and `both_4px_3px`; no method selection, runtime inference, new tuning, or data augmentation occurs.
- All figures are derived from the pinned 36-scene DEV artifact. Source artifact and new individual-scene results remain local and are **not committed**.

## Verified DEV outcomes

| Original gate | Scene-macro false-accepted occlusion rate | Micro false-accept rate | Micro accepted-visible coverage | Affected event proportion | Longest error run | Scenes > 10% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base `appearance_fb_055` | 5.123% | 2.936% | 46.844% | 13.084% | 117 frames | 4/33 |
| Template-consensus 4px | **1.165%** | **1.500%** | **35.335%** | **5.140%** | **117 frames** | 1/33 |
| Affine-consensus 3px | 4.765% | 2.723% | 43.012% | 10.280% | 117 frames | 4/33 |
| Template + affine | 1.109% | 1.473% | 32.703% | 4.206% | 117 frames | 1/33 |

Template-versus-base *paired* DEV changes: **-3.958 percentage points** scene-macro false-acceptance, **-7.944 percentage points** affected-event rate, **-11.509 percentage points** visible-coverage, and **-10.092 percentage points** unconditional visible PCK@8. All these are changes relative to the same, already inspected DEV scenes; they cannot be interpreted as new generalization gains.

### Paired whole-scene bootstrap (4096 resamples, fixed seed 20261103)

We draw 36 complete scene records with replacement per replicate. The baseline and template records from each selected scene stay paired; the four queries, frames, and occlusion events within the scene are **never independently resampled**.

The empirical 2.5–97.5% percentile interval for the **template minus baseline** scene-macro occlusion false-acceptance change is **[-8.380, -0.906] percentage points**. The corresponding empirical interval for visible-coverage change is **[-14.960, -8.121] percentage points**. For affected-event proportion the interval is **[-12.764, -4.366] percentage points**.

Only **1/4096 (0.0244%)** resamples jointly displayed lower scene-macro false acceptance and at most five percentage points of absolute visible-coverage loss. **Five percentage points is purely an illustrative sensitivity condition, not a pre-registered approval threshold**. This arithmetic shows why a safety gain does not automatically imply an acceptable safety–utility trade-off.

These descriptive intervals rely on reweighting the *observed scenes* only. The 36 scenes were already studied. They do not cover arbitrary future domain shift, provide an independent guarantee, or establish scene IID. The bootstrap replicates are also not 4096 new independent experiments.

### Leave-one-block and event-concentration stress audit

Six fixed blocks each contain six consecutive DEV scene IDs. Deleting any one block keeps the sign of the template-vs-base scene-macro false-acceptance difference negative, between **-2.604 and -4.572 percentage points**, but cannot establish safety on an unobserved block. Crucially, removing the block `kinetics_12..17` makes the **observed longest template false-accept run** fall from **117 frames to 13 frames**, demonstrating concentration of severe failure in a narrow scene subset.

The single scene `kinetics_17` contributes **139/162 (85.80%)** of all template false-accepted occlusion frames; the top two scenes contribute **94.44%**. This concentration is not a causal proof of the visual failure mechanism, but it is a strong reason to include worst-event duration and per-scene tail safety in *future* pre-registered evaluation.

## Scientific interpretation and boundaries

1. **Average safety versus event-tail safety are different estimands.** Lower scene-macro false-acceptance, or even lower mean event-affected rate, can coexist with a continuous 117-frame invisible-target error. No bootstrap interval on the mean certifies a bound on longest future failure.
2. **Safety versus useful coverage is a paired multi-objective question.** Reject-only template gating loses substantial correct visible tracking and unconditional PCK, even when mean occlusion metrics improve.
3. **Scene-blocked uncertainty is necessary, not sufficient.** Frame-level or episode-level pseudo-independent confidence intervals would overstate support. Our scene-bootstrap is only descriptive and cannot simulate novel classes, lighting, camera motion, or occlusion mechanisms.
4. **No algorithm promotion.** The results do not justify the template gate, background-relative residual, RGB/NCC gates, or temporal consensus for production. The target's visibility must be inferred from independent causal evidence at inference time; DEV labels are never a gate.
5. **No frozen-run mutation.** Batch 59's existing 42-video class-disjoint plan and its sole `MVVG_PROJECT4_10MIN` cron remain unchanged. Its frozen protocol must finish before any future independent event-level safety experiment can be registered on a new SHA and new videos.

## Reproduction / audit

On authorized Tang, from the Batch 67 independent source checkout:

```bash
../.venv/bin/python -m pytest -q tests/test_batch67_dev_scene_cluster_robustness.py
../.venv/bin/python -m experiments.batch67_dev_scene_cluster_robustness \
  --source ../repo-batch66-dev-clean/outputs/research/batch66_dev_episode_safety.json
../.venv/bin/python -m pytest -q
```

The output is **exclusive-create** and SHA-256 pinned. Re-run with a new output pathname rather than overwriting any research result. The new audit checks all metadata and replayed counts before reporting; it cannot be used to open TEST or to alter a frozen running plan. Keep aggregate private diagnostic JSON and its sidecar on Tang. Commit only this reusable analysis implementation, synthetic tests, and this document.

**Measured output SHA-256:** `ac1dfe500b7eb4b2fe7112368dfcbf596fc666115fd5b95eddb2698aff9c2b6a`.

**Next decision gate:** prioritize completing Batch 59's currently frozen data and held-out TEST pipeline; for a subsequent prospective batch, specify episode metrics, risk/coverage acceptance requirements, and scene-level uncertainty **before** selecting or scoring new independent videos.
