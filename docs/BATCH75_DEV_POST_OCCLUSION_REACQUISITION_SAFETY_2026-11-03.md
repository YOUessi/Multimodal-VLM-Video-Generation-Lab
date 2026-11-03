# Batch 75 — Historical natural DEV post-occlusion reacquisition safety audit

Date: 2026-11-03 (Tang Asia/Shanghai). Repository: `YOUessi/Multimodal-VLM-Video-Generation-Lab`; source baseline is the tree of GitHub `main` at `70c5971d1d03139e3817cea1473fae7a7af9f8f2`.

**Decision: EXPERIMENT COMPLETE; NEGATIVE REACQUISITION EVIDENCE. No model/algorithm deployment.** The evaluation is **retrospective and exploratory** on the previously studied Batch56 36 natural DEV videos, not a new independent natural TEST. No held-out TEST answer was read or scored.

## Research motivation

Batches 62–68 showed that a visual/geometry consensus veto lowers occlusion false accepts but also sacrifices visible tracking. Batch 66 audited false accepts *during* occlusion. Batches 71–73 used artificial scenes to demonstrate that matching a single visually identical impostor does not establish physical identity.

This batch provides a missing natural-video diagnostic: **does the tracker correctly re-acquire a target AFTER it becomes visible again?** It is possible to achieve a low occlusion false-accept rate simply by continuing to abstain forever; an event-level post-occlusion metric is therefore necessary alongside false accept, coverage and CPU cost.

## Exact protocol and definitions

The scorer uses the SHA-256-pinned DEV prediction, DEV selection and DEV scoring files from Batch56b. The known DEV oracle is streamed only through the 36 DEV scene objects with the existing `batch62_dev_consensus_veto.load_dev_only_oracle` boundary check; no TEST objects are parsed. This utility never changes inference predictions or gates.

For each point query, start strictly *after* its query frame. Identify contiguous occlusion episodes from the DEV oracle. Following each occlusion, the post-occlusion visible window ends at the next occlusion or the end of the video. For a candidate method, define an event as **reacquired** if at least one predicted, accepted point during that visible window is within 8 image pixels of the DEV truth. The first-correct lag is measured relative to the first visible frame after occlusion.

Events ending with the video still occluded are **right-censored**. They contribute to occlusion false-accept counts but are excluded from reacquisition denominators. The scorer also tracks initial accepted-but-wrong visible predictions, post-visible coverage and PCK8, and the longest wrong-visible run. Here, "wrong visible position" does NOT establish a physical identity switch; this is a spatial-accuracy metric only.

No new inference-time thresholds were fitted. Four pre-existing choices were evaluated: `baseline`, `template_4px`, `affine_3px`, `both_4px_3px`.

## Verified historical DEV results

Observed on 36 natural DEV scenes, 144 original point queries: **214 occlusion episodes**, of which **64 have no observable subsequent visible frame** and **150 have a post-occlusion visible window** (total 4,753 post-visible frames).

| Variant | Correctly reacquired episodes / 150 | Post-visible correct PCK8 / 4,753 | Post-visible accepted frames / 4,753 | Wrong visible predictions / accepted post-visible | False accepts during occlusion |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | **5 (3.33%)** | **214 (4.50%)** | 287 (6.04%) | 73 / 287 (25.44%) | 317 / 10,797 |
| Template veto | **2 (1.33%)** | **36 (0.76%)** | 75 (1.58%) | 39 / 75 (52.00%) | 162 / 10,797 |
| Affine veto | **4 (2.67%)** | **136 (2.86%)** | 209 (4.40%) | 73 / 209 (34.93%) | 294 / 10,797 |
| Combined veto | **1 (0.67%)** | **34 (0.72%)** | 73 (1.54%) | 39 / 73 (53.42%) | 159 / 10,797 |

The baseline successfully reacquired targets in only **5 of 150** evaluable episodes. Template gating decreased occlusion false-accepted frames from 317 to 162 but also reduced correct reacquisition events from 5 to 2. The combined gate reduced occlusion false accepts further to 159, with **only 1 of 150** events reacquired.

First accepted post-occlusion points were visibly incorrect in 4/150 baseline events, 2/150 template, 4/150 affine and 2/150 combined. All variants retained a longest **24-frame run of accepted, spatially wrong points after an occlusion**.

Scene-concentrated findings: baseline recoveries occurred only in `kinetics_10` (two events), `kinetics_16`, `kinetics_18` and `kinetics_23`. Template retained only `kinetics_18` and `kinetics_23`. The studied `kinetics_17` scene still produced visibly wrong post-occlusion accepts and **zero** recovered events across all variants.

The "wrong fraction of accepted post-visible predictions" can *rise* when a conservative veto removes correct predictions even while reducing total wrong predictions. This ratio is conditional on acceptance and cannot be interpreted as a higher absolute rate of wrong visible frames or as proof of identity switching.

## Validation and reproducibility

- New implementation: `experiments/batch75_dev_reacquisition_audit.py`.
- New regression suite: `tests/test_batch75_dev_reacquisition_audit.py`; **40 passed**.
- Initial complete test run: 45 failures from an isolated development clone missing the existing shared `data -> ../data` symlink. No source defects were associated with those failures; logs preserved.
- Restored *only* this new worktree's symlink; second complete run: **1417 passed, 1 skipped, 3 existing Pillow deprecation warnings, exit code 0**.
- Final immutable DEV report and SHA pin: `outputs/research/batch75_dev_reacquisition_safety.json`, `...json.sha256` (kept on Tang, not committed because it contains DEV-derived scene statistics). SHA-256: `182b31ba48e35084028fc440258adeb7879ecc5cb904226f7764d004ac4ae191`.
- The event scorer replays the existing, frozen per-scene occlusion frame counts for every method. Historical 317 / 162 / 294 / 159 occlusion false accepts are reproduced exactly, making accidental changes to the interpretation of the previously pinned baseline detectable.

### Execution commands

```bash
cd /home/you/projects/Multimodal-VLM-Video-Generation-Lab/repo-batch75-dev
../.venv/bin/python -m pytest -q tests/test_batch75_dev_reacquisition_audit.py
../.venv/bin/python -m experiments.batch75_dev_reacquisition_audit \
  --output outputs/research/batch75_dev_reacquisition_safety.json
../.venv/bin/python -m pytest -q
```

The immutable output command deliberately refuses to overwrite an existing report or pin. To reproduce it, use a **new** output path rather than removing or mutating the existing record.

## Deployment decision and limitations

**Do not deploy these gates.** Lowering occlusion false acceptance alone can create an operationally useless perpetual-abstention tracker. Correct post-occlusion reacquisition and first accepted-point accuracy should be registered prospectively as additional safety/effectiveness endpoints on a *new, untouched natural video split*, including confidence intervals across independent videos and scene-group worst cases.

This audit uses an already-studied DEV population. It cannot establish a new blind improvement; 144 point queries and their frames are nested within just 36 scenes. Event-level results cannot be used for TEST tuning, and even PCK8-correct reappearance is not proof of physical identity continuity.

The frozen Batch59 natural-video evaluation remains the independent generalization gate. Its data download is still active, its code SHA and cron were not modified, and no competing GPU work or second natural TEST scorer was launched. After the independent Batch59 run finishes, compare coverage, post-occlusion recovery, occlusion false accepts, longest error runs, and VLM baselines before a deployment decision.
