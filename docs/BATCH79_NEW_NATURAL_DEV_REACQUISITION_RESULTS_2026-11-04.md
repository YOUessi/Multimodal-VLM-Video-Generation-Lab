# Batch 79 — New natural-video DEV reacquisition and occlusion safety

**Status:** Completed and pinned **exploratory DEV-only audit**; **no new independent TEST results**, no deployment authorization. Experiment code, tests and pre-evaluation protocol were published first to GitHub `main` as `8a818d330a2adc54c9ec08df3fbde1a6d6b32b76`. The corresponding frozen Tang local commit is `668b1d2bfb131444e3722638a44abae787e7fce7`, with identical source Git tree `594f468b9017ec97eb0dceaacf24b6ca2e0fb901`.

## Dataset and provenance

The experiment uses the **new Batch 59 / Batch 77 pinned natural Kinetics DEV set**, not the older Batch 56 DEV or CLEVRER. It contains 36 independent video scenes, 144 total initial queries (four per scene). The pre-existing prediction and scoring stages completed before this audit, with pinned frozen runner commit `7734940ec29bdf34deb784f4a5c562bdcf7803ce`.

This retrospective scorer reads only 36 DEV scene objects from the private oracle, using a deliberately **streaming** parser that stops before the six sealed TEST scene keys. CPU prediction, selection, and original scene-level DEV scoring were already frozen. We do not inspect or re-score TEST, and do not modify the frozen model, gate thresholds, or Batch 77 plan.

Immutable local DEV output:

`outputs/research/batch79_fresh_dev_reacquisition.json`

SHA-256: `2968cd10466b9d75bcddab8435da71b560b0d167b5b63a09787fa74d44f1d4b4`

## Results: previously frozen CPU trackers

For each query, the analysis counts a contiguous occlusion and its following visible period. There were 177 occlusion events; **66 were right-censored** because there was no subsequently visible frame, leaving **111 evaluable reacquisition events**. The events contain 9,769 occluded frames, and the evaluable follow-up windows contain 3,706 visible frames. Each method uses the same denominator.

| Frozen tracker | Evaluable events reacquired (PCK8) | Occluded false accepts | Occluded false accept rate | Visible follow-up PCK8 correct | Visible follow-up wrong accepts | Longest wrong visible run |
|---|---:|---:|---:|---:|---:|---:|
| **`klt_fb` (original frozen DEV selection)** | **7/111 (6.31%)** | **407/9769 (4.17%)** | 4.17% | 179/3706 | 235 | **133 frames** |
| `appearance_fb_055` | 3/111 (2.70%) | 89/9769 (0.91%) | 0.91% | 42/3706 | 0 | 0 |
| `appearance_fb_085` | 1/111 (0.90%) | 5/9769 (0.05%) | 0.05% | 16/3706 | 0 | 0 |
| `gradient_fb_050` | 5/111 (4.50%) | 33/9769 (0.34%) | 0.34% | 53/3706 | 2 | 2 |
| `template_recover4` | 6/111 (5.41%) | 1066/9769 (10.91%) | 10.91% | 58/3706 | 131 | 32 |

**Key negative result:** the preselected `klt_fb` has the highest recovery count among these five methods, but still recovers only 7 of 111 events and yields **235 incorrect PCK8 visible follow-up positions**, including a **133-frame accepted incorrect-position run**. This is a *position* error, not proven persistent physical-ID switching.

In scene-level safety analysis, `klt_fb` has 17 scenes with any false acceptance among the 33 scenes with occlusion support. The worst supported scene has **95.45%** occluded-frame false accepts. The conservative `appearance_fb_085` has only 5 false accepts overall, but succeeds on just one recoverable event.

Paired 2,000-resample scene-level bootstrap against `appearance_fb_055` gives `klt_fb` minus control **recovery-rate difference** interval **[0.0, 0.07767]** and **occluded false-accept difference** interval **[0.01128, 0.06053]**. These are exploratory descriptive intervals that depend on scene exchangeability, not prospective TEST confidence intervals. The scene-level comparison was not used to change the original frozen selection.

The previously frozen Batch77 group safety screen already returned **`BLOCK_GROUP_RISK_OR_MISSING_SUPPORT`**, so these findings corroborate a deployment block rather than changing the selection.

## Correctness and reproducibility

- 34/34 new Batch79-specific tests passed.
- Full repository suite: **1,523 passed, 1 skipped, 3 pre-existing Pillow deprecation warnings**, exit code 0.
- Immutable DEV result was created once and matched its SHA-256 pin.
- Every method's occlusion count and false-accepted occlusion count was cross-checked **per scene** against independently produced frozen DEV scores; no mismatches occurred.
- The code, tests, and protocol were already on GitHub before DEV oracle scoring.
- The first command incorrectly used file execution (`python experiments/batch79_fresh_dev_reacquisition.py`), which failed with `ModuleNotFoundError: No module named 'experiments'` **before opening any oracle or creating output**. The successful, unchanged frozen-code invocation uses module mode:

```bash
python -m experiments.batch79_fresh_dev_reacquisition \
  --input-dir ../repo-batch77-tested/outputs/kinetics_batch59 \
  --data-dir ../data/tapvid_kinetics_batch59/batch59_frozen \
  --output outputs/research/batch79_fresh_dev_reacquisition.json
```

The failure and success command outputs were preserved locally in `/tmp/mvvg_batch79_eval.log` and `/tmp/mvvg_batch79_eval_module.log`. Neither affected the already-frozen experiment algorithm.

## Separate Batch 77 execution recovery

After the 42 video archives downloaded and DEV CPU stages finished, Batch 77 paused before its VLM stage with a **missing required model path**. The two SmolVLM2-2.2B model shards (4.96 GB and 4.03 GB) and safetensors index already existed in the project's shared `models/` directory; the frozen repo had no `models` link.

A local, **gitignored** `repo-batch77-tested/models -> ../models` symlink was created after checking GPU compute was idle, and file existence was verified. This adds no new weights, no tracked source change, and no GPU competitor. A single guarded frozen runner tick was started; subsequent state reported `RUNNING:R08_DEV_SMOLVLM2_2_2B_REFERENCE_CROP_PREDICTIONS`. At report time, there is **no verified VLM result yet**. Do not mistake a running R08 for a completed VLM benchmark.

**Subsequent safety correction (2026-11-04 03:28 UTC+8):** The Tang host rebooted at 03:19, and at 03:20 an explicit ignored `.automation/PAUSE` marker was created with `.automation/crontab_before_user_pause.txt`; the user crontab was deliberately cleared. The manual one-stage R08 tick above was started after that marker, before this pause was discovered. It remains restricted to `--max-stages 1`. **Do not recreate cron or remove PAUSE without an explicit new authorization.** This later evidence supersedes any earlier statement that the Tang 10-minute cron remains active.

## Decision

**Do not deploy or retune on this DEV.** The independent video DEV material strengthens the finding that acceptance/coverage alone is insufficient; the choice between `klt_fb` and conservative appearance gates carries severe, and highly clustered, safety/recovery tradeoffs. Wait for the original Batch77 frozen VLM DEV gate, CPU/VLM TEST prediction pinning, one-time TEST scoring and final audit before any independent generalization claim. Any future algorithm must be separately developed on DEV and tested on a **different** untouched TEST.
