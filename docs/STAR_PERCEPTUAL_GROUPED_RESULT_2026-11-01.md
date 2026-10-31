# STAR content + perceptual-candidate grouped LoRA — batch18 results

Verified on Tang at 2026-11-01 04:21 CST. **All six approved B83–B88 stages completed with returncode 0**. Frozen local source SHA: `4069ec42612dfcd74566da152dbf3e9dfdb24935`; GitHub pinned upstream: `918ccb18e91d84fe7f6781bac3f71e94c9527964`. All **59 regression tests passed**.

## Retrospective data-integrity remedy

The 200 MVBench STAR action_sequence video records contain 193 distinct full MP4 SHA256 byte hashes. Two additional strict pHash+dHash **candidate** near-duplicate relationships (source IDs 31/83 and 123/173) were joined into the grouping before assigning 120 train / 40 validation / 40 test rows. This yields **191 connected components**. All exact-video-byte groups and both flagged perceptual candidate edges were placed within a single split (zero cross-split audited edges).

Grouping is retrospective. The public task and some previous test results had already been inspected during development, so this is **not a new, never-touched blind evaluation**. Strict perceptual hashing does not detect all semantically similar videos; candidates are not independently human-confirmed as exact duplicates.

## Fresh training and selection

Two newly initialized SmolVLM2-500M PEFT LoRA adapters were trained for 120 optimization steps at seeds 19 and 37, using only the train split. Three-frame validation results: seed 19 **21/40**, seed 37 **20/40**. Selected seed 19 before evaluating the newly frozen retrospective test. Previously trained adapters were not reused.

## Results on 40 retrospective STAR action-sequence test videos

| Sampled frames | Base | Newly trained LoRA | Delta pp | Paired bootstrap 95% CI pp |
|---:|---:|---:|---:|---|
| 1 | 15/40 (37.5%) | 19/40 (47.5%) | +10.0 | [-2.5, 25.0] |
| 3 | 19/40 (47.5%) | 25/40 (62.5%) | +15.0 | [-5.0, 35.0] |
| 5 | 21/40 (52.5%) | 22/40 (55.0%) | +2.5 | [-15.0, 20.0] |

**All 95% confidence intervals contain zero**. The +15 pp in three-frame action QA is a promising exploratory observation but **does not prove a stable improvement or generalization**. The test set is small; this study is retrospective and public benchmark pretraining contamination is possible.

## External task result, different model snapshot

The earlier **exact-SHA grouped** STAR LoRA was evaluated on a distinct 200-video MVBench scene_transition task not used for training. Results were 155/200 base versus 155/200 LoRA at 3 frames; 157/200 versus 160/200 at 5 frames (+1.5 pp; 95% CI [-4.5,8.0]). This was **not** repeated using the newer perceptual-candidate-grouped adapter; do not combine these model variants as though they were one experiment.

## Other project achievements and open tasks

- On 40 grouped STAR clips, visual candidate reuse achieved **2.00x cold and 14.95x warm** mean CPU speedups for combined 3-frame/5-frame selection, with exactly identical RGB model input frames. These are CPU selection timings, **not** overall VLM speedups.
- Six locally generated ZeroScope v2 576w videos (two prompts, 8/14/20 Diffusion steps) were verified; peak allocated VRAM about 3.33 GiB, generation ~11–25 seconds per 12-frame clip. Wan2.1 has **not** been executed.
- Human video-generation quality scores are uncollected, not inferred from pixel statistics. No video-generation LoRA fine-tuning or proven quality uplift.
- GitHub source and tests are public; original videos, model weights and LoRA adapters remain on Tang.
- Local state for batch18: `repo-batch18-tested/.automation/state.json` reports `ALL_APPROVED_STAGES_DONE`. The 10-minute cron stays configured but with no further approved stages it **checks only** until a new frozen batch is installed.
- Next rigorous milestone: independently frozen *new* real-video test tasks, strong random/control baselines, paired power analysis and evaluation of video representation; do not retune these already-inspected heldouts.

Local raw artifacts: `repo-batch18-tested/outputs/star_perceptual_grouped_sft_v1/` including `manifest.json`, two training logs and adapters, `chosen_seed.json`, per-item `test_base.json` / `test_lora.json`, `test_comparison.json`, and `RETROSPECTIVE_AUDIT_RESULT.md`.
