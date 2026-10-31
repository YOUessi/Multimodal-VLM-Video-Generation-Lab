# External real-video cross-task generalization — 2026-10-31

**Batch13 (B61–B65) completed 5/5 on Tang.** This tests the STAR content-SHA-group-disjoint LoRA model on a different MVBench task, **scene_transition**. All 200 public scene-transition clips were used for evaluation only, not training or model selection.

## Source integrity

- Dataset: PKU-Alignment/MVBench `scene_transition`, official question/answer choices and public MP4 clips.
- Official source revision: `e17b0cae0729ebf38322ad0fa692c7ca72495afb`.
- Scene QA JSON SHA256: `edab8a7992851061cac170f7e50782961ec21826f99e70ee72ae73a9dafb79f9`.
- Scene archive SHA256: `1215b6f2e69820089320baa1e3f6fc8b73be4b4e3ee3948fe4b15a34cec22690`.
- Frozen 200-row selection SHA256: `2fb05cb0415ff55da67213550428ccae039b929fcb35ee8fd497632b1c8ce9f7`.
- All 200 scene videos have distinct SHA256 contents and **zero exact-file overlaps** with the 193 distinct STAR video contents.
- The STAR SFT used 120 training, 40 validation and 40 grouped heldout videos; best seed 19 was chosen before this external dataset was evaluated.

## Actual GPU result — all 200 videos

| Sample frames | Base model | STAR grouped SFT LoRA | Delta pp | Paired bootstrap 95% CI pp |
|---:|---:|---:|---:|---|
| 3 | 155/200 (77.5%) | 155/200 (77.5%) | +0.0 | [-6.5, 7.0] |
| 5 | 157/200 (78.5%) | 160/200 (80.0%) | +1.5 | [-4.5, 8.0] |

**Interpretation:** No reliable cross-task gain. The confidence intervals include zero. Two LoRA prediction outputs were unparseable at five frames and counted as incorrect. This negative/neutral result is retained without posthoc prompt tuning.

This external-task check is more informative than the simple synthetic color benchmark, but still subject to any possible training contamination in SmolVLM2's foundation pretraining (the public benchmark itself may be familiar to a pretrained model). Content hashes exclude exact source clip overlap, not perceptual near-duplicates.

## Frozen execution and scripts

- Tang frozen local batch13 SHA: `c13287f59f2699ba7f53f9637e9e67cf6990588c`.
- Frozen GitHub upstream ref: `70880bdc32fea691d72fb9f46cc37bb924363efe`.
- 44 source tests passed; all five batch13 stages returned 0.
- The evaluation code is `experiments/mvbench_scene_external.py` and `configs/mvbench_scene_external_v1.json`.
- Local raw evidence: `repo-batch13-tested/outputs/mvbench_scene_external_v1/` with manifest, baseline predictions, grouped-LoRA predictions, paired report and CI.
- Original videos and adapters remain exclusively on Tang.

## Current research conclusion

The training infrastructure and evaluation protocols execute correctly, but a 500M-parameter model with 120-step STAR action-sequence LoRA **has not shown reliable independent improvement** on grouped STAR heldout or cross-task scene transitions. The next algorithmic work should target temporal representation, answer bias and evidence utilization rather than simply longer training.
