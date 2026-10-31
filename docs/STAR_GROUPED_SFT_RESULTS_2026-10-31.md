# STAR action-sequence SFT after exact-content leakage remediation (2026-10-31)

**Status: B55–B60 all completed with exit code 0 on Tang.** This document supersedes the old ID-only STAR split results and any old adaptive-frame accuracy measured on that invalid split. Old logs remain solely for audit.

## Data leakage found and corrected
- The 200 STAR `action_sequence` video filenames correspond to **193 SHA-256-distinct complete file contents**.
- Seven pairs have identical bytes; five pairs crossed train, validation or test in the original ID-only split.
- Original heldout STAR accuracies `14/40`, `19/40`, `20/40` etc. **cannot be used as independent generalization evidence**.
- A new 120/40/40 split assigns whole video-content SHA groups to exactly one split.
- Full-file SHA content hash overlap between new training, validation and test: **zero**.
- Every original video was decoded and frame-sampled in the new preflight.
- New frozen grouped manifest SHA256: `a747e2c682d0dbe1ad276b1ae558836208d14e190e853a1c4bbaa46b113c9a57`.

## Actual new model training

SmolVLM2-500M-Video-Instruct, LoRA rank 4, alpha 8, 120 optimizer steps per seed, source-video train split only. New adapters trained from base weights at seeds 19 and 37; neither reuses the potentially leaked round2 adapter. Validation comparison: seed 19 **23/40**, seed 37 **23/40**, both with 3-frame inputs. Precommitted tie-break selects smaller seed **19**, before reading new test answers.

## Authoritative grouped-heldout result

| Frame budget | Base SmolVLM2 | New content-disjoint STAR-trained LoRA | Difference pp | Paired bootstrap 95% CI pp |
|---:|---:|---:|---:|---|
| 1 frame | 17/40 | 15/40 | **-5.0** | [-17.5, 7.5] |
| 3 frames | 16/40 | 17/40 | **+2.5** | [-5.0, 10.0] |
| 5 frames | 18/40 | 20/40 | **+5.0** | [-5.0, 15.0] |

**Interpretation:** All intervals include zero; no stable independent improvement established. Only 40 test items and one public action-sequence task. Exact-byte SHA grouping does not eliminate visually near-identical videos or contamination in pretrained foundation-model corpora.

## Provenance and reproducibility
- Frozen local batch12 source SHA: `4df2a30ba7f12df833862702746c198cd9a7fb6f`
- Official GitHub batch12 archive ref: `434e3586799f9ba96adfdd57f5ff35edac0b5f4d`
- 41/41 frozen source tests passed.
- Local evidence: `repo-batch12-tested/outputs/star_grouped_sft_v1/AUTHORITATIVE_RESULTS.md`, `AUTHORITATIVE_RESULTS.json`, per-sample `test_base.json` / `test_lora.json`, frozen `manifest.json`, `chosen_seed.json`, two optimizer traces and adapters.
- Code: `experiments/star_content_grouped_sft.py`, `experiments/star_grouped_report.py`, `configs/star_grouped_sft_v1.json`.

## Next approved independent check
Batch13 tests a different official MVBench task, `scene_transition`, with **200 external video examples**. The batch13 preflight verifies source hashes and that scene video bytes do not match any STAR action-sequence video bytes; runs both unmodified base and the validation-selected grouped STAR LoRA without training on scene-transition labels. External-task results must be reported with paired CI and possible negative outcomes preserved.
