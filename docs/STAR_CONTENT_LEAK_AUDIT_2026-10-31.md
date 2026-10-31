# STAR action-sequence split leakage audit — 2026-10-31

**Previous round2 120/40/40 video-ID-disjoint evaluation is NOT content-disjoint and its heldout generalization claims must be regarded as invalid.** Preserve its logs as historical/debugging evidence only.

All 200 local optimized STAR action_sequence MP4s were re-read and SHA-256 hashed on Tang. Only **193 unique file byte contents** were found; **7 groups** each contain two identical files. **5 duplicate-content groups span the frozen training/validation/test boundaries.**

| SHA-identical original file IDs (0-based source index) | Original split pair | Leakage risk |
|---|---|---|
| 98 and 84 | train / test | Direct train/test duplication |
| 100 and 138 | train / train | Within train only |
| 105 and 171 | train / test | Direct train/test duplication |
| 159 and 161 | train / test | Direct train/test duplication |
| 64 and 195 | val / test | Validation/test duplication |
| 135 and 186 | val / test | Validation/test duplication |
| 144 and 192 | test / test | Within test only |

The earlier 40-sample STAR test scores (14/40 vs 14/40; 19/40 vs 20/40; 20/40 vs 20/40) and adaptive frame-selection results inherit this problem and **do not demonstrate independent generalization**. This applies equally to the subsequent cost/accuracy comparison. The measured CPU costs remain hardware observations, but any claims about STAR heldout accuracy must be qualified.

## Remediation
1. Re-split the complete 200-sample STAR action_sequence task by **full content SHA-256**, not by filename/annotation ID, with identical bytes assigned to exactly one split.
2. Preserve exactly 120 train / 40 validation / 40 test source rows using fixed-seed deterministic group allocation. Freeze manifest with source/video SHA, decoded frame count, split assignments.
3. Retrain *new* LoRA adapters from foundation weights; do not reuse the earlier potentially leaked round2 adapters.
4. Select model/checkpoint from validation only, evaluate the test once, and save all raw predictions and 95% paired bootstrap CIs.
5. Keep historical round2 results and logs, but publish grouped-split results as the authoritative STAR evaluation.
6. Audit other public benchmarks for duplicate content before asserting independent test disjointness.

No data were deleted. This audit was discovered during the shared temporal cache benchmark, when content-hash cache correctly hit on two distinct video filenames.
