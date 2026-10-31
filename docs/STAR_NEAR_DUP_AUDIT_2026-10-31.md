# STAR near-duplicate QA audit — 2026-10-31

The frozen batch16 run completed (55/55 unit tests passed).

Using three sampled frames per clip and strict perceptual hash thresholds (pHash Hamming <= 8 and dHash Hamming <= 6 on every sampled frame), 19,900 video pairs were compared across the existing content-hash grouped STAR action-sequence dataset. No answer labels or model predictions were used.

**Two cross-split near-duplicate candidates** were flagged, both between training and validation:

| Source index | Split | Similar source index | Split | pHash distances | dHash distances |
|---:|---|---:|---|---|---|
| 31 | train | 83 | val | [6,2,2] | [5,2,1] |
| 123 | train | 173 | val | [0,0,0] | [0,0,2] |

No candidates crossed into the heldout test under this strict audit. These are candidates for review, not confirmed duplicates, and the heuristic does not guarantee absence of other near duplicates.

The older source-ID-only STAR evaluation is invalid due to confirmed SHA-identical video overlap. The newer SHA-grouped evaluation eliminates exact-byte cross-split copies but may still have near-duplicate training/validation pairs. Its model-selection statistics require this qualification.

Local results remain on Tang in `outputs/perceptual_near_duplicate_v1/`. The original source split and experiment logs have not been changed. New experiments should freeze a group-connected split before training, without posthoc filtering by test accuracy.
