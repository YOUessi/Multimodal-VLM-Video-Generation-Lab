# Batch 38B — Independent TAP-Vid RGB-Stacking, true point occlusion and 500M endpoint VLM (SEALED results, 2026-11-02)

**Frozen source SHA:** `b1e3f4e7c74214adca05c2718056e6cdac9a0fe2`. On authorized Tang, all **13/13** stages passed exactly once in isolated `repo-batch38b-tested`; **206/206** regression tests passed twice, with three non-blocking pre-existing Pillow warnings. TEST results were examined **only after** CPU and 500M VLM predictions had been independently SHA256 sealed. The present document is a later documentation-only commit: it is not a new algorithm release and does not alter the frozen source or thresholds.

## Provenance, safety and failure chain

- Official TAP-Vid RGB-Stacking `https://storage.googleapis.com/dm-tapnet/tapvid_rgb_stacking.zip`, 187,291,581 bytes, **MD5** `46ec560a2387a819a4856a4ac5b2e27d`, **SHA256** `ba0cec640deb65ab510bc3920cdaf8b7a2e756fce9fb339199c557ade436b0cc`, verified ZIP CRC. Official data package states CC BY; source attribution remains mandatory. These are **synthetic robotics videos**, not independently recorded real camera footage.
- Original 50 video records; hash-only selection (without checking metrics) freezes 4 DEV and 8 different TEST source videos, 125 sampled frames each, four first-visible physically tracked point queries per video: **1,500 RGB frames and 48 query-point tracks**. No repeat of DAVIS source videos. Every point has explicit occlusion labels. On each video, the CPU algorithms see all future RGB frames, only the initial public (x,y,t), and never the private future coordinates/occlusions during prediction.
- Important pre-scoring fix: the official TAP-Vid points are in **normalized xy**, whereas PCK@8px and KLT operate in pixel coordinates. Conversion multiplies by `(width,height)` exactly as in the upstream TAP-Vid reader. Test added. This was **not inferred from scored TEST**.
- First Batch38 frozen attempt (SHA `5aa45c10d8dc6c797431d462cb706f7cc43aacba`) failed at data-preparation A02 because the code attempted to hash its oracle file before writing it. All 12 extracted video arrays and failure logs were preserved. **No DEV or TEST inference or scoring was done in the failed batch.** New independent frozen source Batch38B fixes file-hash preparation order, adds a full prepare/audit regression, retains same source/split/methods and never overwrites earlier failed logs.
- Private source/oracle/query bytes, predictions, images, checkpoint and run logs remain on Tang. Publication scan: **175 tracked source files; zero forbidden media/weights/secrets**. GPU lock `/tmp/tang_shared_gpu_experiments.lock` and independent SHA/source guards were used. GitHub contains source and this postscore report only.

## Primary *sealed TEST* — 8 scenes × 4 queries × 124 target frames

The 3,968 query-frame positions consist of **3,266 GT-visible** point positions and **702 explicitly GT-occluded** positions; the point identity is the same physical sampled point throughout. Unconditional PCK@8px counts an abstention on visible ground truth as wrong; “accepted precision” is conditional on non-abstained visible predictions. Absent false accepts are **not** included in unconditional visible PCK and must be considered separately.

| Frozen CPU method | Visible PCK@8 (correct / 3,266) | Accepted on visible | Correct / wrong accepts on visible | Accepted precision@8 | GT-occluded false accepts / 702 | CPU median per target frame* |
|---|---:|---:|---:|---:|---:|---:|
| Stationary first point | **2,483 / 3,266 = 76.03%** | 3,266 / 3,266 = 100% | 2,483 / 783 | 76.03% | **702 / 702 = 100%** | ~0 ms |
| KLT forward | 1,115 / 3,266 = 34.14% | 1,756 / 3,266 = 53.77% | 1,115 / 641 | 63.50% | 257 / 702 = 36.61% | 0.051 ms |
| KLT forward–backward | 1,041 / 3,266 = 31.87% | 1,195 / 3,266 = 36.59% | 1,041 / 154 | **87.11%** | **45 / 702 = 6.41%** | 0.032 ms |
| Static template + up to four misses | 985 / 3,266 = 30.16% | 1,245 / 3,266 = 38.12% | 985 / 260 | 79.12% | 75 / 702 = 10.68% | 0.035 ms |

*CPU timings are method-only, split-aggregated medians of per-scene per-query amortized times; they **exclude decoding, preprocessing and other system overheads**. Early tracker failure artificially reduces cost. Do not conclude a method is faster at equivalent usefulness. On these 8 source-scene clusters, descriptive bootstrap 95% intervals for unconditional PCK@8 are approximately stationary **[64.66%, 85.55%]** and FB-KLT **[20.01%, 43.91%]**; small n=8 cannot support a population/generalization claim.

### Why the zero-motion baseline wins (a serious negative result)

A postscore descriptive audit (not used to modify any method) found median source-point displacement from the query frame of **0.144 px** across GT-visible TEST positions, with **2,483 / 3,266 = 76.03%** of visible targets within 8 pixels of their original location. This **exactly explains** the stationary baseline's high PCK@8: the test is dominated by still/near-still point positions. It is **not** evidence of robust moving-target tracking. The 90th-percentile displacement was about **53.69 px**, meaning there is also a substantial motion tail.

For **783 visible positions with displacement >8px** (a postscore diagnostic stratum, not a new model-selection criterion), stationary gets **0/783**, KLT forward **175/783 (22.35%)**, FB-KLT **160/783 (20.43%)**, and template **45/783 (5.75%)**. For the 591 positions with displacement >16px, KLT forward reaches **147/591 (24.87%)** and FB-KLT **136/591 (23.01%)**. None is a strong motion tracker on this sample.

The 702 explicitly absent TEST positions form **30 occlusion runs**, median length **18.5 sampled frames**, maximum 110. Of **21 first-visible frames after an occlusion**, correct@8 is stationary **8/21**, forward KLT **1/21**, FB-KLT **1/21**, and template **2/21**. Thus the existing “track-or-permanently-abstain” strategies do **not** meaningfully recover exact point identity after disappearance. This is an exploratory failure slice, not independent validation or post-TEST tuning.

### DEV context (not TEST-dependent selection)

DEV has 1,613 visible and 371 truly occluded positions across four other clips. Stationary gets 1,504/1,613 (93.24%) visible PCK8 but falsely accepts all 371 occlusions. KLT forward gets 429/1,613 (26.60%) with 37/371 (9.97%) occlusion false accepts. FB-KLT gets 400/1,613 (24.80%) with 17/371 (4.58%). Template gets 646/1,613 (40.05%) with 79/371 (21.29%). Significant scene-composition differences reinforce the need for scene-level inference; DEV was not used to adjust any method or grammar in this batch.

## SmolVLM2-500M endpoint probe: negative **interface-conformance** result

The frozen checkpoint is SmolVLM2-500M-Video-Instruct, SHA256 `b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3`. Same output space (pixel x,y **or** `ABSENT`) and fixed target identity as the CPU baseline, but **NOT the same evidence budget**: VLM receives just the marked query RGB frame and the final RGB frame, not all 125 frames. Eight DEV and sixteen TEST endpoint queries were selected before inference and SHA-sealed before gold scoring; the TEST subset has **10 visible / 6 occluded** endpoints.

**24/24 VLM responses violated the predeclared exact grammar** (8 DEV, 16 TEST). Common raw replies were `120 120`, `123 123`, `100 100` and `123`, whereas the parser requires `x,y` or `ABSENT`. Consequently, **0/10 visible TEST endpoints correct and 0/6 absent TEST endpoints falsely accepted**, but both counts are a consequence of **universal invalid-output abstention**, NOT demonstrated occlusion skill. The raw model may be producing rough coordinates in a different syntax: an exploratory future *DEV-only* parser/prompt contract test would be warranted, followed by a **new unscored holdout**, never retrospective reparsing the already opened TEST to claim an improved result. Measured TEST greedy generation averaged ~0.308 seconds per query (not full pipeline/model-load latency), peak allocated model GPU memory ~1.88 GiB.

Frozen TEST CPU endpoints on those identical 16 queries: stationary 8/10 visible correct but 6/6 occlusions falsely accepted, KLT forward 2/10 correct and 2/6 absent false accept, FB 2/10 correct and 0/6 false accept, template 1/10 correct and 1/6 false accept. These are very small exploratory endpoint counts, not fair model rankings.

## Reproducibility and strict interpretation

- Frozen output SHA256s: CPU DEV `71d2fb9217ba940ff26732cb938a2f07c14d3308ce7453c2975fab66e84e0427`, CPU TEST `47d34e23b728750b1d49da2516dfa9180c1e0f1229a34cb3d922cc07e45314e2`; VLM DEV `f89441423fb1f6ef23e6367caf5d074c2a027ae87c71b3637bc35290842bab22`, VLM TEST `a56f31807d0496081ffc213783826441292abac821bd1b35ee1a37bbb8be251f`.
- All 13 stage records in `.automation/state.json` are `done` with `returncode=0`; `.automation/ALL_APPROVED_STAGES_DONE` exactly marks frozen SHA `b1e3f4e...`. Cron `MVVG_PROJECT4_10MIN` is guarded against re-running completed stages. Original Batch38 failure is kept separately. Postscore diagnostic JSON and all private gold remain ignored/local.
- These are correlated frames within **8 TEST clips** and **not 3,968 independent subjects**. The dataset may be in pretrained model corpora; no model-unseen or natural-camera generalization claim. PCK@8 in raster pixels and genuine TAP-Vid point occlusion are distinct from previous DAVIS object-mask centroid PCK, CLEVRER motion QA, 3D world motion and true camera pose.

**Research bottleneck and next independent experiment:** ensure a motion-balanced, predeclared *new* DEV/TEST selection with point motion, long occlusion and reappearance, plus explicit risk–coverage analysis and point identity preservation. Fix 500M's grammar/interface only against DEV; evaluate on entirely different unscored video identities. For real-camera occlusion generalization, obtain legal natural footage with explicit point-level visibility or annotated identities; synthetic RGB-Stacking is not a substitute. Do not pick parameter thresholds from this scored Batch38B TEST.
