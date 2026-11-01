# Batch 38 — Independent TAP-Vid RGB-Stacking Point Visibility and Endpoint VLM (PRE-TEST; 2026-11-02)

## Why this batch, evidence boundaries

Batch37B scored 8 previously unscored real DAVIS sequences at fixed 16-frame evaluation: 120 visible object-centroid positions and **zero annotated absent positions**. Fixed FB KLT scored 43/120 PCK@8; consensus 40/120; guarded 28/120. The guarded method reached conditional precision 28/29 but abandoned 91 visible frames, and neither absence nor actual physical point identity could be measured. Batch37B TEST has been scored, and **none of its images or labels may tune these methods**.

Batch38 opens an independently sourced official **TAP-Vid RGB-Stacking** corpus with **explicit ground-truth occluded flags for individual tracked material points**, not mask area shrink. These are **synthetic robotic manipulation videos, NOT independent real-camera footage**; claims must remain domain-specific. Official source: https://storage.googleapis.com/dm-tapnet/tapvid_rgb_stacking.zip. Study parent: Doersch et al., TAP-Vid (2022). No new model-training data are used. Absolute image-plane point coordinates are not background-relative/object-world motion or calibrated camera pose.

The official archive's HEAD declares 187,291,581 bytes, ETag/MD5 `46ec560a2387a819a4856a4ac5b2e27d`. A bounded HTTP-206 90-part downloader must verify byte lengths, official MD5 and ZIP CRC; archive SHA256 is recorded. The downloaded original media live solely under ignored, local shared `data/tapvid_rgb_stacking_v1`. All source images, future point tracks, occlusion gold, model weights, generated query-mark images and private predictions remain local on Tang. No media/weights/secrets in commits.

## Predeclared deterministic frozen split and query policy

Original archive contains 50 source clips, each **250 RGB 256×256 frames with 30 physically tracked points** and 30 corresponding occlusion tracks. Deterministically hash-order video IDs using salt `BATCH38-INDEPENDENT-RGB-STACKING-20261102-SEALED-47`. Take first **4 DEV clips**, next **8 TEST clips**, independent source videos, without checking metrics or TEST labels. Preserve every second source frame, 125 frames/clip. Hash-order fixed point IDs within each clip; select up to 4 points. Query point is the earliest annotated visible sampled frame; its (x,y,t) is public to the predictor, subsequent positions/visibility remain **private scoring oracle**. No scoring before SHA-pinned prediction; no TEST query reselection or algorithm threshold tuning.

DEV may be used for diagnostics, but this batch's source, split, four CPU method contracts, and VLM grammar are frozen before **either** DEV or TEST prediction; all frozen candidates are evaluated without DEV selection and without TEST tuning.

## Fixed methods, common output space, deliberately different visual evidence

Four CPU methods return image-plane material point `(x,y)` or abstain: stationary point, KLT forward (LK 21×21, pyramid 3), forward-backward KLT (<=1.5 px FB error) and static initial 21×21 texture template correlation with local radius 32 px, NCC >=0.75 and a maximum of four missed frames. KLT failure permanently abstains; no unvalidated object re-identification. CPU methods see all sampled source frames from their public query onward.

An independent **task-output-matched but NOT input-budget-matched** SmolVLM2-500M zero-shot endpoint probe receives *only* the public first-annotated-visible query frame marked with a red cross and the last frame (not 125 intermediate frames). For **first two fixed hash-selected query IDs per scene**, greedily generate <=32 tokens, parse exactly `x,y` integers in bounds or `ABSENT`; malformed output is abstention. CPU endpoints for **those exact same queries at that exact final frame** provide a reference. Do not compare this to prior different-label-space SmolVLM2 direction QA; avoid claims of model ranking, fair input budgets, or intrinsic ability from this zero-shot probe. It is an exploratory, frozen input/interface baseline (not claim that 500M has been trained for point tracking).

## Fixed scores and reporting

Primary: visible-point **unconditional PCK@8px** on all future annotated visible point positions (abstentions wrong), unconditional coverage, precision among accepted, false acceptance on explicitly ground-truth-occluded positions, occlusion/visibility accuracy, and scene-cluster variation. Also CPU method time **per scene-query target frame**, explicitly excluding file decode and setup; early abstention makes methods cheaper, which is not necessarily better. VLM endpoint metrics must report point correctness/absence on its much smaller fixed endpoint query subset, invalid grammar rate and GPU seconds. Separate DEV and TEST in reports; **label-blind TEST predictions and VLM replies must be SHA256-sealed before scoring TEST once**. The same oracle visibility labels used for metrics must **never** enter algorithms or prompts. No new blind-dataset or pretrained model-unseen claim.

Failure: preserve exact original logs, local media, prediction digests and frozen source SHA; never retry a failed stage on the same scored plan after modifying its code. Correct any defect with new GitHub main SHA and new isolated frozen batch; never replace prior scored results. Safe source publication guard runs after scores and full regressions. A finished cron batch must not run empty forever.

## Sequence of immutable verification

Freeze source, configs/plan and tests on GitHub main, import **exact matching Git commit SHA** to isolated tested checkout. Verify official source MD5/CRC, freeze 12 video-frame bytes and first-frame public queries/private scoring oracle, pass >=200 regression tests, CPU DEV prediction+pin, DEV scoring, GPU DEV VLM+pin, DEV matched endpoint scoring, CPU TEST prediction+pin, GPU TEST VLM prediction+pin, one-time private TEST CPU score, one-time private matched endpoint TEST score, source/security guard, final tests. GPU stages require global Tang `/tmp/tang_shared_gpu_experiments.lock`, exclusive NVIDIA compute and free memory. Each stage is one-shot; failed work is retained.

This is a **prospective methods protocol only**, not a result statement. Until the frozen TEST stages genuinely finish, report no TEST effect sizes.
