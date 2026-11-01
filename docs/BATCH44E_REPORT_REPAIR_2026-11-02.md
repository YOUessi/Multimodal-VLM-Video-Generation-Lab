# Batch44E — Batch44D report artifact repair (2026-11-02)

## Why a new frozen repair was required

Batch44D completed all prediction and scoring stages. Its DEV and TEST CPU result files, DEV-only method selection, VLM gate, TEST VLM skip artifact, and prediction JSONs were individually SHA-256 pinned. TEST CPU scoring ran exactly once.

The report builder then returned exit code 0 but wrote `BATCH44B_TAPVIDMV_MULTIVIEW_REPORT.md`, a stale filename retained from the copied Batch44B script. The frozen Batch44D plan required `BATCH44D_TAPVIDMV_MULTIVIEW_REPORT.md`, so its final report-pin stage correctly remained pending. The failed clone/transport and report-name mismatch logs were preserved on Tang; Batch44D outputs were not edited or rescored.

## Repair

Batch44E changes the report builder to use an explicit report-repair config, writes the Batch44D filename, and validates every existing Batch44D result/prediction JSON against its adjacent SHA-256 pin. It also checks the Batch44D source SHA, DEV/TEST split identity, DEV-only selection marker, data/config hashes, sealed prediction hashes, and the VLM gate contract.

The repair plan is report-only:
- it reads the public manifest/calibration and already sealed aggregate JSON outputs;
- it does not open private TEST oracle files;
- it does not rerun RGB prediction, VLM inference, or TEST scoring;
- it does not use the GPU;
- it writes only the corrected Markdown report and its SHA-256 pin.

The source repair is committed to GitHub main and frozen in a new `repo-batch44e-tested` worktree. Batch44D's original frozen directory and failed R21 state remain preserved and paused.

## Research findings from the sealed Batch44D evaluation

The selected DEV method was `xview_klt_forward_12px`. It achieved DEV visible PCK@8 52.48%, coverage 55.36%, occlusion false-accept 11.67%, and CPU P50/P95 0.1601/0.1889 ms per target frame.

On the 10-scene TEST split (90 query instances), the same frozen method achieved visible PCK@8 43.80%, coverage 50.68%, accepted-point precision@8 86.43%, occlusion false-accept 31.44% (61/194), out-of-frame false-accept 0.48% (2/416), and CPU P50/P95 0.1684/0.1950 ms per target frame. The TEST occlusion false-accept rate exceeded the predeclared 25% DEV constraint; no threshold was changed afterward.

KLT-FB, although ineligible under the DEV gate, scored TEST PCK@8 54.67%, coverage 58.95%, occlusion false-accept 23.71%, and CPU P50/P95 0.1881/0.3180 ms/frame. This is a clear DEV-to-TEST selection instability; it is reported as a diagnostic, not used to replace the frozen selected method.

For cross-view 3-D reconstruction, the selected method triangulated 8/20 all-three-visible TEST groups (40% coverage), with mean world error 3.34 cm, P50 2.39 cm, P90 6.27 cm, 87.5% within 5 cm, and 100% within 10 cm. KLT-FB triangulated 5/20 groups (25%) with all 5 within 5 cm; KLT Forward triangulated 10/20 but had mean error 5.93 cm and P90 11.13 cm.

The camera-motion diagnostic found mean screen-vs-world-only displacement cosine 0.1696, while oracle camera-subtracted residual vs world-only cosine was 0.8911. The camera-only screen-displacement share proxy averaged 0.8395 (median 0.9867). ORB affine compensation reached only 0.3109 mean residual/world cosine, with 2.85 px median and 17.19 px P90 static-point reprojection error, and cost a mean 540 ms per camera-video registration. This supports the conclusion that screen-space affine compensation is not a substitute for calibrated camera poses in these sequences.

SmolVLM2-500M predicted VISIBLE for all 99 DEV queries. It correctly classified 85 visible points but falsely accepted all 7 non-visible points (2 occluded, 5 out-of-frame); balanced accuracy was 50%, non-visible false-accept rate 100%, and the DEV admission gate blocked. TEST VLM inference and scoring were skipped by design.

## Scope limitation

The 21 TAPVid-MV EgoExo4D/PACE sequences were unused in Batch43 and split by collection group (11 DEV / 10 TEST), but they are the same source groups used in Batch44C's DEV-only repair audit. Batch44C did not score TEST. This is a within-source-family holdout and measurement-repair result, not a new dataset-family generalization claim. The next research batch must use a genuinely new video family or separately sourced independent recordings, with subset-stratified DEV constraints and no post-hoc TEST tuning.


## Corrected report pin

The corrected local report is `outputs/tapvidmv_batch44d/BATCH44D_TAPVIDMV_MULTIVIEW_REPORT.md` (12,725 bytes), SHA-256 `6bf6f9e7012d8f01a984a01623f3b47c26bce4cabe5490bd4ead4ff12de95200`. The prior 12,477-byte report and its pin were moved to a clearly named `pre-caveat` archive; neither was overwritten or deleted.