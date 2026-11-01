# Batch44D — TAPVid-MV collection-group holdout with calibrated cross-view consensus

- Frozen source commit: 1cd9c017bd28e3d9f4d0342f720c4b4b4e6668de
- Public source index SHA-256: eb9ff78058eca50479af98eca5bba21716dcec183ce83d8e70349d9cea58f2f5
- Frozen manifest SHA-256: 71b391d19ea2d8717ff570981f53838f1a80232e564ea521763256b2a84d824a
- Scenes: 21 total (11 DEV / 10 TEST), sequence/collection-group disjoint
- Query instances: 189 (3 shared 3-D point identities × 3 camera views per scene)
- Source data: public TAPVid-MV Rerun viewer recordings for EgoExo4D and PACE; source recordings and extracted frames remain local

## Protocol and label semantics

- Each selected 3-D point is visible in all three camera views at the deterministic query frame: the earliest exact depth snapshot with at least three shared visible point IDs and at least 30 future frames. The same point identity is queried independently in each view.
- CPU predictors see all intermediate RGB frames. The VLM baseline sees only the selected query frame with a red-cross marker and the exact target-frame image.
- VISIBLE, OCCLUDED, OUT, UNKNOWN, and MISSING are kept separate. Visibility is inferred only at exact released depth-cloud snapshots; no interpolation of visibility labels is performed.
- PCK@8 all in-frame measures 2-D point localization for all in-frame projections, including frames whose visibility is unknown. PCK@8 depth-visible is restricted to depth-visible snapshots.
- TAP-Vid Jaccard uses visible/occluded/out labels at exact depth snapshots; unknown and missing labels are excluded.
- The joint tracker reads a separately hashed public calibration file containing only camera poses, intrinsics and resolutions; 3-D point tracks, track IDs and depth clouds remain private oracle data.

## DEV-only method selection

- Selection status: PASS
- Selected method: xview_klt_forward_12px
- Eligible candidates: xview_klt_forward_12px, xview_klt_forward_8px, xview_klt_forward_4px, xview_klt_fb_12px, xview_klt_fb_8px, xview_klt_fb_4px
- Rule: maximize scene-macro Jaccard mean among candidates satisfying predeclared coverage and occlusion/out-of-frame false-accept limits; otherwise use frozen fallback

| Method | PCK@8 all in-frame | PCK@8 depth-visible | Visible coverage | Precision@8 accepted | Occluded false accept | Out-of-frame false accept | Jaccard scene-macro | CPU P50/P95 ms/frame |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| affine_stabilized_klt | 46.5% | 48.7% | 51.7% | 94.2% | 43.3% | 0.7% | 32.5% | 0.3636 / 0.5519 |
| appearance_fb_055 | 39.4% | 43.5% | 45.4% | 95.7% | 6.7% | 2.5% | 25.7% | 0.1686 / 0.3350 |
| appearance_fb_070 | 34.2% | 38.2% | 39.3% | 97.2% | 5.0% | 1.0% | 23.2% | 0.1709 / 0.3319 |
| appearance_fb_085 | 24.2% | 26.1% | 26.5% | 98.4% | 3.3% | 0.3% | 16.4% | 0.1700 / 0.3326 |
| klt_fb | 52.5% | 55.1% | 60.2% | 91.6% | 48.3% | 3.0% | 32.7% | 0.1622 / 0.3201 |
| klt_forward | 58.4% | 61.7% | 81.6% | 75.7% | 89.2% | 13.7% | 32.1% | 0.1464 / 0.1570 |
| stationary_point | 15.0% | 9.5% | 100.0% | 9.5% | 100.0% | 100.0% | 3.1% | 0.0001 / 0.0002 |
| template_recover4 | 33.1% | 36.5% | 46.8% | 77.9% | 2.5% | 2.3% | 22.1% | 0.0243 / 0.0994 |
| xview_klt_fb_12px | 39.2% | 42.5% | 44.2% | 96.3% | 8.3% | 0.6% | 26.8% | 0.1766 / 0.3501 |
| xview_klt_fb_4px | 37.8% | 41.1% | 42.0% | 97.8% | 7.5% | 0.6% | 26.3% | 0.1765 / 0.3505 |
| xview_klt_fb_8px | 39.1% | 42.5% | 44.1% | 96.5% | 7.5% | 0.6% | 26.8% | 0.1767 / 0.3498 |
| xview_klt_forward_12px | 48.3% | 52.5% | 55.4% | 94.8% | 11.7% | 0.7% | 31.3% | 0.1601 / 0.1889 |
| xview_klt_forward_4px | 43.6% | 47.6% | 48.5% | 98.0% | 9.2% | 0.7% | 29.3% | 0.1605 / 0.1890 |
| xview_klt_forward_8px | 47.7% | 51.7% | 53.9% | 96.0% | 10.8% | 0.7% | 31.0% | 0.1603 / 0.1890 |

## TEST CPU results

TEST is descriptive only. All candidate methods and thresholds were frozen before TEST scoring; no TEST result was used for selection.

| Method | PCK@8 all in-frame | PCK@8 depth-visible | Visible coverage | Precision@8 accepted | Occluded false accept | Out-of-frame false accept | Jaccard scene-macro | CPU P50/P95 ms/frame |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| affine_stabilized_klt | 48.1% | 49.7% | 51.7% | 96.1% | 26.3% | 0.2% | 32.7% | 0.3599 / 0.5547 |
| appearance_fb_055 | 36.4% | 38.1% | 40.7% | 93.6% | 0.5% | 0.2% | 23.0% | 0.1968 / 0.3368 |
| appearance_fb_070 | 30.3% | 32.3% | 34.6% | 93.2% | 0.5% | 0.2% | 19.8% | 0.1930 / 0.3350 |
| appearance_fb_085 | 19.3% | 20.0% | 21.7% | 91.9% | 0.5% | 0.2% | 12.7% | 0.1942 / 0.3342 |
| klt_fb | 52.5% | 54.7% | 58.9% | 92.7% | 23.7% | 0.2% | 32.3% | 0.1881 / 0.3180 |
| klt_forward | 57.5% | 59.5% | 76.0% | 78.4% | 86.1% | 1.2% | 31.2% | 0.1513 / 0.1613 |
| stationary_point | 11.1% | 10.3% | 100.0% | 10.3% | 100.0% | 100.0% | 4.2% | 0.0001 / 0.0003 |
| template_recover4 | 26.4% | 27.1% | 39.8% | 68.1% | 18.6% | 4.1% | 15.0% | 0.0151 / 0.0998 |
| xview_klt_fb_12px | 34.1% | 35.9% | 38.5% | 93.2% | 3.1% | 0.2% | 21.4% | 0.2028 / 0.3470 |
| xview_klt_fb_4px | 32.6% | 33.9% | 35.4% | 95.7% | 1.0% | 0.2% | 20.5% | 0.2032 / 0.3474 |
| xview_klt_fb_8px | 33.8% | 35.6% | 38.0% | 93.6% | 3.1% | 0.2% | 21.3% | 0.2029 / 0.3471 |
| xview_klt_forward_12px | 43.0% | 43.8% | 50.7% | 86.4% | 31.4% | 0.5% | 24.4% | 0.1684 / 0.1950 |
| xview_klt_forward_4px | 36.3% | 37.6% | 40.1% | 93.8% | 9.8% | 0.5% | 22.3% | 0.1686 / 0.1948 |
| xview_klt_forward_8px | 41.9% | 43.0% | 48.3% | 89.0% | 23.2% | 0.5% | 24.1% | 0.1687 / 0.1947 |

## TEST subset diagnostics

| Source subset | Method | Visible PCK@8 scene-macro | Visible coverage scene-macro | Occluded false accept scene-macro | Jaccard scene-macro |
|---|---|---:|---:|---:|---:|
| egoexo4d | xview_klt_forward_12px | 25.6% | 46.5% | 47.3% | 5.4% |
| pace | xview_klt_forward_12px | 48.1% | 55.5% | 10.7% | 29.1% |

## TEST image-quality and camera-motion strata

| Scene | Camera | Mean luma P10/P90 | Contrast P10/P90 | Laplacian variance P10/P90 | ORB keypoints P10/P90 |
|---|---:|---:|---:|---:|---:|
| scene_11 | 0 | 71.6 / 74.8 | 24.8 / 28.2 | 97.1 / 134.5 | 438.6 / 969.8 |
| scene_11 | 1 | 103.3 / 103.7 | 47.4 / 47.5 | 758.6 / 779.7 | 943.0 / 951.0 |
| scene_11 | 2 | 104.1 / 104.7 | 32.4 / 32.8 | 617.5 / 630.3 | 932.4 / 939.0 |
| scene_12 | 0 | 69.6 / 76.0 | 20.0 / 24.3 | 70.3 / 252.7 | 398.2 / 772.2 |
| scene_12 | 1 | 104.8 / 107.8 | 38.3 / 41.0 | 2177.4 / 2538.3 | 838.5 / 861.7 |
| scene_12 | 2 | 106.9 / 108.4 | 36.8 / 37.5 | 1566.6 / 1609.7 | 802.1 / 821.0 |
| scene_13 | 0 | 88.5 / 98.0 | 58.8 / 63.7 | 994.7 / 1820.6 | 630.0 / 734.0 |
| scene_13 | 1 | 91.5 / 99.3 | 64.0 / 68.5 | 708.6 / 1040.0 | 685.0 / 820.0 |
| scene_13 | 2 | 86.8 / 102.3 | 61.1 / 67.7 | 529.2 / 934.6 | 751.0 / 858.0 |
| scene_14 | 0 | 86.4 / 97.8 | 54.1 / 65.3 | 853.4 / 2981.5 | 666.5 / 835.2 |
| scene_14 | 1 | 89.5 / 98.9 | 59.3 / 65.4 | 516.1 / 1770.5 | 780.5 / 916.7 |
| scene_14 | 2 | 91.5 / 105.4 | 63.6 / 67.7 | 364.5 / 898.3 | 526.1 / 808.9 |
| scene_15 | 0 | 87.7 / 96.2 | 59.7 / 65.1 | 851.3 / 3033.5 | 757.6 / 848.2 |
| scene_15 | 1 | 84.7 / 95.6 | 62.0 / 67.3 | 728.6 / 2020.3 | 885.8 / 928.0 |
| scene_15 | 2 | 90.1 / 104.5 | 64.5 / 69.9 | 354.3 / 645.1 | 789.2 / 872.6 |
| scene_16 | 0 | 91.6 / 97.9 | 54.5 / 63.4 | 915.2 / 2315.0 | 728.7 / 812.6 |
| scene_16 | 1 | 89.7 / 94.9 | 59.1 / 67.0 | 557.7 / 1283.5 | 837.3 / 891.2 |
| scene_16 | 2 | 90.8 / 102.3 | 61.3 / 65.4 | 477.0 / 907.3 | 713.4 / 804.8 |
| scene_17 | 0 | 81.4 / 98.9 | 56.2 / 67.5 | 366.0 / 1918.6 | 566.4 / 724.0 |
| scene_17 | 1 | 86.9 / 98.7 | 62.8 / 68.1 | 309.4 / 875.9 | 741.9 / 867.4 |
| scene_17 | 2 | 91.2 / 105.7 | 59.3 / 67.0 | 259.0 / 775.7 | 485.4 / 716.6 |
| scene_18 | 0 | 91.2 / 100.7 | 58.0 / 63.7 | 525.6 / 2296.7 | 647.1 / 813.7 |
| scene_18 | 1 | 90.9 / 98.6 | 63.3 / 67.7 | 355.6 / 998.0 | 767.2 / 912.5 |
| scene_18 | 2 | 94.0 / 103.5 | 60.7 / 65.4 | 394.9 / 1261.6 | 704.4 / 839.0 |
| scene_19 | 0 | 85.5 / 92.6 | 65.4 / 75.8 | 808.9 / 2468.3 | 691.0 / 811.0 |
| scene_19 | 1 | 86.0 / 92.6 | 70.7 / 79.0 | 514.2 / 1235.9 | 746.0 / 857.0 |
| scene_19 | 2 | 85.0 / 97.1 | 65.2 / 71.6 | 547.7 / 1261.3 | 756.0 / 852.0 |
| scene_20 | 0 | 91.0 / 98.2 | 64.9 / 70.5 | 671.9 / 1666.9 | 530.3 / 795.9 |
| scene_20 | 1 | 89.0 / 98.4 | 65.8 / 70.8 | 348.1 / 1076.4 | 666.0 / 819.2 |
| scene_20 | 2 | 90.4 / 99.3 | 64.2 / 68.0 | 448.6 / 818.6 | 510.9 / 864.5 |

## Cross-view 3-D reconstruction

| Method | 3-view GT-visible groups | Triangulated | Coverage | World error P50 / P90 (m) | Within 5 cm | Within 10 cm | Oracle pixel-triangulation error P50 (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| affine_stabilized_klt | 20 | 4 | 20.0% | 0.019 / 0.024 | 100.0% | 100.0% | 0.000 |
| appearance_fb_055 | 20 | 3 | 15.0% | 0.024 / 0.024 | 100.0% | 100.0% | 0.000 |
| appearance_fb_070 | 20 | 2 | 10.0% | 0.020 / 0.023 | 100.0% | 100.0% | 0.000 |
| appearance_fb_085 | 20 | 1 | 5.0% | 0.016 / 0.016 | 100.0% | 100.0% | 0.000 |
| klt_fb | 20 | 5 | 25.0% | 0.024 / 0.024 | 100.0% | 100.0% | 0.000 |
| klt_forward | 20 | 10 | 50.0% | 0.024 / 0.111 | 70.0% | 90.0% | 0.000 |
| stationary_point | 20 | 20 | 100.0% | 0.134 / 3.383 | 5.0% | 40.0% | 0.000 |
| template_recover4 | 20 | 2 | 10.0% | 0.049 / 0.074 | 50.0% | 100.0% | 0.000 |
| xview_klt_fb_12px | 20 | 5 | 25.0% | 0.024 / 0.024 | 100.0% | 100.0% | 0.000 |
| xview_klt_fb_4px | 20 | 4 | 20.0% | 0.020 / 0.024 | 100.0% | 100.0% | 0.000 |
| xview_klt_fb_8px | 20 | 5 | 25.0% | 0.024 / 0.024 | 100.0% | 100.0% | 0.000 |
| xview_klt_forward_12px | 20 | 8 | 40.0% | 0.024 / 0.063 | 87.5% | 100.0% | 0.000 |
| xview_klt_forward_4px | 20 | 4 | 20.0% | 0.020 / 0.024 | 100.0% | 100.0% | 0.000 |
| xview_klt_forward_8px | 20 | 7 | 35.0% | 0.024 / 0.034 | 100.0% | 100.0% | 0.000 |

## Camera motion versus world-relative motion

- Screen displacement vs world-only projection cosine: mean 0.1696 (n=12,980)
- ORB affine-compensated residual vs world-only projection cosine: mean 0.3109 (n=12,104)
- Oracle camera-subtracted residual vs world-only projection cosine: mean 0.8911 (n=12,754)
- Camera-only screen displacement share proxy: mean 0.8395
- Net camera translation: median 0.128 m; net rotation: median 17.997°
- ORB affine static-point reprojection error: P50 2.853 px; registration coverage 95.4%

The camera-only share is a 2-D counterfactual displacement proxy, not a physical causal fraction. Affine compensation is a screen-space diagnostic, not calibrated camera pose recovery.

## SmolVLM2-500M visibility baseline

- DEV gate: BLOCK; checks: {"balanced_accuracy": false, "class_support": false, "coverage": true, "false_accept_nonvisible_rate": false, "invalid_rate": true}
- DEV balanced accuracy / coverage / invalid rate: 50.0% / 100.0% / 0.0%
- DEV false-accept rate on non-visible points: 100.0%
- TEST VLM inference and scoring: SKIPPED because the predeclared DEV gate did not pass; TEST VLM labels were not opened.

## Dataset and evaluation limitations

- This is a 21-sequence exploratory subset, not a broad benchmark-wide generalization claim.
- These 21 sequences were unused in Batch43 but were used for DEV-only pipeline repair in Batch44C; Batch44C did not score TEST. This is a held-out TEST within the same source groups, not a new dataset-family generalization experiment.
- The Rerun viewer recordings display a subsampled point-track set; metadata-declared track counts are not the number of rendered track identities.
- Per-view visibility is inferred from exact released depth point-cloud snapshots rather than a separate official binary occlusion annotation. Frames without snapshots remain UNKNOWN.
- The 3-D track IDs and depth snapshots remain evaluation-only. The joint method uses public camera poses/intrinsics but no 3-D target coordinates or depth labels.
- Three selected shared point identities per scene remain a small cross-view sample. Cross-view consensus coverage and 3-D reconstruction statistics have wide uncertainty and should not be interpreted as deployment accuracy.
- Source video rights remain governed by the individual EgoExo4D/PACE terms. No source RRD, video, frame array, private oracle, or model weights are published in GitHub.

## Reproducibility

- Source index SHA-256: eb9ff78058eca50479af98eca5bba21716dcec183ce83d8e70349d9cea58f2f5
- Manifest SHA-256: 71b391d19ea2d8717ff570981f53838f1a80232e564ea521763256b2a84d824a
- DEV prediction SHA-256: b5c7a96ff791430494d1264e7fc673cd02bb5dad3d3cd36c18cbf81d0e712ab5
- TEST prediction SHA-256: 6ea5441986f8b5db3df0af4daf1ef53f99305603de17e89d159ebb15dd406102
- DEV-only selection and VLM admission-gate artifacts are hash-pinned in the local ignored outputs directory.
- All TEST scoring was run once after prediction hashes and source/config pins were verified.