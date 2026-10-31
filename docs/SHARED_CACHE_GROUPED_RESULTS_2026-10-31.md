# Shared scenelet feature cache on SHA-grouped STAR videos — 2026-10-31

**Batch14 B66–B69: 4/4 stages completed, return code 0; 46/46 offline regression tests passed.** Frozen Tang source SHA: `c207a50ef729743f9f15f9e054448c760b633660`. GitHub upstream freeze: `35ec17c5ab29fde10f6c98f06b1946cf6b829810`.

## What was changed
- Label-blind 48-candidate frame selection uses the **same** fixed 0.65 grayscale-motion + 0.35 hue-change scoring, 5%/95% endpoint guard and temporal spacing as earlier.
- Underlying STAR test is the new **full-video SHA-group-disjoint** 40-sample split; old leaked ID-only STAR scores were not used.
- Exact selected frame indices match the frozen sampling plan for both 3-frame and 5-frame requests.
- **Every selected PIL RGB image byte** matches the original uncached implementation.
- A per-source fresh temporary cache ensures true cold measurements, even if different filenames have identical MP4 bytes.
- A shared cache keyed by full video-file SHA256 enables repeated-video feature reuse; 3- and 5-frame requests share one feature extraction and unique frame decoding.
- Three warm-cache repetitions per real video; cold and uncached measurements evaluated on all 40 videos.

## Real measured CPU costs, both 3-frame and 5-frame requests per video

| Workflow | Mean CPU time per video | Median time | P95 |
|---|---:|---:|---:|
| Original two independent computations | **1.290 s** | 1.167 s | 2.183 s |
| Shared calculations, fresh cold feature cache | **0.646 s** | 0.583 s | 1.137 s |
| Repeated-video shared SHA warm cache | **0.086 s** | 0.074 s | 0.173 s |

Mean speedups: **2.00× cold, 14.95× warm** relative to the original pair of computations. These are *CPU-only* selection/decode measurements, not complete end-to-end VLM serving speedups; cache hits are representative only of repeat-video requests.

## Validity restrictions
- Full SHA duplicates are grouped within one dataset split; cache behavior with duplicated names was separately tested.
- No changes to model weights, prompts, labels, or selected pixel data; **no model accuracy improvement is claimed**.
- Cold processing of a new video still examines 48 candidates and has a nontrivial CPU cost.
- The file cache contains only local visual signals and metadata, not uploaded media.

## Evidence
- Local raw metrics: `repo-batch14-tested/outputs/shared_cache_grouped_v1/shared_cache.json`.
- Local Markdown report: `repo-batch14-tested/outputs/shared_cache_grouped_v1/SHARED_CACHE_REPORT.md`.
- Scripts: `experiments/shared_temporal_cache.py`, `experiments/adaptive_temporal_sampling.py`, `experiments/verify_adaptive_plan.py`.
- Frozen configs: `configs/shared_cache_grouped_v1.json`, `configs/adaptive_grouped_v1.json`.
- Regression suite covers byte-identical RGB frames, repeated-file SHA hits, isolated cold cache, and frozen frame endpoints.

**Conclusion:** This is a measurable engineering contribution for long-video frame selection and repeated-input caching. It should not be described as VLM reasoning improvement or video generation quality improvement.
