# Batch 35 — fail-closed cross-snapshot provenance and post-hoc gate-risk audit (2026-11-01)

## What was broken

Batch 34's stage S9 failed after its CV and VLM inference and first heldout aggregate report had already completed: **134 tests passed, one test failed**. The failed test attempted to replace the evaluator's project root with an empty temporary directory and required the scorer to raise rather than use somebody else's frozen outputs. It did not raise because the shared `domain_shift_motion_eval.read` and `manifest` functions implicitly read their module-level `ROOT`, ignoring the explicit study-scoped root. This is a serious **cross-snapshot score provenance contract violation**, not a cosmetic testing issue. Batch 34 retains its original immutable SHA, failure log, 280 source clips, and scored outputs; it was **not repaired in place**.

## New implementation

1. Extend the shared JSON reader and manifest verifier with optional explicit `root`, retaining backward compatibility for preexisting projects. The parallax evaluator now scopes all manifest, media-byte SHA, CV, and VLM report reads to its **own** project root. If a required frozen source/model file is absent, scoring fails closed instead of reading the current working directory or another checkout.
2. The scorer rejects incorrect CV header coverage or answer-label-blind markers and a VLM artifact whose pinned `model_name`, `revision` or source SHA do not match the frozen config. Regression tests reproduce the original missing-manifest failure, missing-model fallback, unscoped read failure, duplicate predictions, invalid accepted direction, rescue/harm accounting and media exclusion from report.
3. New read-only `parallax_gate_risk_audit.py` accepts **only** the Batch 34 frozen SHA **2809c0a4eed8453a5f373716a104853f96ff0abb** and the five precomputed artifact-byte SHA-256 digests. Every one of the 280 synthetic MP4 files is reverified against the manifest. It attributes accepted gate overrides into **rescues**, **harms**, **wrong-both**, **correct-both**, broken down by method, nuisance slice, model and frame budget; bootstrap confidence intervals cluster by the 100 independent forward/reverse scene pairs. Its output must reproduce Batch 34's previously published sample-count and hybrid accuracy aggregates exactly. No threshold search, prompt changes, optimizer updates, model selection, GPU inference, or new TEST scoring occurs.

## Findings on the already-scored, synthetic 200-video / 100-scene-pair Batch 34 heldout

| Frozen model | Frames | Gate | VLM only | Gate+VLM | Accepted | Wrong gates | Rescue | Harm |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 500M | 3 | plain HSV | 42/200 | 101/200 | 121 | 34 | 69 | 10 |
| 500M | 3 | affine | 42/200 | 102/200 | 89 | 9 | 62 | 2 |
| 500M | 3 | shape+affine | 42/200 | 54/200 | 14 | 0 | 12 | 0 |
| 2.2B | 3 | plain HSV | 41/200 | 107/200 | 121 | 34 | 66 | 0 |
| 2.2B | 3 | affine | 41/200 | 97/200 | 89 | 9 | 57 | 1 |
| 2.2B | 3 | shape+affine | 41/200 | 53/200 | 14 | 0 | 12 | 0 |
| 2.2B | 5 | plain HSV | 43/200 | 105/200 | 121 | 34 | 63 | 1 |
| 2.2B | 5 | affine | 43/200 | 99/200 | 89 | 9 | 57 | 1 |
| 2.2B | 5 | shape+affine | 43/200 | 55/200 | 14 | 0 | 12 | 0 |

**Key negative result:** the strict shape+affine tool has **0 observed wrong accepted** decisions but only **14/200 accepted (7% coverage)**, reducing hybrid correct answers versus less conservative tools. Zero gate errors with this coverage is not a robust accuracy or model improvement. Plain HSV accepts **121/200 with 34 wrong** and the affine variant **89/200 with 9 wrong**, demonstrating a **coverage–risk trade-off**, not a universally better method.

**Do not conflate gate error with harm to this specific baseline.** In the 2.2B/3-frame case, plain HSV's **34 wrong** gate decisions happened to override 34 already-wrong VLM answers, hence 0 harms to VLM-correct predictions. The same frozen plain gate causes **10 harms** to the 500M baseline. A zero-harm figure for one weak VLM cannot certify a safe gate.

The post-hoc method produces per-mode risk and source-pair-cluster confidence intervals, but this set is now **already exposed**. No empirical parameter choices may be made from these TEST labels; prospective new hypotheses must be chosen using new DEV-only experiments and evaluated on genuinely new independent scenes/camera sources. All clips are procedurally generated; none of the statistics establish natural camera footage performance, 3D depth recovery, or real-world object identity tracking.

## Release and acceptance

After the new code is committed, freeze a fresh **Batch 35** checkout and run the ordered, no-GPU recovery stages: focused provenance tests → full 140+ suite → SHA-pinned read-only risk audit → artifact-vs-previous-result agreement → source security guard. Do **not** rearm Batch 34's failed S9 or rewrite its frozen outputs. Only source, tests, configs, protocols and **aggregate** results may be pushed; model checkpoints, video data, raw test outputs and secrets remain local. A new prospective natural-video study would additionally require video licensing/consent and independent motion/identity annotations.
