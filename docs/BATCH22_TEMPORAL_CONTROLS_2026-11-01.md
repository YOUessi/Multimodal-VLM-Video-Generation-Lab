# Batch 22 — matched temporal input and order controls (2026-11-01)

All **five approved E1–E5 stages completed with returncode 0** on Tang; **78/78 tests passed** and source-security guard passed. Frozen source local SHA `681387096b0791205311cb9c30a22e2b14bb982b`; pinned GitHub source `2fc5f32fbde4b7b158d97efdcb2271be592cc4de`.

## Protocol

Use frozen SmolVLM2-2.2B pretrained model on all 400 published MVBench questions: 200 CLEVRER synthetic `moving_direction`, 200 scene-transition camera videos. Both 3-frame and 5-frame uniform reference model outputs were already evaluated in batch19.

- **Repeated center**: duplicate the same 50%-time image 3 or 5 times. The input image count is matched, the original source-video question labels remain valid, and this removes visible temporal progression.
- **Reversed order**: show the same 3 or 5 source frames in reverse chronological order. This can change the true answer, so the original labels are **invalid**; report **prediction-change rate only**, NOT reversed-video accuracy.
- No additional training, test-driven prompt tuning, filter changes, or posthoc frame-selection hyperparameter selection. 1,600 new real GPU predictions; all original raw strings and parsed choices remain on Tang.

## Actual measured results

| Task | Frames | Normal chronological | Repeat identical center | Delta pp; paired bootstrap 95% CI | Reversed clip: changes in selected option |
|---|---:|---:|---:|---|---:|
| Moving Direction (CLEVRER simulated) | 3 | 72/200 | 75/200 | +1.5 [-0.0,+3.5] | 6/200 |
| Moving Direction (CLEVRER simulated) | 5 | 73/200 | 74/200 | +0.5 [-1.5,+3.0] | 5/200 |
| Scene Transition (natural videos) | 3 | 182/200 | 171/200 | -5.5 [-10.5,-1.0] | 7/200 |
| Scene Transition (natural videos) | 5 | 180/200 | 170/200 | -5.0 [-9.5,-0.5] | 15/200 |

**Interpretation:** For the current CLEVRER direction QA, **static center-only evidence performs similarly to chronological video**; the model may rely on static cues/question priors rather than tracked motion, but this experiment cannot isolate the architecture component responsible. In natural scene transitions, varied scene evidence does help: repeating the midpoint loses ~5 percentage points, with paired 95% intervals excluding zero. Prediction choices change only rarely on chronological reversal, but lack of change **does not itself prove** the model understands or ignores temporal ordering: changing video order may or may not alter the correct option, depending on question semantics.

The motion task is simulated, not real camera footage. All material is public and may overlap with some original foundation-model pretraining distribution. Confidence intervals resample these evaluation examples; they do not quantify seed or training-data uncertainty.

## Outputs and reproducibility

- Frozen script `experiments/temporal_order_controls.py`
- Frozen config `configs/temporal_controls_v1.json`
- Source tests `tests/test_temporal_order_controls.py`
- Tang raw results `repo-batch22-tested/outputs/temporal_controls_v1/controlled_predictions.json` (1,600 per-example predictions), `temporal_controls_report.json`, `TEMPORAL_CONTROLS_REPORT.md`
- No model weights or original public videos committed to GitHub.

## Next approved exploratory test

A label-blind 12-probe motion-based frame selector under exactly three/five visual-frame budgets, matched against the earlier uniform baseline on 500M and 2.2B; frozen rules and CPU selection overhead to be reported. **The same public tasks were previously inspected**, so a new result from this further study is an exploratory ablation, **not** an unseen final evaluation. No claim of significance without paired CIs.
