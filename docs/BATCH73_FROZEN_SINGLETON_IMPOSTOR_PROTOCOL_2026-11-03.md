# Batch 73 — Frozen prospective singleton impostor identity-limit experiment

Date: 2026-11-03. Sole source of truth: `YOUessi/Multimodal-VLM-Video-Generation-Lab`, `main`.

## Motivation and falsifiable question

Batch 72 showed that a *unique high-scoring visual patch* and a best-vs-second margin can reacquire an object following a 46-frame opaque interval in a small synthetic benchmark. It did **not** establish that this patch belonged to the original physical object. In the prospective Batch 73 intervention, the original identity either reappears or remains unavailable while a *single, pixel-identical copy* emerges. Every public observation and the query coordinate are exactly identical for both latent identity assignments. This explicitly tests the assumption **one visible high-confidence template match ⇒ verified physical identity**.

The data are new **procedural synthetic video** (not new natural video, not a 3D camera dataset). No method may learn or change thresholds on the new DEV or TEST. This cannot establish independent natural-domain efficacy. Batch 59's frozen natural-video run and its GPU ownership remain unchanged.

## Frozen generator, split and observability contract

- `experiments/batch73_singleton_impostor.py` generates **four regimes**: `exact_singleton`, `pan_gain_singleton`, `blur_singleton`, `retextured_singleton`.
- Each regime has **3 DEV pairs and 2 TEST pairs**: 20 independent pairs, 40 videos, 96 grayscale 256×256 frames each. Distinct seed ranges prevent DEV/TEST overlap.
- Each pair contains a `present` world and `absent` world. Input query and **all 96 frames have identical SHA-256 hashes within each pair**.
- [0,18): original object visible. [18,64): fully covered by an opaque screen. [64,96): exactly one visually plausible candidate appears. In `present`, it is the original target. In `absent`, it is an impostor and the original target remains unavailable.
- Source of post-occlusion physical identity is a **private latent assignment** only, not provided to algorithms. This is a thought-experiment-style non-identifiability construction, not a claim that photographs certify identity.
- `pan_gain_singleton` applies a 2D background shift plus global gain, **not real 3D camera motion**. Blur and retexturing perturb local visual evidence in other regimes.
- Public frames, query coordinates, regimes and seeds are written to an immutable manifest. The truth is stored solely under separate local `data/.../private`, never read in prediction.

## Frozen methods, prior knobs and measurement

Exactly four methods, with no changes or parameter sweeps:

1. `appearance_fb_055`: previously frozen conservative appearance tracker.
2. `template_recover4`: previously frozen template recovery baseline.
3. `best_template_score`: Batch 72 17×17 globally searched NCC patch; fixed threshold 0.86.
4. `identity_margin_veto`: same NCC and second-best independent candidate rejection with fixed margin 0.07 and ±17 pixel exclusion radius.

All four operate on pixels only; none receives `world`, post visibility or real coordinate labels. Use PCK@8 exclusively for genuine, visible target locations. Absent-world candidate acceptance is **false identity acceptance**, not correct reacquisition. Report positive-world recovery and coverage separately from post-absence false-accept, hidden-interval false-accept, longest runs and CPU calculation time. A valid method that always abstains gets zero post-target coverage, **not** false credit for safety.

Unit of independence: the **pair**, not two pixel-identical videos. For every method, in balanced indistinguishable pairs, `2 × correctly accepted present frames ≤ accepted post frames across both worlds`. No model observing only the pixels can distinguish the two latent worlds better than chance without an external identity signal. The bound assumes equal probability for the two constructed worlds, not real-world frequencies.

## Frozen test firewall and stage order

Before generating or scoring **any** new TEST video, commit this source, all tests and this protocol to GitHub `main`, record immutable commit SHA and verify working-tree cleanliness. Then:

```bash
../.venv/bin/python -m pytest tests/test_batch73_singleton_impostor.py -q
../.venv/bin/python -m pytest -q
../.venv/bin/python -m experiments.batch73_singleton_impostor prepare
../.venv/bin/python -m experiments.batch73_singleton_impostor predict-dev
../.venv/bin/python -m experiments.batch73_singleton_impostor score-dev
../.venv/bin/python -m experiments.batch73_singleton_impostor predict-test
../.venv/bin/python -m experiments.batch73_singleton_impostor score-test
../.venv/bin/python -m experiments.batch73_singleton_impostor audit
```

Preparation writes all generated images and private labels once and pins public SHA-256. TEST prediction is write-once and occurs **before** the single TEST scoring operation. Prediction opens no private label directory; both-world prediction traces must be identical. Source SHA, tracker config and artifact hashes are validated before actions. Any failure is retained and investigated before rerun with a new freeze; no overwriting artifacts or tuning on TEST.

## Decision and interpretation

This intervention should **falsify** the claim that the best-vs-runner-up visual margin reliably verifies object identity when only an impostor is visible. A result with correct visible recovery in the present world and equal false acceptance in the absent world is a **negative safety result**, even if PCK on the visible world is perfect. If a method abstains in both worlds, it avoids false identity but sacrifices recovery.

Do not deploy a method or infer real-world accident rates from this test. A deployment hypothesis would require externally verifiable identity cues, natural-video independent test transfer and prespecified false acceptance and coverage budgets. No VLM is used in this CPU-only batch; the Batch 59 GPU VLM baseline remains the priority prospective natural test.
