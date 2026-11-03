"""Batch73 prospective singleton-impostor non-identifiability stress test.

The public video has exactly ONE compelling reacquisition candidate. In world
'present' it is the original queried object; in world 'absent' it is a copy
while the original is still unavailable.  The public pixels and query are
bit-identical within each pair; only private identity truth differs.

A single strong template match, even with a perfect best-vs-runner-up margin,
cannot establish object continuity. All seeds, methods, scoring, and thresholds
are frozen before new TEST truth is opened.  Synthetic 2D pan != 3D camera.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time

import cv2
import numpy as np

from experiments import batch70_shared_identity_stress as b
from experiments import batch72_cue_observability as old

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT.parent / "data" / "batch73_singleton_impostor_v1"
OUT = PROJECT / "outputs" / "batch73_singleton_impostor_v1"
CONFIG = PROJECT / "configs" / "kinetics_batch56.json"

REGIMES = ("exact_singleton", "pan_gain_singleton",
           "blur_singleton", "retextured_singleton")
WORLDS = ("present", "absent")
DEV_PAIRS = 3
TEST_PAIRS = 2
FRAMES = 96
WIDTH = 256
OCC_START = 18
OCC_STOP = 64
PATCH = 17
PCK = 8.0
METHODS = ("appearance_fb_055", "template_recover4",
           "best_template_score", "identity_margin_veto")
PROTOCOL = "batch73-prospective-singleton-impostor-paired-v1"
KIND = "BATCH73_FROZEN_SINGLETON_IDENTITY_V1"


def specs() -> list[dict]:
    rows = []
    for split, n, seed0 in (("dev", DEV_PAIRS, 731_103),
                            ("test", TEST_PAIRS, 739_907)):
        for j, regime in enumerate(REGIMES):
            for k in range(n):
                pair_id = f"{split}_{j:02d}_{k:02d}"
                seed = seed0 + 131*j + 17*k
                for world in WORLDS:
                    rows.append(dict(id=f"{pair_id}_{world}",
                                     pair_id=pair_id, seed=seed,
                                     split=split, regime=regime,
                                     world=world, nframes=FRAMES,
                                     query_time=0))
    return rows


def geometry(regime: str, t: int) -> tuple[list[float], list[float]]:
    shift = [0.35*t, -0.12*t] if regime == "pan_gain_singleton" else [0., 0.]
    pre = [float(round(70 + .22*t + shift[0])),
           float(round(118 + shift[1]))]
    post = [float(round(176 - .14*(t-OCC_STOP) + shift[0])),
            float(round(116 + shift[1]))]
    return (pre if t < OCC_STOP else post), shift


def render(spec: dict) -> tuple[np.ndarray, dict]:
    """Pixel data never depends on which latent world is true."""
    if spec["regime"] not in REGIMES or spec["world"] not in WORLDS:
        raise ValueError("BATCH73_INVALID_SPEC")
    regime = spec["regime"]
    rng = np.random.default_rng(spec["seed"])
    bg = cv2.GaussianBlur(rng.integers(40, 202, (WIDTH, WIDTH),
                                        dtype=np.uint8), (5, 5), .9)
    patch = cv2.GaussianBlur(rng.integers(30, 224, (PATCH, PATCH),
                                           dtype=np.uint8), (3, 3), .5)
    retextured = patch.copy()
    # A controlled shift of local identity evidence; no oracle access.
    retextured[::5, ::5] = 255 - retextured[::5, ::5]
    frames = np.empty((FRAMES, WIDTH, WIDTH), dtype=np.uint8)
    truth_xy, visible, phase, background_xy = [], [], [], []
    for t in range(FRAMES):
        xy, shift = geometry(regime, t)
        img = cv2.warpAffine(bg, np.float32([[1, 0, shift[0]],
                                              [0, 1, shift[1]]]),
                             (WIDTH, WIDTH),
                             borderMode=cv2.BORDER_REFLECT_101)
        if OCC_START <= t < OCC_STOP:
            cx = int(round(121 + shift[0]))
            cy = int(round(118 + shift[1]))
            cv2.rectangle(img, (cx - 94, cy - 33), (cx + 94, cy + 33),
                          108, -1)
            cv2.rectangle(img, (cx - 94, cy - 33), (cx + 94, cy + 33),
                          85, 2)
            label = "opaque"
        else:
            x, y = map(int, xy)
            sprite = (retextured if regime == "retextured_singleton"
                      and t >= OCC_STOP else patch)
            img[y-8:y+9, x-8:x+9] = sprite
            label = "pre" if t < OCC_START else "post"
        if regime == "pan_gain_singleton":
            gain = .80 + .16*math.sin(t/11.)
            img = np.clip(img.astype(np.float32)*gain + 7, 0, 255).astype(np.uint8)
        if regime == "blur_singleton" and t >= OCC_STOP:
            img = cv2.GaussianBlur(img, (3, 3), .6)
        frames[t] = img
        target_visible = (t < OCC_START or
                          (t >= OCC_STOP and spec["world"] == "present"))
        visible.append(bool(target_visible))
        truth_xy.append(xy if target_visible else None)
        phase.append(label)
        background_xy.append(shift)
    return frames, dict(xy=truth_xy, visible=visible,
                        phase=phase, background_translation_xy=background_xy,
                        latent_world=spec["world"],
                        absent_world_post_candidate_is_impostor=True,
                        synthetic_not_natural_or_3d=True)


def assert_manifest(man: dict) -> None:
    if (man.get("kind") != KIND or man.get("protocol") != PROTOCOL or
            len(man.get("scenes", [])) != len(specs()) or
            not isinstance(man.get("source_sha"), str)):
        raise RuntimeError("BATCH73_MANIFEST_CONTRACT_DRIFT")
    for row, expected in zip(man["scenes"], specs()):
        if any(row.get(k) != v for k, v in expected.items()):
            raise RuntimeError("BATCH73_SPLIT_SEED_OR_WORLD_DRIFT")
        if (not isinstance(row.get("frames_sha256"), str) or
                len(row["frames_sha256"]) != 64 or
                not isinstance(row.get("query_xy"), list)):
            raise RuntimeError("BATCH73_PUBLIC_PIN_MISSING")


def guard(man: dict) -> None:
    b.assert_clean_source(man["source_sha"])
    if b.digest(CONFIG.read_bytes()) != man["config_sha256"]:
        raise RuntimeError("BATCH73_CONFIG_DRIFT")


def prepare(data: Path = DATA) -> dict:
    b.assert_clean_source(b.source_sha())
    if (data / "manifest.json").exists() or (data / "public").exists():
        raise RuntimeError("BATCH73_DATA_ALREADY_EXISTS")
    rows = []
    pairs = {}
    for spec in specs():
        pixels, truth = render(spec)
        p = data / "public" / f"{spec['id']}.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("xb") as f:
            np.save(f, pixels, allow_pickle=False)
        b.save_json(data / "private" / f"{spec['id']}.json", truth)
        row = dict(spec, frames_sha256=b.digest(p.read_bytes()),
                   query_xy=geometry(spec["regime"], 0)[0])
        rows.append(row)
        pairs.setdefault(spec["pair_id"], []).append(row["frames_sha256"])
    if not all(len(s) == 2 and s[0] == s[1] for s in pairs.values()):
        raise RuntimeError("BATCH73_COUNTERFACTUAL_PIXEL_DRIFT")
    man = dict(kind=KIND, protocol=PROTOCOL, source_sha=b.source_sha(),
               config_sha256=b.digest(CONFIG.read_bytes()), scenes=rows,
               independent_units="paired_latent_worlds",
               pixel_identical_pair_worlds=True,
               thresholds_frozen_from_batch72=True,
               synthetic_not_natural_or_3d=True)
    assert_manifest(man)
    print("BATCH73_PREPARED", b.save_json(data / "manifest.json", man),
          "pairs", len(pairs), flush=True)
    return man


def load_frames(data: Path, spec: dict) -> list[np.ndarray]:
    p = data / "public" / f"{spec['id']}.npy"
    if b.digest(p.read_bytes()) != spec["frames_sha256"]:
        raise RuntimeError("BATCH73_FRAMES_HASH_DRIFT")
    a = np.load(p, allow_pickle=False)
    if a.shape != (FRAMES, WIDTH, WIDTH) or a.dtype != np.uint8:
        raise RuntimeError("BATCH73_FRAMES_FORMAT_DRIFT")
    return [np.ascontiguousarray(f) for f in a]


def predict(split: str, data: Path = DATA, output: Path = OUT) -> dict:
    if split not in ("dev", "test"):
        raise ValueError("BATCH73_INVALID_SPLIT")
    man = b.read_pinned(data / "manifest.json")
    assert_manifest(man)
    guard(man)
    if split == "test":
        b.read_pinned(output / "dev_results.json")
    if (output / f"{split}_predictions.json").exists():
        raise RuntimeError("BATCH73_PREDICTION_ALREADY_EXISTS")
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    cfg = json.loads(CONFIG.read_text())["tracker"]
    rows = []
    for spec in man["scenes"]:
        if spec["split"] != split:
            continue
        frames = load_frames(data, spec)
        q = dict(query_time=0, query_xy=spec["query_xy"])
        start = time.perf_counter()
        affines, _, _ = b.core.estimate_affines(frames, cfg)
        traces, _ = b.core.predict_query(frames, q, affines, cfg)
        legacy_ms = 1000 * (time.perf_counter()-start)
        start = time.perf_counter()
        traces["best_template_score"] = old.identity_match(
            frames, q["query_xy"], False)
        best_ms = 1000 * (time.perf_counter()-start)
        start = time.perf_counter()
        traces["identity_margin_veto"] = old.identity_match(
            frames, q["query_xy"], True)
        margin_ms = 1000 * (time.perf_counter()-start)
        if any(k not in traces or len(traces[k]) != FRAMES for k in METHODS):
            raise RuntimeError("BATCH73_MISSING_METHOD")
        rows.append(dict(id=spec["id"], pair_id=spec["pair_id"],
                         regime=spec["regime"], world=spec["world"],
                         traces={m: traces[m] for m in METHODS},
                         cpu_ms=dict(legacy=round(legacy_ms, 3),
                                     best=round(best_ms, 3),
                                     margin=round(margin_ms, 3))))
    by_pair = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], []).append(row)
    if not all(len(pair) == 2 and pair[0]["traces"] == pair[1]["traces"]
               for pair in by_pair.values()):
        raise RuntimeError("BATCH73_PIXEL_IDENTICAL_TRACE_DRIFT")
    payload = dict(kind="BATCH73_LABEL_BLIND_PREDICTIONS",
                   split=split, source_sha=man["source_sha"],
                   manifest_sha256=b.digest((data/"manifest.json").read_bytes()),
                   oracle_read_in_prediction=False, predictions=rows)
    print("BATCH73_PREDICT_PINNED", split,
          b.save_json(output/f"{split}_predictions.json", payload),
          len(rows), flush=True)
    return payload


def score_trace(trace: list, truth: dict) -> dict:
    if (len(trace) != FRAMES or len(truth.get("visible", [])) != FRAMES or
            len(truth.get("xy", [])) != FRAMES or
            len(truth.get("phase", [])) != FRAMES):
        raise RuntimeError("BATCH73_TRACE_TRUTH_LENGTH_DRIFT")
    out = dict(pre_visible=0, pre_correct=0, hidden=0,
               hidden_accepted=0, post_present=0,
               post_present_accepted=0, post_present_correct=0,
               post_present_wrong=0, post_absent=0,
               post_absent_accepted=0, longest_hidden_error=0,
               longest_post_absent_error=0, first_correct_delay=None)
    hrun = arun = 0
    for t in range(1, FRAMES):
        p = trace[t]
        if p is not None and (not isinstance(p, (list, tuple)) or len(p) != 2 or
                              not all(math.isfinite(float(v)) for v in p)):
            raise RuntimeError("BATCH73_INVALID_TRACE")
        expected = ("pre" if t < OCC_START else "opaque" if t < OCC_STOP
                    else "post")
        if truth["phase"][t] != expected:
            raise RuntimeError("BATCH73_PHASE_DRIFT")
        if truth["visible"][t] != (truth["xy"][t] is not None):
            raise RuntimeError("BATCH73_VISIBILITY_ORACLE_DRIFT")
        if t < OCC_START:
            if not truth["visible"][t]:
                raise RuntimeError("BATCH73_PRE_VISIBILITY_DRIFT")
            out["pre_visible"] += 1
            out["pre_correct"] += (p is not None and
                                   math.dist(p, truth["xy"][t]) <= PCK)
        elif t < OCC_STOP:
            if truth["visible"][t]:
                raise RuntimeError("BATCH73_OPAQUE_VISIBILITY_DRIFT")
            out["hidden"] += 1
            out["hidden_accepted"] += p is not None
            hrun = hrun + 1 if p is not None else 0
            out["longest_hidden_error"] = max(out["longest_hidden_error"], hrun)
        elif truth["visible"][t]:
            out["post_present"] += 1
            out["post_present_accepted"] += p is not None
            correct = p is not None and math.dist(p, truth["xy"][t]) <= PCK
            out["post_present_correct"] += correct
            out["post_present_wrong"] += p is not None and not correct
            if correct and out["first_correct_delay"] is None:
                out["first_correct_delay"] = t - OCC_STOP
        else:
            out["post_absent"] += 1
            out["post_absent_accepted"] += p is not None
            arun = arun + 1 if p is not None else 0
            out["longest_post_absent_error"] = max(
                out["longest_post_absent_error"], arun)
    return out


def totals(rows: list[dict]) -> dict:
    result = {}
    for method in METHODS:
        metrics = [r["metrics"][method] for r in rows]
        keys = ("pre_visible", "pre_correct", "hidden", "hidden_accepted",
                "post_present", "post_present_accepted",
                "post_present_correct", "post_present_wrong",
                "post_absent", "post_absent_accepted")
        v = {k: sum(x[k] for x in metrics) for k in keys}
        v["longest_hidden_error"] = max(
            (x["longest_hidden_error"] for x in metrics), default=0)
        v["longest_post_absent_error"] = max(
            (x["longest_post_absent_error"] for x in metrics), default=0)
        v["post_correct_coverage"] = (v["post_present_correct"]/v["post_present"]
                                       if v["post_present"] else None)
        v["post_absent_false_accept_rate"] = (
            v["post_absent_accepted"]/v["post_absent"] if v["post_absent"] else None)
        v["hidden_false_accept_rate"] = (
            v["hidden_accepted"]/v["hidden"] if v["hidden"] else None)
        v["post_reacquired_worlds"] = sum(
            x["first_correct_delay"] is not None for x in metrics)
        result[method] = v
    return result


def score(split: str, data: Path = DATA, output: Path = OUT) -> dict:
    if split not in ("dev", "test"):
        raise ValueError("BATCH73_INVALID_SPLIT")
    man = b.read_pinned(data/"manifest.json")
    assert_manifest(man)
    guard(man)
    if split == "test":
        b.read_pinned(output/"dev_results.json")
    pred_path = output/f"{split}_predictions.json"
    prediction = b.read_pinned(pred_path)
    if (prediction.get("split") != split or
            prediction.get("source_sha") != man["source_sha"] or
            prediction.get("oracle_read_in_prediction") is not False or
            prediction.get("manifest_sha256") !=
            b.digest((data/"manifest.json").read_bytes())):
        raise RuntimeError("BATCH73_LABEL_FIREWALL_DRIFT")
    expected = [x for x in man["scenes"] if x["split"] == split]
    got = prediction["predictions"]
    if len(got) != len(expected):
        raise RuntimeError("BATCH73_PREDICTION_COUNT_DRIFT")
    if (output/f"{split}_results.json").exists():
        raise RuntimeError("BATCH73_RESULTS_ALREADY_EXIST")
    rows = []
    for spec, p in zip(expected, got):
        if (p["id"], p["pair_id"], p["world"]) != (
                spec["id"], spec["pair_id"], spec["world"]):
            raise RuntimeError("BATCH73_PREDICTION_ORDER_DRIFT")
        truth = b.read_pinned(data/"private"/f"{spec['id']}.json")
        rows.append(dict(id=spec["id"], pair_id=spec["pair_id"],
                         regime=spec["regime"], world=spec["world"],
                         metrics={m: score_trace(p["traces"][m], truth)
                                  for m in METHODS}, cpu_ms=p["cpu_ms"]))
    grouped = {}
    for p in got:
        grouped.setdefault(p["pair_id"], []).append(p)
    if not all(len(xs) == 2 and xs[0]["traces"] == xs[1]["traces"]
               for xs in grouped.values()):
        raise RuntimeError("BATCH73_PAIRED_PREDICTION_INCONSISTENCY")
    # This bound is pairwise, not an aggregated substitute for identity truth.
    for regime in REGIMES:
        block = [r for r in rows if r["regime"] == regime]
        agg = totals(block)
        for method, t in agg.items():
            accepted = t["post_present_accepted"] + t["post_absent_accepted"]
            if 2*t["post_present_correct"] > accepted:
                raise RuntimeError("BATCH73_IDENTIFIABILITY_BOUND_BROKEN")
    costs = {}
    for method in ("legacy", "best", "margin"):
        a = [r["cpu_ms"][method] for r in rows]
        costs[method] = dict(p50=statistics.median(a),
                             p95=float(np.percentile(a, 95)))
    answer = dict(kind="BATCH73_SINGLETON_IMPOSTOR_RESULTS",
                  split=split, source_sha=man["source_sha"],
                  prediction_sha256=b.digest(pred_path.read_bytes()),
                  independent_pairs=len(grouped),
                  counterfactual_worlds_are_correlated=True,
                  no_thresholds_fit_on_new_data=True,
                  dataset_is_synthetic=True,
                  totals=totals(rows),
                  regimes={r: totals([x for x in rows if x["regime"] == r])
                           for r in REGIMES},
                  cpu_ms=costs, scenes=rows)
    print("BATCH73_SCORE_PINNED", split,
          b.save_json(output/f"{split}_results.json", answer), flush=True)
    for m, v in answer["totals"].items():
        print(m, "post_correct", v["post_present_correct"],
              "post_absent_false", v["post_absent_accepted"],
              "hidden_false", v["hidden_accepted"], flush=True)
    return answer


def audit(data: Path = DATA, output: Path = OUT) -> None:
    man = b.read_pinned(data/"manifest.json")
    assert_manifest(man)
    guard(man)
    by_pair = {}
    for spec in man["scenes"]:
        p = data/"public"/f"{spec['id']}.npy"
        if b.digest(p.read_bytes()) != spec["frames_sha256"]:
            raise RuntimeError("BATCH73_FRAME_PIN_TAMPER")
        b.read_pinned(data/"private"/f"{spec['id']}.json")
        by_pair.setdefault(spec["pair_id"], []).append(spec["frames_sha256"])
    if not all(len(v) == 2 and v[0] == v[1] for v in by_pair.values()):
        raise RuntimeError("BATCH73_PUBLIC_PAIR_TAMPER")
    for split in ("dev", "test"):
        p = output/f"{split}_predictions.json"
        preds = b.read_pinned(p)
        result = b.read_pinned(output/f"{split}_results.json")
        if (result["prediction_sha256"] != b.digest(p.read_bytes()) or
                preds["oracle_read_in_prediction"] is not False):
            raise RuntimeError("BATCH73_PREDICTION_PIN_TAMPER")
    print("BATCH73_FINAL_AUDIT_PASS", man["source_sha"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "predict-dev",
                      "score-dev", "predict-test", "score-test", "audit"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUT)
    a = parser.parse_args()
    if a.action == "prepare":
        prepare(a.data)
    elif a.action.startswith("predict-"):
        predict(a.action[8:], a.data, a.output)
    elif a.action.startswith("score-"):
        score(a.action[6:], a.data, a.output)
    else:
        audit(a.data, a.output)


if __name__ == "__main__":
    main()
