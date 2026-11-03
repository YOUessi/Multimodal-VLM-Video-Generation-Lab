"""Batch72 prospective cue-observability / recovery benchmark.

Two counterfactual assignments share pre-occlusion query identity.  Stable
appearance evidence is the ONLY visual difference between the two possible
post-occlusion worlds in observable conditions.  When cues are erased, cloned,
or were never distinctive, both worlds have identical public pixels and
differ only in private identity truth.

All methods, thresholds, scenes, seeds and metrics are frozen before TEST.
Synthetic 2D background translation is NOT physical 3D camera motion.
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

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT.parent / "data" / "batch72_cue_observability_v1"
OUT = PROJECT / "outputs" / "batch72_cue_observability_v1"
CONFIG = PROJECT / "configs" / "kinetics_batch56.json"
REGIMES = ("persistent_cue", "pan_gain_persistent", "identical_twins",
           "erased_cue", "cloned_cue")
AMBIGUOUS = frozenset(("identical_twins", "erased_cue", "cloned_cue"))
DEV_PAIRS = 3
TEST_PAIRS = 2
FRAMES = 96
WIDTH = 256
OCC_START = 18
OCC_STOP = 64
PCK = 8.0
PATCH = 17
# Fixed a priori, NOT selected on DEV or TEST
MATCH_SCORE = 0.86
IDENTITY_MARGIN = 0.07
METHODS = ("appearance_fb_055", "template_recover4",
           "best_template_score", "identity_margin_veto")
PROTOCOL = "batch72-prospective-paired-cue-observability-v1"
MANIFEST_KIND = "BATCH72_FROZEN_CUE_OBSERVABILITY_V1"


def specs() -> list[dict]:
    rows = []
    for split, count, seed_base in (
        ("dev", DEV_PAIRS, 720_101), ("test", TEST_PAIRS, 729_901)
    ):
        for group, regime in enumerate(REGIMES):
            for k in range(count):
                pair_id = f"{split}_{group:02d}_{k:02d}"
                seed = seed_base + 337 * group + 19 * k
                for world in ("a", "b"):
                    rows.append(dict(id=f"{pair_id}_{world}", pair_id=pair_id,
                                     world=world, split=split, regime=regime,
                                     seed=seed, nframes=FRAMES,
                                     query_time=0))
    return rows


def centers(regime: str, t: int) -> tuple[list[float], list[float], list[float]]:
    shift = ([0.30 * t, -0.12 * t] if regime == "pan_gain_persistent"
             else [0.0, 0.0])
    left = [float(round(70 + 0.17*t + shift[0])),
            float(round(118 + shift[1]))]
    right = [float(round(186 - 0.17*t + shift[0])),
             float(round(118 + shift[1]))]
    return left, right, shift


def render(spec: dict) -> tuple[np.ndarray, dict]:
    regime, world, seed = spec["regime"], spec["world"], spec["seed"]
    if regime not in REGIMES or world not in ("a", "b"):
        raise RuntimeError("BATCH72_INVALID_RENDER_SPEC")
    rng = np.random.default_rng(seed)
    bg = cv2.GaussianBlur(rng.integers(45, 205, (WIDTH, WIDTH), dtype=np.uint8),
                          (5, 5), 0.9)
    # Both sprites are real images. Unique cue is local texture, not an oracle ID.
    target = cv2.GaussianBlur(rng.integers(28, 229, (PATCH, PATCH),
                                           dtype=np.uint8), (3, 3), 0.5)
    distractor = cv2.GaussianBlur(rng.integers(28, 229, (PATCH, PATCH),
                                               dtype=np.uint8), (3, 3), 0.5)
    erased = cv2.GaussianBlur(rng.integers(30, 225, (PATCH, PATCH),
                                           dtype=np.uint8), (3, 3), 0.5)
    frames = np.empty((FRAMES, WIDTH, WIDTH), dtype=np.uint8)
    truth_xy, occluded, backgrounds = [], [], []
    for t in range(FRAMES):
        left, right, shift = centers(regime, t)
        img = cv2.warpAffine(bg, np.float32([[1, 0, shift[0]],
                                             [0, 1, shift[1]]]),
                             (WIDTH, WIDTH),
                             borderMode=cv2.BORDER_REFLECT_101)
        hidden = OCC_START <= t < OCC_STOP
        if hidden:
            x1 = int(round(38 + shift[0]))
            x2 = int(round(218 + shift[0]))
            y1 = int(round(89 + shift[1]))
            y2 = int(round(146 + shift[1]))
            img[y1:y2, x1:x2] = 112
            cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), 88, 2)
        else:
            if t < OCC_START:
                sprites = (target, target if regime == "identical_twins"
                           else distractor)
            elif regime == "identical_twins":
                sprites = (target, target)
            elif regime == "erased_cue":
                sprites = (erased, erased)
            elif regime == "cloned_cue":
                sprites = (target, target)
            else:
                sprites = ((target, distractor) if world == "a"
                           else (distractor, target))
            for xy, sprite in zip((left, right), sprites):
                x, y = map(int, xy)
                img[y-8:y+9, x-8:x+9] = sprite
        if regime == "pan_gain_persistent":
            gain = 0.80 + 0.16*math.sin(t/11.0)
            img = np.clip(img.astype(np.float32)*gain + 7, 0, 255
                          ).astype(np.uint8)
        frames[t] = img
        # Both worlds share initial query and pre-occlusion identity.
        post_xy = (right if world == "b" else left)
        truth_xy.append(left if t < OCC_STOP else post_xy)
        occluded.append(hidden)
        backgrounds.append(shift)
    return frames, dict(xy=truth_xy, occluded=occluded,
                        background_translation_xy=backgrounds,
                        latent_world=world, synthetic_not_3d_camera=True)


def identity_match(frames: list[np.ndarray], query_xy: list[float],
                   require_margin: bool) -> list[list[float] | None]:
    """Global appearance memory, with optional nonredundant runner-up veto.

    There is no oracle candidate box, visibility signal, camera truth or
    post-occlusion template update. For ties the cautious method abstains.
    """
    if not frames or len(frames[0].shape) != 2:
        raise ValueError("BATCH72_EXPECT_GRAYSCALE_FRAMES")
    cx, cy = [int(round(float(v))) for v in query_xy]
    pad = PATCH // 2
    if not (pad <= cx < frames[0].shape[1] - pad and
            pad <= cy < frames[0].shape[0] - pad):
        raise ValueError("BATCH72_QUERY_PATCH_OOB")
    reference = frames[0][cy-pad:cy+pad+1, cx-pad:cx+pad+1]
    if float(np.std(reference)) < 4:
        return [list(map(float, query_xy))] + [None]*(len(frames)-1)
    track: list[list[float] | None] = [list(map(float, query_xy))]
    for frame in frames[1:]:
        similarity = cv2.matchTemplate(frame, reference, cv2.TM_CCOEFF_NORMED)
        if not np.isfinite(similarity).all():
            track.append(None)
            continue
        _, top, _, top_loc = cv2.minMaxLoc(similarity)
        if top < MATCH_SCORE:
            track.append(None)
            continue
        if require_margin:
            # Reject exact or near-exact spatially independent duplicates;
            # nearby sliding positions of the SAME sprite are not new objects.
            remaining = similarity.copy()
            x, y = top_loc
            radius = PATCH
            remaining[max(0, y-radius):min(remaining.shape[0], y+radius+1),
                      max(0, x-radius):min(remaining.shape[1], x+radius+1)] = -1.0
            second = float(np.max(remaining))
            if (top - second) < IDENTITY_MARGIN:
                track.append(None)
                continue
        track.append([float(top_loc[0] + pad), float(top_loc[1] + pad)])
    return track


def assert_manifest(man: dict) -> None:
    if (man.get("kind") != MANIFEST_KIND or man.get("protocol") != PROTOCOL or
            not isinstance(man.get("source_sha"), str) or
            not isinstance(man.get("scenes"), list) or
            len(man["scenes"]) != len(specs())):
        raise RuntimeError("BATCH72_INVALID_MANIFEST")
    for got, expected in zip(man["scenes"], specs()):
        if any(got.get(key) != val for key, val in expected.items()):
            raise RuntimeError("BATCH72_SPLIT_OR_SEED_DRIFT")
        if not isinstance(got.get("frames_sha256"), str) or not isinstance(
                got.get("query_xy"), list):
            raise RuntimeError("BATCH72_MISSING_PUBLIC_PINS")


def source_guard(man: dict) -> None:
    b.assert_clean_source(man["source_sha"])
    if b.digest(CONFIG.read_bytes()) != man["config_sha256"]:
        raise RuntimeError("BATCH72_TRACKER_CONFIG_DRIFT")


def prepare(data: Path = DATA) -> dict:
    if (data/"manifest.json").exists() or (data/"public").exists():
        raise RuntimeError("BATCH72_DATA_ALREADY_EXISTS")
    rows = []
    pair_shas: dict[str, list[str]] = {}
    for spec in specs():
        pixels, truth = render(spec)
        assert pixels.shape == (FRAMES, WIDTH, WIDTH)
        filename = data/"public"/f"{spec['id']}.npy"
        filename.parent.mkdir(parents=True, exist_ok=True)
        with filename.open("xb") as f:
            np.save(f, pixels, allow_pickle=False)
        b.save_json(data/"private"/f"{spec['id']}.json", truth)
        row = dict(spec)
        row["frames_sha256"] = b.digest(filename.read_bytes())
        row["query_xy"] = truth["xy"][0]
        rows.append(row)
        pair_shas.setdefault(spec["pair_id"], []).append(row["frames_sha256"])
    for row in rows:
        values = pair_shas[row["pair_id"]]
        equal = values[0] == values[1]
        if equal != (row["regime"] in AMBIGUOUS):
            raise RuntimeError("BATCH72_OBSERVABILITY_CONTRACT_FAILED")
    man = dict(kind=MANIFEST_KIND, protocol=PROTOCOL, source_sha=b.source_sha(),
               config_sha256=b.digest(CONFIG.read_bytes()), scenes=rows,
               independent_units="pairs_not_worlds",
               synthetic_not_natural_or_3d=True,
               preregistered_match_score=MATCH_SCORE,
               preregistered_identity_margin=IDENTITY_MARGIN)
    assert_manifest(man)
    print("BATCH72_PREPARED", b.save_json(data/"manifest.json", man),
          "pairs", len(pair_shas), "scenes", len(rows), flush=True)
    return man


def load_frames(data: Path, spec: dict) -> list[np.ndarray]:
    path = data/"public"/f"{spec['id']}.npy"
    if b.digest(path.read_bytes()) != spec["frames_sha256"]:
        raise RuntimeError("BATCH72_PUBLIC_FRAME_DRIFT")
    arr = np.load(path, allow_pickle=False)
    if arr.dtype != np.uint8 or arr.shape != (FRAMES, WIDTH, WIDTH):
        raise RuntimeError("BATCH72_PUBLIC_FRAME_FORMAT_DRIFT")
    return [np.ascontiguousarray(x) for x in arr]


def predict(split: str, data: Path = DATA, output: Path = OUT) -> dict:
    if split not in ("dev", "test"):
        raise RuntimeError("BATCH72_INVALID_SPLIT")
    man = b.read_pinned(data/"manifest.json")
    assert_manifest(man)
    source_guard(man)
    if split == "test":
        b.read_pinned(output/"dev_results.json")
    if (output/f"{split}_predictions.json").exists():
        raise RuntimeError("BATCH72_PREDICTIONS_ALREADY_EXIST")
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    cfg = json.loads(CONFIG.read_text())["tracker"]
    rows = []
    for spec in man["scenes"]:
        if spec["split"] != split:
            continue
        frames = load_frames(data, spec)
        q = dict(query_time=0, query_xy=spec["query_xy"])
        t0 = time.perf_counter()
        transforms, _, _ = b.core.estimate_affines(frames, cfg)
        traces, _ = b.core.predict_query(frames, q, transforms, cfg)
        frozen_ms = (time.perf_counter() - t0)*1000
        t0 = time.perf_counter()
        traces["best_template_score"] = identity_match(frames, q["query_xy"], False)
        best_ms = (time.perf_counter() - t0)*1000
        t0 = time.perf_counter()
        traces["identity_margin_veto"] = identity_match(frames, q["query_xy"], True)
        veto_ms = (time.perf_counter() - t0)*1000
        if any(k not in traces or len(traces[k]) != FRAMES for k in METHODS):
            raise RuntimeError("BATCH72_TRACE_METHOD_DRIFT")
        rows.append(dict(id=spec["id"], pair_id=spec["pair_id"],
                         world=spec["world"], regime=spec["regime"],
                         traces={k: traces[k] for k in METHODS},
                         cpu_ms={"legacy": round(frozen_ms, 3),
                                 "best_template_score": round(best_ms, 3),
                                 "identity_margin_veto": round(veto_ms, 3)}))
    answer = dict(kind="BATCH72_LABEL_BLIND_PREDICTIONS",
                  split=split, source_sha=man["source_sha"],
                  manifest_sha256=b.digest((data/"manifest.json").read_bytes()),
                  oracle_read_in_prediction=False, predictions=rows)
    print("BATCH72_PREDICT_PINNED", split,
          b.save_json(output/f"{split}_predictions.json", answer),
          len(rows), flush=True)
    return answer


def score_scene(trace: list, truth: dict) -> dict:
    if len(trace) != FRAMES or len(truth["xy"]) != FRAMES or len(
            truth["occluded"]) != FRAMES:
        raise RuntimeError("BATCH72_TRACE_ORACLE_SIZE_DRIFT")
    q = dict(pre_visible=0, pre_accepted=0, pre_correct=0,
             hidden=0, hidden_accepted=0, longest_hidden_false_run=0,
             post_visible=0, post_accepted=0, post_correct=0,
             post_wrong_accepted=0, post_first_correct_delay=None)
    run = 0
    for t in range(1, FRAMES):
        p = trace[t]
        if p is not None and (len(p) != 2 or not all(
                math.isfinite(float(v)) for v in p)):
            raise RuntimeError("BATCH72_INVALID_PREDICTION")
        if truth["occluded"][t] != (OCC_START <= t < OCC_STOP):
            raise RuntimeError("BATCH72_OCCLUSION_TRUTH_DRIFT")
        if OCC_START <= t < OCC_STOP:
            q["hidden"] += 1
            q["hidden_accepted"] += p is not None
            run = (run+1 if p is not None else 0)
            q["longest_hidden_false_run"] = max(
                q["longest_hidden_false_run"], run)
            continue
        correct = p is not None and math.dist(p, truth["xy"][t]) <= PCK
        if t < OCC_START:
            q["pre_visible"] += 1
            q["pre_accepted"] += p is not None
            q["pre_correct"] += correct
        else:
            q["post_visible"] += 1
            q["post_accepted"] += p is not None
            q["post_correct"] += correct
            q["post_wrong_accepted"] += (p is not None and not correct)
            if correct and q["post_first_correct_delay"] is None:
                q["post_first_correct_delay"] = t - OCC_STOP
    return q


def aggregate(rows: list[dict]) -> dict:
    result = {}
    count_keys = ("pre_visible", "pre_accepted", "pre_correct", "hidden",
                  "hidden_accepted", "post_visible", "post_accepted",
                  "post_correct", "post_wrong_accepted")
    for method in METHODS:
        metrics = [row["metrics"][method] for row in rows]
        totals = {k: sum(x[k] for x in metrics) for k in count_keys}
        totals["longest_hidden_false_run"] = max(
            [x["longest_hidden_false_run"] for x in metrics], default=0)
        n = totals["post_visible"]
        totals.update(
            post_correct_coverage=(totals["post_correct"]/n if n else None),
            post_acceptance_coverage=(totals["post_accepted"]/n if n else None),
            post_wrong_acceptance_rate=(totals["post_wrong_accepted"]/n
                                        if n else None),
            hidden_false_accept_rate=(totals["hidden_accepted"]/
                                      totals["hidden"] if totals["hidden"] else None),
            post_reacquired_scenes=sum(
                x["post_first_correct_delay"] is not None for x in metrics),
            post_reacquisition_delay_median=(
                statistics.median(x["post_first_correct_delay"] for x in metrics
                                  if x["post_first_correct_delay"] is not None)
                if any(x["post_first_correct_delay"] is not None for x in metrics)
                else None))
        result[method] = totals
    return result


def score(split: str, data: Path = DATA, output: Path = OUT) -> dict:
    if split not in ("dev", "test"):
        raise RuntimeError("BATCH72_INVALID_SPLIT")
    man = b.read_pinned(data/"manifest.json")
    assert_manifest(man)
    source_guard(man)
    if split == "test":
        b.read_pinned(output/"dev_results.json")
    pred = b.read_pinned(output/f"{split}_predictions.json")
    if (pred["split"] != split or pred["source_sha"] != man["source_sha"] or
            pred["manifest_sha256"] !=
            b.digest((data/"manifest.json").read_bytes()) or
            pred["oracle_read_in_prediction"] is not False):
        raise RuntimeError("BATCH72_PREDICTION_FIREWALL_DRIFT")
    if (output/f"{split}_results.json").exists():
        raise RuntimeError("BATCH72_RESULTS_ALREADY_EXIST")
    expected = [x for x in man["scenes"] if x["split"] == split]
    if len(expected) != len(pred["predictions"]):
        raise RuntimeError("BATCH72_PREDICTION_COUNT_DRIFT")
    rows = []
    for spec, row in zip(expected, pred["predictions"]):
        if row["id"] != spec["id"] or row["pair_id"] != spec["pair_id"]:
            raise RuntimeError("BATCH72_PREDICTION_ORDER_DRIFT")
        oracle = b.read_pinned(data/"private"/f"{spec['id']}.json")
        rows.append(dict(id=spec["id"], pair_id=spec["pair_id"],
                         world=spec["world"], regime=spec["regime"],
                         metrics={m: score_scene(row["traces"][m], oracle)
                                  for m in METHODS}, cpu_ms=row["cpu_ms"]))
    per_regime = {regime: aggregate([r for r in rows if r["regime"] == regime])
                  for regime in REGIMES}
    for regime in AMBIGUOUS:
        by_pair = {}
        for r in pred["predictions"]:
            if r["regime"] == regime:
                by_pair.setdefault(r["pair_id"], []).append(r["traces"])
        for traces in by_pair.values():
            if len(traces) != 2 or traces[0] != traces[1]:
                raise RuntimeError("BATCH72_PIXEL_IDENTICAL_WORLD_TRACE_DRIFT")
        for method in METHODS:
            total = per_regime[regime][method]
            if 2*total["post_correct"] > total["post_accepted"]:
                raise RuntimeError("BATCH72_AMBIGUITY_INFORMATION_BOUND_BROKEN")
    costs = {m: sorted([r["cpu_ms"][m] for r in rows])
             for m in ("legacy", "best_template_score",
                       "identity_margin_veto")}
    answer = dict(kind="BATCH72_PROSPECTIVE_CUE_RESULTS",
                  split=split, source_sha=man["source_sha"],
                  prediction_sha256=b.digest(
                      (output/f"{split}_predictions.json").read_bytes()),
                  independent_pairs=len(rows)//2,
                  correlated_counterfactual_worlds=True,
                  new_synthetic_not_natural=True, tuned_on_test=False,
                  methods=METHODS, scenes=rows, totals=aggregate(rows),
                  regimes=per_regime,
                  cpu_ms={k: {"p50": statistics.median(v),
                              "p95": float(np.percentile(v, 95))}
                          for k, v in costs.items()})
    print("BATCH72_SCORE_PINNED", split,
          b.save_json(output/f"{split}_results.json", answer), flush=True)
    for method in METHODS:
        t = answer["totals"][method]
        print(method, "post_correct", t["post_correct"],
              "post_wrong", t["post_wrong_accepted"],
              "hidden_false", t["hidden_accepted"], flush=True)
    return answer


def audit(data: Path = DATA, output: Path = OUT) -> None:
    man = b.read_pinned(data/"manifest.json")
    assert_manifest(man)
    source_guard(man)
    pairs: dict[str, list[str]] = {}
    for scene in man["scenes"]:
        p = data/"public"/f"{scene['id']}.npy"
        if b.digest(p.read_bytes()) != scene["frames_sha256"]:
            raise RuntimeError("BATCH72_PUBLIC_TAMPER")
        b.read_pinned(data/"private"/f"{scene['id']}.json")
        pairs.setdefault(scene["pair_id"], []).append(scene["frames_sha256"])
    for spec in man["scenes"]:
        pair = pairs[spec["pair_id"]]
        if len(pair) != 2 or (pair[0] == pair[1]) != (
                spec["regime"] in AMBIGUOUS):
            raise RuntimeError("BATCH72_PAIR_OBSERVABILITY_DRIFT")
    for split in ("dev", "test"):
        p = output/f"{split}_predictions.json"
        pred = b.read_pinned(p)
        result = b.read_pinned(output/f"{split}_results.json")
        if (result["prediction_sha256"] != b.digest(p.read_bytes()) or
                pred["oracle_read_in_prediction"] is not False):
            raise RuntimeError("BATCH72_PIN_AUDIT_FAIL")
    print("BATCH72_FINAL_AUDIT_PASS", man["source_sha"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=(
        "prepare", "predict-dev", "score-dev", "predict-test",
        "score-test", "audit"))
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
