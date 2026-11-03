"""Batch71 preregistered counterfactual identity-ambiguity stress benchmark.

A single public pixel video corresponds to two physically possible hidden
identity assignments. Predict once per independent pair, BEFORE reading either
private oracle, then score both counterfactual worlds without tuning. Synthetic
2D pixels are NOT natural video or calibrated 3D camera evidence.

Protocol: prepare -> predict-dev -> score-dev -> predict-test -> score-test -> audit.
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
DATA = PROJECT.parent / "data" / "batch71_counterfactual_identity_v1"
OUT = PROJECT / "outputs" / "batch71_counterfactual_identity_v1"
CONFIG = PROJECT / "configs" / "kinetics_batch56.json"

WIDTH = 256
FRAMES = 96
HIDDEN_START = 18
HIDDEN_STOP = 64  # exclusive; post-occlusion ambiguity begins here
PCK = 8.0
REGIMES = ("static_twins", "planar_pan_twins", "illumination_twins")
DEV_PER_REGIME = 3
TEST_PER_REGIME = 2
METHODS = b.METHODS
MANIFEST_KIND = "BATCH71_PAIRED_COUNTERFACTUAL_IDENTITY_V1"
PROTOCOL = "frozen-twin-world-identity-ambiguity-no-test-selection-v1"


def specs() -> list[dict]:
    records = []
    for split, count, base in (
        ("dev", DEV_PER_REGIME, 710_111),
        ("test", TEST_PER_REGIME, 719_991),
    ):
        for group, regime in enumerate(REGIMES):
            for ix in range(count):
                records.append({
                    "id": f"{split}_{group:02d}_{ix:02d}",
                    "split": split,
                    "regime": regime,
                    "seed": base + 113 * group + 19 * ix,
                    "query_time": 0,
                    "nframes": FRAMES,
                })
    return records


def positions(spec: dict, t: int) -> tuple[list[float], list[float], list[float]]:
    shift = ([0.26 * t, -0.11 * t] if
             spec["regime"] == "planar_pan_twins" else [0., 0.])
    # Distinguishable positions, indistinguishable visual appearance.
    left = [float(round(71 + 0.19 * t + shift[0])),
            float(round(117 + shift[1]))]
    right = [float(round(185 - 0.19 * t + shift[0])),
             float(round(117 + shift[1]))]
    return left, right, shift


def render(spec: dict) -> tuple[np.ndarray, dict, dict]:
    """Construct EXACTLY ONE video shared by two different latent worlds.

    Both world identities start at the left sprite. A full occluder hides both
    objects at t=18..63. While hidden, they may exchange identities; after t=64
    the query identity is the left (world A) or right (world B) sprite. There
    is no observable physical cue about which exchange happened.
    """
    if spec["regime"] not in REGIMES:
        raise RuntimeError("BATCH71_UNKNOWN_REGIME")
    rng = np.random.default_rng(spec["seed"])
    bg = cv2.GaussianBlur(
        rng.integers(35, 205, (WIDTH, WIDTH), dtype=np.uint8), (5, 5), 0.8)
    sprite = cv2.GaussianBlur(
        rng.integers(30, 225, (17, 17), dtype=np.uint8), (3, 3), 0.5)
    frames = np.empty((FRAMES, WIDTH, WIDTH), dtype=np.uint8)
    world_a, world_b, occlusion, shifts = [], [], [], []
    for t in range(FRAMES):
        left, right, shift = positions(spec, t)
        frame = cv2.warpAffine(bg, np.float32([[1, 0, shift[0]],
                                               [0, 1, shift[1]]]),
                               (WIDTH, WIDTH), borderMode=cv2.BORDER_REFLECT_101)
        hidden = HIDDEN_START <= t < HIDDEN_STOP
        if hidden:
            # Opaque foreground panel covers BOTH sprites; no identity signal.
            xa = int(round(38 + shift[0]))
            xb = int(round(220 + shift[0]))
            ya = int(round(88 + shift[1]))
            yb = int(round(146 + shift[1]))
            frame[ya:yb, xa:xb] = 110
            cv2.rectangle(frame, (xa, ya), (xb - 1, yb - 1), 91, 2)
        else:
            for xy in (left, right):
                x, y = int(xy[0]), int(xy[1])
                frame[y - 8:y + 9, x - 8:x + 9] = sprite
        if spec["regime"] == "illumination_twins":
            gain = 0.84 + 0.13 * math.sin(t / 13.0)
            frame = np.clip(frame.astype(np.float32) * gain + 6, 0, 255
                            ).astype(np.uint8)
        frames[t] = frame
        world_a.append(left)
        # Two worlds have identical query up to occlusion; identity is latent
        # during occlusion, and swaps only upon reappearance.
        world_b.append(right if t >= HIDDEN_STOP else left)
        occlusion.append(hidden)
        shifts.append(shift)
    truth_a = {"xy": world_a, "occluded": list(occlusion),
               "shift_xy": [list(v) for v in shifts],
               "latent_assignment": "left_object_after_ambiguous_occlusion"}
    truth_b = {"xy": world_b, "occluded": list(occlusion),
               "shift_xy": [list(v) for v in shifts],
               "latent_assignment": "right_object_after_ambiguous_occlusion"}
    assert world_a[0] == world_b[0]
    for t in range(HIDDEN_STOP, FRAMES):
        assert math.dist(world_a[t], world_b[t]) > 2 * PCK
    return frames, truth_a, truth_b


def assert_manifest(doc: dict) -> None:
    if (doc.get("kind") != MANIFEST_KIND or doc.get("protocol") != PROTOCOL
            or not isinstance(doc.get("source_sha"), str)
            or doc.get("source_sha") == ""
            or not isinstance(doc.get("scenes"), list)
            or len(doc["scenes"]) != len(specs())):
        raise RuntimeError("BATCH71_INVALID_MANIFEST")
    for row, reference in zip(doc["scenes"], specs()):
        if any(row.get(k) != v for k, v in reference.items()):
            raise RuntimeError("BATCH71_SPLIT_SEED_OR_REGIME_DRIFT")
        if not isinstance(row.get("frames_sha256"), str):
            raise RuntimeError("BATCH71_MISSING_PUBLIC_PIN")
        if row.get("query_xy") is None:
            raise RuntimeError("BATCH71_MISSING_PUBLIC_QUERY")


def source_guard(man: dict) -> None:
    b.assert_clean_source(man["source_sha"])
    if b.digest(CONFIG.read_bytes()) != man["config_sha256"]:
        raise RuntimeError("BATCH71_TRACKER_CONFIGURATION_CHANGED")


def prepare(data: Path = DATA) -> None:
    if (data / "manifest.json").exists() or (data / "public").exists():
        raise RuntimeError("BATCH71_DATA_ALREADY_EXISTS")
    records = []
    for spec in specs():
        frames, world_a, world_b = render(spec)
        video_path = data / "public" / (spec["id"] + ".npy")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        with video_path.open("xb") as handle:
            np.save(handle, frames, allow_pickle=False)
        for world, oracle in (("a", world_a), ("b", world_b)):
            b.save_json(data / "private" / f"{spec['id']}_{world}.json", oracle)
        row = dict(spec)
        row["query_xy"] = world_a["xy"][0]
        row["frames_sha256"] = b.digest(video_path.read_bytes())
        records.append(row)
    man = {
        "kind": MANIFEST_KIND, "protocol": PROTOCOL,
        "source_sha": b.source_sha(),
        "config_sha256": b.digest(CONFIG.read_bytes()),
        "pairs_are_independent_units": True,
        "two_counterfactual_worlds_share_exact_public_frames": True,
        "synthetic_not_natural_or_3d": True,
        "scenes": records,
    }
    assert_manifest(man)
    print("BATCH71_DATA_FROZEN", b.save_json(data / "manifest.json", man),
          "independent_pairs", len(records), flush=True)


def load_frames(data: Path, row: dict) -> list[np.ndarray]:
    p = data / "public" / (row["id"] + ".npy")
    if b.digest(p.read_bytes()) != row["frames_sha256"]:
        raise RuntimeError("BATCH71_PUBLIC_DATA_DRIFT")
    frames = np.load(p, allow_pickle=False)
    if frames.shape != (FRAMES, WIDTH, WIDTH) or frames.dtype != np.uint8:
        raise RuntimeError("BATCH71_PUBLIC_FRAME_FORMAT")
    return [np.ascontiguousarray(f) for f in frames]


def predict(split: str, data: Path = DATA, output: Path = OUT) -> dict:
    if split not in ("dev", "test"):
        raise RuntimeError("BATCH71_INVALID_SPLIT")
    man = b.read_pinned(data / "manifest.json")
    assert_manifest(man)
    source_guard(man)
    if split == "test":
        b.read_pinned(output / "dev_results.json")
    if (output / f"{split}_predictions.json").exists():
        raise RuntimeError("BATCH71_PREDICTIONS_ALREADY_EXIST")
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    cfg = json.loads(CONFIG.read_text())["tracker"]
    rows = []
    for row in man["scenes"]:
        if row["split"] != split:
            continue
        frames = load_frames(data, row)
        query = {"query_time": 0, "query_xy": row["query_xy"]}
        start = time.perf_counter()
        transforms, _, _ = b.core.estimate_affines(frames, cfg)
        trace, timings = b.core.predict_query(frames, query, transforms, cfg)
        trace["template_consensus_4px"] = [
            b.agree(trace[b.BASE], trace["template_recover4"], i)
            for i in range(FRAMES)]
        elapsed = 1000 * (time.perf_counter() - start)
        if any(k not in trace or len(trace[k]) != FRAMES for k in METHODS):
            raise RuntimeError("BATCH71_METHOD_TRACE_DRIFT")
        rows.append({
            "id": row["id"], "regime": row["regime"],
            "traces": {k: trace[k] for k in METHODS},
            "cpu_wall_ms": round(elapsed, 3),
            "component_cpu_ms": {k: timings.get(k) for k in METHODS},
        })
    doc = {
        "kind": "BATCH71_LABEL_BLIND_PAIRED_PREDICTIONS",
        "source_sha": man["source_sha"],
        "manifest_sha256": b.digest((data / "manifest.json").read_bytes()),
        "config_sha256": man["config_sha256"],
        "split": split, "private_oracle_read_during_prediction": False,
        "one_prediction_per_counterfactual_pair": True,
        "predictions": rows,
    }
    print("BATCH71_PREDICTIONS_PINNED", split,
          b.save_json(output / f"{split}_predictions.json", doc),
          len(rows), flush=True)
    return doc


def score_pair(traces: dict, a: dict, c: dict) -> dict:
    if any(len(x["xy"]) != FRAMES or len(x["occluded"]) != FRAMES
           for x in (a, c)):
        raise RuntimeError("BATCH71_PRIVATE_ORACLE_DRIFT")
    if a["xy"][0] != c["xy"][0]:
        raise RuntimeError("BATCH71_DIFFERENT_PUBLIC_QUERY")
    stats = {}
    for method in METHODS:
        trace = traces[method]
        if len(trace) != FRAMES:
            raise RuntimeError("BATCH71_TRACE_LENGTH_DRIFT")
        q = dict(pre_visible=0, pre_accepted=0, pre_correct=0,
                 hidden=0, hidden_accepted=0, max_hidden_run=0,
                 post_visible=0, post_accepted=0, post_correct_a=0,
                 post_correct_b=0, post_accepted_wrong_both=0)
        run = 0
        for t in range(1, FRAMES):
            if a["occluded"][t] != c["occluded"][t]:
                raise RuntimeError("BATCH71_VISIBILITY_WORLD_DRIFT")
            p = trace[t]
            if p is not None and (len(p) != 2 or
                                  not all(math.isfinite(float(v)) for v in p)):
                raise RuntimeError("BATCH71_INVALID_PREDICTED_POINT")
            if HIDDEN_START <= t < HIDDEN_STOP:
                if not a["occluded"][t]:
                    raise RuntimeError("BATCH71_HIDDEN_WINDOW_DRIFT")
                q["hidden"] += 1
                run = run + 1 if p is not None else 0
                q["hidden_accepted"] += p is not None
                q["max_hidden_run"] = max(run, q["max_hidden_run"])
            elif t < HIDDEN_START:
                if a["occluded"][t] or a["xy"][t] != c["xy"][t]:
                    raise RuntimeError("BATCH71_PRE_OCCLUSION_WORLD_DRIFT")
                q["pre_visible"] += 1
                if p is not None:
                    q["pre_accepted"] += 1
                    q["pre_correct"] += math.dist(p, a["xy"][t]) <= PCK
            else:
                if a["occluded"][t] or math.dist(a["xy"][t], c["xy"][t]) <= 2*PCK:
                    raise RuntimeError("BATCH71_POST_WORLD_AMBIGUITY_INVALID")
                q["post_visible"] += 1
                if p is not None:
                    q["post_accepted"] += 1
                    win_a = math.dist(p, a["xy"][t]) <= PCK
                    win_b = math.dist(p, c["xy"][t]) <= PCK
                    if win_a and win_b:
                        raise RuntimeError("BATCH71_DUAL_WORLD_ACCURACY_IMPOSSIBLE")
                    q["post_correct_a"] += win_a
                    q["post_correct_b"] += win_b
                    q["post_accepted_wrong_both"] += not (win_a or win_b)
        stats[method] = q
    return stats


def aggregate(rows: list[dict]) -> dict:
    result = {}
    keys = ("pre_visible", "pre_accepted", "pre_correct", "hidden",
            "hidden_accepted", "post_visible", "post_accepted",
            "post_correct_a", "post_correct_b", "post_accepted_wrong_both")
    for method in METHODS:
        values = {k: sum(row["metrics"][method][k] for row in rows) for k in keys}
        values["max_hidden_run"] = max(
            (row["metrics"][method]["max_hidden_run"] for row in rows), default=0)
        n = values["post_visible"]
        accepted = values["post_accepted"]
        wins = values["post_correct_a"] + values["post_correct_b"]
        if wins > accepted:
            raise RuntimeError("BATCH71_PAIRED_IMPOSSIBILITY_VIOLATED")
        values.update(
            post_coverage=accepted / n if n else None,
            post_pck8_random_world=wins / (2*n) if n else None,
            post_pck8_ceiling_given_acceptance=accepted/(2*n) if n else None,
            post_wrong_accepted_fraction_random_world=
                (2*accepted-wins)/(2*accepted) if accepted else None,
            hidden_false_accept_rate=(values["hidden_accepted"]/values["hidden"]
                                      if values["hidden"] else None),
            pre_pck8=(values["pre_correct"]/values["pre_visible"]
                      if values["pre_visible"] else None))
        result[method] = values
    return result


def score(split: str, data: Path = DATA, output: Path = OUT) -> dict:
    if split not in ("dev", "test"):
        raise RuntimeError("BATCH71_INVALID_SPLIT")
    man = b.read_pinned(data / "manifest.json")
    assert_manifest(man)
    source_guard(man)
    if split == "test":
        b.read_pinned(output / "dev_results.json")
    pred = b.read_pinned(output / f"{split}_predictions.json")
    if (pred["split"] != split or pred["source_sha"] != man["source_sha"] or
            pred["manifest_sha256"] != b.digest((data / "manifest.json").read_bytes()) or
            pred["config_sha256"] != man["config_sha256"] or
            pred["private_oracle_read_during_prediction"] is not False or
            pred["one_prediction_per_counterfactual_pair"] is not True):
        raise RuntimeError("BATCH71_PREDICTION_FIREWALL_DRIFT")
    if (output / f"{split}_results.json").exists():
        raise RuntimeError("BATCH71_RESULTS_ALREADY_EXIST")
    expected = [s for s in man["scenes"] if s["split"] == split]
    if len(pred["predictions"]) != len(expected):
        raise RuntimeError("BATCH71_SPLIT_SIZE_DRIFT")
    rows = []
    for record, scene in zip(pred["predictions"], expected):
        if record["id"] != scene["id"] or record["regime"] != scene["regime"]:
            raise RuntimeError("BATCH71_PREDICTION_ORDER_DRIFT")
        world_a = b.read_pinned(data / "private" / f"{scene['id']}_a.json")
        world_b = b.read_pinned(data / "private" / f"{scene['id']}_b.json")
        rows.append({
            "id": scene["id"], "regime": scene["regime"],
            "metrics": score_pair(record["traces"], world_a, world_b),
            "cpu_wall_ms": record["cpu_wall_ms"],
        })
    times = [row["cpu_wall_ms"] for row in rows]
    answer = {
        "kind": "BATCH71_PAIRED_IDENTITY_RESULTS",
        "split": split, "source_sha": man["source_sha"],
        "prediction_sha256": b.digest(
            (output / f"{split}_predictions.json").read_bytes()),
        "pair_is_independent_unit": True,
        "synthetic_not_natural": True, "tuned_on_test": False,
        "paired_worlds_per_public_scene": 2,
        "scenes": rows, "totals": aggregate(rows),
        "cpu_wall_ms_p50": statistics.median(times),
        "cpu_wall_ms_p95": float(np.percentile(times, 95)),
    }
    print("BATCH71_SCORE_PINNED", split,
          b.save_json(output / f"{split}_results.json", answer), flush=True)
    for method in METHODS:
        row = answer["totals"][method]
        print(method, "post_accepted", row["post_accepted"],
              "post_pck8_random_world", round(row["post_pck8_random_world"], 4),
              "hidden_fa", round(row["hidden_false_accept_rate"], 4),
              flush=True)
    return answer


def audit(data: Path = DATA, output: Path = OUT) -> None:
    man = b.read_pinned(data / "manifest.json")
    assert_manifest(man)
    source_guard(man)
    for scene in man["scenes"]:
        p = data / "public" / f"{scene['id']}.npy"
        if b.digest(p.read_bytes()) != scene["frames_sha256"]:
            raise RuntimeError("BATCH71_FRAME_HASH_DRIFT")
        a = b.read_pinned(data / "private" / f"{scene['id']}_a.json")
        c = b.read_pinned(data / "private" / f"{scene['id']}_b.json")
        if a["xy"][0] != c["xy"][0]:
            raise RuntimeError("BATCH71_WORLD_QUERY_DRIFT")
    for split in ("dev", "test"):
        pred = b.read_pinned(output / f"{split}_predictions.json")
        result = b.read_pinned(output / f"{split}_results.json")
        if (result["prediction_sha256"] !=
                b.digest((output / f"{split}_predictions.json").read_bytes()) or
                pred["private_oracle_read_during_prediction"] is not False):
            raise RuntimeError("BATCH71_AUDIT_PIN_DRIFT")
    print("BATCH71_FINAL_AUDIT_PASS", man["source_sha"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=(
        "prepare", "predict-dev", "score-dev", "predict-test", "score-test", "audit"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.data)
    elif args.action.startswith("predict-"):
        predict(args.action[8:], args.data, args.output)
    elif args.action.startswith("score-"):
        score(args.action[6:], args.data, args.output)
    else:
        audit(args.data, args.output)


if __name__ == "__main__":
    main()
