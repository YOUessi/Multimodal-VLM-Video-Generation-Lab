"""Batch72 tests: preregistered synthetic causal identity cue and data firewall."""
from pathlib import Path
import copy
import math

import numpy as np
import pytest

from experiments import batch72_cue_observability as m


def specimen(regime="persistent_cue", world="a", seed=720101):
    return dict(id=f"dev_00_00_{world}", pair_id="dev_00_00",
                world=world, split="dev", regime=regime, seed=seed,
                nframes=m.FRAMES, query_time=0)


def test_specs_independent_pairs_and_seed_disjoint():
    specs = m.specs()
    assert len(specs) == 2 * len(m.REGIMES) * (m.DEV_PAIRS + m.TEST_PAIRS)
    assert sum(r["split"] == "dev" for r in specs) == 30
    assert sum(r["split"] == "test" for r in specs) == 20
    assert len(set(x["pair_id"] for x in specs)) == 25
    assert len(set(x["seed"] for x in specs)) == 25
    for p in set(s["pair_id"] for s in specs):
        worlds = [s for s in specs if s["pair_id"] == p]
        assert [r["world"] for r in worlds] == ["a", "b"]
        assert worlds[0]["seed"] == worlds[1]["seed"]


@pytest.mark.parametrize("regime", m.REGIMES)
def test_render_deterministic_for_both_assignments(regime):
    for world in ("a", "b"):
        s = specimen(regime, world)
        x = m.render(s)
        y = m.render(s)
        assert x[0].shape == (96, 256, 256)
        assert x[0].dtype == np.uint8
        assert np.array_equal(x[0], y[0])
        assert x[1] == y[1]


@pytest.mark.parametrize("regime", m.REGIMES)
def test_observability_pair_fingerprint_and_latent_oracles(regime):
    px_a, a = m.render(specimen(regime, "a"))
    px_b, b = m.render(specimen(regime, "b"))
    assert np.array_equal(px_a[:m.OCC_STOP], px_b[:m.OCC_STOP])
    assert np.array_equal(px_a, px_b) == (regime in m.AMBIGUOUS)
    assert a["xy"][0] == b["xy"][0]
    assert a["occluded"] == b["occluded"]
    for t in range(m.OCC_STOP, m.FRAMES):
        assert math.dist(a["xy"][t], b["xy"][t]) > 2 * m.PCK


@pytest.mark.parametrize("regime", m.REGIMES)
def test_hidden_window_is_fully_opaque_and_query_unchanged(regime):
    images, truth = m.render(specimen(regime))
    assert truth["occluded"][17] is False
    assert truth["occluded"][18] is True
    assert truth["occluded"][63] is True
    assert truth["occluded"][64] is False
    for t in (18, 40, 63):
        left, right, _ = m.centers(regime, t)
        for x, y in (left, right):
            expected = (int(np.uint8(np.clip(112*(0.80 + 0.16*math.sin(t/11.0)) + 7, 0, 255)))
                        if regime == "pan_gain_persistent" else 112)
            assert images[t, int(y), int(x)] == expected
    assert truth["xy"][0] == m.centers(regime, 0)[0]


def test_background_translation_only_planar_no_fake_3d():
    static = m.render(specimen("persistent_cue"))[1]
    shifted = m.render(specimen("pan_gain_persistent"))[1]
    assert static["background_translation_xy"][-1] == [0., 0.]
    assert shifted["background_translation_xy"][-1] != [0., 0.]
    assert shifted["synthetic_not_3d_camera"] is True


@pytest.mark.parametrize("world", ("a", "b"))
def test_unique_cue_reacquired_on_opposite_side_if_observable(world):
    frames, truth = m.render(specimen("persistent_cue", world))
    trace = m.identity_match(list(frames), truth["xy"][0], True)
    assert all(p is None for p in trace[m.OCC_START:m.OCC_STOP])
    assert sum(p is not None and math.dist(p, truth["xy"][t]) <= m.PCK
               for t, p in enumerate(trace[m.OCC_STOP:], m.OCC_STOP)) >= 28


def test_ambiguous_clone_rejected_by_margin_but_best_score_accepts():
    frames, truth = m.render(specimen("cloned_cue"))
    best = m.identity_match(list(frames), truth["xy"][0], False)
    safe = m.identity_match(list(frames), truth["xy"][0], True)
    assert sum(p is not None for p in best[m.OCC_STOP:]) >= 28
    assert all(p is None for p in safe[m.OCC_STOP:])


def test_identical_twins_refuse_spurious_identity_recovery():
    frames, truth = m.render(specimen("identical_twins"))
    safe = m.identity_match(list(frames), truth["xy"][0], True)
    assert all(p is None for p in safe[m.OCC_STOP:])


def test_erased_cue_cannot_supply_durable_identity():
    frames, truth = m.render(specimen("erased_cue"))
    safe = m.identity_match(list(frames), truth["xy"][0], True)
    assert all(p is None for p in safe[m.OCC_STOP:])


@pytest.mark.parametrize("world", ("a", "b"))
def test_illumination_and_pan_preserve_unique_signature(world):
    frames, truth = m.render(specimen("pan_gain_persistent", world))
    safe = m.identity_match(list(frames), truth["xy"][0], True)
    assert sum(p is not None and math.dist(p, truth["xy"][t]) <= m.PCK
               for t, p in enumerate(safe[m.OCC_STOP:], m.OCC_STOP)) >= 28


def test_safe_template_match_query_near_border_rejected():
    image = np.zeros((64, 64), dtype=np.uint8)
    with pytest.raises(ValueError, match="QUERY_PATCH_OOB"):
        m.identity_match([image], [0, 5], True)
    with pytest.raises(ValueError, match="EXPECT_GRAYSCALE"):
        m.identity_match([np.zeros((16, 16, 3), dtype=np.uint8)],
                         [10, 10], True)


def test_constant_query_patch_abstains():
    image = np.ones((64, 64), dtype=np.uint8) * 95
    got = m.identity_match([image, image], [30., 30.], True)
    assert got == [[30., 30.], None]


def test_score_hidden_false_accept_and_reacquisition_delay():
    _, oracle = m.render(specimen())
    perfect = [list(v) for v in oracle["xy"]]
    stats = m.score_scene(perfect, oracle)
    assert stats["pre_visible"] == stats["pre_correct"] == 17
    assert stats["hidden"] == stats["hidden_accepted"] == 46
    assert stats["longest_hidden_false_run"] == 46
    assert stats["post_visible"] == stats["post_correct"] == 32
    assert stats["post_first_correct_delay"] == 0


def test_score_complete_abstention_and_invalid_trace_fail_closed():
    _, oracle = m.render(specimen())
    trace = [oracle["xy"][0]] + [None] * (m.FRAMES - 1)
    stats = m.score_scene(trace, oracle)
    assert stats["hidden_accepted"] == stats["post_accepted"] == 0
    assert stats["post_first_correct_delay"] is None
    with pytest.raises(RuntimeError, match="SIZE_DRIFT"):
        m.score_scene(trace[:-1], oracle)
    trace[72] = [float("nan"), 4.]
    with pytest.raises(RuntimeError, match="INVALID_PREDICTION"):
        m.score_scene(trace, oracle)


def test_score_rejects_corrupted_occlusion_truth():
    _, oracle = m.render(specimen())
    oracle["occluded"][30] = False
    with pytest.raises(RuntimeError, match="OCCLUSION_TRUTH_DRIFT"):
        m.score_scene([None]*m.FRAMES, oracle)


def test_aggregate_recapture_and_wrong_visible_acceptance():
    _, truth = m.render(specimen())
    good = [list(x) for x in truth["xy"]]
    wrong = copy.deepcopy(good)
    for t in range(m.OCC_STOP, m.FRAMES):
        wrong[t] = [0., 0.]
    rows = []
    for trace in (good, wrong):
        stats = m.score_scene(trace, truth)
        rows.append({"metrics": {key: stats for key in m.METHODS}})
    totals = m.aggregate(rows)[m.METHODS[0]]
    assert totals["post_visible"] == 64
    assert totals["post_correct"] == 32
    assert totals["post_wrong_accepted"] == 32
    assert totals["post_reacquired_scenes"] == 1
    assert totals["post_correct_coverage"] == 0.5
    assert totals["post_wrong_acceptance_rate"] == 0.5


def test_strict_manifest_protocol_and_seed_contract():
    man = dict(kind=m.MANIFEST_KIND, protocol=m.PROTOCOL, source_sha="x",
               scenes=[{**s, "frames_sha256": "a"*64,
                        "query_xy": [70., 118.]}
                       for s in m.specs()])
    m.assert_manifest(man)
    man["scenes"][0]["seed"] += 1
    with pytest.raises(RuntimeError, match="SPLIT_OR_SEED_DRIFT"):
        m.assert_manifest(man)
    man["scenes"][0]["seed"] -= 1
    man["protocol"] = "modified"
    with pytest.raises(RuntimeError, match="INVALID_MANIFEST"):
        m.assert_manifest(man)


def test_write_once_and_pin_detects_tamper(tmp_path):
    p = tmp_path/"report.json"
    m.b.save_json(p, {"v": 1})
    with pytest.raises(FileExistsError):
        m.b.save_json(p, {"v": 2})
    p.write_text('{"v":999}')
    with pytest.raises(RuntimeError, match="HASH_DRIFT"):
        m.b.read_pinned(p)


def test_no_private_label_in_prediction_stage(tmp_path, monkeypatch):
    scene = specimen()
    frames, truth = m.render(scene)
    public = [{**scene, "frames_sha256": "a"*64,
               "query_xy": truth["xy"][0]}]
    monkeypatch.setattr(m, "specs", lambda: [scene])
    m.b.save_json(tmp_path/"manifest.json",
                  dict(kind=m.MANIFEST_KIND, protocol=m.PROTOCOL,
                       source_sha="frozen", config_sha256="frozen",
                       scenes=public))
    monkeypatch.setattr(m, "source_guard", lambda man: None)
    monkeypatch.setattr(m, "load_frames", lambda data, row: list(frames))
    monkeypatch.setattr(m.b.core, "estimate_affines",
                        lambda f, cfg: ([None]*len(f), [], None))
    monkeypatch.setattr(m.b.core, "predict_query",
                        lambda fs, q, aff, cfg:
                        ({name: [None]*len(fs) for name in m.METHODS}, {}))
    real_open = Path.open

    def prevent_private(self, *args, **kwargs):
        if "private" in self.parts:
            raise AssertionError("BATCH72_PREDICTION_LEAKED_ORACLE")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", prevent_private)
    row = m.predict("dev", tmp_path, tmp_path/"out")
    assert row["oracle_read_in_prediction"] is False
    assert len(row["predictions"]) == 1


def test_test_stage_requires_dev_score_pin(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "source_guard", lambda man: None)
    scene = specimen()
    monkeypatch.setattr(m, "specs", lambda: [scene])
    m.b.save_json(tmp_path/"manifest.json",
                  dict(kind=m.MANIFEST_KIND, protocol=m.PROTOCOL,
                       source_sha="x", config_sha256="x",
                       scenes=[{**scene, "frames_sha256": "a"*64,
                                "query_xy": [70., 118.]}]))
    with pytest.raises(RuntimeError, match="MISSING_PIN"):
        m.predict("test", tmp_path, tmp_path/"out")


def test_invalid_split_rejected_before_any_access():
    with pytest.raises(RuntimeError, match="INVALID_SPLIT"):
        m.predict("future")
    with pytest.raises(RuntimeError, match="INVALID_SPLIT"):
        m.score("future")


def test_synthetic_ambiguity_bound_on_identical_worlds():
    for regime in m.AMBIGUOUS:
        _, a = m.render(specimen(regime, "a"))
        _, bb = m.render(specimen(regime, "b"))
        trace = [list(v) for v in a["xy"]]
        one = m.score_scene(trace, a)
        two = m.score_scene(trace, bb)
        assert one["post_accepted"] == two["post_accepted"] == 32
        assert one["post_correct"] + two["post_correct"] <= 32
