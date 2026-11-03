"""Batch74 tests: a read-only, fail-closed audit of frozen experiment liveness."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import batch74_frozen_run_watchdog as m


SHA = "a" * 40
NEW_MAIN = "b" * 40
CRON = (
    "*/10 * * * * /bin/bash -lc 'cd /work/repo-batch59-tested && "
    "test ! -e .automation/ALL_APPROVED_STAGES_DONE && "
    "../.venv/bin/python automation/runner_batch59.py tick --max-stages 2' "
    "# MVVG_PROJECT4_10MIN"
)
PS = ("12345 ../.venv/bin/python automation/runner_batch59.py tick --max-stages 2\n"
      "12346 ../.venv/bin/python experiments/kinetics_batch59_data.py download\n")
GIB = 1024 ** 3


def state(running=True):
    return {"plan_version": "frozen-v1", "frozen_sha": SHA,
            "stages": {"R00_SELECT_AND_DOWNLOAD_42_FRESH_CLASS_SHARDS":
                       {"status": "running" if running else "done"},
                       "R01_TARGETED_BATCH59_REGRESSION":
                       {"status": "pending"}}}


def shard(count=26, pending=16, p95=100 * 1024 ** 2, age=2):
    return {"complete": count, "expected": count + pending,
            "remaining": pending, "complete_bytes": count * 50 * 1024 ** 2,
            "partial_count": 1, "active_partial": "k700_val_594.tar.gz.partial",
            "active_partial_bytes": 25 * 1024 ** 2,
            "active_partial_age_seconds": age, "completed_shard_p50_bytes": 50 * 1024 ** 2,
            "completed_shard_p95_bytes": p95}


def audit(**kwargs):
    d = dict(state=state(), checkout_sha=SHA, cron=CRON, ps=PS, gpu="",
             shards=shard(), log="", free_bytes=12 * GIB,
             now=100000.0, done_marker=False, remote_main_sha=NEW_MAIN)
    d.update(kwargs)
    return m.analyze(**d)


def codes(report):
    return [x["code"] for x in report["alerts"]]


def test_valid_snapshot_running_old_sha_does_not_trigger_source_drift():
    a = audit()
    assert a["status"] == "OK"
    assert a["remote_main_sha"] == NEW_MAIN
    assert a["frozen_sha"] == SHA
    assert a["download"]["complete"] == 26
    assert a["download"]["remaining"] == 16
    assert not codes(a)


@pytest.mark.parametrize("new_sha", [None, "c"*40, "", "123"])
def test_frozen_checkout_mismatch_is_blocking(new_sha):
    a = audit(checkout_sha=new_sha)
    assert a["status"] == "BLOCK"
    assert "FROZEN_SOURCE_DRIFT" in codes(a)


@pytest.mark.parametrize("bad_cron", ["", CRON+"\n"+CRON, "# "+CRON,
                                     CRON.replace("*/10", "*/5"),
                                     CRON.replace("repo-batch59-tested", "repo-batch75-tested"),
                                     CRON.replace("ALL_APPROVED_STAGES_DONE", "WRONG_MARKER")])
def test_bad_cron_cannot_pass(bad_cron):
    a = audit(cron=bad_cron)
    assert a["status"] == "BLOCK"
    assert "CRON_CONTRACT" in codes(a)


def test_cron_comment_does_not_count():
    assert m.parse_cron("# disabled MVVG_PROJECT4_10MIN")["matching_entries"] == 0


def test_dirty_tracked_files_block_frozen_run():
    a = audit(tracked_checkout_dirty=True)
    assert a["status"] == "BLOCK"
    assert "FROZEN_TRACKED_FILE_MUTATION" in codes(a)


def test_unknown_tracked_status_must_not_claim_clean():
    a = audit(tracked_checkout_dirty=None)
    assert a["status"] == "WARN"
    assert "FROZEN_TREE_STATUS_UNKNOWN" in codes(a)


def test_gpu_compute_process_is_separate_warning():
    a = audit(gpu="999, python, 8500 MiB\n")
    assert a["status"] == "WARN"
    assert "GPU_COMPUTE_OCCUPIED" in codes(a)
    assert a["gpu_compute_process_count"] == 1


def test_gpu_query_unavailable_is_not_pretended_idle():
    a = audit(gpu=None)
    assert "GPU_QUERY_UNKNOWN" in codes(a)
    assert a["gpu_compute_process_count"] is None


@pytest.mark.parametrize("new_ps", [
    "12345 python automation/runner_batch59.py tick\n",
    "12346 python experiments/kinetics_batch59_data.py download\n",
    "",
])
def test_missing_process_detected(new_ps):
    a = audit(ps=new_ps)
    assert a["status"] == "WARN"


def test_duplicate_runner_blocks():
    a = audit(ps=PS+"12347 python automation/runner_batch59.py tick\n")
    assert "DUPLICATE_BATCH59_PROCESS" in codes(a)


def test_duplicate_download_blocks():
    a = audit(ps=PS+"12347 python experiments/kinetics_batch59_data.py download\n")
    assert "DUPLICATE_BATCH59_PROCESS" in codes(a)


def test_stale_partial_warns_but_does_not_kill_download():
    a = audit(shards=shard(age=m.STALE_SECONDS+1))
    assert "NO_PARTIAL_BYTE_PROGRESS" in codes(a)
    assert a["status"] == "WARN"


def test_recent_partial_healthy():
    assert "NO_PARTIAL_BYTE_PROGRESS" not in codes(audit(shards=shard(age=5)))


def test_below_ten_gib_guard_is_blocking():
    a = audit(free_bytes=9*GIB)
    assert "DISK_BELOW_GUARD" in codes(a)


def test_high_shard_forecast_warns_about_future_headroom():
    a = audit(shards=shard(p95=300*1024**2))
    assert "DISK_FORECAST_TIGHT" in codes(a)
    assert a["storage"]["forecast_is_heuristic"]


def test_stage_multiple_running_is_invalid():
    s = state()
    s["stages"]["R01_TARGETED_BATCH59_REGRESSION"]["status"] = "running"
    a = audit(state=s)
    assert "CONCURRENT_STAGES" in codes(a)


def test_bad_state_rejected():
    a = audit(state={"frozen_sha": SHA})
    assert "MALFORMED_STATE" in codes(a)


def test_done_marker_cannot_be_set_while_running_or_pending():
    a = audit(done_marker=True)
    assert "PREMATURE_DONE_MARKER" in codes(a)


def test_done_marker_missing_after_all_done():
    s = state(False)
    s["stages"]["R01_TARGETED_BATCH59_REGRESSION"]["status"] = "done"
    a = audit(state=s, done_marker=False)
    assert "TERMINAL_MARKER_MISSING" in codes(a)


def test_final_state_with_marker_is_consistent():
    s = state(False)
    s["stages"]["R01_TARGETED_BATCH59_REGRESSION"]["status"] = "done"
    a = audit(state=s, done_marker=True, ps="")
    assert "PREMATURE_DONE_MARKER" not in codes(a)


def test_archives_over_expected_rejected():
    s = shard(count=43, pending=0)
    s["expected"] = 42
    a = audit(shards=s)
    assert "SHARD_COUNT_OVERFLOW" in codes(a)


def test_log_ready_more_than_disk_warns():
    a = audit(log="\n".join(f"KINETICS_BATCH45_SHARD_READY {i:03d} hash" for i in range(30)))
    assert "READY_VS_DISK_COUNT" in codes(a)


def test_progress_parses_restartable_28_without_treating_as_error():
    raw = ("KINETICS_SHARD_RESUME_PROGRESS k700_val_594.tar.gz "
           "attempt=10 rc=28 bytes=33049860/174045245\n")
    d = m.parse_download_log(raw)
    assert d["last_progress"]["curl_rc"] == 28
    assert d["last_progress"]["bytes"] == 33049860
    assert d["last_progress"]["fraction"] == round(33049860/174045245, 5)
    assert d["ready_log_count"] == 0
    assert "GPU" not in d


def test_completed_readiness_deduplicates_repeated_messages():
    text = ("KINETICS_BATCH45_SHARD_READY 086 h\n"
            "KINETICS_BATCH45_SHARD_READY 086 h\n"
            "KINETICS_BATCH45_SHARD_READY 594 h\n")
    assert m.parse_download_log(text)["ready_log_count"] == 2


@pytest.mark.parametrize("raw", ["", "random log", "rc=28 curl timeout"])
def test_malformed_log_does_not_invent_metrics(raw):
    assert m.parse_download_log(raw)["last_progress"] is None


@pytest.mark.parametrize("values,p,expected", [
    ([], .95, 0), ([1], .95, 1),
    ([1, 2, 3, 4], .5, 2), ([1, 2, 3, 4], .95, 4),
    ([2, 8, 4, 6, 10], .8, 8),
])
def test_nearest_rank_percentile(values, p, expected):
    assert m.percentile(values, p) == expected


def test_process_parser_does_not_count_cron_shell_or_diagnostic_grep():
    x = m.process_presence(
        "1 sh -c python automation/runner_batch59.py tick\n"
        "2 grep -E python runner_batch59.py tick\n"
        "3 python automation/runner_batch59.py tick\n"
        "4 python experiments/kinetics_batch59_data.py download\n"
    )
    # Shell is not a python process and is excluded by argv prefix validation.
    assert x["runner_pids"] == [3]
    assert x["downloader_pids"] == [4]


def test_shards_count_without_reading_archive_bytes(tmp_path):
    root = tmp_path / "shards"
    root.mkdir()
    for name, size in [
        ("k700_val_001.tar.gz", 1),
        ("k700_val_002.tar.gz", 3),
        ("k700_val_002.tar.tar.gz.partial", 6),
        ("UNRELATED.mp4", 9000),
    ]:
        f = root / name
        f.write_bytes(b"x" * size)
    d = m.inspect_shards(root, now=10000000, expected=42)
    assert d["complete"] == 2
    assert d["partial_count"] == 1
    assert d["remaining"] == 40
    assert d["completed_shard_p50_bytes"] == 2
    assert d["completed_shard_p95_bytes"] == 3


def test_missing_shard_folder_is_accurately_reported(tmp_path):
    d = m.inspect_shards(tmp_path / "nonexistent", now=100, expected=42)
    assert d["complete"] == 0 and d["remaining"] == 42
    assert d["active_partial"] is None


def test_snapshot_whitelists_inputs_and_avoids_oracle_and_media(tmp_path, monkeypatch):
    auto = tmp_path / ".automation"
    (auto / "logs").mkdir(parents=True)
    (auto / "state.json").write_text(json.dumps(state()))
    (auto / "logs/R00_SELECT_AND_DOWNLOAD_42_FRESH_CLASS_SHARDS.log").write_text("")
    # If implementation ever reads the oracle, fail the regression.
    secret = tmp_path / "data/tapvid_kinetics_batch59/batch59_frozen/oracle-private.json"
    secret.parent.mkdir(parents=True)
    secret.write_text("{ deliberately not valid json }")
    original = Path.read_text
    def no_oracle(self, *args, **kwargs):
        assert "oracle-private" not in str(self)
        assert not str(self).endswith(".mp4")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", no_oracle)
    monkeypatch.setattr(m, "call", lambda argv, timeout=12: (
        SHA if "rev-parse" in argv else
        CRON if argv[0] == "crontab" else
        PS if argv[0] == "ps" else "" ))
    monkeypatch.setattr(m.shutil, "disk_usage",
                        lambda _: SimpleNamespace(free=12*GIB))
    a = m.snapshot(tmp_path, now=10000)
    assert a["schema"] == "batch74-readonly-audit-v1"
    assert a["stages"]["running"] == ["R00_SELECT_AND_DOWNLOAD_42_FRESH_CLASS_SHARDS"]
    assert a["status"] in ("OK", "WARN")
    assert "oracle-private" not in json.dumps(a).lower()


def test_audit_has_no_run_commands():
    import inspect
    source = inspect.getsource(m)
    for forbidden in ("os.system(", "subprocess.Popen(", "os.remove(", "unlink(", "rmtree("):
        assert forbidden not in source
