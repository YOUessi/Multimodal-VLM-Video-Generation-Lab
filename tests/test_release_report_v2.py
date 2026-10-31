"""Final report must never silently use the invalid v1 MCQ metric."""
from pathlib import Path
import importlib.util
p=Path(__file__).resolve().parents[1]/"experiments"/"release_report_v2.py"
spec=importlib.util.spec_from_file_location("release",p)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def test_sha256_is_stable(tmp_path):
    f=tmp_path/"f";f.write_bytes(b"abc")
    assert mod.sha256(f)=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
def test_contract_requires_real_artifacts(tmp_path,monkeypatch):
    monkeypatch.setattr(mod,"ROOT",tmp_path)
    try:mod.build()
    except FileNotFoundError:pass
    else:raise AssertionError("Should reject incomplete release artifacts")
