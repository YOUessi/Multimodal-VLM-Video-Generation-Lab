"""Public source publication guard tests with fake path lists."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("guard",ROOT/"experiments/source_security_guard.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_ban_weights_credentials_media_and_output_dirs():
    values=["README.md","experiments/train.py","models/m.safetensors",
       "outputs/generated.mp4","data/private.json","secret.env","photo.jpg"]
    banned=m.audit_tracked(values)
    assert "README.md" not in banned
    assert "experiments/train.py" not in banned
    assert all(x in banned for x in values[2:])
def test_no_false_positives_on_source():
    assert m.audit_tracked(["README.md","tests/test_example.py",
                           "experiments/generation_step_grid.py",
                           ".github/workflows/ci.yml"])==[]
