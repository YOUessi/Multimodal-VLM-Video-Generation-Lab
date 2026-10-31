"""Source-only test of frozen descriptive posthoc motion error slicing."""
from pathlib import Path
import importlib.util,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
def test_error_slice_source_is_descriptive_no_training():
    p=ROOT/"experiments/object_motion_error_slices.py"
    s=p.read_text()
    assert "ERROR_SLICES.md" in s
    assert "SOURCE" not in s or "MANIFEST" in s
    assert "optimizer.step()" not in s
    assert "model.generate(" not in s
    assert "External CLEVRER" in s
