"""VLM point-coordinate parsing and strict abstention contract."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import tapvid_rgb_vlm as vlm


@pytest.mark.parametrize("answer,expected,kind", [
    ("ABSENT", None, "explicit_absent"),
    ("0,0", [0,0], "valid_xy"),
    ("255, 255", [255,255], "valid_xy"),
    ("88, 122", [88,122], "valid_xy"),
    ("256,0", None, "invalid_format"),
    ("-1,5", None, "invalid_format"),
    ("POINT: 20,30", None, "invalid_format"),
    ("Point is missing", None, "invalid_format"),
    ("ABSENT. ", None, "invalid_format"),
    ("999,999", None, "invalid_format"),
    ("12,13\n14,15", None, "invalid_format"),
    ("", None, "invalid_format"),
])
def test_vlm_strict_grammar_and_boundaries(answer,expected,kind):
    xy, status=vlm.strict_point(answer,256,256)
    assert xy==expected and status==kind


def test_cross_only_modifies_source_frame():
    frames=np.zeros((256,256,3),dtype=np.uint8)
    result=vlm.query_image(frames,[40,70])
    assert result.size==(256,256)
    assert frames.sum()==0
    p=np.asarray(result)
    assert p[70,40].sum()>0
    assert p[10,10].sum()==0


def test_prompt_does_not_reveal_gold():
    assert "ABSENT" in vlm.PROMPT
    assert "future mask" not in vlm.PROMPT.lower()
    assert vlm.QUERIES_PER_SCENE==2
    assert vlm.MAX_NEW_TOKENS <= 48
