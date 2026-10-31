"""Answer parsing must recognize a leading choice with label text."""
from pathlib import Path
import importlib.util
p=Path(__file__).resolve().parents[1]/"experiments"/"mvbench_rescore.py"
spec=importlib.util.spec_from_file_location("mvbench_rescore",p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_leading_choice():
    assert m.answer_letter("C. Spinning something so it continues spinning.")=="C"
    assert m.answer_letter("A. Dropping something onto something.")=="A"
    assert m.answer_letter("B) Pouring water")=="B"
def test_explicit_format_and_invalid():
    assert m.answer_letter("ANSWER: B")=="B"
    assert m.answer_letter("Option C")=="C"
    assert m.answer_letter("Correct answer is A")=="A"
    assert m.answer_letter("Not sure") is None
    assert m.answer_letter("A. Something\nB. Something else")=="A"
