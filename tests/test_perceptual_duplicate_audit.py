"""No-label pHash+dHash audit: do not modify a public heldout dataset."""
import importlib.util,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/"experiments/perceptual_duplicate_audit.py"
spec=importlib.util.spec_from_file_location("near_dup",P)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_hamming_distance():
    assert m.hamming(0,0)==0
    assert m.hamming(0,15)==4
    assert m.hamming(0b10101,0b10001)==1
def test_exact_sha_is_handled_by_grouped_split_not_phash():
    a={"content_sha256":"same","p":[1,1,1],"d":[1,1,1]}
    b={"content_sha256":"same","p":[1,1,1],"d":[1,1,1]}
    assert m.suspicious(a,b,{"phash":8,"dhash":6}) is None
def test_strict_three_frame_gate_and_candidate_distances():
    a={"content_sha256":"A","p":[0,0,0],"d":[0,0,0]}
    b={"content_sha256":"B","p":[1,1,1],"d":[1,1,1]}
    c={"content_sha256":"C","p":[1,1,511],"d":[1,1,511]}
    assert m.suspicious(a,b,{"phash":8,"dhash":6}) == {
        "phash_distances":[1,1,1],"dhash_distances":[1,1,1]}
    assert m.suspicious(a,c,{"phash":8,"dhash":6}) is None
def test_near_duplicate_config_is_non_destructive():
    cfg=json.loads((ROOT/"configs/perceptual_near_dup_v1.json").read_text())
    assert cfg["manifest"]=="outputs/star_grouped_sft_v1/manifest.json"
    assert cfg["unmodified_source_data"] and cfg["no_answer_labels_used"]
    assert cfg["max_phash_distance"]<=10 and cfg["max_dhash_distance"]<=10
