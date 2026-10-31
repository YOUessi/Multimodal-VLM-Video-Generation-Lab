"""Group exact byte hashes and strict perceptual candidates before SFT."""
from pathlib import Path
import importlib.util,json,sys,collections
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
s=importlib.util.spec_from_file_location("perceptual_group",ROOT/"experiments/star_perceptual_grouped_sft.py")
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_connected_components_close_transitive_edges():
    r=[{"source_index":i,"content_sha256":str(i)} for i in range(5)]
    graph=m.linked_components(r,[(0,1),(1,2)])
    values=[set(x) for x in graph.values()]
    assert {0,1,2} in values
    assert {3} in values and {4} in values
    assert len(graph)==3

def test_exact_sha_merges_without_edges():
    rows=[{"source_index":0,"content_sha256":"X"},
          {"source_index":1,"content_sha256":"Y"},
          {"source_index":2,"content_sha256":"X"}]
    components=m.linked_components(rows,[])
    assert len(components)==2
    assert {0,2} in [set(x) for x in components.values()]

def test_fixed_remediation_plan():
    cfg=json.loads((ROOT/"configs/star_perceptual_grouped_v1.json").read_text())
    assert cfg["max_phash_distance"]==8
    assert cfg["max_dhash_distance"]==6
    assert cfg["split_sizes"]=={"train":120,"val":40,"test":40}
    assert cfg["output_dir"]=="outputs/star_perceptual_grouped_sft_v1"
    assert "Retrospective" in cfg["disclosure"]
