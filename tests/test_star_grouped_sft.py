"""Content-hash grouping must prevent duplicate video bytes crossing splits."""
from pathlib import Path
import importlib.util,json,sys,collections
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
spec=importlib.util.spec_from_file_location("grouped",ROOT/"experiments/star_content_grouped_sft.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_grouped_split_zero_overlap_fixed_counts():
    # 200 source rows, 193 content groups: 7 exact duplicates.
    groups={}
    cursor=0
    for i in range(7):
        groups[f"duplicated_{i}"]=[cursor,cursor+1]
        cursor+=2
    while cursor<200:
        groups[f"unique_{cursor}"]=[cursor]
        cursor+=1
    assert len(groups)==193
    caps={"train":120,"val":40,"test":40}
    result=m.assign_content_groups(groups,caps,20261031)
    assert collections.Counter(result.values())==collections.Counter(caps)
    assert all(len({result[i] for i in group})==1 for group in groups.values())
    assert result==m.assign_content_groups(groups,caps,20261031)

def test_reject_oversized_content_group():
    groups={"too_big":[0,1,2],"other":[3]}
    try:m.assign_content_groups(groups,{"train":2,"val":1,"test":1},1)
    except RuntimeError:pass
    else:raise AssertionError("Invalid group split accepted")

def test_pinned_config_and_isolated_outputs():
    cfg=json.loads((ROOT/"configs/star_grouped_sft_v1.json").read_text())
    assert cfg["output_dir"]=="outputs/star_grouped_sft_v1"
    assert cfg["split_sizes"]=={"train":120,"val":40,"test":40}
    assert len(set(cfg["training_seeds"]))==2
    assert len(cfg["source_annotations_sha256"])==64
    assert "SHA256" in cfg["split_policy"]
