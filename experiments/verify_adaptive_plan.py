"""Validate precomputed label-blind STAR adaptive frame-selection manifest.

No video is decoded and no test labels are used to select frames. The prior
40-video plan is immutable; this verifies it before starting GPU inference.
"""
import argparse, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def validate(config):
    import sys
    sys.path.insert(0,str(ROOT/"experiments"))
    from adaptive_temporal_sampling import candidate_positions
    cfg=json.loads((ROOT/config).read_text(encoding="utf8"))
    plan_path=ROOT/cfg["output_dir"]/"sampling_plan.json"
    plan=json.loads(plan_path.read_text(encoding="utf8"))
    star=json.loads((ROOT/cfg["star_manifest"]).read_text(encoding="utf8"))
    assert plan["selection_is_label_blind"] is True
    assert plan["star_manifest_sha256"]==star["sha256"]
    test_rows={r["source_index"]:r for r in star["rows"] if r["split"]=="test"}
    assert len(test_rows)==40 and len(plan["records"])==40
    digest=hashlib.sha256(json.dumps(plan["records"],sort_keys=True).encode()).hexdigest()
    if digest!=plan["plan_sha256"]:raise RuntimeError("ADAPTIVE_PLAN_SHA_DRIFT")
    expected=set(test_rows)
    received=set()
    for entry in plan["records"]:
        sid=entry["id"]
        if sid not in expected or sid in received:raise RuntimeError("TEST_VIDEO_ID_OVERLAP_OR_DUPLICATE")
        received.add(sid)
        row=test_rows[sid]
        if entry["video"]!=row["video"] or entry["gold"]!=row["answer"]:
            raise RuntimeError("TEST_MANIFEST_DRIFT")
        decoded=row["decoded_frames"]
        (lo,hi),probes=candidate_positions(decoded,cfg["probe_count"])
        for n in cfg["frame_counts"]:
            selected=entry["frames"][str(n)]["frame_indices"]
            if len(selected)!=n or sorted(set(selected))!=selected:
                raise RuntimeError("INVALID_ADAPTIVE_POSITIONS")
            if selected[0]!=lo or selected[-1]!=hi:
                raise RuntimeError("ENDPOINT_VIOLATION")
            if any(j not in probes for j in selected):
                raise RuntimeError("CHOSEN_FRAME_NOT_IN_FROZEN_PROBE_SET")
    assert received==expected
    print("ADAPTIVE_MANIFEST_VERIFIED",len(received),
          "frozen_star_sha",star["sha256"],"sampling_sha",digest,flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/adaptive_sampling_v2.json")
    args=p.parse_args()
    validate(args.config)

if __name__=="__main__":main()
