"""Post-hoc descriptive error slices for frozen paired object-motion VLM experiments.

Does NOT change prompts, tracker, model choice or videos. Ground truth is read
only here after both model predictions are complete. Group scores are
descriptive: small per-color and per-shape strata are not independent claims.
"""
from __future__ import annotations
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
from object_motion_counterfactual_data import DIRECTIONS,LABELS
from object_motion_counterfactual_eval import CONDITIONS,direction_from_letter

def calculate(cfg):
    folder=ROOT/cfg["output_dir"]
    manifest=json.loads((ROOT/cfg["dataset_dir"]/"manifest.json").read_text())
    source={r["id"]:r for r in manifest["rows"] if r["split"]=="test"}
    if len(source)!=240:raise RuntimeError("EXPECTED_240_TEST_CLIPS")
    results={}
    for modelname in cfg["models"]:
        data=json.loads((folder/(modelname+".json")).read_text())
        if data["dataset_manifest_sha"]!=manifest["rows_sha256"]:
            raise RuntimeError("MODEL_MANIFEST_MISMATCH")
        rows=[]
        for item in data["predictions"]:
            ref=source.get(item["id"])
            if ref is None:raise RuntimeError("UNKNOWN_TEST_ID")
            guess=direction_from_letter(ref,item["pred_letter"])
            rows.append({"model":modelname,"id":item["id"],"pair_id":ref["pair_id"],
                "mode":item["mode"],"target":ref["target_name"],
                "color":ref["target_name"].split()[0],
                "shape":ref["target_name"].split()[1],
                "reversed":bool(ref["reversed"]),
                "direction":ref["gold_direction"],
                "prediction_direction":guess,
                "correct":guess==ref["gold_direction"]})
        if len(rows)!=240*len(CONDITIONS):
            raise RuntimeError("ERROR_SLICES_INCOMPLETE")
        bymode={}
        for mode in CONDITIONS:
            subset=[x for x in rows if x["mode"]==mode]
            all_modes=Counter(x["direction"] for x in subset)
            classes={}
            for direction in DIRECTIONS:
                group=[x for x in subset if x["direction"]==direction]
                if len(group)!=48:raise RuntimeError("DIRECTION_BALANCE_BROKEN")
                outcomes=Counter(x["prediction_direction"] or "invalid" for x in group)
                classes[direction]={"correct":sum(x["correct"] for x in group),
                    "total":len(group),"model_predicted_direction_counts":dict(outcomes)}
            shapes={}
            for shape in ("sphere","cube","triangle"):
                group=[x for x in subset if x["shape"]==shape]
                shapes[shape]={"correct":sum(x["correct"] for x in group),"total":len(group)}
            colors={}
            for color in ("red","blue","green","purple","cyan","orange"):
                group=[x for x in subset if x["color"]==color]
                colors[color]={"correct":sum(x["correct"] for x in group),"total":len(group)}
            direction_order={}
            for reverse in (False,True):
                group=[x for x in subset if x["reversed"] is reverse]
                direction_order["reverse" if reverse else "forward"]={
                    "correct":sum(x["correct"] for x in group),"total":len(group)}
            bymode[mode]={"total":240,"correct":sum(x["correct"] for x in subset),
                          "direction_class":classes,"target_shape":shapes,
                          "target_color":colors,"forward_reverse":direction_order}
        results[modelname]=bymode
    doc={"test_source_sha":manifest["rows_sha256"],"data":"Procedural 2D objects, fixed camera",
         "statistical_status":"Exploratory descriptive slices, NOT independent claims",
         "models":results}
    (folder/"ERROR_SLICES.json").write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n")
    lines=["# Counterfactual motion error analysis","",
       "All 240 frozen test clips. Direction classes are balanced: 48 clips per class.",
       "These are descriptive slices inspected after the main test; no hyperparameter retuning or inferential claims on small subgroups.",
       "","| Model | Condition | Stationary | Up-right | Up-left | Down-right | Down-left |",
       "|---|---|---|---|---|---|---|"]
    for name,modes in results.items():
        for mode,group in modes.items():
            values=[str(group["direction_class"][d]["correct"])+"/48" for d in DIRECTIONS]
            lines.append("| "+name+" | "+mode+" | "+" | ".join(values)+" |")
    lines+=["","## Interpretation",
            "- Any subgroup difference may be due to random synthetic scene factors and small per-direction counts.",
            "- Source videos were synthesized with distinct target/distractor colors and a fixed camera.",
            "- A color-centroid CV tracker can trivially read these simple motion signals, and should always be reported as a domain-specific baseline.",
            "- External CLEVRER/natural-motion generalization remains unverified; do not advertise test-set-tuned gains."]
    (folder/"ERROR_SLICES.md").write_text("\n".join(lines)+"\n")
    print("MOTION_ERROR_SLICES_RECORDED",folder/"ERROR_SLICES.md",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/object_motion_v1.json")
    args=p.parse_args()
    cfg=json.loads((ROOT/args.config).read_text())
    calculate(cfg)
