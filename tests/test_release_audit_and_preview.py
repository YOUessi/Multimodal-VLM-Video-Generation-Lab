"""Release reporting integrity tests without private weights or public video datasets."""
import hashlib,importlib.util,json
from pathlib import Path
P=Path(__file__).resolve().parents[1]/"experiments/release_audit_and_preview.py"
spec=importlib.util.spec_from_file_location("release_audit",P)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(m,"ROOT",tmp_path)
    rows=[]
    # 7 exact duplicate pairs entirely in train + other unique videos = 193
    for i in range(200):
        data=(f"dup{i//2}" if i<14 else f"unique{i}").encode()
        file=tmp_path/f"video_{i:04d}.mp4"
        file.write_bytes(data)
        split="train" if i<120 else "val" if i<160 else "test"
        rows.append({"source_index":i,"video":file.name,
                     "content_sha256":hashlib.sha256(data).hexdigest(),
                     "split":split})
    return {"rows":rows,"rows_sha256":hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()}

def test_grouped_release_integrity_valid(tmp_path,monkeypatch):
    manifest=fixture(tmp_path,monkeypatch)
    result=m.checked_manifest(manifest)
    assert result["unique_full_sha256_contents"]==193
    assert result["cross_split_sha_overlap"]==0
    assert result["duplicate_content_groups"]==7
    assert result["train_val_test"]=={"train":120,"val":40,"test":40}

def test_grouped_release_rejects_duplicate_train_test(tmp_path,monkeypatch):
    manifest=fixture(tmp_path,monkeypatch)
    # Same content under i=0 and i=1; move only the second across the boundary.
    manifest["rows"][1]["split"]="test"
    manifest["rows_sha256"]=hashlib.sha256(
        json.dumps(manifest["rows"],sort_keys=True).encode()).hexdigest()
    try:m.checked_manifest(manifest)
    except RuntimeError as e:assert "STAR_CONTENT_SPLIT_LEAK" in str(e)
    else:raise AssertionError("Release incorrectly accepted duplicate video crossing splits")

def test_video_html_escapes_untrusted_prompts():
    records=[{"id":"<bad>","prompt":'Scene "x" <script>','diffusion_steps':8,
              "output":"outputs/diffusion_grid/steps_8/rolling_ball.mp4",
              "elapsed_sec":12.99,"gpu_peak_gib":3.327}]
    rendered=m.video_cards(records)
    assert "&lt;script&gt;" in rendered
    assert "<script>" not in rendered
    assert "../diffusion_grid/steps_8/rolling_ball.mp4" in rendered
