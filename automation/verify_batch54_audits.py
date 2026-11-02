"""Read-only SHA and DEV-origin audit for the two immutable Batch54 reports."""
from __future__ import annotations
import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import dev_cluster_occlusion_certificate as clusters


def verify(root: Path, audit_dir: Path) -> dict:
    expected = {
        "batch48b_dev_cluster.json": root / "repo-batch48b-tested",
        "batch53b_dev_cluster.json": root / "repo-batch53b-tested",
    }
    result = {}
    for name, frozen in expected.items():
        path = audit_dir / name
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if path.with_suffix(path.suffix + ".sha256").read_text().strip() != digest:
            raise RuntimeError("BATCH54_PIN_MISMATCH:" + name)
        source, selected, config, provenance = clusters.read_pinned_dev(frozen)
        expected_bytes = clusters.canonical_json(
            clusters.calculate(source, selected, config, provenance)
        )
        if raw != expected_bytes:
            raise RuntimeError("BATCH54_DEV_INPUT_OR_REPORT_DRIFT:" + name)
        result[name] = digest
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.work_root, args.audit_dir)
    for key, value in result.items():
        print("BATCH54_DEV_ONLY_AUDIT_VERIFIED", key, value)


if __name__ == "__main__":
    main()
