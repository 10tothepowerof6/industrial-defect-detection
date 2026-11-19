"""Run from the repository root with its local Python environment."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from industrial_defect_detection.data.audit import audit_dataset, save_preview  # noqa: E402
from industrial_defect_detection.data.mvtec import discover_samples  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MVTec tile and metal_nut")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/day19/audit"))
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"Output already exists; choose a new directory: {args.output_dir}")
    report = audit_dataset(args.dataset_root)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if report["passed"]:
        samples = discover_samples(args.dataset_root)
        for category, defect in (("tile", "glue_strip"), ("metal_nut", "bent")):
            for group in ("good", defect):
                sample = next(
                    s for s in samples if s.category == category and s.defect_type == group
                )
                save_preview(args.dataset_root, sample, args.output_dir / f"{category}_{group}.png")
    print(json.dumps({k: v for k, v in report.items() if k != "inventory"}, indent=2))
    if not report["passed"]:
        raise SystemExit(f"Audit failed. Inspect {args.output_dir / 'audit.json'}")


if __name__ == "__main__":
    main()
