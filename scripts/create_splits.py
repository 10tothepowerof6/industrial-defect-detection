"""Write audited, deterministic manifests for the selected held-out defects."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from industrial_defect_detection.data.audit import audit_dataset  # noqa: E402
from industrial_defect_detection.data.manifests import build_manifest, manifest_files  # noqa: E402
from industrial_defect_detection.data.mvtec import CATEGORIES, discover_samples  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("splits/seed42"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-held-out", required=True)
    parser.add_argument("--metal-nut-held-out", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"Output already exists; choose a new directory: {args.output_dir}")
    audit = audit_dataset(args.dataset_root)
    if not audit["passed"]:
        parser.error("Dataset audit failed; run audit_dataset.py and resolve its report first")
    samples = discover_samples(args.dataset_root)
    held_out = {"tile": args.tile_held_out, "metal_nut": args.metal_nut_held_out}
    files = {}
    for category in CATEGORIES:
        for protocol in ("official_ad", "custom_comparison"):
            defect = held_out[category] if protocol == "custom_comparison" else ""
            rows = build_manifest(samples, category, protocol, args.seed, defect)
            csv_content, metadata = manifest_files(rows, args.seed, audit["inventory"])
            name = f"{protocol}_{category}_{defect or 'baseline'}"
            files[f"{name}.csv"] = csv_content
            files[f"{name}.json"] = metadata
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir.parent) as temporary:
        staging = Path(temporary) / "manifests"
        staging.mkdir()
        for name, content in files.items():
            (staging / name).write_bytes(content)
        os.rename(staging, args.output_dir)
    print(f"Created {len(files) // 2} manifests in {args.output_dir}")
    print("Pending Hung's review before model development uses these splits.")


if __name__ == "__main__":
    main()
