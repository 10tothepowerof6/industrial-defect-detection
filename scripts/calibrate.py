"""Choose thresholds from the normal validation images of a manifest and save them.

Only rows with split == val and is_anomalous == 0 are read. Test rows are never loaded.

  python scripts/calibrate.py --artifact-dir <artifact_dir> --dataset-root datasets \
      --manifest splits/seed42/custom_comparison_tile_crack.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from industrial_defect_detection.data.manifests import (  # noqa: E402
    load_manifest_sample,
    read_manifest,
)
from industrial_defect_detection.evaluation.calibration import (  # noqa: E402
    calibrate_thresholds,
    resolve_quantiles,
)
from industrial_defect_detection.evaluation.checks import check_calibration_manifest  # noqa: E402
from industrial_defect_detection.inference.loader import load_predictor  # noqa: E402
from industrial_defect_detection.runs import file_sha256, read_json, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    # Priority: this CLI value, then calibration.* in the artifact's config, then the default.
    parser.add_argument("--image-quantile", type=float, help="Default: artifact config, else 0.95")
    parser.add_argument("--pixel-quantile", type=float, help="Default: artifact config, else 0.995")
    parser.add_argument("--device")
    parser.add_argument("--force", action="store_true", help="Replace an existing calibration")
    args = parser.parse_args()

    metadata_path = args.artifact_dir / "metadata.json"
    metadata = read_json(metadata_path)
    if metadata.get("status") == "calibrated" and not args.force:
        parser.error("Artifact is already calibrated; pass --force to recalibrate")
    try:
        check_calibration_manifest(metadata, args.manifest)
    except ValueError as error:
        parser.error(str(error))
    predictor = load_predictor(args.artifact_dir, args.device)
    rows = read_manifest(args.manifest, args.dataset_root)
    normal_val = [r for r in rows if r["split"] == "val" and r["is_anomalous"] == 0]
    if not normal_val:
        parser.error("Manifest has no normal validation images")
    if {r["category"] for r in normal_val} != {predictor.category}:
        parser.error(f"Manifest category does not match the {predictor.category} artifact")

    predictions = []
    for i, row in enumerate(normal_val, 1):
        image, _ = load_manifest_sample(args.dataset_root, row)
        predictions.append(predictor.predict(image))
        print(f"\r{i}/{len(normal_val)} normal val images", end="", flush=True)
    print()
    quantiles = resolve_quantiles(metadata, args.image_quantile, args.pixel_quantile)
    calibration = calibrate_thresholds(
        predictions, quantiles["image_quantile"], quantiles["pixel_quantile"]
    )
    calibration["quantile_sources"] = {
        "image": quantiles["image_quantile_source"],
        "pixel": quantiles["pixel_quantile_source"],
    }
    calibration["validation_manifest"] = {
        "path": args.manifest.as_posix(),
        "sha256": file_sha256(args.manifest),
    }
    metadata["status"] = "calibrated"
    metadata["calibration"] = calibration
    write_json(metadata_path, metadata)
    print(
        f"quantiles image={quantiles['image_quantile']} ({quantiles['image_quantile_source']}) "
        f"pixel={quantiles['pixel_quantile']} ({quantiles['pixel_quantile_source']})\n"
        f"image_threshold={calibration['image_threshold']:.6f} "
        f"pixel_threshold={calibration['pixel_threshold']:.6f} "
        f"from {calibration['normal_val_images']} normal val images -> {metadata_path}"
    )


if __name__ == "__main__":
    main()
