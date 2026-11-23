"""Choose the U-Net image-score border on validation images only (never test).

Pass one --artifact-dir/--manifest pair per category; the same border is chosen for all:
  python scripts/select_score_border.py --dataset-root datasets \
      --artifact-dir <unet tile v1> --manifest splits/seed42/custom_comparison_tile_crack.csv \
      --artifact-dir <unet metal_nut v1> \
      --manifest splits/seed42/custom_comparison_metal_nut_scratch.csv \
      --output outputs/score-border-selection.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from industrial_defect_detection.data.manifests import (  # noqa: E402
    load_manifest_sample,
)
from industrial_defect_detection.evaluation.checks import load_development_manifest  # noqa: E402
from industrial_defect_detection.inference.loader import load_predictor  # noqa: E402
from industrial_defect_detection.models.supervised.scoring import (  # noqa: E402
    BORDER_CANDIDATES,
    PLATEAU_TOLERANCE,
    SELECTION_RULE,
    border_table,
    choose_border,
    eligible_borders,
)
from industrial_defect_detection.runs import file_sha256, git_state, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--artifact-dir", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--candidates", type=int, nargs="+", default=list(BORDER_CANDIDATES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    if len(args.artifact_dir) != len(args.manifest):
        parser.error("Give one --manifest per --artifact-dir")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if 0 not in args.candidates:
        parser.error("Candidates must include 0 (the v1 rule) as the baseline")

    tables, sources = {}, {}
    for artifact_dir, manifest in zip(args.artifact_dir, args.manifest, strict=True):
        predictor = load_predictor(artifact_dir, args.device)
        meta = predictor.metadata
        if meta["model_id"] != "unet_resnet18":
            parser.error(f"{artifact_dir} is not a U-Net artifact")
        development = load_development_manifest(meta, args.dataset_root)
        if file_sha256(manifest) != meta["training_manifest"]["sha256"]:
            parser.error(f"{manifest} is not the training manifest of {artifact_dir}")
        val = [r for r in development if r["split"] == "val"]
        good_maps, defect_maps = [], []
        for i, row in enumerate(val, 1):
            image, _ = load_manifest_sample(args.dataset_root, row)
            score_map = predictor.predict(image).score_map
            (defect_maps if row["is_anomalous"] else good_maps).append(score_map)
            print(f"\r{meta['category']}: {i}/{len(val)} val images", end="", flush=True)
        print()
        table = border_table(good_maps, defect_maps, tuple(args.candidates))
        tables[meta["category"]] = table
        sources[meta["category"]] = {
            "artifact_run_id": meta["run_id"],
            "manifest": manifest.as_posix(),
            "manifest_sha256": file_sha256(manifest),
            "val_good_images": len(good_maps),
            "val_defect_images": len(defect_maps),
            "eligible": eligible_borders(table),
        }
        print(f"{'border':>6} {'good q95':>9} {'good median':>11} {'defect min':>10} {'AUROC':>7}")
        for row in table:
            print(
                f"{row['border']:>6} {row['good_q95']:>9.4f} {row['good_median']:>11.4f} "
                f"{row['defect_min']:>10.4f} {row['auroc']:>7.4f}"
            )
        print(f"eligible: {sources[meta['category']]['eligible']}")

    chosen = choose_border(tables)
    write_json(
        args.output,
        {
            "rule": SELECTION_RULE,
            "tolerance": PLATEAU_TOLERANCE,
            "candidates": args.candidates,
            "data": "validation split of each training manifest; no test image was read",
            "chosen_border_px": chosen,
            "categories": {c: {**sources[c], "table": tables[c]} for c in tables},
            "git": git_state(),
        },
    )
    print(f"Chosen border: {chosen} px -> {args.output}")


if __name__ == "__main__":
    main()
