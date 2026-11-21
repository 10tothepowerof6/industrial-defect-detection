"""Evaluate a calibrated artifact on the test rows of a manifest with the shared metrics.

  python scripts/evaluate.py --artifact-dir <artifact_dir> --dataset-root datasets \
      --manifest splits/seed42/custom_comparison_tile_crack.csv

Writes outputs/<protocol>/<model_id>/<category>/<run_id>/eval-<manifest name>/ with
metrics.json, predictions.csv, maps/<sha256(sample_id)>.npz, masks/*.png and visualizations/.
"""

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from industrial_defect_detection.data.manifests import (  # noqa: E402
    load_manifest_sample,
    read_manifest,
)
from industrial_defect_detection.evaluation.checks import EvaluationRecord  # noqa: E402
from industrial_defect_detection.evaluation.metrics import (  # noqa: E402
    evaluate_records,
    prediction_rows,
)
from industrial_defect_detection.inference.loader import (  # noqa: E402
    calibrated_predict,
    load_predictor,
)
from industrial_defect_detection.runs import (  # noqa: E402
    environment,
    file_sha256,
    git_state,
    resolve,
    write_json,
)
from industrial_defect_detection.visualization.overlay import panel  # noqa: E402


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Default: outputs/.../eval-<manifest>")
    parser.add_argument("--device")
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    predictor = load_predictor(args.artifact_dir, args.device)
    meta = predictor.metadata
    rows = read_manifest(args.manifest, args.dataset_root)
    test_rows = [r for r in rows if r["split"] == "test"]
    if {r["category"] for r in test_rows} != {predictor.category}:
        parser.error(f"Manifest category does not match the {predictor.category} artifact")
    output_dir = args.output_dir or resolve("outputs").joinpath(
        meta["protocol"], meta["model_id"], meta["category"], meta["run_id"],
        f"eval-{args.manifest.stem}",
    )
    if output_dir.exists():
        parser.error(f"Output already exists; choose a new directory: {output_dir}")

    first_image, _ = load_manifest_sample(args.dataset_root, test_rows[0])
    for _ in range(args.warmup):
        calibrated_predict(predictor, first_image)

    records, images, latencies = [], {}, []
    for i, row in enumerate(test_rows, 1):
        image, mask = load_manifest_sample(args.dataset_root, row)
        sync()
        start = time.perf_counter()
        prediction = calibrated_predict(predictor, image)
        sync()
        latencies.append(time.perf_counter() - start)
        records.append(EvaluationRecord.from_manifest_row(row, prediction, mask))
        images[row["sample_id"]] = image
        print(f"\r{i}/{len(test_rows)} test images", end="", flush=True)
    print()

    metrics = evaluate_records(records)
    metrics["manifest"] = {"path": args.manifest.as_posix(), "sha256": file_sha256(args.manifest)}
    metrics["timing"] = {
        "scope": "preprocessing + model + postprocessing to original size, batch 1",
        "device": str(predictor.device) if hasattr(predictor, "device") else None,
        "warmup_runs": args.warmup,
        "measured_images": len(latencies),
        "mean_ms": 1000 * float(np.mean(latencies)),
        "median_ms": 1000 * float(np.median(latencies)),
        "p95_ms": 1000 * float(np.percentile(latencies, 95)),
    }
    if torch.cuda.is_available():
        metrics["timing"]["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
    metrics["artifact_size_mb"] = sum(
        p.stat().st_size for p in args.artifact_dir.rglob("*") if p.is_file()
    ) / 2**20
    metrics["environment"] = environment()
    metrics["git"] = git_state()

    for sub in ("maps", "masks", "visualizations"):
        (output_dir / sub).mkdir(parents=True)
    table = prediction_rows(records)
    # One display range for the whole run, so heatmaps of different images are comparable.
    low = min(float(r.prediction.score_map.min()) for r in records)
    high = max(float(r.prediction.score_map.max()) for r in records)
    for record, row in zip(records, table, strict=True):
        key = hashlib.sha256(record.sample_id.encode()).hexdigest()
        prediction = record.prediction
        np.savez_compressed(output_dir / "maps" / f"{key}.npz", score_map=prediction.score_map)
        Image.fromarray(prediction.pred_mask.astype(np.uint8) * 255).save(
            output_dir / "masks" / f"{key}.png"
        )
        verdict = "DEFECT" if prediction.is_defective else "PASS"
        title = (
            f"{record.sample_id} [{record.group}] {verdict} "
            f"score={prediction.image_score:.4f} thr={prediction.image_threshold:.4f}"
        )
        picture = panel(images[record.sample_id], prediction.score_map, prediction.pred_mask,
                        low, high, record.gt_mask, title)
        name = record.sample_id.replace("/", "__").rsplit(".", 1)[0]
        Image.fromarray(picture).save(output_dir / "visualizations" / f"{name}.png")
        row["sample_key"] = key
        row["map_path"] = f"maps/{key}.npz"
        row["mask_path"] = f"masks/{key}.png"

    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    write_json(output_dir / "metrics.json", metrics)

    image, pixel, groups = metrics["image"], metrics["pixel"], metrics["groups"]

    def fmt(value):
        return "null" if value is None else f"{value:.4f}"

    print(f"Image AUROC {fmt(image['auroc'])} AP {fmt(image['average_precision'])} "
          f"F1 {fmt(image['f1'])} FPR {fmt(image['false_positive_rate'])}")
    print(f"Pixel AUROC {fmt(pixel['auroc'])} AP {fmt(pixel['average_precision'])} "
          f"Dice(defect) {fmt(pixel['dice_defect_images'])}")
    print(f"Recall known {fmt(groups['known'].get('recall'))} "
          f"held-out {fmt(groups['held_out'].get('recall'))} "
          f"good FPR {fmt(groups['good']['false_positive_rate'])}")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
