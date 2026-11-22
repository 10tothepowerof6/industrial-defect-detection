"""Predict with the shared loader; opt into calibrated decisions with --calibrated."""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from industrial_defect_detection.data.manifests import (  # noqa: E402
    load_manifest_sample,
    read_manifest,
)
from industrial_defect_detection.evaluation.checks import EvaluationRecord  # noqa: E402
from industrial_defect_detection.inference.loader import calibrated_predict, load_predictor  # noqa: E402
from industrial_defect_detection.runs import file_sha256  # noqa: E402
from industrial_defect_detection.visualization.overlay import panel  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--calibrated", action="store_true", help="Use saved thresholds for decisions/masks"
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output already exists; choose a new output directory")
    predictor = load_predictor(args.artifact_dir, args.device)
    if args.calibrated and predictor.metadata.get("status") != "calibrated":
        parser.error("Artifact is not calibrated; run scripts/calibrate.py first")
    if args.manifest:
        rows = read_manifest(args.manifest, args.dataset_root)
        meta = predictor.metadata
        if file_sha256(args.manifest) != meta["training_manifest"]["sha256"]:
            parser.error("Validation manifest differs from the artifact's recorded manifest")
        rows = [row for row in rows if row["split"] == "val"]
        if not rows:
            parser.error("Manifest has no validation images")
        items = [(row["sample_id"], row) for row in rows]
    else:
        items = [(args.image.name, None)]
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "maps").mkdir()
    (args.output_dir / "visualizations").mkdir()
    if args.calibrated:
        (args.output_dir / "masks").mkdir()
    records, previews, ranges = [], [], []
    for i, (sample_id, row) in enumerate(items):
        if row is None:
            with Image.open(args.image) as file:
                image = np.array(file.convert("RGB"))
            mask = None
        else:
            image, mask = load_manifest_sample(args.dataset_root, row)
        started = time.perf_counter()
        raw = calibrated_predict(predictor, image) if args.calibrated else predictor.predict(image)
        elapsed = time.perf_counter() - started
        if row is not None:
            EvaluationRecord.from_manifest_row(row, raw, mask)
        key = hashlib.sha256(sample_id.encode()).hexdigest()
        np.savez_compressed(args.output_dir / "maps" / f"{key}.npz", score_map=raw.score_map)
        records.append(
            {
                "sample_id": sample_id,
                "category": raw.category,
                "model_id": raw.model_id,
                "run_id": raw.run_id,
                "split": row["split"] if row else "external",
                "defect_type": row["defect_type"] if row else "unknown",
                "image_score": raw.image_score,
                "height": image.shape[0],
                "width": image.shape[1],
                "map_path": f"maps/{key}.npz",
                "raw_predict_seconds": elapsed,
            }
        )
        if args.calibrated:
            Image.fromarray(raw.pred_mask.astype(np.uint8) * 255).save(
                args.output_dir / "masks" / f"{key}.png"
            )
            records[-1].update(
                image_threshold=raw.image_threshold,
                pixel_threshold=raw.pixel_threshold,
                is_defective=int(raw.is_defective),
                mask_path=f"masks/{key}.png",
            )
        ranges.append((float(raw.score_map.min()), float(raw.score_map.max())))
        # First validation sample per defect type, including good; no held-out test browsing.
        group = row["defect_type"] if row else "external"
        if not any(p[0] == group for p in previews):
            previews.append((group, image, raw, mask, key))
        print(f"Prediction {i + 1}/{len(items)}: {sample_id}", flush=True)
    low, high = min(v[0] for v in ranges), max(v[1] for v in ranges)
    for group, image, prediction, mask, key in previews:
        title = (
            ("DEFECT" if prediction.is_defective else "PASS")
            if args.calibrated
            else "Raw scores; no decision"
        )
        preview = panel(
            image,
            prediction.score_map,
            prediction.pred_mask if args.calibrated else None,
            low,
            high,
            mask,
            title,
        )
        Image.fromarray(preview).save(
            args.output_dir / "visualizations" / f"{group}-{key[:12]}.png"
        )
    with (args.output_dir / ("predictions.csv" if args.calibrated else "raw-predictions.csv")).open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "kind": ("calibrated" if args.calibrated else "raw")
        + ("_validation" if args.manifest else "_external"),
        "calibrated": args.calibrated,
        "decision_available": args.calibrated,
        "sample_count": len(records),
        "artifact_dir": str(args.artifact_dir),
        "checkpoint_sha256": file_sha256(args.artifact_dir / predictor.metadata["checkpoint"]),
        "display_range": [low, high],
        "display_only": True,
        "timing_note": "Informal raw predict timing; not a warmed-up latency benchmark",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved predictions, maps and previews: {args.output_dir}")


if __name__ == "__main__":
    main()
