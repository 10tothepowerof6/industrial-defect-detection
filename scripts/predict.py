"""Export raw PatchCore predictions on one image or validation, without calibration."""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from industrial_defect_detection.data.manifests import (  # noqa: E402
    load_manifest_sample,
    read_manifest,
)
from industrial_defect_detection.evaluation.checks import EvaluationRecord  # noqa: E402
from industrial_defect_detection.models.anomaly.patchcore import (  # noqa: E402
    PatchcorePredictor,
    sha256,
)


def raw_panel(image, score_map, gt_mask, low, high):
    scaled = np.clip((score_map - low) / max(high - low, 1e-12), 0, 1)
    colored = colormaps["turbo"](scaled)[..., :3] * 255
    overlay = (0.55 * image + 0.45 * colored).astype(np.uint8)
    truth = image.copy()
    region = gt_mask.astype(bool)
    truth[region] = (truth[region] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    canvas = Image.new("RGB", (768, 278), "black")
    for i, (value, title) in enumerate(
        zip(
            [image, truth, overlay],
            ["image", "ground truth", "raw heatmap (no decision)"],
            strict=True,
        )
    ):
        canvas.paste(Image.fromarray(value).resize((256, 256)), (i * 256, 22))
        ImageDraw.Draw(canvas).text((i * 256 + 3, 4), title, fill="white")
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output already exists; choose a new output directory")
    predictor = PatchcorePredictor(args.artifact_dir, args.device)
    if args.manifest:
        rows = read_manifest(args.manifest, args.dataset_root)
        meta = predictor.metadata
        if sha256(args.manifest) != meta["training_manifest"]["sha256"]:
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
    records, previews, ranges = [], [], []
    for i, (sample_id, row) in enumerate(items):
        if row is None:
            with Image.open(args.image) as file:
                image = np.array(file.convert("RGB"))
            mask = None
        else:
            image, mask = load_manifest_sample(args.dataset_root, row)
        started = time.perf_counter()
        raw = predictor.predict(image)
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
        ranges.append((float(raw.score_map.min()), float(raw.score_map.max())))
        # First validation sample per defect type, including good; no held-out test browsing.
        group = row["defect_type"] if row else "external"
        if not any(p[0] == group for p in previews):
            previews.append((group, image, raw.score_map, mask, key))
        print(f"Raw prediction {i + 1}/{len(items)}: {sample_id}", flush=True)
    low, high = min(v[0] for v in ranges), max(v[1] for v in ranges)
    for group, image, score_map, mask, key in previews:
        # No GT is available for external images, so omit a misleading zero-mask panel.
        if mask is None:
            scaled = np.clip((score_map - low) / max(high - low, 1e-12), 0, 1)
            preview = Image.fromarray((colormaps["turbo"](scaled)[..., :3] * 255).astype(np.uint8))
        else:
            preview = raw_panel(image, score_map, mask, low, high)
        preview.save(args.output_dir / "visualizations" / f"{group}-{key[:12]}.png")
    with (args.output_dir / "raw-predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "kind": "raw_validation" if args.manifest else "raw_external",
        "calibrated": False,
        "decision_available": False,
        "sample_count": len(records),
        "artifact_dir": str(args.artifact_dir),
        "checkpoint_sha256": predictor.metadata["checkpoint_sha256"],
        "display_range": [low, high],
        "display_only": True,
        "timing_note": "Informal raw predict timing; not a warmed-up latency benchmark",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved raw maps and previews: {args.output_dir}")


if __name__ == "__main__":
    main()
