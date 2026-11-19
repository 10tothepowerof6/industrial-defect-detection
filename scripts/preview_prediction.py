"""Produce a clearly labeled synthetic prediction for H01 integration checks."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from industrial_defect_detection.inference.contracts import RawPrediction  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/day19/prediction_example"))
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"Output already exists; choose a new directory: {args.output_dir}")
    score_map = np.full((32, 48), 0.1, dtype=np.float32)
    score_map[8:16, 12:24] = 0.9
    raw = RawPrediction(0.9, score_map, "tile", "patchcore", "synthetic-contract-example")
    prediction = raw.with_thresholds(image_threshold=0.5, pixel_threshold=0.5)
    summary = prediction.summary()
    summary["synthetic"] = True
    summary["purpose"] = "Interface example only; not a trained-model prediction or metric"
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir / "scores.npz", score_map=prediction.score_map)
    Image.fromarray(prediction.pred_mask.astype(np.uint8) * 255).save(args.output_dir / "mask.png")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
