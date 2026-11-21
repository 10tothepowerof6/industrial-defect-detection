"""Choose image and pixel thresholds from normal validation images only."""

from collections.abc import Sequence

import numpy as np

from industrial_defect_detection.inference.contracts import Prediction, RawPrediction

IMAGE_QUANTILE = 0.95
PIXEL_QUANTILE = 0.995
QUANTILE_METHOD = "linear"


def calibrate_thresholds(
    normal_val: Sequence[RawPrediction],
    image_quantile: float = IMAGE_QUANTILE,
    pixel_quantile: float = PIXEL_QUANTILE,
) -> dict:
    """Return thresholds and the record needed to reproduce them.

    Every prediction must come from a good validation image. Labels of test images
    are never seen here, so the thresholds cannot leak test information.
    """
    if not normal_val:
        raise ValueError("Calibration needs at least one normal validation prediction")
    for name, q in (("image_quantile", image_quantile), ("pixel_quantile", pixel_quantile)):
        if not 0 < q < 1:
            raise ValueError(f"{name} must be strictly between 0 and 1")
    identities = {(p.category, p.model_id, p.run_id) for p in normal_val}
    if len(identities) != 1:
        raise ValueError(f"Calibration mixes several category/model/run identities: {identities}")
    image_scores = np.array([p.image_score for p in normal_val], dtype=np.float64)
    pixel_scores = np.concatenate([p.score_map.ravel() for p in normal_val])
    image_threshold = float(np.quantile(image_scores, image_quantile, method=QUANTILE_METHOD))
    pixel_threshold = float(np.quantile(pixel_scores, pixel_quantile, method=QUANTILE_METHOD))
    category, model_id, run_id = identities.pop()
    return {
        "status": "calibrated",
        "category": category,
        "model_id": model_id,
        "run_id": run_id,
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "method": {
            "source": "normal validation images",
            "image": f"quantile {image_quantile} of image scores",
            "pixel": f"quantile {pixel_quantile} of all pixel scores at original image size",
            "numpy_quantile_method": QUANTILE_METHOD,
            "decision_rule": "score > threshold",
        },
        "normal_val_images": len(normal_val),
        "normal_val_pixels": int(pixel_scores.size),
    }


def apply_calibration(raw: RawPrediction, calibration: dict) -> Prediction:
    """Attach saved thresholds after checking they belong to the same model run."""
    if calibration.get("status") != "calibrated":
        raise ValueError("Calibration record is missing or not calibrated")
    expected = (calibration["category"], calibration["model_id"], calibration["run_id"])
    if (raw.category, raw.model_id, raw.run_id) != expected:
        raise ValueError(
            f"Calibration belongs to {expected}, prediction to "
            f"{(raw.category, raw.model_id, raw.run_id)}"
        )
    return raw.with_thresholds(calibration["image_threshold"], calibration["pixel_threshold"])
