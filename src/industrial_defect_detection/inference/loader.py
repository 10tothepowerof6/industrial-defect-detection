"""Load any calibrated or raw artifact through one entry point keyed by model ID.

Each model package registers a predictor class with:
  - __init__(artifact_dir, device=None)
  - attributes model_id, category, run_id, metadata
  - predict(image: uint8 RGB [H,W,3]) -> RawPrediction at the original image size
"""

from pathlib import Path

from industrial_defect_detection.runs import read_json

from .contracts import Prediction


def _unet(artifact_dir: Path, device: str | None):
    from industrial_defect_detection.models.supervised.adapter import UNetPredictor

    return UNetPredictor(artifact_dir, device)


def _patchcore(artifact_dir: Path, device: str | None):
    from industrial_defect_detection.models.anomaly.patchcore import PatchcorePredictor

    return PatchcorePredictor(artifact_dir, device)


# EfficientAD will be registered in T04.
PREDICTORS = {"unet_resnet18": _unet, "patchcore": _patchcore}


def load_predictor(artifact_dir: Path, device: str | None = None):
    metadata_path = Path(artifact_dir) / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"No metadata.json in {artifact_dir}; is this an artifact dir?")
    model_id = read_json(metadata_path).get("model_id")
    if model_id not in PREDICTORS:
        raise ValueError(
            f"No predictor registered for model_id={model_id!r}; known: {sorted(PREDICTORS)}"
        )
    return PREDICTORS[model_id](Path(artifact_dir), device)


def calibrated_predict(predictor, image) -> Prediction:
    """Predict with the thresholds stored in the artifact; refuse uncalibrated artifacts."""
    calibration = predictor.metadata.get("calibration")
    if predictor.metadata.get("status") != "calibrated" or not calibration:
        raise ValueError(
            f"Artifact {predictor.artifact_dir} is not calibrated; run scripts/calibrate.py first"
        )
    from industrial_defect_detection.evaluation.calibration import apply_calibration

    return apply_calibration(predictor.predict(image), calibration)
