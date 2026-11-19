"""Model-independent prediction objects for adapters, evaluation, and the demo."""

from dataclasses import dataclass
from numbers import Real

import numpy as np

from industrial_defect_detection.data.mvtec import CATEGORIES

MODEL_IDS = {"patchcore", "efficientad", "unet_resnet18"}


def require_finite_number(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")


@dataclass(frozen=True)
class RawPrediction:
    """Scores before calibration. The score map must already use original image size."""

    image_score: float
    score_map: np.ndarray
    category: str
    model_id: str
    run_id: str

    def __post_init__(self) -> None:
        require_finite_number(self.image_score, "image_score")
        if self.category not in CATEGORIES:
            raise ValueError(f"Unknown product category: {self.category}")
        if self.model_id not in MODEL_IDS:
            raise ValueError(f"Unknown model ID: {self.model_id}")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be a nonempty string")
        if not isinstance(self.score_map, np.ndarray) or self.score_map.dtype != np.float32:
            raise ValueError("score_map must be a float32 NumPy array")
        if self.score_map.ndim != 2 or self.score_map.size == 0:
            raise ValueError("score_map must have nonempty shape [height, width]")
        if not np.isfinite(self.score_map).all():
            raise ValueError("score_map contains NaN or infinity")

    def with_thresholds(self, image_threshold: float, pixel_threshold: float) -> "Prediction":
        """Apply thresholds selected elsewhere using normal validation only."""
        return Prediction(
            image_score=self.image_score,
            score_map=self.score_map,
            category=self.category,
            model_id=self.model_id,
            run_id=self.run_id,
            image_threshold=image_threshold,
            pixel_threshold=pixel_threshold,
        )


@dataclass(frozen=True)
class Prediction(RawPrediction):
    """Calibrated scores. Decisions are derived to avoid mismatched labels and masks."""

    image_threshold: float
    pixel_threshold: float

    def __post_init__(self) -> None:
        super().__post_init__()
        require_finite_number(self.image_threshold, "image_threshold")
        require_finite_number(self.pixel_threshold, "pixel_threshold")

    @property
    def is_defective(self) -> bool:
        return bool(self.image_score > self.image_threshold)

    @property
    def pred_mask(self) -> np.ndarray:
        return self.score_map > self.pixel_threshold

    def summary(self) -> dict:
        return {
            "model_id": self.model_id,
            "category": self.category,
            "run_id": self.run_id,
            "image_score": float(self.image_score),
            "image_threshold": float(self.image_threshold),
            "pixel_threshold": float(self.pixel_threshold),
            "is_defective": self.is_defective,
            "height": self.score_map.shape[0],
            "width": self.score_map.shape[1],
            "defect_pixels": int(self.pred_mask.sum()),
        }
