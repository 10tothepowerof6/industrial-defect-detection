"""Validate predictions and ground truth before any metric is computed.

Every model adapter (PatchCore, EfficientAD, U-Net) hands the evaluator the same
objects, so shape, label, and identity problems are caught here once instead of
being rediscovered inside each metric.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from industrial_defect_detection.data.manifests import read_manifest
from industrial_defect_detection.inference.contracts import Prediction, RawPrediction
from industrial_defect_detection.runs import file_sha256, resolve


def resize_score_map(score_map: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinearly resize a model-resolution map to the original image size [H, W]."""
    if not isinstance(score_map, np.ndarray) or score_map.ndim != 2 or score_map.size == 0:
        raise ValueError("score_map must be a nonempty 2-D NumPy array")
    if height <= 0 or width <= 0:
        raise ValueError("Target height and width must be positive")
    source = np.ascontiguousarray(score_map, dtype=np.float32)
    if source.shape == (height, width):
        return source.copy()
    # A 2-D float32 array opens as Pillow mode "F", so values are not quantized to 8 bits.
    resized = Image.fromarray(source).resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def check_ground_truth(mask: np.ndarray, is_anomalous: int, sample_id: str = "") -> np.ndarray:
    """Return the ground-truth mask as bool after checking it matches the image label."""
    where = f" for {sample_id}" if sample_id else ""
    if isinstance(is_anomalous, bool) or is_anomalous not in (0, 1):
        raise ValueError(f"is_anomalous must be the integer 0 or 1{where}")
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.size == 0:
        raise ValueError(f"Ground-truth mask must be a nonempty [H, W] array{where}")
    if mask.dtype != np.bool_:
        if mask.dtype != np.uint8:
            raise ValueError(
                f"Ground-truth mask must be bool or uint8 0/1, got {mask.dtype}{where}"
            )
        if mask.max() > 1:
            raise ValueError(f"Ground-truth mask must contain only 0/1 values{where}")
    binary = mask.astype(bool)
    if is_anomalous and not binary.any():
        raise ValueError(f"Anomalous image has an empty ground-truth mask{where}")
    if not is_anomalous and binary.any():
        raise ValueError(f"Good image has defect pixels in its ground-truth mask{where}")
    return binary


@dataclass(frozen=True)
class EvaluationRecord:
    """One test image: manifest identity, model output, and checked ground truth."""

    sample_id: str
    defect_type: str
    is_anomalous: int
    held_out: bool
    prediction: RawPrediction
    gt_mask: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a nonempty string")
        if not isinstance(self.prediction, RawPrediction):
            raise ValueError(f"prediction must be a RawPrediction for {self.sample_id}")
        if (self.defect_type == "good") == bool(self.is_anomalous):
            raise ValueError(f"defect_type disagrees with is_anomalous for {self.sample_id}")
        if self.held_out and not self.is_anomalous:
            raise ValueError(f"Only defect images can be held out: {self.sample_id}")
        mask = check_ground_truth(self.gt_mask, self.is_anomalous, self.sample_id)
        if mask.shape != self.prediction.score_map.shape:
            raise ValueError(
                f"score_map shape {self.prediction.score_map.shape} differs from ground truth "
                f"{mask.shape} for {self.sample_id}; resize the map to the original image "
                "with resize_score_map before evaluation"
            )
        object.__setattr__(self, "gt_mask", mask)

    @property
    def group(self) -> str:
        """good, known (defect type seen in training), or held_out."""
        if not self.is_anomalous:
            return "good"
        return "held_out" if self.held_out else "known"

    @classmethod
    def from_manifest_row(
        cls, row: dict, prediction: RawPrediction, gt_mask: np.ndarray
    ) -> "EvaluationRecord":
        """Build a record from a manifest row read with read_manifest."""
        if prediction.category != row["category"]:
            raise ValueError(
                f"Prediction category {prediction.category} differs from manifest "
                f"{row['category']} for {row['sample_id']}"
            )
        return cls(
            sample_id=row["sample_id"],
            defect_type=row["defect_type"],
            is_anomalous=int(row["is_anomalous"]),
            held_out=bool(row["held_out_defect"]) and row["defect_type"] == row["held_out_defect"],
            prediction=prediction,
            gt_mask=gt_mask,
        )


def check_record_set(records: Sequence[EvaluationRecord], require_calibrated: bool = False) -> dict:
    """Check that records come from one model run and return its identity."""
    if not records:
        raise ValueError("No evaluation records were given")
    ids = [r.sample_id for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs in evaluation records")
    identities = {
        (r.prediction.category, r.prediction.model_id, r.prediction.run_id) for r in records
    }
    if len(identities) != 1:
        raise ValueError(f"Records mix several category/model/run identities: {sorted(identities)}")
    calibrated = [isinstance(r.prediction, Prediction) for r in records]
    if any(calibrated) and not all(calibrated):
        raise ValueError("Records mix calibrated and uncalibrated predictions")
    if require_calibrated and not all(calibrated):
        raise ValueError("Evaluation requires calibrated predictions; run calibration first")
    if all(calibrated):
        thresholds = {(r.prediction.image_threshold, r.prediction.pixel_threshold) for r in records}
        if len(thresholds) != 1:
            raise ValueError("All calibrated predictions in one run must share thresholds")
    category, model_id, run_id = identities.pop()
    return {
        "category": category,
        "model_id": model_id,
        "run_id": run_id,
        "calibrated": all(calibrated),
        "sample_count": len(records),
    }


def fold_name(held_out_defect: str) -> str:
    """Fold label used in configs and run IDs: baseline or holdout-<defect>."""
    return f"holdout-{held_out_defect}" if held_out_defect else "baseline"


def manifest_identity(rows: Sequence[dict]) -> dict:
    """category / protocol / fold of a validated manifest (one identity per manifest)."""
    first = rows[0]
    return {
        "category": first["category"],
        "protocol": first["protocol"],
        "fold": fold_name(first["held_out_defect"]),
    }


def check_evaluation_manifest(
    artifact_metadata: dict,
    development_rows: Sequence[dict],
    evaluation_rows: Sequence[dict],
    allow_cross_manifest: bool = False,
) -> None:
    """Refuse to evaluate on images the model used for training or model selection.

    development_rows is the manifest recorded in the artifact at training time. Its
    train and val rows are development data; none of them may appear in the test
    rows being evaluated. A different protocol/fold is refused unless explicitly
    allowed (for example reusing an anomaly model fitted on the same normal train),
    and even then the overlap check still applies.
    """
    trained = manifest_identity(development_rows)
    evaluated = manifest_identity(evaluation_rows)
    if trained["category"] != evaluated["category"]:
        raise ValueError(
            f"Artifact was trained on {trained['category']}, manifest is {evaluated['category']}"
        )
    if (trained["protocol"], trained["fold"]) != (evaluated["protocol"], evaluated["fold"]):
        if not allow_cross_manifest:
            raise ValueError(
                f"Artifact was trained with {trained['protocol']}/{trained['fold']}, but the "
                f"manifest is {evaluated['protocol']}/{evaluated['fold']}. Use the training "
                "manifest, or pass --allow-cross-manifest when reusing a model on purpose"
            )
    expected_fold = artifact_metadata.get("fold")
    if expected_fold and expected_fold.removeprefix("smoke-") != trained["fold"]:
        raise ValueError(
            f"Artifact metadata says fold {expected_fold}, its training manifest is "
            f"{trained['fold']}"
        )
    development = {r["sample_id"] for r in development_rows if r["split"] in {"train", "val"}}
    overlap = sorted(
        r["sample_id"]
        for r in evaluation_rows
        if r["split"] == "test" and r["sample_id"] in development
    )
    if overlap:
        raise ValueError(
            f"{len(overlap)} test images were used for training or validation of this "
            f"artifact (first: {overlap[:3]}); evaluating on them would inflate the results"
        )


def load_development_manifest(artifact_metadata: dict, dataset_root: Path) -> list[dict]:
    """Read the manifest the artifact was trained with and verify it has not changed."""
    record = artifact_metadata.get("training_manifest")
    if not record or not record.get("path") or not record.get("sha256"):
        raise ValueError(
            "Artifact metadata has no training_manifest {path, sha256}; it is needed to prove "
            "that test images were not used for training or validation"
        )
    path = resolve(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Training manifest recorded in the artifact is missing: {path}")
    if file_sha256(path) != record["sha256"]:
        raise ValueError(f"Training manifest {path} changed since the artifact was trained")
    return read_manifest(path, dataset_root)


def check_same_test_ids(first: Iterable[str], second: Iterable[str]) -> None:
    """Fail when two models were evaluated on different test images."""
    first_ids, second_ids = set(first), set(second)
    if first_ids != second_ids:
        missing = sorted(first_ids - second_ids)[:5]
        extra = sorted(second_ids - first_ids)[:5]
        raise ValueError(f"Test IDs differ; only in first: {missing}, only in second: {extra}")
