"""Logic behind the Streamlit demo, kept outside the UI so it can be tested.

The demo never re-implements prediction: it goes through the shared loader and
calibrated_predict, exactly like scripts/predict.py --calibrated.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from industrial_defect_detection.data.mvtec import CATEGORIES, IMAGE_SUFFIXES
from industrial_defect_detection.inference.contracts import Prediction
from industrial_defect_detection.inference.loader import (
    PREDICTORS,
    calibrated_predict,
)
from industrial_defect_detection.runs import read_json
from industrial_defect_detection.visualization.overlay import (
    DEFECT_RGB,
    GROUND_TRUTH_RGB,
    blend,
    heat_overlay,
    heatmap,
)

MODEL_NAMES = {"unet_resnet18": "U-Net (ResNet-18)", "patchcore": "PatchCore"}
# Known limitation shown next to the model: v1 image score is dominated by a border pixel.
KNOWN_ISSUES = {("unet_resnet18", "metal_nut"): "unet_metal_nut_border"}


@dataclass(frozen=True)
class ArtifactInfo:
    path: Path
    model_id: str
    category: str
    protocol: str
    fold: str
    run_id: str
    calibrated: bool

    @property
    def label(self) -> str:
        return f"{MODEL_NAMES.get(self.model_id, self.model_id)} · {self.fold} · {self.run_id}"

    @property
    def known_issue(self) -> str | None:
        return KNOWN_ISSUES.get((self.model_id, self.category))


def find_artifacts(root: Path) -> list[ArtifactInfo]:
    """Every artifact under artifacts/<protocol>/<model_id>/<category>/<run_id>/ with a loader."""
    found = []
    for metadata_path in sorted(Path(root).glob("*/*/*/*/metadata.json")):
        try:
            meta = read_json(metadata_path)
        except (OSError, ValueError):
            continue
        if meta.get("model_id") not in PREDICTORS or meta.get("category") not in CATEGORIES:
            continue
        found.append(
            ArtifactInfo(
                path=metadata_path.parent,
                model_id=meta["model_id"],
                category=meta["category"],
                protocol=meta.get("protocol", ""),
                fold=meta.get("fold", ""),
                run_id=meta.get("run_id", metadata_path.parent.name),
                calibrated=meta.get("status") == "calibrated" and bool(meta.get("calibration")),
            )
        )
    return found


def sample_images(dataset_root: Path, category: str) -> dict[str, list[Path]]:
    """Original test images of a category grouped by defect type (good first)."""
    test_dir = Path(dataset_root) / category / "test"
    if not test_dir.is_dir():
        return {}
    groups = {}
    for folder in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if images:
            groups[folder.name] = images
    return dict(sorted(groups.items(), key=lambda item: item[0] != "good"))


def ground_truth_mask(image_path: Path) -> np.ndarray | None:
    """Mask for an original MVTec test image; None for good or external images."""
    image_path = Path(image_path)
    if image_path.parent.name == "good" or image_path.parents[1].name != "test":
        return None
    mask_path = (
        image_path.parents[2]
        / "ground_truth"
        / image_path.parent.name
        / f"{image_path.stem}_mask.png"
    )
    if not mask_path.is_file():
        return None
    with Image.open(mask_path) as mask:
        return np.array(mask.convert("L")) > 0


def read_image(source) -> np.ndarray:
    """uint8 RGB [H,W,3] from a path or an uploaded file; ValueError if unreadable."""
    try:
        with Image.open(source) as image:
            return np.array(image.convert("RGB"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Cannot read image: {error}") from error


def inspect(predictor, image: np.ndarray) -> tuple[Prediction, float]:
    """Calibrated prediction (same call as predict.py --calibrated) and its time in ms."""
    try:
        import torch

        synchronize = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    except ImportError:
        synchronize = lambda: None  # noqa: E731
    synchronize()
    started = time.perf_counter()
    prediction = calibrated_predict(predictor, image)
    synchronize()
    return prediction, 1000 * (time.perf_counter() - started)


def display_range(prediction: Prediction) -> tuple[float, float]:
    """Display only: 0 to twice the pixel threshold, so the threshold sits mid-scale.

    The same rule for every model and image keeps colors comparable; scores are
    never rescaled per image for decisions or metrics.
    """
    return 0.0, 2.0 * max(float(prediction.pixel_threshold), 1e-12)


def result_panels(
    image: np.ndarray, prediction: Prediction, gt_mask: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Images for the UI, all at the original image size."""
    low, high = display_range(prediction)
    panels = {"image": image}
    if gt_mask is not None:
        panels["ground_truth"] = blend(image, gt_mask, GROUND_TRUTH_RGB)
    panels["heatmap"] = heat_overlay(image, heatmap(prediction.score_map, low, high))
    panels["mask"] = blend(image, prediction.pred_mask, DEFECT_RGB)
    return panels
