"""Offline PatchCore adapter using Anomalib's model and explicit preprocessing."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from PIL import Image

from industrial_defect_detection.data.mvtec import CATEGORIES
from industrial_defect_detection.evaluation.checks import resize_score_map
from industrial_defect_detection.inference.contracts import RawPrediction

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
WEIGHTS_PREFIX = "f37072fd"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only cpu and cuda devices are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose cpu explicitly if needed")
    return device


def preprocess(image: np.ndarray, preprocessing: dict) -> torch.Tensor:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or not image.size
    ):
        raise ValueError("Expected a nonempty uint8 RGB image [H,W,3]")
    size = preprocessing["resize"]
    resized = Image.fromarray(image).resize(
        (size["width"], size["height"]), Image.Resampling.BILINEAR
    )
    tensor = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float() / 255
    mean = torch.tensor(preprocessing["normalize_mean"])[:, None, None]
    std = torch.tensor(preprocessing["normalize_std"])[:, None, None]
    return (tensor - mean) / std


def make_model(settings: dict) -> PatchcoreModel:
    if settings["backbone"] != "resnet18":
        raise ValueError("This adapter supports the verified resnet18 backbone only")
    return PatchcoreModel(
        layers=settings["layers"],
        backbone=settings["backbone"],
        pre_trained=False,
        num_neighbors=settings["num_neighbors"],
    )


def load_pretrained(model: PatchcoreModel, weights: Path) -> str:
    if not weights.is_file():
        raise FileNotFoundError(
            f"Missing pretrained weights: {weights}. Download {WEIGHTS_URL} explicitly; "
            "training never downloads weights automatically."
        )
    digest = sha256(weights)
    if not digest.startswith(WEIGHTS_PREFIX):
        raise ValueError("ResNet-18 checkpoint hash does not match the configured official weights")
    source = torch.load(weights, map_location="cpu", weights_only=True)
    extractor = model.feature_extractor.feature_extractor
    # features_only omits the classifier and the unused final ResNet stage.
    selected = {}
    for key, value in extractor.state_dict().items():
        # The official legacy checkpoint predates BatchNorm's batch-count buffers.
        # These counters do not affect frozen eval-mode features.
        if key.endswith("num_batches_tracked") and key not in source:
            selected[key] = torch.zeros_like(value)
        else:
            selected[key] = source[key]
    extractor.load_state_dict(selected, strict=True)
    return digest


class PatchcorePredictor:
    """Implements H03's predictor interface without depending on its unmerged loader."""

    def __init__(self, artifact_dir: Path, device: str | None = None):
        self.artifact_dir = Path(artifact_dir)
        self.metadata = json.loads(
            (self.artifact_dir / "metadata.json").read_text(encoding="utf-8")
        )
        meta = self.metadata
        if meta["schema_version"] != 1 or meta["model_id"] != "patchcore":
            raise ValueError("Unsupported artifact schema or model_id")
        if meta["category"] not in CATEGORIES or not meta["run_id"]:
            raise ValueError("Invalid artifact category or run_id")
        if meta["preprocessing"]["color"] != "RGB":
            raise ValueError("Unsupported artifact color space")
        if meta["preprocessing"]["resize"]["interpolation"] != "bilinear":
            raise ValueError("Unsupported artifact resize interpolation")
        self.model_id, self.category, self.run_id = (
            meta["model_id"],
            meta["category"],
            meta["run_id"],
        )
        self.device = select_device(device or "cpu")
        checkpoint = (self.artifact_dir / meta["checkpoint"]).resolve()
        if not checkpoint.is_relative_to(self.artifact_dir.resolve()):
            raise ValueError("Checkpoint path escapes artifact directory")
        if sha256(checkpoint) != meta["checkpoint_sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
        self.model = make_model(meta["model"])
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        bank = state["memory_bank"]
        if bank.ndim != 2 or bank.shape[0] < meta["model"]["num_neighbors"]:
            raise ValueError("Missing or undersized PatchCore memory bank")
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> RawPrediction:
        tensor = preprocess(image, self.metadata["preprocessing"]).unsqueeze(0).to(self.device)
        result = self.model(tensor)
        score_map = result.anomaly_map[0, 0].float().cpu().numpy()
        score_map = resize_score_map(score_map, image.shape[0], image.shape[1])
        return RawPrediction(
            float(result.pred_score.reshape(-1)[0].cpu()),
            score_map,
            self.category,
            self.model_id,
            self.run_id,
        )
