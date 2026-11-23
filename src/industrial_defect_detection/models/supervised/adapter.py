"""Load a U-Net artifact and turn one original-size image into a RawPrediction."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from industrial_defect_detection.inference.contracts import RawPrediction
from industrial_defect_detection.runs import read_json

from .data import normalize
from .model import build_unet
from .scoring import border_max, image_score_border
from .train import MODEL_ID


class UNetPredictor:
    """Reads weights and preprocessing from the artifact only; nothing is hardcoded."""

    model_id = MODEL_ID

    def __init__(self, artifact_dir: Path, device: str | None = None):
        self.artifact_dir = Path(artifact_dir)
        metadata_path = self.artifact_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing {metadata_path}; train the model first")
        self.metadata = read_json(metadata_path)
        if self.metadata.get("model_id") != MODEL_ID:
            raise ValueError(f"{self.artifact_dir} is not a {MODEL_ID} artifact")
        checkpoint_path = self.artifact_dir / self.metadata["checkpoint"]
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint {checkpoint_path}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # encoder_weights=None: everything comes from the checkpoint, not a local weight cache.
        self.model = build_unet(self.metadata["model"]["encoder"], encoder_weights=None)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state["state_dict"])
        self.model.to(self.device).eval()
        resize = self.metadata["preprocessing"]["resize"]
        self.input_size = (resize["height"], resize["width"])
        self.border = image_score_border(self.metadata)

    @property
    def category(self) -> str:
        return self.metadata["category"]

    @property
    def run_id(self) -> str:
        return self.metadata["run_id"]

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> RawPrediction:
        """image: uint8 RGB [H,W,3] at original size. Map and score use that same size."""
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Expected a uint8 RGB image with shape [H, W, 3]")
        height, width = image.shape[:2]
        resized = np.asarray(
            Image.fromarray(image).resize(self.input_size[::-1], Image.Resampling.BILINEAR)
        )
        tensor = normalize(resized)[None].to(self.device)
        probabilities = torch.sigmoid(self.model(tensor).float())
        probabilities = F.interpolate(
            probabilities, size=(height, width), mode="bilinear", align_corners=False
        )
        score_map = probabilities[0, 0].clamp(0, 1).cpu().numpy().astype(np.float32)
        return RawPrediction(
            image_score=border_max(score_map, self.border),
            score_map=score_map,
            category=self.category,
            model_id=MODEL_ID,
            run_id=self.run_id,
        )
