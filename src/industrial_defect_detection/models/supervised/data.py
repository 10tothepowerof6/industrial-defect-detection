"""Manifest-backed tensors for U-Net training, with label-safe paired augmentation."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from industrial_defect_detection.data.manifests import load_manifest_sample

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize(image: np.ndarray) -> torch.Tensor:
    """uint8 RGB [H,W,3] -> float32 ImageNet-normalized [3,H,W]."""
    scaled = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(scaled.transpose(2, 0, 1)))


def augment(
    image: np.ndarray, mask: np.ndarray, settings: dict, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same geometric change to image and mask. Never changes pixel labels."""
    if settings.get("horizontal_flip") and rng.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    if settings.get("vertical_flip") and rng.random() < 0.5:
        image, mask = image[::-1], mask[::-1]
    if settings.get("rotate90"):
        turns = int(rng.integers(4))
        image, mask = np.rot90(image, turns), np.rot90(mask, turns)
    return image, mask


class ManifestSegmentationDataset(Dataset):
    """Images and binary masks for one split, resized once and kept in memory.

    MVTec tile/metal_nut have a few hundred images per split, so caching avoids
    decoding 700-840 px PNG files every epoch.
    """

    def __init__(
        self,
        rows: list[dict],
        dataset_root: Path,
        image_size: tuple[int, int],
        augmentation: dict | None = None,
        seed: int = 0,
    ):
        if not rows:
            raise ValueError("Dataset split is empty")
        self.rows = rows
        self.augmentation = augmentation or {}
        self.rng = np.random.default_rng(seed)
        self.samples = [load_manifest_sample(dataset_root, row, image_size) for row in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        image, mask = self.samples[index]
        if self.augmentation:
            image, mask = augment(image, mask, self.augmentation, self.rng)
        return {
            "image": normalize(image),
            "mask": torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))[None],
            "is_anomalous": int(self.rows[index]["is_anomalous"]),
            "index": index,
        }
