"""Read original MVTec images without changing the downloaded dataset."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

CATEGORIES = ("tile", "metal_nut")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def dataset_path(root: Path, relative_path: str) -> Path:
    """Resolve a portable dataset path and reject references outside its root."""
    relative = Path(relative_path)
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(resolved_root):
        raise ValueError(f"Dataset path must remain inside {resolved_root}: {relative_path}")
    return path


@dataclass(frozen=True)
class MVTecSample:
    sample_id: str
    image_path: str
    mask_path: str
    category: str
    defect_type: str
    source_split: Literal["train", "test"]

    @property
    def is_anomalous(self) -> bool:
        return self.defect_type != "good"


def discover_samples(root: Path, categories: tuple[str, ...] = CATEGORIES) -> list[MVTecSample]:
    samples = []
    for category in categories:
        if category not in CATEGORIES:
            raise ValueError(f"Unsupported category: {category}")
        category_dir = root / category
        for required in ("train/good", "test/good", "ground_truth"):
            if not (category_dir / required).is_dir():
                raise FileNotFoundError(f"Missing dataset directory: {category_dir / required}")
        for source_split in ("train", "test"):
            groups = sorted((category_dir / source_split).iterdir())
            for group in groups:
                if not group.is_dir():
                    continue
                if source_split == "train" and group.name != "good":
                    raise ValueError(f"Original MVTec training data must be good only: {group}")
                images = sorted(p for p in group.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
                if not images:
                    raise ValueError(f"Empty image group: {group}")
                for image in images:
                    relative = image.relative_to(root).as_posix()
                    mask_path = ""
                    if group.name != "good":
                        mask = category_dir / "ground_truth" / group.name / f"{image.stem}_mask.png"
                        mask_path = mask.relative_to(root).as_posix()
                    samples.append(
                        MVTecSample(
                            relative, relative, mask_path, category, group.name, source_split
                        )
                    )
    if len({s.sample_id for s in samples}) != len(samples):
        raise ValueError("Duplicate sample IDs or repeated categories")
    return samples


def load_sample(
    root: Path, sample: MVTecSample, image_size: tuple[int, int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return uint8 RGB [H,W,3] and uint8 binary mask [H,W]. Size uses (H,W)."""
    with Image.open(dataset_path(root, sample.image_path)) as image_file:
        image = image_file.convert("RGB")
        original_size = image.size
    if sample.is_anomalous:
        if not sample.mask_path:
            raise ValueError(f"Missing mask path for anomalous image: {sample.sample_id}")
        with Image.open(dataset_path(root, sample.mask_path)) as mask_file:
            if mask_file.size != original_size:
                raise ValueError(f"Image/mask size mismatch: {sample.sample_id}")
            mask = mask_file.convert("L")
    else:
        if sample.mask_path:
            raise ValueError(f"Good sample must not reference a mask: {sample.sample_id}")
        mask = Image.new("L", original_size, 0)
    if image_size is not None:
        if len(image_size) != 2 or any(d <= 0 for d in image_size):
            raise ValueError("image_size must contain positive height and width")
        size = (image_size[1], image_size[0])
        image = image.resize(size, Image.Resampling.BILINEAR)
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return np.array(image, dtype=np.uint8), (np.array(mask) > 0).astype(np.uint8)
