"""Audit image integrity, mask alignment, and duplicate image content."""

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .mvtec import IMAGE_SUFFIXES, MVTecSample, dataset_path, discover_samples, load_sample


def image_digest(image: np.ndarray) -> str:
    digest = hashlib.sha256(str(image.shape).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def audit_dataset(root: Path) -> dict:
    samples = discover_samples(root)
    counts = Counter()
    dimensions = Counter()
    duplicates = defaultdict(list)
    errors = []
    inventory = []
    for sample in samples:
        counts[f"{sample.category}/{sample.source_split}/{sample.defect_type}"] += 1
        try:
            image, mask = load_sample(root, sample)
            if sample.is_anomalous:
                with Image.open(dataset_path(root, sample.mask_path)) as mask_file:
                    labels = set(np.unique(np.array(mask_file)).tolist())
                if not labels.issubset({0, 255}):
                    raise ValueError(f"Unexpected raw mask values: {sorted(labels)}")
                if not mask.any():
                    raise ValueError("Anomalous image has an empty ground-truth mask")
            digest = image_digest(image)
            duplicates[digest].append(sample.sample_id)
            dimensions[f"{sample.category}/{image.shape[0]}x{image.shape[1]}"] += 1
            inventory.append(
                {
                    "sample_id": sample.sample_id,
                    "image_sha256": digest,
                    "height": image.shape[0],
                    "width": image.shape[1],
                }
            )
        except (OSError, UnidentifiedImageError, ValueError) as error:
            errors.append({"sample_id": sample.sample_id, "error": str(error)})

    expected_masks = {s.mask_path for s in samples if s.mask_path}
    actual_masks = {
        p.relative_to(root).as_posix()
        for category in {s.category for s in samples}
        for p in (root / category / "ground_truth").rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    }
    for orphan in sorted(actual_masks - expected_masks):
        errors.append({"sample_id": orphan, "error": "Ground-truth mask has no matching image"})
    duplicate_groups = [ids for ids in duplicates.values() if len(ids) > 1]
    return {
        "schema_version": 1,
        "sample_count": len(samples),
        "valid_sample_count": len(inventory),
        "counts": dict(sorted(counts.items())),
        "dimensions": dict(sorted(dimensions.items())),
        "errors": errors,
        "duplicate_image_groups": duplicate_groups,
        "inventory": inventory,
        "passed": not errors and not duplicate_groups,
    }


def save_preview(root: Path, sample: MVTecSample, output_path: Path) -> None:
    """Save RGB / ground truth / overlay panels using a bounded preview size."""
    image, mask = load_sample(root, sample, image_size=(256, 256))
    overlay = image.copy()
    overlay[mask.astype(bool)] = (
        image[mask.astype(bool)].astype(np.float32) * 0.5 + np.array([255, 0, 0]) * 0.5
    ).astype(np.uint8)
    mask_rgb = np.repeat((mask * 255)[..., None], 3, axis=2)
    panels = np.concatenate((image, mask_rgb, overlay), axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panels).save(output_path)
