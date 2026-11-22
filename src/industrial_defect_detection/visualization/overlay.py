"""Shared heatmap and overlay rendering for evaluation figures and the demo.

Color scaling is for display only. Metrics and PASS/DEFECT always use raw scores.
"""

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw

DEFECT_RGB = np.array([255, 40, 40], dtype=np.float32)
GROUND_TRUTH_RGB = np.array([40, 220, 40], dtype=np.float32)


def heatmap(score_map: np.ndarray, low: float, high: float) -> np.ndarray:
    """Colorize a score map with a fixed [low, high] range shared by one model run."""
    span = high - low if high > low else 1.0
    scaled = np.clip((score_map - low) / span, 0.0, 1.0)
    return (colormaps["jet"](scaled)[..., :3] * 255).astype(np.uint8)


def blend(image: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    out = image.astype(np.float32).copy()
    region = mask.astype(bool)
    out[region] = out[region] * (1 - alpha) + color * alpha
    return out.astype(np.uint8)


def heat_overlay(image: np.ndarray, colored: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return (image.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha).astype(
        np.uint8
    )


def panel(
    image: np.ndarray,
    score_map: np.ndarray,
    pred_mask: np.ndarray | None,
    low: float,
    high: float,
    gt_mask: np.ndarray | None = None,
    title: str = "",
    tile_size: int = 256,
) -> np.ndarray:
    """Image | ground truth (if known) | heatmap overlay | predicted mask overlay."""
    tiles = [image]
    labels = ["image"]
    if gt_mask is not None:
        tiles.append(blend(image, gt_mask, GROUND_TRUTH_RGB))
        labels.append("ground truth")
    tiles.append(heat_overlay(image, heatmap(score_map, low, high)))
    labels.append("score heatmap")
    if pred_mask is not None:
        tiles.append(blend(image, pred_mask, DEFECT_RGB))
        labels.append("predicted mask")
    resized = []
    for tile, label in zip(tiles, labels, strict=True):
        picture = Image.fromarray(tile).resize((tile_size, tile_size), Image.Resampling.BILINEAR)
        ImageDraw.Draw(picture).text((4, 4), label, fill=(255, 255, 255))
        resized.append(np.asarray(picture))
    row = np.concatenate(resized, axis=1)
    if not title:
        return row
    header = Image.new("RGB", (row.shape[1], 18), (0, 0, 0))
    ImageDraw.Draw(header).text((4, 3), title, fill=(255, 255, 255))
    return np.concatenate([np.asarray(header), row], axis=0)
