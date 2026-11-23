"""U-Net image score: max of the probability map, optionally ignoring a border band.

v1 artifacts take the max over the whole map. On metal_nut the max of every good
image is a single corner pixel (a convolution padding artifact), so v2 artifacts
ignore a band of `border` pixels at the original image size. The map and the
pixel mask are unchanged; only the image score is affected.
"""

import numpy as np

from industrial_defect_detection.evaluation.metrics import auroc

BORDER_CANDIDATES = (0, 4, 8, 16, 32)
PLATEAU_TOLERANCE = 0.10
SELECTION_RULE = (
    "On validation images only: for each border b, take the 0.95 quantile of good image "
    "scores (q95). Eligible b: q95 within 10% of q95 at the widest candidate border (the "
    "border no longer drives good scores) and good-vs-known-defect AUROC not lower than "
    "at b=0. Choose the smallest b eligible in every category."
)


def image_score_border(metadata: dict) -> int:
    """Border band ignored by the image score; 0 (whole map) for v1 artifacts."""
    border = (metadata.get("score") or {}).get("image_score_border_px", 0)
    if isinstance(border, bool) or not isinstance(border, int) or border < 0:
        raise ValueError(f"image_score_border_px must be a non-negative integer, got {border!r}")
    return border


def border_max(score_map: np.ndarray, border: int) -> float:
    """Maximum of the map after removing `border` pixels on every side."""
    if border < 0:
        raise ValueError("border must be non-negative")
    if border == 0:
        return float(score_map.max())
    height, width = score_map.shape
    if 2 * border >= min(height, width):
        raise ValueError(f"border {border} leaves no pixels in a {height}x{width} map")
    return float(score_map[border:-border, border:-border].max())


def border_table(
    good_maps: list[np.ndarray],
    defect_maps: list[np.ndarray],
    candidates: tuple[int, ...] = BORDER_CANDIDATES,
) -> list[dict]:
    """Validation statistics of the image score for each candidate border."""
    rows = []
    for border in candidates:
        good = np.array([border_max(m, border) for m in good_maps])
        defect = np.array([border_max(m, border) for m in defect_maps])
        scores = np.concatenate([good, defect])
        labels = np.concatenate([np.zeros(good.size, bool), np.ones(defect.size, bool)])
        rows.append(
            {
                "border": border,
                "good_q95": float(np.quantile(good, 0.95)),
                "good_median": float(np.median(good)),
                "defect_min": float(defect.min()) if defect.size else None,
                "auroc": auroc(scores, labels),
            }
        )
    return rows


def eligible_borders(table: list[dict], tolerance: float = PLATEAU_TOLERANCE) -> list[int]:
    """Borders that satisfy SELECTION_RULE for one category."""
    widest = max(table, key=lambda row: row["border"])
    base = next(row for row in table if row["border"] == 0)
    eligible = []
    for row in table:
        plateau = abs(row["good_q95"] - widest["good_q95"]) <= tolerance * widest["good_q95"]
        separates = base["auroc"] is None or (
            row["auroc"] is not None and row["auroc"] >= base["auroc"] - 1e-12
        )
        if plateau and separates:
            eligible.append(row["border"])
    return eligible


def choose_border(tables: dict[str, list[dict]], tolerance: float = PLATEAU_TOLERANCE) -> int:
    """Smallest border eligible in every category (one rule for every U-Net artifact)."""
    common = None
    for table in tables.values():
        eligible = set(eligible_borders(table, tolerance))
        common = eligible if common is None else common & eligible
    if not common:
        raise ValueError("No border satisfies the selection rule in every category")
    return min(common)
