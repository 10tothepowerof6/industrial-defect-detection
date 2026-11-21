"""One metric implementation shared by PatchCore, EfficientAD, and U-Net.

Undefined metrics (for example AUROC without one of the two classes) are returned
as None and their reason is listed under "undefined", next to the sample counts.
"""

from collections import Counter
from collections.abc import Sequence

import numpy as np

from .checks import EvaluationRecord, check_record_set

GROUPS = ("good", "known", "held_out")


def _ranking_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False/true positive counts at each distinct threshold, highest score first."""
    order = np.argsort(scores, kind="stable")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    del order
    last_of_tie = np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1])
    last_of_tie = np.append(last_of_tie, sorted_scores.size - 1)
    count_dtype = np.int64 if sorted_labels.size >= 2**31 else np.int32
    tps = np.cumsum(sorted_labels, dtype=count_dtype)[last_of_tie].astype(np.int64)
    fps = last_of_tie.astype(np.int64) + 1 - tps
    return fps, tps


def _class_problem(labels: np.ndarray) -> str | None:
    positives = int(labels.sum())
    if positives == 0:
        return "no positive samples"
    if positives == labels.size:
        return "no negative samples"
    return None


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Exact area under the ROC curve; ties count as half, like scikit-learn."""
    labels = np.asarray(labels, dtype=bool)
    if _class_problem(labels):
        return None
    fps, tps = _ranking_curve(np.asarray(scores), labels)
    fpr = np.concatenate(([0.0], fps / fps[-1]))
    tpr = np.concatenate(([0.0], tps / tps[-1]))
    return float(np.trapezoid(tpr, fpr))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Step-wise average precision, matching sklearn.metrics.average_precision_score."""
    labels = np.asarray(labels, dtype=bool)
    if not labels.any():
        return None
    fps, tps = _ranking_curve(np.asarray(scores), labels)
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(recall, prepend=0.0) * precision))


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def decision_metrics(predicted: np.ndarray, labels: np.ndarray) -> dict:
    """Precision, recall, F1 and false-positive rate for boolean decisions."""
    predicted = np.asarray(predicted, dtype=bool)
    labels = np.asarray(labels, dtype=bool)
    tp = int((predicted & labels).sum())
    fp = int((predicted & ~labels).sum())
    fn = int((~predicted & labels).sum())
    tn = int((~predicted & ~labels).sum())
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": _ratio(fp, fp + tn),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def dice_iou(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Dice and IoU of two boolean masks; both empty scores 1, only one empty scores 0."""
    predicted = np.asarray(predicted, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = int((predicted & target).sum())
    predicted_area = int(predicted.sum())
    target_area = int(target.sum())
    if predicted_area == 0 and target_area == 0:
        return 1.0, 1.0
    union = predicted_area + target_area - intersection
    return 2 * intersection / (predicted_area + target_area), intersection / union


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _undefined(section: dict, reasons: dict) -> dict:
    section["undefined"] = {k: v for k, v in reasons.items() if section.get(k) is None}
    return section


def evaluate_records(records: Sequence[EvaluationRecord]) -> dict:
    """Compute every required image, pixel, and group metric from calibrated records."""
    identity = check_record_set(records, require_calibrated=True)
    first = records[0].prediction
    labels = np.array([r.is_anomalous for r in records], dtype=bool)
    scores = np.array([r.prediction.image_score for r in records], dtype=np.float64)
    decisions = np.array([r.prediction.is_defective for r in records], dtype=bool)
    groups = np.array([r.group for r in records])
    image_class = _class_problem(labels)

    image = {
        "auroc": auroc(scores, labels),
        "average_precision": average_precision(scores, labels),
        **decision_metrics(decisions, labels),
    }
    image = _undefined(
        image,
        {
            "auroc": image_class,
            "average_precision": "no defect images",
            "precision": "no image was predicted defective",
            "recall": "no defect images",
            "f1": "no defect images and no defect predictions",
            "false_positive_rate": "no good images",
        },
    )

    pixel_labels = np.concatenate([r.gt_mask.ravel() for r in records])
    pixel_scores = np.concatenate([r.prediction.score_map.ravel() for r in records])
    pixel_class = _class_problem(pixel_labels)
    pixel = {
        "auroc": auroc(pixel_scores, pixel_labels),
        "average_precision": average_precision(pixel_scores, pixel_labels),
    }
    del pixel_scores, pixel_labels
    overlaps = [dice_iou(r.prediction.pred_mask, r.gt_mask) for r in records]
    defect_overlaps = [o for o, r in zip(overlaps, records, strict=True) if r.is_anomalous]
    good_overlaps = [o for o, r in zip(overlaps, records, strict=True) if not r.is_anomalous]
    pixel.update(
        {
            "dice_defect_images": _mean([d for d, _ in defect_overlaps]),
            "iou_defect_images": _mean([i for _, i in defect_overlaps]),
            "dice_good_images": _mean([d for d, _ in good_overlaps]),
            "iou_good_images": _mean([i for _, i in good_overlaps]),
            "dice_iou_rule": "per image, then mean; both masks empty = 1, one empty = 0",
        }
    )
    pixel = _undefined(
        pixel,
        {
            "auroc": pixel_class,
            "average_precision": "no defect pixels",
            "dice_defect_images": "no defect images",
            "iou_defect_images": "no defect images",
            "dice_good_images": "no good images",
            "iou_good_images": "no good images",
        },
    )

    group_metrics = {}
    good = groups == "good"
    for group in GROUPS:
        members = groups == group
        count = int(members.sum())
        group_overlaps = [o for o, g in zip(overlaps, groups, strict=True) if g == group]
        entry = {"images": count}
        if group == "good":
            entry["false_positive_rate"] = _ratio(int(decisions[members].sum()), count)
            reasons = {"false_positive_rate": "no good images"}
        else:
            # Ranking metrics for one defect group use the good test images as negatives.
            subset = members | good
            entry["recall"] = _ratio(int(decisions[members].sum()), count)
            entry["auroc_vs_good"] = auroc(scores[subset], labels[subset])
            entry["average_precision_vs_good"] = average_precision(scores[subset], labels[subset])
            reason = "no images in this group" if not count else _class_problem(labels[subset])
            reasons = {
                "recall": "no images in this group",
                "auroc_vs_good": reason,
                "average_precision_vs_good": "no images in this group",
            }
        entry["mean_dice"] = _mean([d for d, _ in group_overlaps])
        entry["mean_iou"] = _mean([i for _, i in group_overlaps])
        reasons.update({"mean_dice": "no images in this group", "mean_iou": "no images in this group"})
        group_metrics[group] = _undefined(entry, reasons)

    per_defect = {}
    for defect_type in sorted({r.defect_type for r in records} - {"good"}):
        members = [i for i, r in enumerate(records) if r.defect_type == defect_type]
        per_defect[defect_type] = {
            "group": records[members[0]].group,
            "images": len(members),
            "recall": float(decisions[members].mean()),
            "mean_dice": _mean([overlaps[i][0] for i in members]),
            "mean_iou": _mean([overlaps[i][1] for i in members]),
        }

    return {
        "schema_version": 1,
        **identity,
        "image_threshold": float(first.image_threshold),
        "pixel_threshold": float(first.pixel_threshold),
        "counts": {
            "images": len(records),
            "good_images": int((~labels).sum()),
            "defect_images": int(labels.sum()),
            "pixels": int(sum(r.gt_mask.size for r in records)),
            "defect_pixels": int(sum(r.gt_mask.sum() for r in records)),
            "by_group": dict(Counter(groups.tolist())),
        },
        "image": image,
        "pixel": pixel,
        "groups": group_metrics,
        "per_defect_type": per_defect,
    }


def prediction_rows(records: Sequence[EvaluationRecord]) -> list[dict]:
    """One row per image for predictions.csv."""
    rows = []
    for record in records:
        prediction = record.prediction
        dice, iou = dice_iou(prediction.pred_mask, record.gt_mask)
        rows.append(
            {
                "sample_id": record.sample_id,
                "defect_type": record.defect_type,
                "group": record.group,
                "is_anomalous": record.is_anomalous,
                "image_score": float(prediction.image_score),
                "image_threshold": float(prediction.image_threshold),
                "is_defective": int(prediction.is_defective),
                "correct": int(prediction.is_defective == bool(record.is_anomalous)),
                "pixel_threshold": float(prediction.pixel_threshold),
                "predicted_defect_pixels": int(prediction.pred_mask.sum()),
                "true_defect_pixels": int(record.gt_mask.sum()),
                "dice": dice,
                "iou": iou,
            }
        )
    return rows
