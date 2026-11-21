import unittest

import numpy as np

from industrial_defect_detection.evaluation.calibration import (
    apply_calibration,
    calibrate_thresholds,
    resolve_quantiles,
)
from industrial_defect_detection.evaluation.checks import EvaluationRecord
from industrial_defect_detection.evaluation.metrics import (
    auroc,
    average_precision,
    decision_metrics,
    dice_iou,
    evaluate_records,
    prediction_rows,
)
from industrial_defect_detection.inference.contracts import RawPrediction

SHAPE = (8, 8)


def square(value=1.0, background=0.0):
    score_map = np.full(SHAPE, background, dtype=np.float32)
    score_map[2:5, 2:5] = value
    return score_map


def gt_square():
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[2:5, 2:5] = 1
    return mask


def raw(image_score, score_map, run_id="synthetic"):
    return RawPrediction(image_score, score_map, "tile", "unet_resnet18", run_id)


def record(sample_id, defect_type, image_score, score_map, held_out=False, thresholds=(0.5, 0.5)):
    is_anomalous = int(defect_type != "good")
    mask = gt_square() if is_anomalous else np.zeros(SHAPE, dtype=np.uint8)
    prediction = raw(image_score, score_map).with_thresholds(*thresholds)
    return EvaluationRecord(sample_id, defect_type, is_anomalous, held_out, prediction, mask)


class RankingMetricTests(unittest.TestCase):
    def test_perfect_reversed_and_tied_scores(self):
        labels = np.array([0, 0, 1, 1], dtype=bool)
        self.assertEqual(auroc(np.array([0.1, 0.2, 0.8, 0.9]), labels), 1.0)
        self.assertEqual(auroc(np.array([0.9, 0.8, 0.2, 0.1]), labels), 0.0)
        self.assertEqual(auroc(np.full(4, 0.5), labels), 0.5)
        self.assertEqual(average_precision(np.array([0.1, 0.2, 0.8, 0.9]), labels), 1.0)
        self.assertEqual(average_precision(np.full(4, 0.5), labels), 0.5)

    def test_missing_class_is_undefined(self):
        self.assertIsNone(auroc(np.array([0.1, 0.2]), np.array([0, 0])))
        self.assertIsNone(auroc(np.array([0.1, 0.2]), np.array([1, 1])))
        self.assertIsNone(average_precision(np.array([0.1, 0.2]), np.array([0, 0])))
        self.assertEqual(average_precision(np.array([0.1, 0.2]), np.array([1, 1])), 1.0)

    def test_matches_scikit_learn_with_ties(self):
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
        except ImportError:
            self.skipTest("scikit-learn is not installed")
        rng = np.random.default_rng(0)
        for trial in range(20):
            labels = rng.random(300) < 0.3
            # Rounding creates many tied scores, the hard case for curve code.
            scores = np.round(rng.random(300) + labels * 0.3, 1).astype(np.float32)
            with self.subTest(trial=trial):
                self.assertAlmostEqual(auroc(scores, labels), roc_auc_score(labels, scores))
                self.assertAlmostEqual(
                    average_precision(scores, labels), average_precision_score(labels, scores)
                )


class DecisionAndOverlapTests(unittest.TestCase):
    def test_decision_metrics(self):
        result = decision_metrics(
            np.array([1, 1, 0, 0, 1], dtype=bool), np.array([1, 0, 1, 0, 1], dtype=bool)
        )
        self.assertEqual(result["confusion"], {"tp": 2, "fp": 1, "fn": 1, "tn": 1})
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertAlmostEqual(result["f1"], 2 / 3)
        self.assertAlmostEqual(result["false_positive_rate"], 0.5)

    def test_decision_metrics_without_positive_predictions(self):
        result = decision_metrics(np.zeros(3, dtype=bool), np.array([0, 0, 1], dtype=bool))
        self.assertIsNone(result["precision"])
        self.assertEqual(result["recall"], 0.0)

    def test_dice_iou_empty_rules(self):
        empty = np.zeros(SHAPE, dtype=bool)
        full = gt_square().astype(bool)
        self.assertEqual(dice_iou(empty, empty), (1.0, 1.0))
        self.assertEqual(dice_iou(full, empty), (0.0, 0.0))
        self.assertEqual(dice_iou(empty, full), (0.0, 0.0))
        self.assertEqual(dice_iou(full, full), (1.0, 1.0))
        half = full.copy()
        half[2:5, 2] = False  # 6 of 9 pixels
        dice, iou = dice_iou(half, full)
        self.assertAlmostEqual(dice, 12 / 15)
        self.assertAlmostEqual(iou, 6 / 9)


class CalibrationTests(unittest.TestCase):
    def test_quantiles_from_normal_validation(self):
        normal = [raw(float(i), np.full(SHAPE, i / 10, np.float32)) for i in range(1, 21)]
        calibration = calibrate_thresholds(normal)
        self.assertAlmostEqual(calibration["image_threshold"], np.quantile(np.arange(1, 21), 0.95))
        pixels = np.concatenate([p.score_map.ravel() for p in normal])
        self.assertAlmostEqual(calibration["pixel_threshold"], float(np.quantile(pixels, 0.995)))
        self.assertEqual(calibration["normal_val_images"], 20)
        self.assertEqual(calibration["normal_val_pixels"], 20 * 64)
        calibrated = apply_calibration(normal[0], calibration)
        self.assertEqual(calibrated.image_threshold, calibration["image_threshold"])

    def test_quantile_priority_cli_then_artifact_config_then_default(self):
        config = {"config": {"calibration": {"image_quantile": 0.9, "pixel_quantile": 0.99}}}
        self.assertEqual(
            resolve_quantiles(config, 0.8, None),
            {
                "image_quantile": 0.8,
                "image_quantile_source": "cli",
                "pixel_quantile": 0.99,
                "pixel_quantile_source": "artifact_config",
            },
        )
        resolved = resolve_quantiles({"config": {"calibration": {"image_quantile": 0.9}}})
        self.assertEqual(
            (resolved["image_quantile"], resolved["image_quantile_source"]),
            (0.9, "artifact_config"),
        )
        self.assertEqual(
            (resolved["pixel_quantile"], resolved["pixel_quantile_source"]), (0.995, "default")
        )
        self.assertEqual(resolve_quantiles({})["image_quantile_source"], "default")

    def test_resolved_quantiles_change_and_are_saved_with_thresholds(self):
        normal = [raw(float(i), np.full(SHAPE, i / 10, np.float32)) for i in range(1, 21)]
        quantiles = resolve_quantiles({"config": {"calibration": {"image_quantile": 0.5}}})
        calibration = calibrate_thresholds(
            normal, quantiles["image_quantile"], quantiles["pixel_quantile"]
        )
        self.assertEqual(calibration["image_quantile"], 0.5)
        self.assertEqual(calibration["pixel_quantile"], 0.995)
        self.assertAlmostEqual(calibration["image_threshold"], 10.5)

    def test_rejects_wrong_inputs(self):
        with self.assertRaises(ValueError):
            calibrate_thresholds([])
        with self.assertRaises(ValueError):
            calibrate_thresholds([raw(0.1, square())], image_quantile=1.0)
        with self.assertRaises(ValueError):
            calibrate_thresholds([raw(0.1, square()), raw(0.2, square(), run_id="other")])
        calibration = calibrate_thresholds([raw(0.1, square())])
        with self.assertRaisesRegex(ValueError, "belongs to"):
            apply_calibration(raw(0.1, square(), run_id="other"), calibration)


class EvaluateRecordsTests(unittest.TestCase):
    def test_perfect_predictions(self):
        records = [
            record("g0", "good", 0.1, square(0.0)),
            record("g1", "good", 0.2, square(0.0)),
            record("k0", "oil", 0.9, square(1.0)),
            record("h0", "crack", 0.8, square(1.0), held_out=True),
        ]
        result = evaluate_records(records)
        self.assertEqual(result["image"]["auroc"], 1.0)
        self.assertEqual(result["image"]["f1"], 1.0)
        self.assertEqual(result["image"]["false_positive_rate"], 0.0)
        self.assertEqual(result["pixel"]["auroc"], 1.0)
        self.assertEqual(result["pixel"]["dice_defect_images"], 1.0)
        self.assertEqual(result["pixel"]["dice_good_images"], 1.0)
        self.assertEqual(result["groups"]["held_out"]["recall"], 1.0)
        self.assertEqual(result["groups"]["held_out"]["auroc_vs_good"], 1.0)
        self.assertEqual(result["counts"]["by_group"], {"good": 2, "known": 1, "held_out": 1})
        self.assertEqual(result["per_defect_type"]["crack"]["group"], "held_out")
        self.assertEqual(result["image"]["undefined"], {})

    def test_missed_held_out_defect_and_false_alarm(self):
        records = [
            record("g0", "good", 0.9, square(1.0)),  # false alarm with a wrong mask
            record("g1", "good", 0.1, square(0.0)),
            record("k0", "oil", 0.8, square(1.0)),
            record("h0", "crack", 0.2, square(0.0), held_out=True),  # missed entirely
        ]
        result = evaluate_records(records)
        self.assertEqual(result["groups"]["known"]["recall"], 1.0)
        self.assertEqual(result["groups"]["held_out"]["recall"], 0.0)
        self.assertEqual(result["groups"]["good"]["false_positive_rate"], 0.5)
        self.assertEqual(result["groups"]["good"]["mean_dice"], 0.5)
        self.assertEqual(result["per_defect_type"]["crack"]["mean_dice"], 0.0)
        rows = {r["sample_id"]: r for r in prediction_rows(records)}
        self.assertEqual(rows["h0"]["correct"], 0)
        self.assertEqual(rows["g1"]["correct"], 1)

    def test_baseline_without_held_out_reports_null_with_reason(self):
        records = [
            record("g0", "good", 0.1, square(0.0)),
            record("k0", "oil", 0.9, square(1.0)),
        ]
        held_out = evaluate_records(records)["groups"]["held_out"]
        self.assertEqual(held_out["images"], 0)
        self.assertIsNone(held_out["recall"])
        self.assertIn("recall", held_out["undefined"])

    def test_only_good_images_leaves_ranking_metrics_undefined(self):
        result = evaluate_records([record("g0", "good", 0.1, square(0.0))])
        self.assertIsNone(result["image"]["auroc"])
        self.assertEqual(result["image"]["undefined"]["auroc"], "no positive samples")
        self.assertIsNone(result["pixel"]["auroc"])

    def test_requires_calibrated_predictions(self):
        uncalibrated = EvaluationRecord(
            "g0", "good", 0, False, raw(0.1, square(0.0)), np.zeros(SHAPE, np.uint8)
        )
        with self.assertRaisesRegex(ValueError, "calibrat"):
            evaluate_records([uncalibrated])


if __name__ == "__main__":
    unittest.main()
