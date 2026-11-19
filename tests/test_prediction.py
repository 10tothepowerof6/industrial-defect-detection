import unittest

import numpy as np

from industrial_defect_detection.inference.contracts import Prediction, RawPrediction


class PredictionTests(unittest.TestCase):
    def raw(self, **changes):
        values = {
            "image_score": 0.7,
            "score_map": np.array([[0.2, 0.5], [0.9, 0.1]], dtype=np.float32),
            "category": "tile",
            "model_id": "patchcore",
            "run_id": "test-run",
        }
        values.update(changes)
        return RawPrediction(**values)

    def test_raw_prediction_has_no_calibrated_decision(self):
        raw = self.raw()
        self.assertFalse(hasattr(raw, "is_defective"))
        self.assertFalse(hasattr(raw, "pred_mask"))

    def test_threshold_ties_are_normal_and_mask_is_boolean(self):
        prediction = self.raw(image_score=0.5).with_thresholds(0.5, 0.5)
        self.assertIsInstance(prediction, Prediction)
        self.assertFalse(prediction.is_defective)
        self.assertEqual(prediction.pred_mask.dtype, np.bool_)
        np.testing.assert_array_equal(prediction.pred_mask, [[False, False], [True, False]])

    def test_image_decision_is_independent_of_pixel_threshold(self):
        prediction = self.raw().with_thresholds(0.6, 1.0)
        self.assertTrue(prediction.is_defective)
        self.assertFalse(prediction.pred_mask.any())

    def test_rejects_invalid_maps(self):
        for value in (
            np.ones((2, 2), dtype=np.float64),
            np.ones((2, 2, 1), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.array([[np.nan]], dtype=np.float32),
            np.array([[np.inf]], dtype=np.float32),
        ):
            with self.subTest(shape=value.shape):
                with self.assertRaises(ValueError):
                    self.raw(score_map=value)

    def test_rejects_invalid_ids_and_scores(self):
        for changes in (
            {"image_score": np.nan},
            {"image_score": True},
            {"category": "unknown"},
            {"model_id": "unknown"},
            {"run_id": " "},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.raw(**changes)

    def test_rejects_nonfinite_or_boolean_thresholds(self):
        for image, pixel in ((np.inf, 0.5), (0.5, np.nan), (True, 0.5)):
            with self.assertRaises(ValueError):
                self.raw().with_thresholds(image, pixel)


if __name__ == "__main__":
    unittest.main()
