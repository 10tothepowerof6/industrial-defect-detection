import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from industrial_defect_detection.evaluation.checks import (
    EvaluationRecord,
    check_evaluation_manifest,
    check_ground_truth,
    check_record_set,
    check_same_test_ids,
    fold_name,
    load_development_manifest,
    resize_score_map,
)
from industrial_defect_detection.inference.contracts import RawPrediction

SPLITS = Path(__file__).resolve().parents[1] / "splits" / "seed42"


def raw(score_map=None, **changes):
    values = {
        "image_score": 0.7,
        "score_map": np.zeros((4, 6), dtype=np.float32) if score_map is None else score_map,
        "category": "tile",
        "model_id": "unet_resnet18",
        "run_id": "synthetic-run",
    }
    values.update(changes)
    return RawPrediction(**values)


def row(sample_id="tile/test/crack/000.png", defect="crack", held_out="crack"):
    return {
        "sample_id": sample_id,
        "category": "tile",
        "defect_type": defect,
        "is_anomalous": int(defect != "good"),
        "held_out_defect": held_out,
    }


def defect_mask(shape=(4, 6)):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[1:3, 2:4] = 1
    return mask


class ResizeTests(unittest.TestCase):
    def test_resize_keeps_float_values_and_orientation(self):
        source = np.zeros((2, 4), dtype=np.float32)
        source[:, 2:] = 1.0
        resized = resize_score_map(source, 4, 8)
        self.assertEqual(resized.shape, (4, 8))
        self.assertEqual(resized.dtype, np.float32)
        # Left side stays low and right side stays high: the map is not transposed.
        self.assertLess(resized[:, 0].max(), 0.01)
        self.assertGreater(resized[:, -1].min(), 0.99)
        # Bilinear interpolation produces in-between values, not 8-bit steps.
        self.assertTrue(((resized > 0.01) & (resized < 0.99)).any())

    def test_same_size_returns_copy(self):
        source = np.ones((3, 3), dtype=np.float32)
        resized = resize_score_map(source, 3, 3)
        resized[0, 0] = 5
        self.assertEqual(source[0, 0], 1)

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            resize_score_map(np.ones((2, 2, 1), dtype=np.float32), 4, 4)
        with self.assertRaises(ValueError):
            resize_score_map(np.ones((2, 2), dtype=np.float32), 0, 4)


class GroundTruthTests(unittest.TestCase):
    def test_accepts_binary_uint8_and_bool(self):
        self.assertEqual(check_ground_truth(defect_mask(), 1).dtype, np.bool_)
        self.assertFalse(check_ground_truth(np.zeros((2, 2), dtype=bool), 0).any())

    def test_rejects_label_mask_disagreement(self):
        with self.assertRaisesRegex(ValueError, "empty ground-truth"):
            check_ground_truth(np.zeros((4, 6), dtype=np.uint8), 1, "x")
        with self.assertRaisesRegex(ValueError, "Good image"):
            check_ground_truth(defect_mask(), 0, "x")

    def test_rejects_unbinarized_or_wrong_type(self):
        for mask, label in (
            (defect_mask() * 255, 1),
            (defect_mask().astype(np.float32), 1),
            (np.zeros((2, 2, 1), dtype=np.uint8), 0),
        ):
            with self.subTest(dtype=mask.dtype, shape=mask.shape):
                with self.assertRaises(ValueError):
                    check_ground_truth(mask, label)
        with self.assertRaises(ValueError):
            check_ground_truth(defect_mask(), True)


class RecordTests(unittest.TestCase):
    def test_groups_from_manifest(self):
        held = EvaluationRecord.from_manifest_row(row(), raw(), defect_mask())
        known = EvaluationRecord.from_manifest_row(
            row("tile/test/oil/000.png", "oil"), raw(), defect_mask()
        )
        good = EvaluationRecord.from_manifest_row(
            row("tile/test/good/000.png", "good"), raw(), np.zeros((4, 6), np.uint8)
        )
        self.assertEqual((held.group, known.group, good.group), ("held_out", "known", "good"))
        self.assertEqual(held.gt_mask.dtype, np.bool_)

    def test_baseline_manifest_has_no_held_out_group(self):
        record = EvaluationRecord.from_manifest_row(row(held_out=""), raw(), defect_mask())
        self.assertEqual(record.group, "known")

    def test_shape_mismatch_explains_resize(self):
        with self.assertRaisesRegex(ValueError, "resize_score_map"):
            EvaluationRecord.from_manifest_row(
                row(), raw(np.zeros((2, 3), np.float32)), defect_mask()
            )

    def test_category_mismatch(self):
        with self.assertRaisesRegex(ValueError, "category"):
            EvaluationRecord.from_manifest_row(row(), raw(category="metal_nut"), defect_mask())


class RecordSetTests(unittest.TestCase):
    def records(self, *predictions):
        return [
            EvaluationRecord(f"tile/test/oil/{i:03}.png", "oil", 1, False, p, defect_mask())
            for i, p in enumerate(predictions)
        ]

    def test_identity_and_calibration_state(self):
        info = check_record_set(self.records(raw(), raw()))
        self.assertEqual(info["model_id"], "unet_resnet18")
        self.assertFalse(info["calibrated"])
        with self.assertRaisesRegex(ValueError, "calibrat"):
            check_record_set(self.records(raw()), require_calibrated=True)
        calibrated = raw().with_thresholds(0.5, 0.5)
        self.assertTrue(check_record_set(self.records(calibrated, calibrated))["calibrated"])

    def test_rejects_mixed_runs_states_and_thresholds(self):
        for predictions in (
            (raw(), raw(run_id="other")),
            (raw(), raw(model_id="patchcore")),
            (raw(), raw().with_thresholds(0.5, 0.5)),
            (raw().with_thresholds(0.5, 0.5), raw().with_thresholds(0.6, 0.5)),
        ):
            with self.subTest(predictions=predictions):
                with self.assertRaises(ValueError):
                    check_record_set(self.records(*predictions))

    def test_rejects_duplicates_and_empty(self):
        record = self.records(raw())[0]
        with self.assertRaises(ValueError):
            check_record_set([record, record])
        with self.assertRaises(ValueError):
            check_record_set([])

    def test_same_test_ids(self):
        check_same_test_ids(["a", "b"], ["b", "a"])
        with self.assertRaisesRegex(ValueError, "only in first"):
            check_same_test_ids(["a", "b"], ["a"])


def read_rows(name):
    with (SPLITS / name).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


class EvaluationManifestTests(unittest.TestCase):
    """The evaluated test images must never include the artifact's train/val images."""

    def setUp(self):
        self.custom = read_rows("custom_comparison_tile_crack.csv")
        self.official = read_rows("official_ad_tile_baseline.csv")
        self.unet = {"fold": "holdout-crack"}

    def test_fold_name(self):
        self.assertEqual(fold_name(""), "baseline")
        self.assertEqual(fold_name("crack"), "holdout-crack")

    def test_training_manifest_is_accepted(self):
        check_evaluation_manifest(self.unet, self.custom, self.custom)

    def test_official_manifest_is_refused_for_custom_unet(self):
        # Regression: the official tile test split contains the 38 train and 12 val defect
        # images that U-Net (custom, holdout-crack) learned from or used to pick a checkpoint.
        with self.assertRaisesRegex(ValueError, "allow-cross-manifest"):
            check_evaluation_manifest(self.unet, self.custom, self.official)
        with self.assertRaisesRegex(ValueError, "50 test images were used"):
            check_evaluation_manifest(self.unet, self.custom, self.official, True)

    def test_cross_manifest_without_overlap_is_allowed_when_requested(self):
        # An anomaly model fitted on official normal train may be reused on the custom fold:
        # normal train/val are identical, so no custom test image is development data.
        anomaly = {"fold": "baseline"}
        with self.assertRaises(ValueError):
            check_evaluation_manifest(anomaly, self.official, self.custom)
        check_evaluation_manifest(anomaly, self.official, self.custom, True)

    def test_category_and_metadata_fold_mismatch(self):
        metal_nut = read_rows("custom_comparison_metal_nut_scratch.csv")
        with self.assertRaisesRegex(ValueError, "trained on tile"):
            check_evaluation_manifest(self.unet, self.custom, metal_nut, True)
        with self.assertRaisesRegex(ValueError, "metadata says fold holdout-oil"):
            check_evaluation_manifest({"fold": "holdout-oil"}, self.custom, self.custom)
        check_evaluation_manifest({"fold": "smoke-holdout-crack"}, self.custom, self.custom)

    def test_development_manifest_must_be_recorded_and_unchanged(self):
        with self.assertRaisesRegex(ValueError, "training_manifest"):
            load_development_manifest({}, Path("datasets"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.csv"
            path.write_text("changed\n", encoding="utf-8")
            record = {"training_manifest": {"path": str(path), "sha256": "0" * 64}}
            with self.assertRaisesRegex(ValueError, "changed since"):
                load_development_manifest(record, Path(folder))
            record["training_manifest"]["path"] = str(Path(folder) / "missing.csv")
            with self.assertRaises(FileNotFoundError):
                load_development_manifest(record, Path(folder))


if __name__ == "__main__":
    unittest.main()
