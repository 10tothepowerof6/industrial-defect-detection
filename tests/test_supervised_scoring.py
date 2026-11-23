import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from industrial_defect_detection.models.supervised.scoring import (
    ImageTooSmallError,
    border_max,
    border_table,
    check_image_size,
    choose_border,
    eligible_borders,
    image_score_border,
)

ROOT = Path(__file__).resolve().parents[1]
TILE_V1 = (
    ROOT / "artifacts/custom_comparison/unet_resnet18/tile/20261121-150943-seed42-holdout-crack"
)


def load_derive():
    path = ROOT / "scripts" / "derive_unet_score_variant.py"
    spec = importlib.util.spec_from_file_location("derive_unet_score_variant", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.derive


def good_map(corner=0.25, background=0.04):
    score_map = np.full((128, 128), background, np.float32)
    score_map[0, -1] = corner  # the padding artifact seen on metal_nut
    return score_map


def defect_map(peak=0.99):
    score_map = good_map()
    score_map[60:64, 60:64] = peak
    return score_map


class BorderMaxTests(unittest.TestCase):
    def test_corner_spike_is_ignored_only_with_a_border(self):
        self.assertAlmostEqual(border_max(good_map(), 0), 0.25)
        self.assertAlmostEqual(border_max(good_map(), 1), 0.04)
        self.assertAlmostEqual(border_max(defect_map(), 8), 0.99)

    def test_invalid_borders(self):
        with self.assertRaises(ValueError):
            border_max(good_map(), -1)
        with self.assertRaises(ValueError):
            border_max(good_map(), 64)  # nothing left of a 128x128 map

    def test_image_size_limit_for_a_border(self):
        check_image_size(65, 65, 32)  # one pixel left on each axis
        check_image_size(64, 64, 0)  # v1 has no limit
        for height, width in ((64, 64), (64, 128), (128, 64), (10, 10)):
            with self.subTest(size=(height, width)):
                with self.assertRaises(ImageTooSmallError) as caught:
                    check_image_size(height, width, 32)
                self.assertEqual((caught.exception.height, caught.exception.width), (height, width))
                self.assertIn("exceed 64 px", str(caught.exception))

    def test_metadata_border_defaults_to_v1(self):
        self.assertEqual(image_score_border({}), 0)
        self.assertEqual(image_score_border({"score": {"image_score_border_px": 8}}), 8)
        for bad in (-1, 2.5, True, "8"):
            with self.assertRaises(ValueError):
                image_score_border({"score": {"image_score_border_px": bad}})


class SelectionTests(unittest.TestCase):
    def test_smallest_border_on_the_plateau_is_chosen(self):
        table = border_table([good_map() for _ in range(10)], [defect_map() for _ in range(3)])
        q95 = {row["border"]: row["good_q95"] for row in table}
        self.assertAlmostEqual(q95[0], 0.25)
        self.assertAlmostEqual(q95[4], 0.04)
        self.assertEqual(eligible_borders(table), [4, 8, 16, 32])
        self.assertEqual(choose_border({"metal_nut": table}), 4)

    def test_no_border_needed_when_there_is_no_artifact(self):
        clean = [good_map(corner=0.04) for _ in range(10)]
        table = border_table(clean, [defect_map() for _ in range(3)])
        self.assertEqual(choose_border({"tile": table}), 0)

    def test_choice_must_hold_in_every_category(self):
        table_a = [
            {"border": 0, "good_q95": 0.30, "auroc": 1.0},
            {"border": 4, "good_q95": 0.05, "auroc": 1.0},
            {"border": 8, "good_q95": 0.04, "auroc": 1.0},
        ]
        table_b = [
            {"border": 0, "good_q95": 0.20, "auroc": 0.9},
            {"border": 4, "good_q95": 0.04, "auroc": 0.8},  # separation got worse
            {"border": 8, "good_q95": 0.04, "auroc": 0.95},
        ]
        self.assertEqual(eligible_borders(table_a), [8])  # 0.05 is 25% above 0.04
        self.assertEqual(eligible_borders(table_b), [8])
        self.assertEqual(choose_border({"a": table_a, "b": table_b}), 8)
        with self.assertRaises(ValueError):
            choose_border(
                {
                    "a": table_a,
                    "c": [dict(r, auroc=0.1) for r in table_b if r["border"]] + [table_b[0]],
                }
            )


class DeriveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.v1 = Path(self.temporary.name) / "run-v1"
        self.v1.mkdir()
        (self.v1 / "model.ckpt").write_bytes(b"weights")
        self.meta = {
            "model_id": "unet_resnet18",
            "run_id": "run-v1",
            "checkpoint": "model.ckpt",
            "status": "calibrated",
            "calibration": {"image_threshold": 0.25},
            "score": {"image_score": "max of the probability map"},
        }
        (self.v1 / "metadata.json").write_text(json.dumps(self.meta), encoding="utf-8")
        self.derive = load_derive()

    def test_v2_reuses_the_checkpoint_and_needs_new_calibration(self):
        target = self.derive(self.v1, 8)
        meta = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(target.name, "run-v1-border8")
        self.assertEqual(meta["run_id"], "run-v1-border8")
        self.assertEqual(meta["status"], "uncalibrated")
        self.assertNotIn("calibration", meta)
        self.assertEqual(meta["score"]["image_score_border_px"], 8)
        self.assertEqual(meta["derived_from"]["run_id"], "run-v1")
        self.assertEqual((target / "model.ckpt").read_bytes(), b"weights")
        # v1 is untouched.
        self.assertEqual(json.loads((self.v1 / "metadata.json").read_text()), self.meta)

    def test_refuses_variants_zero_border_and_overwrite(self):
        target = self.derive(self.v1, 8)
        with self.assertRaisesRegex(ValueError, "original v1"):
            self.derive(target, 4)
        with self.assertRaisesRegex(ValueError, "positive"):
            self.derive(self.v1, 0)
        with self.assertRaises(FileExistsError):
            self.derive(self.v1, 8)

    def test_real_checkpoint_same_map_different_image_score(self):
        if not TILE_V1.is_dir():
            self.skipTest("Needs the U-Net tile v1 artifact")
        from industrial_defect_detection.inference.loader import load_predictor

        copy = Path(self.temporary.name) / "copy" / TILE_V1.name
        copy.mkdir(parents=True)
        for name in ("metadata.json", "model.ckpt"):
            shutil.copy2(TILE_V1 / name, copy / name)
        v2 = self.derive(copy, 8)
        image = np.full((96, 128, 3), 120, np.uint8)
        first = load_predictor(copy, "cpu").predict(image)
        second = load_predictor(v2, "cpu").predict(image)
        np.testing.assert_array_equal(first.score_map, second.score_map)
        self.assertAlmostEqual(first.image_score, float(first.score_map.max()))
        self.assertAlmostEqual(second.image_score, border_max(second.score_map, 8))
        self.assertEqual(second.run_id, f"{TILE_V1.name}-border8")

    def real_v1_and_v2(self, border):
        if not TILE_V1.is_dir():
            self.skipTest("Needs the U-Net tile v1 artifact")
        copy = Path(self.temporary.name) / "copy" / TILE_V1.name
        copy.mkdir(parents=True)
        for name in ("metadata.json", "model.ckpt"):
            shutil.copy2(TILE_V1 / name, copy / name)
        return copy, self.derive(copy, border)

    def test_small_images_are_refused_before_inference(self):
        # Regression (PR #22 review): 64x64 and 64x128 images crashed a border=32 artifact.
        from industrial_defect_detection.inference.loader import load_predictor

        v1, v2 = self.real_v1_and_v2(32)
        predictor = load_predictor(v2, "cpu")
        model = predictor.model
        predictor.model = None  # any inference attempt would fail with a different error
        for height, width in ((64, 64), (64, 128)):
            with self.subTest(size=(height, width)):
                with self.assertRaises(ImageTooSmallError):
                    predictor.predict(np.zeros((height, width, 3), np.uint8))
        predictor.model = model
        result = predictor.predict(np.zeros((65, 65, 3), np.uint8))
        self.assertAlmostEqual(result.image_score, border_max(result.score_map, 32))
        # v1 keeps scoring small images over the whole map.
        small = load_predictor(v1, "cpu").predict(np.zeros((64, 64, 3), np.uint8))
        self.assertAlmostEqual(small.image_score, float(small.score_map.max()))


if __name__ == "__main__":
    unittest.main()
