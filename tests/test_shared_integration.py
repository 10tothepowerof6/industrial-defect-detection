import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from industrial_defect_detection.evaluation.checks import check_calibration_manifest
from industrial_defect_detection.inference.loader import load_predictor
from industrial_defect_detection.runs import file_sha256, git_state
from industrial_defect_detection.visualization.overlay import panel


class SharedIntegrationTests(unittest.TestCase):
    def test_calibration_rejects_changed_manifest_before_loading_images(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "split.csv"
            manifest.write_text("frozen manifest")
            metadata = {"training_manifest": {"sha256": file_sha256(manifest)}}
            check_calibration_manifest(metadata, manifest)
            manifest.write_text("changed normal train/val assignments")
            with self.assertRaisesRegex(ValueError, "Calibration manifest differs"):
                check_calibration_manifest(metadata, manifest)
            with self.assertRaises(ValueError):
                check_calibration_manifest({}, manifest)

    def test_loader_reports_missing_unknown_and_routes_unet(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(FileNotFoundError):
                load_predictor(root)
            (root / "metadata.json").write_text(json.dumps({"model_id": "unsupported"}))
            with self.assertRaisesRegex(ValueError, "No predictor"):
                load_predictor(root)
            (root / "metadata.json").write_text(json.dumps({"model_id": "unet_resnet18"}))
            with patch(
                "industrial_defect_detection.models.supervised.adapter.UNetPredictor"
            ) as adapter:
                self.assertIs(load_predictor(root, "cpu"), adapter.return_value)
                adapter.assert_called_once_with(root, "cpu")

    def test_raw_panel_has_no_fake_mask_or_ground_truth(self):
        image = np.zeros((16, 20, 3), np.uint8)
        scores = np.zeros((16, 20), np.float32)
        raw = panel(image, scores, None, 0, 1, tile_size=32)
        calibrated = panel(image, scores, scores > 0.5, 0, 1, tile_size=32)
        self.assertEqual(raw.shape, (32, 64, 3))
        self.assertEqual(calibrated.shape, (32, 96, 3))

    def test_provenance_contains_real_commit_and_surfaces_git_errors(self):
        state = git_state()
        self.assertEqual(len(state["commit"]), 40)
        self.assertIsInstance(state["dirty"], bool)
        import subprocess

        with patch(
            "industrial_defect_detection.runs.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "git"),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                git_state()


if __name__ == "__main__":
    unittest.main()
