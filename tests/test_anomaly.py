import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from industrial_defect_detection.models.anomaly.patchcore import (
    MEAN,
    STD,
    PatchcorePredictor,
    load_pretrained,
    make_model,
    preprocess,
    sha256,
)
from industrial_defect_detection.models.anomaly.train import ROOT, validate_experiment


class AnomalyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.config = yaml.safe_load((ROOT / "configs/anomaly/patchcore_tile.yaml").read_text())
        self.rows = [
            {
                "category": "tile",
                "protocol": "custom_comparison",
                "held_out_defect": "crack",
                "split": split,
                "is_anomalous": label,
            }
            for split, label in [("train", 0), ("train", 1), ("val", 0), ("test", 1)]
        ]

    def test_only_good_train_is_selected(self):
        selected = validate_experiment(self.config, self.rows)
        self.assertEqual(selected, [self.rows[0]])

    def test_mismatched_fold_category_protocol_are_rejected(self):
        for key, value in [
            ("fold", "holdout-oil"),
            ("category", "metal_nut"),
            ("protocol", "official_ad"),
        ]:
            config = copy.deepcopy(self.config)
            config["experiment"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_experiment(config, self.rows)

    def test_preprocessing_rgb_shape_and_imagenet_normalization(self):
        params = {
            "resize": {"height": 32, "width": 48},
            "normalize_mean": MEAN,
            "normalize_std": STD,
        }
        tensor = preprocess(np.full((60, 70, 3), 255, np.uint8), params)
        self.assertEqual(tuple(tensor.shape), (3, 32, 48))
        np.testing.assert_allclose(tensor[:, 0, 0].numpy(), (1 - np.array(MEAN)) / STD, rtol=1e-6)
        with self.assertRaises(ValueError):
            preprocess(np.ones((60, 70, 3), np.float32), params)

    def test_missing_or_wrong_weights_never_download(self):
        model = make_model(self.config["model"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing.pth"
            with self.assertRaises(FileNotFoundError):
                load_pretrained(model, path)
            path.write_bytes(b"wrong checkpoint")
            with self.assertRaisesRegex(ValueError, "hash"):
                load_pretrained(model, path)

    def test_offline_checkpoint_roundtrip_and_checksum(self):
        # Random features are deliberately used only for this serialization test.
        model = make_model(self.config["model"])
        model.memory_bank = torch.randn(16, 384)
        model.eval()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            torch.save(model.state_dict(), root / "model.ckpt")
            metadata = {
                "schema_version": 1,
                "model_id": "patchcore",
                "category": "tile",
                "run_id": "test",
                "model": self.config["model"],
                "checkpoint": "model.ckpt",
                "checkpoint_sha256": sha256(root / "model.ckpt"),
                "preprocessing": {
                    "color": "RGB",
                    "resize": {"height": 32, "width": 48, "interpolation": "bilinear"},
                    "normalize_mean": MEAN,
                    "normalize_std": STD,
                },
            }
            (root / "metadata.json").write_text(json.dumps(metadata))
            image = np.full((40, 55, 3), 120, np.uint8)
            predictor = PatchcorePredictor(root)
            first = predictor.predict(image)
            second = PatchcorePredictor(root).predict(image)
            self.assertEqual(first.score_map.shape, (40, 55))
            self.assertEqual(first.image_score, second.image_score)
            np.testing.assert_array_equal(first.score_map, second.score_map)
            self.assertEqual(tuple(predictor.model.memory_bank.shape), (16, 384))
            with (root / "model.ckpt").open("ab") as file:
                file.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum"):
                PatchcorePredictor(root)


if __name__ == "__main__":
    unittest.main()
