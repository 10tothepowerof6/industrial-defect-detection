import csv
import unittest
from pathlib import Path

from industrial_defect_detection.models.supervised.train import check_fold, split_rows
from industrial_defect_detection.runs import load_config

ROOT = Path(__file__).resolve().parents[1]


def manifest_rows(name):
    with (ROOT / "splits" / "seed42" / name).open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["is_anomalous"] = int(row["is_anomalous"])
    return rows


class FoldCheckTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs" / "supervised" / "unet_resnet18_tile.yaml")
        self.rows = manifest_rows("custom_comparison_tile_crack.csv")

    def with_fold(self, fold):
        config = {**self.config, "experiment": {**self.config["experiment"], "fold": fold}}
        return config

    def test_matching_fold_trains_without_the_held_out_defect(self):
        splits = split_rows(self.rows, self.config)
        self.assertEqual(len(splits["train"]), 222)
        self.assertEqual(len(splits["val"]), 58)
        for split in splits.values():
            self.assertNotIn("crack", {r["defect_type"] for r in split})

    def test_wrong_fold_is_refused_before_training(self):
        # Regression: holdout-oil with the holdout-crack manifest would train on 10 oil images
        # while metadata claims oil was held out.
        oil_in_train = [r for r in self.rows if r["defect_type"] == "oil" and r["split"] == "train"]
        self.assertEqual(len(oil_in_train), 10)
        for fold in ("holdout-oil", "baseline", "smoke-holdout-oil"):
            with self.subTest(fold=fold):
                with self.assertRaisesRegex(ValueError, "needs 'holdout-crack'"):
                    split_rows(self.rows, self.with_fold(fold))

    def test_smoke_test_prefix_keeps_the_real_fold(self):
        # scripts/train_supervised.py --smoke-test renames the fold to smoke-<fold>.
        check_fold(self.rows, self.with_fold("smoke-holdout-crack"))


if __name__ == "__main__":
    unittest.main()
