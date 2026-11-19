import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from industrial_defect_detection.data.audit import audit_dataset
from industrial_defect_detection.data.manifests import (
    build_manifest,
    manifest_files,
    read_manifest,
    validate_manifest,
)
from industrial_defect_detection.data.mvtec import (
    dataset_path,
    discover_samples,
    load_sample,
)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        serial = 0
        for category, known, held_out in (
            ("tile", "glue_strip", "crack"),
            ("metal_nut", "bent", "scratch"),
        ):
            for split, group, count in (
                ("train", "good", 10),
                ("test", "good", 2),
                ("test", known, 5),
                ("test", held_out, 5),
            ):
                folder = self.root / category / split / group
                folder.mkdir(parents=True)
                for i in range(count):
                    serial += 1
                    image = np.full((8, 12, 3), serial, dtype=np.uint8)
                    Image.fromarray(image).save(folder / f"{i:03}.png")
                    if group != "good":
                        masks = self.root / category / "ground_truth" / group
                        masks.mkdir(parents=True, exist_ok=True)
                        mask = np.zeros((8, 12), dtype=np.uint8)
                        mask[2:6, 3:9] = 255
                        Image.fromarray(mask).save(masks / f"{i:03}_mask.png")
        self.samples = discover_samples(self.root)

    def test_good_mask_and_defect_mask_resize(self):
        good = next(s for s in self.samples if not s.is_anomalous)
        defective = next(s for s in self.samples if s.is_anomalous)
        image, mask = load_sample(self.root, good)
        self.assertEqual(image.shape, (8, 12, 3))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertFalse(mask.any())
        image, mask = load_sample(self.root, defective, (16, 24))
        self.assertEqual(image.shape, (16, 24, 3))
        self.assertEqual(mask.shape, (16, 24))
        self.assertEqual(set(np.unique(mask)), {0, 1})

    def test_missing_and_misaligned_masks_are_reported(self):
        sample = next(s for s in self.samples if s.is_anomalous)
        mask_path = self.root / sample.mask_path
        Image.new("L", (2, 2), 255).save(mask_path)
        report = audit_dataset(self.root)
        self.assertFalse(report["passed"])
        self.assertIn("mismatch", report["errors"][0]["error"])
        mask_path.unlink()
        self.assertTrue(audit_dataset(self.root)["errors"])

    def test_corrupt_image_and_orphan_mask_are_reported(self):
        (self.root / self.samples[0].image_path).write_bytes(b"not an image")
        orphan = self.root / "tile/ground_truth/crack/orphan_mask.png"
        Image.new("L", (12, 8), 255).save(orphan)
        report = audit_dataset(self.root)
        self.assertEqual(len(report["errors"]), 2)
        self.assertFalse(report["passed"])

    def test_duplicate_decoded_pixels_are_reported(self):
        first = self.root / self.samples[0].image_path
        second = self.root / self.samples[-1].image_path
        with Image.open(first) as image:
            image.save(second, compress_level=1)
        report = audit_dataset(self.root)
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["duplicate_image_groups"]), 1)

    def test_split_is_deterministic_and_held_out_is_test_only(self):
        rows = build_manifest(self.samples, "tile", "custom_comparison", 42, "crack")
        self.assertEqual(
            rows, build_manifest(self.samples, "tile", "custom_comparison", 42, "crack")
        )
        self.assertEqual(len(rows), 22)
        self.assertEqual(len({r["sample_id"] for r in rows}), len(rows))
        self.assertTrue(all(r["split"] == "test" for r in rows if r["defect_type"] == "crack"))
        good = [r for r in rows if r["defect_type"] == "good"]
        self.assertEqual(sum(r["split"] == "train" for r in good), 8)
        self.assertEqual(sum(r["split"] == "val" for r in good), 2)
        self.assertEqual(sum(r["split"] == "test" for r in good), 2)

    def test_official_test_and_normal_assignments_match_custom(self):
        official = build_manifest(self.samples, "tile", "official_ad", 42)
        custom = build_manifest(self.samples, "tile", "custom_comparison", 42, "crack")
        self.assertTrue(all(r["split"] == "test" for r in official if "/test/" in r["sample_id"]))
        assignments = {r["sample_id"]: r["split"] for r in official}
        self.assertTrue(
            all(
                assignments[r["sample_id"]] == r["split"]
                for r in custom
                if r["defect_type"] == "good"
            )
        )

    def test_validator_rejects_held_out_leakage_and_duplicate_ids(self):
        rows = build_manifest(self.samples, "tile", "custom_comparison", 42, "crack")
        bad = copy.deepcopy(rows)
        next(r for r in bad if r["defect_type"] == "crack")["split"] = "train"
        with self.assertRaisesRegex(ValueError, "leaked"):
            validate_manifest(bad)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_manifest(rows + [rows[0]])

    def test_manifest_roundtrip_and_hash_are_reproducible(self):
        rows = build_manifest(self.samples, "tile", "custom_comparison", 42, "crack")
        audit = audit_dataset(self.root)
        first = manifest_files(rows, 42, audit["inventory"])
        self.assertEqual(first, manifest_files(rows, 42, audit["inventory"]))
        csv_path = self.root / "manifest.csv"
        csv_path.write_bytes(first[0])
        self.assertEqual(rows, read_manifest(csv_path, self.root))

    def test_dataset_path_rejects_escape(self):
        with self.assertRaises(ValueError):
            dataset_path(self.root, "../outside.png")

    def test_invalid_held_out_and_small_groups_fail(self):
        with self.assertRaisesRegex(ValueError, "existing held-out"):
            build_manifest(self.samples, "tile", "custom_comparison", 42, "scratch")
        with self.assertRaisesRegex(ValueError, "Too few"):
            one_good = next(
                s for s in self.samples if s.category == "metal_nut" and s.source_split == "train"
            )
            build_manifest([one_good], "metal_nut", "official_ad", 42)


if __name__ == "__main__":
    unittest.main()
