import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from industrial_defect_detection.demo import (
    default_comparison,
    display_range,
    find_artifacts,
    ground_truth_mask,
    inspect,
    read_image,
    result_panels,
    sample_images,
)
from industrial_defect_detection.inference.contracts import RawPrediction
from industrial_defect_detection.inference.loader import load_predictor

ROOT = Path(__file__).resolve().parents[1]
CRACK_IMAGE = ROOT / "datasets" / "tile" / "test" / "crack" / "000.png"
SCRATCH_IMAGE = ROOT / "datasets" / "metal_nut" / "test" / "scratch" / "000.png"


def write_metadata(root, protocol, model_id, category, run_id, calibrated, border=None):
    folder = root / protocol / model_id / category / run_id
    folder.mkdir(parents=True)
    meta = {
        "model_id": model_id,
        "category": category,
        "protocol": protocol,
        "fold": "holdout-crack",
        "run_id": run_id,
    }
    if border is not None:
        meta["score"] = {"image_score_border_px": border}
    if calibrated:
        meta.update(status="calibrated", calibration={"image_threshold": 1.0})
    (folder / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")


class CatalogTests(unittest.TestCase):
    def test_find_artifacts_reads_identity_and_calibration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_metadata(root, "custom_comparison", "unet_resnet18", "tile", "r1", True)
            write_metadata(root, "custom_comparison", "patchcore", "tile", "r2", False)
            write_metadata(root, "custom_comparison", "efficientad", "tile", "r3", True)
            (root / "backbones").mkdir()
            found = {a.run_id: a for a in find_artifacts(root)}
        self.assertEqual(set(found), {"r1", "r2"})  # no loader for efficientad yet
        self.assertTrue(found["r1"].calibrated)
        self.assertFalse(found["r2"].calibrated)
        self.assertIn("U-Net", found["r1"].label)

    def test_known_issue_only_for_unet_metal_nut(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_metadata(root, "c", "unet_resnet18", "metal_nut", "a", True)
            write_metadata(root, "c", "unet_resnet18", "tile", "b", True)
            write_metadata(root, "c", "patchcore", "metal_nut", "c", True)
            write_metadata(root, "c", "unet_resnet18", "metal_nut", "a-border8", True, 8)
            issues = {a.run_id: a.known_issue for a in find_artifacts(root)}
        self.assertEqual(
            issues, {"a": "unet_metal_nut_border", "a-border8": None, "b": None, "c": None}
        )

    def test_versions_and_default_comparison_prefer_v2(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_metadata(root, "c", "unet_resnet18", "metal_nut", "a", True)
            write_metadata(root, "c", "unet_resnet18", "metal_nut", "a-border8", True, 8)
            write_metadata(root, "c", "patchcore", "metal_nut", "p", True)
            artifacts = find_artifacts(root)
        versions = {a.run_id: a.score_version for a in artifacts}
        self.assertEqual(versions, {"a": "v1", "a-border8": "v2", "p": None})
        self.assertEqual([a.run_id for a in default_comparison(artifacts)], ["a-border8", "p"])
        v1_only = [a for a in artifacts if a.run_id != "a-border8"]
        self.assertEqual([a.run_id for a in default_comparison(v1_only)], ["a", "p"])


class ImageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for group in ("good", "crack"):
            folder = self.root / "tile" / "test" / group
            folder.mkdir(parents=True)
            Image.fromarray(np.zeros((4, 6, 3), np.uint8)).save(folder / "000.png")
        masks = self.root / "tile" / "ground_truth" / "crack"
        masks.mkdir(parents=True)
        mask = np.zeros((4, 6), np.uint8)
        mask[1:3, 2:4] = 255
        Image.fromarray(mask).save(masks / "000_mask.png")

    def test_samples_list_good_first(self):
        groups = sample_images(self.root, "tile")
        self.assertEqual(list(groups), ["good", "crack"])
        self.assertEqual(sample_images(self.root, "metal_nut"), {})

    def test_ground_truth_only_for_defect_test_images(self):
        mask = ground_truth_mask(self.root / "tile" / "test" / "crack" / "000.png")
        self.assertEqual(int(mask.sum()), 4)
        self.assertIsNone(ground_truth_mask(self.root / "tile" / "test" / "good" / "000.png"))
        self.assertIsNone(ground_truth_mask(self.root / "elsewhere.png"))

    def test_read_image_converts_and_rejects_corrupt_files(self):
        buffer = io.BytesIO()
        Image.fromarray(np.zeros((3, 5), np.uint8)).save(buffer, format="PNG")  # grayscale
        buffer.seek(0)
        image = read_image(buffer)
        self.assertEqual((image.shape, image.dtype), ((3, 5, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, "Cannot read image"):
            read_image(io.BytesIO(b"not an image"))

    def test_panels_keep_original_size_and_fixed_display_range(self):
        score_map = np.zeros((4, 6), np.float32)
        score_map[1:3, 2:4] = 0.9
        prediction = RawPrediction(0.9, score_map, "tile", "unet_resnet18", "r").with_thresholds(
            0.5, 0.25
        )
        self.assertEqual(display_range(prediction), (0.0, 0.5))
        panels = result_panels(np.zeros((4, 6, 3), np.uint8), prediction, np.ones((4, 6), bool))
        self.assertEqual(list(panels), ["image", "ground_truth", "heatmap", "mask"])
        self.assertTrue(all(p.shape == (4, 6, 3) for p in panels.values()))
        self.assertEqual(
            list(result_panels(panels["image"], prediction)), ["image", "heatmap", "mask"]
        )


class AppMatchesCliTests(unittest.TestCase):
    """The demo must give the same decision as scripts/predict.py --calibrated."""

    def test_every_local_calibrated_artifact(self):
        artifacts = [a for a in find_artifacts(ROOT / "artifacts") if a.calibrated]
        images = {"tile": CRACK_IMAGE, "metal_nut": SCRATCH_IMAGE}
        if not artifacts or not all(p.is_file() for p in images.values()):
            self.skipTest("Needs local calibrated artifacts and the MVTec dataset")
        for artifact in artifacts:
            with self.subTest(artifact=artifact.label), tempfile.TemporaryDirectory() as folder:
                image_path = images[artifact.category]
                out = Path(folder) / "cli"
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "predict.py"),
                        "--artifact-dir",
                        str(artifact.path),
                        "--image",
                        str(image_path),
                        "--output-dir",
                        str(out),
                        "--device",
                        "cpu",
                        "--calibrated",
                    ],
                    check=True,
                    capture_output=True,
                    cwd=ROOT,
                )
                with (out / "predictions.csv").open(encoding="utf-8") as file:
                    cli = next(csv.DictReader(file))
                prediction, _ = inspect(
                    load_predictor(artifact.path, "cpu"), read_image(image_path)
                )
                self.assertEqual(int(prediction.is_defective), int(cli["is_defective"]))
                self.assertAlmostEqual(prediction.image_score, float(cli["image_score"]), places=5)
                self.assertEqual(prediction.image_threshold, float(cli["image_threshold"]))


def load_app_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "streamlit_app", ROOT / "app" / "streamlit_app.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # main() only runs under "streamlit run"
    return module


class AppTextTests(unittest.TestCase):
    def test_both_languages_have_the_same_texts_and_names(self):
        app = load_app_module()
        self.assertEqual(set(app.TEXT["vi"]), set(app.TEXT["en"]))
        self.assertEqual(set(app.NAMES["vi"]), set(app.NAMES["en"]))
        self.assertEqual(app.NAMES["vi"]["tile"], "gạch ốp")

    def test_every_dataset_defect_type_has_a_name(self):
        app = load_app_module()
        for category in ("tile", "metal_nut"):
            groups = sample_images(ROOT / "datasets", category)
            if not groups:
                self.skipTest("Needs the MVTec dataset")
            self.assertLessEqual(set(groups), set(app.NAMES["vi"]), category)


class StreamlitAppTests(unittest.TestCase):
    def test_app_runs_in_both_languages_and_inspects_a_sample(self):
        from streamlit.testing.v1 import AppTest

        if not any(
            a.calibrated and a.category == "tile" for a in find_artifacts(ROOT / "artifacts")
        ):
            self.skipTest("Needs a local calibrated tile artifact")
        app = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180)
        app.run()
        self.assertFalse(app.exception)
        self.assertIn("Kiểm tra lỗi", app.title[0].value)
        self.assertIn("tile (gạch ốp)", app.sidebar.radio[1].options)
        defect_options = app.sidebar.selectbox[1].options
        self.assertIn("good (ảnh tốt)", defect_options)
        self.assertIn("crack (vết nứt) · lỗi giữ riêng, chưa thấy khi train", defect_options)
        app.sidebar.radio[0].set_value("en").run()
        self.assertIn("defect inspection", app.title[0].value)
        app.sidebar.button[0].click().run()
        self.assertFalse(app.exception)
        verdicts = [e.value for e in (*app.success, *app.error)]
        self.assertTrue(any("PASS" in v or "DEFECT" in v for v in verdicts), verdicts)

    def test_compare_mode_runs_the_default_pair(self):
        from streamlit.testing.v1 import AppTest

        tile = [
            a for a in find_artifacts(ROOT / "artifacts") if a.calibrated and a.category == "tile"
        ]
        if len({a.model_id for a in tile}) < 2:
            self.skipTest("Needs calibrated U-Net and PatchCore tile artifacts")
        app = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180)
        app.run()
        app.sidebar.radio[2].set_value("compare").run()  # language, product, mode, source
        self.assertEqual(len(app.sidebar.multiselect[0].value), 2)
        app.sidebar.button[0].click().run()
        self.assertFalse(app.exception)
        headers = [h.value for h in app.subheader]
        self.assertEqual(len(headers), 2)
        self.assertTrue(any(h.startswith("U-Net") for h in headers), headers)
        self.assertIn("PatchCore", headers)


if __name__ == "__main__":
    unittest.main()
