"""Physical sphere geometry, 3D extraction and competition-test checkpoints."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml
from radiomics import featureextractor

from LIDCfeatureExtraction import main as extract_lidc
from LNDBfeatureExtraction import main as extract_lndb
from src.radiomics_extraction import check_mask_geometry, create_sphere, extract_features, load_paths, prepare_run

ROOT = Path(__file__).resolve().parents[1]


class SphereTests(unittest.TestCase):
    def test_physical_distance_with_fractional_center_spacing_and_direction(self):
        image = sitk.GetImageFromArray(np.arange(11 * 13 * 17, dtype=np.int16).reshape(11, 13, 17))
        image.SetOrigin((20., -30., 40.))
        image.SetSpacing((0.7, 1.1, 2.5))
        angle = 0.4
        image.SetDirection((np.cos(angle), -np.sin(angle), 0., np.sin(angle), np.cos(angle), 0., 0., 0., 1.))
        before = sitk.GetArrayFromImage(image)
        center = image.TransformContinuousIndexToPhysicalPoint((7.2, 6.4, 5.3))
        selected, mask = create_sphere(image, center, 3.4)
        expected = np.zeros_like(before, dtype=np.uint8)
        # Compare with world-coordinate distances on the whole voxel grid.
        for z, y, x in np.ndindex(expected.shape):
            point = image.TransformIndexToPhysicalPoint((x, y, z))
            expected[z, y, x] = np.linalg.norm(np.array(point) - center) <= 3.4
        self.assertIs(selected, image)
        check_mask_geometry(image, mask)
        self.assertEqual(mask.GetPixelID(), sitk.sitkUInt8)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(mask), expected)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(image), before)

    def test_radius_boundary_and_scan_edge_do_not_wrap(self):
        image = sitk.GetImageFromArray(np.zeros((9, 9, 9), dtype=np.int16))
        _, mask = create_sphere(image, (4., 4., 4.), 2)
        values = sitk.GetArrayFromImage(mask)
        self.assertEqual(int(values.sum()), 33)
        self.assertEqual(values[4, 4, 6], 1)
        self.assertEqual(values[4, 4, 7], 0)
        _, edge_mask = create_sphere(image, (0., 0., 0.), 2)
        z, y, x = np.ogrid[:9, :9, :9]
        np.testing.assert_array_equal(sitk.GetArrayFromImage(edge_mask), x*x + y*y + z*z <= 4)

    def test_invalid_radius_and_centroid_are_rejected(self):
        image = sitk.GetImageFromArray(np.zeros((9, 9, 9), dtype=np.int16))
        for radius in (0, -1, np.nan, np.inf):
            with self.subTest(radius=radius), self.assertRaisesRegex(ValueError, "positive finite"):
                create_sphere(image, (4., 4., 4.), radius)
        for center in ((np.nan, 4., 4.), (4., np.inf, 4.), (-1., 4., 4.), (4., 4., 9.)):
            with self.subTest(center=center), self.assertRaises(ValueError):
                create_sphere(image, center, 2)

    def test_config_matches_segmentation_without_artificial_shape(self):
        config = yaml.safe_load((ROOT / "configs/radiomics_sphere_3d.yaml").read_text())
        segmentation = yaml.safe_load((ROOT / "configs/radiomics_segmentation_3d.yaml").read_text())
        self.assertEqual(config["setting"], segmentation["setting"])
        self.assertEqual(config["imageType"], segmentation["imageType"])
        self.assertEqual(config["featureClass"], {name: value for name, value in segmentation["featureClass"].items() if name != "shape"})

    def test_real_extraction_preserves_filters_and_excludes_shape(self):
        image = sitk.GetImageFromArray(np.random.default_rng(42).normal(-500, 100, (32, 32, 32)).astype(np.float32))
        image.SetSpacing((0.7, 0.7, 0.7))
        center = image.TransformContinuousIndexToPhysicalPoint((15.5, 15.5, 15.5))
        selected, mask = create_sphere(image, center, 3)
        extractor = featureextractor.RadiomicsFeatureExtractor(str(ROOT / "configs/radiomics_sphere_3d.yaml"))
        with contextlib.redirect_stderr(io.StringIO()):
            features = extract_features(selected, mask, extractor)
        self.assertFalse(any("_shape_" in name or "_shape2D_" in name or name.startswith("diagnostics_") for name in features))
        self.assertIn("wavelet-HHH_glcm_Contrast", features)
        for sigma in range(1, 6):
            self.assertIn(f"log-sigma-{sigma}-0-mm-3D_firstorder_Mean", features)
        for name in ("lbp-3D-m1", "lbp-3D-m2", "lbp-3D-k"):
            self.assertIn(f"{name}_firstorder_Mean", features)
        self.assertTrue(all(np.isfinite(value) for value in features.values()))

    def test_checkpoint_fingerprints_roi_radius_dimension_and_binning(self):
        with TemporaryDirectory(dir="/tmp/opencode", prefix="sphere-resume-") as temporary:
            output = Path(temporary) / "features.csv"
            config = ROOT / "configs/radiomics_sphere_3d.yaml"
            directory = prepare_run(output, config, 6, False, [], roi="sphere", dimension=3, bin_width=10)
            settings = json.loads((directory / "settings.json").read_text())
            self.assertEqual(settings["roi"], "sphere")
            self.assertEqual(settings["radius_mm"], 6)
            self.assertEqual(settings["dimension"], 3)
            self.assertEqual(settings["bin_width"], 10)
            self.assertNotIn("consensus", settings)
            self.assertEqual(prepare_run(output, config, 6, False, [], roi="sphere", dimension=3, bin_width=10), directory)
            for changes in ({"radius_mm": 7}, {"roi": "segmentation"}, {"bin_width": 25}):
                kwargs = {"radius_mm": 6, "clip": False, "sources": [], "roi": "sphere", "dimension": 3, "bin_width": 10, **changes}
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "Settings or inputs changed"):
                    prepare_run(output, config, **kwargs)
            with self.assertRaisesRegex(ValueError, "dimension"):
                prepare_run(output, config, 6, False, [], roi="sphere", dimension=2)

    def test_cli_requires_3d_and_rejects_2d_config_and_consensus(self):
        for main in (extract_lidc, extract_lndb):
            for args in (["--roi", "sphere"],
                         ["--roi", "sphere", "--dimension", "3", "--config", str(ROOT / "configs/radiomics_circle_2d.yaml")],
                         ["--roi", "sphere", "--dimension", "3", "--consensus", "0.75"]):
                with self.subTest(main=main.__module__, args=args), patch("sys.argv", ["extract", *args]), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)

    def test_test_cli_uses_centroids_no_reader_masks_and_resumes_successes(self):
        with TemporaryDirectory(dir="/tmp/opencode", prefix="sphere-test-cli-") as temporary:
            root = Path(temporary)
            image = sitk.GetImageFromArray(np.zeros((9, 11, 13), dtype=np.int16))
            image.SetSpacing((0.8, 0.8, 2.5))
            image.SetOrigin((10., -20., 30.))
            centers = [image.TransformIndexToPhysicalPoint((6, 5, 4)), image.TransformIndexToPhysicalPoint((6, 5, 10))]
            nodules_path = root / "test.csv"
            pd.DataFrame({"LNDbID": [1, 1], "FindingID": [2, 3], "x": [p[0] for p in centers],
                          "y": [p[1] for p in centers], "z": [p[2] for p in centers], "prediction": [5, 5]}).to_csv(nodules_path, index=False)
            paths = load_paths()
            paths["outputs"]["features"] = root
            paths["lndb"]["ct_dir"] = root
            output = root / "lndb_test_sphere_r6_bw10_hu.csv"
            arguments = ["extract", "--roi", "sphere", "--dimension", "3", "--radius", "6", "--bin-width", "10",
                         "--test-nodules", str(nodules_path), "--config", str(ROOT / "configs/radiomics_segmentation_3d.yaml")]
            with patch("sys.argv", arguments), patch("LNDBfeatureExtraction.load_paths", return_value=paths), \
                    patch("LNDBfeatureExtraction.load_scan", return_value=image), \
                    patch("LNDBfeatureExtraction.load_reader_mask") as readers, \
                    patch("LNDBfeatureExtraction.match_nodules_to_masks") as matched, \
                    patch("LNDBfeatureExtraction.extract_features", return_value={"original_firstorder_Mean": -500.}) as extracted, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(extract_lndb(), 1)
                selected, mask, extractor, clip = extracted.call_args.args
                self.assertIs(selected, image)
                check_mask_geometry(image, mask)
                self.assertNotIn("shape", extractor.enabledFeatures)
                self.assertEqual(extractor.settings["binWidth"], 10)
                self.assertEqual(extractor.enabledImagetypes["LBP3D"]["binWidth"], 0.25)
                self.assertFalse(clip)
                self.assertEqual(pd.read_csv(output)["FindingID"].tolist(), [2])
                errors = pd.read_csv(output.with_suffix(".scans") / "LNDb-0001.errors.csv")
                self.assertEqual(errors["FindingID"].tolist(), [3])
                self.assertIn("outside the scan", errors["error"].iloc[0])
                self.assertEqual(extract_lndb(), 1)
                self.assertEqual(extracted.call_count, 1)
                readers.assert_not_called()
                matched.assert_not_called()


if __name__ == "__main__":
    unittest.main()
