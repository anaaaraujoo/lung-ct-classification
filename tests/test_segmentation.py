"""Small mask tests that do not require downloading CT datasets."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

from LIDCfeatureExtraction import create_lidc_segmentation
from LNDBfeatureExtraction import create_lndb_segmentation, load_reader_mask, reader_findings
from src.radiomics_extraction import (
    check_mask_geometry,
    combine_reader_masks,
    create_circle,
    extract_features,
    file_checksum,
    largest_mask_slice,
    prepare_run,
)

ROOT = Path(__file__).resolve().parents[1]


def make_image(values):
    image = sitk.GetImageFromArray(values)
    image.SetSpacing((0.8, 0.8, 2.5))
    image.SetOrigin((10., -20., 30.))
    return image


class ConsensusTests(unittest.TestCase):
    def test_half_includes_ties_and_one_reader(self):
        first = np.array([[[1, 0, 1, 0]]], dtype=bool)
        second = np.array([[[0, 1, 1, 0]]], dtype=bool)
        np.testing.assert_array_equal(combine_reader_masks([first]), first)
        np.testing.assert_array_equal(combine_reader_masks([first, second]), [[[1, 1, 1, 0]]])

    def test_three_readers_require_two_votes(self):
        masks = [np.array([[[1, 1, 0]]]), np.array([[[1, 0, 1]]]), np.array([[[0, 0, 1]]])]
        np.testing.assert_array_equal(combine_reader_masks(iter(masks)), [[[1, 0, 1]]])

    def test_invalid_or_empty_masks_are_not_dropped(self):
        valid = np.ones((1, 2, 2), dtype=bool)
        for masks in ([], [valid, np.zeros_like(valid)], [valid, np.ones((2, 2, 2))],
                      [valid * 2], [valid[0]]):
            with self.subTest(masks=len(masks)), self.assertRaises(ValueError):
                combine_reader_masks(masks)
        disjoint = [np.eye(3, dtype=bool)[i].reshape(1, 1, 3) for i in range(3)]
        with self.assertRaisesRegex(ValueError, "consensus mask is empty"):
            combine_reader_masks(disjoint)


class SliceTests(unittest.TestCase):
    def setUp(self):
        self.image = make_image(np.arange(4 * 8 * 9, dtype=np.int16).reshape(4, 8, 9))
        values = np.zeros((4, 8, 9), dtype=np.uint8)
        values[0, 2, 3] = 1
        values[2:4, 1:5, 2:7] = 1
        self.mask = make_image(values)

    def test_largest_area_keeps_one_slice_and_geometry(self):
        image, mask, index = largest_mask_slice(self.image, self.mask)
        self.assertEqual(index, 2)  # Earliest of the two equal-area slices.
        self.assertEqual(image.GetSize(), (9, 8, 1))
        self.assertEqual(image.GetOrigin(), self.image.TransformIndexToPhysicalPoint((0, 0, 2)))
        check_mask_geometry(image, mask)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(mask)[0], sitk.GetArrayFromImage(self.mask)[2])

    def test_geometry_mismatches_raise(self):
        for attribute, value in [("SetOrigin", (11., -20., 30.)), ("SetSpacing", (0.7, 0.8, 2.5)),
                                 ("SetDirection", (-1., 0., 0., 0., 1., 0., 0., 0., 1.))]:
            mask = sitk.Image(self.mask)
            getattr(mask, attribute)(value)
            with self.subTest(attribute=attribute), self.assertRaises(ValueError):
                largest_mask_slice(self.image, mask)
        with self.assertRaises(ValueError):
            largest_mask_slice(self.image, self.mask[:, :, :1])

    def test_empty_and_non_binary_masks_raise(self):
        for mask in (self.mask * 0, self.mask * 3):
            with self.assertRaises(ValueError):
                largest_mask_slice(self.image, mask)

    def test_circles_still_choose_nearest_centroid_slice(self):
        center = self.image.TransformContinuousIndexToPhysicalPoint((4., 3., 0.6))
        image, mask = create_circle(self.image, center, 1.2)
        self.assertEqual(image.GetOrigin(), self.image.TransformIndexToPhysicalPoint((0, 0, 1)))
        check_mask_geometry(image, mask)

    def test_real_extractor_includes_shape2d_and_existing_filters(self):
        image, mask, _ = largest_mask_slice(self.image, self.mask)
        extractor = featureextractor.RadiomicsFeatureExtractor(str(ROOT / "configs/radiomics_segmentation_2d.yaml"))
        with contextlib.redirect_stderr(io.StringIO()):
            features = extract_features(image, mask, extractor)
        self.assertIn("original_shape2D_PixelSurface", features)
        self.assertTrue(np.isfinite(features["original_shape2D_PixelSurface"]))
        self.assertIn("wavelet-HL_glcm_Contrast", features)
        self.assertIn("logarithm_firstorder_Mean", features)
        self.assertFalse(any(name.startswith("diagnostics_") or name.startswith("log-sigma") for name in features))


class ReaderAssociationTests(unittest.TestCase):
    def test_lndb_uses_reader_labels_and_ignores_non_nodules(self):
        nodule = {"FindingID": 99, "RadID": "1,2,3", "RadFinding": "4,8,7", "Nodule": "1,0,1"}
        self.assertEqual(reader_findings(nodule), [(1, 4), (3, 7)])
        values = np.zeros((3, 8, 9), dtype=np.uint8)
        values[1, 1:4, 2:6] = 4
        first = make_image(values)
        third = make_image(np.where(values == 4, 7, 0).astype(np.uint8))
        image = make_image(np.zeros_like(values, dtype=np.int16))
        _, mask, index = create_lndb_segmentation(image, nodule, {1: first, 3: third})
        self.assertEqual(index, 1)
        self.assertEqual(int(sitk.GetArrayFromImage(mask).sum()), 12)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            create_lndb_segmentation(image, {**nodule, "RadFinding": "5,8,7"}, {1: first, 3: third})

    def test_lndb_rejects_bad_reader_lists(self):
        valid = {"RadID": "1,2", "RadFinding": "3,4", "Nodule": "1,1"}
        for changed in [{"RadFinding": "3"}, {"RadID": "1,1"}, {"RadID": "1,2.5"},
                        {"RadID": np.nan}, {"Nodule": "0,0"}, {"Nodule": "1,2"}, {"RadFinding": "0,4"}]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                reader_findings({**valid, **changed})
        self.assertEqual(reader_findings({"RadID": 1., "RadFinding": 2., "Nodule": 1.}), [(1, 2)])

    def test_reader_zip_is_unchanged(self):
        with TemporaryDirectory(dir="/tmp/opencode", prefix="mask-zip-test-") as directory:
            directory = Path(directory)
            original = make_image(np.ones((2, 3, 4), dtype=np.uint8))
            header = directory / "LNDb-0002_rad3.mhd"
            sitk.WriteImage(original, str(header))
            archive_path = directory / "masks.zip"
            with ZipFile(archive_path, "w") as archive:
                for suffix in ("mhd", "raw"):
                    path = header.with_suffix(f".{suffix}")
                    archive.write(path, f"masks/{path.name}")
            before = file_checksum(archive_path)
            loaded = load_reader_mask(archive_path, 2, 3)
            check_mask_geometry(original, loaded)
            np.testing.assert_array_equal(sitk.GetArrayFromImage(original), sitk.GetArrayFromImage(loaded))
            self.assertEqual(file_checksum(archive_path), before)

    def test_lidc_verifies_saved_readers_and_transposes_mask(self):
        image = make_image(np.zeros((5, 8, 9), dtype=np.int16))
        cluster = [SimpleNamespace(scan=SimpleNamespace(id=12), centroid=np.array([3., 4., 2.]))]
        saved = pd.DataFrame({"annotation_idx": [0], "scan_id": [12], "row": [3.], "column": [4.], "slice": [2.]})
        combined = np.zeros((3, 4, 2), dtype=bool)
        combined[:, :, 1] = True
        bbox = (slice(2, 5), slice(3, 7), slice(1, 3))
        with patch("LIDCfeatureExtraction.consensus", return_value=(combined, bbox, [combined])):
            _, mask, index = create_lidc_segmentation(image, cluster, saved)
            self.assertEqual(index, 2)
            expected = np.zeros((1, 8, 9), dtype=np.uint8)
            expected[0, 2:5, 3:7] = 1
            np.testing.assert_array_equal(sitk.GetArrayFromImage(mask), expected)
        for changed in [saved.assign(row=4.), saved.assign(scan_id=99), saved.assign(annotation_idx=1), saved.head(0)]:
            with self.assertRaises(ValueError):
                create_lidc_segmentation(image, cluster, changed)


class ResumeTests(unittest.TestCase):
    def test_segmentation_settings_and_source_changes(self):
        with TemporaryDirectory(dir="/tmp/opencode", prefix="mask-resume-test-") as directory:
            directory = Path(directory)
            config = directory / "config.yaml"
            config.write_text("setting: {}\n")
            source = directory / "masks.zip"
            source.write_bytes(b"first mask source")
            output = directory / "features.csv"
            progress = prepare_run(output, config, None, False, [source], roi="segmentation")
            settings = json.loads((progress / "settings.json").read_text())
            self.assertEqual(settings["consensus"], 0.5)
            self.assertEqual(settings["slice_selection"], "largest_mask_area")
            self.assertIsNone(settings["radius_mm"])
            self.assertEqual(prepare_run(output, config, 99, False, [source], roi="segmentation"), progress)
            with self.assertRaises(ValueError):
                prepare_run(output, config, 6, False, [source])
            source.write_bytes(b"changed mask source")
            with self.assertRaises(ValueError):
                prepare_run(output, config, None, False, [source], roi="segmentation")


if __name__ == "__main__":
    unittest.main()
