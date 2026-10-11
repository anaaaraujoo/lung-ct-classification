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

from LIDCfeatureExtraction import create_lidc_segmentation, main as extract_lidc
from LNDBfeatureExtraction import create_lndb_segmentation, load_reader_mask, main as extract_lndb, match_nodules_to_masks, reader_findings
from src.radiomics_extraction import (
    check_mask_geometry,
    combine_reader_masks,
    create_circle,
    extract_features,
    file_checksum,
    largest_mask_slice,
    load_paths,
    prepare_run,
    segmentation_roi,
)

ROOT = Path(__file__).resolve().parents[1]


def make_image(values):
    image = sitk.GetImageFromArray(values)
    image.SetSpacing((0.8, 0.8, 2.5))
    image.SetOrigin((10., -20., 30.))
    return image


class PathTests(unittest.TestCase):
    def test_yaml_paths_expand_home_and_use_repository_root(self):
        with TemporaryDirectory(dir="/tmp/opencode", prefix="paths-test-") as directory:
            config = Path(directory) / "paths.yaml"
            config.write_text("lndb:\n  gt: ~/Downloads/trainset_csv/trainNodules_gt.csv\n"
                              "outputs:\n  features: data\n  models: /tmp/opencode/model outputs\n")
            paths = load_paths(config)
            self.assertEqual(paths["lndb"]["gt"], Path.home() / "Downloads/trainset_csv/trainNodules_gt.csv")
            self.assertEqual(paths["outputs"]["features"], ROOT / "data")
            self.assertEqual(paths["outputs"]["models"], Path("/tmp/opencode/model outputs"))

    def test_default_yaml_preserves_current_inputs(self):
        paths = load_paths()
        self.assertEqual(paths["lidc"]["targets"], ROOT / "data/processed/lidc_nodule_targets.csv")
        self.assertEqual(paths["lndb"]["segmentation_targets"], ROOT / "data/processed/lndb_ge3_nodule_targets.csv")
        self.assertEqual(paths["radiomics"]["segmentation"], ROOT / "configs/radiomics_segmentation_2d.yaml")
        self.assertEqual(paths["radiomics"]["segmentation_3d"], ROOT / "configs/radiomics_segmentation_3d.yaml")
        self.assertEqual(paths["radiomics"]["sphere_3d"], ROOT / "configs/radiomics_sphere_3d.yaml")
        self.assertEqual(paths["training"]["lidc_features"], ROOT / "data/lidc_circle_r21_hu.csv")
        self.assertEqual(paths["training"]["lndb_features"], ROOT / "data/lndb_circle_r11_hu.csv")

    def test_invalid_paths_are_rejected(self):
        with TemporaryDirectory(dir="/tmp/opencode", prefix="bad-paths-test-") as directory:
            config = Path(directory) / "paths.yaml"
            for contents in ("", "[]", "lndb: data", "lndb:\n  gt:", "lndb:\n  gt: 3"):
                config.write_text(contents)
                with self.subTest(contents=contents), self.assertRaises(ValueError):
                    load_paths(config)


class ConsensusTests(unittest.TestCase):
    def test_half_includes_ties_and_one_reader(self):
        first = np.array([[[1, 0, 1, 0]]], dtype=bool)
        second = np.array([[[0, 1, 1, 0]]], dtype=bool)
        np.testing.assert_array_equal(combine_reader_masks([first]), first)
        np.testing.assert_array_equal(combine_reader_masks([first, second]), [[[1, 1, 1, 0]]])

    def test_three_readers_require_two_votes(self):
        masks = [np.array([[[1, 1, 0]]]), np.array([[[1, 0, 1]]]), np.array([[[0, 0, 1]]])]
        np.testing.assert_array_equal(combine_reader_masks(iter(masks)), [[[1, 0, 1]]])

    def test_configurable_consensus(self):
        first = np.array([[[1, 0, 1]]])
        second = np.array([[[0, 1, 1]]])
        np.testing.assert_array_equal(combine_reader_masks([first, second], 0.75), [[[0, 0, 1]]])
        np.testing.assert_array_equal(combine_reader_masks([first, second], 1), [[[0, 0, 1]]])
        for level in (0, -0.5, 1.01, np.nan, np.inf):
            with self.subTest(level=level), self.assertRaises(ValueError):
                combine_reader_masks([first, second], level)

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


class BinWidthTests(unittest.TestCase):
    def test_override_changes_discretization_not_config_or_feature_names(self):
        image = make_image(np.arange(4 * 8 * 9, dtype=np.int16).reshape(4, 8, 9) * 3)
        mask = make_image(np.ones((4, 8, 9), dtype=np.uint8))
        image, mask, _ = largest_mask_slice(image, mask)
        for roi in ("circle", "segmentation"):
            config = ROOT / f"configs/radiomics_{roi}_2d.yaml"
            before = config.read_bytes()
            baseline = featureextractor.RadiomicsFeatureExtractor(str(config))
            override = featureextractor.RadiomicsFeatureExtractor(str(config), binWidth=10)
            with self.subTest(roi=roi), contextlib.redirect_stderr(io.StringIO()):
                original = extract_features(image, mask, baseline)
                changed = extract_features(image, mask, override)
                self.assertEqual(baseline.settings["binWidth"], 25)
                self.assertEqual(override.settings["binWidth"], 10)
                self.assertEqual(original.keys(), changed.keys())
                self.assertNotEqual(original["original_firstorder_Entropy"], changed["original_firstorder_Entropy"])
                self.assertEqual(config.read_bytes(), before)

    def test_both_cli_overrides_names_checkpoints_and_resume(self):
        with TemporaryDirectory(dir="/tmp/opencode", prefix="bin-width-cli-") as directory:
            root = Path(directory)
            image = make_image(np.zeros((3, 8, 9), dtype=np.int16))
            center = image.TransformIndexToPhysicalPoint((4, 3, 1))
            tables = {
                "lidc_targets": pd.DataFrame({"patient_id": ["patient"], "series_uid": ["series"], "nodule_id": [0]}),
                "lidc_folds": pd.DataFrame({"patient_id": ["patient"], "fold": [1]}),
                "lidc_annotations": pd.DataFrame({"patient_id": ["patient"], "series_uid": ["series"],
                                                  "nodule_id": [0], "centroid": ["[3 4 1]"]}),
                "lndb_targets": pd.DataFrame({"LNDbID": [1], "NoduleID": [1]}),
                "lndb_folds": pd.DataFrame({"LNDbID": [1], "fold": [1]}),
                "findings": pd.DataFrame({"LNDbID": [1], "FindingID": [1], "x": [center[0]],
                                          "y": [center[1]], "z": [center[2]], "RadID": [1],
                                          "RadFinding": [1], "Nodule": [1],
                                          "DiamEq_Rad": [2 * np.cbrt(3 * 125 / (4 * np.pi))]}),
                "gt": pd.DataFrame({"LNDbID": [1], "FindingID": [1], "x": [center[0]],
                                    "y": [center[1]], "z": [center[2]], "RadID": [1],
                                    "RadFindingID": [1], "Nodule": [1], "Volume": [125]}),
            }
            for name, table in tables.items():
                table.to_csv(root / f"{name}.csv", index=False)
            (root / "masks.zip").write_bytes(b"mask source")
            paths = {
                "lidc": {name: root / f"lidc_{name}.csv" for name in ("targets", "folds", "annotations")},
                "lndb": {"targets": root / "lndb_targets.csv", "segmentation_targets": root / "lndb_targets.csv",
                         "folds": root / "lndb_folds.csv", "findings": root / "findings.csv",
                         "gt": root / "gt.csv", "masks": root / "masks.zip", "ct_dir": root},
                "radiomics": {roi: ROOT / f"configs/radiomics_{roi}_2d.yaml" for roi in ("circle", "segmentation")},
                "outputs": {"features": root},
            }
            paths["radiomics"]["segmentation_3d"] = ROOT / "configs/radiomics_segmentation_3d.yaml"
            paths["radiomics"]["sphere_3d"] = ROOT / "configs/radiomics_sphere_3d.yaml"
            scan = SimpleNamespace(to_volume=lambda verbose: sitk.GetArrayFromImage(image).transpose(1, 2, 0),
                                   pixel_spacing=0.8, slice_spacing=2.5, cluster_annotations=lambda verbose: [[]])
            slice_image = image[:, :, 1:2]
            slice_mask = sitk.Cast(slice_image == 0, sitk.sitkUInt8)
            for module, main, dataset in (("LIDCfeatureExtraction", extract_lidc, "lidc"),
                                          ("LNDBfeatureExtraction", extract_lndb, "lndb")):
                for roi, dimension in (("circle", 2), ("sphere", 3), ("segmentation", 2), ("segmentation", 3)):
                    for width, custom_output in ((None, False), (10, False), (0.25, False), (10, True)):
                        args = [module, "--roi", roi, "--radius", "6", "--dimension", str(dimension)]
                        if width is not None:
                            args += ["--bin-width", str(width)]
                        prefix = "lndb_ge3" if dataset == "lndb" and roi == "segmentation" else dataset
                        roi_name = f"{roi}_r6" if roi in ("circle", "sphere") else f"segmentation_{dimension}d_c50"
                        suffix = f"_bw{width:g}" if width is not None else ""
                        output = root / f"{prefix}_{roi_name}{suffix}_hu.csv"
                        if custom_output:
                            output = root / f"{prefix}_{roi_name}_custom.csv"
                            args += ["--output", str(output)]
                        with self.subTest(dataset=dataset, roi=roi, dimension=dimension, width=width, custom_output=custom_output), \
                                contextlib.ExitStack() as stack:
                            stack.enter_context(patch("sys.argv", args))
                            stack.enter_context(patch(f"{module}.load_paths", return_value=paths))
                            selected = (slice_image, slice_mask, 1) if dimension == 2 else (image, sitk.Cast(image == 0, sitk.sitkUInt8), None)
                            extracted = stack.enter_context(patch(f"{module}.extract_features", side_effect=
                                lambda image, mask, extractor, clip: {"original_firstorder_Entropy": extractor.settings["binWidth"]}))
                            if dataset == "lidc":
                                queried = stack.enter_context(patch(f"{module}.pl.query"))
                                queried.return_value.filter.return_value.one.return_value = scan
                                stack.enter_context(patch(f"{module}.create_lidc_segmentation", return_value=selected))
                            else:
                                stack.enter_context(patch(f"{module}.load_scan", return_value=image))
                                stack.enter_context(patch(f"{module}.load_reader_mask", return_value=image))
                                stack.enter_context(patch(f"{module}.create_lndb_segmentation", return_value=selected))
                            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                            self.assertEqual(main(), 0)
                            settings = json.loads((output.with_suffix(".scans") / "settings.json").read_text())
                            self.assertEqual(settings.get("bin_width"), width)
                            if dimension == 3:
                                self.assertEqual(settings["dimension"], 3)
                                if roi == "segmentation":
                                    self.assertEqual(settings["slice_selection"], "full_volume")
                                else:
                                    self.assertEqual(settings["radius_mm"], 6)
                                    self.assertNotIn("consensus", settings)
                                    self.assertNotIn("shape", extracted.call_args.args[2].enabledFeatures)
                                    self.assertNotIn("shape2D", extracted.call_args.args[2].enabledFeatures)
                                self.assertNotIn("slice_index", pd.read_csv(output))
                                self.assertEqual(extracted.call_args.args[0].GetSize(), image.GetSize())
                                self.assertEqual(extracted.call_args.args[2].enabledImagetypes["LBP3D"]["binWidth"], 0.25)
                            self.assertEqual(pd.read_csv(output)["original_firstorder_Entropy"].tolist(), [width if width is not None else 25])
                            before = output.read_bytes()
                            self.assertEqual(main(), 0)
                            self.assertEqual(extracted.call_count, 1)
                            self.assertEqual(output.read_bytes(), before)

    def test_invalid_cli_widths_are_rejected_before_loading_inputs(self):
        for main in (extract_lidc, extract_lndb):
            for width in ("0", "-1", "nan", "inf", "-inf"):
                with self.subTest(main=main.__module__, width=width), \
                        patch("sys.argv", ["extract", f"--bin-width={width}"]), \
                        patch(f"{main.__module__}.load_paths") as loaded, contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        main()
                    self.assertEqual(error.exception.code, 2)
                    loaded.assert_not_called()


class VolumeTests(unittest.TestCase):
    def test_full_volume_retains_geometry_and_all_slices(self):
        image = make_image(np.zeros((5, 8, 9), dtype=np.int16))
        values = np.zeros((5, 8, 9), dtype=np.uint8)
        values[1:4, 2:5, 3:7] = 1
        mask = make_image(values)
        selected, selected_mask, index = segmentation_roi(image, mask, dimension=3)
        self.assertIs(selected, image)
        self.assertIsNone(index)
        check_mask_geometry(selected, selected_mask)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(selected_mask), values)
        for bad in (mask * 0, mask * 2, mask[:, :, :1]):
            with self.assertRaises(ValueError):
                segmentation_roi(image, bad, dimension=3)
        with self.assertRaises(ValueError):
            segmentation_roi(image, mask, dimension=4)

    def test_lndb_full_volume_preserves_reader_consensus(self):
        first = np.zeros((5, 8, 9), dtype=np.uint8)
        first[1:3, 2:5, 3:7] = 4
        second = np.zeros_like(first)
        second[2:4, 2:5, 3:7] = 7
        image = make_image(np.zeros_like(first, dtype=np.int16))
        masks = {1: make_image(first), 3: make_image(second)}
        nodule = {"mask_RadID": "1,3", "mask_RadFindingID": "4,7"}
        for level, expected in ((0.5, (first == 4) | (second == 7)), (1, (first == 4) & (second == 7))):
            selected, mask, index = create_lndb_segmentation(image, nodule, masks, level, dimension=3)
            self.assertIs(selected, image)
            self.assertIsNone(index)
            np.testing.assert_array_equal(sitk.GetArrayFromImage(mask), expected)

    def test_lidc_full_volume_preserves_reader_validation_and_axis_order(self):
        image = make_image(np.zeros((5, 8, 9), dtype=np.int16))
        cluster = [SimpleNamespace(scan=SimpleNamespace(id=12), centroid=np.array([3., 4., 2.]))]
        saved = pd.DataFrame({"annotation_idx": [0], "scan_id": [12], "row": [3.], "column": [4.], "slice": [2.]})
        combined = np.ones((3, 4, 2), dtype=bool)
        bbox = (slice(2, 5), slice(3, 7), slice(1, 3))
        with patch("LIDCfeatureExtraction.consensus", return_value=(combined, bbox, [combined])):
            selected, mask, index = create_lidc_segmentation(image, cluster, saved, 0.75, dimension=3)
            self.assertIs(selected, image)
            self.assertIsNone(index)
            expected = np.zeros((5, 8, 9), dtype=np.uint8)
            expected[1:3, 2:5, 3:7] = 1
            np.testing.assert_array_equal(sitk.GetArrayFromImage(mask), expected)
        with self.assertRaises(ValueError):
            create_lidc_segmentation(image, cluster, saved.assign(scan_id=99), dimension=3)

    def test_real_3d_config_extracts_shape_log_lbp_and_3d_wavelets(self):
        values = np.random.default_rng(42).normal(-500, 100, (32, 32, 32)).astype(np.float32)
        image = sitk.GetImageFromArray(values)
        image.SetSpacing((0.7, 0.7, 0.7))
        labels = np.zeros_like(values, dtype=np.uint8)
        labels[12:20, 12:20, 12:20] = 1
        mask = sitk.GetImageFromArray(labels)
        mask.CopyInformation(image)
        config = ROOT / "configs/radiomics_segmentation_3d.yaml"
        extractor = featureextractor.RadiomicsFeatureExtractor(str(config))
        self.assertFalse(extractor.settings["force2D"])
        self.assertFalse(extractor.settings["normalize"])
        self.assertEqual(extractor.settings["binWidth"], 25)
        self.assertEqual(extractor.settings["resampledPixelSpacing"], [0.7, 0.7, 0.7])
        self.assertEqual(extractor.enabledImagetypes["LBP3D"]["binWidth"], 0.25)
        with contextlib.redirect_stderr(io.StringIO()):
            features = extract_features(image, mask, extractor)
        self.assertEqual(len(features), 2060)
        self.assertIn("original_shape_MeshVolume", features)
        self.assertIn("wavelet-HHH_glcm_Contrast", features)
        for sigma in range(1, 6):
            self.assertIn(f"log-sigma-{sigma}-0-mm-3D_firstorder_Mean", features)
        for name in ("lbp-3D-m1", "lbp-3D-m2", "lbp-3D-k"):
            self.assertIn(f"{name}_firstorder_Mean", features)
        self.assertTrue(all(np.isfinite(value) for value in features.values()))
        self.assertFalse(any("_shape2D_" in name or name.startswith("diagnostics_") for name in features))

    def test_cli_rejects_3d_circles_and_2d_configs(self):
        for main in (extract_lidc, extract_lndb):
            arguments = (["--dimension", "3"], ["--dimension", "4"],
                         ["--roi", "segmentation", "--dimension", "3", "--config", str(ROOT / "configs/radiomics_segmentation_2d.yaml")])
            for args in arguments:
                with self.subTest(main=main.__module__, args=args), patch("sys.argv", ["extract", *args]), \
                        contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        main()
                    self.assertEqual(error.exception.code, 2)

class ReaderAssociationTests(unittest.TestCase):
    def test_lndb_uses_gt_labels_not_allnods_reader_ids(self):
        nodule = {"FindingID": 99, "RadID": "4,5", "RadFinding": "1,1",
                  "mask_RadID": "1,3", "mask_RadFindingID": "4,7"}
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
            create_lndb_segmentation(image, {**nodule, "mask_RadFindingID": "5,7"}, {1: first, 3: third})

    def test_lndb_rejects_bad_reader_lists(self):
        valid = {"mask_RadID": "1,2", "mask_RadFindingID": "3,4"}
        for changed in [{"mask_RadFindingID": "3"}, {"mask_RadID": "1,1"}, {"mask_RadID": "1,2.5"},
                        {"mask_RadID": np.nan}, {"mask_RadID": "0,2"}, {"mask_RadFindingID": "0,4"}]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                reader_findings({**valid, **changed})
        self.assertEqual(reader_findings({"mask_RadID": 1., "mask_RadFindingID": 2.}), [(1, 2)])

    def test_lndb_matches_location_size_and_labels_not_group_or_reader_ids(self):
        gt = pd.DataFrame({"LNDbID": [68, 131, 311], "FindingID": [1, 1, 7], "Nodule": [1, 1, 1],
                           "RadID": ["1,2", "1", "2"], "RadFindingID": ["1,1", "1", "1"],
                           "x": [10., 40., 70.], "y": [20., 50., 80.], "z": [30., 60., 90.],
                           "Volume": [125., 20., 25.]})
        nodules = gt[["LNDbID", "x", "y", "z"]].copy()
        nodules["FindingID"], nodules["RadID"] = [1, 1, 6], ["1,3", "4", "5,4"]
        nodules["RadFinding"], nodules["Nodule"] = ["1,1", "1", "1,2"], ["1,1", "1", "1,0"]
        nodules["DiamEq_Rad"] = 2 * np.cbrt(3 * gt["Volume"] / (4 * np.pi))
        nodules["target_classification_mean"] = [2, 3, 4]
        nodules.loc[2, "x"] += 1  # Averaging non-nodule annotations can shift the centre.
        mapping = match_nodules_to_masks(nodules, gt)
        self.assertEqual(mapping["FindingID"].tolist(), [1, 1, 6])
        self.assertEqual(mapping["gt_FindingID"].tolist(), [1, 1, 7])
        self.assertEqual(mapping["mask_RadID"].tolist(), ["1,2", "1", "2"])
        self.assertEqual(mapping["match_distance_mm"].tolist(), [0., 0., 1.])
        joined = nodules.merge(mapping, on=["LNDbID", "FindingID"], validate="one_to_one")
        self.assertEqual(joined["target_classification_mean"].tolist(), [2, 3, 4])
        for wrong in (gt.assign(x=999.), gt.assign(Volume=999.), gt.assign(RadFindingID="9")):
            with self.assertRaisesRegex(ValueError, "found 0"):
                match_nodules_to_masks(nodules, wrong)
        # A matching centre alone is insufficient. Keep size and label checks.
        nearby_wrong = gt.assign(Volume=999., FindingID=99)
        self.assertEqual(len(match_nodules_to_masks(nodules, pd.concat([gt, nearby_wrong]))), 3)
        with self.assertRaisesRegex(ValueError, "found 2"):
            match_nodules_to_masks(nodules, pd.concat([gt, gt.assign(FindingID=99)]))
        duplicate = pd.concat([nodules, nodules.iloc[[0]].assign(FindingID=99)])
        with self.assertRaisesRegex(ValueError, "same _gt nodule"):
            match_nodules_to_masks(duplicate, gt)

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
        with patch("LIDCfeatureExtraction.consensus", return_value=(combined, bbox, [combined])) as mocked:
            _, mask, index = create_lidc_segmentation(image, cluster, saved, 0.75)
            mocked.assert_called_once_with(cluster, clevel=0.75)
            self.assertEqual(index, 2)
            expected = np.zeros((1, 8, 9), dtype=np.uint8)
            expected[0, 2:5, 3:7] = 1
            np.testing.assert_array_equal(sitk.GetArrayFromImage(mask), expected)
        for changed in [saved.assign(row=4.), saved.assign(scan_id=99), saved.assign(annotation_idx=1), saved.head(0)]:
            with self.assertRaises(ValueError):
                create_lidc_segmentation(image, cluster, changed)


class ResumeTests(unittest.TestCase):
    def test_3d_checkpoints_cannot_resume_as_2d(self):
        with TemporaryDirectory(dir="/tmp/opencode", prefix="volume-resume-") as directory:
            output = Path(directory) / "features.csv"
            config = ROOT / "configs/radiomics_segmentation_3d.yaml"
            progress = prepare_run(output, config, None, False, [], roi="segmentation", dimension=3)
            settings = json.loads((progress / "settings.json").read_text())
            self.assertEqual(settings["dimension"], 3)
            self.assertEqual(settings["slice_selection"], "full_volume")
            self.assertEqual(prepare_run(output, config, None, False, [], roi="segmentation", dimension=3), progress)
            with self.assertRaisesRegex(ValueError, "Settings or inputs changed"):
                prepare_run(output, config, None, False, [], roi="segmentation", dimension=2)
            with self.assertRaisesRegex(ValueError, "Settings or inputs changed"):
                prepare_run(output, config, None, False, [], roi="segmentation", dimension=3, consensus_level=0.75)

    def test_bin_width_overrides_are_part_of_checkpoint_settings(self):
        with TemporaryDirectory(dir="/tmp/opencode", prefix="bin-width-resume-") as directory:
            root = Path(directory)
            config = root / "config.yaml"
            config.write_text("setting:\n  binWidth: 25\n")
            output = root / "features.csv"
            progress = prepare_run(output, config, 6, False, [], bin_width=10)
            self.assertEqual(json.loads((progress / "settings.json").read_text())["bin_width"], 10)
            self.assertEqual(prepare_run(output, config, 6, False, [], bin_width=10), progress)
            for width in (None, 25, 0.25):
                with self.subTest(width=width), self.assertRaisesRegex(ValueError, "Settings or inputs changed"):
                    prepare_run(output, config, 6, False, [], bin_width=width)
            for width in (0, -1, np.nan, np.inf):
                with self.subTest(width=width), self.assertRaisesRegex(ValueError, "positive finite"):
                    prepare_run(output, config, 6, False, [], bin_width=width)

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
                prepare_run(output, config, None, False, [source], roi="segmentation", consensus_level=0.75)
            with self.assertRaises(ValueError):
                prepare_run(output, config, 6, False, [source])
            source.write_bytes(b"changed mask source")
            with self.assertRaises(ValueError):
                prepare_run(output, config, None, False, [source], roi="segmentation")


if __name__ == "__main__":
    unittest.main()
