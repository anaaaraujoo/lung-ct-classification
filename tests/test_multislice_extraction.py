"""Shared LIDC/LNDb axial stacks, training populations and extraction checkpoints."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

import LIDCfeatureExtraction
import LNDBfeatureExtraction
from LIDCfeatureExtraction import KEYS, main as extract_lidc
from LNDBfeatureExtraction import main as extract_lndb
from src.modeling import FEATURE_PATTERN, load_data
from src.multislice_extraction import create_circle_slices, prepare_run_slices, segmentation_slices, slice_stack
from src.radiomics_extraction import check_mask_geometry, extract_features

ROOT = Path(__file__).resolve().parents[1]


def lidc_fixture(root):
    values = np.broadcast_to(np.arange(7, dtype=np.int16)[:, None, None], (7, 9, 10)).copy()
    image = sitk.GetImageFromArray(values)
    image.SetSpacing((0.8, 0.8, 2.5))
    tables = {
        "targets": pd.DataFrame({"patient_id": ["patient"] * 2, "series_uid": ["series"] * 2,
                                 "nodule_id": [0, 1], "ratings": ["[2,3]", "[4]"],
                                 "target_classification_mean": [3, 4], "target_regression_mean": [2.5, 4.],
                                 "target_binary_mean": [0, 1]}),
        "folds": pd.DataFrame({"patient_id": ["patient"], "fold": [1]}),
        "annotations": pd.DataFrame({"patient_id": ["patient"] * 2, "series_uid": ["series"] * 2,
                                     "nodule_id": [0, 1], "centroid": ["[4 5 3]", "[4 5 0]"]}),
    }
    for name, table in tables.items():
        table.to_csv(root / f"{name}.csv", index=False)
    paths = {"lidc": {name: root / f"{name}.csv" for name in tables},
             "radiomics": {"circle": ROOT / "configs/radiomics_circle_2d.yaml",
                           "segmentation": ROOT / "configs/radiomics_segmentation_2d.yaml"},
             "outputs": {"features": root}}
    scan = SimpleNamespace(to_volume=lambda verbose: values.transpose(1, 2, 0), pixel_spacing=0.8,
                           slice_spacing=2.5, cluster_annotations=lambda verbose: [[], []])
    return image, paths, scan


def lndb_fixture(root):
    image, _, _ = lidc_fixture(root)
    centres = [image.TransformIndexToPhysicalPoint((5, 4, index)) for index in (3, 0)]
    targets = pd.DataFrame({"LNDbID": [1, 1], "NoduleID": [7, 8], "ratings": ["[2,3]", "[4]"],
                            "target_classification_mean": [3, 4], "target_regression_mean": [2.5, 4.],
                            "target_binary_mean": [0, 1]})
    findings = pd.DataFrame({"LNDbID": [1, 1], "FindingID": [7, 8], "x": [point[0] for point in centres],
                             "y": [point[1] for point in centres], "z": [point[2] for point in centres],
                             "RadID": [4, 5], "RadFinding": [4, 7], "Nodule": [1, 1],
                             "DiamEq_Rad": [2 * np.cbrt(3 * 125 / (4 * np.pi))] * 2})
    gt = findings.rename(columns={"RadFinding": "RadFindingID"}).drop(columns="DiamEq_Rad")
    gt["FindingID"], gt["RadID"], gt["Volume"] = [70, 80], [1, 2], [125, 125]
    tables = {"targets": targets, "segmentation_targets": targets.iloc[:1],
              "folds": pd.DataFrame({"LNDbID": [1], "fold": [2]}), "findings": findings, "gt": gt}
    for name, table in tables.items():
        table.to_csv(root / f"lndb_{name}.csv", index=False)
    (root / "masks.zip").write_bytes(b"reader mask source")
    paths = {"lndb": {name: root / f"lndb_{name}.csv" for name in tables},
             "radiomics": {roi: ROOT / f"configs/radiomics_{roi}_2d.yaml" for roi in ("circle", "segmentation")},
             "outputs": {"features": root}}
    paths["lndb"].update(ct_dir=root, masks=root / "masks.zip")
    return image, paths


class StackTests(unittest.TestCase):
    def test_stack_indices_at_edges_and_for_even_counts(self):
        cases = {(3, 1): [3], (3, 2): [3, 4], (3, 3): [2, 3, 4], (3, 5): [1, 2, 3, 4, 5],
                 (0, 5): [0, 1, 2], (1, 5): [0, 1, 2, 3], (6, 5): [4, 5, 6], (3, 10): list(range(7))}
        for (centre, count), expected in cases.items():
            with self.subTest(centre=centre, count=count):
                self.assertEqual(slice_stack(centre, count, 7), expected)
        with self.assertRaises(ValueError):
            slice_stack(3, 0, 7)

    def test_both_extractors_use_the_shared_slice_functions(self):
        for name in ("create_circle_slices", "extract_slice_features", "prepare_run_slices", "segmentation_slices"):
            self.assertIs(getattr(LIDCfeatureExtraction, name), getattr(LNDBfeatureExtraction, name))

    def test_circle_geometry_values_and_physical_radius(self):
        image = sitk.GetImageFromArray(np.arange(7 * 9 * 10, dtype=np.int16).reshape(7, 9, 10))
        image.SetSpacing((0.8, 1.1, 2.5))
        image.SetOrigin((10., -20., 30.))
        image.SetDirection((0., -1., 0., 1., 0., 0., 0., 0., 1.))
        for centre_index, expected_indices in (((4.2, 3.4, 3.3), [1, 2, 3, 4, 5]), ((4.2, 3.4, 0.3), [0, 1, 2])):
            centre = image.TransformContinuousIndexToPhysicalPoint(centre_index)
            actual = create_circle_slices(image, centre, 2., 5)
            self.assertEqual([s[0] for s in actual], expected_indices)
            for index, selected, mask in actual:
                check_mask_geometry(selected, mask)
                self.assertEqual(selected.GetOrigin(), image.TransformIndexToPhysicalPoint((0, 0, index)))
                np.testing.assert_array_equal(sitk.GetArrayFromImage(selected)[0], sitk.GetArrayFromImage(image)[index])
                projected = image.TransformContinuousIndexToPhysicalPoint((*centre_index[:2], float(index)))
                expected = np.zeros((image.GetHeight(), image.GetWidth()), dtype=np.uint8)
                for y, x in np.ndindex(expected.shape):
                    point = image.TransformIndexToPhysicalPoint((x, y, index))
                    expected[y, x] = np.linalg.norm(np.array(point) - projected) <= 2.
                np.testing.assert_array_equal(sitk.GetArrayFromImage(mask)[0], expected)

    def test_segmentation_uses_largest_area_and_skips_empty_slices(self):
        image = sitk.GetImageFromArray(np.zeros((7, 9, 10), dtype=np.int16))
        values = np.zeros((7, 9, 10), dtype=np.uint8)
        values[[1, 5, 6], 3:5, 3:5] = 1
        values[3, 2:6, 2:7] = 1
        mask = sitk.GetImageFromArray(values)
        slices = segmentation_slices(image, mask, 5)
        self.assertEqual([s[0] for s in slices], [1, 3, 5])
        for index, selected, selected_mask in slices:
            check_mask_geometry(selected, selected_mask)
            np.testing.assert_array_equal(sitk.GetArrayFromImage(selected_mask)[0], values[index])

    def test_real_circle_features_match_uploaded_lndb_feature_columns(self):
        image = sitk.GetImageFromArray(np.arange(5 * 16 * 16, dtype=np.int16).reshape(5, 16, 16))
        image.SetSpacing((0.8, 0.8, 2.5))
        centre = image.TransformIndexToPhysicalPoint((8, 8, 2))
        extractor = featureextractor.RadiomicsFeatureExtractor(str(ROOT / "configs/radiomics_circle_2d.yaml"))
        index, selected, mask = create_circle_slices(image, centre, 3., 3)[1]
        self.assertEqual(index, 2)
        with contextlib.redirect_stderr(io.StringIO()):
            features = extract_features(selected, mask, extractor)
        columns = pd.read_csv(ROOT / "data/lndb_circle_r6_5slice_hu.csv", nrows=0).filter(regex=FEATURE_PATTERN).columns
        self.assertEqual(set(features), set(columns))


class LIDCStackTests(unittest.TestCase):
    def test_atomic_nodule_retry_targets_folds_and_resume(self):
        with TemporaryDirectory(dir="/tmp/opencode", prefix="lidc-stack-") as temporary:
            root = Path(temporary)
            image, paths, scan = lidc_fixture(root)
            originals = {path: path.read_bytes() for path in paths["lidc"].values()}
            output = root / "lidc_circle_r11_5slice_hu.csv"
            failed_once = False

            def extract(selected, mask, extractor, clip):
                nonlocal failed_once
                check_mask_geometry(selected, mask)
                self.assertEqual(extractor.settings["binWidth"], 25)
                self.assertFalse(clip)
                value = float(sitk.GetArrayViewFromImage(selected)[0, 0, 0])
                if value == 2 and not failed_once:
                    failed_once = True
                    raise ValueError("slice failed")
                return {"original_firstorder_Mean": value}

            args = ["extract", "--roi", "circle", "--radius", "11", "--slices", "5"]
            with patch("sys.argv", args), patch("LIDCfeatureExtraction.load_paths", return_value=paths), \
                    patch("LIDCfeatureExtraction.pl.query") as queried, \
                    patch("src.multislice_extraction.extract_features", side_effect=extract) as extracted, \
                    contextlib.redirect_stdout(io.StringIO()):
                queried.return_value.filter.return_value.one.return_value = scan
                self.assertEqual(extract_lidc(), 1)
                first = pd.read_csv(output)
                self.assertEqual(first["nodule_id"].tolist(), [1, 1, 1])
                errors_path = output.with_suffix(".scans") / "patient_series.errors.csv"
                self.assertEqual(pd.read_csv(errors_path)["nodule_id"].tolist(), [0])
                checkpoint = pd.read_csv(output.with_suffix(".scans") / "patient_series.features.csv")
                self.assertEqual(checkpoint["nodule_id"].tolist(), [1, 1, 1])
                before_retry = extracted.call_count
                self.assertEqual(extract_lidc(), 0)
                self.assertEqual(extracted.call_count - before_retry, 5)
                self.assertFalse(errors_path.exists())
                before_resume = extracted.call_count
                self.assertEqual(extract_lidc(), 0)
                self.assertEqual(extracted.call_count, before_resume)
                self.assertEqual(queried.call_count, 2)
            table = pd.read_csv(output)
            self.assertEqual(table.groupby("nodule_id").size().to_dict(), {0: 5, 1: 3})
            self.assertEqual(table.loc[table["nodule_id"] == 0, "slice_index"].tolist(), [1, 2, 3, 4, 5])
            self.assertEqual(table.loc[table["nodule_id"] == 0, "slice_in_nodule"].tolist(), [0, 1, 2, 3, 4])
            self.assertEqual(table["fold"].tolist(), [1] * 8)
            self.assertEqual(table["target_regression_mean"].tolist(), [2.5] * 5 + [4.] * 3)
            self.assertEqual(table["target_binary_mean"].tolist(), [0] * 5 + [1] * 3)
            data, features = load_data(output, "lidc", paths=paths, multi_slice=True)
            self.assertEqual(len(data), 8)
            self.assertEqual(features, ["original_firstorder_Mean"])
            settings = json.loads((output.with_suffix(".scans") / "slices.json").read_text())
            self.assertEqual(settings, {"slice_count": 5})
            for path, before in originals.items():
                self.assertEqual(path.read_bytes(), before)

    def test_segmentation_cli_keeps_consensus_bin_override_and_custom_output(self):
        with TemporaryDirectory(dir="/tmp/opencode", prefix="lidc-seg-stack-") as temporary:
            root = Path(temporary)
            image, paths, scan = lidc_fixture(root)
            values = np.zeros(tuple(reversed(image.GetSize())), dtype=np.uint8)
            values[1:6, 2:7, 3:8] = 1
            mask = sitk.GetImageFromArray(values)
            mask.CopyInformation(image)
            output = root / "custom.csv"
            args = ["extract", "--roi", "segmentation", "--slices", "5", "--consensus", "0.75",
                    "--bin-width", "10", "--output", str(output)]
            with patch("sys.argv", args), patch("LIDCfeatureExtraction.load_paths", return_value=paths), \
                    patch("LIDCfeatureExtraction.pl.query") as queried, \
                    patch("LIDCfeatureExtraction.create_lidc_segmentation", return_value=(image, mask, None)) as segmented, \
                    patch("src.multislice_extraction.extract_features", side_effect=lambda image, mask, extractor, clip:
                          {"original_firstorder_Entropy": extractor.settings["binWidth"]}), \
                    contextlib.redirect_stdout(io.StringIO()):
                queried.return_value.filter.return_value.one.return_value = scan
                self.assertEqual(extract_lidc(), 0)
                self.assertEqual(segmented.call_args.args[-2:], (0.75, 3))
            table = pd.read_csv(output)
            self.assertEqual(table["original_firstorder_Entropy"].tolist(), [10.] * 6)
            self.assertEqual(table.groupby("nodule_id")["slice_index"].apply(list).to_dict(), {0: [1, 2, 3], 1: [1, 2, 3]})
            settings = json.loads((output.with_suffix(".scans") / "settings.json").read_text())
            self.assertEqual(settings["consensus"], 0.75)
            self.assertEqual(settings["bin_width"], 10)

    def test_invalid_counts_and_3d_combinations_fail_before_loading_inputs(self):
        for main in (extract_lidc, extract_lndb):
            for args in (["--slices", "0"], ["--slices=-1"],
                         ["--roi", "sphere", "--dimension", "3", "--slices", "5"],
                         ["--roi", "segmentation", "--dimension", "3", "--slices", "5"]):
                with self.subTest(dataset=main.__module__, args=args), patch("sys.argv", ["extract", *args]), \
                        patch(f"{main.__module__}.load_paths") as loaded, contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        main()
                    self.assertEqual(error.exception.code, 2)
                    loaded.assert_not_called()

    def test_checkpoints_cannot_change_counts_or_switch_to_single_slice(self):
        with TemporaryDirectory(dir="/tmp/opencode", prefix="stack-resume-") as temporary:
            root = Path(temporary)
            config = ROOT / "configs/radiomics_circle_2d.yaml"
            output = root / "multi.csv"
            directory = prepare_run_slices(output, config, 11, False, [], slice_count=5)
            self.assertEqual(prepare_run_slices(output, config, 11, False, [], slice_count=5), directory)
            for count in (1, 3, 7):
                with self.subTest(count=count), self.assertRaisesRegex(ValueError, "Slice count changed"):
                    prepare_run_slices(output, config, 11, False, [], slice_count=count)
            with self.assertRaisesRegex(ValueError, "Settings or inputs changed"):
                prepare_run_slices(output, config, 11, False, [], slice_count=5, bin_width=10)
            single = root / "single.csv"
            prepare_run_slices(single, config, 11, False, [], slice_count=1)
            with self.assertRaisesRegex(ValueError, "no matching slice settings"):
                prepare_run_slices(single, config, 11, False, [], slice_count=5)


class LNDbStackTests(unittest.TestCase):
    def test_training_preserves_targets_folds_and_default_output_name(self):
        with TemporaryDirectory(dir="/tmp/opencode", prefix="lndb-stack-") as temporary:
            root = Path(temporary)
            image, paths = lndb_fixture(root)
            originals = {path: path.read_bytes() for path in paths["lndb"].values() if path.is_file()}
            output = root / "lndb_circle_r6_5slice_hu.csv"
            args = ["extract", "--roi", "circle", "--radius", "6", "--slices", "5"]
            with patch("sys.argv", args), patch("LNDBfeatureExtraction.load_paths", return_value=paths), \
                    patch("LNDBfeatureExtraction.load_scan", return_value=image) as loaded, \
                    patch("LNDBfeatureExtraction.load_reader_mask") as readers, \
                    patch("src.multislice_extraction.extract_features", side_effect=lambda image, mask, extractor, clip:
                          {"original_firstorder_Mean": float(sitk.GetArrayViewFromImage(image)[0, 0, 0])}) as extracted, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(extract_lndb(), 0)
                self.assertEqual(extracted.call_count, 8)
                self.assertEqual(extract_lndb(), 0)
                self.assertEqual(extracted.call_count, 8)
                loaded.assert_called_once()
                readers.assert_not_called()
            table = pd.read_csv(output)
            self.assertEqual(table.groupby("FindingID").size().to_dict(), {7: 5, 8: 3})
            self.assertEqual(table.loc[table["FindingID"] == 7, "slice_index"].tolist(), [1, 2, 3, 4, 5])
            self.assertEqual(table.loc[table["FindingID"] == 8, "slice_in_nodule"].tolist(), [0, 1, 2])
            self.assertEqual(table["target_regression_mean"].tolist(), [2.5] * 5 + [4.] * 3)
            self.assertEqual(table["ratings"].tolist(), ["[2,3]"] * 5 + ["[4]"] * 3)
            self.assertEqual(table["fold"].tolist(), [2] * 8)
            data, features = load_data(output, "lndb", paths=paths, multi_slice=True)
            self.assertEqual(len(data), 8)
            self.assertEqual(features, ["original_firstorder_Mean"])
            for path, before in originals.items():
                self.assertEqual(path.read_bytes(), before)

    def test_test_nodules_are_atomic_and_failed_nodules_are_retried(self):
        with TemporaryDirectory(dir="/tmp/opencode", prefix="lndb-test-stack-") as temporary:
            root = Path(temporary)
            image, paths = lndb_fixture(root)
            nodules = pd.read_csv(paths["lndb"]["findings"])[["LNDbID", "FindingID", "x", "y", "z"]]
            nodules["prediction"] = 5
            test_path = root / "test.csv"
            nodules.to_csv(test_path, index=False)
            original = test_path.read_bytes()
            output = root / "custom_test.csv"
            failed_once = False

            def extract(image, mask, extractor, clip):
                nonlocal failed_once
                check_mask_geometry(image, mask)
                self.assertEqual(extractor.settings["binWidth"], 10)
                self.assertTrue(clip)
                value = float(sitk.GetArrayViewFromImage(image)[0, 0, 0])
                if value == 2 and not failed_once:
                    failed_once = True
                    raise ValueError("slice failed")
                return {"original_firstorder_Mean": value}

            args = ["extract", "--roi", "circle", "--radius", "6", "--slices", "5", "--clip", "--bin-width", "10",
                    "--test-nodules", str(test_path), "--output", str(output)]
            with patch("sys.argv", args), patch("LNDBfeatureExtraction.load_paths", return_value=paths), \
                    patch("LNDBfeatureExtraction.load_scan", return_value=image) as loaded, \
                    patch("LNDBfeatureExtraction.load_reader_mask") as readers, \
                    patch("LNDBfeatureExtraction.match_nodules_to_masks") as matched, \
                    patch("src.multislice_extraction.extract_features", side_effect=extract) as extracted, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(extract_lndb(), 1)
                self.assertEqual(pd.read_csv(output)["FindingID"].tolist(), [8] * 3)
                checkpoint = output.with_suffix(".scans") / "LNDb-0001.features.csv"
                self.assertEqual(pd.read_csv(checkpoint)["FindingID"].tolist(), [8] * 3)
                errors = output.with_suffix(".scans") / "LNDb-0001.errors.csv"
                self.assertEqual(pd.read_csv(errors)["FindingID"].tolist(), [7])
                before_retry = extracted.call_count
                self.assertEqual(extract_lndb(), 0)
                self.assertEqual(extracted.call_count - before_retry, 5)
                self.assertFalse(errors.exists())
                before_resume = extracted.call_count
                self.assertEqual(extract_lndb(), 0)
                self.assertEqual(extracted.call_count, before_resume)
                self.assertEqual(loaded.call_count, 2)
                readers.assert_not_called()
                matched.assert_not_called()
            table = pd.read_csv(output)
            self.assertEqual(len(table), 8)
            self.assertNotIn("prediction", table.columns)
            self.assertNotIn("fold", table.columns)
            self.assertFalse(table.duplicated(["LNDbID", "FindingID", "slice_in_nodule"]).any())
            self.assertEqual(test_path.read_bytes(), original)

    def test_segmentation_uses_corrected_reader_mapping_and_requested_consensus(self):
        with TemporaryDirectory(dir="/tmp/opencode", prefix="lndb-seg-stack-") as temporary:
            root = Path(temporary)
            image, paths = lndb_fixture(root)
            gt = pd.read_csv(paths["lndb"]["gt"]).iloc[:1].copy()
            gt["RadID"], gt["RadFindingID"] = "1,3", "4,9"
            gt.to_csv(paths["lndb"]["gt"], index=False)
            findings = pd.read_csv(paths["lndb"]["findings"], dtype={name: str for name in ("RadID", "RadFinding", "Nodule")})
            findings.loc[0, ["RadID", "RadFinding", "Nodule"]] = ["4,5", "4,9", "1,1"]
            findings.to_csv(paths["lndb"]["findings"], index=False)
            first = np.zeros(tuple(reversed(image.GetSize())), dtype=np.uint8)
            first[1:6, 2:7, 3:8] = 4
            third = np.zeros_like(first)
            third[2:5, 3:6, 4:7] = 9
            masks = {}
            for reader, values in ((1, first), (3, third)):
                mask = sitk.GetImageFromArray(values)
                mask.CopyInformation(image)
                masks[reader] = mask
            output = root / "lndb_ge3_segmentation_2d_c75_5slice_bw10_hu.csv"
            args = ["extract", "--roi", "segmentation", "--slices", "5", "--consensus", "0.75", "--bin-width", "10"]

            def extract(selected, mask, extractor, clip):
                index = round(selected.GetOrigin()[2] / image.GetSpacing()[2])
                np.testing.assert_array_equal(sitk.GetArrayViewFromImage(mask)[0], ((first == 4) & (third == 9))[index])
                return {"original_firstorder_Mean": extractor.settings["binWidth"]}

            with patch("sys.argv", args), patch("LNDBfeatureExtraction.load_paths", return_value=paths), \
                    patch("LNDBfeatureExtraction.load_scan", return_value=image), \
                    patch("LNDBfeatureExtraction.load_reader_mask", side_effect=lambda path, scan, reader: masks[reader]) as readers, \
                    patch("src.multislice_extraction.extract_features", side_effect=extract), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(extract_lndb(), 0)
            self.assertEqual([call.args[-1] for call in readers.call_args_list], [1, 3])
            table = pd.read_csv(output)
            self.assertEqual(table["slice_index"].tolist(), [2, 3, 4])
            self.assertEqual(table["slice_in_nodule"].tolist(), [0, 1, 2])
            self.assertEqual(table["FindingID"].tolist(), [7] * 3)
            self.assertEqual(table["target_regression_mean"].tolist(), [2.5] * 3)
            self.assertEqual(table["original_firstorder_Mean"].tolist(), [10.] * 3)
            mapping = pd.read_csv(output.with_suffix(".scans") / "mask_mapping.csv")
            self.assertEqual(mapping["gt_FindingID"].tolist(), [70])
            settings = json.loads((output.with_suffix(".scans") / "settings.json").read_text())
            self.assertEqual(settings["consensus"], 0.75)
            self.assertEqual(settings["bin_width"], 10)

    def test_multislice_test_segmentation_is_rejected(self):
        with patch("sys.argv", ["extract", "--roi", "segmentation", "--slices", "5", "--test-nodules", "unused.csv"]), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            extract_lndb()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
