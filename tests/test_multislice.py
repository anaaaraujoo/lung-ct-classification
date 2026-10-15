"""Merge compatibility for multi-slice inputs, binary targets and saved predictions."""
import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd

from src.make_predictions import main as make_submission
from src.modeling import EXCLUDED_LNDB_IDS, SLICE_IN_NODULE, TARGET, aggregate_slices, load_data, main, metrics

KEYS = ["LNDbID", "FindingID"]
FEATURE = "original_firstorder_Mean"


class MultiSliceTests(unittest.TestCase):
    def test_ct_exclusion_removes_every_nodule_and_slice_before_fold_assignment(self):
        with TemporaryDirectory(dir="/tmp/opencode", prefix="multislice-dedup-") as temporary:
            root = Path(temporary)
            targets = pd.DataFrame({"LNDbID": [161, 161, 181, 181], "NoduleID": [1, 2, 1, 2],
                                    TARGET: [2, 4, 3, 5], "target_binary_mean": [0, 1, np.nan, 1]})
            targets.to_csv(root / "targets.csv", index=False)
            pd.DataFrame({"LNDbID": [161, 181], "fold": [1, 3]}).to_csv(root / "folds.csv", index=False)
            table = targets.loc[targets.index.repeat(5)].rename(columns={"NoduleID": "FindingID"}).reset_index(drop=True)
            table[SLICE_IN_NODULE] = np.tile(np.arange(5), 4)
            table[FEATURE] = np.arange(len(table), dtype=float)
            table.to_csv(root / "features.csv", index=False)
            paths = {"lndb": {"targets": root / "targets.csv", "folds": root / "folds.csv"}}
            for binary in (False, True):
                with self.subTest(binary=binary):
                    full, _ = load_data(root / "features.csv", "lndb", paths=paths, binary=binary, multi_slice=True)
                    self.assertEqual(len(full), 15 if binary else 20)
                    data, features = load_data(root / "features.csv", "lndb", paths=paths, binary=binary,
                                               multi_slice=True, exclude_duplicate_lndb=True)
                    self.assertEqual(len(data), 10)
                    self.assertEqual(data["LNDbID"].tolist(), [161] * 10)
                    self.assertEqual(data["FindingID"].tolist(), [1] * 5 + [2] * 5)
                    self.assertEqual(data[SLICE_IN_NODULE].tolist(), list(range(5)) * 2)
                    self.assertEqual(data["fold"].tolist(), [1] * 10)
                    self.assertEqual(features, [FEATURE])

    def test_triangular_weights_sort_slices_and_keep_nodules_separate(self):
        rows = pd.DataFrame({"LNDbID": [2, 1, 1, 1, 1, 1], "FindingID": [1] * 6,
                             SLICE_IN_NODULE: [0, 4, 2, 0, 3, 1]})
        prediction = [4, 1, 5, 1, 1, 1]
        result = aggregate_slices(rows, prediction, KEYS)
        self.assertEqual(result.loc[(2, 1)], 4)
        self.assertAlmostEqual(result.loc[(1, 1)], 21 / 9)

    def test_load_multislice_uses_saved_targets_folds_and_binary_population(self):
        with TemporaryDirectory(dir="/tmp/opencode", prefix="multislice-load-") as temporary:
            root = Path(temporary)
            targets = pd.DataFrame({"LNDbID": [1, 2, 3], "NoduleID": [1] * 3,
                                    TARGET: [3, 3, 4], "target_binary_mean": [0, np.nan, 1]})
            targets.to_csv(root / "targets.csv", index=False)
            pd.DataFrame({"LNDbID": [1, 2, 3], "fold": [1, 3, 5]}).to_csv(root / "folds.csv", index=False)
            table = pd.DataFrame({"LNDbID": [1, 1, 2, 2, 3, 3], "FindingID": [1] * 6,
                                  SLICE_IN_NODULE: [0, 1] * 3, FEATURE: [1., 2., 3., 4., 5., 6.],
                                  "original_shape_MeshVolume": [10.] * 6, TARGET: [5] * 6, "fold": [99] * 6})
            table.to_csv(root / "features.csv", index=False)
            paths = {"lndb": {"targets": root / "targets.csv", "folds": root / "folds.csv"}}
            data, features = load_data(root / "features.csv", "lndb", paths=paths, multi_slice=True)
            self.assertEqual(features, [FEATURE, "original_shape_MeshVolume"])
            self.assertEqual(data[TARGET].tolist(), [3, 3, 3, 3, 4, 4])
            self.assertEqual(data["fold"].tolist(), [1, 1, 3, 3, 5, 5])
            binary, _ = load_data(root / "features.csv", "lndb", paths=paths, binary=True, multi_slice=True)
            self.assertEqual(binary["LNDbID"].tolist(), [1, 1, 3, 3])
            self.assertEqual(binary["target_binary_mean"].tolist(), [0, 0, 1, 1])
            with self.assertRaisesRegex(ValueError, "unique"):
                load_data(root / "features.csv", "lndb", paths=paths)
            for invalid in (table.drop(columns=SLICE_IN_NODULE), table.assign(slice_in_nodule=np.nan),
                            pd.concat([table, table.iloc[[0]]])):
                invalid.to_csv(root / "features.csv", index=False)
                with self.assertRaises(ValueError):
                    load_data(root / "features.csv", "lndb", paths=paths, multi_slice=True)
            table[table["LNDbID"] != 3].to_csv(root / "features.csv", index=False)
            with self.assertRaisesRegex(ValueError, "all 3 training nodules"):
                load_data(root / "features.csv", "lndb", paths=paths, multi_slice=True)

    def test_cli_saves_one_prediction_per_nodule_for_both_targets(self):
        with TemporaryDirectory(dir="/tmp/opencode", prefix="multislice-cli-") as temporary:
            root = Path(temporary)
            targets = pd.DataFrame({"LNDbID": [*range(1, 25), 181], "NoduleID": [1] * 25,
                                    TARGET: np.tile([1, 2, 3, 4, 5], 5), "fold": np.repeat(np.arange(1, 6), 5)})
            targets["target_binary_mean"] = np.where(targets[TARGET] == 3, np.nan, (targets[TARGET] > 3).astype(float))
            targets.drop(columns="fold").to_csv(root / "targets.csv", index=False)
            targets[["LNDbID", "fold"]].to_csv(root / "folds.csv", index=False)
            table = targets.loc[targets.index.repeat(3), ["LNDbID", TARGET]].reset_index(drop=True)
            table["FindingID"] = 1
            table[SLICE_IN_NODULE] = np.tile([0, 1, 2], 25)
            table[FEATURE] = table[TARGET].astype(float)
            table.to_csv(root / "features.csv", index=False)
            paths = {"lndb": {"targets": root / "targets.csv", "folds": root / "folds.csv"},
                     "lidc": {"targets": root / "unused.csv"},
                     "training": {"lidc_features": root / "unused.csv", "lndb_features": root / "features.csv"},
                     "outputs": {"models": root}}
            for binary, exclude in ((False, False), (True, False), (False, True), (True, True)):
                args = ["modeling", "--lndb-only", "--multi-slice", "--class-weight", "balanced"]
                if binary:
                    args += ["--binary"]
                if exclude:
                    args += ["--exclude-duplicate-lndb"]
                with self.subTest(binary=binary, exclude=exclude), patch("sys.argv", args), \
                         patch("src.modeling.load_paths", return_value=paths), contextlib.redirect_stdout(io.StringIO()):
                    main()
                output = root / f"rf_features_balanced{'_binary' if binary else ''}_multislice{'_dedup' if exclude else ''}"
                saved = joblib.load(output / "model.joblib")
                target = "target_binary_mean" if binary else TARGET
                self.assertTrue(saved["multi_slice"])
                self.assertEqual(saved["excluded_lndb_ids"], sorted(EXCLUDED_LNDB_IDS) if exclude else [])
                self.assertEqual(saved["target"], target)
                self.assertEqual(saved["model"].named_steps["randomforestclassifier"].n_estimators, 500)
                oof = pd.read_csv(output / "validation_predictions.csv")
                self.assertEqual(len(oof), (20 if binary else 25) - int(exclude))
                self.assertEqual(181 in oof["LNDbID"].values, not exclude)
                self.assertFalse(oof.duplicated(KEYS).any())
                scores = pd.read_csv(output / "metrics.csv")
                for name, value in metrics(oof[target], oof["prediction"], binary).items():
                    self.assertAlmostEqual(scores.iloc[-1][name], value)
                labels = [0, 1] if binary else [1, 2, 3, 4, 5]
                self.assertTrue(oof["prediction"].isin(labels).all())

    def test_submission_preserves_remote_rounding_offsets_and_nodule_fallback(self):
        with TemporaryDirectory(dir="/tmp/opencode", prefix="multislice-predictions-") as temporary:
            root = Path(temporary)
            table = pd.DataFrame({"LNDbID": [1] * 5 + [2], "FindingID": [1] * 6,
                                  SLICE_IN_NODULE: [4, 2, 0, 3, 1, 0], FEATURE: [1.] * 6})
            table.to_csv(root / "features.csv", index=False)
            pd.DataFrame({"LNDbID": [1, 2, 3], "FindingID": [1] * 3, "prediction": [1] * 3}).to_csv(root / "submission.csv", index=False)
            model = Mock()
            # The first nodule rounds to class 4, the second predicts class 2.
            model.predict.return_value = np.array([0, 4, 0, 4, 4, 1])
            saved = {"model": model, "features": [FEATURE], "multi_slice": True, "label_offset": 1, "target": TARGET}
            args = ["make_predictions", "--model", "unused.joblib", "--features", str(root / "features.csv"),
                    "--submission", str(root / "submission.csv"), "--output", str(root / "predictions.csv")]
            with patch("sys.argv", args), patch("src.make_predictions.joblib.load", return_value=saved), \
                    contextlib.redirect_stdout(io.StringIO()):
                make_submission()
            result = pd.read_csv(root / "predictions.csv")
            self.assertEqual(result["prediction"].tolist(), [4, 2, 2])
            self.assertEqual(pd.read_csv(root / "submission.csv")["prediction"].tolist(), [1, 1, 1])
            table.iloc[[0, 5]].drop(columns=SLICE_IN_NODULE).to_csv(root / "features.csv", index=False)
            with patch("sys.argv", args), patch("src.make_predictions.joblib.load", return_value=saved), \
                    self.assertRaisesRegex(ValueError, "slice_in_nodule"):
                make_submission()


if __name__ == "__main__":
    unittest.main()
