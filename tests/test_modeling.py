"""Checks for the saved-fold RF tuning, without using competition data."""
import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.modeling import MIN_SAMPLES_GRID, TARGET, fit_model, lndb_cv, load_data, make_model, metrics, tune_rf
from src.make_predictions import main as make_submission


class RFTuningTests(unittest.TestCase):
    def test_grid_and_defaults(self):
        self.assertEqual(MIN_SAMPLES_GRID["randomforestclassifier__min_samples_leaf"], [1, 5, 10, 20])
        self.assertEqual(MIN_SAMPLES_GRID["randomforestclassifier__min_samples_split"], [2, 10, 20, 40])
        model = make_model("balanced").named_steps["randomforestclassifier"]
        self.assertEqual(model.n_estimators, 500)
        self.assertEqual(model.min_samples_leaf, 1)
        self.assertEqual(model.min_samples_split, 2)
        self.assertIsNone(model.max_depth)
        self.assertEqual(model.class_weight, "balanced")

    def test_inner_splits_keep_lidc_out_of_validation(self):
        # Outer fold 3 has already been removed from the LNDb training rows.
        train = pd.DataFrame({"fold": [1, 3, 5, 1, 1, 2, 2, 4, 4, 5, 5]})
        splits = list(lndb_cv(train, 3).split())
        self.assertEqual(len(splits), 4)
        validation_rows = []
        for training, validation in splits:
            self.assertTrue(set(range(3)).issubset(training))
            self.assertTrue((validation >= 3).all())
            self.assertFalse(set(training) & set(validation))
            self.assertEqual(train.iloc[validation].fold.nunique(), 1)
            self.assertNotEqual(train.iloc[validation].fold.iloc[0], 3)
            validation_rows.extend(validation)
        self.assertEqual(sorted(validation_rows), list(range(3, len(train))))

    def test_lndb_only_splits(self):
        train = pd.DataFrame({"fold": [1, 2, 4, 5]})
        self.assertEqual(len(list(lndb_cv(train, 0).split())), 4)

    def test_search_refits_pipeline_and_saves_results(self):
        train = pd.DataFrame({
            "fold": np.repeat([1, 2, 4, 5], 5),
            TARGET: np.tile([1, 2, 3, 4, 5], 4),
            "original_firstorder_Mean": np.tile([1., 2., 3., 4., 5.], 4),
        })
        train.loc[0, "original_firstorder_Mean"] = np.nan
        features = ["original_firstorder_Mean"]
        # One candidate suffices to check GridSearchCV wiring, retaining 500 trees.
        grid = {"randomforestclassifier__min_samples_leaf": [1],
                "randomforestclassifier__min_samples_split": [2]}
        with TemporaryDirectory(prefix="rf-tuning-test-", dir="/tmp/opencode") as directory:
            output = Path(directory)
            with patch("src.modeling.MIN_SAMPLES_GRID", grid), contextlib.redirect_stdout(io.StringIO()):
                search = tune_rf(make_model("balanced"), train, features, 0, output, "test")
            baseline = make_model("balanced").fit(train[features], train[TARGET])
            np.testing.assert_array_equal(search.predict(train[features]), baseline.predict(train[features]))
            self.assertEqual(search.n_splits_, 4)
            self.assertEqual(search.best_estimator_.named_steps["randomforestclassifier"].n_estimators, 500)
            np.testing.assert_array_equal(search.best_estimator_.named_steps["simpleimputer"].statistics_, [3.])
            scores = pd.read_csv(output / "grid_test.csv")
            self.assertEqual(len(scores), 1)
            self.assertIn("mean_train_score", scores)
            self.assertIn("split3_test_score", scores)


class PopulationTests(unittest.TestCase):
    def test_explicit_targets_select_same_nodules_from_larger_circle_table(self):
        with TemporaryDirectory(prefix="rf-population-test-", dir="/tmp/opencode") as directory:
            root = Path(directory)
            processed = root / "data/processed"
            processed.mkdir(parents=True)
            targets = pd.DataFrame({"LNDbID": [1, 2, 3], "NoduleID": [1, 1, 1],
                                    TARGET: [1, 3, 4], "ratings": ["[1]", "[3]", "[4]"]})
            targets.to_csv(processed / "lndb_nodule_targets.csv", index=False)
            selected_path = processed / "lndb_ge3_nodule_targets.csv"
            targets.iloc[[0, 2]].to_csv(selected_path, index=False)
            pd.DataFrame({"LNDbID": [1, 2, 3], "fold": [1, 2, 5]}).to_csv(
                processed / "lndb_patient_folds.csv", index=False,
            )
            table = pd.DataFrame({"LNDbID": [1, 2, 3], "FindingID": [1, 1, 1],
                                  "original_firstorder_Mean": [10., 20., 30.],
                                  "original_shape2D_PixelSurface": [1., 2., 3.],
                                  "slice_index": [12, 13, 14], "RadID": [1, 2, 3],
                                  "diagnostics_Image_Mean": [1., 2., 3.], TARGET: [5, 5, 5], "fold": [99, 99, 99]})
            path = root / "features.csv"
            table.to_csv(path, index=False)
            locations = {"lndb": {"targets": processed / "lndb_nodule_targets.csv",
                                   "folds": processed / "lndb_patient_folds.csv"}}
            with patch("src.modeling.load_paths", return_value=locations):
                full, features = load_data(path, "lndb")
                selected, _ = load_data(path, "lndb", selected_path)
                self.assertEqual(len(full), 3)
                self.assertEqual(selected["LNDbID"].tolist(), [1, 3])
                self.assertEqual(selected["fold"].tolist(), [1, 5])
                self.assertEqual(selected[TARGET].tolist(), [1, 4])
                self.assertEqual(features, ["original_firstorder_Mean", "original_shape2D_PixelSurface"])
                # A failed extraction cannot silently reduce the requested population.
                table.iloc[[0]].to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, "all 2 training nodules"):
                    load_data(path, "lndb", selected_path)
                with self.assertRaisesRegex(ValueError, "all 3 training nodules"):
                    load_data(path, "lndb")

    def test_duplicate_features_are_rejected_even_outside_selected_population(self):
        with TemporaryDirectory(prefix="rf-population-test-", dir="/tmp/opencode") as directory:
            root = Path(directory)
            targets = pd.DataFrame({"LNDbID": [1], "FindingID": [1], TARGET: [3]})
            targets_path = root / "targets.csv"
            targets.to_csv(targets_path, index=False)
            table = pd.DataFrame({"LNDbID": [1, 2, 2], "FindingID": [1, 1, 1],
                                  "original_firstorder_Mean": [1., 2., 2.]})
            path = root / "features.csv"
            table.to_csv(path, index=False)
            locations = {"lndb": {"targets": targets_path, "folds": root / "unused_folds.csv"}}
            with self.assertRaisesRegex(ValueError, "feature rows must have unique"):
                load_data(path, "lndb", targets_path, locations)


class BinaryTests(unittest.TestCase):
    def test_binary_model_cannot_fill_competition_submission(self):
        with patch("sys.argv", ["make_predictions.py", "--model", "unused.joblib"]), \
                patch("src.make_predictions.joblib.load", return_value={"target": "target_binary_mean"}):
            with self.assertRaisesRegex(ValueError, "five-class model"):
                make_submission()

    def test_saved_binary_targets_select_population_and_preserve_folds(self):
        with TemporaryDirectory(prefix="binary-population-", dir="/tmp/opencode") as directory:
            root = Path(directory)
            targets = pd.DataFrame({"LNDbID": [1, 2, 3, 4], "NoduleID": [1, 1, 1, 1],
                                    TARGET: [3, 3, 3, 4], "target_binary_mean": [0, np.nan, 1, 1]})
            targets.to_csv(root / "targets.csv", index=False)
            pd.DataFrame({"LNDbID": [1, 2, 3, 4], "fold": [5, 2, 3, 1]}).to_csv(root / "folds.csv", index=False)
            table = pd.DataFrame({"LNDbID": [1, 2, 3, 4], "FindingID": [1, 1, 1, 1],
                                  "original_firstorder_Mean": [10., 20., 30., 40.],
                                  "original_shape2D_PixelSurface": [1., 2., 3., 4.],
                                  "target_binary_mean": [1, 1, 0, 0], "fold": [99, 99, 99, 99]})
            path = root / "features.csv"
            table.to_csv(path, index=False)
            locations = {"lndb": {"targets": root / "targets.csv", "folds": root / "folds.csv"}}
            data, features = load_data(path, "lndb", paths=locations, binary=True)
            self.assertEqual(data["LNDbID"].tolist(), [1, 3, 4])
            self.assertEqual(data["target_binary_mean"].tolist(), [0, 1, 1])
            self.assertEqual(data["fold"].tolist(), [5, 3, 1])
            self.assertEqual(features, ["original_firstorder_Mean", "original_shape2D_PixelSurface"])
            # Missing features are permitted only for the excluded, undefined target.
            table[table["LNDbID"] != 2].to_csv(path, index=False)
            self.assertEqual(len(load_data(path, "lndb", paths=locations, binary=True)[0]), 3)
            table[table["LNDbID"] == 1].to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "all 3 training nodules"):
                load_data(path, "lndb", paths=locations, binary=True)

    def test_binary_metrics_and_rf_defaults(self):
        score = metrics([0, 0, 1, 1], [0, 1, 1, 1], binary=True)
        self.assertAlmostEqual(score["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(score["macro_f1"], (2 / 3 + 0.8) / 2)
        self.assertAlmostEqual(score["mae"], 0.25)
        five_class = make_model("balanced").named_steps["randomforestclassifier"]
        binary = make_model("balanced", binary=True).named_steps["randomforestclassifier"]
        self.assertEqual(five_class.get_params(), binary.get_params())

    def test_binary_xgboost_keeps_class_weights_with_zero_label_offset(self):
        model = make_model("balanced", model_kind="xgboost", binary=True)
        classifier = model.named_steps["xgbclassifier"]
        self.assertEqual(classifier.objective, "binary:logistic")
        self.assertEqual(classifier.eval_metric, "logloss")
        X = pd.DataFrame({"original_firstorder_Mean": [0., 0., 0., 1.]})
        y = pd.Series([0, 0, 0, 1])
        with patch.object(model, "fit", return_value=model) as fitted:
            fit_model(model, X, y, "balanced", 0)
        np.testing.assert_array_equal(fitted.call_args.args[1], y)
        np.testing.assert_allclose(fitted.call_args.kwargs["xgbclassifier__sample_weight"], [2/3, 2/3, 2/3, 2])

    def test_binary_rf_tuning_uses_binary_labels_and_target(self):
        train = pd.DataFrame({"fold": np.repeat([1, 2, 4, 5], 4), TARGET: [5] * 16,
                              "target_binary_mean": np.tile([0, 0, 1, 1], 4),
                              "original_firstorder_Mean": np.tile([0., 0., 1., 1.], 4)})
        grid = {"randomforestclassifier__min_samples_leaf": [1], "randomforestclassifier__min_samples_split": [2]}
        with TemporaryDirectory(prefix="binary-tuning-", dir="/tmp/opencode") as directory:
            with patch("src.modeling.MIN_SAMPLES_GRID", grid), contextlib.redirect_stdout(io.StringIO()):
                search = tune_rf(make_model("balanced", binary=True), train, ["original_firstorder_Mean"],
                                 0, Path(directory), "test", binary=True)
            self.assertEqual(search.best_score_, 1.)
            np.testing.assert_array_equal(search.best_estimator_.classes_, [0, 1])


if __name__ == "__main__":
    unittest.main()
