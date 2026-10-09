"""Checks for the saved-fold RF tuning, without using competition data."""
import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.modeling import MIN_SAMPLES_GRID, TARGET, lndb_cv, load_data, make_model, tune_rf


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
            with patch("src.modeling.ROOT", root):
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
            with self.assertRaisesRegex(ValueError, "feature rows must have unique"):
                load_data(path, "lndb", targets_path)


if __name__ == "__main__":
    unittest.main()
