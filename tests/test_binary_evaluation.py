"""Operating points and disjoint patient-fold threshold calibration."""
import contextlib
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.binary_evaluation import (binary_metrics, calibration_splits, positive_score,
                                   run_calibration, zero_fn_summary, zero_fn_threshold)


class BinaryEvaluationTests(unittest.TestCase):
    def test_threshold_keeps_tied_positives_and_is_the_highest_valid_one(self):
        target, score = [0, 1, 0, 1], [0.1, 0.2, 0.2, 0.8]
        threshold = zero_fn_threshold(target, score)
        self.assertEqual(threshold, 0.2)
        result = binary_metrics(target, np.asarray(score) >= threshold)
        self.assertEqual((result["FN"], result["FP"]), (0, 1))
        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["specificity"], 0.5)
        self.assertGreater(binary_metrics(target, np.asarray(score) >= np.nextafter(threshold, 1))["FN"], 0)

    def test_calibration_does_not_guarantee_recall_on_unseen_rows(self):
        threshold = zero_fn_threshold([1, 0, 1], [0.3, 0.4, 0.8])
        result = binary_metrics([1, 0], np.asarray([0.2, 0.1]) >= threshold)
        self.assertEqual(result["FN"], 1)

    def test_summary_does_not_round_away_zero_threshold(self):
        result = zero_fn_summary([0, 1, 1], [0.0, 0.0, 0.8])
        self.assertEqual(result["threshold"], 0.0)
        self.assertEqual(result["FN"], 0)
        self.assertEqual(result["specificity"], 0)

    def test_invalid_inputs_fail(self):
        for target, score in [([], []), ([0], [0.1, 0.3]), ([2], [0.3]), ([1], [np.nan])]:
            with self.assertRaises(ValueError):
                zero_fn_threshold(target, score)
        with self.assertRaisesRegex(ValueError, "positive"):
            zero_fn_threshold([0, 0], [0.1, 0.2])
        with self.assertRaisesRegex(ValueError, "binary"):
            binary_metrics([0, 1], [0.2, 1])
        with self.assertRaisesRegex(ValueError, "both"):
            zero_fn_summary([1, 1], [0.1, 0.2])

    def test_patient_splits_are_disjoint_and_cover_each_test_row_once(self):
        table = pd.DataFrame({"LNDbID": np.repeat(range(10), 2), "fold": np.repeat([1, 2, 3, 4, 5] * 2, 2)})
        tested = np.zeros(len(table), dtype=int)
        for fold, calibration_fold, train, calibration, test in calibration_splits(table):
            self.assertNotEqual(fold, calibration_fold)
            self.assertTrue((train.astype(int) + calibration.astype(int) + test.astype(int)).eq(1).all())
            train_patients = set(table.loc[train, "LNDbID"])
            self.assertTrue(train_patients.isdisjoint(table.loc[calibration | test, "LNDbID"]))
            self.assertTrue(set(table.loc[test, "LNDbID"]).isdisjoint(table.loc[calibration, "LNDbID"]))
            tested += test.to_numpy(dtype=int)
        np.testing.assert_array_equal(tested, 1)

    def test_patient_in_multiple_folds_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "patient"):
            list(calibration_splits(pd.DataFrame({"LNDbID": [1, 1, 2], "fold": [1, 2, 3]})))

    def test_probability_uses_class_labels_not_positions(self):
        class Model:
            classes_ = np.array([5, 1, 4, 3, 2])

            def predict_proba(self, values):
                return np.array([[0.2, 0.1, 0.3, 0.1, 0.3]])

        np.testing.assert_allclose(positive_score(Model(), [[1]], binary=False), [0.5])
        np.testing.assert_allclose(positive_score(Model(), [[1]], binary=True), [0.1])

    def test_experiment_does_not_fit_on_calibration_or_test_patients(self):
        features = ["original_firstorder_Mean", "original_firstorder_UID"]
        table = pd.DataFrame([
            {"LNDbID": 10 * fold + j, "FindingID": 1, "fold": fold,
             "target_binary_mean": j % 2, "target_classification_mean": 4 if j % 2 else 2,
             features[0]: 0.8 if j % 2 else 0.1, features[1]: 10 * fold + j}
            for fold in range(1, 6) for j in range(4)
        ])
        lidc = table.iloc[:4].copy()
        lidc[features[1]] += 1000
        fitted_ids = []

        class Model:
            def __init__(self, binary):
                self.classes_ = np.array([0, 1] if binary else [1, 2, 3, 4, 5])

            def predict_proba(self, values):
                score = values[features[0]].to_numpy()
                if len(self.classes_) == 2:
                    return np.column_stack([1 - score, score])
                return np.column_stack([np.zeros(len(score)), 1 - score,
                                        np.zeros(len(score)), score, np.zeros(len(score))])

        def fit(model, values, target, weight, offset):
            fitted_ids.append(set(values[features[1]]))
            return model

        with TemporaryDirectory(dir="/tmp/opencode") as directory:
            output = Path(directory)
            with patch("src.binary_evaluation.load_paths", return_value={"lndb": {"segmentation_targets": "targets.csv"}}), \
                    patch("src.binary_evaluation.load_data", side_effect=[(table, features), (lidc, features)]) as load, \
                    patch("src.binary_evaluation.make_model", side_effect=lambda *args, **kw: Model(kw["binary"])), \
                    patch("src.binary_evaluation.fit_model", side_effect=fit), \
                    contextlib.redirect_stdout(io.StringIO()):
                run_calibration(output)
            self.assertTrue(load.call_args_list[0].kwargs["exclude_duplicate_lndb"])
            for index, ids in enumerate(fitted_ids):
                fold = index % 5 + 1
                calibration_fold = fold % 5 + 1
                excluded = set(table.loc[table.fold.isin([fold, calibration_fold]), features[1]])
                self.assertTrue(ids.isdisjoint(excluded))
                self.assertTrue(set(lidc[features[1]]).issubset(ids))
            predictions = pd.read_csv(output / "validation_predictions.csv")
            self.assertEqual(len(predictions), 80)
            self.assertFalse(predictions.duplicated(["model", "LNDbID", "FindingID"]).any())
            self.assertTrue(predictions.prediction.eq(predictions.target_binary_mean).all())

            with patch("src.binary_evaluation.load_paths", return_value={"lndb": {"segmentation_targets": "targets.csv"}}), \
                    patch("src.binary_evaluation.load_data", side_effect=[(table, features), (lidc, features)]), \
                    patch("src.binary_evaluation.make_model", side_effect=lambda *args, **kw: Model(kw["binary"])) as factory, \
                    patch("src.binary_evaluation.fit_model", side_effect=fit), \
                    contextlib.redirect_stdout(io.StringIO()):
                run_calibration(output, top_k=1)
            self.assertEqual(factory.call_count, 5)
            self.assertTrue(all(call.kwargs["k"] == 1 and call.kwargs["model_kind"] == "xgboost"
                                for call in factory.call_args_list))
            selected = pd.read_csv(output / "validation_predictions.csv")
            self.assertEqual(len(selected), 20)
            self.assertEqual(selected.model.unique().tolist(), ["Binary XGBoost, top 1 features"])
            self.assertTrue(selected.prediction.eq(selected.target_binary_mean).all())
            for fold, ids in enumerate(fitted_ids[-5:], start=1):
                excluded = set(table.loc[table.fold.isin([fold, fold % 5 + 1]), features[1]])
                self.assertTrue(ids.isdisjoint(excluded))


if __name__ == "__main__":
    unittest.main()
