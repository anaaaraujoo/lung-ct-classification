"""Diameter filtering and high-sensitivity tuning stay inside training folds."""
import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.binary_tuning import (DIAMETER_COLUMN, diameter_filtered_features, parameter_candidates,
                               run_experiment, tune_parameters)


class BinaryTuningTests(unittest.TestCase):
    def test_filter_removes_positive_and_negative_size_proxies(self):
        train = pd.DataFrame({DIAMETER_COLUMN: [1, 2, 3, 4, 5, 6],
                              "volume_proxy": [1, 8, 27, 64, 125, 216],
                              "inverse_size": [6, 5, 4, 3, 2, 1],
                              "texture": [2, 5, 1, 6, 4, 3]})
        features = list(train)
        self.assertEqual(diameter_filtered_features(train, features, 0.7), ["texture"])
        self.assertEqual(diameter_filtered_features(train, features), features)

    def test_invalid_or_empty_filter_fails(self):
        train = pd.DataFrame({DIAMETER_COLUMN: [1, 2, 3]})
        for maximum in [-1, 0, 1, 2]:
            with self.assertRaises(ValueError):
                diameter_filtered_features(train, list(train), maximum)
        with self.assertRaisesRegex(ValueError, "No features"):
            diameter_filtered_features(train, list(train), 0.7)

    def test_grids_include_unchanged_baseline(self):
        for kind in ["xgboost", "brf"]:
            self.assertEqual(parameter_candidates(kind)[0], {})
        with self.assertRaises(ValueError):
            parameter_candidates("unknown")

    def test_search_uses_disjoint_inner_fit_calibration_and_validation_rows(self):
        features = ["original_firstorder_Mean", "original_firstorder_UID"]
        table = pd.DataFrame([
            {"LNDbID": 10 * fold + j, "FindingID": 1, "fold": fold,
             "target_binary_mean": j % 2, features[0]: 0.8 if j % 2 else 0.1,
             features[1]: 10 * fold + j}
            for fold in [1, 3, 5] for j in range(4)
        ])
        lidc = table.iloc[:4].copy()
        lidc[features[1]] += 1000
        events = []

        class Model:
            classes_ = np.array([0, 1])

            def __init__(self, params):
                self.noise = bool(params)

            def predict_proba(self, values):
                events.append(("score", set(values[features[1]])))
                score = np.full(len(values), 0.5) if self.noise else values[features[0]].to_numpy()
                return np.column_stack([1 - score, score])

        def fit(model, values, target, weight, offset):
            events.append(("fit", set(values[features[1]])))
            return model

        with patch("src.binary_tuning.parameter_candidates", return_value=[{"noisy": True}, {}]), \
                patch("src.binary_tuning.new_model", side_effect=lambda kind, params: (Model(params), None)), \
                patch("src.binary_tuning.fit_model", side_effect=fit), \
                contextlib.redirect_stdout(io.StringIO()):
            best, result = tune_parameters(lidc, table, features, "brf")
        self.assertEqual(best, {})
        self.assertEqual(result.loc[result.selected, "specificity"].item(), 1)
        self.assertEqual(result.loc[result.selected, "FN"].item(), 0)
        self.assertEqual(json.loads(result.loc[result.selected, "params"].item()), {})
        for index in range(0, len(events), 3):
            fit_rows, calibration, validation = [ids for _, ids in events[index:index + 3]]
            self.assertTrue(fit_rows.isdisjoint(calibration | validation))
            self.assertTrue(calibration.isdisjoint(validation))
            self.assertTrue(set(lidc[features[1]]).issubset(fit_rows))

        # Fewer FN takes priority even if specificity is lower.
        with patch("src.binary_tuning.parameter_candidates", return_value=[{"noisy": True}, {}]), \
                patch("src.binary_tuning.new_model", side_effect=lambda kind, params: (Model(params), None)), \
                patch("src.binary_tuning.fit_model", side_effect=fit), \
                patch("src.binary_tuning.binary_metrics", side_effect=[
                    {"FN": 1, "specificity": 1.0}, {"FN": 0, "specificity": 0.1},
                ]), contextlib.redirect_stdout(io.StringIO()):
            best, _ = tune_parameters(lidc, table, features, "brf")
        self.assertEqual(best, {})

    def test_outer_search_and_feature_filter_never_receive_outer_calibration_or_test(self):
        features = ["original_firstorder_Mean", "original_firstorder_UID"]
        table = pd.DataFrame([
            {"LNDbID": 10 * fold + j, "FindingID": 1, "fold": fold,
             "target_binary_mean": j % 2, features[0]: 0.8 if j % 2 else 0.1,
             features[1]: 10 * fold + j}
            for fold in range(1, 6) for j in range(4)
        ])
        lidc = table.iloc[:4].copy()
        lidc[features[1]] += 1000
        training_ids, search_folds, filter_ids = [], [], []

        class Model:
            classes_ = np.array([0, 1])

            def predict_proba(self, values):
                score = values[features[0]].to_numpy()
                return np.column_stack([1 - score, score])

        def fit(model, values, target, weight, offset):
            training_ids.append(set(values[features[1]]))
            return model

        def search(other, source, columns, kind, maximum):
            search_folds.append(set(source.fold))
            return {}, pd.DataFrame([{"candidate": 0, "params": "{}", "selected": True}])

        def filtered(train, columns, maximum):
            filter_ids.append(set(train[features[1]]))
            return columns

        with TemporaryDirectory(dir="/tmp/opencode") as directory:
            output = Path(directory)
            with patch("src.binary_tuning.load_paths", return_value={"lndb": {"segmentation_targets": "targets.csv"}}), \
                    patch("src.binary_tuning.load_data", side_effect=[(table, features), (lidc, features)]), \
                    patch("src.binary_tuning.new_model", side_effect=lambda kind, params: (Model(), None)), \
                    patch("src.binary_tuning.fit_model", side_effect=fit), \
                    patch("src.binary_tuning.tune_parameters", side_effect=search), \
                    patch("src.binary_tuning.diameter_filtered_features", side_effect=filtered), \
                    contextlib.redirect_stdout(io.StringIO()):
                run_experiment(output, maximum=0.7, tune=True)
            for index, ids in enumerate(training_ids):
                fold = index % 5 + 1
                calibration = fold % 5 + 1
                excluded = set(table.loc[table.fold.isin([fold, calibration]), features[1]])
                self.assertTrue(ids.isdisjoint(excluded))
                self.assertEqual(filter_ids[index], ids)
                self.assertEqual(search_folds[index], set(range(1, 6)) - {fold, calibration})
            predictions = pd.read_csv(output / "validation_predictions.csv")
            self.assertEqual(len(predictions), 40)
            self.assertTrue(predictions.prediction.eq(predictions.target_binary_mean).all())
            metrics = pd.read_csv(output / "metrics.csv")
            self.assertEqual(len(metrics[metrics.fold.eq("overall_oof")]), 2)


if __name__ == "__main__":
    unittest.main()
