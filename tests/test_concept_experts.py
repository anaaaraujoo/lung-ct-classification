"""Reader concepts, patient-safe risk heads and separately calibrated maximum."""
import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src.auxiliary import ATTRIBUTES, CONCEPTS, load_concepts
from src.binary_evaluation import calibration_splits
from src.concept_experts import (BASELINE, MAXIMUM, ConceptExpert, forest, main,
                                 maximum_positive_score, run_experiment)


class ConceptTests(unittest.TestCase):
    def test_reader_concepts_preserve_old_defaults_and_decode_calcification(self):
        self.assertEqual(len(ATTRIBUTES), 5)
        self.assertEqual(len(CONCEPTS), 7)
        with TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            for dataset in ["lidc", "lndb"]:
                keys = (["patient_id", "series_uid", "nodule_id"] if dataset == "lidc"
                        else ["LNDbID", "FindingID"])
                rows = []
                for uid, calcification in [(1, 6), (1, 3), (2, 6), (3, 4.5)]:
                    row = {key: uid for key in keys}
                    row.update({name: 2 for name in CONCEPTS})
                    row["calcification"] = calcification
                    row["lobulation"] = 0 if uid == 2 else 2
                    rows.append(row)
                annotations = pd.DataFrame(rows)
                if dataset == "lndb":
                    annotations = annotations.rename(columns={name: name.capitalize() for name in CONCEPTS})
                source = root / f"{dataset}.csv"
                annotations.to_csv(source, index=False)
                paths = {dataset: {"annotations" if dataset == "lidc" else "findings": source}}
                table = pd.DataFrame([{key: uid for key in keys} for uid in [1, 2, 3]])
                merged = load_concepts(table, dataset, paths)
                self.assertEqual(merged.calcification.tolist(), [1, 0, 1])
                self.assertTrue(pd.isna(merged.lobulation.iloc[1]))

    def test_maximum_uses_positive_scores_and_rejects_missing_or_invalid_values(self):
        scores = pd.DataFrame({"shape": [0.1, 0.9], "spiculation": [0.8, 0.2]}, index=[7, 3])
        result = maximum_positive_score(scores)
        self.assertEqual(result.tolist(), [0.8, 0.9])
        self.assertEqual(result.index.tolist(), [7, 3])
        for invalid in [pd.DataFrame(), scores.assign(shape=np.nan), scores.assign(shape=1.1),
                        pd.DataFrame([[0.1, 0.2]], columns=["same", "same"])]:
            with self.assertRaises(ValueError):
                maximum_positive_score(invalid)

    def test_risk_head_uses_only_one_cross_fitted_concept_with_disjoint_patients(self):
        groups = np.repeat(np.arange(6), 2)
        values = np.column_stack([np.arange(12), np.arange(12) % 2])
        target = values[:, 1].astype(int)
        ratings = 1 + 4 * target
        events = []

        class Regressor:
            def fit(self, X, y):
                self.seen = set(X[:, 0])
                events.append(("fit", self.seen))
                return self

            def predict(self, X):
                events.append(("predict", set(X[:, 0])))
                return 1 + 4 * X[:, 1]

        with patch("src.concept_experts.concept_regressor", side_effect=lambda *args: Regressor()):
            model = ConceptExpert("spiculation", inner_splits=3).fit(values, target, ratings, groups)
        for index in range(0, 6, 2):
            fit_rows, held_out = events[index][1], events[index + 1][1]
            fit_groups = {groups[int(uid)] for uid in fit_rows}
            held_out_groups = {groups[int(uid)] for uid in held_out}
            self.assertTrue(fit_groups.isdisjoint(held_out_groups))
        self.assertEqual(model.classifier_.n_features_in_, 1)
        self.assertAlmostEqual(model.recovery_, 1)
        self.assertEqual(model.predict(values).tolist(), target.tolist())
        self.assertEqual(model.predict_proba(values).shape, (12, 2))

    def test_real_predictor_handles_missing_ratings_and_binary_calcification(self):
        generator = np.random.default_rng(42)
        values = generator.normal(size=(24, 3))
        target = np.tile([0, 1], 12)
        ratings = target.astype(float)
        ratings[0] = np.nan
        model = ConceptExpert("calcification", n_estimators=2, n_jobs=1, inner_splits=3)
        model.fit(values, target, ratings, np.repeat(np.arange(8), 3))
        probabilities = model.predict_proba(values)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1))
        self.assertTrue(((model.predict_concept(values) >= 0) & (model.predict_concept(values) <= 1)).all())
        self.assertEqual(model.recovery_metric_, "AUC")

    def test_invalid_training_data_fail(self):
        values = np.arange(12).reshape(6, 2)
        for name, target, ratings, groups in [
            ("malignancy", [0, 1] * 3, [2] * 6, range(6)),
            ("margin", [0] * 6, [2] * 6, range(6)),
            ("margin", [0, 1] * 3, [0] * 6, range(6)),
            ("margin", [0, 1] * 3, [np.nan] * 6, range(6)),
            ("margin", [0, 1] * 3, [2] * 6, [1] * 6),
            ("calcification", [0, 1] * 3, [3] * 6, range(6)),
        ]:
            with self.subTest(name=name, target=target, ratings=ratings), self.assertRaises(ValueError):
                ConceptExpert(name, n_estimators=2).fit(values, target, ratings, groups)

    def test_experiment_keeps_calibration_and_test_patients_out_of_training(self):
        uid = "original_firstorder_UID"
        volume = "original_shape_MeshVolume"
        table = pd.DataFrame([
            {"LNDbID": 10 * fold + j, "FindingID": 1, "fold": fold,
             "target_binary_mean": j % 2, uid: 10 * fold + j,
             volume: 10 * fold + j, "patient_id": np.nan}
            for fold in range(1, 6) for j in range(4)
        ])
        lidc = table.iloc[:4].copy()
        lidc[uid] += 1000
        lidc[volume] += 1000
        lidc["patient_id"] = ["p1", "p1", "p2", "p2"]
        fit_ids = []

        class Model:
            classes_ = np.array([0, 1])
            recovery_ = 0.5
            recovery_metric_ = "Spearman"

            def fit(self, X, y, *args):
                fit_ids.append(set(X.iloc[:, -1]))
                return self

            def predict_proba(self, X):
                scores = np.where(X.iloc[:, -1].to_numpy() % 2, 0.8, 0.2)
                return np.column_stack([1 - scores, scores])

            def predict_concept(self, X):
                return np.full(len(X), 3)

        def attributes(data, dataset, paths):
            return data.assign(**{name: 0 if name == "calcification" else 3 for name in CONCEPTS})

        with TemporaryDirectory(dir="/tmp/opencode") as directory:
            output = Path(directory)
            with patch("src.concept_experts.load_paths", return_value={"lndb": {"segmentation_targets": "targets.csv"}}), \
                    patch("src.concept_experts.load_data", side_effect=[(table, [uid, volume]), (lidc, [uid, volume])]), \
                    patch("src.concept_experts.load_concepts", side_effect=attributes), \
                    patch("src.concept_experts.ConceptExpert", side_effect=lambda *args: Model()), \
                    patch("src.concept_experts.forest", side_effect=lambda *args: Model()), \
                    patch("src.concept_experts.joblib.dump"), contextlib.redirect_stdout(io.StringIO()):
                run_experiment(output, n_jobs=1)
            for index, ids in enumerate(fit_ids):
                fold = index // 9 + 1
                calibration_fold = fold % 5 + 1
                excluded = set(table.loc[table.fold.isin([fold, calibration_fold]), volume])
                self.assertTrue(ids.isdisjoint(excluded))
                self.assertTrue(set(lidc[volume]).issubset(ids))
            predictions = pd.read_csv(output / "validation_predictions.csv")
            self.assertEqual(len(predictions), 20 * 10)
            self.assertFalse(predictions.duplicated(["model", "LNDbID", "FindingID"]).any())
            self.assertTrue(predictions.prediction.eq(predictions.target_binary_mean).all())
            maximum = predictions[predictions.model.eq(MAXIMUM)]
            expert_scores = predictions[~predictions.model.isin([MAXIMUM, BASELINE])].pivot(
                index=["LNDbID", "FindingID"], columns="model", values="score",
            )
            np.testing.assert_allclose(maximum.set_index(["LNDbID", "FindingID"]).score,
                                       maximum_positive_score(expert_scores))
            calibration = pd.read_csv(output / "calibration_predictions.csv")
            self.assertTrue(calibration.loc[calibration.target_binary_mean.eq(1), "prediction"].eq(1).all())
            self.assertTrue(pd.read_csv(output / "metrics.csv").completed_folds.eq(5).all())
            settings = json.loads((output / "settings.json").read_text())
            self.assertEqual(settings["maximum_members"], CONCEPTS + ["shape"])

    def test_saved_models_replay_scores_and_held_out_annotations_do_not_change_them(self):
        columns = ["original_firstorder_Mean", "original_shape_MeshVolume"]
        source = pd.DataFrame([
            {"LNDbID": 10 * fold + j, "FindingID": 1, "fold": fold,
             "target_binary_mean": j % 2, columns[0]: j % 2, columns[1]: 5 + j,
             **{name: j % 2 if name == "calcification" else 1 + 4 * (j % 2) for name in CONCEPTS}}
            for fold in range(1, 6) for j in range(4)
        ])
        lidc = source.iloc[:8].assign(patient_id=[f"p{i // 2}" for i in range(8)]).copy()
        split = next(calibration_splits(source))
        actual_forest = forest

        def small_forest(n_jobs):
            return actual_forest(n_jobs).set_params(balancedrandomforestclassifier__n_estimators=2)

        with TemporaryDirectory(dir="/tmp/opencode") as directory:
            results = []
            for run in range(2):
                table = source.copy()
                if run:
                    held_out = table.fold.isin([1, 2])
                    for name in CONCEPTS:
                        table.loc[held_out, name] = 1 if name == "calcification" else 3
                output = Path(directory) / str(run)
                output.mkdir()
                with patch("src.concept_experts.load_paths", return_value={"lndb": {"segmentation_targets": "targets.csv"}}), \
                        patch("src.concept_experts.load_data", side_effect=[(table, columns), (lidc, columns)]), \
                        patch("src.concept_experts.load_concepts", side_effect=lambda rows, *args: rows), \
                        patch("src.concept_experts.calibration_splits", return_value=[split]), \
                        patch("src.concept_experts.forest", side_effect=small_forest), \
                        contextlib.redirect_stdout(io.StringIO()):
                    run_experiment(output, n_jobs=1, n_estimators=2, inner_splits=2)
                predictions = pd.read_csv(output / "validation_predictions.csv")
                results.append(predictions)
                saved = joblib.load(output / "fold_1.joblib")
                for name, bundle in saved["models"].items():
                    actual = bundle["model"].predict_proba(table.loc[split[-1], bundle["features"]])[:, 1]
                    np.testing.assert_allclose(predictions.loc[predictions.model.eq(name), "score"], actual)
            pd.testing.assert_frame_equal(results[0], results[1])

    def test_command_refuses_to_overwrite_saved_work(self):
        with TemporaryDirectory(dir="/tmp/opencode") as directory:
            sentinel = Path(directory) / "existing.csv"
            sentinel.write_text("keep me\n")
            with patch("sys.argv", ["concept_experts", "--output", directory]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main()
            self.assertEqual(sentinel.read_text(), "keep me\n")


if __name__ == "__main__":
    unittest.main()
