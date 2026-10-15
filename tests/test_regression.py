"""Continuous reader targets, fold-local preprocessing and rounded-class evaluation."""
import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src.make_predictions import main as make_submission
from src.modeling import EXCLUDED_LNDB_IDS, TARGET, metrics
from src.regression import REGRESSION_TARGET, load_regression_data, main, make_model, regression_metrics, to_classes


class RegressionTests(unittest.TestCase):
    def test_half_up_rounding_and_clipping_match_targets(self):
        predictions = [-2., 1., 1.49, 1.5, 2.5, 3., 3.5, 4.5, 5., 7.]
        np.testing.assert_array_equal(to_classes(predictions), [1, 1, 1, 2, 3, 3, 4, 5, 5, 5])

    def test_metrics_use_continuous_predictions_without_clipping(self):
        score = regression_metrics([1., 2.5, 5.], [0., 2., 6.])
        self.assertAlmostEqual(score["mae"], 2.5 / 3)
        self.assertAlmostEqual(score["rmse"], np.sqrt(2.25 / 3))

    def test_pipeline_uses_training_medians_scaling_and_ridge(self):
        model = make_model()
        X = pd.DataFrame({"first": [1., 3., np.nan], "empty": [np.nan] * 3})
        model.fit(X, [1., 2.5, 4.])
        np.testing.assert_array_equal(model.named_steps["simpleimputer"].statistics_, [2., 0.])
        np.testing.assert_array_equal(model.named_steps["standardscaler"].mean_, [2., 0.])
        self.assertEqual(model.named_steps["ridge"].alpha, 1.)
        model.predict(pd.DataFrame({"first": [1000., np.nan], "empty": [np.nan] * 2}))
        np.testing.assert_array_equal(model.named_steps["simpleimputer"].statistics_, [2., 0.])
        np.testing.assert_array_equal(model.named_steps["standardscaler"].mean_, [2., 0.])

    def test_loading_uses_saved_float_targets_and_excludes_metadata(self):
        with TemporaryDirectory(dir="/tmp/opencode", prefix="regression-targets-") as temporary:
            root = Path(temporary)
            targets = pd.DataFrame({"LNDbID": [1, 2, 3], "NoduleID": [1, 1, 1],
                                    REGRESSION_TARGET: [2.5, 3., 3.5], TARGET: [3, 3, 4],
                                    "target_binary_mean": [0, np.nan, 1], "ratings": ["[2,3]", "[3]", "[3,4]"]})
            targets.to_csv(root / "targets.csv", index=False)
            pd.DataFrame({"LNDbID": [1, 2, 3], "fold": [1, 3, 5]}).to_csv(root / "folds.csv", index=False)
            table = pd.DataFrame({"LNDbID": [1, 2, 3], "FindingID": [1, 1, 1], "x": [11., 22., 33.],
                                  "RadID": [7, 8, 9], "diagnostics_Image_Mean": [99.] * 3,
                                  "original_firstorder_Mean": [1., np.inf, -np.inf],
                                  REGRESSION_TARGET: [5., 5., 5.], "fold": [99, 99, 99]})
            table.to_csv(root / "features.csv", index=False)
            paths = {"lndb": {"targets": root / "targets.csv", "folds": root / "folds.csv"}}
            data, features = load_regression_data(root / "features.csv", "lndb", paths=paths)
            self.assertEqual(features, ["original_firstorder_Mean"])
            self.assertEqual(data[REGRESSION_TARGET].tolist(), [2.5, 3., 3.5])
            self.assertEqual(data[TARGET].tolist(), [3, 3, 4])
            self.assertEqual(data["fold"].tolist(), [1, 3, 5])
            self.assertTrue(data[features].iloc[1:].isna().all().all())
            selected = root / "selected.csv"
            targets.iloc[[0, 2]].to_csv(selected, index=False)
            subset, _ = load_regression_data(root / "features.csv", "lndb", selected, paths)
            self.assertEqual(subset[REGRESSION_TARGET].tolist(), [2.5, 3.5])
            for value in (np.nan, np.inf, 0., 6.):
                with self.subTest(value=value):
                    targets.assign(target_regression_mean=value).to_csv(root / "targets.csv", index=False)
                    with self.assertRaisesRegex(ValueError, "finite reader means"):
                        load_regression_data(root / "features.csv", "lndb", paths=paths)
            targets.drop(columns=REGRESSION_TARGET).to_csv(root / "targets.csv", index=False)
            with self.assertRaisesRegex(ValueError, "finite reader means"):
                load_regression_data(root / "features.csv", "lndb", paths=paths)

    def test_cli_saves_both_predictions_metrics_and_continuous_model(self):
        with TemporaryDirectory(dir="/tmp/opencode", prefix="regression-cli-") as temporary:
            root = Path(temporary)
            means = np.tile([1., 1.75, 2.5, 3., 3.5, 4.75], 5)
            folds = np.repeat(np.arange(1, 6), 6)
            ids = np.array([*range(1, len(means)), 181])
            targets = pd.DataFrame({"LNDbID": ids, "NoduleID": 1, REGRESSION_TARGET: means, TARGET: to_classes(means)})
            targets.to_csv(root / "targets.csv", index=False)
            pd.DataFrame({"LNDbID": ids, "fold": folds}).to_csv(root / "folds.csv", index=False)
            features = ["original_firstorder_Mean", "wavelet-HHH_glcm_Contrast"]
            table = pd.DataFrame({"LNDbID": ids, "FindingID": 1, features[0]: means + folds * 10., features[1]: means ** 2})
            table.loc[0, features[0]] = np.nan
            table.to_csv(root / "features.csv", index=False)
            lidc = pd.DataFrame({"patient_id": ["p1", "p2"], "series_uid": ["s1", "s2"], "nodule_id": [0, 0],
                                 REGRESSION_TARGET: [1.25, 4.5], TARGET: [1, 5]})
            lidc.to_csv(root / "lidc_targets.csv", index=False)
            pd.DataFrame({"patient_id": ["p1", "p2"], "fold": [1, 5]}).to_csv(root / "lidc_folds.csv", index=False)
            lidc_features = lidc[["patient_id", "series_uid", "nodule_id"]].copy()
            lidc_features[features[0]], lidc_features[features[1]] = [2., 6.], [1., 9.]
            lidc_features.to_csv(root / "lidc_features.csv", index=False)
            paths = {"lndb": {"targets": root / "targets.csv", "folds": root / "folds.csv"},
                     "lidc": {"targets": root / "lidc_targets.csv", "folds": root / "lidc_folds.csv"},
                     "training": {"lndb_features": root / "features.csv", "lidc_features": root / "lidc_features.csv"},
                     "outputs": {"models": root}}
            for lndb_only, original_only, exclude in ((False, False, False), (True, True, False),
                                                       (True, True, True)):
                output = root / f"results_{lndb_only}_{exclude}"
                args = ["regression", "--output", str(output)]
                if lndb_only:
                    args += ["--lndb-only"]
                if original_only:
                    args += ["--original-only"]
                if exclude:
                    args += ["--exclude-duplicate-lndb"]
                run_paths = paths if not lndb_only else {
                    "lndb": paths["lndb"], "outputs": paths["outputs"],
                    "training": {"lndb_features": root / "features.csv"},
                }
                fitted = []

                def record_model():
                    model = make_model()
                    fitted.append(model)
                    return model

                with self.subTest(lndb_only=lndb_only, exclude=exclude), patch("sys.argv", args), \
                        patch("src.regression.load_paths", return_value=run_paths), \
                        patch("src.regression.make_model", side_effect=record_model), \
                        contextlib.redirect_stdout(io.StringIO()):
                    main()
                self.assertEqual(len(fitted), 6)
                selected_features = features[:1] if original_only else features
                for fold, model in zip(range(1, 6), fitted):
                    train = table.loc[(folds != fold) & ((ids != 181) if exclude else True), selected_features]
                    if not lndb_only:
                        train = pd.concat([lidc_features[selected_features], train])
                    expected = train.median().to_numpy()
                    np.testing.assert_allclose(model.named_steps["simpleimputer"].statistics_, expected)
                    np.testing.assert_allclose(model.named_steps["standardscaler"].mean_, train.fillna(train.median()).mean())
                predictions = pd.read_csv(output / "validation_predictions.csv")
                self.assertEqual(len(predictions), len(table) - int(exclude))
                np.testing.assert_array_equal(predictions[REGRESSION_TARGET], means[:-1] if exclude else means)
                np.testing.assert_array_equal(predictions["prediction"], to_classes(predictions["prediction_continuous"]))
                self.assertTrue((predictions["prediction_continuous"] % 1 != 0).any())
                regression = pd.read_csv(output / "regression_metrics.csv")
                classification = pd.read_csv(output / "metrics.csv")
                self.assertEqual(regression["fold"].tolist(), ["1", "2", "3", "4", "5", "overall_oof"])
                self.assertEqual(classification["fold"].tolist(), regression["fold"].tolist())
                expected = regression_metrics(predictions[REGRESSION_TARGET], predictions["prediction_continuous"])
                for name, value in expected.items():
                    self.assertAlmostEqual(regression.iloc[-1][name], value)
                for name, value in metrics(predictions[TARGET], predictions["prediction"]).items():
                    self.assertAlmostEqual(classification.iloc[-1][name], value)
                self.assertEqual(pd.read_csv(output / "confusion_matrix.csv", index_col=0).to_numpy().sum(), len(predictions))
                self.assertEqual(pd.read_csv(output / "recall_per_class.csv")["class"].tolist(), [1, 2, 3, 4, 5])
                saved = joblib.load(output / "model.joblib")
                self.assertEqual(saved["target"], REGRESSION_TARGET)
                self.assertEqual(saved["excluded_lndb_ids"], sorted(EXCLUDED_LNDB_IDS) if exclude else [])
                self.assertEqual(saved["features"], selected_features)
                self.assertEqual(saved["class_conversion"], "half_up_clip_1_5")
                self.assertEqual(saved["lidc_csv"], None if lndb_only else str(root / "lidc_features.csv"))
                self.assertTrue(np.isfinite(saved["model"].predict(table[selected_features])).all())

    def test_competition_conversion_rounds_regressor_and_preserves_classifiers(self):
        with TemporaryDirectory(dir="/tmp/opencode", prefix="regression-submission-") as temporary:
            root = Path(temporary)
            table = pd.DataFrame({"LNDbID": [1, 2, 3], "FindingID": [1, 1, 1], "original_firstorder_Mean": [1., 2., 3.]})
            table.to_csv(root / "features.csv", index=False)
            submission = pd.DataFrame({"LNDbID": [1, 2, 3, 4], "FindingID": [1] * 4, "prediction": [2] * 4})
            submission.to_csv(root / "submission.csv", index=False)
            before = (root / "submission.csv").read_bytes()
            for target, raw, expected in ((REGRESSION_TARGET, [2.5, 4.5, 7.], [3, 5, 5, 5]),
                                          (TARGET, [2, 4, 4], [2, 4, 4, 4])):
                with self.subTest(target=target), patch("sys.argv", ["make_predictions", "--model", "unused.joblib",
                        "--features", str(root / "features.csv"), "--submission", str(root / "submission.csv"),
                        "--output", str(root / "predictions.csv")]), patch("src.make_predictions.joblib.load") as loaded, \
                        contextlib.redirect_stdout(io.StringIO()):
                    model = unittest.mock.Mock()
                    model.predict.return_value = np.array(raw)
                    loaded.return_value = {"model": model, "target": target, "features": ["original_firstorder_Mean"]}
                    make_submission()
                    self.assertEqual(pd.read_csv(root / "predictions.csv")["prediction"].tolist(), expected)
                    self.assertEqual((root / "submission.csv").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
