"""Checks for the saved-fold RF tuning, without using competition data."""
import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import ParameterGrid

from src.modeling import EXCLUDED_LNDB_IDS, MIN_SAMPLES_GRID, QUICK_FOREST_GRID, QUICK_XGBOOST_GRID, TARGET, fit_model, lndb_cv, load_data, main as train_classifier, make_model, metrics, prefer_normalized_pairs, tune_balanced_model, tune_rf
from src.make_predictions import main as make_submission
from src.regression import load_regression_data


class RFTuningTests(unittest.TestCase):
    def test_normalized_pairs_keep_other_features(self):
        features = ["original_glrlm_RunLengthNonUniformity",
                    "original_glrlm_RunLengthNonUniformityNormalized",
                    "wavelet-HL_gldm_DependenceNonUniformity",
                    "wavelet-HL_gldm_DependenceNonUniformityNormalized",
                    "original_firstorder_Mean"]
        self.assertEqual(prefer_normalized_pairs(features), [features[1], features[3], features[4]])
        with self.assertRaisesRegex(ValueError, "No raw/Normalized"):
            prefer_normalized_pairs([features[4]])

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

    def test_balanced_forest_tuning_uses_inner_folds_and_refits(self):
        train = pd.DataFrame({"fold": [-1] * 5 + [1] * 5 + [2] * 5 + [4] * 5 + [5] * 5,
                              TARGET: [1, 2, 3, 4, 5] * 5,
                              "original_firstorder_Mean": list(range(5)) * 5})
        model = make_model(balanced_forest=True).set_params(balancedrandomforestclassifier__n_estimators=3)
        grid = {"balancedrandomforestclassifier__min_samples_leaf": [1, 3]}
        with TemporaryDirectory(prefix="brf-grid-", dir="/tmp/opencode") as directory:
            with contextlib.redirect_stdout(io.StringIO()):
                fitted, score, params = tune_balanced_model(model, train, ["original_firstorder_Mean"],
                                                            5, Path(directory), "fold_3", None, 0, True, grid)
            self.assertEqual(len(pd.read_csv(Path(directory) / "grid_fold_3.csv")), 2)
            self.assertIn(params["balancedrandomforestclassifier__min_samples_leaf"], [1, 3])
            self.assertTrue(0 <= score <= 1)
            self.assertEqual(fitted.named_steps["balancedrandomforestclassifier"].n_estimators, 3)
            np.testing.assert_array_equal(fitted.named_steps["simpleimputer"].statistics_, [2.])

    def test_xgboost_tuning_weights_only_inner_training_and_offsets_labels(self):
        train = pd.DataFrame({"fold": [-1] * 5 + [1] * 5 + [2] * 5 + [4] * 5 + [5] * 5,
                              TARGET: [1, 2, 3, 4, 5] * 5,
                              "original_firstorder_Mean": list(range(5)) * 5})
        model = make_model("balanced", model_kind="xgboost").set_params(xgbclassifier__n_estimators=2)
        structure = {"xgbclassifier__max_depth": [2, 3]}
        rounds = [{"xgbclassifier__learning_rate": [0.05], "xgbclassifier__n_estimators": [2]},
                  {"xgbclassifier__learning_rate": [0.1], "xgbclassifier__n_estimators": [3]}]
        from sklearn.utils.class_weight import compute_sample_weight
        with TemporaryDirectory(prefix="xgb-grid-", dir="/tmp/opencode") as directory:
            with patch("src.modeling.compute_sample_weight", wraps=compute_sample_weight) as weights, \
                    contextlib.redirect_stdout(io.StringIO()):
                fitted, score, params = tune_balanced_model(model, train, ["original_firstorder_Mean"],
                                                            5, Path(directory), "fold_3", "balanced", 1,
                                                            structure_grid=structure, rounds_grid=rounds)
            self.assertEqual([len(call.args[1]) for call in weights.call_args_list], [20] * 16 + [25])
            self.assertEqual(set(params), {"xgbclassifier__max_depth", "xgbclassifier__learning_rate",
                                           "xgbclassifier__n_estimators"})
            self.assertEqual(fitted.named_steps["xgbclassifier"].n_classes_, 5)
            self.assertTrue(set(fitted.predict(train[["original_firstorder_Mean"]]) + 1) <= {1, 2, 3, 4, 5})
            self.assertTrue(0 <= score <= 1)
            grid = pd.read_csv(Path(directory) / "grid_fold_3.csv")
            self.assertEqual(grid.stage.tolist(), ["structure", "structure", "rounds", "rounds"])
            self.assertEqual(len([key for key in grid if key.startswith("split")]), 4)

    def test_quick_tuning_uses_a_four_candidate_grid_and_one_inner_fold(self):
        forest_grid = list(ParameterGrid(QUICK_FOREST_GRID))
        xgboost_grid = list(ParameterGrid(QUICK_XGBOOST_GRID))
        self.assertEqual(len(forest_grid), len(xgboost_grid))
        self.assertEqual(len(forest_grid), 4)
        self.assertEqual({(p["balancedrandomforestclassifier__max_features"],
                           p["balancedrandomforestclassifier__min_samples_leaf"]) for p in forest_grid},
                         {("sqrt", 1), ("sqrt", 3), (0.1, 1), (0.1, 3)})
        self.assertEqual({(p["xgbclassifier__max_depth"], p["xgbclassifier__min_child_weight"])
                          for p in xgboost_grid}, {(2, 1), (2, 5), (3, 1), (3, 5)})
        train = pd.DataFrame({"fold": [-1] * 5 + [1] * 5 + [2] * 5 + [4] * 5 + [5] * 5,
                              TARGET: [1, 2, 3, 4, 5] * 4 + [1, 2, 3, 5, 5],
                              "original_firstorder_Mean": list(range(5)) * 5})
        model = make_model("balanced", model_kind="xgboost").set_params(xgbclassifier__n_estimators=2)
        from sklearn.utils.class_weight import compute_sample_weight
        with TemporaryDirectory(prefix="quick-tuning-", dir="/tmp/opencode") as directory:
            with patch("src.modeling.compute_sample_weight", wraps=compute_sample_weight) as weights, \
                    contextlib.redirect_stdout(io.StringIO()) as stdout:
                _, _, params = tune_balanced_model(model, train, ["original_firstorder_Mean"], 5,
                                                    Path(directory), "fold_3", "balanced", 1, quick=True)
            self.assertIn("inner LNDb fold 5", stdout.getvalue())
            self.assertEqual(len(weights.call_args_list), 5)
            self.assertEqual([len(call.args[1]) for call in weights.call_args_list], [20] * 4 + [25])
            self.assertIn(params["xgbclassifier__max_depth"], [2, 3])
            self.assertIn(params["xgbclassifier__min_child_weight"], [1, 5])
            self.assertEqual(len(pd.read_csv(Path(directory) / "grid_fold_3.csv")), 4)

    def test_cli_rejects_misnamed_baselines_and_saves_real_tuning(self):
        with TemporaryDirectory(prefix="tuning-cli-", dir="/tmp/opencode") as directory:
            root = Path(directory)
            lidc = pd.DataFrame({"patient_id": [f"p{label}" for label in range(1, 6)],
                                 "series_uid": [f"s{label}" for label in range(1, 6)],
                                 "nodule_id": [1] * 5, TARGET: list(range(1, 6)),
                                 "original_firstorder_Mean": list(range(1, 6))})
            lndb = pd.DataFrame([{"LNDbID": 10 * fold + label, "FindingID": 1, "fold": fold,
                                  TARGET: label, "original_firstorder_Mean": float(label)}
                                 for fold in range(1, 6) for label in range(1, 6)])
            lidc.drop(columns=TARGET).to_csv(root / "lidc.csv", index=False)
            lndb.drop(columns=[TARGET, "fold"]).to_csv(root / "lndb.csv", index=False)
            lidc.drop(columns="original_firstorder_Mean").to_csv(root / "lidc_targets.csv", index=False)
            lndb.drop(columns=["original_firstorder_Mean", "fold"]).rename(
                columns={"FindingID": "NoduleID"}).to_csv(root / "lndb_targets.csv", index=False)
            lidc[["patient_id"]].assign(fold=1).to_csv(root / "lidc_folds.csv", index=False)
            lndb[["LNDbID", "fold"]].to_csv(root / "lndb_folds.csv", index=False)
            paths = {"lidc": {"targets": root / "lidc_targets.csv", "folds": root / "lidc_folds.csv"},
                     "lndb": {"targets": root / "lndb_targets.csv", "folds": root / "lndb_folds.csv"},
                     "training": {"lidc_features": root / "lidc.csv", "lndb_features": root / "lndb.csv"},
                     "outputs": {"models": root}}

            def tiny_model(*args, **kwargs):
                model = make_model(*args, **kwargs)
                classifier = "balancedrandomforestclassifier" if "balancedrandomforestclassifier" in model.named_steps else "xgbclassifier"
                return model.set_params(**{f"{classifier}__n_estimators": 2})

            for name, flags in (("brf", ["--model", "rf", "--balanced-forest"]),
                                ("xgboost", ["--model", "xgboost", "--class-weight", "balanced"])):
                with self.subTest(name=name):
                    args = ["modeling.py", "--lidc", str(root / "lidc.csv"), "--lndb", str(root / "lndb.csv"),
                            *flags, "--output", str(root / f"{name}_grid4")]
                    with patch("sys.argv", args), patch("src.modeling.load_paths", return_value=paths), \
                            patch("src.modeling.make_model", side_effect=tiny_model), \
                            contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit):
                            train_classifier()
                    self.assertFalse((root / f"{name}_grid4").exists())

                    output = root / f"{name}_grid4_tuned"
                    args[-1] = str(output)
                    args.insert(-2, "--quick-tune")
                    with patch("sys.argv", args), patch("src.modeling.load_paths", return_value=paths), \
                            patch("src.modeling.make_model", side_effect=tiny_model), \
                            contextlib.redirect_stdout(io.StringIO()):
                        train_classifier()
                    self.assertEqual(len(pd.read_csv(output / "tuning.csv")), 6)
                    self.assertEqual(len(pd.read_csv(output / "grid_fold_1.csv")), 4)
                    self.assertEqual(len(pd.read_csv(output / "validation_predictions.csv")), 25)
                    saved = joblib.load(output / "model.joblib")
                    self.assertEqual(saved["tuning_scoring"], "balanced_accuracy")
                    self.assertEqual(len(saved["best_params"]), 2)


class PopulationTests(unittest.TestCase):
    def test_confirmed_ct_copies_are_excluded_from_every_target_and_roi(self):
        self.assertEqual(EXCLUDED_LNDB_IDS, {181, 182, 183, 184, 185, 188, 189, 194, 195, 198, 226})
        retained = [161, 162, 163, 164, 165, 168, 169, 174, 175, 179, 225]
        excluded = sorted(EXCLUDED_LNDB_IDS)
        with TemporaryDirectory(prefix="lndb-dedup-", dir="/tmp/opencode") as directory:
            root = Path(directory)
            targets = pd.DataFrame({"LNDbID": retained + excluded, "NoduleID": [1] * 22,
                                    TARGET: [3] * 22, "target_regression_mean": [2.5] * 22,
                                    "target_binary_mean": [0] * 22, "ratings": ["[2,3]"] * 22})
            targets.to_csv(root / "targets.csv", index=False)
            folds = pd.DataFrame({"LNDbID": retained + excluded, "fold": [1, 3, 5, 2, 4, 1, 3, 5, 2, 4, 1] * 2})
            folds.to_csv(root / "folds.csv", index=False)
            table = targets.rename(columns={"NoduleID": "FindingID"}).copy()
            columns = ["original_firstorder_Mean", "original_shape_MeshVolume", "wavelet-HHH_glcm_Contrast"]
            for column in columns:
                table[column] = np.arange(22, dtype=float)
            table.to_csv(root / "features.csv", index=False)
            originals = {path: path.read_bytes() for path in root.glob("*.csv")}
            paths = {"lndb": {"targets": root / "targets.csv", "folds": root / "folds.csv"}}
            for binary in (False, True):
                for explicit in (None, root / "targets.csv", root / "features.csv"):
                    with self.subTest(binary=binary, explicit=explicit):
                        full, _ = load_data(root / "features.csv", "lndb", explicit, paths, binary=binary)
                        self.assertEqual(full["LNDbID"].tolist(), retained + excluded)
                        data, features = load_data(root / "features.csv", "lndb", explicit, paths, binary=binary,
                                                   exclude_duplicate_lndb=True)
                        self.assertEqual(data["LNDbID"].tolist(), retained)
                        self.assertEqual(data["fold"].tolist(), folds["fold"].tolist()[:11])
                        self.assertEqual(data["ratings"].tolist(), ["[2,3]"] * 11)
                        self.assertEqual(features, columns)
                        self.assertFalse(data["LNDbID"].isin(EXCLUDED_LNDB_IDS).any())
            regression, _ = load_regression_data(root / "features.csv", "lndb", paths=paths)
            self.assertEqual(regression["LNDbID"].tolist(), retained + excluded)
            regression, _ = load_regression_data(root / "features.csv", "lndb", paths=paths,
                                                 exclude_duplicate_lndb=True)
            self.assertEqual(regression["LNDbID"].tolist(), retained)
            self.assertEqual(regression["target_regression_mean"].tolist(), [2.5] * 11)
            for path, content in originals.items():
                self.assertEqual(path.read_bytes(), content)

    def test_missing_excluded_features_are_allowed_but_retained_features_are_required(self):
        with TemporaryDirectory(prefix="lndb-dedup-missing-", dir="/tmp/opencode") as directory:
            root = Path(directory)
            targets = pd.DataFrame({"LNDbID": [161, 181], "NoduleID": [1, 1], TARGET: [2, 4]})
            targets.to_csv(root / "targets.csv", index=False)
            pd.DataFrame({"LNDbID": [161, 181], "fold": [1, 3]}).to_csv(root / "folds.csv", index=False)
            table = pd.DataFrame({"LNDbID": [161], "FindingID": [1], "original_firstorder_Mean": [1.]})
            paths = {"lndb": {"targets": root / "targets.csv", "folds": root / "folds.csv"}}
            table.to_csv(root / "features.csv", index=False)
            with self.assertRaisesRegex(ValueError, "all 2 training nodules"):
                load_data(root / "features.csv", "lndb", paths=paths)
            data, _ = load_data(root / "features.csv", "lndb", paths=paths, exclude_duplicate_lndb=True)
            self.assertEqual(data["LNDbID"].tolist(), [161])
            # Rows for discarded CTs cannot satisfy completeness for retained CTs.
            table.assign(LNDbID=181).to_csv(root / "features.csv", index=False)
            with self.assertRaisesRegex(ValueError, "all 1 training nodules"):
                load_data(root / "features.csv", "lndb", paths=paths, exclude_duplicate_lndb=True)
            targets.iloc[[1]].to_csv(root / "selected.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Targets must contain unique, non-missing nodule IDs"):
                load_data(root / "features.csv", "lndb", root / "selected.csv", paths,
                          exclude_duplicate_lndb=True)

    def test_lidc_nodule_numbers_are_not_subject_to_lndb_exclusions(self):
        with TemporaryDirectory(prefix="lidc-not-dedup-", dir="/tmp/opencode") as directory:
            root = Path(directory)
            targets = pd.DataFrame({"patient_id": ["p181"], "series_uid": ["s181"], "nodule_id": [181], TARGET: [4]})
            targets.to_csv(root / "targets.csv", index=False)
            pd.DataFrame({"patient_id": ["p181"], "fold": [3]}).to_csv(root / "folds.csv", index=False)
            targets.assign(original_firstorder_Mean=1.).to_csv(root / "features.csv", index=False)
            paths = {"lidc": {"targets": root / "targets.csv", "folds": root / "folds.csv"}}
            data, _ = load_data(root / "features.csv", "lidc", paths=paths)
            self.assertEqual(data["nodule_id"].tolist(), [181])
            self.assertEqual(data["fold"].tolist(), [3])
            excluded, _ = load_data(root / "features.csv", "lidc", paths=paths, exclude_duplicate_lndb=True)
            self.assertEqual(excluded["nodule_id"].tolist(), [181])

    def test_3d_shape_log_and_lbp_are_features_not_metadata(self):
        with TemporaryDirectory(prefix="volume-population-", dir="/tmp/opencode") as directory:
            root = Path(directory)
            pd.DataFrame({"LNDbID": [1], "NoduleID": [1], TARGET: [4]}).to_csv(root / "targets.csv", index=False)
            pd.DataFrame({"LNDbID": [1], "fold": [3]}).to_csv(root / "folds.csv", index=False)
            columns = ["original_shape_MeshVolume", "log-sigma-1-0-mm-3D_glcm_Contrast", "lbp-3D-m1_firstorder_Mean"]
            table = pd.DataFrame({"LNDbID": [1], "FindingID": [1], "mask_RadID": [99], "gt_FindingID": [7],
                                  "diagnostics_Image_Mean": [500], **{name: [2.] for name in columns}})
            table.to_csv(root / "features.csv", index=False)
            paths = {"lndb": {"targets": root / "targets.csv", "folds": root / "folds.csv"}}
            data, features = load_data(root / "features.csv", "lndb", paths=paths)
            self.assertEqual(features, columns)
            self.assertEqual(data["fold"].tolist(), [3])
            self.assertEqual(data[TARGET].tolist(), [4])

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
