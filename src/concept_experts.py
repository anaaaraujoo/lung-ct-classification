"""Binary segmentation experts taught with shared radiologist concepts."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .auxiliary import CONCEPTS, load_concepts
from .binary_evaluation import (BINARY_TARGET, binary_metrics, calibration_splits,
                                checked_scores, positive_score, zero_fn_threshold)
from .modeling import EXCLUDED_LNDB_IDS, LNDB_KEYS, ROOT, load_data, make_model, select_families
from .radiomics_extraction import load_paths

SHAPE = "shape"
BASELINE = "all_radiomics"
MAXIMUM = "maximum"


def maximum_positive_score(scores):
    """Take the strongest expert score for each nodule before threshold calibration."""
    if not isinstance(scores, pd.DataFrame) or scores.empty or scores.columns.duplicated().any():
        raise ValueError("Scores must be a non-empty table with unique expert columns.")
    values = scores.to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Every expert needs a finite score between 0 and 1 on every row.")
    return scores.max(axis=1).rename("score")


def concept_regressor(n_estimators, n_jobs):
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        RandomForestRegressor(n_estimators=n_estimators, max_features="sqrt",
                              min_samples_leaf=3, random_state=42, n_jobs=n_jobs),
    )


class ConceptExpert(ClassifierMixin, BaseEstimator):
    """Radiomics -> one predicted concept -> binary malignancy score.

    The risk head sees only that concept, not the original radiomics or a reader
    annotation. Its training inputs are cross-fitted by patient. At inference,
    only radiomics are needed. Calcification predicts reported presence.
    """

    def __init__(self, concept, n_estimators=200, n_jobs=2, inner_splits=4):
        self.concept = concept
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.inner_splits = inner_splits

    def fit(self, X, y, ratings, groups):
        if self.concept not in CONCEPTS:
            raise ValueError(f"Unknown concept: {self.concept}")
        values = np.asarray(X, dtype=float)
        target, _ = checked_scores(y, np.zeros(len(values)))
        ratings = np.asarray(ratings, dtype=float)
        groups = np.asarray(groups)
        if values.ndim != 2 or ratings.shape != (len(values),) or groups.shape != ratings.shape:
            raise ValueError("Radiomics, targets, ratings and patient groups must have matching rows.")
        if target.shape != ratings.shape or not np.isin(target, [0, 1]).all() or np.unique(target).size != 2:
            raise ValueError("Training needs both binary target classes.")
        known = np.isfinite(ratings)
        valid = np.isin(ratings[known], [0, 1]) if self.concept == "calcification" else (
            (ratings[known] >= 1) & (ratings[known] <= 5))
        if not valid.all() or not known.any() or pd.isna(groups).any():
            raise ValueError("Concept ratings or patient groups are invalid.")
        splits = min(self.inner_splits, len(np.unique(groups)))
        if splits < 2:
            raise ValueError("Concept fitting needs at least two patient groups and inner splits.")
        cross_fitted = np.full(len(values), np.nan)
        splitter = GroupKFold(n_splits=splits, shuffle=True, random_state=42)
        for fit_rows, held_out in splitter.split(values, target, groups):
            fit_rows = fit_rows[known[fit_rows]]
            if not len(fit_rows):
                raise ValueError("An inner fitting split has no known concept ratings.")
            model = concept_regressor(self.n_estimators, self.n_jobs)
            model.fit(values[fit_rows], ratings[fit_rows])
            cross_fitted[held_out] = model.predict(values[held_out])
        if self.concept == "calcification":
            self.recovery_metric_ = "AUC"
            self.recovery_ = (roc_auc_score(ratings[known], cross_fitted[known])
                              if np.unique(ratings[known]).size == 2 else np.nan)
        else:
            self.recovery_metric_ = "Spearman"
            self.recovery_ = (float(spearmanr(ratings[known], cross_fitted[known]).statistic)
                              if np.unique(ratings[known]).size > 1
                              and np.unique(cross_fitted[known]).size > 1 else np.nan)
        self.classifier_ = make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        ).fit(cross_fitted[:, None], target)
        self.regressor_ = concept_regressor(self.n_estimators, self.n_jobs)
        self.regressor_.fit(values[known], ratings[known])
        self.classes_ = self.classifier_.classes_
        self.n_features_in_ = values.shape[1]
        return self

    def predict_concept(self, X):
        return self.regressor_.predict(np.asarray(X, dtype=float))

    def predict_proba(self, X):
        return self.classifier_.predict_proba(self.predict_concept(X)[:, None])

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def forest(n_jobs):
    return make_model(balanced_forest=True, binary=True).set_params(
        balancedrandomforestclassifier__n_jobs=n_jobs,
    )


def prediction_rows(rows, score, threshold, name, calibration_fold):
    result = rows[LNDB_KEYS + ["fold", BINARY_TARGET]].copy()
    result["diameter_mm"] = 2 * (3 * rows.original_shape_MeshVolume / (4 * np.pi)) ** (1 / 3)
    result["model"] = name
    result["calibration_fold"] = calibration_fold
    result["threshold"] = threshold
    result["score"] = np.asarray(score)
    result["prediction"] = (np.asarray(score) >= threshold).astype(int)
    return result


def run_experiment(output, n_jobs=2, n_estimators=200, inner_splits=4):
    """Fit on three LNDb folds + LIDC, calibrate on one, test on the last."""
    output = Path(output)
    paths = load_paths()
    lndb, features = load_data(
        ROOT / "data/lndb_ge3_segmentation_3d_c50_hu.csv", "lndb", binary=True,
        targets_path=paths["lndb"]["segmentation_targets"], paths=paths, exclude_duplicate_lndb=True,
    )
    lidc, other_features = load_data(ROOT / "data/lidc_segmentation_3d_c50_hu.csv", "lidc", binary=True, paths=paths)
    if set(features) != set(other_features):
        raise ValueError("LIDC and LNDb must share segmentation features.")
    lndb = load_concepts(lndb, "lndb", paths)
    lidc = load_concepts(lidc, "lidc", paths)
    lndb["patient_group"] = "lndb:" + lndb.LNDbID.astype(str)
    lidc["patient_group"] = "lidc:" + lidc.patient_id.astype(str)
    shape_features = select_families(features, ["shape"])
    if not shape_features or lndb.original_shape_MeshVolume.le(0).any():
        raise ValueError("Segmentation needs shape features and positive nodule volumes.")
    settings = {"concepts": CONCEPTS, "n_jobs": n_jobs, "n_estimators": n_estimators,
                "inner_splits": inner_splits, "risk_head": "balanced logistic regression on one predicted concept",
                "calcification": "any reader reports a code below 6, codes 1–5 are not an ordinal scale",
                "maximum_members": CONCEPTS + [SHAPE], "small_diameter_mm": 6,
                "excluded_lndb_ids": sorted(EXCLUDED_LNDB_IDS)}
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    pd.Series(sorted(EXCLUDED_LNDB_IDS), name="LNDbID").to_csv(output / "excluded_ct_copies.csv", index=False)
    predictions, calibration_predictions, recovery = [], [], []
    for fold, calibration_fold, train_rows, calibration_rows, test_rows in calibration_splits(lndb):
        train = pd.concat([lidc, lndb.loc[train_rows]], ignore_index=True)
        calibration, test = lndb.loc[calibration_rows], lndb.loc[test_rows]
        calibration_scores = pd.DataFrame(index=calibration.index)
        test_scores = pd.DataFrame(index=test.index)
        fitted = {}
        for name in CONCEPTS + [SHAPE, BASELINE]:
            print(f"Fold {fold}: fitting {name}", flush=True)
            columns = shape_features if name == SHAPE else features
            if name in CONCEPTS:
                model = ConceptExpert(name, n_estimators, n_jobs, inner_splits)
                model.fit(train[columns], train[BINARY_TARGET], train[name], train.patient_group)
                recovery.append({"fold": fold, "concept": name, "metric": model.recovery_metric_,
                                 "value": model.recovery_})
            else:
                model = forest(n_jobs).fit(train[columns], train[BINARY_TARGET])
            fitted[name] = {"model": model, "features": columns}
            calibration_scores[name] = positive_score(model, calibration[columns])
            test_scores[name] = positive_score(model, test[columns])
        members = CONCEPTS + [SHAPE]
        calibration_scores[MAXIMUM] = maximum_positive_score(calibration_scores[members])
        test_scores[MAXIMUM] = maximum_positive_score(test_scores[members])
        thresholds = {}
        for name in test_scores:
            threshold = zero_fn_threshold(calibration[BINARY_TARGET], calibration_scores[name])
            thresholds[name] = threshold
            result = prediction_rows(test, test_scores[name], threshold, name, calibration_fold)
            if name in CONCEPTS:
                result["concept_prediction"] = fitted[name]["model"].predict_concept(test[features])
            if name == MAXIMUM:
                result["winning_expert"] = test_scores[members].idxmax(axis=1).to_numpy()
            predictions.append(result)
            calibration_predictions.append(prediction_rows(
                calibration, calibration_scores[name], threshold, name, calibration_fold,
            ).assign(test_fold=fold))
        models = {"models": fitted, "thresholds": thresholds, "maximum_members": members,
                  "test_fold": int(fold), "calibration_fold": int(calibration_fold)}
        joblib.dump(models, output / f"fold_{fold}.joblib", compress=3)
        pd.concat(predictions, ignore_index=True).to_csv(output / "validation_predictions.csv", index=False)
        pd.concat(calibration_predictions, ignore_index=True).to_csv(output / "calibration_predictions.csv", index=False)
        pd.DataFrame(recovery).to_csv(output / "concept_recovery.csv", index=False)
        scores = pd.concat(predictions, ignore_index=True)
        overall = [{"model": name, **binary_metrics(rows[BINARY_TARGET], rows.prediction),
                    "AUC": roc_auc_score(rows[BINARY_TARGET], rows.score)}
                   for name, rows in scores.groupby("model", sort=False)]
        pd.DataFrame(overall).assign(completed_folds=scores.fold.nunique()).to_csv(output / "metrics.csv", index=False)
        maximum_result = next(row for row in overall if row["model"] == MAXIMUM)
        print(f"Fold {fold} finished. Maximum so far: FN={maximum_result['FN']}, "
              f"FP={maximum_result['FP']}, accuracy={maximum_result['accuracy']:.3f}", flush=True)
    print(pd.DataFrame(overall).round(3).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models/segmentation_concept_experts")
    parser.add_argument("--n-jobs", type=int, default=2, help="CPU workers. Default: 2.")
    parser.add_argument("--n-estimators", type=int, default=200, help="Trees per concept predictor.")
    parser.add_argument("--inner-splits", type=int, default=4, help="Patient splits for learning each risk head.")
    args = parser.parse_args()
    if args.n_jobs < 1 or args.n_estimators < 1 or args.inner_splits < 2:
        parser.error("Use positive CPU/tree counts and at least two inner splits.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=args.n_jobs):
        run_experiment(args.output, args.n_jobs, args.n_estimators, args.inner_splits)


if __name__ == "__main__":
    # Keep saved estimator classes importable when this file runs via python -m.
    from .concept_experts import main

    main()
