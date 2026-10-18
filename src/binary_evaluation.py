"""Evaluate segmentation-based alerts and independently selected thresholds."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from .modeling import (EXCLUDED_LNDB_IDS, LNDB_KEYS, TARGET, fit_model, load_data,
                       make_model, select_families)
from .radiomics_extraction import load_paths

BINARY_TARGET = "target_binary_mean"


def checked_scores(target, score):
    target, score = np.asarray(target), np.asarray(score, dtype=float)
    if target.ndim != 1 or score.shape != target.shape or not len(target):
        raise ValueError("Target and score must be non-empty vectors of equal length.")
    if not np.isin(target, [0, 1]).all() or not np.isfinite(score).all():
        raise ValueError("Targets must be binary and scores finite.")
    return target.astype(int), score


def zero_fn_threshold(target, score):
    """Highest threshold retaining every positive in these calibration rows.

    Predict positive for score >= threshold. This is not a guarantee for new rows.
    """
    target, score = checked_scores(target, score)
    if not np.any(target == 1):
        raise ValueError("Calibration needs at least one positive.")
    return float(score[target == 1].min())


def binary_metrics(target, prediction):
    target, prediction = checked_scores(target, prediction)
    if not np.isin(prediction, [0, 1]).all():
        raise ValueError("Predictions must be binary.")
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {"nodules": len(target), "TP": int(tp), "FN": int(fn),
            "TN": int(tn), "FP": int(fp), "sensitivity": sensitivity,
            "specificity": specificity, "accuracy": (tp + tn) / len(target),
            "balanced_accuracy": (sensitivity + specificity) / 2,
            "precision": tp / (tp + fp) if tp + fp else 0.0}


def zero_fn_summary(target, score):
    """Descriptive ROC point chosen on the evaluated rows, not a validation result."""
    target, score = checked_scores(target, score)
    if np.unique(target).size != 2:
        raise ValueError("ROC evaluation needs both binary classes.")
    threshold = zero_fn_threshold(target, score)
    return {"threshold": threshold, **binary_metrics(target, score >= threshold),
            "AUC": roc_auc_score(target, score)}


def calibration_splits(table):
    """Rotate disjoint training, calibration and test patient folds.

    For each test fold, use the next saved fold for threshold calibration. The
    model is not refitted after calibration. All supplied rows must be LNDb.
    """
    if table["fold"].isna().any() or table.groupby("LNDbID")["fold"].nunique().gt(1).any():
        raise ValueError("Every patient needs exactly one saved fold.")
    folds = sorted(table["fold"].unique())
    if len(folds) < 3:
        raise ValueError("Need at least three patient folds.")
    for index, test_fold in enumerate(folds):
        calibration_fold = folds[(index + 1) % len(folds)]
        test = table["fold"].eq(test_fold)
        calibration = table["fold"].eq(calibration_fold)
        yield test_fold, calibration_fold, ~(test | calibration), calibration, test


def positive_score(model, values, binary=True):
    """Probability score for a binary-trained or five-class-derived alert."""
    probabilities = model.predict_proba(values)
    positive = model.classes_ == 1 if binary else np.isin(model.classes_, [4, 5])
    return probabilities[:, positive].sum(axis=1)


def run_calibration(output):
    """Fixed segmentation recipes with a separate threshold-calibration fold."""
    paths = load_paths()
    lndb, features = load_data(
        "data/lndb_ge3_segmentation_3d_c50_hu.csv", "lndb",
        targets_path=paths["lndb"]["segmentation_targets"], paths=paths,
        exclude_duplicate_lndb=True,
    )
    lidc, _ = load_data("data/lidc_segmentation_3d_c50_hu.csv", "lidc", paths=paths)
    recipes = {
        "Binary XGBoost, all features": (True, "xgboost", False, None),
        "Binary Balanced RF, all features": (True, "rf", True, None),
        "Binary RF, intensity": (True, "rf", False, ["firstorder"]),
        "Five-class RF, intensity, P(4)+P(5)": (False, "rf", False, ["firstorder"]),
    }
    predictions, fold_metrics = [], []
    for name, (binary, kind, balanced_forest, families) in recipes.items():
        columns = select_families(features, families) if families else features
        source = lndb[lndb[BINARY_TARGET].notna()].copy() if binary else lndb
        other = lidc[lidc[BINARY_TARGET].notna()].copy() if binary else lidc
        target_column = BINARY_TARGET if binary else TARGET
        weight = None if balanced_forest else "balanced"
        for fold, calibration_fold, train_rows, calibration_rows, test_rows in calibration_splits(source):
            train = pd.concat([other, source.loc[train_rows]], ignore_index=True)
            calibration = source.loc[calibration_rows & source[BINARY_TARGET].notna()]
            test = source.loc[test_rows & source[BINARY_TARGET].notna()]
            model = make_model(weight, balanced_forest=balanced_forest, model_kind=kind, binary=binary)
            fit_model(model, train[columns], train[target_column].astype(int), weight, 0)
            threshold = zero_fn_threshold(
                calibration[BINARY_TARGET], positive_score(model, calibration[columns], binary),
            )
            score = positive_score(model, test[columns], binary)
            prediction = score >= threshold
            rows = test[LNDB_KEYS + ["fold", BINARY_TARGET]].copy()
            rows["model"] = name
            rows["calibration_fold"] = calibration_fold
            rows["calibration_positives"] = int(calibration[BINARY_TARGET].sum())
            rows["threshold"] = threshold
            rows["score"] = score
            rows["prediction"] = prediction.astype(int)
            predictions.append(rows)
            result = {"model": name, "fold": fold, "calibration_fold": calibration_fold,
                      "threshold": threshold, **binary_metrics(test[BINARY_TARGET], prediction)}
            fold_metrics.append(result)
            print(f"{name}, fold {fold}: FN={result['FN']}, FP={result['FP']}, "
                  f"accuracy={result['accuracy']:.3f}", flush=True)
    predictions = pd.concat(predictions, ignore_index=True)
    overall = [{"model": name, "fold": "overall_oof", **binary_metrics(
        rows[BINARY_TARGET], rows["prediction"])} for name, rows in predictions.groupby("model", sort=False)]
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    pd.DataFrame(fold_metrics + overall).to_csv(output / "metrics.csv", index=False)
    pd.Series(sorted(EXCLUDED_LNDB_IDS), name="LNDbID").to_csv(output / "excluded_ct_copies.csv", index=False)
    print(pd.DataFrame(overall).round(3).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/segmentation_binary_calibration"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    run_calibration(args.output)


if __name__ == "__main__":
    main()
