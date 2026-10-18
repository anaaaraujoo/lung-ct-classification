"""Tune segmentation-based binary alerts without using calibration or test labels."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .binary_evaluation import (BINARY_TARGET, binary_metrics, calibration_splits,
                                positive_score, zero_fn_threshold)
from .modeling import (EXCLUDED_LNDB_IDS, LNDB_KEYS, fit_model, load_data,
                       make_model, size_confounded_features)
from .radiomics_extraction import load_paths

DIAMETER_COLUMN = "original_shape_Maximum3DDiameter"
MODEL_NAMES = {"xgboost": "Binary XGBoost, all features",
               "brf": "Binary Balanced RF, all features"}


def diameter_filtered_features(train, features, maximum=None):
    """Remove features with |Spearman correlation| > maximum on training rows only."""
    if maximum is None:
        return list(features)
    if not 0 < maximum < 1:
        raise ValueError("Maximum diameter correlation must be between 0 and 1.")
    dropped = size_confounded_features(features, train, maximum, column=DIAMETER_COLUMN)
    kept = [name for name in features if name not in dropped]
    if not kept:
        raise ValueError("No features remain after diameter filtering.")
    return kept


def parameter_candidates(kind):
    """Small prespecified grids, including each unchanged baseline."""
    if kind == "xgboost":
        return [
            {},
            {"xgbclassifier__max_depth": 2, "xgbclassifier__min_child_weight": 5},
            {"xgbclassifier__max_depth": 3, "xgbclassifier__min_child_weight": 5},
            {"xgbclassifier__max_depth": 3, "xgbclassifier__min_child_weight": 5,
             "xgbclassifier__reg_lambda": 10},
        ]
    if kind == "brf":
        return [
            {},
            {"balancedrandomforestclassifier__max_features": 0.1},
            {"balancedrandomforestclassifier__min_samples_leaf": 5},
            {"balancedrandomforestclassifier__max_features": 0.1,
             "balancedrandomforestclassifier__min_samples_leaf": 5},
        ]
    raise ValueError(f"Unknown binary model: {kind}")


def new_model(kind, params):
    balanced_forest = kind == "brf"
    weight = None if balanced_forest else "balanced"
    model = make_model(weight, balanced_forest=balanced_forest,
                       model_kind="rf" if balanced_forest else kind, binary=True)
    return model.set_params(**params), weight


def tune_parameters(lidc, lndb, features, kind, maximum=None):
    """Choose parameters on the three outer-training LNDb folds only.

    Each inner rotation has disjoint fit, threshold-calibration and validation
    patients. Rank candidates first by total validation FN, then by specificity.
    No outer calibration or test rows enter this search.
    """
    splits = []
    for _, _, fit_rows, calibration_rows, validation_rows in calibration_splits(lndb):
        train = pd.concat([lidc, lndb.loc[fit_rows]], ignore_index=True)
        columns = diameter_filtered_features(train, features, maximum)
        splits.append((train, lndb.loc[calibration_rows], lndb.loc[validation_rows], columns))
    results, best_params, best_rank = [], None, None
    for candidate, params in enumerate(parameter_candidates(kind)):
        targets, predictions = [], []
        for train, calibration, validation, columns in splits:
            model, weight = new_model(kind, params)
            fit_model(model, train[columns], train[BINARY_TARGET], weight, 0)
            threshold = zero_fn_threshold(
                calibration[BINARY_TARGET], positive_score(model, calibration[columns]),
            )
            predictions.extend(positive_score(model, validation[columns]) >= threshold)
            targets.extend(validation[BINARY_TARGET])
        result = {"candidate": candidate, "params": json.dumps(params, sort_keys=True),
                  **binary_metrics(targets, predictions)}
        results.append(result)
        rank = (-result["FN"], result["specificity"])
        if best_rank is None or rank > best_rank:
            best_params, best_rank = params, rank
        print(f"  {kind} candidate {candidate}: inner FN={result['FN']}, "
              f"specificity={result['specificity']:.3f}", flush=True)
    results = pd.DataFrame(results)
    results["selected"] = results.params.eq(json.dumps(best_params, sort_keys=True))
    return best_params, results


def run_experiment(output, maximum=None, tune=False):
    paths = load_paths()
    lndb, features = load_data(
        "data/lndb_ge3_segmentation_3d_c50_hu.csv", "lndb", binary=True,
        targets_path=paths["lndb"]["segmentation_targets"], paths=paths,
        exclude_duplicate_lndb=True,
    )
    lidc, lidc_features = load_data("data/lidc_segmentation_3d_c50_hu.csv", "lidc", binary=True, paths=paths)
    if set(features) != set(lidc_features):
        raise ValueError("LIDC and LNDb must share the same segmentation features.")
    settings = {"diameter_column": DIAMETER_COLUMN, "max_diameter_correlation": maximum,
                "tune": tune, "inner_selection": "minimum FN, then maximum specificity",
                "excluded_lndb_ids": sorted(EXCLUDED_LNDB_IDS),
                "parameter_candidates": {kind: parameter_candidates(kind) for kind in MODEL_NAMES}}
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    predictions, scores, tuning, selected_features = [], [], [], []
    for kind, name in MODEL_NAMES.items():
        for fold, calibration_fold, train_rows, calibration_rows, test_rows in calibration_splits(lndb):
            training_lndb = lndb.loc[train_rows]
            params = {}
            if tune:
                print(f"{name}, outer fold {fold}: inner search", flush=True)
                params, search = tune_parameters(lidc, training_lndb, features, kind, maximum)
                tuning.append(search.assign(model=name, outer_fold=fold))
                pd.concat(tuning, ignore_index=True).to_csv(output / "tuning.csv", index=False)
            train = pd.concat([lidc, training_lndb], ignore_index=True)
            calibration, test = lndb.loc[calibration_rows], lndb.loc[test_rows]
            columns = diameter_filtered_features(train, features, maximum)
            model, weight = new_model(kind, params)
            fit_model(model, train[columns], train[BINARY_TARGET], weight, 0)
            threshold = zero_fn_threshold(
                calibration[BINARY_TARGET], positive_score(model, calibration[columns]),
            )
            score = positive_score(model, test[columns])
            prediction = score >= threshold
            rows = test[LNDB_KEYS + ["fold", BINARY_TARGET]].copy()
            rows["model"], rows["calibration_fold"] = name, calibration_fold
            rows["threshold"], rows["score"], rows["prediction"] = threshold, score, prediction.astype(int)
            predictions.append(rows)
            result = {"model": name, "fold": fold, "features": len(columns),
                      "threshold": threshold, "params": json.dumps(params, sort_keys=True),
                      **binary_metrics(test[BINARY_TARGET], prediction)}
            scores.append(result)
            selected_features.extend({"model": name, "fold": fold, "feature": column} for column in columns)
            pd.concat(predictions, ignore_index=True).to_csv(output / "validation_predictions.csv", index=False)
            pd.DataFrame(scores).to_csv(output / "metrics.csv", index=False)
            pd.DataFrame(selected_features).to_csv(output / "selected_features.csv", index=False)
            print(f"{name}, fold {fold}: {len(columns)} features, FN={result['FN']}, "
                  f"FP={result['FP']}, accuracy={result['accuracy']:.3f}", flush=True)
    all_predictions = pd.concat(predictions, ignore_index=True)
    overall = [{"model": name, "fold": "overall_oof", **binary_metrics(
        rows[BINARY_TARGET], rows.prediction)} for name, rows in all_predictions.groupby("model", sort=False)]
    pd.DataFrame(scores + overall).to_csv(output / "metrics.csv", index=False)
    print(pd.DataFrame(overall).round(3).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-diameter-correlation", type=float,
                        help="Drop features above this absolute Spearman correlation with maximum 3D diameter.")
    parser.add_argument("--tune", action="store_true", help="Select parameters on inner patient folds by FN and specificity.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_diameter_correlation is not None and not 0 < args.max_diameter_correlation < 1:
        parser.error("--max-diameter-correlation must be between 0 and 1.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    run_experiment(args.output, args.max_diameter_correlation, args.tune)


if __name__ == "__main__":
    main()
