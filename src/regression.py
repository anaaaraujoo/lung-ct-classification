"""Fit Linear Regression w/ L2 regularization to continuous means and compare rounded classes on saved LNDb folds."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix, mean_absolute_error, recall_score, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__:
    from .modeling import EXCLUDED_LNDB_IDS, ROOT, TARGET, load_data, metrics
    from .radiomics_extraction import load_paths
else:
    from modeling import EXCLUDED_LNDB_IDS, ROOT, TARGET, load_data, metrics
    from radiomics_extraction import load_paths

REGRESSION_TARGET = "target_regression_mean"


def to_classes(predictions):
    """Match the saved targets' half-up rounding, limiting classes to 1–5."""
    return np.clip(np.floor(np.asarray(predictions) + 0.5), 1, 5).astype(int)


def load_regression_data(path, dataset, targets_path=None, paths=None, exclude_duplicate_lndb=False):
    data, features = load_data(path, dataset, targets_path=targets_path, paths=paths,
                               exclude_duplicate_lndb=exclude_duplicate_lndb)
    if (REGRESSION_TARGET not in data or not np.isfinite(data[REGRESSION_TARGET]).all()
            or not data[REGRESSION_TARGET].between(1, 5).all()):
        raise ValueError(f"{dataset}: regression targets must be finite reader means from 1 to 5.")
    return data, features


def make_model():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge())


def regression_metrics(y, predictions):
    # Evaluate raw continuous outputs, clipping only the converted classes.
    return {"mae": mean_absolute_error(y, predictions), "rmse": root_mean_squared_error(y, predictions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=ROOT / "configs/paths.yaml", help="YAML with file locations.")
    parser.add_argument("--lidc", type=Path, help="Override the YAML LIDC feature CSV.")
    parser.add_argument("--lndb", type=Path, help="Override the YAML LNDb feature CSV.")
    parser.add_argument("--lidc-targets", type=Path, help="Explicit LIDC nodule population.")
    parser.add_argument("--lndb-targets", type=Path, help="Explicit LNDb nodule population, e.g. the saved ge3 targets.")
    parser.add_argument("--lndb-only", action="store_true", help="Train using only the LNDb training folds.")
    parser.add_argument("--exclude-duplicate-lndb", action="store_true",
                        help="Exclude confirmed duplicate LNDb CT scans. Default: use all scans.")
    parser.add_argument("--original-only", action="store_true", help="Use only features from the unfiltered Original image.")
    parser.add_argument("--output", type=Path, help="Directory for this experiment's results and model.")
    args = parser.parse_args()
    paths = load_paths(args.paths)
    args.lndb = args.lndb or paths["training"]["lndb_features"]

    lndb, features = load_regression_data(args.lndb, "lndb", args.lndb_targets, paths,
                                          exclude_duplicate_lndb=args.exclude_duplicate_lndb)
    lidc = lndb.head(0)
    if not args.lndb_only:
        args.lidc = args.lidc or paths["training"]["lidc_features"]
        lidc, lidc_features = load_regression_data(args.lidc, "lidc", args.lidc_targets, paths)
        if set(features) != set(lidc_features):
            raise ValueError("LIDC and LNDb must contain the same radiomics features.")
        features = lidc_features
    if args.original_only:
        features = [name for name in features if name.startswith("original_")]
        if not features:
            parser.error("No Original features found in the supplied CSVs.")
    folds = sorted(lndb["fold"].unique())
    if len(folds) != 5:
        raise ValueError("Expected five saved LNDb folds.")
    name = args.lndb.stem if args.lndb_only else f"{args.lidc.stem}_{args.lndb.stem}"
    output = args.output or paths["outputs"]["models"] / f"ridge_{name}{'_original' if args.original_only else ''}{'_dedup' if args.exclude_duplicate_lndb else ''}"
    output.mkdir(parents=True, exist_ok=True)
    predictions = pd.Series(index=lndb.index, dtype=float)
    scores, continuous_scores = [], []
    print(f"Ridge: {len(features)} features, target: {REGRESSION_TARGET}", flush=True)
    for fold in folds:
        held_out = lndb["fold"].eq(fold)
        train = pd.concat([lidc, lndb.loc[~held_out]], ignore_index=True)
        model = make_model()
        print(f"Fold {fold}: {len(train)} training / {held_out.sum()} validation nodules", flush=True)
        model.fit(train[features], train[REGRESSION_TARGET])
        prediction = model.predict(lndb.loc[held_out, features])
        predictions.loc[held_out] = prediction
        continuous_scores.append({"fold": fold, **regression_metrics(lndb.loc[held_out, REGRESSION_TARGET], prediction)})
        scores.append({"fold": fold, **metrics(lndb.loc[held_out, TARGET], to_classes(prediction))})
        print(pd.Series(scores[-1]).to_string(), flush=True)
        print(pd.Series(continuous_scores[-1]).drop("fold").add_prefix("regression_").to_string(), flush=True)

    classes = to_classes(predictions)
    continuous_scores.append({"fold": "overall_oof", **regression_metrics(lndb[REGRESSION_TARGET], predictions)})
    scores.append({"fold": "overall_oof", **metrics(lndb[TARGET], classes)})
    pd.DataFrame(continuous_scores).to_csv(output / "regression_metrics.csv", index=False)
    pd.DataFrame(scores).to_csv(output / "metrics.csv", index=False)
    oof = lndb[["LNDbID", "FindingID", "fold", REGRESSION_TARGET, TARGET]].copy()
    oof["prediction_continuous"] = predictions
    oof["prediction"] = classes
    oof.to_csv(output / "validation_predictions.csv", index=False)
    labels = [1, 2, 3, 4, 5]
    pd.DataFrame(confusion_matrix(lndb[TARGET], classes, labels=labels), index=labels, columns=labels).to_csv(
        output / "confusion_matrix.csv", index_label="true_class"
    )
    pd.DataFrame({"class": labels, "recall": recall_score(lndb[TARGET], classes, labels=labels, average=None, zero_division=0)}).to_csv(
        output / "recall_per_class.csv", index=False
    )

    print(f"Fitting final model on all {'LNDb' if args.lndb_only else 'LIDC + LNDb'} training nodules", flush=True)
    train = pd.concat([lidc, lndb], ignore_index=True)
    model = make_model().fit(train[features], train[REGRESSION_TARGET])
    lidc_targets_path = None if args.lndb_only else (args.lidc_targets or paths["lidc"]["targets"])
    lndb_targets_path = args.lndb_targets or paths["lndb"]["targets"]
    joblib.dump({"model": model, "features": features, "target": REGRESSION_TARGET,
                 "model_kind": "ridge", "class_conversion": "half_up_clip_1_5", "label_offset": 0,
                 "lidc_csv": None if args.lndb_only else str(args.lidc.resolve()),
                 "lndb_csv": str(args.lndb.resolve()), "original_only": args.original_only,
                 "lidc_targets_csv": None if args.lndb_only else str(lidc_targets_path.resolve()),
                 "lndb_targets_csv": str(lndb_targets_path.resolve()),
                  "excluded_lndb_ids": sorted(EXCLUDED_LNDB_IDS) if args.exclude_duplicate_lndb else []},
                 output / "model.joblib")
    print(f"Saved regression and classification metrics, validation predictions and model to {output}")


if __name__ == "__main__":
    main()
