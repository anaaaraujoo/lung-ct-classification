"""Random Forest using the saved LNDb patient folds, with LIDC added to training."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, mean_absolute_error, recall_score
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
TARGET = "target_classification_mean"
FEATURE_PATTERN = r"_(firstorder|glcm|glrlm|glszm|gldm|ngtdm)_"


def load_data(path, dataset):
    table = pd.read_csv(path)
    features = table.filter(regex=FEATURE_PATTERN).columns.tolist()
    keys = ["patient_id", "series_uid", "nodule_id"] if dataset == "lidc" else ["LNDbID", "FindingID"]
    targets = pd.read_csv(ROOT / f"data/processed/{dataset}_nodule_targets.csv").rename(columns={"NoduleID": "FindingID"})
    folds = pd.read_csv(ROOT / f"data/processed/{dataset}_patient_folds.csv")
    data = targets.merge(folds, on=keys[0], validate="many_to_one", how="left")
    data = data.merge(table[keys + features], on=keys, validate="one_to_one", how="left", indicator=True)
    if not features or len(table) != len(targets) or data["_merge"].ne("both").any():
        raise ValueError(f"{path}: expected features for all {len(targets)} training nodules.")
    if data["fold"].isna().any() or not data[TARGET].isin([1, 2, 3, 4, 5]).all():
        raise ValueError(f"{dataset}: missing folds or invalid target classes.")
    data[features] = data[features].replace([np.inf, -np.inf], np.nan)
    return data.drop(columns="_merge"), features


def make_model(class_weight=None, k=None):
    steps = [SimpleImputer(strategy="median", keep_empty_features=True)]
    if k is not None:
        steps.append(SelectKBest(f_classif, k=k))
    steps.append(RandomForestClassifier(n_estimators=500, class_weight=class_weight, random_state=42, n_jobs=-1))
    return make_pipeline(*steps)


def metrics(y, prediction):
    return {"balanced_accuracy": balanced_accuracy_score(y, prediction),
            "macro_f1": f1_score(y, prediction, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0),
            "qwk": cohen_kappa_score(y, prediction, labels=[1, 2, 3, 4, 5], weights="quadratic"),
            "mae": mean_absolute_error(y, prediction)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lidc", type=Path, default=ROOT / "data/lidc_circle_r21_hu.csv")
    parser.add_argument("--lndb", type=Path, default=ROOT / "data/lndb_circle_r11_hu.csv")
    parser.add_argument("--lndb-only", action="store_true", help="Train using only the LNDb training folds.")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--k", type=int, help="Select the K best features inside each training fold. Default: use all.")
    parser.add_argument("--output", type=Path, help="Directory for this experiment's results and model.")
    args = parser.parse_args()
    lndb, features = load_data(args.lndb, "lndb")
    lidc = lndb.head(0)
    if not args.lndb_only:
        lidc, lidc_features = load_data(args.lidc, "lidc")
        if set(features) != set(lidc_features):
            raise ValueError("LIDC and LNDb must contain the same radiomics features.")
        features = lidc_features
    class_weight = None if args.class_weight == "none" else args.class_weight
    if args.k is not None and not 1 <= args.k <= len(features):
        parser.error(f"--k must be between 1 and {len(features)}.")
    folds = sorted(lndb["fold"].unique())
    if len(folds) != 5:
        raise ValueError("Expected five saved LNDb folds.")
    name = args.lndb.stem if args.lndb_only else f"{args.lidc.stem}_{args.lndb.stem}"
    output = args.output or ROOT / "models" / f"rf_{name}{'_balanced' if class_weight else ''}{f'_k{args.k}' if args.k is not None else ''}"
    output.mkdir(parents=True, exist_ok=True)
    predictions = pd.Series(index=lndb.index, dtype=float)
    scores = []
    for fold in folds:
        held_out = lndb["fold"].eq(fold)
        train = pd.concat([lidc, lndb.loc[~held_out]], ignore_index=True)
        model = make_model(class_weight, args.k)  # Imputation, selection and weights use training patients only.
        print(f"Fold {fold}: {len(train)} training / {held_out.sum()} validation nodules", flush=True)
        model.fit(train[features], train[TARGET])
        prediction = model.predict(lndb.loc[held_out, features])
        predictions.loc[held_out] = prediction
        scores.append({"fold": fold, **metrics(lndb.loc[held_out, TARGET], prediction)})
        print(pd.Series(scores[-1]).to_string(), flush=True)
    scores.append({"fold": "overall_oof", **metrics(lndb[TARGET], predictions)})
    pd.DataFrame(scores).to_csv(output / "metrics.csv", index=False)
    oof = lndb[["LNDbID", "FindingID", "fold", TARGET]].copy()
    oof["prediction"] = predictions.astype(int)
    oof.to_csv(output / "validation_predictions.csv", index=False)
    labels = [1, 2, 3, 4, 5]
    confusion = pd.DataFrame(confusion_matrix(lndb[TARGET], predictions, labels=labels), index=labels, columns=labels)
    confusion.to_csv(output / "confusion_matrix.csv", index_label="true_class")
    pd.DataFrame({"class": labels, "recall": recall_score(lndb[TARGET], predictions, labels=labels, average=None, zero_division=0)}).to_csv(output / "recall_per_class.csv", index=False)
    print(f"Fitting final model on all {'LNDb' if args.lndb_only else 'LIDC + LNDb'} training nodules", flush=True)
    train = pd.concat([lidc, lndb], ignore_index=True)
    model = make_model(class_weight, args.k)
    model.fit(train[features], train[TARGET])
    joblib.dump({"model": model, "features": features, "target": TARGET,
                 "lidc_csv": None if args.lndb_only else str(args.lidc.resolve()),
                 "lndb_csv": str(args.lndb.resolve()), "class_weight": class_weight, "k": args.k}, output / "model.joblib")
    print(f"Saved metrics, validation predictions and model to {output}")


if __name__ == "__main__":
    main()
