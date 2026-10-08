"""Compare classifiers using the saved LNDb patient folds, adding LIDC to training."""
import argparse
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, mean_absolute_error, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

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


def make_model(class_weight=None, k=None, smote=False, balanced_forest=False, model_kind="rf"):
    steps = [SimpleImputer(strategy="median", keep_empty_features=True)]
    if k is not None:
        steps.append(SelectKBest(f_classif, k=k))
    if smote or model_kind == "logistic":
        steps.append(StandardScaler())
    if smote:
        steps.append(SMOTE(random_state=42))
    if model_kind == "logistic":
        classifier = LogisticRegression(penalty="l1", solver="saga", C=1, class_weight=class_weight, max_iter=20000, random_state=42)
    elif model_kind == "xgboost":
        classifier = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                   colsample_bytree=0.8, objective="multi:softprob", eval_metric="mlogloss",
                                   tree_method="hist", random_state=42, n_jobs=-1)
    elif balanced_forest:
        classifier = BalancedRandomForestClassifier(n_estimators=500, sampling_strategy="all", replacement=True,
                                                    bootstrap=False, random_state=42, n_jobs=-1)
    else:
        classifier = RandomForestClassifier(n_estimators=500, class_weight=class_weight, random_state=42, n_jobs=-1)
    steps.append(classifier)
    return make_pipeline(*steps)


def fit_model(model, X, y, class_weight, label_offset):
    # XGBoost needs classes 0–4 and sample weights instead of a class_weight parameter.
    params = {"xgbclassifier__sample_weight": compute_sample_weight(class_weight, y)} if label_offset and class_weight else {}
    return model.fit(X, y - label_offset, **params)


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
    parser.add_argument("--model", choices=["rf", "logistic", "xgboost"], default="rf")
    parser.add_argument("--original-only", action="store_true", help="Use only features from the unfiltered Original image.")
    parser.add_argument("--smote", action="store_true", help="Scale features and balance classes with SMOTE on training data only.")
    parser.add_argument("--balanced-forest", action="store_true", help="Balance each tree's training sample by undersampling.")
    parser.add_argument("--k", type=int, help="Select the K best features inside each training fold. Default: use all.")
    parser.add_argument("--output", type=Path, help="Directory for this experiment's results and model.")
    args = parser.parse_args()
    if args.smote and args.class_weight != "none":
        parser.error("Use --smote without --class-weight balanced to test resampling separately.")
    if args.balanced_forest and (args.smote or args.class_weight != "none"):
        parser.error("Use --balanced-forest without SMOTE or class weights to test undersampling separately.")
    if args.balanced_forest and args.model != "rf":
        parser.error("--balanced-forest requires --model rf.")
    if args.smote or args.balanced_forest:
        # imbalanced-learn 0.12 uses these older APIs, which still work in sklearn 1.6.
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\.base",
                                message=r"`BaseEstimator\._(validate_data|check_feature_names|check_n_features)` is deprecated")
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\.utils\.deprecation",
                                message="'force_all_finite' was renamed to 'ensure_all_finite'")
    lndb, features = load_data(args.lndb, "lndb")
    lidc = lndb.head(0)
    if not args.lndb_only:
        lidc, lidc_features = load_data(args.lidc, "lidc")
        if set(features) != set(lidc_features):
            raise ValueError("LIDC and LNDb must contain the same radiomics features.")
        features = lidc_features
    if args.original_only:
        features = [column for column in features if column.startswith("original_")]
        if not features:
            parser.error("No Original features found in the supplied CSVs.")
    class_weight = None if args.class_weight == "none" else args.class_weight
    label_offset = int(args.model == "xgboost")
    if args.k is not None and not 1 <= args.k <= len(features):
        parser.error(f"--k must be between 1 and {len(features)}.")
    folds = sorted(lndb["fold"].unique())
    if len(folds) != 5:
        raise ValueError("Expected five saved LNDb folds.")
    name = args.lndb.stem if args.lndb_only else f"{args.lidc.stem}_{args.lndb.stem}"
    kind = "brf" if args.balanced_forest else args.model
    output = args.output or ROOT / "models" / f"{kind}_{name}{'_balanced' if class_weight else ''}{f'_k{args.k}' if args.k is not None else ''}{'_smote' if args.smote else ''}{'_original' if args.original_only else ''}"
    output.mkdir(parents=True, exist_ok=True)
    predictions = pd.Series(index=lndb.index, dtype=float)
    scores = []
    print(f"{kind}: {len(features)} features, class weights: {args.class_weight}", flush=True)
    for fold in folds:
        held_out = lndb["fold"].eq(fold)
        train = pd.concat([lidc, lndb.loc[~held_out]], ignore_index=True)
        model = make_model(class_weight, args.k, args.smote, args.balanced_forest, args.model)
        print(f"Fold {fold}: {len(train)} training / {held_out.sum()} validation nodules", flush=True)
        fit_model(model, train[features], train[TARGET], class_weight, label_offset)
        prediction = model.predict(lndb.loc[held_out, features]) + label_offset
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
    model = make_model(class_weight, args.k, args.smote, args.balanced_forest, args.model)
    fit_model(model, train[features], train[TARGET], class_weight, label_offset)
    joblib.dump({"model": model, "features": features, "target": TARGET,
                 "lidc_csv": None if args.lndb_only else str(args.lidc.resolve()),
                 "lndb_csv": str(args.lndb.resolve()), "class_weight": class_weight, "k": args.k,
                 "smote": args.smote, "balanced_forest": args.balanced_forest,
                 "model_kind": args.model, "label_offset": label_offset, "original_only": args.original_only}, output / "model.joblib")
    print(f"Saved metrics, validation predictions and model to {output}")


if __name__ == "__main__":
    main()
