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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, make_scorer, mean_absolute_error, recall_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# Support both `python src/modeling.py` and importing `src.modeling`.
if __package__:
    from .radiomics_extraction import load_paths
else:
    from radiomics_extraction import load_paths

ROOT = Path(__file__).resolve().parents[1]
TARGET = "target_classification_mean"
FEATURE_PATTERN = r"_(shape2D|firstorder|glcm|glrlm|glszm|gldm|ngtdm)_"
MIN_SAMPLES_GRID = {
    "randomforestclassifier__min_samples_leaf": [1, 5, 10, 20],
    "randomforestclassifier__min_samples_split": [2, 10, 20, 40],
}


def load_data(path, dataset, targets_path=None, paths=None, binary=False):
    """Load every requested nodule and use its saved patient fold."""
    table = pd.read_csv(path)
    features = table.filter(regex=FEATURE_PATTERN).columns.tolist()
    keys = ["patient_id", "series_uid", "nodule_id"] if dataset == "lidc" else ["LNDbID", "FindingID"]
    locations = (paths or load_paths())[dataset]
    targets_file = targets_path or locations["targets"]
    targets = pd.read_csv(targets_file).rename(columns={"NoduleID": "FindingID"})
    # An explicit target table defines the experiment's population, including for the circle comparison.
    target_columns = keys + [name for name in targets if name == "ratings" or name.startswith("target_")]
    targets = targets[target_columns]
    if targets.empty or targets[keys].isna().any().any() or targets.duplicated(keys).any():
        raise ValueError("Targets must contain unique, non-missing nodule IDs.")
    target = "target_binary_mean" if binary else TARGET
    labels = [0, 1] if binary else [1, 2, 3, 4, 5]
    if binary:
        targets = targets[targets[target].notna()].copy()
        if targets.empty:
            raise ValueError(f"{dataset}: no nodules have a binary target.")
    if table[keys].isna().any().any() or table.duplicated(keys).any():
        raise ValueError(f"{path}: feature rows must have unique, non-missing nodule IDs.")
    folds = pd.read_csv(locations["folds"])
    data = targets.merge(folds, on=keys[0], validate="many_to_one", how="left")
    data = data.merge(table[keys + features], on=keys, validate="one_to_one", how="left", indicator=True)
    if not features or (targets_path is None and not binary and len(table) != len(targets)) or data["_merge"].ne("both").any():
        raise ValueError(f"{path}: expected features for all {len(targets)} training nodules.")
    if data["fold"].isna().any() or not data[target].isin(labels).all():
        raise ValueError(f"{dataset}: missing folds or invalid target classes.")
    data[target] = data[target].astype(int)
    data[features] = data[features].replace([np.inf, -np.inf], np.nan)
    return data.drop(columns="_merge"), features


def make_model(class_weight=None, k=None, smote=False, balanced_forest=False, model_kind="rf", binary=False):
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
                                   colsample_bytree=0.8, objective="binary:logistic" if binary else "multi:softprob",
                                   eval_metric="logloss" if binary else "mlogloss",
                                   tree_method="hist", random_state=42, n_jobs=-1)
    elif balanced_forest:
        classifier = BalancedRandomForestClassifier(n_estimators=500, sampling_strategy="all", replacement=True,
                                                    bootstrap=False, random_state=42, n_jobs=-1)
    else:
        classifier = RandomForestClassifier(n_estimators=500, class_weight=class_weight, random_state=42, n_jobs=-1)
    steps.append(classifier)
    return make_pipeline(*steps)


def fit_model(model, X, y, class_weight, label_offset):
    # XGBoost uses sample weights for both binary and five-class training.
    params = {"xgbclassifier__sample_weight": compute_sample_weight(class_weight, y)} if "xgbclassifier" in model.named_steps and class_weight else {}
    return model.fit(X, y - label_offset, **params)


def metrics(y, prediction, binary=False):
    labels = [0, 1] if binary else [1, 2, 3, 4, 5]
    return {"balanced_accuracy": balanced_accuracy_score(y, prediction),
            "macro_f1": f1_score(y, prediction, labels=labels, average="macro", zero_division=0),
            "qwk": cohen_kappa_score(y, prediction, labels=labels, weights="quadratic"),
            "mae": mean_absolute_error(y, prediction)}


def lndb_cv(train, n_lidc):
    """Validate on saved LNDb folds, keeping the leading LIDC rows in training only."""
    test_fold = train["fold"].to_numpy().copy()
    test_fold[:n_lidc] = -1
    return PredefinedSplit(test_fold)


def tune_rf(model, train, features, n_lidc, output, fold, binary=False):
    target = "target_binary_mean" if binary else TARGET
    labels = [0, 1] if binary else [1, 2, 3, 4, 5]
    scorer = make_scorer(f1_score, labels=labels, average="macro", zero_division=0)
    search = GridSearchCV(model, MIN_SAMPLES_GRID, scoring=scorer, cv=lndb_cv(train, n_lidc),
                          n_jobs=1, error_score="raise", return_train_score=True, verbose=2)
    # Parallelize the 500 trees, not searches and trees simultaneously.
    search.fit(train[features], train[target])
    pd.DataFrame(search.cv_results_).to_csv(output / f"grid_{fold}.csv", index=False)
    print(f"Best inner macro F1: {search.best_score_:.4f}, parameters: {search.best_params_}", flush=True)
    return search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=ROOT / "configs/paths.yaml", help="YAML with file locations.")
    parser.add_argument("--lidc", type=Path, help="Override the YAML LIDC feature CSV.")
    parser.add_argument("--lndb", type=Path, help="Override the YAML LNDb feature CSV.")
    parser.add_argument("--lidc-targets", type=Path, help="Explicit LIDC nodule population. Default: all cleaned LIDC targets.")
    parser.add_argument("--lndb-targets", type=Path, help="Explicit LNDb nodule population, e.g. the saved ge3 targets.")
    parser.add_argument("--lndb-only", action="store_true", help="Train using only the LNDb training folds.")
    parser.add_argument("--binary", action="store_true", help="Use saved target_binary_mean: mean < 3 is 0, mean > 3 is 1, mean = 3 is excluded.")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--model", choices=["rf", "logistic", "xgboost"], default="rf")
    parser.add_argument("--original-only", action="store_true", help="Use only features from the unfiltered Original image.")
    parser.add_argument("--smote", action="store_true", help="Scale features and balance classes with SMOTE on training data only.")
    parser.add_argument("--balanced-forest", action="store_true", help="Balance each tree's training sample by undersampling.")
    parser.add_argument("--k", type=int, help="Select the K best features inside each training fold. Default: use all.")
    parser.add_argument("--tune-rf", action="store_true", help="Tune RF minimum leaf and split sizes inside the saved training folds, selecting by macro F1.")
    parser.add_argument("--output", type=Path, help="Directory for this experiment's results and model.")
    args = parser.parse_args()
    args.paths = args.paths.expanduser().resolve()
    paths = load_paths(args.paths)
    args.lidc = args.lidc or paths["training"]["lidc_features"]
    args.lndb = args.lndb or paths["training"]["lndb_features"]
    target = "target_binary_mean" if args.binary else TARGET
    labels = [0, 1] if args.binary else [1, 2, 3, 4, 5]
    if args.smote and args.class_weight != "none":
        parser.error("Use --smote without --class-weight balanced to test resampling separately.")
    if args.balanced_forest and (args.smote or args.class_weight != "none"):
        parser.error("Use --balanced-forest without SMOTE or class weights to test undersampling separately.")
    if args.balanced_forest and args.model != "rf":
        parser.error("--balanced-forest requires --model rf.")
    if args.tune_rf and (args.model != "rf" or args.balanced_forest or args.smote):
        parser.error("--tune-rf requires ordinary RF without SMOTE or --balanced-forest.")
    if args.smote or args.balanced_forest:
        # imbalanced-learn 0.12 uses these older APIs, which still work in sklearn 1.6.
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\.base",
                                message=r"`BaseEstimator\._(validate_data|check_feature_names|check_n_features)` is deprecated")
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\.utils\.deprecation",
                                message="'force_all_finite' was renamed to 'ensure_all_finite'")
    lndb, features = load_data(args.lndb, "lndb", args.lndb_targets, paths, args.binary)
    lidc = lndb.head(0)
    if not args.lndb_only:
        lidc, lidc_features = load_data(args.lidc, "lidc", args.lidc_targets, paths, args.binary)
        if set(features) != set(lidc_features):
            raise ValueError("LIDC and LNDb must contain the same radiomics features.")
        features = lidc_features
    if args.original_only:
        features = [column for column in features if column.startswith("original_")]
        if not features:
            parser.error("No Original features found in the supplied CSVs.")
    class_weight = None if args.class_weight == "none" else args.class_weight
    label_offset = int(args.model == "xgboost" and not args.binary)
    if args.k is not None and not 1 <= args.k <= len(features):
        parser.error(f"--k must be between 1 and {len(features)}.")
    folds = sorted(lndb["fold"].unique())
    if len(folds) != 5:
        raise ValueError("Expected five saved LNDb folds.")
    name = args.lndb.stem if args.lndb_only else f"{args.lidc.stem}_{args.lndb.stem}"
    kind = "brf" if args.balanced_forest else args.model
    output = args.output or paths["outputs"]["models"] / f"{kind}_{name}{'_balanced' if class_weight else ''}{f'_k{args.k}' if args.k is not None else ''}{'_smote' if args.smote else ''}{'_original' if args.original_only else ''}{'_tuned' if args.tune_rf else ''}{'_binary' if args.binary else ''}"
    output.mkdir(parents=True, exist_ok=True)
    predictions = pd.Series(index=lndb.index, dtype=float)
    scores = []
    tuning, training_scores = [], []
    print(f"{kind}: {len(features)} features, class weights: {args.class_weight}, target: {target}", flush=True)
    for fold in folds:
        held_out = lndb["fold"].eq(fold)
        train = pd.concat([lidc, lndb.loc[~held_out]], ignore_index=True)
        model = make_model(class_weight, args.k, args.smote, args.balanced_forest, args.model, args.binary)
        print(f"Fold {fold}: {len(train)} training / {held_out.sum()} validation nodules", flush=True)
        if args.tune_rf:
            search = tune_rf(model, train, features, len(lidc), output, f"fold_{fold}", args.binary)
            model = search.best_estimator_
            tuning.append({"fold": fold, "inner_macro_f1": search.best_score_, **search.best_params_})
            training_prediction = model.predict(train[features])
            training_scores.append({"fold": fold, "accuracy": accuracy_score(train[target], training_prediction),
                                    **metrics(train[target], training_prediction, args.binary)})
            pd.DataFrame(tuning).to_csv(output / "tuning.csv", index=False)
            pd.DataFrame(training_scores).to_csv(output / "training_metrics.csv", index=False)
        else:
            fit_model(model, train[features], train[target], class_weight, label_offset)
        prediction = model.predict(lndb.loc[held_out, features]) + label_offset
        predictions.loc[held_out] = prediction
        scores.append({"fold": fold, **metrics(lndb.loc[held_out, target], prediction, args.binary)})
        print(pd.Series(scores[-1]).to_string(), flush=True)
    scores.append({"fold": "overall_oof", **metrics(lndb[target], predictions, args.binary)})
    pd.DataFrame(scores).to_csv(output / "metrics.csv", index=False)
    oof = lndb[["LNDbID", "FindingID", "fold", target]].copy()
    oof["prediction"] = predictions.astype(int)
    oof.to_csv(output / "validation_predictions.csv", index=False)
    confusion = pd.DataFrame(confusion_matrix(lndb[target], predictions, labels=labels), index=labels, columns=labels)
    confusion.to_csv(output / "confusion_matrix.csv", index_label="true_class")
    pd.DataFrame({"class": labels, "recall": recall_score(lndb[target], predictions, labels=labels, average=None, zero_division=0)}).to_csv(output / "recall_per_class.csv", index=False)
    print(f"Fitting final model on all {'LNDb' if args.lndb_only else 'LIDC + LNDb'} training nodules", flush=True)
    train = pd.concat([lidc, lndb], ignore_index=True)
    model = make_model(class_weight, args.k, args.smote, args.balanced_forest, args.model, args.binary)
    tuning_info = {}
    if args.tune_rf:
        # Choose final parameters using all five LNDb folds, never competition test data.
        search = tune_rf(model, train, features, len(lidc), output, "final", args.binary)
        model = search.best_estimator_
        tuning.append({"fold": "final", "inner_macro_f1": search.best_score_, **search.best_params_})
        pd.DataFrame(tuning).to_csv(output / "tuning.csv", index=False)
        tuning_info = {"param_grid": MIN_SAMPLES_GRID, "tuning_scoring": "macro_f1",
                       "best_params": search.best_params_, "inner_cv": "saved LNDb folds, LIDC training only"}
    else:
        fit_model(model, train[features], train[target], class_weight, label_offset)
    lidc_targets_path = args.lidc_targets or paths["lidc"]["targets"]
    lndb_targets_path = args.lndb_targets or paths["lndb"]["targets"]
    joblib.dump({"model": model, "features": features, "target": target,
                 "lidc_csv": None if args.lndb_only else str(args.lidc.resolve()),
                 "lndb_csv": str(args.lndb.resolve()), "class_weight": class_weight, "k": args.k,
                 "lidc_targets_csv": None if args.lndb_only else str(lidc_targets_path.resolve()),
                 "lndb_targets_csv": str(lndb_targets_path.resolve()),
                 "smote": args.smote, "balanced_forest": args.balanced_forest,
                 "model_kind": args.model, "label_offset": label_offset, "original_only": args.original_only,
                 **tuning_info}, output / "model.joblib")
    print(f"Saved metrics, validation predictions and model to {output}")


if __name__ == "__main__":
    main()
