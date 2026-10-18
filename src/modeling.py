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
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, make_scorer, mean_absolute_error, recall_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# Support both `python src/modeling.py` and importing `src.modeling`.
if __package__:
    from .auxiliary import ATTRIBUTES, WithAttributes, load_attributes
    from .radiomics_extraction import load_paths
else:
    from auxiliary import ATTRIBUTES, WithAttributes, load_attributes
    from radiomics_extraction import load_paths

ROOT = Path(__file__).resolve().parents[1]
TARGET = "target_classification_mean"
FEATURE_PATTERN = r"_(shape|shape2D|firstorder|glcm|glrlm|glszm|gldm|ngtdm)_"
# Radiomics families. A feature belongs to the family named in its middle token,
# for example original_shape_MeshVolume or wavelet-HL_gldm_DependenceNonUniformity.
FEATURE_FAMILIES = ("shape", "shape2D", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")
# Groups make the common splits easy to ask for on the command line.
FAMILY_GROUPS = {"texture": ("glcm", "glrlm", "glszm", "gldm", "ngtdm"),
                 "intensity": ("firstorder",),
                 "shape": ("shape", "shape2D")}
SLICE_IN_NODULE = "slice_in_nodule"
# Nodule volume in mm3, used by the size experiments. Segmentation masks provide it.
SIZE_COLUMN = "original_shape_MeshVolume"
LIDC_KEYS = ["patient_id", "series_uid", "nodule_id"]
LNDB_KEYS = ["LNDbID", "FindingID"]

# Equal raw SHA-256 and image headers in notebooks/radiomics_decisions_eda.ipynb.
# Retain the lower LNDbID in each confirmed pair.
EXCLUDED_LNDB_IDS = frozenset({181, 182, 183, 184, 185, 188, 189, 194, 195, 198, 226})
MIN_SAMPLES_GRID = {
    "randomforestclassifier__min_samples_leaf": [1, 5, 10, 20],
    "randomforestclassifier__min_samples_split": [2, 10, 20, 40],
}
BALANCED_FOREST_GRID = {
    "balancedrandomforestclassifier__max_depth": [None, 10],
    "balancedrandomforestclassifier__max_features": ["sqrt", 0.1, 0.3],
    "balancedrandomforestclassifier__min_samples_leaf": [1, 3, 5],
}
XGBOOST_STRUCTURE_GRID = {
    "xgbclassifier__max_depth": [2, 3, 4],
    "xgbclassifier__min_child_weight": [1, 5],
    "xgbclassifier__colsample_bytree": [0.6, 0.8, 1.0],
}
XGBOOST_ROUNDS_GRID = [
    {"xgbclassifier__learning_rate": [0.1], "xgbclassifier__n_estimators": [150]},
    {"xgbclassifier__learning_rate": [0.05], "xgbclassifier__n_estimators": [300]},
    {"xgbclassifier__learning_rate": [0.03], "xgbclassifier__n_estimators": [500]},
]
QUICK_FOREST_GRID = {
    "balancedrandomforestclassifier__max_features": ["sqrt", 0.1],
    "balancedrandomforestclassifier__min_samples_leaf": [1, 3],
}
QUICK_XGBOOST_GRID = {
    "xgbclassifier__max_depth": [2, 3],
    "xgbclassifier__min_child_weight": [1, 5],
}


def load_data(path, dataset, targets_path=None, paths=None, binary=False, multi_slice=False,
              exclude_duplicate_lndb=False):
    """Load requested nodules on saved folds, optionally excluding LNDb CT copies."""
    table = pd.read_csv(path)
    features = table.filter(regex=FEATURE_PATTERN).columns.tolist()
    keys = LIDC_KEYS if dataset == "lidc" else LNDB_KEYS
    slice_keys = keys + [SLICE_IN_NODULE] if multi_slice else keys
    locations = (paths or load_paths())[dataset]
    targets_file = targets_path or locations["targets"]
    targets = pd.read_csv(targets_file).rename(columns={"NoduleID": "FindingID"})
    if dataset == "lndb" and exclude_duplicate_lndb:
        table = table[~table["LNDbID"].isin(EXCLUDED_LNDB_IDS)].copy()
        targets = targets[~targets["LNDbID"].isin(EXCLUDED_LNDB_IDS)].copy()
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
    if multi_slice and (SLICE_IN_NODULE not in table.columns or table[SLICE_IN_NODULE].isna().any()):
        raise ValueError(f"{path}: multi-slice datasets need a non-missing {SLICE_IN_NODULE} column.")
    if table[keys].isna().any().any() or table.duplicated(slice_keys).any():
        raise ValueError(f"{path}: feature rows must have unique, non-missing"
                         + (" nodule and slice IDs." if multi_slice else " nodule IDs."))
    folds = pd.read_csv(locations["folds"])
    data = targets.merge(folds, on=keys[0], validate="many_to_one", how="left")
    data = data.merge(table[slice_keys + features], on=keys,
                      validate="one_to_many" if multi_slice else "one_to_one",
                      how="left", indicator=True)
    # Multi-slice tables repeat each nodule once per slice.
    nodules = table[keys].drop_duplicates().shape[0] if multi_slice else len(table)
    if not features or (targets_path is None and not binary and nodules != len(targets)) or data["_merge"].ne("both").any():
        raise ValueError(f"{path}: expected features for all {len(targets)} training nodules.")
    if data["fold"].isna().any() or not data[target].isin(labels).all():
        raise ValueError(f"{dataset}: missing folds or invalid target classes.")
    data[target] = data[target].astype(int)
    data[features] = data[features].replace([np.inf, -np.inf], np.nan)
    return data.drop(columns="_merge"), features

def feature_inventory(directory):
    """Summarise the saved training feature CSVs."""
    prefixes = (
        "lidc_circle_", "lidc_segmentation_", "lidc_sphere_",
        "lndb_circle_", "lndb_ge3_segmentation_", "lndb_sphere_",
    )
    rows = []

    for path in sorted(Path(directory).glob("*.csv")):
        if not path.name.startswith(prefixes):
            continue

        name = path.stem
        dataset = "LIDC" if name.startswith("lidc_") else "LNDb"
        keys = LIDC_KEYS if dataset == "LIDC" else LNDB_KEYS

        header = pd.read_csv(path, nrows=0)
        identifiers = pd.read_csv(path, usecols=keys)

        region = (
            "sphere" if "_sphere_" in name
            else "segmentation" if "_segmentation_" in name
            else "circle"
        )
        method = (
            "2.5D" if "_5slice_" in name
            else "3D" if region == "sphere" or "_3d_" in name
            else "2D"
        )

        rows.append({
            "dataset": dataset,
            "region": region,
            "method": method,
            "CSV": path.name,
            "nodules": len(identifiers.drop_duplicates()),
            "rows": len(identifiers),
            "features": len(header.filter(regex=FEATURE_PATTERN).columns),
        })

    return pd.DataFrame(rows)

def attach_size(data, dataset, paths):
    """Add nodule size from the shape table when the features carry no shape.

    Circle and sphere features describe a fixed region, so they have no shape
    columns. The size experiments still need a size, and the masks give one for
    the same nodules.
    """
    if SIZE_COLUMN in data.columns:
        return data
    keys = LIDC_KEYS if dataset == "lidc" else LNDB_KEYS
    table = paths["shape"][dataset]
    shape = pd.read_csv(table, usecols=keys + [SIZE_COLUMN])
    if shape[keys].duplicated().any():
        raise ValueError(f"{table}: repeated nodule IDs.")
    merged = data.merge(shape, on=keys, validate="many_to_one", how="left")
    if merged[SIZE_COLUMN].isna().any():
        raise ValueError(f"{table} does not cover every {dataset} nodule in this run.")
    return merged


def prefer_normalized_pairs(features):
    raw = {name.removesuffix("Normalized") for name in features if name.endswith("Normalized")}
    if not raw.intersection(features):
        raise ValueError("No raw/Normalized radiomics feature pairs found.")
    return [name for name in features if name not in raw]


def feature_family(name):
    """Radiomics family a feature belongs to, read from its name."""
    for token in name.split("_"):
        if token in FEATURE_FAMILIES:
            return token
    raise ValueError(f"{name} does not look like a radiomics feature.")


def select_families(features, requested):
    """Keep the requested radiomics families, accepting group names like texture."""
    wanted = set()
    for group in requested:
        group = group.strip()
        if group in FAMILY_GROUPS:
            wanted.update(FAMILY_GROUPS[group])
        elif group in FEATURE_FAMILIES:
            wanted.add(group)
        else:
            raise ValueError(f"Unknown feature family {group!r}. Use any of "
                             f"{sorted(FEATURE_FAMILIES + tuple(FAMILY_GROUPS))}.")
    return [name for name in features if feature_family(name) in wanted]


def size_confounded_features(features, frame, maximum, column=SIZE_COLUMN):
    """Features whose Spearman correlation with nodule size exceeds the limit."""
    if column not in frame.columns:
        raise ValueError(f"{column} is missing. Size experiments need segmentation features.")
    correlations = frame[features].corrwith(frame[column], method="spearman").abs()
    return [name for name in features if correlations[name] > maximum]


def aggregate_slices(rows, values, keys):
    """Weighted mean of per-slice values, one row per nodule. Centre slices weigh most.

    Pass one value per slice, or a table with one column per class to average
    probabilities. Returns a Series for single values and a table otherwise.
    """
    single = not isinstance(values, pd.DataFrame)
    frame = pd.DataFrame(np.asarray(values, dtype=float), index=rows.index)
    frame.columns = ["value"] if single else list(values.columns)
    values_columns = list(frame.columns)
    frame[keys] = rows[keys].to_numpy()
    frame[SLICE_IN_NODULE] = rows[SLICE_IN_NODULE].to_numpy()
    frame = frame.sort_values(keys + [SLICE_IN_NODULE], kind="stable")
    grouped = frame.groupby(list(keys), sort=False)
    position = grouped.cumcount().to_numpy()
    size = grouped[SLICE_IN_NODULE].transform("size").to_numpy()
    # Triangular weights 1, 2, ..., centre, ..., 2, 1 over the nodule's slices.
    frame["weight"] = np.minimum(position + 1, size - position)
    for column in values_columns:
        frame[column] = frame[column] * frame["weight"]
    totals = frame.groupby(list(keys), sort=False)[values_columns + ["weight"]].sum()
    result = totals[values_columns].div(totals["weight"], axis=0)
    return result["value"] if single else result


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


def tune_balanced_model(model, train, features, n_lidc, output, fold, class_weight, label_offset,
                        balanced_forest=False, structure_grid=None, rounds_grid=None, quick=False):
    """Select parameters on inner LNDb folds, then refit on the outer training rows."""
    if structure_grid is None:
        structure_grid = (QUICK_FOREST_GRID if balanced_forest else QUICK_XGBOOST_GRID) if quick else (
            BALANCED_FOREST_GRID if balanced_forest else XGBOOST_STRUCTURE_GRID)
    splits = list(lndb_cv(train, n_lidc).split())
    if quick:
        splits = [max(splits, key=lambda split: train.iloc[split[1]][TARGET].eq(5).sum())]
        print(f"Quick tuning on inner LNDb fold {train.iloc[splits[0][1]]['fold'].iloc[0]}", flush=True)

    def search(grid, fixed, stage):
        rows = []
        for params in ParameterGrid(grid):
            settings = {**fixed, **params}
            scores = []
            for inner_train, inner_valid in splits:
                candidate = clone(model).set_params(**settings)
                fit_model(candidate, train.iloc[inner_train][features], train.iloc[inner_train][TARGET],
                          class_weight, label_offset)
                prediction = candidate.predict(train.iloc[inner_valid][features]) + label_offset
                scores.append(balanced_accuracy_score(train.iloc[inner_valid][TARGET], prediction))
            rows.append({"stage": stage, **settings, **{f"split{i}_score": score for i, score in enumerate(scores)},
                         "mean_inner_balanced_accuracy": np.mean(scores)})
            print(f"{stage}: {settings} -> {np.mean(scores):.4f}", flush=True)
        return rows, max(rows, key=lambda row: row["mean_inner_balanced_accuracy"])

    rows, best = search(structure_grid, {}, "structure")
    if not balanced_forest and not quick:
        later, best = search(rounds_grid or XGBOOST_ROUNDS_GRID,
                             {key: best[key] for key in ParameterGrid(structure_grid)[0]}, "rounds")
        rows.extend(later)
    pd.DataFrame(rows).to_csv(output / f"grid_{fold}.csv", index=False)
    params = {key: best[key] for key in model.get_params() if key in best}
    fitted = clone(model).set_params(**params)
    fit_model(fitted, train[features], train[TARGET], class_weight, label_offset)
    print(f"Best inner balanced accuracy: {best['mean_inner_balanced_accuracy']:.4f}, parameters: {params}", flush=True)
    return fitted, best["mean_inner_balanced_accuracy"], params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=ROOT / "configs/paths.yaml", help="YAML with file locations.")
    parser.add_argument("--lidc", type=Path, help="Override the YAML LIDC feature CSV.")
    parser.add_argument("--lndb", type=Path, help="Override the YAML LNDb feature CSV.")
    parser.add_argument("--lidc-targets", type=Path, help="Explicit LIDC nodule population. Default: all cleaned LIDC targets.")
    parser.add_argument("--lndb-targets", type=Path, help="Explicit LNDb nodule population, e.g. the saved ge3 targets.")
    parser.add_argument("--lndb-only", action="store_true", help="Train using only the LNDb training folds.")
    parser.add_argument("--exclude-duplicate-lndb", action="store_true",
                        help="Exclude confirmed duplicate LNDb CT scans. Default: use all scans.")
    parser.add_argument("--binary", action="store_true", help="Use saved target_binary_mean: mean < 3 is 0, mean > 3 is 1, mean = 3 is excluded.")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--model", choices=["rf", "logistic", "xgboost"], default="rf")
    parser.add_argument("--original-only", action="store_true", help="Use only features from the unfiltered Original image.")
    parser.add_argument("--family", help="Keep only these radiomics families, comma separated. "
                                         "Use firstorder, glcm, glrlm, glszm, gldm, ngtdm, shape, or the groups "
                                         "texture (all five texture families) and intensity (firstorder).")
    parser.add_argument("--normalized-pairs", action="store_true", help="Keep normalized radiomics features instead of their raw counterparts.")
    parser.add_argument("--size-only", action="store_true",
                        help=f"Use only {SIZE_COLUMN}, the volume baseline for size experiments.")
    parser.add_argument("--max-size-correlation", type=float,
                        help="Drop features whose Spearman correlation with nodule size exceeds this value. "
                             "Correlations are measured on training rows only.")
    parser.add_argument("--auxiliary-attributes", action="store_true",
                        help="Add the radiologists' own attributes (lobulation, spiculation, sphericity, "
                             "margin, texture) as extra features, predicted from the radiomics inside each "
                             "fold. They exist only for training nodules, never for a test nodule. Requires "
                             "--model rf without tuning or --multi-slice.")
    parser.add_argument("--smote", action="store_true", help="Scale features and balance classes with SMOTE on training data only.")
    parser.add_argument("--balanced-forest", action="store_true", help="Balance each tree's training sample by undersampling.")
    parser.add_argument("--multi-slice", action="store_true",
                        help="Accept several rows per nodule, one per slice numbered by "
                             "slice_in_nodule, and predict each nodule from the weighted "
                             "mean of its slice predictions (centre slice weighs most).")
    parser.add_argument("--k", type=int, help="Select the K best features inside each training fold. Default: use all.")
    parser.add_argument("--tune-rf", action="store_true", help="Tune RF minimum leaf and split sizes inside the saved training folds, selecting by macro F1.")
    parser.add_argument("--tune", action="store_true", help="Tune Balanced RF or weighted XGBoost on inner LNDb folds by balanced accuracy.")
    parser.add_argument("--quick-tune", action="store_true", help="Grid-search four Balanced RF or weighted XGBoost settings on one inner LNDb fold per outer fold.")
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
    do_tune = args.tune or args.quick_tune
    if do_tune and (args.tune and args.quick_tune or args.tune_rf or args.binary or args.multi_slice or args.smote or
                   not (args.balanced_forest or args.model == "xgboost" and args.class_weight == "balanced")):
        parser.error("Tuning requires five-class Balanced RF or XGBoost with --class-weight balanced, without SMOTE or multi-slice. Choose one tuning flag.")
    if args.auxiliary_attributes and (args.model != "rf" or do_tune or args.tune_rf or args.multi_slice):
        parser.error("--auxiliary-attributes requires --model rf without tuning or --multi-slice.")
    if args.smote or args.balanced_forest:
        # imbalanced-learn 0.12 uses these older APIs, which still work in sklearn 1.6.
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\.base",
                                message=r"`BaseEstimator\._(validate_data|check_feature_names|check_n_features)` is deprecated")
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\.utils\.deprecation",
                                message="'force_all_finite' was renamed to 'ensure_all_finite'")
    lndb, features = load_data(args.lndb, "lndb", targets_path=args.lndb_targets, paths=paths,
                               binary=args.binary, multi_slice=args.multi_slice,
                               exclude_duplicate_lndb=args.exclude_duplicate_lndb)
    lidc = lndb.head(0)
    if not args.lndb_only:
        lidc, lidc_features = load_data(args.lidc, "lidc", targets_path=args.lidc_targets, paths=paths,
                                       binary=args.binary, multi_slice=args.multi_slice)
        if set(features) != set(lidc_features):
            raise ValueError("LIDC and LNDb must contain the same radiomics features.")
        features = lidc_features
    if args.auxiliary_attributes:
        # The attributes are targets of their own, so they never enter `features`.
        lndb = load_attributes(lndb, "lndb", paths)
        if not args.lndb_only:
            lidc = load_attributes(lidc, "lidc", paths)
        print(f"Auxiliary attributes: {', '.join(ATTRIBUTES)}", flush=True)
    if args.size_only or args.max_size_correlation is not None:
        # Circle and sphere features have no shape columns, so take the size from the masks.
        if SIZE_COLUMN not in lndb.columns:
            lndb = attach_size(lndb, "lndb", paths)
            print(f"Nodule size taken from {paths['shape']['lndb']}", flush=True)
        if not args.lndb_only and SIZE_COLUMN not in lidc.columns:
            lidc = attach_size(lidc, "lidc", paths)
            print(f"Nodule size taken from {paths['shape']['lidc']}", flush=True)
    if args.original_only:
        features = [column for column in features if column.startswith("original_")]
        if not features:
            parser.error("No Original features found in the supplied CSVs.")
    if args.family:
        try:
            features = select_families(features, args.family.split(","))
        except ValueError as error:
            parser.error(str(error))
        if not features:
            parser.error(f"No features left in the requested families: {args.family}")
    if args.normalized_pairs:
        features = prefer_normalized_pairs(features)
    if args.size_only:
        if SIZE_COLUMN not in lndb.columns:
            parser.error(f"--size-only needs {SIZE_COLUMN} in the feature CSV or in the shape table.")
        features = [SIZE_COLUMN]
    class_weight = None if args.class_weight == "none" else args.class_weight
    label_offset = int(args.model == "xgboost" and not args.binary)
    if args.k is not None and not 1 <= args.k <= len(features):
        parser.error(f"--k must be between 1 and {len(features)}.")
    folds = sorted(lndb["fold"].unique())
    if len(folds) != 5:
        raise ValueError("Expected five saved LNDb folds.")
    name = args.lndb.stem if args.lndb_only else f"{args.lidc.stem}_{args.lndb.stem}"
    kind = "brf" if args.balanced_forest else args.model
    output = args.output or paths["outputs"]["models"] / f"{kind}_{name}{'_balanced' if class_weight else ''}{f'_k{args.k}' if args.k is not None else ''}{'_smote' if args.smote else ''}{'_original' if args.original_only else ''}{'_aux' if args.auxiliary_attributes else ''}{'_family-' + args.family.replace(',', '+').replace(' ', '') if args.family else ''}{'_normalized_pairs' if args.normalized_pairs else ''}{'_sizeonly' if args.size_only else ''}{f'_maxsize{args.max_size_correlation}' if args.max_size_correlation is not None else ''}{'_quick_tuned' if args.quick_tune else '_tuned' if args.tune_rf or args.tune else ''}{'_binary' if args.binary else ''}{'_multislice' if args.multi_slice else ''}{'_dedup' if args.exclude_duplicate_lndb else ''}"
    if not (do_tune or args.tune_rf) and any(word in output.name.lower() for word in ("grid", "tuned")):
        parser.error("Output name suggests tuning, but no tuning flag was supplied.")
    if (do_tune or args.tune_rf) and output.exists() and any(output.iterdir()):
        parser.error(f"Tuned output already contains files: {output}. Choose a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    # Validation stays at nodule level: one row per nodule, its first slice.
    lndb_keys = ["LNDbID", "FindingID"]
    nodules = lndb.drop_duplicates(lndb_keys) if args.multi_slice else lndb
    predictions = pd.Series(index=nodules.index, dtype=float)
    probability_names = [f"proba_{label}" for label in labels]
    probabilities = pd.DataFrame(index=nodules.index, columns=probability_names, dtype=float)
    scores = []
    tuning, training_scores = [], []

    def new_model():
        base = make_model(class_weight, args.k, args.smote, args.balanced_forest, args.model, args.binary)
        return WithAttributes(base) if args.auxiliary_attributes else base

    def fit_on(model, frame, feature_columns):
        if args.auxiliary_attributes:
            # The attributes are extra inputs the classifier never sees at test.
            model.fit(frame[feature_columns], frame[target], frame[ATTRIBUTES])
        else:
            fit_model(model, frame[feature_columns], frame[target], class_weight, label_offset)
        return model

    print(f"{kind}: {len(features)} features, class weights: {args.class_weight}, target: {target}", flush=True)
    for fold in folds:
        held_out = lndb["fold"].eq(fold)
        held_out_nodules = nodules["fold"].eq(fold)
        train = pd.concat([lidc, lndb.loc[~held_out]], ignore_index=True)
        model = new_model()
        print(f"Fold {fold}: {len(train)} training / {held_out_nodules.sum()} validation nodules", flush=True)
        fold_features = features
        if args.max_size_correlation is not None:
            dropped = size_confounded_features(features, train, args.max_size_correlation)
            fold_features = [column for column in features if column not in dropped]
            print(f"Fold {fold}: dropped {len(dropped)} of {len(features)} features correlated with size"
                  f" above {args.max_size_correlation}", flush=True)
        if args.tune_rf:
            search = tune_rf(model, train, fold_features, len(lidc), output, f"fold_{fold}", args.binary)
            model = search.best_estimator_
            tuning.append({"fold": fold, "inner_macro_f1": search.best_score_, **search.best_params_})
            training_prediction = model.predict(train[fold_features])
            training_scores.append({"fold": fold, "accuracy": accuracy_score(train[target], training_prediction),
                                    **metrics(train[target], training_prediction, args.binary)})
            pd.DataFrame(tuning).to_csv(output / "tuning.csv", index=False)
            pd.DataFrame(training_scores).to_csv(output / "training_metrics.csv", index=False)
        elif do_tune:
            model, score, params = tune_balanced_model(model, train, fold_features, len(lidc), output,
                                                       f"fold_{fold}", class_weight, label_offset,
                                                       args.balanced_forest, quick=args.quick_tune)
            tuning.append({"fold": fold, "inner_balanced_accuracy": score, **params})
            pd.DataFrame(tuning).to_csv(output / "tuning.csv", index=False)
        else:
            fit_on(model, train, fold_features)
        held_out_rows = lndb.loc[held_out, fold_features]
        slice_prediction = model.predict(held_out_rows) + label_offset
        slice_probabilities = None
        if hasattr(model, "predict_proba"):
            values = model.predict_proba(held_out_rows)
            names = [f"proba_{int(label) + label_offset}" for label in model.classes_]
            slice_probabilities = pd.DataFrame(values, index=held_out_rows.index, columns=names)[probability_names]
        if args.multi_slice:
            # Train on every slice, then combine the slice predictions per nodule.
            prediction = aggregate_slices(lndb.loc[held_out], slice_prediction, lndb_keys)
            index = pd.MultiIndex.from_frame(nodules.loc[held_out_nodules, lndb_keys])
            prediction = np.rint(prediction.reindex(index)).clip(min(labels), max(labels)).astype(int)
            predictions.loc[held_out_nodules] = prediction.to_numpy()
            if slice_probabilities is not None:
                aggregated = aggregate_slices(lndb.loc[held_out], slice_probabilities, lndb_keys)
                probabilities.loc[held_out_nodules, probability_names] = aggregated.reindex(index).to_numpy()
            scores.append({"fold": fold, **metrics(nodules.loc[held_out_nodules, target], prediction, args.binary)})
        else:
            predictions.loc[held_out] = slice_prediction
            if slice_probabilities is not None:
                probabilities.loc[held_out, probability_names] = slice_probabilities.to_numpy()
            scores.append({"fold": fold, **metrics(lndb.loc[held_out, target], slice_prediction, args.binary)})
        print(pd.Series(scores[-1]).to_string(), flush=True)
    scores.append({"fold": "overall_oof", **metrics(nodules[target], predictions, args.binary)})
    pd.DataFrame(scores).to_csv(output / "metrics.csv", index=False)
    oof = nodules[["LNDbID", "FindingID", "fold", target]].copy()
    oof["prediction"] = predictions.astype(int)
    if not probabilities.isna().to_numpy().any():
        # Saved so several models can be combined on their shared nodules.
        if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6):
            raise ValueError("Class probabilities must sum to one.")
        for name in probability_names:
            oof[name] = probabilities[name].to_numpy()
    oof.to_csv(output / "validation_predictions.csv", index=False)
    confusion = pd.DataFrame(confusion_matrix(nodules[target], predictions, labels=labels), index=labels, columns=labels)
    confusion.to_csv(output / "confusion_matrix.csv", index_label="true_class")
    pd.DataFrame({"class": labels, "recall": recall_score(nodules[target], predictions, labels=labels, average=None, zero_division=0)}).to_csv(output / "recall_per_class.csv", index=False)
    print(f"Fitting final model on all {'LNDb' if args.lndb_only else 'LIDC + LNDb'} training"
          f" {'slices' if args.multi_slice else 'nodules'}", flush=True)
    train = pd.concat([lidc, lndb], ignore_index=True)
    model = new_model()
    final_features = features
    if args.max_size_correlation is not None:
        dropped = size_confounded_features(features, train, args.max_size_correlation)
        final_features = [column for column in features if column not in dropped]
        print(f"Final model: dropped {len(dropped)} of {len(features)} features correlated with size"
              f" above {args.max_size_correlation}", flush=True)
    tuning_info = {}
    if args.tune_rf:
        # Choose final parameters using all five LNDb folds, never competition test data.
        search = tune_rf(model, train, final_features, len(lidc), output, "final", args.binary)
        model = search.best_estimator_
        tuning.append({"fold": "final", "inner_macro_f1": search.best_score_, **search.best_params_})
        pd.DataFrame(tuning).to_csv(output / "tuning.csv", index=False)
        tuning_info = {"param_grid": MIN_SAMPLES_GRID, "tuning_scoring": "macro_f1",
                        "best_params": search.best_params_, "inner_cv": "saved LNDb folds, LIDC training only"}
    elif do_tune:
        model, score, params = tune_balanced_model(model, train, final_features, len(lidc), output, "final",
                                                   class_weight, label_offset, args.balanced_forest,
                                                   quick=args.quick_tune)
        tuning.append({"fold": "final", "inner_balanced_accuracy": score, **params})
        pd.DataFrame(tuning).to_csv(output / "tuning.csv", index=False)
        grid = ((QUICK_FOREST_GRID if args.balanced_forest else QUICK_XGBOOST_GRID) if args.quick_tune else
                BALANCED_FOREST_GRID if args.balanced_forest else
                {"structure": XGBOOST_STRUCTURE_GRID, "rounds": XGBOOST_ROUNDS_GRID})
        tuning_info = {"param_grid": grid,
                       "tuning_scoring": "balanced_accuracy", "best_params": params,
                       "inner_cv": ("one saved LNDb fold with most class-5 nodules, LIDC training only" if args.quick_tune
                                    else "saved LNDb folds, LIDC training only")}
    else:
        fit_on(model, train, final_features)
    lidc_targets_path = args.lidc_targets or paths["lidc"]["targets"]
    lndb_targets_path = args.lndb_targets or paths["lndb"]["targets"]
    joblib.dump({"model": model, "features": final_features, "target": target,
                 "lidc_csv": None if args.lndb_only else str(args.lidc.resolve()),
                 "lndb_csv": str(args.lndb.resolve()), "class_weight": class_weight, "k": args.k,
                 "lidc_targets_csv": None if args.lndb_only else str(lidc_targets_path.resolve()),
                 "lndb_targets_csv": str(lndb_targets_path.resolve()),
                 "smote": args.smote, "balanced_forest": args.balanced_forest,
                  "model_kind": args.model, "label_offset": label_offset, "original_only": args.original_only,
                  "family": args.family,
                  "auxiliary_attributes": args.auxiliary_attributes,
                  "normalized_pairs": args.normalized_pairs,
                  "multi_slice": args.multi_slice,
                  "size_only": args.size_only,
                  "max_size_correlation": args.max_size_correlation,
                  "excluded_lndb_ids": sorted(EXCLUDED_LNDB_IDS) if args.exclude_duplicate_lndb else [],
                 **tuning_info}, output / "model.joblib")
    print(f"Saved metrics, validation predictions and model to {output}")


if __name__ == "__main__":
    main()
