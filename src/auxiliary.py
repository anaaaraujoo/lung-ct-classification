"""Give a classifier the radiologists' own attributes as extra features.

Lobulation, spiculation, sphericity, margin and texture are radiologist ratings.
They exist for the LIDC and LNDb training nodules, and never for a nodule the
model has to score. A forest predicts each one from the radiomics instead, and
those predictions join the classifier's features.

The attribute predictions the classifier trains on come from inner folds, so a
training row never feeds the attribute model that describes it. Without that,
the classifier would learn on cleaner attributes than the ones it meets later.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline

ATTRIBUTES = ["lobulation", "spiculation", "sphericity", "margin", "texture"]
CONCEPTS = ATTRIBUTES + ["calcification", "subtlety"]
LIDC_KEYS = ["patient_id", "series_uid", "nodule_id"]
LNDB_KEYS = ["LNDbID", "FindingID"]
# The two annotation tables use different spellings for the same five ratings.
LNDB_NAMES = {"Lobulation": "lobulation", "Spiculation": "spiculation", "Sphericity": "sphericity",
              "Margin": "margin", "Texture": "texture"}


def load_attributes(data, dataset, paths):
    """Add the mean attribute rating of the readers to a training table.

    Readers rate each nodule separately, so the table holds the mean over them,
    matching how the malignancy target is built.
    """
    if dataset == "lidc":
        source = paths["lidc"]["annotations"]
        keys = LIDC_KEYS
        table = pd.read_csv(source)[LIDC_KEYS + ATTRIBUTES]
    else:
        source = paths["lndb"]["findings"]
        keys = LNDB_KEYS
        table = pd.read_csv(Path(source).expanduser())[LNDB_KEYS + list(LNDB_NAMES)]
        table = table.rename(columns=LNDB_NAMES)
    ratings = table.groupby(keys, as_index=False)[ATTRIBUTES].mean()
    merged = data.merge(ratings, on=keys, validate="one_to_one", how="left")
    if merged[ATTRIBUTES].isna().all(axis=1).any():
        raise ValueError(f"{source} has no attributes for some {dataset} nodules.")
    return merged


def load_concepts(data, dataset, paths):
    """Load shared reader concepts, keeping calcification as presence, not a scale.

    Calcification codes 1–5 describe types and 6 means none. LNDb also contains
    averaged codes, which cannot identify a type. A value below 6 establishes
    that at least one reader reported calcification. This same presence target
    is used for LIDC. Invalid zero ratings and missing ratings remain unknown.
    """
    if dataset not in {"lidc", "lndb"}:
        raise ValueError("Dataset must be lidc or lndb.")
    keys = LIDC_KEYS if dataset == "lidc" else LNDB_KEYS
    source = paths[dataset]["annotations" if dataset == "lidc" else "findings"]
    names = {name if dataset == "lidc" else name.capitalize(): name for name in CONCEPTS}
    table = pd.read_csv(Path(source).expanduser(), usecols=keys + list(names)).rename(columns=names)
    for name in CONCEPTS:
        values = pd.to_numeric(table[name], errors="coerce")
        maximum = 6 if name == "calcification" else 5
        table[name] = values.where(values.between(1, maximum))
    known = table.calcification.notna()
    table.loc[known, "calcification"] = table.loc[known, "calcification"].lt(6).astype(float)
    aggregation = {name: "max" if name == "calcification" else "mean" for name in CONCEPTS}
    ratings = table.groupby(keys, as_index=False).agg(aggregation)
    merged = data.merge(ratings, on=keys, validate="one_to_one", how="left")
    if merged[CONCEPTS].isna().all(axis=1).any():
        raise ValueError(f"{source} has no concepts for some {dataset} nodules.")
    return merged


def default_regressor():
    """Predicts one radiologist rating from the radiomics features.

    max_features="sqrt" keeps a fit to seconds: the regression default reads every
    one of the 2060 features at every split.
    """
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         RandomForestRegressor(n_estimators=300, max_features="sqrt",
                                               random_state=42, n_jobs=-1))


class WithAttributes(BaseEstimator, ClassifierMixin):
    """A classifier that also sees the predicted radiologist attributes."""

    def __init__(self, classifier, regressor=None, inner_splits=4):
        self.classifier = classifier
        self.regressor = regressor
        self.inner_splits = inner_splits

    def _new_regressor(self):
        return clone(self.regressor) if self.regressor is not None else default_regressor()

    def fit(self, X, y, attributes):
        values = np.asarray(X, dtype=float)
        ratings = attributes[ATTRIBUTES].to_numpy(dtype=float)
        out_of_fold = np.full(ratings.shape, np.nan, dtype=float)
        splitter = KFold(self.inner_splits, shuffle=True, random_state=42)
        for train_index, test_index in splitter.split(values):
            for column in range(len(ATTRIBUTES)):
                known = train_index[~np.isnan(ratings[train_index, column])]
                if len(known) == 0:
                    continue
                model = self._new_regressor().fit(values[known], ratings[known, column])
                out_of_fold[test_index, column] = model.predict(values[test_index])
        if np.isnan(out_of_fold).any():
            raise ValueError("Some rows have no predicted attributes. Check the attribute coverage.")
        # How well the radiomics recover each rating, from predictions the rated
        # nodule never trained on. A floor for what the classifier can use.
        self.attribute_scores_ = {}
        for column, name in enumerate(ATTRIBUTES):
            known = ~np.isnan(ratings[:, column])
            self.attribute_scores_[name] = float(spearmanr(out_of_fold[known, column], ratings[known, column]).statistic)
        print("Attribute recovery, out-of-fold Spearman: "
              + ", ".join(f"{name} {value:.2f}" for name, value in self.attribute_scores_.items()), flush=True)
        self.attribute_models_ = []
        for column in range(len(ATTRIBUTES)):
            known = ~np.isnan(ratings[:, column])
            self.attribute_models_.append(self._new_regressor().fit(values[known], ratings[known, column]))
        self.classifier_ = clone(self.classifier).fit(np.column_stack([values, out_of_fold]), y)
        self.classes_ = self.classifier_.classes_
        return self

    def _augment(self, X):
        values = np.asarray(X, dtype=float)
        predicted = np.column_stack([model.predict(values) for model in self.attribute_models_])
        return np.column_stack([values, predicted])

    def predict(self, X):
        return self.classifier_.predict(self._augment(X))

    def predict_proba(self, X):
        return self.classifier_.predict_proba(self._augment(X))
