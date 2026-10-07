"""Prepare reader-level LIDC and LNDb malignancy ratings."""

import numpy as np
import pandas as pd

def prepare_lndb_annotations(characteristics, all_nodules):
    # Remove findings that exist only in medical reports.
    nodules = all_nodules[
        ["LNDbID", "FindingID", "RadID", "RadFinding", "Nodule", "DiamEq_Rad"]
    ].dropna(subset=["RadID"]).copy()

    # allNods groups annotations of the same nodule into one row.
    # FindingID identifies that nodule, while RadFinding lists the ID used by each radiologist.
    nodules = nodules.rename(columns={"FindingID": "NoduleID", "RadFinding": "FindingID"})

    # Create one row per radiologist.
    reader_columns = ["RadID", "FindingID", "Nodule"]
    for column in reader_columns:
        nodules[column] = nodules[column].astype(str).str.split(",")

    nodules = nodules.explode(reader_columns, ignore_index=True)
    nodules[reader_columns] = nodules[reader_columns].astype(float)
    # Keep readers who considered the finding a nodule.
    nodules = nodules[nodules["Nodule"] == 1]

    # Add the individual malignancy ratings.
    ratings = characteristics[["LNDbID", "RadID", "FindingID", "malignancy"]]
    annotations = nodules.merge(ratings, on=["LNDbID", "RadID", "FindingID"])

    # Remove nodule annotations without a malignancy rating.
    annotations = annotations[annotations["malignancy"] != 0]

    return annotations[["LNDbID", "NoduleID", "RadID", "malignancy", "DiamEq_Rad"]].reset_index(drop=True)

# if there's a tie then return nan
def _find_unique_mode(ratings):
    modes = ratings.mode()
    if (len(modes)) == 1:
        return modes.iloc[0]
    return np.nan

def aggregate_ratings(annotations, group_columns):
    return annotations.groupby(group_columns, as_index=False).agg(
        ratings=("malignancy", list),
        malignancy_mean=("malignancy", "mean"),
        malignancy_median=("malignancy", "median"),
        malignancy_mode=("malignancy", _find_unique_mode),
    )

def create_targets(nodules):
    targets = nodules.copy()

    # Continuous regression targets.
    targets["target_regression_mean"] = targets["malignancy_mean"]
    targets["target_regression_median"] = targets["malignancy_median"]

    # Integer classes from 1 to 5. Half-up rounding makes 2.5 become 3.
    targets["target_classification_mean"] = np.floor(
        targets["malignancy_mean"] + 0.5
    ).astype(int)

    # Binary target from the unrounded mean. A mean of 3 remains missing.
    targets["target_binary_mean"] = pd.NA
    targets.loc[targets["malignancy_mean"] < 3, "target_binary_mean"] = 0
    targets.loc[targets["malignancy_mean"] > 3, "target_binary_mean"] = 1
    targets["target_binary_mean"] = targets["target_binary_mean"].astype("Int64") #Int64 to support nNA values

    return targets
