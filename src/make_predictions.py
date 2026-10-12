"""Fill the competition submission using a saved classifier or rounded regressor."""
import argparse
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["LNDbID", "FindingID"]
# Scripts in src/ run with src/ on the path, not the repository root.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.modeling import SLICE_IN_NODULE, aggregate_slices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Saved model.joblib from src/modeling.py or src/regression.py.")
    parser.add_argument("--features", type=Path, default=ROOT / "data/lndb_test_circle_r11_hu.csv")
    parser.add_argument("--submission", type=Path, default=ROOT / "predictions.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "predictions.csv")
    args = parser.parse_args()

    saved = joblib.load(args.model)
    if saved.get("target") == "target_binary_mean":
        raise ValueError("Competition submissions require a five-class model.")
    model = saved["model"]
    features = saved["features"]
    multi_slice = saved.get("multi_slice", False)
    table = pd.read_csv(args.features)
    if not set(features).issubset(table.columns):
        raise ValueError("The test features do not match the saved model.")
    if multi_slice:
        if SLICE_IN_NODULE not in table.columns or table[SLICE_IN_NODULE].isna().any():
            raise ValueError(f"The test features need a non-missing {SLICE_IN_NODULE} column.")
        if table.duplicated(KEYS + [SLICE_IN_NODULE]).any():
            raise ValueError("Each nodule slice must appear once.")
    elif table.duplicated(KEYS).any():
        raise ValueError("The test features repeat nodules, but this model was trained on single-slice rows.")

    X = table[features].replace([np.inf, -np.inf], np.nan)
    slice_predictions = model.predict(X) + saved.get("label_offset", 0)
    if multi_slice:
        # Combine the slice predictions per nodule, centre slice weighing most.
        aggregated = aggregate_slices(table, slice_predictions, KEYS)
        predicted = table[KEYS].drop_duplicates()
        index = pd.MultiIndex.from_frame(predicted)
        predictions = aggregated.reindex(index).to_numpy()
    else:
        predicted = table[KEYS].copy()
        predictions = slice_predictions
    if saved.get("target") == "target_regression_mean":
        from src.regression import to_classes
        predictions = to_classes(predictions)
    elif multi_slice:
        predictions = np.rint(predictions).clip(1, 5)
    predicted["prediction"] = predictions.astype(int)

    submission = pd.read_csv(args.submission).drop(columns=["prediction"])
    result = submission.merge(predicted, on=KEYS, how="left")
    fallback = int(predicted["prediction"].mode().iloc[0]) if len(predicted) else 2
    missing = result["prediction"].isna().sum()
    result["prediction"] = result["prediction"].fillna(fallback).astype(int)
    if not result["prediction"].between(1, 5).all():
        raise ValueError("Predictions must be integer classes from 1 to 5.")

    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} predictions to {args.output}")
    print(f"Model predictions: {len(result) - missing}. Fallback class {fallback}: {missing}.")
    if missing:
        print("Fallback rows have no extracted features. Check the extraction errors before submitting.")


if __name__ == "__main__":
    main()
