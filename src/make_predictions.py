"""Fill the competition submission using a saved classifier or rounded regressor."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["LNDbID", "FindingID"]


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
    table = pd.read_csv(args.features)
    if not set(features).issubset(table.columns):
        raise ValueError("The test features do not match the saved model.")

    X = table[features].replace([np.inf, -np.inf], np.nan)
    predictions = model.predict(X) + saved.get("label_offset", 0)
    if saved.get("target") == "target_regression_mean":
        if __package__:
            from .regression import to_classes
        else:
            from regression import to_classes
        predictions = to_classes(predictions)
    predicted = table[KEYS].copy()
    predicted["prediction"] = predictions.astype(int)

    submission = pd.read_csv(args.submission).drop(columns=["prediction"])
    result = submission.merge(predicted, on=KEYS, how="left")
    fallback = int(pd.Series(predictions).mode().iloc[0]) if len(predictions) else 2
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
