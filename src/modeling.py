from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


CSV_PATH = "data/radiomics_causey_2d.csv"
LABEL_AGGREGATION = "mean"  # "mean" or "median"
RANDOM_STATE = 42

MODEL_DIRECTORY = Path("models")
BINARY_MODEL_PATH = MODEL_DIRECTORY / "radiomics_2d_binary.joblib"
SCORE_MODEL_PATH = MODEL_DIRECTORY / "radiomics_2d_score.joblib"


def main():
    df = pd.read_csv(CSV_PATH)
    score_column = f"malignancy_{LABEL_AGGREGATION}"

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("original_")
    ]
    X = df[feature_columns]

    missing_values = X.isna().sum().sum()
    print(f"Missing feature values: {missing_values}")

    if missing_values > 0:
        raise ValueError("The radiomic features contain missing values.")

    # Split patient IDs so that nodules from the same patient cannot
    # appear in both training and testing.
    patient_ids = df["patient_id"].unique()
    train_patients, test_patients = train_test_split(
        patient_ids,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    train_mask = df["patient_id"].isin(train_patients)
    test_mask = df["patient_id"].isin(test_patients)

    # Binary model: <3 is benign, >3 is malignant, and 3 is excluded.
    binary_mask = df[score_column] != 3
    binary_train_mask = train_mask & binary_mask
    binary_test_mask = test_mask & binary_mask

    X_binary_train = X[binary_train_mask]
    X_binary_test = X[binary_test_mask]
    y_binary_train = (df.loc[binary_train_mask, score_column] > 3).astype(int)
    y_binary_test = (df.loc[binary_test_mask, score_column] > 3).astype(int)

    binary_model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    binary_model.fit(X_binary_train, y_binary_train)

    binary_predictions = binary_model.predict(X_binary_test)
    binary_probabilities = binary_model.predict_proba(X_binary_test)[:, 1]

    binary_accuracy = accuracy_score(y_binary_test, binary_predictions)
    sensitivity = recall_score(
        y_binary_test,
        binary_predictions,
        pos_label=1,
        zero_division=0,
    )
    specificity = recall_score(
        y_binary_test,
        binary_predictions,
        pos_label=0,
        zero_division=0,
    )

    if y_binary_test.nunique() == 2:
        binary_auc = roc_auc_score(y_binary_test, binary_probabilities)
    else:
        binary_auc = np.nan

    # Score model: predict the original malignancy score, including 3.
    X_score_train = X[train_mask]
    X_score_test = X[test_mask]
    y_score_train = df.loc[train_mask, score_column]
    y_score_test = df.loc[test_mask, score_column]

    score_model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    score_model.fit(X_score_train, y_score_train)

    score_predictions = np.clip(score_model.predict(X_score_test), 1, 5)
    score_mae = mean_absolute_error(y_score_test, score_predictions)
    score_rmse = np.sqrt(mean_squared_error(y_score_test, score_predictions))

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    joblib.dump(binary_model, BINARY_MODEL_PATH)
    joblib.dump(score_model, SCORE_MODEL_PATH)

    print(f"Aggregation: {LABEL_AGGREGATION}")
    print(f"Training patients: {len(train_patients)}")
    print(f"Testing patients: {len(test_patients)}")

    print("\nBinary model")
    print(f"Nodules: {binary_mask.sum()}")
    print(f"Accuracy: {binary_accuracy:.3f}")
    print(f"AUC: {binary_auc:.3f}")
    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")

    print("\nMalignancy score model")
    print(f"Nodules: {len(df)}")
    print(f"MAE: {score_mae:.3f}")
    print(f"RMSE: {score_rmse:.3f}")

    print(f"\nSaved binary model to {BINARY_MODEL_PATH}")
    print(f"Saved score model to {SCORE_MODEL_PATH}")


if __name__ == "__main__":
    main()
