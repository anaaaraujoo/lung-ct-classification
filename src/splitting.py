import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def create_patient_folds(targets, patient_column, target_column, n_splits):
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    folds = np.zeros(len(targets), dtype=int)

    splits = splitter.split(
        X=targets,
        y=targets[target_column],
        groups=targets[patient_column],
    )

    for fold_number, (training_rows, validation_rows) in enumerate(
        splits,
        start=1,
    ):
        folds[validation_rows] = fold_number

    patient_folds = targets[[patient_column]].copy()
    patient_folds["fold"] = folds

    # Targets has one row per nodule. Keep one fold assignment per patient.
    patient_folds = patient_folds.drop_duplicates()

    return patient_folds.reset_index(drop=True)