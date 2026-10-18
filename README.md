# Lung Cancer Classification using CT Data

**Laboratory of Artificial Intelligence and Data Science — 2026/2027**

This project studies the use of CT images and machine learning to support the assessment of lung nodule malignancy.

We use LIDC-IDRI and LNDb to explore data quality, image preprocessing, radiomics extraction, feature selection and predictive models. The experiments cover five-class classification, binary classification and regression.

The prediction targets come from radiologists’ ratings. These ratings describe their assessment of the nodules, rather than confirmed cancer diagnoses.

## Start here

The main report is [radiomics_analysis.ipynb](notebooks/radiomics_analysis.ipynb).

The notebooks organise the work as follows:

| Notebook | Purpose |
|---|---|
| [data_understanding.ipynb](data_understanding.ipynb) | Initial LIDC exploration, data quality checks and annotation preparation |
| [radiomics_decisions_eda.ipynb](notebooks/radiomics_decisions_eda.ipynb) | Image preprocessing and feature-extraction decisions |
| [radiomics_experiments.ipynb](notebooks/radiomics_experiments.ipynb) | Supporting model comparisons and result calculations |
| [radiomics_analysis.ipynb](notebooks/radiomics_analysis.ipynb) | Main methodology, results, interpretation and discussion |

## Approach

1. Inspect the CT data and reader annotations.
2. Combine reader ratings and create patient-level validation folds.
3. Extract radiomics from circles, spheres and segmentation masks, comparing 2D, multiple-slice and 3D approaches.
4. Compare Random Forest, Balanced Random Forest, XGBoost and Ridge regression.
5. Evaluate feature selection, model settings and prediction thresholds.
6. Analyse model performance and the features influencing predictions.

LNDb is used for evaluation, with LIDC added to training. Validation uses five patient-level folds, with each fold evaluated once. Comparisons use shared nodules.

## Repository structure

- `notebooks/` — main report and supporting analysis.
- `src/` — data preparation, modeling and evaluation code.
- `configs/` — dataset paths and radiomics settings.
- `tests/` — automated tests.
- `data/` — local datasets, extracted features and generated tables.
- `models/` — saved models, predictions and metrics.

CT scans, saved models and most generated CSVs are not included in Git.

## Installation

```bash
git clone https://gitlab.up.pt/por-definir/por-definir-ds
cd por-definir-ds
uv sync
```

Open the notebooks using the Python environment in `.venv`. Run terminal commands from the repository root.

## Dataset configuration

Dataset information and access:

- [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/)
- [LNDb](https://lndb.grand-challenge.org/)

Set local input and output paths in [`configs/paths.yaml`](configs/paths.yaml).

PyLIDC reads the LIDC DICOM location separately from `~/.pylidcrc`:

```ini
[dicom]
path = /path/to/LIDC-IDRI
warn = True
```

## Reproducing the analysis

1. Prepare the annotations and configure the dataset paths.
2. Run sections 1–3 of the main notebook to generate targets and patient folds.
3. Follow the extraction and training commands documented in the notebooks.
4. Run the supporting experiments to generate their result CSVs and ROC figure.
5. Run the evaluation cells in the main notebook.

The notebooks identify the saved runs used for each comparison. Their result cells require the corresponding files under `models/` and `data/processed/`.

The main scripts are:

| Script | Purpose |
|---|---|
| `LIDCfeatureExtraction.py` | Extract LIDC radiomics |
| `LNDBfeatureExtraction.py` | Extract LNDb radiomics |
| `src/modeling.py` | Train classification models |
| `src/regression.py` | Train Ridge regression |
| `src/binary_evaluation.py` | Evaluate binary decisions and threshold calibration |
| `src/binary_tuning.py` | Test binary feature filtering and tuning |
| `src/make_predictions.py` | Generate competition predictions |

Each script provides its available arguments through `--help`.

## Competition

The competition predicts five-class ratings for the LNDb test candidates. These candidates have no supplied segmentation masks, so the submission uses features extracted around their provided locations.

The binary experiments investigate an additional objective: reducing missed positive-rated nodules.

## Tests

```bash
uv run --with pytest python -m pytest -q
```