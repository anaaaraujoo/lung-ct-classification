# por-definir-ds

## Name

## Description

## Badges

## Visuals

## Installation

Clone the repo:

```bash
git clone https://gitlab.up.pt/por-definir/por-definir-ds
cd por-definir-ds
```

### Using uv

```bash
uv sync
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Adding new dependencies

To add a new package, please use the *uv add* command and then sync requirements.txt (for pip users) with the following command:

```bash
uv export --format requirements.txt --no-hashes -o requirements.txt
```

## Usage

### File locations

Edit **`configs/paths.yaml`** to set input paths and output folders for both
extractors and training. Relative paths start at the repository root.
Use `--paths another.yaml` to select another paths file.
Existing command-line path arguments override the YAML.

**Exception:** LIDC DICOM files remain configured in `~/.pylidcrc`, read by
`pylidc`. We do not change that file. Radiomics settings are in
`configs/radiomics_circle_2d.yaml` and `configs/radiomics_segmentation_2d.yaml`.

### LNDb masks: which CSV does what?

| File | Used for |
|---|---|
| `allNods.csv` | The nodule groups used by our saved malignancy targets |
| `trainNodules_gt.csv` | Mask filenames (`RadID`) and labels inside masks (`RadFindingID`) |

**Their grouped `FindingID` values can differ.** We match nodules using the
same CT, centres within 3 mm, matching equivalent diameter and matching individual
finding labels. Missing or ambiguous matches stop extraction.

Example: CT 28, `allNods` finding **6** matches `_gt` finding **7**. We load
the masks listed in `_gt` row 7 and save features with ID **6**, preserving its target.
The correspondence is saved as `mask_mapping.csv` in the output's `.scans` folder.

### Extract segmentation features

Use the supplied reader masks, combine them, and extract features from the axial
slice with the largest mask area. LIDC masks come from `pylidc` reader contours.

```bash
uv run python LIDCfeatureExtraction.py --roi segmentation --consensus 0.5
uv run python LNDBfeatureExtraction.py --roi segmentation --consensus 0.5
```

Consensus: `0.5` = 50%, `0.75` = 75%, `1` = all readers. At 50%, two readers
produce their union. Missing masks are errors, not dropped readers.

At 50%, the default outputs are `data/lidc_segmentation_2d_c50_hu.csv` and
`data/lndb_ge3_segmentation_2d_c50_hu.csv`. Rerun to resume and retry failures.
Changed code, inputs or settings require another `--output`. For a small trial,
add `--limit-scans 1 --output /tmp/opencode/trial.csv`.

### Train and compare

After extraction, train the 500-tree weighted RF:

```bash
uv run python src/modeling.py \
  --lidc data/lidc_segmentation_2d_c50_hu.csv \
  --lndb data/lndb_ge3_segmentation_2d_c50_hu.csv \
  --lndb-targets data/processed/lndb_ge3_nodule_targets.csv \
  --class-weight balanced \
  --output models/rf_segmentation_2d_c50_balanced
```

Compare with circles at radius values 11/6 on the same LNDb population:

```bash
uv run python src/modeling.py \
  --lidc data/lidc_circle_r11_hu.csv \
  --lndb data/lndb_circle_r6_hu.csv \
  --lndb-targets data/processed/lndb_ge3_nodule_targets.csv \
  --class-weight balanced \
  --output models/rf_circle_r11_r6_ge3_balanced
```

The segmentation population is 2,637 LIDC and 622 LNDb nodules. Missing features
stop training. Any deliberate exclusions must use the same explicit target tables
in both comparisons. Targets, saved patient folds and competition outputs remain
unchanged. Validation uses LNDb folds, with all LIDC in training only.

### Check how much the model leans on nodule size

Texture features that count voxels grow with the nodule, so a size-only baseline
is worth reporting next to any radiomics score. Both flags need segmentation
features, which carry nodule volume.

```bash
uv run python src/modeling.py \
  --lidc data/lidc_segmentation_3d_c50_hu.csv \
  --lndb data/lndb_ge3_segmentation_3d_c50_hu.csv \
  --lndb-targets data/processed/lndb_ge3_nodule_targets.csv \
  --class-weight balanced \
  --max-size-correlation 0.7 \
  --output models/rf_segmentation_3d_c50_ge3_maxsize0.7
```

`--size-only` trains on nodule volume alone. `--max-size-correlation` drops every
feature whose Spearman correlation with size exceeds the value. The correlations
use training rows only, inside each fold. Section 10 of
`notebooks/radiomics_experiments.ipynb` reports both.

Circle and sphere features describe a fixed region, so they have no shape columns.
For those, both flags take nodule volume from the masks, read from the `shape`
section of `configs/paths.yaml`, which covers the 622 LNDb ge3 nodules.

### Train one model per feature family

`--family` keeps only the requested radiomics families, so a model can be trained
on one kind of statistic at a time. The names are `firstorder`, `glcm`, `glrlm`,
`glszm`, `gldm`, `ngtdm` and `shape`, plus the groups `intensity` (firstorder) and
`texture` (all five texture families). Separate several with commas.

```bash
uv run python src/modeling.py \
  --lidc data/lidc_sphere_r11_hu.csv \
  --lndb data/lndb_sphere_r6_hu.csv \
  --family glcm \
  --smote \
  --output models/expert_sphere_glcm
```

These options split radiomics by statistic. For binary experts organised around
radiologist concepts instead, see `notebooks/model_ensemble.ipynb`.

### Teach the model the radiologists' attributes

`--auxiliary-attributes` adds the five ratings the readers give every LIDC and
LNDb training nodule (lobulation, spiculation, sphericity, margin, texture) to the
classifier's features. A forest predicts each one from the radiomics first, and the
classifier trains on those predictions. The predictions it trains on come from
inner folds, so a training row never feeds the attribute model that describes it.
The ratings never exist for a nodule the model has to score, so nothing external is
needed at prediction time.

```bash
uv run python src/modeling.py \
  --lidc data/lidc_sphere_r11_hu.csv \
  --lndb data/lndb_sphere_r6_hu.csv \
  --smote \
  --auxiliary-attributes \
  --output models/rf_sphere_smote_aux
```

It needs `--model rf` and cannot be combined with tuning or `--multi-slice`. The
tables are the LIDC annotations and the LNDb `allNods.csv` findings, both read from
`configs/paths.yaml`.

### Binary alerts from segmentation

The tables displayed in `notebooks/radiomics_analysis.ipynb` are generated by
`notebooks/radiomics_experiments.ipynb`. Its opening table links each exported
CSV under `data/processed/` to its calculation and main-notebook section.

The main binary workflow uses `src/binary_evaluation.py` for baseline evaluation
and threshold calibration, then `src/binary_tuning.py` for diameter filtering
and tuning. `src/auxiliary.py` predicts reader attributes from radiomics and
supports the auxiliary-target model in the five-class ranking.
`src/concept_experts.py` trains the experimental concept-guided experts and
combines their scores. Their results are presented in
`notebooks/model_ensemble.ipynb`. Saved models still use these modules, so their
classes and import paths are retained.

Section 11 of `notebooks/radiomics_experiments.ipynb` compares only segmentation-based models.
It separates binary-trained models from five-class models used as binary alert
scores, and reports false negatives, false positives, accuracy, balanced accuracy,
macro F1, recall, specificity, precision and AUC on shared nodules. The labels are radiologist ratings, not diagnoses.

The ROC point with no missed positive is descriptive when its threshold is chosen
on the evaluated rows. To test threshold selection on separate patient folds:

```bash
uv run python -m src.binary_evaluation
```

Each test fold uses the next saved fold for threshold calibration and the other
three for training, with LIDC added only to training. Confirmed duplicate LNDb CT
copies are excluded. The model is not refitted after calibration. Results go to
`models/segmentation_binary_calibration/`. Zero false negatives on calibration
does not guarantee zero false negatives on the test fold.

To filter diameter-related features, or jointly filter and tune the binary models:

```bash
uv run python -m src.binary_tuning \
  --max-diameter-correlation 0.7 \
  --output models/segmentation_binary_diameter07

uv run python -m src.binary_tuning \
  --tune --max-diameter-correlation 0.7 \
  --output models/segmentation_binary_diameter07_tuned
```

The filter removes features whose absolute Spearman correlation with maximum 3D
diameter exceeds the limit, using fitting rows only. Tuning compares four settings
per model, including the unchanged baseline. Inner patient folds independently
fit the model, calibrate its threshold and validate alerts. Settings are selected
by the fewest inner false negatives, then the highest specificity. Outer
calibration and test folds are excluded from that search. Feature lists and
selected parameters are saved beside the results.

### Binary experts by radiologist concept

```bash
uv run python -m src.concept_experts --n-jobs 2
```

Seven experts learn lobulation, spiculation, sphericity, margin, texture,
calcification and subtlety from segmentation radiomics. Each predicts one
concept, then a logistic regression converts that prediction to a binary alert
score. The risk head trains on concept predictions from disjoint patient groups,
never on the reader's actual rating. Calcification means reported presence,
not a numerical ordering of its types. Internal structure is not included because
it is not available in the cleaned LIDC annotations.

An eighth expert reads the mask's 14 shape features directly. Their maximum is
compared with a balanced random forest reading all radiomics. Each outer split
uses separate fitting, threshold-selection and test patients, with LIDC in fitting
only. Confirmed LNDb CT copies are excluded. Results and fold models are saved in
`models/segmentation_concept_experts/`, without overwriting existing results.
The default uses two CPU workers. `notebooks/model_ensemble.ipynb` reads the saved
results and compares FN, FP, ROC curves and nodules with equivalent diameter ≤6 mm.
The labels are reader ratings, not confirmed cancer diagnoses.

### Tests

```bash
uv run python -m unittest discover -s tests -v
```

## Support

Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap

If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing

State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment

Show your appreciation to those who have contributed to the project.

## License

For open source projects, say how it is licensed.

## Project status

If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
