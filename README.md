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
