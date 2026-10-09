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

### 2D radiomics from nodule segmentations

This is a separate malignancy experiment, not an automatic segmentation model.
The circle extraction remains the default and competition outputs are not replaced.

The scripts use the cleaned LIDC targets (2,637 nodules) and the saved LNDb
segmented-nodule targets (622 nodules). LNDb's diameter value of 3 is a sentinel
for point-only findings, already excluded from `lndb_ge3_nodule_targets.csv`.

LIDC reader contours are loaded through `pylidc` and checked against the saved
scan IDs, reader indices and centroids. LNDb uses `allNods.csv` to associate
each grouped finding with its reader-specific label in `masks.zip`.
Only readers who marked the finding as a nodule participate in the consensus.
Missing reader masks are errors, not silently removed from the denominator.

Masks are combined at **at least 50% agreement**, including ties. With two
readers, this is their union. Features are extracted from the axial slice with
the largest consensus-mask area, choosing the first slice in an area tie.
The selected original `slice_index` is saved for inspection, not used by the model.

`configs/radiomics_segmentation_2d.yaml` retains the existing image filters,
0.7 × 0.7 mm in-plane spacing, B-spline CT interpolation, nearest-neighbour mask
interpolation and 25 HU bins. It adds `shape2D`, without LoG, HU clipping or
intensity normalization.

Extract both datasets from the repository root:

```bash
uv run python LIDCfeatureExtraction.py --roi segmentation
uv run python LNDBfeatureExtraction.py --roi segmentation
```

LNDb defaults to `~/Downloads/LNDb`, containing the CTs, `allNods.csv` and
`masks.zip`. Override with `--data-dir`, `--findings` or `--masks` if needed.
LIDC uses the existing `pylidc` DICOM configuration. For a small trial, use
`--limit-scans 1 --output /tmp/opencode/lidc_segmentation_trial.csv`, or the
equivalent LNDb command with a different output path.

Outputs are `data/lidc_segmentation_2d_hu.csv` and
`data/lndb_ge3_segmentation_2d_hu.csv`. Each scan has a checkpoint and an error
CSV in the corresponding `.scans` directory. Repeating the command retries
failed nodules and skips successful ones. Changed settings, source tables,
mask archive or extraction code require a different output path.
CTs and masks are read one scan at a time, preserving the raw files.

After **both extractions are complete**, train the same 500-tree weighted RF:

```bash
uv run python src/modeling.py \
  --lidc data/lidc_segmentation_2d_hu.csv \
  --lndb data/lndb_ge3_segmentation_2d_hu.csv \
  --lndb-targets data/processed/lndb_ge3_nodule_targets.csv \
  --class-weight balanced \
  --output models/rf_segmentation_2d_balanced
```

Compare with the existing circle features at radius values 11/6, selecting
the **same 622 LNDb nodules** without repeating their extraction:

```bash
uv run python src/modeling.py \
  --lidc data/lidc_circle_r11_hu.csv \
  --lndb data/lndb_circle_r6_hu.csv \
  --lndb-targets data/processed/lndb_ge3_nodule_targets.csv \
  --class-weight balanced \
  --output models/rf_circle_r11_r6_ge3_balanced
```

An explicit target table selects the population from a larger feature CSV.
Missing features for any selected nodule stop training. If a smaller population
is deliberately chosen after inspecting extraction failures, supply identical
`--lidc-targets` and `--lndb-targets` tables to **both** comparisons.

Both runs predict `target_classification_mean` (classes 1–5), using the saved
LNDb patient folds with all LIDC in training only. Imputation is fitted within
each training fold. Ratings, reader IDs, coordinates and other metadata are
excluded from the feature whitelist. Cross-fold duplicate CTs remain unchanged
and still limit validation reliability. The mask and slice choice both change
in this comparison, so it evaluates the complete segmentation workflow.

Run the small automated tests:

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
