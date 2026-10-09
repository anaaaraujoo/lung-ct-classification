"""Extract LNDb circle radiomics or 2D radiomics from existing reader masks."""

import argparse
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd
import radiomics
import SimpleITK as sitk
from radiomics import featureextractor

from src.radiomics_extraction import (
    combine_results,
    combine_reader_masks,
    check_mask_geometry,
    create_circle,
    extract_features,
    largest_mask_slice,
    prepare_run,
    save_csv,
    save_errors,
)

ROOT = Path(__file__).resolve().parent
KEYS = ["LNDbID", "FindingID"]


def load_scan(data_dir, lndb_id):
    scan_name = f"LNDb-{lndb_id:04d}"
    image_path = data_dir / f"{scan_name}.mhd"
    if image_path.exists():
        return sitk.ReadImage(str(image_path))

    # Read one zipped scan through a temporary copy, preserving the original ZIP.
    with TemporaryDirectory(prefix=f"{scan_name}-") as temporary:
        with ZipFile(data_dir / f"{scan_name}.zip") as archive:
            archive.extractall(temporary)
        image_path = next(Path(temporary).rglob(f"{scan_name}.mhd"))
        return sitk.ReadImage(str(image_path))


def reader_findings(nodule):
    """Map a grouped finding to the positive nodule labels used by its readers."""
    columns = []
    for name in ("RadID", "RadFinding", "Nodule"):
        values = [float(value) for value in str(nodule[name]).split(",")]
        if not all(np.isfinite(value) and value.is_integer() for value in values):
            raise ValueError(f"Invalid reader mapping in {name}.")
        columns.append([int(value) for value in values])
    if len({len(values) for values in columns}) != 1:
        raise ValueError("RadID, RadFinding and Nodule must have matching lengths.")
    readers = []
    for reader_id, label, is_nodule in zip(*columns):
        if reader_id <= 0 or label <= 0 or is_nodule not in (0, 1):
            raise ValueError("Invalid reader ID, finding label or nodule flag.")
        if is_nodule == 1:
            readers.append((reader_id, label))
    if not readers or len({reader for reader, _ in readers}) != len(readers):
        raise ValueError("Each nodule must have one mask per associated nodule reader.")
    return readers


def load_reader_mask(archive_path, lndb_id, reader_id):
    """Read only this reader's volume from the ZIP, leaving raw data unchanged."""
    name = f"LNDb-{lndb_id:04d}_rad{reader_id}"
    with ZipFile(archive_path) as archive, TemporaryDirectory(prefix=f"{name}-") as temporary:
        for suffix in ("mhd", "raw"):
            archive.extract(f"masks/{name}.{suffix}", temporary)
        return sitk.ReadImage(str(Path(temporary) / "masks" / f"{name}.mhd"))


def create_lndb_segmentation(image, nodule, masks):
    """Select reader-specific labels, not the grouped FindingID."""
    readers = reader_findings(nodule)
    for reader_id, _ in readers:
        check_mask_geometry(image, masks[reader_id])
    # Combine one binary reader mask at a time, keeping memory use bounded.
    reader_masks = (sitk.GetArrayViewFromImage(masks[reader_id]) == label for reader_id, label in readers)
    values = combine_reader_masks(reader_masks)
    mask = sitk.GetImageFromArray(values)
    mask.CopyInformation(image)
    return largest_mask_slice(image, mask)


def extract_features_for_series(lndb_id, nodules, data_dir, extractor, radius_mm, clip, directory,
                                roi="circle", masks_path=None):
    scan_name = f"LNDb-{lndb_id:04d}"
    checkpoint = directory / f"{scan_name}.features.csv"
    rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
    completed = {row["FindingID"] for row in rows}
    pending = nodules[~nodules["FindingID"].isin(completed)]
    if pending.empty:
        print(f"{scan_name}: already extracted")
        return

    roi_name = f"circle_r{radius_mm:g}" if roi == "circle" else "segmentation_2d"
    print(f"{scan_name}: {len(pending)} nodules, {roi_name}", flush=True)
    errors = []
    masks = {}
    try:
        image = load_scan(data_dir, lndb_id)
    except Exception as exc:
        errors = [{**row, "error": str(exc)} for row in pending[KEYS].to_dict("records")]
    else:
        for _, nodule in pending.iterrows():
            row = {"LNDbID": int(lndb_id), "FindingID": int(nodule["FindingID"])}
            try:
                if roi == "segmentation":
                    for reader_id, _ in reader_findings(nodule):
                        if reader_id not in masks:
                            masks[reader_id] = load_reader_mask(masks_path, lndb_id, reader_id)
                    image_slice, mask, row["slice_index"] = create_lndb_segmentation(image, nodule, masks)
                else:
                    center = tuple(nodule[["x", "y", "z"]].astype(float))
                    image_slice, mask = create_circle(image, center, radius_mm)
                row.update(extract_features(image_slice, mask, extractor, clip))
                rows.append(row)
            except Exception as exc:
                errors.append({**row, "error": str(exc)})
        del image

    if rows:
        save_csv(pd.DataFrame(rows), checkpoint)
    save_errors(errors, directory / f"{scan_name}.errors.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi", choices=["circle", "segmentation"], default="circle")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "Downloads/LNDb")
    parser.add_argument("--masks", type=Path, help="Reader masks ZIP. Default: --data-dir/masks.zip.")
    parser.add_argument("--findings", type=Path, help="Training allNods.csv, if stored outside --data-dir.")
    parser.add_argument("--test-nodules", type=Path, help="Competition CSV with LNDbID, FindingID, x, y, z.")
    parser.add_argument("--radius", type=float, default=11, help="Radius in mm. Default: full-training LNDb P99.")
    parser.add_argument("--clip", action="store_true", help="Clip HU to [-1000, 400].")
    parser.add_argument("--scan", type=int, help="Extract only this LNDbID.")
    parser.add_argument("--limit-scans", type=int, help="Limit the number of scans for a small run.")
    parser.add_argument("--config", type=Path, help="Default: configuration matching --roi.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.config = args.config or ROOT / f"configs/radiomics_{args.roi}_2d.yaml"
    args.masks = args.masks or args.data_dir / "masks.zip"
    if args.test_nodules and args.roi == "segmentation":
        parser.error("Competition test nodules have no supplied reader masks. Use --roi circle.")

    if args.test_nodules:
        nodules = pd.read_csv(args.test_nodules)[[*KEYS, "x", "y", "z"]]
        sources = [Path(__file__), args.test_nodules]
        dataset = "lndb_test"
    else:
        findings_path = args.findings or args.data_dir / "allNods.csv"
        dataset = "lndb_ge3" if args.roi == "segmentation" else "lndb"
        targets_path = ROOT / f"data/processed/{dataset}_nodule_targets.csv"
        folds_path = ROOT / "data/processed/lndb_patient_folds.csv"
        finding_columns = [*KEYS, "x", "y", "z"]
        if args.roi == "segmentation":
            finding_columns += ["RadID", "RadFinding", "Nodule"]
        # NoduleID in the targets is the grouped FindingID from allNods.csv.
        nodules = pd.read_csv(targets_path).rename(columns={"NoduleID": "FindingID"})
        nodules = nodules.merge(
            pd.read_csv(findings_path)[finding_columns],
            on=KEYS, how="left", validate="one_to_one",
        ).merge(pd.read_csv(folds_path), on="LNDbID", how="left", validate="many_to_one")
        if nodules["fold"].isna().any():
            raise ValueError("A patient is missing from the saved folds.")
        sources = [Path(__file__), findings_path, targets_path, folds_path]
        if args.roi == "segmentation":
            sources.append(args.masks)

    if nodules[KEYS].isna().any().any() or nodules.duplicated(KEYS).any():
        raise ValueError("Each nodule must have a unique LNDbID + FindingID.")
    if args.scan is not None:
        nodules = nodules[nodules["LNDbID"] == args.scan]
    if args.limit_scans is not None:
        if args.limit_scans < 1:
            raise ValueError("--limit-scans must be positive.")
        scans = nodules["LNDbID"].drop_duplicates().head(args.limit_scans)
        nodules = nodules[nodules["LNDbID"].isin(scans)]
    if nodules.empty:
        raise ValueError("No nodules match the requested scans.")

    intensity = "clipped" if args.clip else "hu"
    roi_name = f"circle_r{args.radius:g}" if args.roi == "circle" else "segmentation_2d"
    output = args.output or ROOT / f"data/{dataset}_{roi_name}_{intensity}.csv"
    directory = prepare_run(output, args.config, args.radius, args.clip, sources, roi=args.roi)
    radiomics.setVerbosity(logging.ERROR)
    extractor = featureextractor.RadiomicsFeatureExtractor(str(args.config))
    for lndb_id, scan_nodules in nodules.groupby("LNDbID", sort=False):
        extract_features_for_series(
            int(lndb_id), scan_nodules, args.data_dir, extractor, args.radius, args.clip, directory,
            roi=args.roi, masks_path=args.masks,
        )
    return int(combine_results(directory, nodules, KEYS, output) > 0)


if __name__ == "__main__":
    raise SystemExit(main())
