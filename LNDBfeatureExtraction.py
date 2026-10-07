"""Extract circular LNDb radiomics from training or competition centroids."""

import argparse
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import pandas as pd
import radiomics
import SimpleITK as sitk
from radiomics import featureextractor

from src.radiomics_extraction import (
    combine_results,
    create_circle,
    extract_features,
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


def extract_features_for_series(lndb_id, nodules, data_dir, extractor, radius_mm, clip, directory):
    scan_name = f"LNDb-{lndb_id:04d}"
    checkpoint = directory / f"{scan_name}.features.csv"
    rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
    completed = {row["FindingID"] for row in rows}
    pending = nodules[~nodules["FindingID"].isin(completed)]
    if pending.empty:
        print(f"{scan_name}: already extracted")
        return

    print(f"{scan_name}: {len(pending)} nodules, radius {radius_mm:g} mm", flush=True)
    errors = []
    try:
        image = load_scan(data_dir, lndb_id)
    except Exception as exc:
        errors = [{**row, "error": str(exc)} for row in pending[KEYS].to_dict("records")]
    else:
        for _, nodule in pending.iterrows():
            row = {"LNDbID": int(lndb_id), "FindingID": int(nodule["FindingID"])}
            try:
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
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "Downloads/LNDb")
    parser.add_argument("--findings", type=Path, help="Training allNods.csv, if stored outside --data-dir.")
    parser.add_argument("--test-nodules", type=Path, help="Competition CSV with LNDbID, FindingID, x, y, z.")
    parser.add_argument("--radius", type=float, default=11, help="Radius in mm. Default: full-training LNDb P99.")
    parser.add_argument("--clip", action="store_true", help="Clip HU to [-1000, 400].")
    parser.add_argument("--scan", type=int, help="Extract only this LNDbID.")
    parser.add_argument("--limit-scans", type=int, help="Limit the number of scans for a small run.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/radiomics_circle_2d.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.test_nodules:
        nodules = pd.read_csv(args.test_nodules)[[*KEYS, "x", "y", "z"]]
        sources = [Path(__file__), args.test_nodules]
        dataset = "lndb_test"
    else:
        findings_path = args.findings or args.data_dir / "allNods.csv"
        targets_path = ROOT / "data/processed/lndb_nodule_targets.csv"
        folds_path = ROOT / "data/processed/lndb_patient_folds.csv"
        # NoduleID in the targets is the grouped FindingID from allNods.csv.
        nodules = pd.read_csv(targets_path).rename(columns={"NoduleID": "FindingID"})
        nodules = nodules.merge(
            pd.read_csv(findings_path)[[*KEYS, "x", "y", "z"]],
            on=KEYS, how="left", validate="one_to_one",
        ).merge(pd.read_csv(folds_path), on="LNDbID", how="left", validate="many_to_one")
        if nodules["fold"].isna().any():
            raise ValueError("A patient is missing from the saved folds.")
        sources = [Path(__file__), findings_path, targets_path, folds_path]
        dataset = "lndb"

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
    output = args.output or ROOT / f"data/{dataset}_circle_r{args.radius:g}_{intensity}.csv"
    directory = prepare_run(output, args.config, args.radius, args.clip, sources)
    radiomics.setVerbosity(logging.ERROR)
    extractor = featureextractor.RadiomicsFeatureExtractor(str(args.config))
    for lndb_id, scan_nodules in nodules.groupby("LNDbID", sort=False):
        extract_features_for_series(
            int(lndb_id), scan_nodules, args.data_dir, extractor, args.radius, args.clip, directory
        )
    return int(combine_results(directory, nodules, KEYS, output) > 0)


if __name__ == "__main__":
    raise SystemExit(main())
