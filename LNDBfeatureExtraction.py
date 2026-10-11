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
    load_paths,
    prepare_run,
    save_csv,
    save_errors,
    validate_consensus,
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


def integer_list(value):
    values = [float(part) for part in str(value).split(",")]
    if not all(np.isfinite(number) and number.is_integer() for number in values):
        raise ValueError("Reader and finding IDs must be finite integers.")
    return [int(number) for number in values]


def reader_findings(nodule):
    """Use _gt mask readers and labels, never allNods reader IDs."""
    columns = [integer_list(nodule[name]) for name in ("mask_RadID", "mask_RadFindingID")]
    if len(columns[0]) != len(columns[1]) or any(value <= 0 for values in columns for value in values):
        raise ValueError("Mask readers and finding labels must be positive and have matching lengths.")
    readers = list(zip(*columns))
    if not readers or len({reader for reader, _ in readers}) != len(readers):
        raise ValueError("Each nodule must have one mask per associated nodule reader.")
    return readers


def match_nodules_to_masks(nodules, gt):
    """Find each allNods nodule in _gt without assuming shared group or reader IDs."""
    gt = gt[gt["Nodule"] == 1].copy()
    # allNods gives equivalent diameter, _gt gives volume. Compare the same size.
    gt["diameter_mm"] = 2 * np.cbrt(3 * gt["Volume"] / (4 * np.pi))
    gt["labels"] = gt["RadFindingID"].map(lambda value: tuple(sorted(integer_list(value))))
    rows = []
    for _, nodule in nodules.iterrows():
        # 1. Only consider nodules in the same CT.
        candidates = gt[gt["LNDbID"] == nodule["LNDbID"]].copy()
        offsets = candidates[["x", "y", "z"]].to_numpy() - nodule[["x", "y", "z"]].to_numpy(dtype=float)
        candidates["distance_mm"] = np.linalg.norm(offsets, axis=1)

        # 2. Compare centres, measured size and individual labels, not reader IDs.
        labels, flags = integer_list(nodule["RadFinding"]), integer_list(nodule["Nodule"])
        if len(labels) != len(flags) or any(flag not in (0, 1) for flag in flags):
            raise ValueError("allNods finding labels and nodule flags must align.")
        labels = tuple(sorted(label for label, flag in zip(labels, flags) if flag == 1))
        same_labels = candidates["labels"] == labels
        same_size = np.isclose(candidates["diameter_mm"], nodule["DiamEq_Rad"], atol=1e-5, rtol=0)
        close_centres = candidates["distance_mm"] <= 3.0
        matches = candidates[same_labels & same_size & close_centres]
        if len(matches) != 1:
            raise ValueError(f"CT {nodule['LNDbID']}, finding {nodule['FindingID']}: expected one _gt match, found {len(matches)}.")

        # 3. Keep the allNods ID for the target. Take mask readers and labels from _gt.
        mask_nodule = matches.iloc[0]
        rows.append({
            "LNDbID": int(nodule["LNDbID"]), "FindingID": int(nodule["FindingID"]),
            "gt_FindingID": int(mask_nodule["FindingID"]),
            "mask_RadID": mask_nodule["RadID"], "mask_RadFindingID": mask_nodule["RadFindingID"],
            "match_distance_mm": float(mask_nodule["distance_mm"]),
        })
    mapping = pd.DataFrame(rows)
    if mapping.duplicated(["LNDbID", "gt_FindingID"]).any():
        raise ValueError("Two allNods nodules matched the same _gt nodule.")
    return mapping


def load_reader_mask(archive_path, lndb_id, reader_id):
    """Read only this reader's volume from the ZIP, leaving raw data unchanged."""
    name = f"LNDb-{lndb_id:04d}_rad{reader_id}"
    with ZipFile(archive_path) as archive, TemporaryDirectory(prefix=f"{name}-") as temporary:
        for suffix in ("mhd", "raw"):
            archive.extract(f"masks/{name}.{suffix}", temporary)
        return sitk.ReadImage(str(Path(temporary) / "masks" / f"{name}.mhd"))


def create_lndb_segmentation(image, nodule, masks, consensus_level=0.5):
    """Select reader-specific labels, not the grouped FindingID."""
    readers = reader_findings(nodule)
    for reader_id, _ in readers:
        check_mask_geometry(image, masks[reader_id])
    # Combine one binary reader mask at a time, keeping memory use bounded.
    reader_masks = (sitk.GetArrayViewFromImage(masks[reader_id]) == label for reader_id, label in readers)
    values = combine_reader_masks(reader_masks, consensus_level)
    mask = sitk.GetImageFromArray(values)
    mask.CopyInformation(image)
    return largest_mask_slice(image, mask)


def extract_features_for_series(lndb_id, nodules, data_dir, extractor, radius_mm, clip, directory,
                                roi="circle", masks_path=None, consensus_level=0.5):
    scan_name = f"LNDb-{lndb_id:04d}"
    checkpoint = directory / f"{scan_name}.features.csv"
    rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
    completed = {row["FindingID"] for row in rows}
    pending = nodules[~nodules["FindingID"].isin(completed)]
    if pending.empty:
        print(f"{scan_name}: already extracted")
        return

    roi_name = f"circle_r{radius_mm:g}" if roi == "circle" else f"segmentation_2d_c{100 * consensus_level:g}"
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
                    image_slice, mask, row["slice_index"] = create_lndb_segmentation(image, nodule, masks, consensus_level)
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
    parser.add_argument("--paths", type=Path, default=ROOT / "configs/paths.yaml", help="YAML with file locations.")
    parser.add_argument("--roi", choices=["circle", "segmentation"], default="circle")
    parser.add_argument("--data-dir", type=Path, help="Override the YAML CT directory and default mask/findings locations.")
    parser.add_argument("--masks", type=Path, help="Override the YAML reader masks ZIP.")
    parser.add_argument("--findings", type=Path, help="Override the YAML allNods.csv.")
    parser.add_argument("--consensus", type=float, default=0.5, help="Required fraction of nodule readers, including ties. Default: 0.5.")
    parser.add_argument("--test-nodules", type=Path, help="Competition CSV with LNDbID, FindingID, x, y, z.")
    parser.add_argument("--radius", type=float, default=11, help="Radius in mm. Default: full-training LNDb P99.")
    parser.add_argument("--clip", action="store_true", help="Clip HU to [-1000, 400].")
    parser.add_argument("--scan", type=int, help="Extract only this LNDbID.")
    parser.add_argument("--limit-scans", type=int, help="Limit the number of scans for a small run.")
    parser.add_argument("--config", type=Path, help="Default: configuration matching --roi.")
    parser.add_argument("--bin-width", type=float, help="Override the YAML intensity bin width. Default: use the YAML value.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.bin_width is not None and (not np.isfinite(args.bin_width) or args.bin_width <= 0):
        parser.error("--bin-width must be a positive finite number.")
    args.paths = args.paths.expanduser().resolve()
    paths = load_paths(args.paths)
    args.config = args.config or paths["radiomics"][args.roi]
    if args.data_dir:
        args.masks = args.masks or args.data_dir / "masks.zip"
        args.findings = args.findings or args.data_dir / "allNods.csv"
    else:
        args.data_dir = paths["lndb"]["ct_dir"]
        args.masks = args.masks or paths["lndb"]["masks"]
        args.findings = args.findings or paths["lndb"]["findings"]
    if args.test_nodules and args.roi == "segmentation":
        parser.error("Competition test nodules have no supplied reader masks. Use --roi circle.")
    validate_consensus(args.consensus)
    if args.roi == "circle" and args.consensus != 0.5:
        parser.error("--consensus applies only to --roi segmentation.")

    if args.test_nodules:
        nodules = pd.read_csv(args.test_nodules)[[*KEYS, "x", "y", "z"]]
        sources = [Path(__file__), args.paths, args.test_nodules]
        dataset = "lndb_test"
    else:
        findings_path = args.findings
        dataset = "lndb_ge3" if args.roi == "segmentation" else "lndb"
        targets_path = paths["lndb"]["segmentation_targets" if args.roi == "segmentation" else "targets"]
        folds_path = paths["lndb"]["folds"]
        finding_columns = [*KEYS, "x", "y", "z"]
        if args.roi == "segmentation":
            finding_columns += ["RadID", "RadFinding", "Nodule", "DiamEq_Rad"]
        # NoduleID in the targets is the grouped FindingID from allNods.csv.
        nodules = pd.read_csv(targets_path).rename(columns={"NoduleID": "FindingID"})
        nodules = nodules.merge(
            pd.read_csv(findings_path)[finding_columns],
            on=KEYS, how="left", validate="one_to_one",
        ).merge(pd.read_csv(folds_path), on="LNDbID", how="left", validate="many_to_one")
        if nodules["fold"].isna().any():
            raise ValueError("A patient is missing from the saved folds.")
        sources = [Path(__file__), args.paths, findings_path, targets_path, folds_path]
        if args.roi == "segmentation":
            mapping = match_nodules_to_masks(nodules, pd.read_csv(paths["lndb"]["gt"]))
            nodules = nodules.merge(mapping, on=KEYS, how="left", validate="one_to_one")
            sources.extend([args.masks, paths["lndb"]["gt"]])

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
    roi_name = f"circle_r{args.radius:g}" if args.roi == "circle" else f"segmentation_2d_c{100 * args.consensus:g}"
    bin_suffix = f"_bw{args.bin_width:g}" if args.bin_width is not None else ""
    output = args.output or paths["outputs"]["features"] / f"{dataset}_{roi_name}{bin_suffix}_{intensity}.csv"
    directory = prepare_run(output, args.config, args.radius, args.clip, sources,
                            roi=args.roi, consensus_level=args.consensus, bin_width=args.bin_width)
    if args.roi == "segmentation":
        save_csv(mapping, directory / "mask_mapping.csv")
    radiomics.setVerbosity(logging.ERROR)
    overrides = {"binWidth": args.bin_width} if args.bin_width is not None else {}
    extractor = featureextractor.RadiomicsFeatureExtractor(str(args.config), **overrides)
    for lndb_id, scan_nodules in nodules.groupby("LNDbID", sort=False):
        extract_features_for_series(
            int(lndb_id), scan_nodules, args.data_dir, extractor, args.radius, args.clip, directory,
            roi=args.roi, masks_path=args.masks, consensus_level=args.consensus,
        )
    return int(combine_results(directory, nodules, KEYS, output) > 0)


if __name__ == "__main__":
    raise SystemExit(main())
