"""Extract circle, sphere or segmented-nodule LIDC radiomics with saved targets and folds."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pylidc as pl
import radiomics
import SimpleITK as sitk
from pylidc.utils import consensus
from radiomics import featureextractor

from src.multislice_extraction import (
    combine_results as combine_slice_results,
    create_circle_slices,
    extract_slice_features,
    prepare_run_slices,
    segmentation_slices,
)
from src.radiomics_extraction import (
    combine_results,
    create_circle,
    create_sphere,
    extract_features,
    load_paths,
    save_csv,
    save_errors,
    segmentation_roi,
    validate_consensus,
)

ROOT = Path(__file__).resolve().parent
KEYS = ["patient_id", "series_uid", "nodule_id"]


def create_lidc_segmentation(image, cluster, saved_annotations, consensus_level=0.5, dimension=2):
    """Verify the saved reader association before reconstructing its contours."""
    saved = saved_annotations.sort_values("annotation_idx")
    if len(cluster) != len(saved) or saved["annotation_idx"].tolist() != list(range(len(cluster))):
        raise ValueError("Reader annotations do not match the saved nodule cluster.")
    for annotation, (_, row) in zip(cluster, saved.iterrows()):
        if annotation.scan.id != row["scan_id"] or not np.allclose(
            annotation.centroid, row[["row", "column", "slice"]].to_numpy(dtype=float),
            atol=1e-5, rtol=0,
        ):
            raise ValueError("Reader centroid or scan differs from the saved annotation.")

    validate_consensus(consensus_level)
    combined, bbox, reader_masks = consensus(cluster, clevel=consensus_level)
    if any(not mask.any() for mask in reader_masks):
        raise ValueError("A reader's nodule mask is empty.")
    # PyLIDC uses [row, column, slice], SimpleITK arrays use [slice, row, column].
    values = np.zeros(tuple(reversed(image.GetSize())), dtype=np.uint8)
    values[bbox[2], bbox[0], bbox[1]] = combined.transpose(2, 0, 1)
    mask = sitk.GetImageFromArray(values)
    mask.CopyInformation(image)
    return segmentation_roi(image, mask, dimension)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=ROOT / "configs/paths.yaml", help="YAML with file locations.")
    parser.add_argument("--roi", choices=["circle", "sphere", "segmentation"], default="circle")
    parser.add_argument("--dimension", type=int, choices=[2, 3], default=2,
                        help="Circle or axial segmentation (2), sphere or full segmentation (3). Default: 2.")
    parser.add_argument("--slices", type=int, default=1,
                        help="Axial slices around the centroid or largest mask slice. Requires --dimension 2. Default: 1.")
    parser.add_argument("--consensus", type=float, default=0.5, help="Required fraction of nodule readers, including ties. Default: 0.5.")
    parser.add_argument("--radius", type=float, default=21, help="Radius in mm. Default: full-training LIDC P99.")
    parser.add_argument("--clip", action="store_true", help="Clip HU to [-1000, 400].")
    parser.add_argument("--patient", help="Extract only this patient ID.")
    parser.add_argument("--limit-scans", type=int, help="Limit the number of CT series for a small run.")
    parser.add_argument("--config", type=Path, help="Default: configuration matching --roi.")
    parser.add_argument("--bin-width", type=float, help="Override the YAML intensity bin width. Default: use the YAML value.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.bin_width is not None and (not np.isfinite(args.bin_width) or args.bin_width <= 0):
        parser.error("--bin-width must be a positive finite number.")
    if args.dimension == 3 and args.roi == "circle":
        parser.error("--dimension 3 requires --roi sphere or segmentation.")
    if args.roi == "sphere" and args.dimension != 3:
        parser.error("--roi sphere requires --dimension 3.")
    if args.slices < 1:
        parser.error("--slices must be positive.")
    if args.slices > 1 and args.dimension != 2:
        parser.error("--slices greater than 1 requires --dimension 2.")
    args.paths = args.paths.expanduser().resolve()
    paths = load_paths(args.paths)
    validate_consensus(args.consensus)
    if args.roi != "segmentation" and args.consensus != 0.5:
        parser.error("--consensus applies only to --roi segmentation.")
    config_name = f"{args.roi}_3d" if args.dimension == 3 else args.roi
    args.config = args.config or paths["radiomics"][config_name]
    overrides = {"binWidth": args.bin_width} if args.bin_width is not None else {}
    extractor = featureextractor.RadiomicsFeatureExtractor(str(args.config), **overrides)
    if args.dimension == 3 and extractor.settings.get("force2D", False):
        parser.error("3D extraction requires a config with force2D: false.")
    if args.roi == "sphere":
        for name in ("shape", "shape2D"):
            extractor.enableFeatureClassByName(name, enabled=False)

    targets_path = paths["lidc"]["targets"]
    folds_path = paths["lidc"]["folds"]
    annotations_path = paths["lidc"]["annotations"]
    nodules = pd.read_csv(targets_path).merge(
        pd.read_csv(folds_path), on="patient_id", how="left", validate="many_to_one"
    )
    if nodules[KEYS + ["fold"]].isna().any().any() or nodules.duplicated(KEYS).any():
        raise ValueError("Targets must have unique nodule IDs and a fold for every patient.")
    if args.patient:
        nodules = nodules[nodules["patient_id"] == args.patient]
    if args.limit_scans is not None:
        if args.limit_scans < 1:
            raise ValueError("--limit-scans must be positive.")
        series = nodules["series_uid"].drop_duplicates().head(args.limit_scans)
        nodules = nodules[nodules["series_uid"].isin(series)]
    if nodules.empty:
        raise ValueError("No nodules match the requested scans.")

    # Use saved centroids and zero-based IDs, avoiding any reordering of clusters.
    annotations = pd.read_csv(annotations_path)
    annotations[["row", "column", "slice"]] = np.stack(
        annotations["centroid"].map(lambda value: np.fromstring(value.strip("[]"), sep=" "))
    )
    centers = annotations.groupby(KEYS)[["row", "column", "slice"]].mean()
    scan_rows = nodules.merge(centers, on=KEYS, how="left", validate="one_to_one")
    if scan_rows[["row", "column", "slice"]].isna().any().any():
        raise ValueError("A nodule is missing its saved centroid.")

    intensity = "clipped" if args.clip else "hu"
    roi_name = f"{args.roi}_r{args.radius:g}" if args.roi in ("circle", "sphere") else f"segmentation_{args.dimension}d_c{100 * args.consensus:g}"
    if args.slices > 1:
        roi_name += f"_{args.slices}slice"
    bin_suffix = f"_bw{args.bin_width:g}" if args.bin_width is not None else ""
    output = args.output or paths["outputs"]["features"] / f"lidc_{roi_name}{bin_suffix}_{intensity}.csv"
    directory = prepare_run_slices(
        output, args.config, args.radius, args.clip,
        [Path(__file__), args.paths, targets_path, folds_path, annotations_path],
        slice_count=args.slices, roi=args.roi, consensus_level=args.consensus,
        bin_width=args.bin_width, dimension=args.dimension,
    )
    radiomics.setVerbosity(logging.ERROR)

    for (patient_id, series_uid), scan_nodules in scan_rows.groupby(["patient_id", "series_uid"], sort=False):
        scan_name = f"{patient_id}_{series_uid}"
        checkpoint = directory / f"{scan_name}.features.csv"
        rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
        completed = {row["nodule_id"] for row in rows}
        pending = scan_nodules[~scan_nodules["nodule_id"].isin(completed)]
        if pending.empty:
            print(f"{patient_id} / {series_uid}: already extracted")
            continue

        print(f"{patient_id}: {len(pending)} nodules, {roi_name}", flush=True)
        errors = []
        try:
            scan = pl.query(pl.Scan).filter(
                pl.Scan.patient_id == patient_id, pl.Scan.series_instance_uid == series_uid
            ).one()
            volume = scan.to_volume(verbose=False)
            image = sitk.GetImageFromArray(volume.transpose(2, 0, 1))
            image.SetSpacing((scan.pixel_spacing, scan.pixel_spacing, scan.slice_spacing))
            del volume
            clusters = scan.cluster_annotations(verbose=False) if args.roi == "segmentation" else None
            saved = annotations[(annotations["patient_id"] == patient_id) & (annotations["series_uid"] == series_uid)]
        except Exception as exc:
            errors = [{**row, "error": str(exc)} for row in pending[KEYS].to_dict("records")]
        else:
            for _, nodule in pending.iterrows():
                row = {"patient_id": patient_id, "series_uid": series_uid, "nodule_id": int(nodule["nodule_id"])}
                try:
                    if args.roi == "segmentation":
                        image_roi, mask, slice_index = create_lidc_segmentation(
                            image, clusters[row["nodule_id"]], saved[saved["nodule_id"] == row["nodule_id"]],
                            args.consensus, 3 if args.slices > 1 else args.dimension,
                        )
                        if args.slices > 1:
                            slices = segmentation_slices(image_roi, mask, args.slices)
                        elif args.dimension == 2:
                            row["slice_index"] = slice_index
                    else:
                        # PyLIDC indices are [row, column, slice], SimpleITK uses [x, y, z].
                        index = tuple(nodule[["column", "row", "slice"]].astype(float))
                        center = image.TransformContinuousIndexToPhysicalPoint(index)
                        if args.slices > 1:
                            slices = create_circle_slices(image, center, args.radius, args.slices)
                        else:
                            create_roi = create_sphere if args.roi == "sphere" else create_circle
                            image_roi, mask = create_roi(image, center, args.radius)
                    if args.slices > 1:
                        rows.extend(extract_slice_features(slices, row, extractor, args.clip))
                    else:
                        row.update(extract_features(image_roi, mask, extractor, args.clip))
                        rows.append(row)
                except Exception as exc:
                    errors.append({**row, "error": str(exc)})
            del image

        if rows:
            save_csv(pd.DataFrame(rows), checkpoint)
        save_errors(errors, directory / f"{scan_name}.errors.csv")

    combine = combine_slice_results if args.slices > 1 else combine_results
    return int(combine(directory, nodules, KEYS, output) > 0)


if __name__ == "__main__":
    raise SystemExit(main())
