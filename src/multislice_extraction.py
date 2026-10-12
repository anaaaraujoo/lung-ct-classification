"""Axial slice stacks, atomic nodule outputs and separate extraction checkpoints."""
import json
from pathlib import Path

import pandas as pd
import SimpleITK as sitk

from .radiomics_extraction import create_circle, extract_features, largest_mask_slice, prepare_run, save_csv


def slice_stack(main_index, count, depth):
    """Centre the stack on the main slice, retaining only indices inside the CT."""
    if count < 1:
        raise ValueError("The slice stack needs at least one slice.")
    start = main_index - (count - 1) // 2
    return [index for index in range(start, start + count) if 0 <= index < depth]


def create_circle_slices(image, center_xyz, radius_mm, count):
    """Repeat the same physical circle on axial slices around the saved centroid."""
    _, circle = create_circle(image, center_xyz, radius_mm)
    index = image.TransformPhysicalPointToContinuousIndex(tuple(float(value) for value in center_xyz))
    slices = []
    for slice_index in slice_stack(int(round(index[2])), count, image.GetDepth()):
        image_slice = image[:, :, slice_index:slice_index + 1]
        mask = sitk.Image(circle)
        mask.CopyInformation(image_slice)
        slices.append((slice_index, image_slice, mask))
    return slices


def segmentation_slices(image, mask, count):
    """Select non-empty consensus slices around the largest-area axial slice."""
    _, _, main_index = largest_mask_slice(image, mask)
    slices = []
    for slice_index in slice_stack(main_index, count, image.GetDepth()):
        mask_slice = mask[:, :, slice_index:slice_index + 1]
        if sitk.GetArrayViewFromImage(mask_slice).any():
            slices.append((slice_index, image[:, :, slice_index:slice_index + 1], mask_slice))
    return slices


def extract_slice_features(slices, row, extractor, clip=False):
    """Return all slice rows together, discarding the nodule if any slice fails."""
    rows = []
    for number, (slice_index, image, mask) in enumerate(slices):
        slice_row = {**row, "slice_in_nodule": number, "slice_index": slice_index}
        slice_row.update(extract_features(image, mask, extractor, clip))
        rows.append(slice_row)
    return rows


def prepare_run_slices(output, config, radius_mm, clip, sources, slice_count=1, **settings):
    """Reject mixing slice counts or single-slice and multi-slice checkpoints."""
    if slice_count < 1:
        raise ValueError("The slice stack needs at least one slice.")
    directory = output.with_suffix(".scans")
    slices_path = directory / "slices.json"
    slices = {"slice_count": slice_count}
    if slices_path.exists():
        if json.loads(slices_path.read_text()) != slices:
            raise ValueError("Slice count changed. Choose a different --output file.")
    elif slice_count > 1 and directory.exists() and any(directory.iterdir()):
        raise ValueError("Output has no matching slice settings. Choose a different --output file.")
    if slice_count > 1:
        sources = [Path(__file__), *sources]
    directory = prepare_run(output, config, radius_mm, clip, sources, **settings)
    if slice_count > 1 and not slices_path.exists():
        slices_path.write_text(json.dumps(slices, indent=2) + "\n")
    return directory


def combine_results(directory, nodules, keys, output):
    """Preserve each nodule's targets and folds on every extracted slice row."""
    paths = sorted(directory.glob("*.features.csv"))
    if paths:
        features = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        result = nodules.merge(features, on=keys, how="inner", validate="one_to_many")
    else:
        result = nodules.head(0)
    save_csv(result, output)
    extracted = len(result[keys].drop_duplicates())
    missing = len(nodules) - extracted
    print(f"Saved {len(result)} slices from {extracted}/{len(nodules)} nodules to {output}")
    if missing:
        print(f"{missing} nodules failed. See *.errors.csv in {directory}, then rerun to retry.")
    return missing
