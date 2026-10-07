"""Shared circular radiomics extraction and per-scan progress saving."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import radiomics
import SimpleITK as sitk


def create_circle(image, center_xyz, radius_mm):
    """Return one axial slice and a circle defined in millimetres."""
    if radius_mm <= 0 or not np.isfinite(radius_mm):
        raise ValueError("Radius must be a positive finite number.")
    if not np.isfinite(center_xyz).all():
        raise ValueError("Nodule centroid contains missing or non-finite coordinates.")

    # SimpleITK accounts for origin, spacing and image direction.
    x, y, z = image.TransformPhysicalPointToContinuousIndex(
        tuple(float(value) for value in center_xyz)
    )
    if any(value < 0 or value > size - 1 for value, size in zip((x, y, z), image.GetSize())):
        raise ValueError("Nodule centroid is outside the scan.")
    slice_index = int(round(z))
    image_slice = image[:, :, slice_index:slice_index + 1]

    spacing_x, spacing_y, _ = image.GetSpacing()
    rows, columns = np.ogrid[:image.GetHeight(), :image.GetWidth()]
    distance_squared = ((columns - x) * spacing_x) ** 2 + ((rows - y) * spacing_y) ** 2
    circle = (distance_squared <= radius_mm ** 2).astype(np.uint8)
    mask = sitk.GetImageFromArray(circle[np.newaxis])
    mask.CopyInformation(image_slice)
    return image_slice, mask


def extract_features(image, mask, extractor, clip=False):
    # Preserve fractional HU values when resampling.
    image = sitk.Cast(image, sitk.sitkFloat32)
    if clip:
        image = sitk.Clamp(image, lowerBound=-1000, upperBound=400)
    return {name: float(value) for name, value in extractor.execute(image, mask).items()
            if not name.startswith("diagnostics_")}


def prepare_run(output, config, radius_mm, clip, sources):
    """Resume only when settings, source tables and extraction code match."""
    if radius_mm <= 0 or not np.isfinite(radius_mm):
        raise ValueError("Radius must be a positive finite number.")
    directory = output.with_suffix(".scans")
    settings_path = directory / "settings.json"
    settings = {
        "radius_mm": radius_mm,
        "clip": clip,
        "pyradiomics_version": radiomics.__version__,
        "config": config.read_text(),
        "sources": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), *sources]
        },
    }
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise ValueError("Settings or inputs changed. Choose a different --output file.")
    else:
        if output.exists() or (directory.exists() and any(directory.iterdir())):
            raise ValueError("Output already exists without matching settings. Choose another --output.")
        directory.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    return directory


def save_csv(table, path):
    """Replace a checkpoint only after its CSV has been fully written."""
    temporary = path.with_name(path.name + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def save_errors(errors, path):
    """Keep an error CSV only while a scan has failed nodules."""
    if errors:
        save_csv(pd.DataFrame(errors), path)
        print(f"{len(errors)} nodules failed. Details: {path}", flush=True)
    else:
        path.unlink(missing_ok=True)


def combine_results(directory, nodules, keys, output):
    """Join successful extractions to input rows, retaining their order."""
    paths = sorted(directory.glob("*.features.csv"))
    if paths:
        features = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        result = nodules.merge(features, on=keys, how="inner", validate="one_to_one")
    else:
        result = nodules.head(0)
    save_csv(result, output)
    missing = len(nodules) - len(result)
    print(f"Saved {len(result)}/{len(nodules)} nodules to {output}")
    if missing:
        print(f"{missing} nodules failed. See *.errors.csv in {directory}, then rerun to retry.")
    return missing
