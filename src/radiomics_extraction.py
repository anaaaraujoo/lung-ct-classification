"""Shared radiomics extraction and per-scan progress saving."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import radiomics
import SimpleITK as sitk
import yaml


def load_paths(path=None):
    """Read file locations, expanding ~ and resolving relative paths from the repo."""
    root = Path(__file__).resolve().parents[1]
    path = Path(path or root / "configs/paths.yaml").expanduser()
    paths = yaml.safe_load(path.read_text())
    if not isinstance(paths, dict) or not all(isinstance(section, dict) for section in paths.values()):
        raise ValueError("Paths YAML must contain sections of named file locations.")
    for section, locations in paths.items():
        for name, value in locations.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{section}.{name} must contain a non-empty path.")
            location = Path(value).expanduser()
            locations[name] = location if location.is_absolute() else root / location
    return paths


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


def create_sphere(image, center_xyz, radius_mm):
    """Return the full CT and a spherical mask defined in millimetres."""
    if radius_mm <= 0 or not np.isfinite(radius_mm):
        raise ValueError("Radius must be a positive finite number.")
    if not np.isfinite(center_xyz).all():
        raise ValueError("Nodule centroid contains missing or non-finite coordinates.")

    center = np.array(image.TransformPhysicalPointToContinuousIndex(
        tuple(float(value) for value in center_xyz)
    ))
    size = np.array(image.GetSize())
    if np.any(center < 0) or np.any(center > size - 1):
        raise ValueError("Nodule centroid is outside the scan.")
    spacing = np.array(image.GetSpacing())
    # Compute distances only near the centre, retaining a full-volume mask.
    lower = np.maximum(np.floor(center - radius_mm / spacing), 0).astype(int)
    upper = np.minimum(np.ceil(center + radius_mm / spacing) + 1, size).astype(int)
    z, y, x = np.ogrid[lower[2]:upper[2], lower[1]:upper[1], lower[0]:upper[0]]
    distance_squared = (
        ((x - center[0]) * spacing[0]) ** 2
        + ((y - center[1]) * spacing[1]) ** 2
        + ((z - center[2]) * spacing[2]) ** 2
    )
    values = np.zeros(tuple(reversed(image.GetSize())), dtype=np.uint8)
    values[lower[2]:upper[2], lower[1]:upper[1], lower[0]:upper[0]] = distance_squared <= radius_mm ** 2
    mask = sitk.GetImageFromArray(values)
    mask.CopyInformation(image)
    return image, mask


def check_mask_geometry(image, mask):
    """Reject misaligned masks rather than silently changing their geometry."""
    if image.GetDimension() != 3 or mask.GetDimension() != 3 or image.GetSize() != mask.GetSize():
        raise ValueError("Mask and CT must have the same 3D dimensions.")
    for attribute in ("GetOrigin", "GetSpacing", "GetDirection"):
        if not np.allclose(getattr(image, attribute)(), getattr(mask, attribute)(), atol=1e-6, rtol=0):
            raise ValueError(f"Mask and CT differ in {attribute[3:].lower()}.")


def validate_consensus(level):
    if not np.isfinite(level) or not 0 < level <= 1:
        raise ValueError("Consensus must be a finite fraction greater than 0 and at most 1.")
    return level


def combine_reader_masks(masks, consensus_level=0.5):
    """Include voxels meeting the requested fraction of associated readers."""
    validate_consensus(consensus_level)
    votes, count = None, 0
    for mask in masks:
        mask = np.asarray(mask)
        if mask.ndim != 3 or not np.isin(mask, [0, 1]).all() or not mask.any():
            raise ValueError("Every reader must have a non-empty binary 3D nodule mask.")
        if votes is None:
            votes = np.zeros(mask.shape, dtype=np.uint16)
        if mask.shape != votes.shape:
            raise ValueError("Reader masks must share the same voxel grid.")
        votes += mask.astype(bool, copy=False)
        count += 1
    if count == 0:
        raise ValueError("No reader masks found for this nodule.")
    # With two readers, 50% includes voxels marked by either reader.
    required = int(np.ceil(count * consensus_level))
    combined = (votes >= required).astype(np.uint8)
    if not combined.any():
        raise ValueError(f"The {100 * consensus_level:g}% consensus mask is empty.")
    return combined


def largest_mask_slice(image, mask):
    """Return one axial slice, its binary mask and the original slice index."""
    check_mask_geometry(image, mask)
    values = sitk.GetArrayViewFromImage(mask)
    if not np.isin(values, [0, 1]).all():
        raise ValueError("The nodule mask must be binary.")
    areas = np.count_nonzero(values, axis=(1, 2))
    if not areas.any():
        raise ValueError("The nodule mask is empty.")
    # np.argmax chooses the first slice if several have the same largest area.
    index = int(np.argmax(areas))
    return image[:, :, index:index + 1], mask[:, :, index:index + 1], index


def segmentation_roi(image, mask, dimension=2):
    """Select the largest axial slice or retain the full consensus volume."""
    if dimension == 2:
        return largest_mask_slice(image, mask)
    if dimension != 3:
        raise ValueError("Segmentation dimension must be 2 or 3.")
    check_mask_geometry(image, mask)
    values = sitk.GetArrayViewFromImage(mask)
    if not np.isin(values, [0, 1]).all() or not values.any():
        raise ValueError("The nodule mask must be non-empty and binary.")
    return image, mask, None


def extract_features(image, mask, extractor, clip=False):
    # Preserve fractional HU values when resampling.
    image = sitk.Cast(image, sitk.sitkFloat32)
    if clip:
        image = sitk.Clamp(image, lowerBound=-1000, upperBound=400)
    return {name: float(value) for name, value in extractor.execute(image, mask).items()
            if not name.startswith("diagnostics_")}


def file_checksum(path):
    """Hash large inputs, such as masks.zip, without loading them into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_run(output, config, radius_mm, clip, sources, roi="circle", consensus_level=0.5, bin_width=None, dimension=2):
    """Resume only when settings, source tables and extraction code match."""
    if roi not in ("circle", "sphere", "segmentation"):
        raise ValueError("ROI must be circle, sphere or segmentation.")
    if dimension not in (2, 3) or (dimension == 3 and roi == "circle") or (dimension == 2 and roi == "sphere"):
        raise ValueError("Use dimension 2 for circles, 3 for spheres, or either for segmentation.")
    if roi in ("circle", "sphere") and (radius_mm is None or radius_mm <= 0 or not np.isfinite(radius_mm)):
        raise ValueError("Radius must be a positive finite number.")
    if bin_width is not None and (not np.isfinite(bin_width) or bin_width <= 0):
        raise ValueError("Bin width must be a positive finite number.")
    directory = output.with_suffix(".scans")
    settings_path = directory / "settings.json"
    settings = {
        "roi": roi,
        "radius_mm": radius_mm if roi in ("circle", "sphere") else None,
        "clip": clip,
        "pyradiomics_version": radiomics.__version__,
        "config": config.read_text(),
        "sources": {
            path.name: file_checksum(path)
            for path in [Path(__file__), *sources]
        },
    }
    if roi == "segmentation":
        validate_consensus(consensus_level)
        settings.update(consensus=consensus_level,
                        slice_selection="largest_mask_area" if dimension == 2 else "full_volume")
    if dimension == 3:
        settings["dimension"] = dimension
    if bin_width is not None:
        settings["bin_width"] = bin_width
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
