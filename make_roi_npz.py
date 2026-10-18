#!/usr/bin/env python3
"""
Create 3-D ROI NPZ files from LNDb data (.mhd/.raw) + allNods.csv.

For every nodule in allNods.csv it:
  1. Loads the matching LNDb-####.mhd/.raw CT volume (int16 HU).
  2. Converts the nodule world centre (x,y,z in mm) to a voxel index using the
     volume's Offset and ElementSpacing.
  3. Crops a cube of side --roi-mm (16 or 32 mm) centred on the nodule, clamping
     to the volume boundary and padding back to the full cube size with the
     volume median HU (image) / False (mask).
  4. Writes one NPZ per nodule under <out-dir>/LNDb-####/LNDb-####_findingNNN.npz
     with keys understood by build_dataset_2d.py:
         image        int16  (zx, zy, zz)  HU, axis order = (x, y, z)
         mask         bool   same shape    True inside the ROI
         spacing      float32 (3,)         voxel spacing (x, y, z) mm
         n_annotators int32  scalar        default 1
         malignancy   float32 (0,)         empty (no label in allNods.csv)
         pad_xy, pad_z int32               voxels clamped per axis (info only)
         plus patient_id, finding_id, centre_vox, centre_world, roi_mm, source

The output directory layout matches build_dataset_2d.py's default input-dir
("reconstructed_3d"), so the next step is:

    python build_dataset_2d.py --input-dir reconstructed_3d --out dataset2d
    python extract_features_manus.py --data dataset2d --out dataset2d_rin \
        --weights /path/to/resnet50_torch.pt
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# MetaImage (.mhd / .raw) reading
# ---------------------------------------------------------------------------

def parse_mhd(mhd_path):
    """Parse a MetaImage header into a dict of key -> str."""
    header = {}
    with open(mhd_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            header[key.strip()] = val.strip()
    return header


def load_volume(mhd_path):
    """
    Load an LNDb .mhd/.raw pair.

    Returns
    -------
    image : np.ndarray (int16), shape (nx, ny, nz)  axis order = (x, y, z)
    spacing : np.ndarray (float32, 3)  (sx, sy, sz) mm
    offset : np.ndarray (float32, 3)   (ox, oy, oz) mm  world coord of voxel (0,0,0)
    dims : tuple (nx, ny, nz)
    """
    hdr = parse_mhd(mhd_path)
    dims = tuple(int(v) for v in hdr["DimSize"].split())
    spacing = np.array([float(v) for v in hdr["ElementSpacing"].split()], dtype=np.float32)
    offset = np.array([float(v) for v in hdr["Offset"].split()], dtype=np.float32)
    raw_path = Path(mhd_path).with_name(hdr["ElementDataFile"])

    ndims = int(hdr.get("NDims", len(dims)))
    assert ndims == len(dims) == 3, "expected a 3-D volume"
    assert spacing.size == 3 and offset.size == 3

    # MET_SHORT = 16-bit signed int; BinaryDataByteOrderMSB=False -> little-endian.
    dtype = "<i2"
    raw = np.fromfile(raw_path, dtype=dtype)
    expected = int(np.prod(dims))
    if raw.size != expected:
        raise ValueError(
            f"{raw_path}: expected {expected} voxels, found {raw.size}")
    image = raw.astype(np.int16).reshape(dims)   # (x, y, z), C-order
    return image, spacing, offset, dims


# ---------------------------------------------------------------------------
# ROI extraction
# ---------------------------------------------------------------------------

def world_to_voxel(world, offset, spacing):
    """Continuous voxel coordinates for a world point (mm)."""
    return (np.asarray(world, dtype=np.float64) - offset) / spacing


def sphere_mask(shape, centre_vox, diameter_mm, spacing):
    """Binary ellipsoid mask (True inside) centred at centre_vox."""
    zz, yy, xx = np.indices(shape)  # same ordering as image axes (x,y,z)
    dx = (xx - centre_vox[0]) * spacing[0]
    dy = (yy - centre_vox[1]) * spacing[1]
    dz = (zz - centre_vox[2]) * spacing[2]
    rr = np.sqrt((dx / (diameter_mm / 2.0)) ** 2
                 + (dy / (diameter_mm / 2.0)) ** 2
                 + (dz / (diameter_mm / 2.0)) ** 2)
    return rr <= 1.0


def extract_cube(image, dims, centre_vox, size_vox, fill_hu):
    """
    Extract `size_vox` voxels centred on `centre_vox`, clamping to the volume
    and padding back to the full cube size with `fill_hu`.

    Returns (cube, pad_xy, pad_z) where pad_* are total voxels clamped per axis.
    """
    lo = centre_vox - size_vox // 2
    hi = lo + size_vox - 1                     # inclusive, exactly size_vox voxels

    # Clamp to valid voxel range; record how much was clamped per axis.
    lo_clamped = np.maximum(lo, 0)
    hi_clamped = np.minimum(hi, np.array(dims) - 1)
    pad_lo = np.maximum(-lo, 0)                # voxels missing below the volume
    pad_hi = np.maximum(hi - (np.array(dims) - 1), 0)  # voxels missing above
    pad_total = pad_lo + pad_hi

    slab = tuple(slice(l, h + 1) for l, h in zip(lo_clamped, hi_clamped))
    crop = image[slab]

    # Pad back to the requested cube size so every ROI NPZ has an identical shape.
    pad_width = [(int(pl), int(ph)) for pl, ph in zip(pad_lo, pad_hi)]
    cube = np.pad(crop, pad_width, mode="constant", constant_values=fill_hu)

    return cube.astype(np.int16), pad_total.astype(np.int32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_volume_for_patient(data_dir, lndb_id):
    """Return path to data/LNDb-####.mhd for the given LNDbID, or None."""
    mhd = Path(data_dir) / f"LNDb-{int(lndb_id):04d}.mhd"
    return mhd if mhd.is_file() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data",
                    help="Directory containing LNDb-####.mhd/.raw and allNods.csv")
    ap.add_argument("--csv", default=None,
                    help="Nodule CSV; defaults to <data-dir>/allNods.csv")
    ap.add_argument("--out-dir", default="reconstructed_3d",
                    help="Output NPZ root (matches build_dataset_2d default input-dir)")
    ap.add_argument("--roi-mm", type=float, choices=[16.0, 32.0], default=32.0,
                    help="Side length of the cube ROI in mm (16 or 32)")
    ap.add_argument("--mask-mode", choices=["full", "sphere"], default="full",
                    help="Mask: 'full' = whole cube (default), 'sphere' = nodule-sized ellipsoid")
    ap.add_argument("--nodule-mm", type=float, default=10.0,
                    help="Nodule diameter (mm) for --mask-mode sphere")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N findings (smoke testing)")
    ap.add_argument("--patient", type=str, default=None,
                    help="Only process findings whose LNDbID matches this value")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    csv_path = Path(args.csv) if args.csv else data_dir / "allNods.csv"
    if not csv_path.is_file():
        raise SystemExit(f"Nodule CSV not found: {csv_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read findings.
    findings = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if not row["Nodule"]:
                continue
            lndb_id = int(row["LNDbID"])
            if args.patient is not None and str(lndb_id) != args.patient:
                continue
            findings.append({
                "lndb_id": lndb_id,
                "finding_id": int(row["FindingID"]),
                "world": np.array([float(row["x"]), float(row["y"]), float(row["z"])]),
            })
    if args.limit is not None:
        findings = findings[:args.limit]

    # Volume cache: patient id -> (image, spacing, offset, dims, mhd_path)
    volumes = {}

    created = []
    skipped = 0
    for finding in findings:
        pid = finding["lndb_id"]
        if pid not in volumes:
            mhd = find_volume_for_patient(data_dir, pid)
            if mhd is None:
                print(f"[skip] no volume for patient LNDb-{pid:04d} "
                      f"(finding {finding['finding_id']})", flush=True)
                skipped += 1
                continue
            try:
                volumes[pid] = load_volume(mhd)
            except Exception as e:  # noqa: BLE001
                print(f"[skip] failed to load {mhd}: {e}", flush=True)
                skipped += 1
                continue
        image, spacing, offset, dims = volumes[pid]

        centre_vox_f = world_to_voxel(finding["world"], offset, spacing)
        centre_vox = np.round(centre_vox_f).astype(np.int64)

        # Skip nodules whose centre lies entirely outside the volume.
        if np.any(centre_vox < 0) or np.any(centre_vox >= np.array(dims)):
            print(f"[skip] patient LNDb-{pid:04d} finding {finding['finding_id']}: "
                  f"centre voxel {centre_vox} outside volume {dims}", flush=True)
            skipped += 1
            continue

        # Voxel size of the requested cube (physical size ~= roi_mm on each side).
        size_vox = np.maximum(1, np.round(args.roi_mm / spacing).astype(np.int64))

        fill_hu = float(np.median(image))
        cube, pad_total = extract_cube(image, dims, centre_vox, size_vox, fill_hu)

        if args.mask_mode == "full":
            mask = np.ones(cube.shape, dtype=bool)
        else:
            # Nodule position within the final (padded) cube.
            lo = centre_vox - size_vox // 2
            nodule_in_cube = centre_vox - np.maximum(lo, 0) + np.maximum(-lo, 0)
            mask = sphere_mask(cube.shape, nodule_in_cube, args.nodule_mm, spacing)

        patient_id = f"LNDb-{pid:04d}"
        nodule_id = f"{patient_id}_finding{int(finding['finding_id']):03d}"

        patient_dir = out_dir / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        npz_path = patient_dir / f"{nodule_id}.npz"

        np.savez_compressed(
            npz_path,
            image=cube,
            mask=mask,
            spacing=spacing.astype(np.float32),
            n_annotators=np.array(1, dtype=np.int32),
            malignancy=np.empty(0, dtype=np.float32),
            pad_xy=pad_total[:2],          # [x_clamped, y_clamped]
            pad_z=pad_total[2:3],          # [z_clamped]
            patient_id=patient_id,
            nodule_id=nodule_id,
            finding_id=np.array(int(finding["finding_id"]), dtype=np.int64),
            centre_vox=centre_vox.astype(np.float64),
            centre_world=finding["world"].astype(np.float64),
            roi_mm=np.array(args.roi_mm, dtype=np.float64),
            source_mhd=f"LNDb-{pid:04d}.mhd",
        )
        created.append({
            "npz": str(npz_path),
            "patient_id": patient_id,
            "nodule_id": nodule_id,
            "centre_world": finding["world"].tolist(),
            "centre_vox": centre_vox.tolist(),
            "shape": cube.shape,
            "roi_mm": args.roi_mm,
            "source": f"LNDb-{pid:04d}.mhd",
        })
        print(f"[ok] {npz_path}  shape={cube.shape} "
              f"centre_vox={centre_vox.tolist()} pad_vox={pad_total.tolist()}",
              flush=True)

    # Index of everything written (handy for auditing / downstream pairing).
    index_path = out_dir / "roi_index.csv"
    with open(index_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["npz", "patient_id", "nodule_id", "centre_x", "centre_y",
                    "centre_z", "centre_vox_x", "centre_vox_y", "centre_vox_z",
                    "shape_x", "shape_y", "shape_z", "roi_mm", "source"])
        for c in created:
            cx, cy, cz = c["centre_vox"]
            sx, sy, sz = c["shape"]
            wx, wy, wz = c["centre_world"]
            w.writerow([c["npz"], c["patient_id"], c["nodule_id"], wx, wy, wz,
                        cx, cy, cz, sx, sy, sz, c["roi_mm"], c["source"]])

    print(f"\nDone: {len(created)} ROI NPZs written to {out_dir}; "
          f"{skipped} findings skipped. Index: {index_path}")
    print("\nNext step:\n"
          f"  python build_dataset_2d.py --input-dir {out_dir} --out dataset2d\n"
          "  python extract_features_manus.py --data dataset2d --out dataset2d_rin "
          "--weights /path/to/resnet50_torch.pt")


if __name__ == "__main__":
    main()
