"""Build a fixed-scale, per-slice HU dataset from reconstructed 3-D ROI NPZs.

Images are the CNN input. Annotator-level labels remain vectors in metadata;
this stage does not average, binarise, or stratify labels. The frozen CNN stage
can export feature-only, one-slice-per-NPZ records for downstream MIL.
"""
from __future__ import annotations
import argparse
import json
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.ndimage as ndi

TARGET_SIZE = 224
TARGET_MM = 32.0
DEFAULT_INPUT = "reconstructed_3d"
NAME_RE = re.compile(r"(?P<patient>LIDC-IDRI-\d+).*?nodule(?P<nodule>\d+)", re.I)
LABEL_KEYS = ("malignancy", "spiculation", "texture")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def discover_npzs(input_dir):
    if not input_dir.is_dir():
        # Accept the common typo if that is the actual directory name.
        alternate = input_dir.with_name("reconstrucuted_3d")
        if input_dir.name == DEFAULT_INPUT and alternate.is_dir():
            input_dir = alternate
        else:
            raise FileNotFoundError(f"ROI folder does not exist: {input_dir}")
    files = sorted(p for p in input_dir.rglob("*.npz") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No .npz files found under {input_dir}")
    return input_dir, files


def parse_ids(path, root):
    match = NAME_RE.search(path.stem)
    if match:
        patient = match.group("patient")
        nodule = f"{patient}_nodule{int(match.group('nodule')):02d}"
        return patient, nodule
    patient = path.parent.name if path.parent != root else "unknown_patient"
    return patient, path.stem


def read_label(d, key, n_annotators):
    if key not in d.files:
        return np.empty(0, dtype=np.float32)
    a = np.asarray(d[key]).reshape(-1).copy()
    if a.size not in (0, n_annotators):
        raise ValueError(f"{key} has {a.size} entries, n_annotators={n_annotators}")
    return a


def inspect_roi(path, patient_id, nodule_id, expand, keep_empty):
    try:
        with np.load(path, allow_pickle=False) as d:
            required = {"image", "mask", "spacing", "n_annotators", "malignancy"}
            missing = required.difference(d.files)
            if missing:
                raise KeyError(f"missing keys {sorted(missing)}")
            shape = d["image"].shape
            mask = np.asarray(d["mask"], dtype=bool)
            if len(shape) != 3 or mask.shape != shape:
                raise ValueError(f"image/mask must share a 3-D shape: {shape} vs {mask.shape}")
            spacing = np.asarray(d["spacing"], dtype=np.float64).reshape(-1)
            if spacing.size != 3 or not np.isfinite(spacing).all() or (spacing <= 0).any():
                raise ValueError(f"invalid spacing {spacing}")
            n_ann = int(np.asarray(d["n_annotators"]).reshape(-1)[0])
            if n_ann < 1:
                raise ValueError(f"invalid n_annotators={n_ann}")
            labels = {key: read_label(d, key, n_ann) for key in LABEL_KEYS}
            pad_xy = np.asarray(d["pad_xy"]).reshape(-1).copy() if "pad_xy" in d.files else np.empty(0, dtype=np.int32)
            pad_z = np.asarray(d["pad_z"]).reshape(-1).copy() if "pad_z" in d.files else np.empty(0, dtype=np.int32)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    if mask.size == 0:
        return None, "empty_volume"

    nx, ny, nz = shape
    counts = mask.sum(axis=(0, 1), dtype=np.int64)
    mask_zs = np.flatnonzero(counts > 0)
    mask_empty = len(mask_zs) == 0
    if mask_empty:
        if not keep_empty:
            return None, "empty_mask"
        center = ((nx - 1) / 2., (ny - 1) / 2.)
        z_lo, z_hi = (2, nz - 3) if nz > 4 else (0, nz - 1)
        core_zs = np.arange(z_lo, z_hi + 1)
        extent_mm, mask_volume = float("nan"), 0.
    else:
        xs, ys, _ = np.nonzero(mask)
        center = (float(xs.mean()), float(ys.mean()))
        z_lo, z_hi = int(mask_zs[0]), int(mask_zs[-1])
        core_zs = mask_zs
        extent_mm = float(max((xs.max()-xs.min()+1)*spacing[0],
                              (ys.max()-ys.min()+1)*spacing[1]))
        mask_volume = float(mask.sum() * np.prod(spacing))
    z0, z1 = max(0, z_lo-expand), min(nz-1, z_hi+expand)
    return dict(path=str(path), patient_id=patient_id, nodule_id=nodule_id,
                n_annotators=n_ann, labels=labels, pad_xy=pad_xy, pad_z=pad_z,
                spacing=spacing, shape=shape,
                center=center, z_list=list(range(z0, z1+1)), core_zs=core_zs,
                counts=counts, mask_empty=mask_empty, extent_mm=extent_mm,
                mask_volume_mm3=mask_volume, n_mask_slices=int(len(mask_zs))), None


def render_slice(sl, spacing, center, fill, size, target_mm):
    mm_px = target_mm / size
    ax, ay = mm_px / spacing[0], mm_px / spacing[1]
    c = (size - 1) / 2.
    off = (center[0]-ax*c, center[1]-ay*c)
    out = ndi.affine_transform(sl.astype(np.float32), (ax, ay), offset=off,
                               output_shape=(size, size), order=1,
                               mode="constant", cval=float(fill), prefilter=False)
    i = np.arange(size)
    vx = (off[0]+ax*i >= 0) & (off[0]+ax*i <= sl.shape[0]-1)
    vy = (off[1]+ay*i >= 0) & (off[1]+ay*i <= sl.shape[1]-1)
    return np.rint(out).astype(np.int16), 1.0-float(vx.mean())*float(vy.mean())


def build(plans, out_prefix, size, target_mm):
    total = sum(len(p["z_list"]) for p in plans)
    if not plans or not total:
        raise RuntimeError("No usable slices. Check input folder and --keep-empty.")
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    x_path = Path(f"{prefix}_X.npy")
    X = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.int16,
                                  shape=(total, size, size))
    pids = sorted({p["patient_id"] for p in plans})
    patient_codes = {pid: i for i, pid in enumerate(pids)}
    nodule_ids = np.empty(total, dtype=np.int64)
    patient_idx = np.empty(total, dtype=np.int64)
    meta, nodule_rows, row = [], [], 0
    for nod_i, p in enumerate(plans):
        with np.load(p["path"], allow_pickle=False) as d:
            image = d["image"]
        fill = float(np.median(image))
        pad_fracs = []
        for slice_order, z in enumerate(p["z_list"]):
            img, pad_frac = render_slice(image[:, :, z], p["spacing"], p["center"],
                                         fill, size, target_mm)
            X[row] = img
            nodule_ids[row] = nod_i
            patient_idx[row] = patient_codes[p["patient_id"]]
            pad_fracs.append(pad_frac)
            meta.append({
                "row_idx": row, "source_roi": p["path"],
                "patient_id": p["patient_id"], "patient_idx": patient_codes[p["patient_id"]],
                "nodule_id": p["nodule_id"], "nodule_idx": nod_i,
                "slice_idx": int(z), "slice_in_nodule": slice_order,
                "n_slices_in_nodule": len(p["z_list"]),
                "has_mask": bool(p["counts"][z] > 0), "mask_empty": p["mask_empty"],
                "z_dist": int(np.min(np.abs(p["core_zs"]-z))),
                "mask_area_mm2": float(p["counts"][z]*p["spacing"][0]*p["spacing"][1]),
                "pad_fraction": pad_frac, "spacing": p["spacing"].tolist(),
                "n_annotators": p["n_annotators"],
                "pad_xy": p["pad_xy"].tolist(), "pad_z": p["pad_z"].tolist(),
                **{k: v.tolist() for k, v in p["labels"].items()},
            })
            row += 1
        nodule_rows.append({
            "nodule_idx": nod_i, "nodule_id": p["nodule_id"],
            "patient_id": p["patient_id"], "patient_idx": patient_codes[p["patient_id"]],
            "source_roi": p["path"], "n_annotators": p["n_annotators"],
            "pad_xy": json.dumps(p["pad_xy"].tolist()),
            "pad_z": json.dumps(p["pad_z"].tolist()),
            **{k: json.dumps(v.tolist()) for k, v in p["labels"].items()},
            "spacing_x": p["spacing"][0], "spacing_y": p["spacing"][1],
            "spacing_z": p["spacing"][2], "vol_nx": p["shape"][0],
            "vol_ny": p["shape"][1], "vol_nz": p["shape"][2],
            "mask_volume_mm3": p["mask_volume_mm3"], "mask_extent_mm": p["extent_mm"],
            "mask_empty": p["mask_empty"], "n_mask_slices": p["n_mask_slices"],
            "n_slices": len(p["z_list"]), "mean_pad_fraction": float(np.mean(pad_fracs)),
        })
    X.flush()
    np.save(f"{prefix}_nodule_ids.npy", nodule_ids)
    np.save(f"{prefix}_patient_idx.npy", patient_idx)
    pd.DataFrame(meta).to_pickle(f"{prefix}_meta.pkl")
    pd.DataFrame(nodule_rows).to_csv(f"{prefix}_nodules.csv", index=False)
    log(f"Wrote {x_path} {X.shape} int16; {len(plans)} nodules, {len(pids)} patients")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", help="Root containing the ROI directory")
    ap.add_argument("--input-dir", default=DEFAULT_INPUT,
                    help="ROI directory, relative to --root by default")
    ap.add_argument("--out", default="dataset2d", help="Output prefix")
    ap.add_argument("--size", type=int, default=TARGET_SIZE)
    ap.add_argument("--target-mm", type=float, default=TARGET_MM)
    ap.add_argument("--expand", type=int, default=2, help="Context slices around mask extent")
    ap.add_argument("--keep-empty", action="store_true", help="Keep empty consensus masks")
    ap.add_argument("--limit", type=int, default=None, help="Limit files for smoke testing")
    args = ap.parse_args()
    if args.size <= 0 or args.target_mm <= 0 or args.expand < 0:
        ap.error("size and target-mm must be positive; expand must be nonnegative")
    root = Path(args.root).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser()
    if not input_dir.is_absolute():
        input_dir = root / input_dir
    input_dir, files = discover_npzs(input_dir)
    if args.limit is not None:
        files = files[:args.limit]
    plans, drops = [], {}
    for path in files:
        patient_id, nodule_id = parse_ids(path, input_dir)
        plan, reason = inspect_roi(path, patient_id, nodule_id, args.expand, args.keep_empty)
        if plan is None:
            drops[reason] = drops.get(reason, 0)+1
        else:
            plans.append(plan)
    keys = [p["nodule_id"] for p in plans]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate parsed nodule IDs found; ensure each ROI NPZ is unique and filenames identify nodules")
    log(f"Found {len(files)} NPZs under {input_dir}; kept {len(plans)}; dropped {drops}")
    build(plans, args.out, args.size, args.target_mm)


if __name__ == "__main__":
    main()