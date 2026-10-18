import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

try:                                   # keeps this module importable without torch
    from torch.utils.data import Dataset as _DatasetBase
except ImportError:                    # pragma: no cover
    _DatasetBase = object

HF_REPO, HF_FILE = "Lab-Rasool/RadImageNet", "ResNet50.pt"   # unofficial mirror
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_model_input(hu, lo, hi, norm):
    """int16 HU (S, S) -> float32 (3, S, S).  Cast BEFORE clipping (int16 math is slow/lossy)."""
    x = np.clip(hu.astype(np.float32), lo, hi)
    x = (x - lo) / (hi - lo)                              # [0, 1]
    if norm == "radimagenet":
        x = np.repeat((x * 2.0 - 1.0)[None], 3, axis=0)   # [-1, 1]
    elif norm == "imagenet":
        x = np.repeat(x[None], 3, axis=0)
        x = (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    else:
        raise ValueError(f"unknown --norm {norm!r}")
    return np.ascontiguousarray(x, dtype=np.float32)


class SliceDataset(_DatasetBase):
    """Opens the memmap lazily inside each worker (no 2 GB copy per worker)."""

    def __init__(self, x_path, n, lo, hi, norm):
        self.x_path, self.n, self.lo, self.hi, self.norm = x_path, n, lo, hi, norm
        self._X = None

    def __len__(self):
        return self.n

    def __getitem__(self, j):
        import torch
        if self._X is None:
            self._X = np.load(self.x_path, mmap_mode="r")
        return torch.from_numpy(to_model_input(self._X[j], self.lo, self.hi, self.norm)), j

# nn.Sequential(*resnet.children()[:9]) index -> torchvision attribute name.
# 2 = relu, 3 = maxpool, 8 = avgpool carry no parameters.
_SEQ_TO_NAME = {"0": "conv1", "1": "bn1", "4": "layer1",
                "5": "layer2", "6": "layer3", "7": "layer4"}
_HEAD_PREFIXES = ("fc.", "classifier.", "linear.")


def normalise_keys(state):
    """Map any common layout to torchvision ResNet-50 key names; drop the head.

    Accepts: 'backbone.<idx>.*' (official RadImageNet), '<idx>.*' (bare
    Sequential), 'conv1.*/layerN.*' (full torchvision model), each optionally
    with a DataParallel 'module.' prefix.
    """
    out = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith(_HEAD_PREFIXES):
            continue
        if k.startswith("backbone."):
            k = k[len("backbone."):]
        head, _, tail = k.partition(".")
        if head.isdigit():
            name = _SEQ_TO_NAME.get(head)
            if name is None:
                continue
            k = f"{name}.{tail}"
        out[k] = v
    return out


def file_sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_backbone(weights, trust_pickle):
    """Return (backbone: (B,3,H,W) -> (B,2048,1,1), eval mode, weights_path)."""
    import torch
    import torch.nn as nn
    from torchvision import models

    if weights:
        path = weights
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)
        print(f"[weights] using the UNOFFICIAL Hugging Face mirror: {path}")

    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        if not trust_pickle:
            raise SystemExit(
                f"torch.load(weights_only=True) failed: {type(e).__name__}: {e}\n"
                "The file may be a pickled full model. If (and only if) you trust "
                "its source, re-run with --trust-pickle.")
        obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    state = obj.state_dict() if isinstance(obj, nn.Module) else obj
    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected checkpoint type: {type(obj)}")

    net = models.resnet50(weights=None)
    missing, unexpected = net.load_state_dict(normalise_keys(state), strict=False)
    missing = [k for k in missing if not k.startswith("fc.") and not k.endswith("num_batches_tracked")]
    unexpected = [k for k in unexpected if not k.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint does not match ResNet-50: {len(missing)} missing "
            f"{missing[:5]}, {len(unexpected)} unexpected {unexpected[:5]}. "
            "Refusing to continue with partly random weights.")
    print(f"[weights] all {len(net.state_dict()) - 2} backbone tensors loaded from {path}")

    backbone = nn.Sequential(*list(net.children())[:-1])     # conv1 ... avgpool
    return backbone.eval(), str(path)


def write_feature_records(feats, meta, slice_dir):
    """Write one feature-only NPZ per slice, retaining source labels and IDs."""
    slice_dir = Path(slice_dir)
    slice_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row, (feature, m) in enumerate(zip(feats, meta)):
        nodule_id, patient_id = str(m["nodule_id"]), str(m["patient_id"])
        slice_idx = int(m["slice_idx"])
        target = slice_dir / f"{nodule_id}_slice{slice_idx:04d}.npz"
        payload = {
            "feature_vector": np.asarray(feature, dtype=np.float32),
            "patient_id": np.asarray(patient_id), "nodule_id": np.asarray(nodule_id),
            "slice_idx": np.asarray(slice_idx, dtype=np.int32),
            "nodule_idx": np.asarray(int(m["nodule_idx"]), dtype=np.int64),
            "n_annotators": np.asarray(int(m["n_annotators"]), dtype=np.int16),
            "source_roi": np.asarray(str(m["source_roi"])),
            "spacing": np.asarray(m["spacing"], dtype=np.float32),
            "pad_xy": np.asarray(m.get("pad_xy", [])),
            "pad_z": np.asarray(m.get("pad_z", [])),
        }
        for key in ("malignancy", "spiculation", "texture"):
            payload[key] = np.asarray(m.get(key, []))
        for key in ("has_mask", "mask_empty", "z_dist", "mask_area_mm2",
                    "pad_fraction", "slice_in_nodule", "n_slices_in_nodule"):
            if key in m:
                payload[key] = np.asarray(m[key])
        np.savez_compressed(target, **payload)
        manifest.append({**{k: v for k, v in m.items() if k not in
                            ("malignancy", "spiculation", "texture")},
                         **{k: json.dumps(np.asarray(m.get(k, [])).tolist())
                            for k in ("malignancy", "spiculation", "texture")},
                         "feature_npz": str(target), "feature_row": row})
    manifest_path = Path(f"{slice_dir.parent / slice_dir.name}_slice_manifest.csv")
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    return slice_dir, manifest_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset2d", help="builder output prefix")
    ap.add_argument("--out", default="dataset2d", help="feature output prefix "
                    "(encode the variant, e.g. dataset2d_rin_m1000_400)")
    ap.add_argument("--weights", default=None, help="official resnet50_torch.pt (preferred)")
    ap.add_argument("--trust-pickle", action="store_true",
                    help="allow torch.load(weights_only=False) for a pickled model")
    ap.add_argument("--lo", type=float, default=-1000.0, help="CT window low (HU)")
    ap.add_argument("--hi", type=float, default=400.0, help="CT window high (HU)")
    ap.add_argument("--norm", choices=["radimagenet", "imagenet"], default="radimagenet")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--meta", default=None, help="builder metadata pickle; defaults to DATA_meta.pkl")
    ap.add_argument("--slice-npz-dir", default=None,
                    help="feature-only per-slice NPZ directory; defaults to OUT_slices")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    import torch
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}   window=({args.lo:g}, {args.hi:g})  norm={args.norm}")

    x_path = f"{args.data}_X.npy"
    X = np.load(x_path, mmap_mode="r")
    bag = np.load(f"{args.data}_nodule_ids.npy")
    meta_path = Path(args.meta or f"{args.data}_meta.pkl")
    meta = pd.read_pickle(meta_path).to_dict(orient="records")
    N = len(X)
    if len(bag) != N or len(meta) != N:
        raise ValueError("X, nodule_ids, and per-slice metadata must have equal lengths")
    if [int(m["row_idx"]) for m in meta] != list(range(N)):
        raise ValueError("metadata row_idx must align exactly with X rows")
    print(f"[data] X={X.shape} {X.dtype} slices={N}  "
          f"nodules={len(set(bag.tolist()))}  "
          f"patients={len(set(m['patient_id'] for m in meta))}")

    backbone, weights_path = load_backbone(args.weights, args.trust_pickle)
    backbone = backbone.to(device)

    ds = SliceDataset(x_path, N, args.lo, args.hi, args.norm)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False,
                                     num_workers=args.workers, pin_memory=(device.type == "cuda"))
    feats = np.zeros((N, 2048), dtype=np.float32)
    written = np.zeros(N, dtype=bool)
    t0 = time.time()
    with torch.inference_mode():
        for i, (batch, idx) in enumerate(dl):
            out = backbone(batch.to(device, non_blocking=True)).flatten(1)
            feats[idx.numpy()] = out.float().cpu().numpy()
            written[idx.numpy()] = True
            if i % 50 == 0:
                done = int(written.sum())
                print(f"  {done}/{N}  {done / max(time.time() - t0, 1e-9):.0f} slices/s", flush=True)

    assert written.all(), "some rows were never written"
    assert np.isfinite(feats).all(), "non-finite features"
    stats = dict(mean=float(feats.mean()), std=float(feats.std()), max=float(feats.max()),
                 frac_zero=float((feats == 0).mean()),
                 row_std_min=float(feats.std(axis=1).min()))
    print(f"[feats] {stats}")
    if stats["row_std_min"] < 1e-6:
        print("[WARN] some rows have constant features (blank slices?)")

    np.save(f"{args.out}_feats.npy", feats)
    slice_dir, manifest_path = write_feature_records(
        feats, meta, args.slice_npz_dir or f"{args.out}_slices")
    import torchvision
    prov = dict(data=args.data, n_slices=N, dim=2048, lo=args.lo, hi=args.hi, norm=args.norm,
                weights=weights_path, weights_sha1=file_sha1(weights_path),
                torch=torch.__version__, torchvision=torchvision.__version__,
                slice_npz_dir=str(slice_dir), slice_manifest=str(manifest_path),
                input_meta=str(meta_path), stats=stats,
                created=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(prov, open(f"{args.out}_feats.json", "w"), indent=2)
    print(f"[done] {args.out}_feats.npy {feats.shape}; {N} feature NPZs in {slice_dir}; "
          f"manifest: {manifest_path}")

if __name__ == "__main__":
    main()