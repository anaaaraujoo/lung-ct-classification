from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from build_2d_dataset import (
    TARGET_MM,
    TARGET_SIZE,
    discover_npzs,
    inspect_roi,
    parse_ids,
    render_slice,
)
import tqdm

CFG = dict(
    slice_mode="replicate",    # "replicate": slice copied to 3 channels | "adjacent": [i-1, i, i+1]
    hu_window=(-1000, 400),
    norm="radimagenet",        # matches extract_features.py --norm radimagenet
    size=TARGET_SIZE,          # network input size
    train_slices=16,           # fixed slice count while fitting the head (needed for batching)
    batch_size=2,
    lr=1e-3,
    test_fraction=0.15,
    validation_fraction=0.15,
    seed=42,
    cam_normalize="nodule",    # "nodule": shared scale across slices | "slice": per-slice scale
)


# Data
def task_label(ratings, task):
    ratings = np.asarray(ratings, dtype=np.float32).reshape(-1)
    if ratings.size == 0 or not np.isin(ratings, [1, 2, 3, 4, 5]).all():
        raise ValueError(f"Expected malignancy ratings in 1..5, got {ratings}")
    mean_rating = float(ratings.mean())
    if task == "binary":
        return None if mean_rating == 3 else int(mean_rating > 3)
    if task == "3-class":
        return 2 if mean_rating == 3 else int(mean_rating > 3)
    return int(np.floor(mean_rating + 0.5)) - 1


def discover_samples(data_dir, task):
    """Plan labeled nodule inputs directly from reconstructed ROI NPZ files."""
    root = Path(data_dir)
    roi_dir = root / "reconstructed_3d" if (root / "reconstructed_3d").is_dir() else root
    roi_dir, paths = discover_npzs(roi_dir)
    if task == "binary":
        class_names = ["benign", "malignant"]
    elif task == "3-class":
        class_names = ["benign", "malignant", "inconclusive"]
    else:
        class_names = ["1", "2", "3", "4", "5"]

    plans = []
    seen_nodules = set()
    skipped = 0
    for path in paths:
        patient_id, nodule_id = parse_ids(path, roi_dir)
        if nodule_id in seen_nodules:
            raise ValueError(f"More than one reconstructed volume maps to {nodule_id}")
        seen_nodules.add(nodule_id)
        plan, reason = inspect_roi(path, patient_id, nodule_id, expand=2, keep_empty=False)
        if plan is None:
            skipped += 1
            continue
        plans.append(plan)

    if not plans:
        raise ValueError(f"No usable reconstructed nodule volumes found under {roi_dir}")

    # Match the notebook's deterministic patient-disjoint 70/15/15 assignment.
    patients = np.unique([plan["patient_id"] for plan in plans])
    shuffled = np.random.default_rng(CFG["seed"]).permutation(patients)
    n_test = max(1, int(round(CFG["test_fraction"] * len(shuffled))))
    test_patients = set(shuffled[:n_test])
    remaining = shuffled[n_test:]
    n_validation = max(
        1,
        int(round((CFG["validation_fraction"] / (1.0 - CFG["test_fraction"])) * len(remaining))),
    )
    validation_patients = set(remaining[:n_validation])
    train_patients = set(remaining[n_validation:])
    for plan in plans:
        patient_id = plan["patient_id"]
        if patient_id in test_patients:
            plan["split"] = "test"
        elif patient_id in validation_patients:
            plan["split"] = "validation"
        else:
            plan["split"] = "train"

    samples = []
    for plan in plans:
        label = task_label(plan["labels"]["malignancy"], task)
        if label is not None:
            samples.append((plan["nodule_id"], label, plan))

    if not samples:
        raise ValueError(f"No labeled nodules available for task={task!r}")
    print(f"Loaded {len(samples)} {task} nodules from {roi_dir}; skipped {skipped} unusable volumes")
    return samples, class_names


def prepare_nodule(images, num_slices=None):
    """Return normalized tensor[K,3,S,S] and display images[K,S,S] from HU slices."""
    vol = np.asarray(images, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected nodule slices with shape (D,H,W), got {vol.shape}")
    depth = vol.shape[0]
    idx = np.arange(depth) if num_slices is None else np.linspace(0, depth - 1, num_slices).round().astype(int)
    lo, hi = CFG["hu_window"]
    vol = (np.clip(vol, lo, hi) - lo) / (hi - lo)

    if CFG["slice_mode"] == "adjacent":
        prev = np.clip(idx - 1, 0, depth - 1)
        nxt = np.clip(idx + 1, 0, depth - 1)
        chans = np.stack([vol[prev], vol[idx], vol[nxt]], axis=1)
    elif CFG["slice_mode"] == "replicate":
        chans = np.repeat(vol[idx][:, None], 3, axis=1)
    else:
        raise ValueError(f"Unknown slice_mode: {CFG['slice_mode']}")

    tensor = torch.from_numpy(np.ascontiguousarray(chans))
    if tensor.shape[-2:] != (CFG["size"], CFG["size"]):
        tensor = F.interpolate(tensor, size=(CFG["size"], CFG["size"]), mode="bilinear", align_corners=False)
    display = tensor[:, 1].clone()
    if CFG["norm"] == "radimagenet":
        tensor = tensor * 2.0 - 1.0
    elif CFG["norm"] == "imagenet":
        mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
    else:
        raise ValueError(f"Unknown normalization: {CFG['norm']}")
    return tensor, display


def prepare_nodule_from_roi(plan, num_slices=None):
    with np.load(plan["path"], allow_pickle=False) as npz:
        volume = npz["image"]
    fill = float(np.median(volume))
    slices = [
        render_slice(volume[:, :, z], plan["spacing"], plan["center"],
                     fill, CFG["size"], TARGET_MM)[0]
        for z in plan["z_list"]
    ]
    return prepare_nodule(np.stack(slices), num_slices=num_slices)


class NoduleDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        _, label, plan = self.samples[i]
        x, _ = prepare_nodule_from_roi(plan, num_slices=CFG["train_slices"])
        return x, label


# Model
def load_radimagenet_weights(backbone: nn.Module, path: str):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, nn.Module):
        state = state.state_dict()
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    cleaned = {}
    for k, v in state.items():
        changed = True
        while changed:  # strip wrapper prefixes such as "module." / "backbone."
            changed = False
            for p in ("module.", "backbone.", "model."):
                if k.startswith(p):
                    k, changed = k[len(p):], True
        head, _, tail = k.partition(".")
        sequential_names = {"0": "conv1", "1": "bn1", "4": "layer1",
                            "5": "layer2", "6": "layer3", "7": "layer4"}
        if head.isdigit():
            name = sequential_names.get(head)
            if name is None:
                continue
            k = f"{name}.{tail}"
        if k.startswith(("fc.", "classifier.", "linear.")):
            continue
        cleaned[k] = v
    missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
    missing = [k for k in missing if not k.startswith("fc.")]
    if missing:
        raise RuntimeError(
            f"Checkpoint does not match torchvision ResNet-50 (missing {len(missing)} keys, e.g. {missing[:3]}). "
            "Adapt the key mapping in load_radimagenet_weights()."
        )
    if unexpected:
        print(f"[weights] ignored {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")

class NoduleClassifier(nn.Module):
    def __init__(self, weights_path: str, num_classes: int):
        super().__init__()
        self.backbone = torchvision.models.resnet50(weights=None)
        load_radimagenet_weights(self.backbone, weights_path)
        self.backbone.fc = nn.Identity()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = nn.Linear(2048, num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # keep BatchNorm statistics frozen
        return self

    @property
    def target_layer(self):
        return self.backbone.layer4

    def forward(self, x):  # x: (B, D, 3, H, W)
        B, D = x.shape[:2]
        feats = self.backbone(x.flatten(0, 1))      # (B*D, 2048)
        feats = feats.view(B, D, -1).mean(dim=1)    # mean over slices
        return self.head(feats)


# Head training
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += len(y)
    return correct / max(total, 1)

def cmd_train(a, device):
    samples, class_names = discover_samples(a.data_dir, a.task)
    train = [sample for sample in samples if sample[2]["split"] == "train"]
    val = [sample for sample in samples if sample[2]["split"] == "validation"]
    print(f"{len(train)} train / {len(val)} validation nodules, task={a.task}, classes={class_names}")

    bs = CFG["batch_size"]
    train_dl = DataLoader(NoduleDataset(train), batch_size=bs, shuffle=True)
    val_dl = DataLoader(NoduleDataset(val), batch_size=bs) if val else None

    model = NoduleClassifier(a.weights, len(class_names)).to(device)
    opt = torch.optim.Adam(model.head.parameters(), lr=CFG["lr"])
    for ep in range(1, a.epochs + 1):
        model.train()
        running = 0.0
        for x, y in tqdm.tqdm(train_dl, desc=f"Epoch {ep:02d}", leave=False):
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
        msg = f"epoch {ep:02d} loss {running / len(train):.4f}"
        if val_dl:
            msg += f" val_acc {evaluate(model, val_dl, device):.3f}"
        print(msg)
    torch.save({"head": model.head.state_dict(), "class_names": class_names, "task": a.task}, a.out)
    print(f"saved head -> {a.out}")


# Grad-CAM
class GradCAM:
    def __init__(self, model: NoduleClassifier, target_layer: nn.Module):
        self.model = model
        self.acts = None
        self.grads = None
        self._handle = target_layer.register_forward_hook(self._on_forward)

    def _on_forward(self, module, inp, out):
        self.acts = out.detach()
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "grads", g.detach()))

    def remove(self):
        self._handle.remove()

    def __call__(self, x, class_idx=None):
        """x: (1, D, 3, H, W). Returns dict with cams (D,H,W) in [0,1], probs, class_idx."""
        self.model.eval()
        # Backbone is frozen: without this no autograd graph would exist.
        x = x.clone().requires_grad_(True)
        self.grads = None
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = logits.softmax(1)[0].detach().cpu()
        if class_idx is None:
            class_idx = int(logits.argmax(1))
        logits[0, class_idx].backward()

        weights = self.grads.mean(dim=(2, 3), keepdim=True)            # (D, 2048, 1, 1)
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))   # (D, 1, 7, 7)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        if CFG["cam_normalize"] == "nodule":
            cam = cam / (cam.max() + 1e-8)
        else:
            mx = cam.flatten(1).max(1)[0].view(-1, 1, 1)
            cam = cam / (mx + 1e-8)
        return {"cams": cam.cpu().numpy(), "probs": probs.numpy(), "class_idx": class_idx}

def save_overlay_grid(display, cams, out_path, title="", cols=8, alpha=0.45):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = display.shape[0]
    rows = int(np.ceil(K / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < K:
            ax.imshow(display[i], cmap="gray", vmin=0, vmax=1)
            ax.imshow(cams[i], cmap="jet", alpha=alpha, vmin=0, vmax=1)
            ax.set_title(f"slice {i}", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def cmd_explain(a, device):
    ckpt = torch.load(a.head, map_location="cpu", weights_only=True)
    class_names = ckpt["class_names"]
    task = a.task or ckpt.get("task", "binary")
    if "task" in ckpt and task != ckpt["task"]:
        raise SystemExit(f"Checkpoint task is {ckpt['task']!r}, not requested task {task!r}.")
    model = NoduleClassifier(a.weights, len(class_names))
    model.head.load_state_dict(ckpt["head"])
    model.to(device).eval()

    samples, data_classes = discover_samples(a.data_dir, task)
    if data_classes != class_names:
        raise SystemExit(f"Checkpoint classes {class_names} do not match dataset classes {data_classes}.")
    samples = [sample for sample in samples if sample[2]["split"] == a.split]
    if a.limit:
        samples = samples[: a.limit]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cam = GradCAM(model, model.target_layer)
    output_rows = []
    for n, (nodule_id, label, plan) in enumerate(samples, 1):
        x, display = prepare_nodule_from_roi(plan)  # all nodule slices
        res = cam(x.unsqueeze(0).to(device))
        pred = res["class_idx"]
        prob = float(res["probs"][pred])
        true = data_classes[label]
        stem = nodule_id
        np.savez_compressed(out / f"{stem}_gradcam.npz", cams=res["cams"], probs=res["probs"],
                            class_idx=pred)
        title = f"{stem} | true: {true or '?'} | pred: {class_names[pred]} (p={prob:.2f})"
        save_overlay_grid(display.numpy(), res["cams"], out / f"{stem}_gradcam.png", title)
        output_rows.append([nodule_id, true, class_names[pred], f"{prob:.4f}"])
        print(f"[{n}/{len(samples)}] {title}")
    cam.remove()

    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "true", "pred", "prob_pred"])
        w.writerows(output_rows)
    print(f"done -> {out}")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="fit the linear head on the frozen backbone")
    tr.add_argument("--data-dir", required=True, help="processed-data directory or its reconstructed_3d folder")
    tr.add_argument("--weights", required=True, help="RadImageNet ResNet-50 checkpoint")
    tr.add_argument("--task", choices=("binary", "3-class", "5-class"), default="binary")
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--out", default="head.pt")

    ex = sub.add_parser("explain", help="Grad-CAM for every nodule in the folder")
    ex.add_argument("--data-dir", required=True, help="processed-data directory or its reconstructed_3d folder")
    ex.add_argument("--weights", required=True)
    ex.add_argument("--head", default="head.pt")
    ex.add_argument("--task", choices=("binary", "3-class", "5-class"), default=None)
    ex.add_argument("--split", choices=("train", "validation", "test"), default="test")
    ex.add_argument("--out-dir", default="gradcam_out")
    ex.add_argument("--limit", type=int, default=None, help="only process the first N nodules")

    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (cmd_train if a.cmd == "train" else cmd_explain)(a, device)

if __name__ == "__main__":
    main()