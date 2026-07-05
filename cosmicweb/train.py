"""Training utilities: patch dataset, cube-symmetry augmentation, metrics, and a training loop."""
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def cube_symmetry_cx(x, y):
    """Apply a random element of the cube symmetry group to a (C, N, N, N) input and (N, N, N) label.

    Flips and axis permutations are valid augmentations because the web labels are
    invariant under rotations and reflections. Input channels share the spatial transform.
    """
    for ax in range(3):
        if np.random.random() < 0.5:
            x = np.flip(x, ax + 1)  # +1 skips the channel axis
            y = np.flip(y, ax)
    perm = np.random.permutation(3)
    x = np.transpose(x, (0,) + tuple(perm + 1))
    y = np.transpose(y, perm)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


class PatchSet(Dataset):
    """Serve 3D sub-cubes from precomputed (input, label) grids, with optional augmentation.

    inputs : dict gid -> (C, N, N, N) float32
    labels : dict gid -> (N, N, N) int
    ids    : which realizations to draw from
    """

    def __init__(self, inputs, labels, ids, patch=32, stride=16, augment=False):
        self.inputs, self.labels, self.augment, self.patch = inputs, labels, augment, patch
        self.index = []
        n = next(iter(inputs.values())).shape[-1]
        for gi in ids:
            for i in range(0, n - patch + 1, stride):
                for j in range(0, n - patch + 1, stride):
                    for k in range(0, n - patch + 1, stride):
                        self.index.append((gi, i, j, k))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, n):
        gi, i, j, k = self.index[n]
        p = self.patch
        x = self.inputs[gi][:, i:i + p, j:j + p, k:k + p]
        y = self.labels[gi][i:i + p, j:j + p, k:k + p].astype(np.int64)
        if self.augment:
            x, y = cube_symmetry_cx(x, y)
        return torch.from_numpy(np.ascontiguousarray(x)).float(), torch.from_numpy(np.ascontiguousarray(y))


def class_weights(label_arrays, n_classes=4):
    """sqrt-inverse-frequency class weights (gentle on the rare knot class)."""
    fr = np.bincount(np.concatenate([a.ravel() for a in label_arrays]), minlength=n_classes)
    w = 1.0 / np.sqrt(fr + 1)
    return (w / w.sum() * n_classes).astype(np.float32)


def confusion_matrix(net, loader, device, n_classes=4):
    """Accumulate a confusion matrix over a loader. Handles both UNet3D and MultiFieldUNet outputs."""
    net.eval()
    C = np.zeros((n_classes, n_classes), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            out = net(x.to(device))
            logits = out[0] if isinstance(out, tuple) else out
            p = logits.argmax(1).cpu().numpy().ravel()
            t = y.numpy().ravel()
            C += np.bincount(n_classes * t + p, minlength=n_classes ** 2).reshape(n_classes, n_classes)
    return C


def metrics_from_confusion(C):
    """Per-class F1 and IoU plus overall accuracy from a confusion matrix."""
    f1, iou = [], []
    for c in range(len(C)):
        tp = C[c, c]
        fp = C[:, c].sum() - tp
        fn = C[c, :].sum() - tp
        f1.append(2 * tp / (2 * tp + fp + fn + 1e-9))
        iou.append(tp / (tp + fp + fn + 1e-9))
    return {"accuracy": float(np.trace(C) / C.sum()), "f1": np.array(f1), "iou": np.array(iou)}


def train_segmenter(net, train_loader, val_loader, device, epochs=40, lr=1e-3, weight=None):
    """Train a segmentation net with cosine LR, mixed precision on CUDA, and best-epoch checkpointing.

    Returns (net_with_best_weights, best_val_macro_f1).
    """
    dev = "cuda" if "cuda" in str(device) else "cpu"
    use_amp = dev == "cuda"
    opt = torch.optim.Adam(net.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w = torch.tensor(weight).to(device) if weight is not None else None
    crit = nn.CrossEntropyLoss(weight=w)
    scaler = torch.amp.GradScaler(dev, enabled=use_amp)
    best_f1, best_state = 0.0, None
    for ep in range(epochs):
        net.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast(device_type=dev, enabled=use_amp):
                out = net(x)
                logits = out[0] if isinstance(out, tuple) else out
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        m = metrics_from_confusion(confusion_matrix(net, val_loader, device))
        if m["f1"].mean() > best_f1:
            best_f1, best_state = float(m["f1"].mean()), copy.deepcopy(net.state_dict())
        print(f"epoch {ep + 1:>3}/{epochs}  val macroF1 {m['f1'].mean():.3f}  acc {m['accuracy']:.3f}")
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, best_f1
