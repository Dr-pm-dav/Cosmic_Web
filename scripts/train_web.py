"""Train the 3D U-Net web classifier on CAMELS N-body data; report held-out test metrics.

Defaults assume a GPU. On CPU, shrink the ID lists and EPOCHS first.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
import cosmicweb as cw

FILE = "cmd_cv128.npy"
L, R, LAM, NBAR = 25.0, 1.0, 0.3, 3.0
PATCH, STRIDE = 32, 16
TRAIN_IDS = list(range(0, 19))
VAL_IDS = [19, 20, 21, 22]
TEST_IDS = [23, 24, 25, 26]
F, EPOCHS, BATCH = 16, 40, 8


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    cw.download(cw.cmd_url("Mtot", folder="Nbody", tag="Nbody_IllustrisTNG"), FILE)

    inp, lab = {}, {}
    for gi in sorted(set(TRAIN_IDS) | set(VAL_IDS) | set(TEST_IDS)):
        delta = cw.to_overdensity(cw.load_grid(FILE, gi))
        lab[gi] = cw.tidal_web(cw.smooth(delta, L, R), L, LAM)
        inp[gi] = cw.log_counts(cw.poisson_tracers(delta, NBAR, seed=100 + gi))[None]  # (1,N,N,N)

    mu = np.mean([inp[g].mean() for g in TRAIN_IDS])
    sd = np.mean([inp[g].std() for g in TRAIN_IDS])
    for g in inp:
        inp[g] = ((inp[g] - mu) / sd).astype(np.float32)

    def loader(ids, aug, sh):
        return DataLoader(cw.PatchSet(inp, lab, ids, PATCH, STRIDE, aug), batch_size=BATCH, shuffle=sh)

    net = cw.UNet3D(1, 4, F).to(device)
    w = cw.class_weights([lab[g] for g in TRAIN_IDS])
    net, best = cw.train_segmenter(net, loader(TRAIN_IDS, True, True), loader(VAL_IDS, False, False),
                                   device, EPOCHS, 1e-3, w)

    m = cw.metrics_from_confusion(cw.confusion_matrix(net, loader(TEST_IDS, False, False), device))
    print(f"\nTEST  accuracy {m['accuracy']:.3f}   mean IoU {m['iou'].mean():.3f}   macro F1 {m['f1'].mean():.3f}")
    for c, name in enumerate(cw.WEB_NAMES):
        print(f"  {name:9s} F1 {m['f1'][c]:.3f}   IoU {m['iou'][c]:.3f}")
    torch.save(net.state_dict(), "web_unet.pt")
    print("saved web_unet.pt")


if __name__ == "__main__":
    main()
