"""Plotting helpers for web maps and fields."""
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from .tweb import WEB_NAMES, WEB_COLORS


def web_cmap():
    """Return (ListedColormap, BoundaryNorm) for the four web classes."""
    return ListedColormap(list(WEB_COLORS)), BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4)


def plot_web(labels, L=25.0, z=None, ax=None, colorbar=True):
    """Plot a slice of a web-classification volume."""
    cmap, norm = web_cmap()
    z = labels.shape[0] // 2 if z is None else z
    ax = ax or plt.gca()
    im = ax.imshow(labels[z].T, origin="lower", cmap=cmap, norm=norm, extent=[0, L, 0, L])
    if colorbar:
        cb = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], fraction=0.046)
        cb.ax.set_yticklabels(WEB_NAMES)
    ax.set_xlabel("Mpc/h")
    ax.set_ylabel("Mpc/h")
    return ax
