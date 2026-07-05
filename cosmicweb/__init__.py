"""cosmicweb -- map the cosmic web and characterize void contents from sparse tracers.

Quick start::

    import cosmicweb as cw
    cw.download(cw.cmd_url("Mtot", folder="Nbody", tag="Nbody_IllustrisTNG"), "cmd.npy")
    rho = cw.load_grid("cmd.npy", 0)
    web = cw.tidal_web(cw.smooth(cw.to_overdensity(rho), 25.0, 1.0), 25.0)
"""
from .tweb import smooth, tidal_web, web_fractions, WEB_NAMES, WEB_COLORS
from .fields import to_overdensity, poisson_tracers, log_counts, log_field
from .data import CAMELS_BASE, cmd_url, download, load_grid, num_realizations
from .models import UNet3D, MultiFieldUNet, double_conv
from .train import (
    PatchSet, cube_symmetry_cx, class_weights,
    confusion_matrix, metrics_from_confusion, train_segmenter,
)
from . import viz

__version__ = "0.1.0"

__all__ = [
    "smooth", "tidal_web", "web_fractions", "WEB_NAMES", "WEB_COLORS",
    "to_overdensity", "poisson_tracers", "log_counts", "log_field",
    "CAMELS_BASE", "cmd_url", "download", "load_grid", "num_realizations",
    "UNet3D", "MultiFieldUNet", "double_conv",
    "PatchSet", "cube_symmetry_cx", "class_weights",
    "confusion_matrix", "metrics_from_confusion", "train_segmenter",
    "viz", "__version__",
]
