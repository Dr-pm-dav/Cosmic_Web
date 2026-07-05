"""Tidal-tensor (T-web) cosmic-web classification."""
import numpy as np
from numpy.fft import fftn, ifftn, fftfreq

WEB_NAMES = ("void", "sheet", "filament", "knot")
WEB_COLORS = ("#08306b", "#6baed6", "#fd8d3c", "#cb181d")


def _kgrid(n, L):
    kax = fftfreq(n, d=L / n) * 2 * np.pi
    return np.meshgrid(kax, kax, kax, indexing="ij")


def smooth(delta, L, R):
    """Gaussian-smooth a periodic field at scale R (same length units as L)."""
    KX, KY, KZ = _kgrid(delta.shape[0], L)
    return ifftn(fftn(delta) * np.exp(-0.5 * (KX ** 2 + KY ** 2 + KZ ** 2) * R ** 2)).real


def tidal_web(delta, L, lambda_th=0.3):
    """Classify each voxel by how many tidal-tensor eigenvalues exceed lambda_th.

    Returns an int8 array with 0=void, 1=sheet, 2=filament, 3=knot.
    """
    n = delta.shape[0]
    KX, KY, KZ = _kgrid(n, L)
    Kv = (KX, KY, KZ)
    k2 = KX ** 2 + KY ** 2 + KZ ** 2
    k2[0, 0, 0] = 1.0
    dk = fftn(delta)
    T = np.zeros((n, n, n, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            t = ifftn((Kv[i] * Kv[j] / k2) * dk).real
            T[..., i, j] = t
            if i != j:
                T[..., j, i] = t
    return np.sum(np.linalg.eigvalsh(T) > lambda_th, axis=-1).astype(np.int8)


def web_fractions(labels):
    """Fraction of voxels in each of the four classes (void, sheet, filament, knot)."""
    f = np.bincount(labels.ravel(), minlength=4).astype(float)
    return f / f.sum()
