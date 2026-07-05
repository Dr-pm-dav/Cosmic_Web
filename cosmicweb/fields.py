"""Density-field helpers: overdensity, sparse tracers, network input."""
import numpy as np


def to_overdensity(rho):
    """Convert a mass-density field to overdensity delta = rho/<rho> - 1."""
    return rho / rho.mean() - 1.0


def poisson_tracers(delta, nbar, seed=0):
    """Poisson-sample a sparse tracer field: mean rate nbar per voxel, proportional to (1+delta)."""
    rng = np.random.default_rng(seed)
    return rng.poisson(nbar * np.clip(1.0 + delta, 0.0, None))


def log_counts(counts):
    """Compress a counts field to log10(1+counts) for use as a network input channel."""
    return np.log10(1.0 + counts).astype(np.float32)


def log_field(field):
    """Compress a positive physical field (gas, HI, ...) to log10(1+field)."""
    return np.log10(1.0 + field).astype(np.float32)
