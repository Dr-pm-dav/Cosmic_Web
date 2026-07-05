"""Download and load CAMELS Multifield Dataset 3D grids over public HTTP."""
import os
import urllib.request
import numpy as np

CAMELS_BASE = "https://users.flatironinstitute.org/~camels/CMD/3D_grids/data"


def cmd_url(field, folder="IllustrisTNG", tag="IllustrisTNG", cmd_set="CV", grid=128, z="0.0"):
    """Build a CAMELS grid download URL.

    folder : directory on the server ('IllustrisTNG', 'SIMBA', 'Astrid', or 'Nbody').
    tag    : the suite tag in the filename. For hydro suites this equals `folder`
             (e.g. 'IllustrisTNG'); for gravity-only runs it is the paired name,
             e.g. 'Nbody_IllustrisTNG'.
    field  : Mtot, Mcdm, Mgas, Mstar, T, Z, P, HI, ne, B, Vcdm, Vgas, MgFe, ...
    """
    return f"{CAMELS_BASE}/{folder}/Grids_{field}_{tag}_{cmd_set}_{grid}_z={z}.npy"


def download(url, path):
    """Download `url` to `path` if not already present; return `path`."""
    if not os.path.exists(path):
        print(f"downloading {os.path.basename(path)} ...")
        urllib.request.urlretrieve(url, path)
    return path


def load_grid(path, index):
    """Memory-map a CAMELS file and return realization `index` as a float64 array."""
    return np.array(np.load(path, mmap_mode="r")[index], dtype=np.float64)


def num_realizations(path):
    """Number of realizations stored in a CAMELS grid file."""
    return int(np.load(path, mmap_mode="r").shape[0])
