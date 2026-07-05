"""Download the CAMELS grids used by the scripts and real-data notebooks (~1 GB, cached locally)."""
import os
from cosmicweb import cmd_url, download

FILES = {
    "cmd_cv128.npy": cmd_url("Mtot", folder="Nbody", tag="Nbody_IllustrisTNG"),  # N-body matter (real/scale)
    "tng_Mtot.npy": cmd_url("Mtot"),   # IllustrisTNG matter    (multifield)
    "tng_Mgas.npy": cmd_url("Mgas"),   # gas density
    "tng_T.npy": cmd_url("T"),         # gas temperature
    "tng_HI.npy": cmd_url("HI"),       # neutral hydrogen
}


def main():
    for path, url in FILES.items():
        download(url, path)
        print(f"  {path}: {os.path.getsize(path) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
