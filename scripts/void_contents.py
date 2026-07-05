"""What is in each cosmic-web environment? Gas, temperature, and HI per class on IllustrisTNG data."""
import numpy as np
import cosmicweb as cw

L, R, LAM = 25.0, 1.0, 0.3
FIELDS = {"Mtot": "tng_Mtot.npy", "Mgas": "tng_Mgas.npy", "T": "tng_T.npy", "HI": "tng_HI.npy"}


def main(gi=0):
    for field, path in FIELDS.items():
        cw.download(cw.cmd_url(field), path)
    web = cw.tidal_web(cw.smooth(cw.to_overdensity(cw.load_grid(FIELDS["Mtot"], gi)), L, R), L, LAM)
    gas = cw.load_grid(FIELDS["Mgas"], gi)
    temp = cw.load_grid(FIELDS["T"], gi)
    hi = cw.load_grid(FIELDS["HI"], gi)
    print(f"{'class':9s}{'vol%':>7s}{'gas/<gas>':>11s}{'median T (K)':>14s}{'HI share':>10s}")
    for c in range(4):
        m = web == c
        print(f"{cw.WEB_NAMES[c]:9s}{100 * m.mean():7.1f}{np.median(gas[m]) / gas.mean():11.3f}"
              f"{np.median(temp[m]):14.3e}{100 * hi[m].sum() / hi.sum():9.1f}%")


if __name__ == "__main__":
    main()
