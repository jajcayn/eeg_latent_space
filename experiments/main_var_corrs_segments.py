import argparse
import os
from itertools import combinations_with_replacement

import numpy as np
import xarray as xr
from utils import run_in_parallel


def corr2_coeff(A, B):
    """
    https://stackoverflow.com/a/30143754
    """
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))


def _compute_corr(args):
    topomaps, comb, lamap = args
    to_corr = topomaps.sel(
        {"subject_id": [comb[0], comb[1]], "latent map": lamap}
    ).dropna("channels")
    corr = (
        xr.DataArray(
            corr2_coeff(to_corr[:, 0, :].values.T, to_corr[:, 1, :].values.T),
            dims=["type1", "type2"],
            coords={
                "type1": topomaps["type"].values,
                "type2": topomaps["type"].values,
            },
        )
        .assign_coords(
            {"latent map": lamap, "subj1": comb[0], "subj2": comb[1]}
        )
        .expand_dims(["latent map", "subj1", "subj2"])
    )

    return corr


def main(folder):
    topomaps = xr.open_dataarray(os.path.join(folder, "topomaps.nc"))
    topomaps = topomaps.set_index({"stack": ["subject_id", "type"]}).unstack()

    all_corrs = run_in_parallel(
        _compute_corr,
        [
            (topomaps, comb, lamap)
            for lamap in topomaps["latent map"].values
            for comb in combinations_with_replacement(
                topomaps["subject_id"].values, r=2
            )
        ],
        workers=22,
    )

    combined = np.abs(xr.combine_by_coords(all_corrs))
    combined.to_netcdf(os.path.join(folder, "topomaps_corrs.nc"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAR correlation computation")
    parser.add_argument("folder", type=str, help="Folder with data VAR stuff")
    args = parser.parse_args()
    main(args.folder)
