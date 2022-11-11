"""
Create folder structure to be share on figshare, i.e. keeps only aggregate
results, optionally archives all subject data into a tgz archive.
"""

import argparse
import os
import string
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

NEEDED_FILES = [
    "latent_stats.csv",
    "dynamic_stats.csv",
]


def main(
    input_glob, output_folder, sampling_freq=250.0, segmentation_join="inner"
):
    output_folder = os.path.abspath(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    df_all = pd.DataFrame()
    xr_all_segmentations = []
    xr_all_topomaps = []
    xr_group_topo = []

    for folder in tqdm(sorted(glob(input_glob))):
        folder = os.path.abspath(folder)
        assert all(
            [
                os.path.exists(os.path.join(folder, file))
                for file in NEEDED_FILES
            ]
        )
        folder_types = os.path.basename(folder).split("_")
        data_type = folder_types[-3]
        if data_type == "seeded":
            data_type = folder_types[-4]
        # properties
        df_latent = pd.read_csv(
            os.path.join(folder, "latent_stats.csv"), index_col=0
        )
        df_latent = df_latent.set_index(["subject_id", "latent map"])
        df_dynamic = pd.read_csv(
            os.path.join(folder, "dynamic_stats.csv"), index_col=0
        )
        df_dynamic = df_dynamic.set_index(["subject_id", "latent map"])
        df_join = df_latent.join(df_dynamic)
        df_join["data_type"] = data_type
        df_all = pd.concat([df_all, df_join], axis=0)

        n_states = df_join["no_states"].unique()[0]
        decomp_type = df_join["decomposition_type"].unique()[0]
        try:
            surr_type = df_join["surrogate_type"].unique()[0]
        except KeyError:
            surr_type = "real_data"

        # group topographies
        group_topo = np.load(os.path.join(folder, "group_mean.npz"))
        xr_topo = (
            xr.DataArray(
                group_topo["latent_maps"],
                # np.append(
                #     group_topo["latent_maps"],
                #     group_topo["corrs_w_template"][:, np.newaxis],
                #     axis=1,
                # ),
                dims=["latent map", "channels"],
                coords={
                    "latent map": list(string.ascii_uppercase)[:n_states],
                    "channels": group_topo["group_channels"].tolist()
                    # + ["corr. w. template"],
                },
            )
            .assign_coords(
                {
                    "no_states": n_states,
                    "decomposition_type": decomp_type,
                    "surrogate_type": surr_type,
                    "data_type": data_type,
                }
            )
            .expand_dims(["data_type", "surrogate_type", "decomposition_type"])
        )
        xr_group_topo.append(xr_topo)

        # segmentation and topo
        all_segmentations = []
        all_topomaps = []
        subj_ids = []
        for subj_file in sorted(glob(folder + "/sub*.npz")):
            data_file = np.load(subj_file)
            segmentation = data_file["latent_segmentation"]
            topo = data_file["latent_maps"]
            all_segmentations.append(segmentation)
            all_topomaps.append(topo)
            subj_id = os.path.basename(subj_file).split(".")[0]
            subj_ids.append(subj_id.split("_")[0])
        # unequal lengths of segmentations -> take maximum and fill with NaNs
        array_segm = np.empty(
            (len(subj_ids), max([len(arr) for arr in all_segmentations]))
        )
        for i, arr in enumerate(all_segmentations):
            array_segm[i, : len(arr)] = arr
        # unequal shape of topomaps
        array_topomaps = np.empty(
            (
                len(subj_ids),
                n_states,
                max([arr.shape[1] for arr in all_topomaps]),
            )
        )
        for i, arr in enumerate(all_topomaps):
            array_topomaps[i, :, : arr.shape[1]] = arr
        xr_segm = (
            xr.DataArray(
                array_segm,
                dims=["subject_id", "time"],
                coords={
                    "subject_id": subj_ids,
                    "time": np.arange(
                        0,
                        array_segm.shape[1] * (1.0 / sampling_freq),
                        1.0 / sampling_freq,
                    ),
                },
            )
            .assign_coords(
                {
                    "no_states": n_states,
                    "decomposition_type": decomp_type,
                    "surrogate_type": surr_type,
                    "data_type": data_type,
                }
            )
            .expand_dims(["data_type", "surrogate_type", "decomposition_type"])
        )
        xr_all_segmentations.append(xr_segm)

        xr_topo = (
            xr.DataArray(
                array_topomaps,
                dims=["subject_id", "latent map", "channels"],
                coords={
                    "subject_id": subj_ids,
                    "latent map": list(string.ascii_uppercase)[:n_states],
                    "channels": np.arange(array_topomaps.shape[2]),
                },
            )
            .assign_coords(
                {
                    "no_states": n_states,
                    "decomposition_type": decomp_type,
                    "surrogate_type": surr_type,
                    "data_type": data_type,
                }
            )
            .expand_dims(["data_type", "surrogate_type", "decomposition_type"])
        )
        xr_all_topomaps.append(xr_topo)
    try:
        df_all["surrogate_type"] = df_all["surrogate_type"].fillna("real_data")
    except KeyError:
        df_all["surrogate_type"] = "real_data"
    df_all.to_csv(os.path.join(output_folder, "stats.csv"))

    combined_group_topo = xr.combine_by_coords(xr_group_topo, join="outer")
    combined_group_topo.to_netcdf(os.path.join(output_folder, "group_topo.nc"))

    combined_topo = xr.combine_by_coords(xr_all_topomaps, join="outer")
    combined_topo.to_netcdf(os.path.join(output_folder, "topomaps.nc"))

    combined_segmentations = xr.combine_by_coords(
        xr_all_segmentations, join=segmentation_join
    )
    combined_segmentations.to_netcdf(
        os.path.join(output_folder, "segmentations.nc")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select only group results for sharing on figshare"
    )
    parser.add_argument(
        "input_data_glob",
        type=str,
        help="Input data glob / folder paths, not files!",
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Output folder",
    )
    parser.add_argument(
        "--data_sampling_freq",
        type=float,
        default=250.0,
        help="Sampling frequency of the data",
    )
    parser.add_argument(
        "--segmentation_join",
        type=str,
        choices=["outer", "inner"],
        default="inner",
        help="join to use for segmentations - outer or inner",
    )

    args = parser.parse_args()
    main(
        args.input_data_glob,
        args.output_folder,
        args.data_sampling_freq,
        args.segmentation_join,
    )
