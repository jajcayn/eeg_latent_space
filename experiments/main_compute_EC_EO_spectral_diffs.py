"""
Script for computing spectral differences in the EC/EO data.
"""
import argparse
import logging
import os
from copy import deepcopy
from glob import glob
from multiprocessing import cpu_count

import mne
import numpy as np
import xarray as xr
from eeg_recording import SingleSubjectRecording
from utils import RESULTS_ROOT, make_dirs, run_in_parallel, set_logger, today

BANDS = [
    (0, 4, "Delta"),
    (4, 8, "Theta"),
    (8, 12, "Alpha"),
    (12, 30, "Beta"),
    (30, 45, "Gamma"),
]


def _compute_freq_stats(args):
    recording, bands, data_type = args
    # create 2s Epochs
    events, event_id = mne.events_from_annotations(recording._data)
    epochs = mne.Epochs(
        recording._data,
        events,
        event_id=event_id,
        event_repeated="merge",
        tmax=2.0,
        tmin=0.0,
        baseline=None,
        preload=True,
    )
    all_data = []
    for band in bands:
        filt_envelope = (
            deepcopy(epochs)
            .filter(band[0], band[1])
            .apply_hilbert(envelope=True)
        )
        # epochs x channels x time in epoch
        band_data = (
            xr.DataArray(
                filt_envelope.get_data() * 1e6,
                dims=["epochs", "channels", "time"],
                coords={
                    "epochs": np.arange(filt_envelope.get_data().shape[0]),
                    "channels": filt_envelope.info["ch_names"],
                    "time": np.arange(0, filt_envelope.get_data().shape[-1])
                    * (1.0 / filt_envelope.info["sfreq"]),
                },
            )
            .assign_coords(
                {
                    "band": band[2],
                    "data_type": data_type,
                    "subject_id": recording.subject_id.split("_")[0],
                }
            )
            .expand_dims(["band", "data_type", "subject_id"])
        )
        # mean over epochs
        band_data = band_data.mean(dim="epochs")
        all_data.append(band_data)

    return xr.concat(all_data, dim="band").stack(
        stacked=["data_type", "subject_id"]
    )


def main(input_data, workers=cpu_count()):
    result_dir = os.path.join(
        RESULTS_ROOT, f"{today()}_EEG_ECvsEO_spectral_diffs"
    )
    make_dirs(result_dir)
    set_logger(log_filename=os.path.join(result_dir, "log"))
    logging.info("Loading subject data...")

    recordings_EC = []
    for data_file in sorted(glob(f"{input_data}/*_EC.set")):
        mne_data = mne.io.read_raw_eeglab(data_file, preload=True)
        subject_id = "-".join(os.path.basename(data_file).split(".")[:-1])
        recordings_EC.append(
            SingleSubjectRecording(subject_id=subject_id, data=mne_data)
        )
    logging.info(f"Loaded {len(recordings_EC)} EC data files")

    recordings_EO = []
    for data_file in sorted(glob(f"{input_data}/*_EO.set")):
        mne_data = mne.io.read_raw_eeglab(data_file, preload=True)
        subject_id = "-".join(os.path.basename(data_file).split(".")[:-1])
        recordings_EO.append(
            SingleSubjectRecording(subject_id=subject_id, data=mne_data)
        )
    logging.info(f"Loaded {len(recordings_EO)} EO data files")

    results_EC = run_in_parallel(
        _compute_freq_stats,
        [(deepcopy(recording), BANDS, "EC") for recording in recordings_EC],
        workers=workers,
    )

    results_EO = run_in_parallel(
        _compute_freq_stats,
        [(deepcopy(recording), BANDS, "EO") for recording in recordings_EO],
        workers=workers,
    )

    all_data = xr.concat(list(results_EC) + list(results_EO), dim="stacked")

    all_data.unstack("stacked").to_netcdf(
        os.path.join(result_dir, "epochs_bands_envelopes.nc")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Subjectwise latent EEG decomposition"
    )
    parser.add_argument(
        "input_data", type=str, help="Folder with data files [EEGLAB]"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="number of processes to launch",
    )
    args = parser.parse_args()
    main(args.input_data, args.workers)
