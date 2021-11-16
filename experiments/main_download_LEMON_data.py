"""
Little script for downloading preprocessed EEG data from mind-brain-body
dataset.

Reference:
    [paper]
    Babayan, A., Erbey, M., Kumral, D., Reinelt, J. D., Reiter, A. M., Robbig,
        J., ... & Villringer, A. (2019). A mind-brain-body dataset of MRI, EEG,
        cognition, emotion, and peripheral physiology in young and old adults.
        Scientific data, 6(1), 1-21.
    [data]
    https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/
"""

import argparse
import ftplib
import logging
import os
import re

import numpy as np
from utils import make_dirs, set_logger

BASE_URL = "ftp.gwdg.de"
DATA_FOLDER = "/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/EEG_Preprocessed/"


def _ftp_connect():
    ftp = ftplib.FTP(BASE_URL)
    ftp.login()
    ftp.cwd(DATA_FOLDER)
    return ftp


def download(no_subjects, path, data_type):
    set_logger()
    make_dirs(path)
    assert data_type in ["EC", "EO", "all"]
    if data_type in ["EC", "EO"]:
        data_type = [data_type]
    else:
        data_type = ["EC", "EO"]
        no_subjects *= 2
    no_subjects = np.inf if no_subjects == 0 else no_subjects
    conn = _ftp_connect()
    # get file list
    try:
        files = sorted(conn.nlst())
    except ftplib.error_perm as resp:
        if str(resp) == "550 No files found":
            raise Exception("No files found")
        else:
            raise
    logging.info(
        f"Found {len(files)} files; downloading data for {no_subjects} subjects"
        f" of type {data_type}"
    )
    done_cnt = 0
    fdt_done = False
    set_done = False
    for file in files:
        _, _, file_type, ext = re.split("[_\\-\\.]", file)
        if file_type not in data_type:
            continue
        write_to = os.path.join(path, file)
        if os.path.exists(write_to):
            logging.info(f"File {file} already exists, skipping...")
            continue
        logging.info(f"Downloading {file}...")
        with open(write_to, "wb") as f:
            conn.retrbinary("RETR " + file, f.write)
        if ext == "fdt":
            fdt_done = True
        elif ext == "set":
            set_done = True
        if fdt_done and set_done:
            done_cnt += 1
            set_done = False
            fdt_done = False
        if done_cnt == no_subjects:
            break
    logging.info("All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LEMON data downloader")
    parser.add_argument(
        "num_subjects",
        type=int,
        help="number of subjects to download (in order by their ID), if `0`, "
        "will download all",
    )
    parser.add_argument(
        "download_folder", type=str, help="where to download the data"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="EC",
        help="data type to download: EC - eyes closed; EO - eyes open; "
        "all - all data",
    )
    args = parser.parse_args()
    download(args.num_subjects, args.download_folder, args.type)
