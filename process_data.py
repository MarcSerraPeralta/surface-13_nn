#!/usr/bin/env python
# coding: utf-8
print("Importing libraries...")

import numpy as np
import xarray as xr
import pathlib
import os
import random

from delft_data.process_data import process_data
from qrennd import Layout

# Load raw data

DATA_DIR = pathlib.Path("delft_data")

FILE_NAMES = [f for f in os.listdir(DATA_DIR) if "Raw_data_dict_" in f]

MEAS_TYPE = float
ROUNDS = ["1_R", "2_R", "4_R", "8_R", "16_R"]

OUTPUT_PREFIX = "rot_surf-code-13_DiCarlo_qubit_IQ_leak_bZ_s{init}_r{rounds}"
OUTPUT_DIR = pathlib.Path("nn_data")

PARTITIONS = {
    "test": 5_000,  # number of shots reserved for testing
    "train": 0.90,  # percentage of the remaining shots
    "dev": None,  # all the remaining shots
}


###################################

(OUTPUT_DIR / "all").mkdir(exist_ok=True, parents=True)


# Process data
print("Processing data...\n")

for filename in FILE_NAMES:
    print(filename)
    print("Loading data... ", end="")
    raw_data_dict = np.load(DATA_DIR / filename, allow_pickle=True).item()
    print("Done")

    proc_data_dict = process_data(raw_data_dict)

    DICARLO_DATA = proc_data_dict["Shots_exp"][0]
    DICARLO_DATA_LEAKAGE = proc_data_dict["Shots_qutrit"][0]

    print("Formatting data to xarray... ", end="")

    for rounds in ROUNDS:
        rounds_int = int(rounds.split("_")[0])
        shots_int = len(DICARLO_DATA["Z1"][rounds]["round {}".format(rounds_int)])

        # COORDINATES
        data_qubit = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]
        anc_qubit = ["Z1", "Z2", "Z3", "Z4"]
        shot = list(range(1, shots_int + 1))
        qec_round = list(range(1, rounds_int + 1))
        rot_basis = False  # from the anc_qubits, i.e. they are the Z stabilizers
        meas_reset = False  # from the way they process the defects

        # VARIABLES
        anc_meas = np.zeros((shots_int, rounds_int, len(anc_qubit), 2))
        data_meas = np.zeros((shots_int, len(data_qubit), 2))
        ideal_data_meas = np.zeros(len(data_qubit))
        ideal_anc_meas = np.zeros((rounds_int, len(anc_qubit)))
        data_init = np.zeros(len(data_qubit))
        post_selection_leakage = np.zeros(shots_int)
        anc_leakage_flag = np.zeros((shots_int, rounds_int, len(anc_qubit)))
        data_leakage_flag = np.zeros((shots_int, len(data_qubit)))

        # GET VARIABLES
        # measurement outcomes
        for k, qubit in enumerate(data_qubit):
            if qubit in filename:
                data_init[k] = 1
            else:
                data_init[k] = 0

        for q_id, qubit in enumerate(anc_qubit):
            for r_id, r in enumerate(qec_round):
                anc_meas[:, r_id, q_id] = DICARLO_DATA[qubit][rounds][f"round {r}"]

        for q_id, qubit in enumerate(data_qubit):
            data_meas[:, q_id] = DICARLO_DATA[qubit][rounds][f"round {rounds_int}"]

        for q_id, qubit in enumerate(data_qubit):
            # the ^ True shouldn't be there but if not
            # the data outcomes in the first round (probability of having
            # a single error is low) do not match the ideal ones
            # and same for the other rounds
            ideal_data_meas[q_id] = bool(data_init[q_id]) ^ bool(rounds_int % 2) ^ True

        # ideal_anc_meas = 0 for the way they build the quantum circuit

        # leakage flags
        for q_id, qubit in enumerate(anc_qubit):
            for r_id, r in enumerate(qec_round):
                anc_leakage_flag[:, r_id, q_id] = (
                    DICARLO_DATA_LEAKAGE[qubit][rounds][f"round {r}"] == 2
                )

        for q_id, qubit in enumerate(data_qubit):
            data_leakage_flag[:, q_id] = (
                DICARLO_DATA_LEAKAGE[qubit][rounds][f"round {rounds_int}"] == 2
            )

        # leakage post selection of the shots
        post_selection_leakage = np.isnan(
            proc_data_dict["Shots_qubit_ps"][0]["D1"][rounds][f"round {rounds_int}"]
        )

        # GET SOFT INFORMATION PARAMETERS
        # cannot store them as dictonaries due to xarray problem with attributes
        # and with lists of lists
        qubits = data_qubit + anc_qubit
        params_qubit = {
            qubit: proc_data_dict[qubit]["classifier_qubit"].params()
            for qubit in qubits
        }
        params = ["mu_0", "mu_1", "sigma", "angle"]
        pdf_0_params = [[params_qubit[qubit][0][k] for k in params] for qubit in qubits]
        pdf_1_params = [[params_qubit[qubit][1][k] for k in params] for qubit in qubits]
        rot_angles = [params_qubit[qubit]["rot_angle"] for qubit in qubits]
        thresholds = [params_qubit[qubit]["threshold"] for qubit in qubits]

        # GENERATE DATASET
        dataset = xr.Dataset(
            data_vars=dict(
                anc_meas=(
                    ["shot", "qec_round", "anc_qubit", "iq"],
                    anc_meas.astype(MEAS_TYPE),
                ),
                data_meas=(["shot", "data_qubit", "iq"], data_meas.astype(MEAS_TYPE)),
                ideal_anc_meas=(
                    ["qec_round", "anc_qubit"],
                    ideal_anc_meas.astype(bool),
                ),
                ideal_data_meas=(
                    [
                        "data_qubit",
                    ],
                    ideal_data_meas.astype(bool),
                ),
                anc_leakage_flag=(
                    ["shot", "qec_round", "anc_qubit"],
                    anc_leakage_flag.astype(bool),
                ),
                data_leakage_flag=(
                    ["shot", "data_qubit"],
                    data_leakage_flag.astype(bool),
                ),
                data_init=(
                    [
                        "data_qubit",
                    ],
                    data_init.astype(bool),
                ),
                pdf_0_params=(
                    [
                        "qubit",
                        "param",
                    ],
                    pdf_0_params,
                ),
                pdf_1_params=(
                    [
                        "qubit",
                        "param",
                    ],
                    pdf_1_params,
                ),
                thresholds=(
                    [
                        "qubit",
                    ],
                    thresholds,
                ),
                rot_angles=(
                    [
                        "qubit",
                    ],
                    rot_angles,
                ),
            ),
            coords=dict(
                data_qubit=data_qubit,
                anc_qubit=anc_qubit,
                shot=shot,
                qec_round=qec_round,
                rot_basis=rot_basis,
                meas_reset=meas_reset,
                qubit=qubits,
                param=params,
                iq=["I", "Q"],
            ),
            attrs=dict(
                data_type="Shots_exp",
                pdf_model="simple_1d_gaussian",
            ),
        )

        da_leakage_ps = xr.DataArray(data=post_selection_leakage)

        # STORE DATASET
        init_bitstring = "".join(map(str, data_init.astype(int)))
        FOLDER = (
            OUTPUT_DIR
            / "all"
            / OUTPUT_PREFIX.format(init=init_bitstring, rounds=rounds_int)
        )
        FOLDER.mkdir(exist_ok=True)
        dataset.to_netcdf(FOLDER / "measurements.nc")
        da_leakage_ps.to_netcdf(FOLDER / "leakage_ps.nc")
    print("\n")

# Split data into training, validation, and testing

FOLDERS = sorted(os.listdir(OUTPUT_DIR / "all"))

np.random.seed(9999)  # reproducible results

for partition in PARTITIONS:
    DATASET_PATH = OUTPUT_DIR / partition
    DATASET_PATH.mkdir(exist_ok=True)

TOTAL_SHOTS = {k: 0 for k in PARTITIONS}

layout = Layout.from_yaml(DATA_DIR / "d3_rotated_layout_surface-13Z.yaml")

for folder in FOLDERS:
    print(folder, end="\r")
    dataset = xr.load_dataset(OUTPUT_DIR / "all" / folder / "measurements.nc")

    total_shots = len(dataset.shot)
    used_shots = np.zeros(total_shots).astype(bool)  # mask = 1 the shot has been used

    # TEST
    non_used_idx = np.nonzero(~used_shots)[0]
    assert len(non_used_idx) >= PARTITIONS["test"]
    np.random.shuffle(non_used_idx)
    test_idx = non_used_idx[: PARTITIONS["test"]]
    partition_dataset = dataset.isel(shot=test_idx)

    # save
    DATASET_PATH = OUTPUT_DIR / "test" / folder
    DATASET_PATH.mkdir(exist_ok=True)
    partition_dataset.to_netcdf(DATASET_PATH / "measurements.nc")
    TOTAL_SHOTS["test"] += len(test_idx)

    # update used shots
    used_shots[test_idx] = True

    # TRAIN
    non_used_idx = np.nonzero(~used_shots)[0]
    np.random.shuffle(non_used_idx)
    train_idx = non_used_idx[: int(len(non_used_idx) * PARTITIONS["train"])]
    partition_dataset = dataset.isel(shot=train_idx)

    # save
    DATASET_PATH = OUTPUT_DIR / "train" / folder
    DATASET_PATH.mkdir(exist_ok=True)
    partition_dataset.to_netcdf(DATASET_PATH / "measurements.nc")
    TOTAL_SHOTS["train"] += len(train_idx)

    # update used shots
    used_shots[train_idx] = True

    # DEV
    non_used_idx = np.nonzero(~used_shots)[0]
    dev_idx = non_used_idx
    partition_dataset = dataset.isel(shot=dev_idx)

    # save
    DATASET_PATH = OUTPUT_DIR / "dev" / folder
    DATASET_PATH.mkdir(exist_ok=True)
    partition_dataset.to_netcdf(DATASET_PATH / "measurements.nc")
    TOTAL_SHOTS["dev"] += len(dev_idx)

    # update used shots
    used_shots[dev_idx] = True

    # ASSERTIONS
    assert used_shots.all()

    # copy layout file in config directory
    for split in ["train", "dev", "test"]:
        CONFIG_PATH = OUTPUT_DIR / split / "config"
        CONFIG_PATH.mkdir(exist_ok=True)
        layout.to_yaml(CONFIG_PATH / "d3_rotated_layout_surface-13Z.yaml")


print("\nTOTAL NUMBER OF SHOTS")
for key, value in TOTAL_SHOTS.items():
    print(key, value)
