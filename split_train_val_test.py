import xarray as xr
import pathlib
import os
import random
import numpy as np

OUTPUT_DIR = pathlib.Path("nn_data")

DATASET = "all"

PARTITIONS = {
    "test": 5_000, # number of shots reserved for testing
    "train": 0.90, # percentage of the remaining shots
    "dev": None, # all the remaining shots
}  

FOLDERS = sorted(os.listdir(OUTPUT_DIR / DATASET))

##################################################

np.random.seed(9999)  # reproducible results

for partition in PARTITIONS:
    DATASET_PATH = OUTPUT_DIR / partition
    DATASET_PATH.mkdir(exist_ok=True)

TOTAL_SHOTS = {k: 0 for k in PARTITIONS}

for folder in FOLDERS:
    print(folder, end="\r")
    dataset = xr.load_dataset(OUTPUT_DIR / DATASET / folder / "measurements.nc")

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


print("\nTOTAL NUMBER OF SHOTS")
for key, value in TOTAL_SHOTS.items():
    print(key, value)
