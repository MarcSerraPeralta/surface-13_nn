# %%
import pathlib
import numpy as np
import xarray as xr
from itertools import product

import warnings

# Ignore all warnings (pandas FutureWarning)
warnings.filterwarnings("ignore")

from qrennd import Config, Layout, get_model, load_datasets, set_coords

# %%
EXP_NAME = "20231219-rot_surf-code-13_DiCarlo_V3_5_IQ"
MODEL_FOLDER = "20231221-103743_lstm90x2_eval90_b64_dr0-20_lr0-0005_no-SI_no-leak"
LAYOUT_NAME = "d3_rotated_layout_surface-13Z.yaml"
TEST_DATASET = ["test"]

# %%
NOTEBOOK_DIR = pathlib.Path.cwd()  # define the path where the notebook is placed.

DATA_DIR = NOTEBOOK_DIR / "nn_data"
if not DATA_DIR.exists():
    raise ValueError(f"Data directory does not exist: {DATA_DIR}")

OUTPUT_DIR = NOTEBOOK_DIR / "nn_output"
if not OUTPUT_DIR.exists():
    raise ValueError(f"Output directory does not exist: {OUTPUT_DIR}")

CONFIG_FILE = OUTPUT_DIR / EXP_NAME / MODEL_FOLDER / "config.yaml"
if not CONFIG_FILE.exists():
    raise ValueError(f"Config file does not exist: {CONFIG_FILE}")

LAYOUT_FILE = DATA_DIR / EXP_NAME / "config" / f"{LAYOUT_NAME}"
if not LAYOUT_FILE.exists():
    raise ValueError(f"Layout file does not exist: {LAYOUT_FILE}")

# %% [markdown]
# # Evaluation


# %%
def evaluate_model(model, config, layout, dataset_name="test"):
    test_data = load_datasets(
        config=config, layout=layout, dataset_name=dataset_name, concat=False
    )
    rounds = config.dataset[dataset_name]["rounds"]
    states = config.dataset[dataset_name]["states"]
    num_shots = config.dataset[dataset_name]["shots"]
    sequences = product(rounds, states)
    list_log_errors = []
    list_log_flips = []
    list_predictions = []

    for data, (num_rounds, state) in zip(test_data, sequences):
        print(f"QEC = {num_rounds} | state = {state}", end="\r")
        prediction = model.predict(data, verbose=0)
        prediction = prediction[0].flatten()
        errors = (prediction > 0.5) != data.log_errors
        list_log_flips.append(data.log_errors)
        list_predictions.append(prediction)
        list_log_errors.append(errors)
        print(
            f"QEC = {num_rounds} | state = {state} | p_L={np.average(errors)}", end="\r"
        )

    list_log_flips = np.array(list_log_flips).reshape(
        len(rounds), len(states), num_shots
    )
    list_predictions = np.array(list_predictions).reshape(
        len(rounds), len(states), num_shots
    )
    list_log_errors = np.array(list_log_errors).reshape(
        len(rounds), len(states), num_shots
    )

    log_fid = xr.Dataset(
        data_vars=dict(
            log_flips=(["qec_round", "state", "shot"], list_log_flips),
            predictions=(["qec_round", "state", "shot"], list_predictions),
            log_errors=(["qec_round", "state", "shot"], list_log_errors),
        ),
        coords=dict(qec_round=rounds, state=states, shot=list(range(1, num_shots + 1))),
    )

    return log_fid


# %%
layout = Layout.from_yaml(LAYOUT_FILE)
set_coords(layout)
config = Config.from_yaml(
    filepath=CONFIG_FILE,
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
)
if not isinstance(TEST_DATASET, list):
    TEST_DATASET = [TEST_DATASET]

# %%
# if results have not been stored, evaluate model
DIR = OUTPUT_DIR / EXP_NAME / MODEL_FOLDER
for test_dataset in TEST_DATASET:
    NAME = f"{test_dataset}.nc"
    if not (DIR / NAME).exists():
        print("Evaluating model...")

        anc_qubits = layout.get_qubits(role="anc")
        num_anc = len(anc_qubits)
        data_qubits = layout.get_qubits(role="data")
        num_data = len(data_qubits)

        rec_features = num_anc
        eval_features = num_anc  # ancillas are only the Z-type stabilizers

        if config.dataset["input"] == "IQ":
            leakage = config.dataset.get("leakage")
            if leakage["anc"]:
                rec_features += num_anc
            if leakage["data"]:
                eval_features += num_data

        model = get_model(
            rec_features=rec_features,
            eval_features=eval_features,
            config=config,
        )

        model.load_weights(DIR / "checkpoint" / "weights.hdf5")
        ds = evaluate_model(model, config, layout, test_dataset)
        ds.to_netcdf(path=DIR / NAME)

        print("Done!")

    else:
        print("Model already evaluated!")

    print("\nRESULTS IN:")
    print("output_dir=", NOTEBOOK_DIR)
    print("exp_name=", EXP_NAME)
    print("run_name=", MODEL_FOLDER)
    print("test_data=", NAME)
