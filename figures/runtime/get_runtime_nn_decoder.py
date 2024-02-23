import numpy as np
import pathlib
from cProfile import Profile
from pstats import Stats

import xarray as xr

from qrennd import Config, Layout, set_coords
from qrennd import get_model


LAYOUT_FILE = pathlib.Path(
    "../../nn_data/20231219-rot_surf-code-13_DiCarlo_V3_5_IQ/config/d3_rotated_layout_surface-13Z.yaml"
)
CONFIG_FILE = pathlib.Path(
    "../../nn_output/20231219-rot_surf-code-13_DiCarlo_V3_5_IQ/20231220-094626_lstm90x2_eval90_b64_dr0-20_lr0-0005_no-SI_no-leak/config.yaml"
)

layout = Layout.from_yaml(LAYOUT_FILE)
set_coords(layout)
config = Config.from_yaml(CONFIG_FILE, "", "")


# Get model with random weights
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
print(model.summary())


# generate random data
list_rounds = [1, 2, 4, 8, 16, 32, 64, 128]
shots = 50_000
batch_size = 1

times = []
for rounds in list_rounds:
    rec_inputs = np.random.normal(size=(shots, rounds, rec_features))
    eval_inputs = np.random.normal(size=(shots, eval_features))

    model_inputs = dict(rec_input=rec_inputs, eval_input=eval_inputs)

    profiler = Profile()
    profiler.enable()
    model.predict(model_inputs, batch_size=batch_size)
    profiler.disable()
    stats = Stats(profiler)
    total_time = stats.total_tt
    print(rounds, total_time / (shots * rounds) * 1e6)  # in micro-seconds
    times.append(total_time)

da = xr.DataArray(times, coords=dict(rounds=list_rounds))
da.to_netcdf("data/runtimes.nc")
