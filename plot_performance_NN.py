# %%
import pathlib
import copy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import xarray as xr
import yaml
from lmfit import Model

from qrennd.utils.analysis import lmfit_par_to_ufloat


# %%
EXP_NAME = "20231219-rot_surf-code-13_DiCarlo_V3_5_IQ"
MODEL_FOLDER = "20231221-103743_lstm90x2_eval90_b64_dr0-20_lr0-0005_no-SI_no-leak"
DATASET = "test.nc"

TITLE = f"{EXP_NAME}\n{MODEL_FOLDER} ({DATASET})"
OUTPUT_DIR = pathlib.Path(EXP_NAME) / MODEL_FOLDER
OUTPUT_NAME = DATASET.replace(".nc", "")

FIT = True

# %%
DIR = pathlib.Path.cwd() / "nn_output"

# %%
fig, ax = plt.subplots(figsize=(7, 5))

# %%
EXP_DIR_ = EXP_NAME
RUN_DIR_ = MODEL_FOLDER
COLOR = "blue"
LABEL_DATA = "NN"
LABEL_FIT = "$\\epsilon_L$ = ({error_rate_100})%"

log_fid = xr.load_dataset(DIR / EXP_DIR_ / RUN_DIR_ / DATASET)

x = log_fid.qec_round.values
y = 1 - log_fid.log_errors.mean(dim=["shot", "state"]).values
y_err = log_fid.log_errors.std(dim=["shot", "state"]).values / np.sqrt(
    len(log_fid.shot) * len(log_fid.state)
)

print(list(y))

if LABEL_DATA:
    ax.errorbar(
        x,
        y,
        fmt=".",
        yerr=y_err,
        color=COLOR,
        markersize=10,
        label=LABEL_DATA,
    )

if FIT:

    def func(qec_round, err_rate=0.1, round_offset=0):
        return 0.5 * (1 + (1 - 2 * err_rate) ** (qec_round - round_offset))

    log_decay_model = Model(func)

    fit = log_decay_model.fit(y, qec_round=x)

    error_rate = lmfit_par_to_ufloat(fit.params["err_rate"])
    t0 = lmfit_par_to_ufloat(fit.params["round_offset"])

    x_fit = np.linspace(0, max(x), 100)
    y_fit = log_decay_model.func(x_fit, error_rate.nominal_value, t0.nominal_value)
    vars_fit = {
        "error_rate": error_rate,
        "t0": t0,
        "error_rate_100": error_rate * 100,
    }
    ax.plot(
        x_fit,
        y_fit,
        linestyle="-",
        color=COLOR,
        label=LABEL_FIT.format(**vars_fit),
    )


# %%
ax.set_xlabel("QEC round")
ax.set_ylabel("logical fidelity")
ax.set_xlim(xmin=0)
# ax.set_yscale("log")
ax.set_ylim(ymax=1, ymin=0.5)
ax.set_yticks(
    np.arange(0.5, 1.01, 0.05), np.round(np.arange(0.5, 1.01, 0.05), decimals=2)
)
ax.legend(loc="best")
ax.grid(which="major")
if TITLE:
    ax.set_title(TITLE)
fig.tight_layout()

# %%
for format_ in ["pdf", "png", "svg"]:
    fig.savefig(
        DIR / OUTPUT_DIR / (OUTPUT_NAME + f".{format_}"),
        format=format_,
    )

# %%
plt.show()
