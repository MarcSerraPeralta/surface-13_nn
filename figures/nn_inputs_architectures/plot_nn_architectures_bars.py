# %%
import pathlib
import copy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import xarray as xr
import yaml
from lmfit import Model

from qrennd.utils.analysis import (
    LogicalErrorProb,
    lmfit_par_to_ufloat,
)


# %%
OUTPUT_DIR = pathlib.Path("img")
OUTPUT_NAME = "nn_architectures_bars"

# %%
DIR = pathlib.Path("../../nn_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LABELS = ["size 1", "size 2", "size 3"]

# Set matplotlib parameters
import numpy as np
import matplotlib

matplotlib.rcParams.update(
    {
        "font.size": 10,
        "font.family": "Latin Modern Roman",
        "font.weight": "normal",
        "text.usetex": True,
    }
)

colors = {
    "red": "#e41a1cff",
    "green": "#4daf4aff",
    "blue": "#377eb8ff",
    "orange": "#ff9933ff",
    "purple": "#984ea3ff",
    "yellow": "#f2c829ff",
}


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def print_params(value, error):
    decimals = 0
    while error < 0.95:
        decimals += 1
        error *= 10

    error = int(np.round(error, 0))
    value = np.round(value, decimals)

    string = "{value:.{decimals}f}({error})".format(
        value=value, decimals=decimals, error=error
    )

    return string


# %%
fig, (ax1, ax2, ax3, ax4) = plt.subplots(
    figsize=cm2inch(17.0637, 5),
    ncols=4,
    gridspec_kw={"width_ratios": [2, 3, 3, 5], "height_ratios": [1]},
)


# %%
def get_log_err_rate(x, y):
    def func(qec_round, err_rate=0.1, round_offset=0):
        return 0.5 * (1 + (1 - 2 * err_rate) ** (qec_round - round_offset))

    log_decay_model = Model(func)

    fit = log_decay_model.fit(y, qec_round=x)

    error_rate = lmfit_par_to_ufloat(fit.params["err_rate"])
    t0 = lmfit_par_to_ufloat(fit.params["round_offset"])

    return error_rate, t0


###########################################################################

LABELS = []

# %%
EXP_DIR = "20231219-rot_surf-code-13_DiCarlo_V3_5_IQ"
MODEL_DIR = "20231221-103743_lstm90x2_eval90_b64_dr0-20_lr0-0005_no-SI_no-leak"
DATASET = "test.nc"
COLOR = "C0"
LABEL_DATA = "size 1"

log_fid = xr.load_dataset(DIR / EXP_DIR / MODEL_DIR / DATASET)

x = log_fid.qec_round.values
y = 1 - log_fid.log_errors.mean(dim=["shot", "state"]).values
y_err = log_fid.log_errors.std(dim=["shot", "state"]).values / np.sqrt(
    len(log_fid.shot) * len(log_fid.state)
)
error_rate, t0 = get_log_err_rate(x, y)

ax1.bar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    color=COLOR,
)
ax1.errorbar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    yerr=[(error_rate * 100).std_dev],
    color="black",
    capsize=5,
)
LABELS.append(LABEL_DATA)

# %%
EXP_DIR = "20231219-rot_surf-code-13_DiCarlo_V3_5_IQ"
MODEL_DIR = "20240114-114308_encoder32x2_lstm100x2_eval100_64_dr0-20-all_lr0-0002_no-SI-anc_no-leak"
DATASET = "test.nc"
COLOR = "C0"
LABEL_DATA = "size 2"

log_fid = xr.load_dataset(DIR / EXP_DIR / MODEL_DIR / DATASET)

x = log_fid.qec_round.values
y = 1 - log_fid.log_errors.mean(dim=["shot", "state"]).values
y_err = log_fid.log_errors.std(dim=["shot", "state"]).values / np.sqrt(
    len(log_fid.shot) * len(log_fid.state)
)
error_rate, t0 = get_log_err_rate(x, y)

ax1.bar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    color=COLOR,
)
ax1.errorbar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    yerr=[(error_rate * 100).std_dev],
    color="black",
    capsize=5,
)
LABELS.append(LABEL_DATA)

ax1.set_xticks(np.arange(len(LABELS)), LABELS)
ax1.set_xlim(xmin=-0.75, xmax=len(LABELS) - 0.25)

#########################################################################################

LABELS = []

# %%
EXP_DIR = "20231219-rot_surf-code-13_DiCarlo_V3_5_IQ"
MODEL_DIR = "20240114-161515_lstm90x2_eval90_b64_dr0-20_lr0-0005_no-SI_leak"
DATASET = "test.nc"
COLOR = "C0"
LABEL_DATA = "size 1"

log_fid = xr.load_dataset(DIR / EXP_DIR / MODEL_DIR / DATASET)

x = log_fid.qec_round.values
y = 1 - log_fid.log_errors.mean(dim=["shot", "state"]).values
y_err = log_fid.log_errors.std(dim=["shot", "state"]).values / np.sqrt(
    len(log_fid.shot) * len(log_fid.state)
)
error_rate, t0 = get_log_err_rate(x, y)

ax2.bar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    color=COLOR,
)
ax2.errorbar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    yerr=[(error_rate * 100).std_dev],
    color="black",
    capsize=5,
)
LABELS.append(LABEL_DATA)

# %%
EXP_DIR = "20231219-rot_surf-code-13_DiCarlo_V3_5_IQ"
MODEL_DIR = (
    "20231225-000208_encoder32x2_lstm100x2_eval100_b64_dr0-20_lr0-0002_no-SI_leak"
)
DATASET = "test.nc"
COLOR = "C0"
LABEL_DATA = "size 2"

log_fid = xr.load_dataset(DIR / EXP_DIR / MODEL_DIR / DATASET)

x = log_fid.qec_round.values
y = 1 - log_fid.log_errors.mean(dim=["shot", "state"]).values
y_err = log_fid.log_errors.std(dim=["shot", "state"]).values / np.sqrt(
    len(log_fid.shot) * len(log_fid.state)
)
error_rate, t0 = get_log_err_rate(x, y)

ax2.bar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    color=COLOR,
)
ax2.errorbar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    yerr=[(error_rate * 100).std_dev],
    color="black",
    capsize=5,
)
LABELS.append(LABEL_DATA)

# %%
EXP_DIR = "20231219-rot_surf-code-13_DiCarlo_V3_5_IQ"
MODEL_DIR = "20240113-192604_encoder64x2_lstm120x2_eval120_64_dr0-20-all_lr0-0002_no-SI-anc_leak"
DATASET = "test.nc"
COLOR = "C0"
LABEL_DATA = "size 3"

log_fid = xr.load_dataset(DIR / EXP_DIR / MODEL_DIR / DATASET)

x = log_fid.qec_round.values
y = 1 - log_fid.log_errors.mean(dim=["shot", "state"]).values
y_err = log_fid.log_errors.std(dim=["shot", "state"]).values / np.sqrt(
    len(log_fid.shot) * len(log_fid.state)
)
error_rate, t0 = get_log_err_rate(x, y)

ax2.bar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    color=COLOR,
)
ax2.errorbar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    yerr=[(error_rate * 100).std_dev],
    color="black",
    capsize=5,
)
LABELS.append(LABEL_DATA)

ax2.set_xticks(np.arange(len(LABELS)), LABELS)
ax2.set_xlim(xmin=-0.75, xmax=len(LABELS) - 0.25)

#########################################################################################

LABELS = []

# %%
EXP_DIR = "20231219-rot_surf-code-13_DiCarlo_V3_5_IQ"
MODEL_DIR = "20240115-100331_lstm90x2_eval90_b64_dr0-20_lr0-0005_SI_no-leak"
DATASET = "test.nc"
COLOR = "C0"
LABEL_DATA = "size 1"

log_fid = xr.load_dataset(DIR / EXP_DIR / MODEL_DIR / DATASET)

x = log_fid.qec_round.values
y = 1 - log_fid.log_errors.mean(dim=["shot", "state"]).values
y_err = log_fid.log_errors.std(dim=["shot", "state"]).values / np.sqrt(
    len(log_fid.shot) * len(log_fid.state)
)
error_rate, t0 = get_log_err_rate(x, y)

ax3.bar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    color=COLOR,
)
ax3.errorbar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    yerr=[(error_rate * 100).std_dev],
    color="black",
    capsize=5,
)
LABELS.append(LABEL_DATA)

# %%
EXP_DIR = "20231219-rot_surf-code-13_DiCarlo_V3_5_IQ"
MODEL_DIR = (
    "20231223-000053_encoder32x2_lstm100x2_eval100_b64_dr0-20_lr0-0002_SI-anc_no-leak"
)
DATASET = "test.nc"
COLOR = "C0"
LABEL_DATA = "size 2"

log_fid = xr.load_dataset(DIR / EXP_DIR / MODEL_DIR / DATASET)

x = log_fid.qec_round.values
y = 1 - log_fid.log_errors.mean(dim=["shot", "state"]).values
y_err = log_fid.log_errors.std(dim=["shot", "state"]).values / np.sqrt(
    len(log_fid.shot) * len(log_fid.state)
)
error_rate, t0 = get_log_err_rate(x, y)

ax3.bar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    color=COLOR,
)
ax3.errorbar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    yerr=[(error_rate * 100).std_dev],
    color="black",
    capsize=5,
)
LABELS.append(LABEL_DATA)

# %%
EXP_DIR = "20231219-rot_surf-code-13_DiCarlo_V3_5_IQ"
MODEL_DIR = "20240114-085821_encoder64x2_lstm120x2_eval120_64_dr0-20-all_lr0-0002_SI-anc_no-leak"
DATASET = "test.nc"
COLOR = "C0"
LABEL_DATA = "size 3"

log_fid = xr.load_dataset(DIR / EXP_DIR / MODEL_DIR / DATASET)

x = log_fid.qec_round.values
y = 1 - log_fid.log_errors.mean(dim=["shot", "state"]).values
y_err = log_fid.log_errors.std(dim=["shot", "state"]).values / np.sqrt(
    len(log_fid.shot) * len(log_fid.state)
)
error_rate, t0 = get_log_err_rate(x, y)

ax3.bar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    color=COLOR,
)
ax3.errorbar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    yerr=[(error_rate * 100).std_dev],
    color="black",
    capsize=5,
)
LABELS.append(LABEL_DATA)

ax3.set_xticks(np.arange(len(LABELS)), LABELS)
ax3.set_xlim(xmin=-0.75, xmax=len(LABELS) - 0.25)

#########################################################################################

LABELS = []

# %%
EXP_DIR = "20231219-rot_surf-code-13_DiCarlo_V3_5_IQ"
MODEL_DIR = "20240114-215800_lstm90x2_eval90_b64_dr0-20_lr0-0005_SI_leak"
DATASET = "test.nc"
COLOR = "C0"
LABEL_DATA = "size 1"

log_fid = xr.load_dataset(DIR / EXP_DIR / MODEL_DIR / DATASET)

x = log_fid.qec_round.values
y = 1 - log_fid.log_errors.mean(dim=["shot", "state"]).values
y_err = log_fid.log_errors.std(dim=["shot", "state"]).values / np.sqrt(
    len(log_fid.shot) * len(log_fid.state)
)
error_rate, t0 = get_log_err_rate(x, y)

ax4.bar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    color=COLOR,
)
ax4.errorbar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    yerr=[(error_rate * 100).std_dev],
    color="black",
    capsize=5,
)
LABELS.append(LABEL_DATA)

# %%
EXP_DIR = "20231219-rot_surf-code-13_DiCarlo_V3_5_IQ"
MODEL_DIR = (
    "20240113-192156_encoder32x2_lstm100x2_eval100_64_dr0-20-all_lr0-0002_SI-anc_leak"
)
DATASET = "test.nc"
COLOR = "C0"
LABEL_DATA = "size 2"

log_fid = xr.load_dataset(DIR / EXP_DIR / MODEL_DIR / DATASET)

x = log_fid.qec_round.values
y = 1 - log_fid.log_errors.mean(dim=["shot", "state"]).values
y_err = log_fid.log_errors.std(dim=["shot", "state"]).values / np.sqrt(
    len(log_fid.shot) * len(log_fid.state)
)
error_rate, t0 = get_log_err_rate(x, y)

ax4.bar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    color=COLOR,
)
ax4.errorbar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    yerr=[(error_rate * 100).std_dev],
    color="black",
    capsize=5,
)
LABELS.append(LABEL_DATA)

# %%
EXP_DIR = "20231219-rot_surf-code-13_DiCarlo_V3_5_IQ"
MODEL_DIR = (
    "20240113-192028_encoder64x2_lstm120x2_eval120_64_dr0-20-all_lr0-0002_SI-anc_leak"
)
DATASET = "test.nc"
COLOR = "C0"
LABEL_DATA = "size 3"

log_fid = xr.load_dataset(DIR / EXP_DIR / MODEL_DIR / DATASET)

x = log_fid.qec_round.values
y = 1 - log_fid.log_errors.mean(dim=["shot", "state"]).values
y_err = log_fid.log_errors.std(dim=["shot", "state"]).values / np.sqrt(
    len(log_fid.shot) * len(log_fid.state)
)
error_rate, t0 = get_log_err_rate(x, y)

ax4.bar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    color=COLOR,
)
ax4.errorbar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    yerr=[(error_rate * 100).std_dev],
    color="black",
    capsize=5,
)
LABELS.append(LABEL_DATA)

# %%
EXP_DIR = "20231219-rot_surf-code-13_DiCarlo_V3_5_IQ"
MODEL_DIR = "20240117-144524_encoder90x2_lstm100x3_eval100_64_dr0-20-all_lr0-0002_SI-anc_leak_continue"
DATASET = "test.nc"
COLOR = "C0"
LABEL_DATA = "size 4"

log_fid = xr.load_dataset(DIR / EXP_DIR / MODEL_DIR / DATASET)

x = log_fid.qec_round.values
y = 1 - log_fid.log_errors.mean(dim=["shot", "state"]).values
y_err = log_fid.log_errors.std(dim=["shot", "state"]).values / np.sqrt(
    len(log_fid.shot) * len(log_fid.state)
)
error_rate, t0 = get_log_err_rate(x, y)

ax4.bar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    color=COLOR,
)
ax4.errorbar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    yerr=[(error_rate * 100).std_dev],
    color="black",
    capsize=5,
)
LABELS.append(LABEL_DATA)

# %%
EXP_DIR = "20231219-rot_surf-code-13_DiCarlo_V3_5_IQ"
MODEL_DIR = "20240111-235859_encoder100x2_lstm100x3_eval100_64_dr0-20-all_lr0-0002_SI-anc_leak_continue"
DATASET = "test.nc"
COLOR = "C0"
LABEL_DATA = "size 5"

log_fid = xr.load_dataset(DIR / EXP_DIR / MODEL_DIR / DATASET)

x = log_fid.qec_round.values
y = 1 - log_fid.log_errors.mean(dim=["shot", "state"]).values
y_err = log_fid.log_errors.std(dim=["shot", "state"]).values / np.sqrt(
    len(log_fid.shot) * len(log_fid.state)
)
error_rate, t0 = get_log_err_rate(x, y)

ax4.bar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    color=COLOR,
)
ax4.errorbar(
    [len(LABELS)],
    [(error_rate * 100).nominal_value],
    yerr=[(error_rate * 100).std_dev],
    color="black",
    capsize=5,
)
LABELS.append(LABEL_DATA)

ax4.set_xticks(np.arange(len(LABELS)), LABELS)
ax4.set_xlim(xmin=-0.75, xmax=len(LABELS) - 0.25)

#########################################################################################

# %%
# ax.set_xlabel("QEC round, $r$")
ax1.set_ylabel("Logical error rate, $\\varepsilon_L$ [\\%]")

# ax.set_yticks(
#    np.arange(0.5, 1.01, 0.05), np.round(np.arange(0.5, 1.01, 0.05), decimals=2)
# )
# ax.legend(loc="best")
# ax.grid(which="major")

axes = {"(a)": ax1, "(b)": ax2, "(c)": ax3, "(d)": ax4}
for label, ax in axes.items():
    ax.text(
        0.925,
        0.975,
        label,
        ha="right",
        va="top",
        transform=ax.transAxes,
    )
    if label != "(a)":
        ax.tick_params(
            axis="y",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
            labelleft=False,
        )  # labels along the bottom edge are off
    ax.set_ylim(ymin=4.65, ymax=5.17)

fig.tight_layout(pad=0.2)
plt.subplots_adjust(wspace=0)  # hspace=0, wspace=0


# %%
for format_ in ["pdf", "png", "svg"]:
    fig.savefig(
        OUTPUT_DIR / (OUTPUT_NAME + f".{format_}"),
        format=format_,
    )

# %%
plt.show()
