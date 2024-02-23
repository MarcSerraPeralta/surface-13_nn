import json
from collections import defaultdict
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

columnwidth = 3.404

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rcParams.update({"text.usetex": True, "font.family": "Computer Modern Roman"})

plt.rc("font", size=BIGGER_SIZE)
plt.rc("axes", titlesize=BIGGER_SIZE)
plt.rc("axes", labelsize=BIGGER_SIZE)
plt.rc("xtick", labelsize=MEDIUM_SIZE)
plt.rc("ytick", labelsize=MEDIUM_SIZE)
plt.rc("legend", fontsize=SMALL_SIZE)
plt.rc("figure", titlesize=MEDIUM_SIZE)

ROUNDS = [1, 2, 4, 8, 16]

STATES = [
    [],  # I
    ["D1", "D2"],  # X1
    ["D8", "D9"],  # X4
    ["D2", "D3", "D5", "D6"],  # X2
    ["D4", "D5", "D7", "D8"],  # X3
    ["D1", "D3", "D5", "D6"],  # X1 X2
    ["D4", "D5", "D7", "D9"],  # X3 X4
    ["D1", "D2", "D8", "D9"],  # X1 X4
    ["D2", "D3", "D4", "D6", "D7", "D8"],  # X2 X3
    ["D1", "D2", "D4", "D5", "D7", "D8"],  # X1 X3
    ["D2", "D3", "D5", "D6", "D8", "D9"],  # X2 X4
    ["D1", "D3", "D4", "D6", "D7", "D8"],  # X1 X2 X3
    ["D2", "D3", "D4", "D6", "D7", "D9"],  # X2 X3 X4
    ["D1", "D3", "D5", "D6", "D8", "D9"],  # X1 X2 X4
    ["D1", "D2", "D4", "D5", "D7", "D9"],  # X1 X3 X4
    ["D1", "D3", "D4", "D6", "D7", "D9"],  # X1 X2 X3 X4
]

STATE_LABELS = [
    "$|000000000\\rangle$",
    "$|110000000\\rangle$",
    "$|000000011\\rangle$",
    "$|011011000\\rangle$",
    "$|110000011\\rangle$",
    "$|101011000\\rangle$",
    "$|000110110\\rangle$",
    "$|000110101\\rangle$",
    "$|110110110\\rangle$",
    "$|110110101\\rangle$",
    "$|101101110\\rangle$",
    "$|101101101\\rangle$",
    "$|101011011\\rangle$",
    "$|011101110\\rangle$",
    "$|011101101\\rangle$",
    "$|011011011\\rangle$",
]


def get_exp_fit_with_errorbars(
    logical_fails_per_round: npt.NDArray[np.int_],
    shots_per_round: npt.NDArray[np.int_],
    ROUNDS: List[int],
) -> Tuple[
    Dict[str, float],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
]:
    """
    Implement logical error rate fit usihng a method similar to Google's papers
    https://arxiv.org/pdf/2310.05900.pdf (p.40) and https://arxiv.org/pdf/2207.06431.pdf
    (p.21) but with the logical fidelity decaying to 1/2 instead of 0.

    The first round (r=0) data points are included in the fit and compensated for by
    an additional parameter r_0 to account for the increased error suppression.

    Parameters
    ----------
    logical_fails_per_round : npt.NDArray[np.int_]
        Number of logical failures per round of experiment.
    shots_per_round : npt.NDArray[np.int_]
        Number of shots per round of experiment.
    ROUNDS : List[int]
        The number of ROUNDS corresponding to each data point, ex. [1, 2, 4, 8, 16].

    Returns
    -------
    Tuple[
        Dict[str, float],
        npt.NDArray[np.float_],
        npt.NDArray[np.float_],
        npt.NDArray[np.float_],
        npt.NDArray[np.float_]
    ]
        A tuple consisting of a dictionary of the curve fit parameters, and arrays of:
        the extrapolated x-values along the rounds in the experiment, the exponential
        decay curve as found in the curve fit, the logical fidelity as calculated from
        the logical failures and the standard error of the fidelity.
    """

    def line_equation(x, a, b):
        """Line equation for curve fitting."""
        return a * x + b

    # Calculate logical fidelity (F = 1 - p_err) and its standard error
    logical_perr_per_round = logical_fails_per_round / shots_per_round
    fidelity = 1 - logical_perr_per_round
    yerr = np.sqrt(
        logical_perr_per_round * (1 - logical_perr_per_round) / shots_per_round
    )

    # Take natural log of fidelity and find a best fit line
    ln_yerr = yerr / fidelity
    popt, pcov = curve_fit(line_equation, ROUNDS, np.log(fidelity - 0.5), sigma=ln_yerr)
    a, b = popt
    perr_a, perr_b = np.sqrt(np.diag(pcov))

    # Calculate decay parameters from curve fit
    epsilon = 0.5 * (1 - np.exp(a))
    r_0 = (np.log(0.5) - b) / np.log(1 - 2 * epsilon)

    # Calculate standard error in the curve fit parameters
    err_epsilon = 0.5 * np.exp(a) * perr_a
    err_r_0 = (1 / np.log(1 - 2 * epsilon)) * perr_b

    # Store curve fit parameters in dict
    fit_params = {
        "epsilon": epsilon,
        "err_epsilon": err_epsilon,
        "r_0": r_0,
        "err_r_0": err_r_0,
    }

    # Extrapolate x values across the range of available rounds and calculate
    # y values based on an exponential decay as found in the fit
    ROUNDS_extrap = np.linspace(ROUNDS[0], ROUNDS[-1], 100)
    y_extrap = [0.5 * (1 + (1 - 2 * epsilon) ** (r - r_0)) for r in ROUNDS_extrap]

    return fit_params, ROUNDS_extrap, y_extrap, fidelity, yerr


# Choose which CLF and data to plot ============================
clf_name = "clf_gauss_1D"
with open("ler_data_NO_PS_16_states.json", "r") as f:
    ler_dict = json.load(f)

# %%
import pathlib
import xarray as xr

OUTPUT_DIR = pathlib.Path("img")
OUTPUT_NAME = "nn_performance"

# %%
DIR = pathlib.Path("../../nn_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Iterate over result dict and plot individual states ==========

state_str_to_label = {str(state): label for state, label in zip(STATES, STATE_LABELS)}

fidelity_at_r8 = defaultdict(list)
fidelity_at_r8_err = defaultdict(list)
epsilons = defaultdict(list)
epsilon_errs = defaultdict(list)
agg_shots = defaultdict(list)
agg_fails = defaultdict(list)

exp_types = ["Fails Hard Info", "Fails Soft Info"]
fig_labels = ["Hard Info", "Soft Info"]
colors = ["C0", "C1"]
linestyles = ["solid", "solid"]

fig, axs = plt.subplot_mosaic(
    [["(b)"], ["(c)"]], figsize=(1.5 * columnwidth, 2.3 * columnwidth)
)
ax0 = axs["(b)"]
ax1 = axs["(c)"]

for label, ax in axs.items():
    trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
    ax.text(
        -0.12,
        0.9,
        label,
        transform=ax.transAxes + trans,
        fontsize=BIGGER_SIZE,
        va="bottom",
        fontfamily="serif",
    )


######################

EXP_DIR = "20231219-rot_surf-code-13_DiCarlo_V3_5_IQ"
MODEL_DIR = "ensemble_lstm90x2_eval90_b64_dr0-20_lr0-0005_no-SI_no-leak"
DATASET = "test_loglikelihood.nc"
idx = 0

log_fid = xr.load_dataset(DIR / EXP_DIR / MODEL_DIR / DATASET)

for state in STATE_LABELS:
    state = state[2:11]
    fails = np.array(
        [log_fid.log_errors.sel(state=state, qec_round=r).sum().values for r in ROUNDS]
    )
    shots = np.array(
        [
            np.cumprod(log_fid.log_errors.sel(state=state, qec_round=r).values.shape)[
                -1
            ]
            for r in ROUNDS
        ]
    )

    data_label, color, linestyle = exp_types[idx], colors[idx], linestyles[idx]

    fit_params, fit_x, fit_y, fidelity_y, yerr = get_exp_fit_with_errorbars(
        fails,
        shots,
        ROUNDS,
    )
    fidelity_at_r8[data_label].append(fidelity_y[3])
    fidelity_at_r8_err[data_label].append(yerr[3])
    agg_fails[data_label].append(fails)
    agg_shots[data_label].append(shots)
    epsilons[data_label].append(eps := fit_params["epsilon"])
    epsilon_errs[data_label].append(err := fit_params["err_epsilon"])
    ax0.plot(fit_x, fit_y, alpha=0.1, linestyle=linestyle, color=color)

######################

EXP_DIR = "20231219-rot_surf-code-13_DiCarlo_V3_5_IQ"
MODEL_DIR = "ensemble_encoder100x2_lstm100x3_eval100_64_dr0-20-all_lr0-0002_SI-anc_leak_continue"
DATASET = "test_loglikelihood.nc"
idx = 1

log_fid = xr.load_dataset(DIR / EXP_DIR / MODEL_DIR / DATASET)

for state in STATE_LABELS:
    state = state[2:11]
    fails = np.array(
        [log_fid.log_errors.sel(state=state, qec_round=r).sum().values for r in ROUNDS]
    )
    shots = np.array(
        [
            np.cumprod(log_fid.log_errors.sel(state=state, qec_round=r).values.shape)[
                -1
            ]
            for r in ROUNDS
        ]
    )

    data_label, color, linestyle = exp_types[idx], colors[idx], linestyles[idx]

    fit_params, fit_x, fit_y, fidelity_y, yerr = get_exp_fit_with_errorbars(
        fails,
        shots,
        ROUNDS,
    )
    fidelity_at_r8[data_label].append(fidelity_y[3])
    fidelity_at_r8_err[data_label].append(yerr[3])
    agg_fails[data_label].append(fails)
    agg_shots[data_label].append(shots)
    epsilons[data_label].append(eps := fit_params["epsilon"])
    epsilon_errs[data_label].append(err := fit_params["err_epsilon"])
    ax0.plot(fit_x, fit_y, alpha=0.1, linestyle=linestyle, color=color)

# Plot aggregated logical fidelity across all 16 states
for exp, color, linestyle, fig_label in zip(exp_types, colors, linestyles, fig_labels):
    fails = np.sum(np.array(agg_fails[exp]), axis=0)
    shots = np.sum(np.array(agg_shots[exp]), axis=0)
    fit_params, fit_x, fit_y, fidelity_y, yerr = get_exp_fit_with_errorbars(
        fails, shots, ROUNDS
    )
    epsilon = fit_params["epsilon"]
    err_epsilon = fit_params["err_epsilon"]
    legend = f"$\epsilon_L=({epsilon*100:.2f} \pm {err_epsilon*100:.2f}) \\%$"
    ax0.plot(fit_x, fit_y, color=color, linestyle=linestyle)
    ax0.scatter(ROUNDS, fidelity_y, marker="o")
    ax0.plot(
        [],
        [],
        marker="o",
        linestyle="solid",
        label=f"{fig_label}\n" + legend,
        color=color,
    )

ax0.set_xticks(ROUNDS, labels=[str(r) for r in ROUNDS])
ax0.set_ylim(0.5, 1.02)
ax0.set_xlabel("Number of rounds")
ax0.set_ylabel("Logical fidelity")
ax0.legend(loc=1, frameon=False)

# Plot bar chart showing LER for each state

attributes = [(fidelity_at_r8[exp], fidelity_at_r8_err[exp]) for exp in exp_types]
x = np.arange(len(STATE_LABELS))
width = 0.33
multiplier = 0

for i, ler_data in enumerate(attributes):
    offset = width * multiplier
    eps, err_eps = ler_data
    ax1.bar(
        x + offset,
        eps,
        yerr=err_eps,
        width=width,
        label=fig_labels[i],
        edgecolor="black",
        linewidth=0.5,
        error_kw={"capthick": 1.5, "capsize": 3, "elinewidth": 1.5},
    )
    multiplier += 1

ax1.text(0.83, 0.9, "$R=8$", color="black", transform=ax.transAxes)
ax1.set_ylabel("Logical fidelity")
ax1.set_xlabel("Physical computational state")
ax1.set_xticks(x + width, STATE_LABELS)
ax1.set_xticklabels(
    [state_str_to_label[state] for state in sorted(ler_dict.keys())[::-1]], rotation=90
)
ax1.set_ylim(0.69, 0.79)
ax1.legend(loc=2, frameon=False, ncol=2)

plt.subplots_adjust(
    left=0.18, bottom=0.19, right=0.98, top=0.98, wspace=0.1, hspace=0.31
)
for format_ in ["pdf", "svg", "png"]:
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_NAME}.{format_}", format=format_)
plt.show()
