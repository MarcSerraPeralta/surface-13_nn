import pathlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import xarray as xr

OUTPUT_DIR = pathlib.Path("img")
OUTPUT_NAME = "runtime"

# load data
da = xr.load_dataarray("data/runtimes.nc")
num_shots = 50_000  # it should be stored in the dataarray
rounds = da.rounds.values
times = da.values / num_shots * 1e6  # microseconds

# perform linear regression
reg = stats.linregress(rounds, times)
print(reg)

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
fig, ax = plt.subplots(figsize=cm2inch(8.355, 6))  # width, height


ax.plot(rounds, times, linestyle="none", color=colors["blue"], marker="o", markersize=2)

ax.plot(rounds, rounds * reg.slope + reg.intercept, linestyle="-", color="gray")

label = f"$\\Delta t$ [$\\mu$s] = ${reg.slope:0.2f}R + {reg.intercept:0.2f}$ \n\n R-squared coeff. $= {reg.rvalue**2:0.4f}$"
ax.text(
    0.055,
    0.925,
    label,
    ha="left",
    va="top",
    transform=ax.transAxes,
)

# %%
ax.set_xlabel("QEC round, $R$")
ax.set_ylabel("Runtime, $\\Delta t$ [$\\mu$s]")
# ax.set_xlim(xmin=-0.5, xmax=len(LABELS) - 0.5)
# ax.set_ylim(ymin=4.55, ymax=5.15)
# ax.set_yticks(
#    np.arange(0.5, 1.01, 0.05), np.round(np.arange(0.5, 1.01, 0.05), decimals=2)
# )
# ax.legend(loc="lower left")
# ax.grid(which="major")

# ax.set_xticks(np.arange(len(LABELS)), LABELS)

fig.tight_layout(pad=0.2)
# plt.subplots_adjust(wspace=0)  # hspace=0, wspace=0

# %%
for format_ in ["pdf", "png", "svg"]:
    fig.savefig(
        OUTPUT_DIR / (OUTPUT_NAME + f".{format_}"),
        format=format_,
    )

# %%
plt.show()
