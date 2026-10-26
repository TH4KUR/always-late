"""Render report plots from committed results, without rerunning analyses."""
from pathlib import Path
import csv
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "reports/functional_validation"
plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 9,
    "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.pad_inches": 0.03,
})


def rows(relative):
    with (RESULTS / relative).open() as stream:
        return list(csv.DictReader(stream))


def save(fig, name):
    fig.savefig(HERE / "figures" / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(HERE / "figures" / (name + ".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    (HERE / "figures").mkdir(exist_ok=True)
    data = rows("figures/f5/ladder_control_table.csv")
    fig, ax = plt.subplots(figsize=(3.35, 2.1), layout="constrained")
    x = list(range(len(data)))
    for label, stem, color, offset in [
        ("Feature", "features", "#17629a", -0.055),
        ("Other direction", "control", "#b95726", 0.055),
    ]:
        values = [float(r["median_" + stem]) for r in data]
        lower = [v - float(r["q25_" + stem]) for r, v in zip(data, values)]
        upper = [float(r["q75_" + stem]) - v for r, v in zip(data, values)]
        ax.errorbar([v + offset for v in x], values, yerr=[lower, upper],
                    fmt="o-", color=color, markersize=3, capsize=2, label=label)
    ax.set_xticks(x, [r["state"].replace("INT", "") for r in data])
    ax.set(xlabel="Weight precision (bits)", ylabel="Effect cosine", ylim=(0.72, 1.01))
    ax.legend(frameon=False, loc="lower left")
    save(fig, "direction-control")

    data = rows("gain_study/results/curves.csv")
    fig, ax = plt.subplots(figsize=(3.35, 2.05), layout="constrained")
    doses = [1.0, 0.3535533905932738, 0.1767766952966369, 0.08838834764831845, 0.0]
    for state, color, marker in [("FP", "#252525", "o"), ("INT6", "#26835a", "s"),
                                 ("INT5", "#17629a", "^"), ("INT4", "#b95726", "D")]:
        medians = [statistics.median(float(r["net"]) for r in data
                   if r["state"] == state and abs(float(r["ratio"]) - dose) < 1e-8)
                   for dose in doses]
        ax.plot(range(len(doses)), medians, marker=marker, color=color,
                markersize=3, linewidth=1, label=state)
    ax.set_xticks(range(len(doses)), ["1", ".354", ".177", ".088", "0"])
    ax.set(xlabel="Activation multiplier (stronger suppression →)",
           ylabel="Net harm (nats/token)")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    save(fig, "gain-down")

    data = rows("gain_study/extension_c/bands.csv")
    fig, ax = plt.subplots(figsize=(3.35, 2.1), layout="constrained")
    for index, (state, color) in enumerate([("FP", "#252525"), ("INT6", "#26835a"),
                                           ("INT5", "#17629a"), ("INT4", "#b95726")]):
        selected = [next(r for r in data if r["state"] == state and float(r["ratio"]) == dose)
                    for dose in [1.1, 1.2]]
        values = [1000 * float(r["median"]) for r in selected]
        lower = [v - 1000 * float(r["lo"]) for r, v in zip(selected, values)]
        upper = [1000 * float(r["hi"]) - v for r, v in zip(selected, values)]
        offset = (index - 1.5) * .11
        ax.errorbar([offset, 1 + offset], values, yerr=[lower, upper],
                    fmt="o", color=color, markersize=3, capsize=2, label=state)
    ax.axhline(0, color="0.55", linestyle="--", linewidth=.7)
    ax.set_xticks([0, 1], ["1.1", "1.2"])
    ax.set(xlabel="Activation multiplier", ylabel="Net harm (10⁻³ nats/token)", xlim=(-.4, 1.4))
    ax.legend(frameon=False, ncol=2, loc="lower left")
    save(fig, "fine-gain")


if __name__ == "__main__":
    main()
