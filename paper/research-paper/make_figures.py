"""Render committed estimates and intervals; no model runs or new inference."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.6, "savefig.pad_inches": 0.035,
})


def read_csv(relative):
    with (ROOT / relative).open() as stream:
        return list(csv.DictReader(stream))


def save(fig, name):
    fig.savefig(OUT / (name + ".pdf"))
    fig.savefig(OUT / (name + ".png"), dpi=200)
    plt.close(fig)


rows = read_csv("reports/functional_validation/results/f5/f5_tables.csv")
states = ["INT8", "INT6", "INT5", "INT4"]
selected = [next(r for r in rows if r["state"] == state and r["group"] == "all")
            for state in states]
fig, ax = plt.subplots(figsize=(3.3, 2.1), layout="constrained")
for key, label, color, marker in [
    ("R1", "Activation correlation", "#28628e", "o"),
    ("W1", "Effect cosine", "#a84c2a", "s"),
]:
    center = np.array([float(r[key + "_median"]) for r in selected])
    lower = np.array([float(r[key + "_median_lo"]) for r in selected])
    upper = np.array([float(r[key + "_median_hi"]) for r in selected])
    ax.errorbar(range(4), center, yerr=[center-lower, upper-center],
                label=label, color=color, marker=marker, markersize=4,
                capsize=2, linewidth=1)
ax.axhline(.9, color="0.5", linestyle="--", linewidth=.7)
ax.set(xticks=range(4), xticklabels=states, ylabel="Median agreement",
       ylim=(.68, 1.01), xlim=(-.2, 3.2))
ax.legend(frameon=False, fontsize=7, loc="lower left")
save(fig, "activation-effect")

rows = read_csv("reports/gptq_extension/results/stage2/transfer_feature_level.csv")
arms = [("legacy RTN", "Legacy RTN", "#747474"),
        ("matched RTN", "Matched RTN", "#b67230"),
        ("GPTQ", "GPTQ", "#28628e")]
fig, ax = plt.subplots(figsize=(3.3, 2.1), layout="constrained")
for arm_index, (key, label, color) in enumerate(arms):
    values, lowers, uppers = [], [], []
    for state in ["INT6", "INT4"]:
        row = next(r for r in rows if r["state"] == state
                   and r["metric"] == "normalized_error")
        values.append(float(row[key]))
        lowers.append(float(row[key + " low"]))
        uppers.append(float(row[key + " high"]))
    values = np.array(values)
    x = np.arange(2) + (arm_index - 1) * .23
    ax.bar(x, values, width=.21, label=label, color=color,
           yerr=[values-np.array(lowers), np.array(uppers)-values],
           capsize=2, error_kw={"elinewidth": .7})
ax.axhline(.056, color="0.45", linestyle="--", linewidth=.7)
ax.set(xticks=[0, 1], xticklabels=["INT6", "INT4"],
       ylabel="Normalized transfer error", ylim=(0, .76))
ax.legend(frameon=False, fontsize=7, loc="upper left")
save(fig, "quantizer-transfer")

# These sidecars document the exact sources used for the rendered estimates.
manifest = {
    "activation-effect": "reports/functional_validation/results/f5/f5_tables.csv",
    "quantizer-transfer": "reports/gptq_extension/results/stage2/transfer_feature_level.csv",
    "operation": "Plot stored estimates and confidence endpoints only; no recomputation of inference.",
}
(OUT / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Rendered two figures from committed result tables.")
