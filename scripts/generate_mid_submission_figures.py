"""Generate All Publication-Ready Figures for Mid-Submission 1.

Implements STEP 8 of the Mid-Submission priority plan:
- Produces the 5 core figures required for the 7-8 page Mid-Submission report:
    Figure 1: Experimental Architecture & Pipeline Flowchart
    Figure 2: Selectivity Distribution across the 27-Language Panel
    Figure 3: Selective vs Universal Survival under INT7 and INT6 (Quintile-Matched)
    Figure 4: Resource-Tier Survival Comparison (High vs Mid vs Low)
    Figure 5: Selective Feature Damage vs Language-Level Cross-Entropy Loss Degradation (Delta L)
- All figures saved at 300 DPI in `figures/`.
- Resilient: Uses pure matplotlib with optional seaborn styling if installed.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", font="sans-serif")
except ImportError:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.autolayout": True,
})


def generate_figure1_pipeline():
    """Figure 1: Conceptual & Experimental Pipeline Diagram."""
    fig, ax = plt.subplots(figsize=(11, 3.2), dpi=300)
    ax.axis("off")

    stages = [
        ("FLORES-200\n(27 Languages)", "#e8f4f8", "#2b5c8f"),
        ("Token-Budget\nEqualizer", "#e8f4f8", "#2b5c8f"),
        ("Gemma-2-2B\nLayer 12 Resid", "#fff2e6", "#d95f02"),
        ("Gemma Scope\nJumpReLU SAE", "#fff2e6", "#d95f02"),
        ("Selectivity\nTagging S(f)", "#eef7ee", "#2ca02c"),
        ("RTN Fake\nQuantization", "#fce8e6", "#d62728"),
        ("Quintile-Matched\nPearson Survival", "#f3e8fd", "#7570b3"),
        ("Behavioral\nLoss Link (ΔL)", "#fdf6e2", "#b58900"),
    ]

    box_w, box_h = 1.05, 0.65
    y = 0.5

    for i, (name, bg, border) in enumerate(stages):
        x = i * 1.35 + 0.2
        rect = patches.FancyBboxPatch(
            (x, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.08,rounding_size=0.1",
            linewidth=1.8,
            edgecolor=border,
            facecolor=bg,
        )
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y, name, ha="center", va="center", fontsize=9.5, fontweight="semibold")

        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x + box_w + 0.28, y),
                xytext=(x + box_w + 0.02, y),
                arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.5, mutation_scale=12),
            )

    ax.set_xlim(0, len(stages) * 1.35 + 0.2)
    ax.set_ylim(0, 1)
    ax.set_title("Figure 1: End-to-End Experimental Architecture and Mechanistic Pipeline", fontsize=12, pad=15)
    out_path = FIGURES_DIR / "fig1_experimental_pipeline.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")


SEL_REPORT = REPO_ROOT / "reports" / "selectivity_results_27lang.json"
SURV_REPORT = REPO_ROOT / "reports" / "phase1_survival_results_27lang_gpu.json"
BEHAVIOR_REPORT = REPO_ROOT / "reports" / "behavioral_loss_link_results.json"
TIER_COLORS = {"high": "#2b5c8f", "mid": "#d95f02", "low": "#7570b3"}


def _load(path: Path) -> dict:
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def generate_figure2_selectivity():
    """Figure 2: real S(f) distribution (supported features) and selective counts by tier."""
    sel = _load(SEL_REPORT)
    feats = np.load(SEL_REPORT.with_suffix(".features.npz"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    scores = feats["raw_s"][feats["supported"]]
    cutoff = sel["shuffled_null_threshold_95th"]
    ax1.hist(scores, bins=60, color="#2b5c8f", alpha=0.75, density=True, edgecolor="none")
    ax1.axvline(cutoff, color="red", linestyle="--", linewidth=1.8, label=f"95th Pct Null Cutoff ({cutoff:.2f})")
    ax1.set_title(f"(A) Selectivity Score S(f), {len(scores):,} supported features")
    ax1.set_xlabel("Deng Selectivity S(f)")
    ax1.set_ylabel("Density")
    ax1.legend()

    tiers = ["high", "mid", "low"]
    counts = [sel["selective_features_by_tier"][t] for t in tiers]
    bars = ax2.bar([t.capitalize() for t in tiers], counts, color=[TIER_COLORS[t] for t in tiers], width=0.45)
    ax2.set_title("(B) Language-Selective Features by Resource Tier")
    ax2.set_xlabel("Resource Tier (FLORES-200)")
    ax2.set_ylabel("Total Language-Selective Features")
    for bar, v in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, v * 1.01, f"{v}", ha="center", va="bottom", fontweight="semibold")
    ax2.set_ylim(0, max(counts) * 1.15)

    out_path = FIGURES_DIR / "fig2_selectivity_populations.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated: {out_path}")


def _annotate(ax, rects, fontsize=10):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=fontsize)


def generate_figure3_survival_comparison():
    """Figure 3: Selective vs quintile-matched universal survival at INT7 and INT6."""
    res = _load(SURV_REPORT)["results_by_bitwidth"]
    labels = [bw for bw in ["INT7", "INT6"] if bw in res]
    selective = [res[bw]["selective_features"] for bw in labels]
    universal = [res[bw]["quintile_matched_universal"] for bw in labels]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    x = np.arange(len(labels))
    width = 0.32
    r1 = ax.bar(x - width/2, [d["survived_pct"] for d in selective], width,
                label=f"Language-Selective (n={selective[0]['count']})", color="#d95f02", alpha=0.9)
    r2 = ax.bar(x + width/2, [d["survived_pct"] for d in universal], width,
                label=f"Universal, Quintile-Matched (n={universal[0]['count']})", color="#2b5c8f", alpha=0.9)
    ax.set_ylabel("Feature Survival Rate (% with r ≥ 0.90)")
    ax.set_title("Feature Survival under RTN Quantization (RQ1)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="semibold")
    lo = min(d["survived_pct"] for d in selective + universal)
    ax.set_ylim(max(0, lo - 15), 103)
    ax.legend(loc="lower left")
    _annotate(ax, list(r1) + list(r2))

    out_path = FIGURES_DIR / "fig3_rq1_selective_vs_universal_survival.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated: {out_path}")


def generate_figure4_tier_survival():
    """Figure 4: Selective-feature survival by resource tier at INT7 and INT6."""
    res = _load(SURV_REPORT)["results_by_bitwidth"]
    tiers = ["high", "mid", "low"]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    x = np.arange(len(tiers))
    width = 0.35
    rects, vals = [], []
    for i, (bw, color) in enumerate([("INT7", "#386cb0"), ("INT6", "#f0027f")]):
        v = [res[bw]["by_resource_tier"][t]["survived_pct"] for t in tiers]
        vals += v
        rects += list(ax.bar(x + (i - 0.5) * width, v, width, label=bw, color=color, alpha=0.85))
    ax.set_ylabel("Survival Rate (% with r ≥ 0.90)")
    ax.set_title("Language-Selective Feature Survival by Resource Tier (RQ2)")
    ax.set_xticks(x)
    ns = [res["INT7"]["by_resource_tier"][t]["count"] for t in tiers]
    ax.set_xticklabels([f"{t.capitalize()} (n={n})" for t, n in zip(tiers, ns)], fontweight="semibold")
    ax.set_xlabel("Resource Tier")
    ax.set_ylim(max(0, min(vals) - 15), 105)
    ax.legend(title="Bit-width")
    _annotate(ax, rects, fontsize=9.5)

    out_path = FIGURES_DIR / "fig4_rq2_tier_survival.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated: {out_path}")


def generate_figure5_behavioral_link():
    """Figure 5: per-language selective-feature damage vs. delta loss (measured, INT6)."""
    if not BEHAVIOR_REPORT.exists():
        print(f"Skipping Figure 5: {BEHAVIOR_REPORT} not found (run scripts/run_behavioral_loss_link.py).")
        return
    beh = _load(BEHAVIOR_REPORT)
    bw = "INT6"
    x_key = f"{bw}_selective_median_r"
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    for t in ["high", "mid", "low"]:
        pts = [d for d in beh["data"] if d["resource_tier"] == t and d.get(x_key) is not None]
        ax.scatter([1 - d[x_key] for d in pts], [d[f"{bw}_delta_loss"] for d in pts],
                   label=f"{t.capitalize()}-Resource", color=TIER_COLORS[t], s=55, alpha=0.85)
    c = beh["correlations"][bw]["spearman_selective_drift_vs_delta_loss"]
    ax.set_title(f"Selective-Feature Drift vs ΔLoss ({bw}; Spearman ρ = {c['rho']:+.2f}, p = {c['p_val']:.3f})")
    ax.set_xlabel("1 − median Pearson r of language's selective features")
    ax.set_ylabel("Cross-Entropy Degradation ΔL (nats/token)")
    ax.legend()

    out_path = FIGURES_DIR / "fig5_behavioral_loss_link.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated: {out_path}")


def generate_all_figures():
    print("=== Generating Publication-Ready Mid-Submission Figures ===")
    generate_figure1_pipeline()
    generate_figure2_selectivity()
    generate_figure3_survival_comparison()
    generate_figure4_tier_survival()
    generate_figure5_behavioral_link()
    print("All figures successfully saved in `figures/`.")


if __name__ == "__main__":
    generate_all_figures()
