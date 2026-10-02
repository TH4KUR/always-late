"""Log the Phase-1 run (selectivity, survival, stats, behavioral link, figures) to Weights & Biases.

Reads the finished report files, so what is logged is exactly what the paper cites.
Usage: python scripts/log_to_wandb.py --tokens 20000 [--project always-late] [--log-file path ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import wandb

from scripts.statistical_analysis import holm_bonferroni_correction, two_proportion_p_value

REPORTS = REPO_ROOT / "reports"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="always-late")
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--log-file", nargs="*", default=[])
    args = ap.parse_args()

    sel = json.loads((REPORTS / "selectivity_results_27lang.json").read_text())
    surv = json.loads((REPORTS / "phase1_survival_results_27lang_gpu.json").read_text())
    beh_path = REPORTS / "behavioral_loss_link_results.json"
    beh = json.loads(beh_path.read_text()) if beh_path.exists() else None

    run = wandb.init(
        project=args.project,
        name=f"phase1-27lang-{args.tokens}tok",
        job_type="phase1",
        config={
            "model": "unsloth/gemma-2-2b",
            "sae": "gemma-scope-2b-pt-res layer_12/width_16k average_l0_82",
            "tokens_per_language": args.tokens,
            "n_languages": sel["n_languages"],
            "bitwidths": list(surv["results_by_bitwidth"]),
            "null_permutations": 20,
            "seed": 42,
            "selectivity_split": "flores dev",
            "survival_split": "flores devtest",
        },
    )

    wandb.summary.update({
        "selectivity/total_selective": sel["total_selective_features"],
        "selectivity/supported_features": sel["supported_features_count"],
        "selectivity/null_threshold": sel["shuffled_null_threshold_95th"],
        "selectivity/encode_seconds": sel["total_encode_seconds"],
        **{f"selectivity/tier_{t}": n for t, n in sel["selective_features_by_tier"].items()},
    })
    wandb.log({"selectivity/by_language": wandb.Table(
        columns=["language", "name", "tier", "script", "tokens", "n_selective"],
        data=[[c, v["name"], v["tier"], v["script"], v["tokens"], v["n_selective_features"]]
              for c, v in sel["languages"].items()])})

    pvals, names = [], []
    for bw, res in surv["results_by_bitwidth"].items():
        s, u = res["selective_features"], res["quintile_matched_universal"]
        wandb.summary.update({
            f"survival/{bw}/selective_pct": s["survived_pct"], f"survival/{bw}/selective_n": s["count"],
            f"survival/{bw}/universal_pct": u["survived_pct"], f"survival/{bw}/universal_n": u["count"],
            f"survival/{bw}/selective_median_r": s["median_pearson_r"],
            f"survival/{bw}/universal_median_r": u["median_pearson_r"],
            f"survival/{bw}/elapsed_s": res["elapsed_seconds"],
        })
        for t, d in res["by_resource_tier"].items():
            wandb.summary.update({f"survival/{bw}/tier_{t}_pct": d["survived_pct"],
                                  f"survival/{bw}/tier_{t}_median_r": d["median_pearson_r"]})
        c1, c2 = round(s["survived_pct"] * s["count"] / 100), round(u["survived_pct"] * u["count"] / 100)
        names.append(f"H1/H2 {bw} selective vs universal")
        pvals.append(two_proportion_p_value(c1, s["count"], c2, u["count"]))
        t = res["by_resource_tier"]
        if "high" in t and "low" in t:
            names.append(f"H3 {bw} high vs low")
            pvals.append(two_proportion_p_value(
                round(t["low"]["survived_pct"] * t["low"]["count"] / 100), t["low"]["count"],
                round(t["high"]["survived_pct"] * t["high"]["count"] / 100), t["high"]["count"]))
    adj = holm_bonferroni_correction(pvals)
    wandb.log({"stats/holm": wandb.Table(
        columns=["test", "raw_p", "adjusted_alpha", "significant"],
        data=[[n, a["raw_p"], a["adjusted_alpha"], bool(a["significant"])] for n, a in zip(names, adj)])})

    if beh:
        for bw, corr in beh["correlations"].items():
            for k, v in corr.items():
                wandb.summary[f"behavior/{bw}/{k}_rho"] = v["rho"]
                wandb.summary[f"behavior/{bw}/{k}_p"] = v["p_val"]

    for fig in sorted((REPO_ROOT / "figures").glob("fig[2-5]_*.png")):
        wandb.log({f"figures/{fig.stem}": wandb.Image(str(fig))})

    art = wandb.Artifact("phase1-reports", type="results")
    for p in [REPORTS / "selectivity_results_27lang.json", REPORTS / "phase1_survival_results_27lang_gpu.json",
              REPORTS / "selectivity_results_27lang.features.npz", beh_path,
              REPORTS / "fertility_vs_english_flores_dev.json", *map(Path, args.log_file)]:
        if p.exists():
            art.add_file(str(p))
    run.log_artifact(art)
    print("wandb run:", run.url)
    run.finish()


if __name__ == "__main__":
    main()
