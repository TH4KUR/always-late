"""Compare a replicated Duan Phase-3 run with their published table, using the pre-committed bands
(reports/duan_codebase_audit.md section 7.2, Q1), and optionally upload the result to wandb AFTER the run, from their own
output files (so nothing is logged from inside their code).

  python scripts/g0/baseline_report.py --run-dir <dir with phase3_summary_final.csv> \\
      --published ../duan-sae-feature-survival/results/tables/phase3_summary_final.csv \\
      [--per-feature <run RTN_INT6_per_feature.csv> --published-per-feature <their phase5b per-feature csv>] [--wandb]
"""
import argparse, csv, json, sys
from pathlib import Path


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def compare_rows(rep_rows, pub_rows):
    rep = {r["condition"]: r for r in rep_rows}
    pub = {r["condition"]: r for r in pub_rows}
    rows = []
    for cond in pub:
        if cond not in rep:
            continue
        a, b = rep[cond], pub[cond]
        sa, sb = _f(a.get("survived_>0.9_pct")), _f(b.get("survived_>0.9_pct"))
        ma, mb = _f(a.get("median_corr")), _f(b.get("median_corr"))
        na, nb = _f(a.get("n_active_features")), _f(b.get("n_active_features"))
        rows.append({"condition": cond, "perplexity_rep": _f(a.get("perplexity")), "perplexity_pub": _f(b.get("perplexity")),
                     "ppl_delta_pct_rep": _f(a.get("ppl_delta_pct")), "ppl_delta_pct_pub": _f(b.get("ppl_delta_pct")),
                     "survived_rep": sa, "survived_pub": sb, "d_survived_pp": None if sa is None or sb is None else sa - sb,
                     "median_corr_rep": ma, "median_corr_pub": mb, "d_median_corr": None if ma is None or mb is None else ma - mb,
                     "n_active_rep": na, "n_active_pub": nb, "active_ratio": None if not na or not nb else na / nb})

    def within(cond, pp):
        for x in rows:
            if x["condition"] == cond and x["d_survived_pp"] is not None:
                return abs(x["d_survived_pp"]) <= pp
        return None

    base = next((x for x in rows if x["condition"] == "FP16 baseline"), None)
    active = None if base is None or base["active_ratio"] is None else abs(base["active_ratio"] - 1.0) <= 0.10
    d7 = next((x["ppl_delta_pct_rep"] for x in rows if x["condition"] == "RTN INT7"), None)
    d6 = next((x["ppl_delta_pct_rep"] for x in rows if x["condition"] == "RTN INT6"), None)
    signs = None if d7 is None or d6 is None else (d7 < 0 < d6)
    bands = {"int7_survival_within_5pp": within("RTN INT7", 5.0), "int6_survival_within_10pp": within("RTN INT6", 10.0),
             "active_features_within_10pct": active, "ppl_signs_int7_improves_int6_worsens": signs}
    vals = list(bands.values())
    q1 = False if any(v is False for v in vals) else (None if any(v is None for v in vals) else True)
    return {"rows": rows, "bands": bands, "q1_pass": q1}


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0
        i = j + 1
    return r


def _spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = _rank(a), _rank(b); n = len(a); ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return None if den == 0 else num / den


def _col(row, *names):
    for n in names:
        if n in row:
            return row[n]
    raise KeyError(names)


def per_feature_agreement(rep_csv, pub_csv):
    def load(p):
        out = {}
        for r in _read(p):
            if str(_col(r, "active", "active_rtn")) == "True":
                out[int(_col(r, "feature_id"))] = (float(_col(r, "corr", "corr_rtn")), float(_col(r, "firing_rate", "firing_rate_rtn")))
        return out
    a, b = load(rep_csv), load(pub_csv)
    ids = sorted(set(a) & set(b))
    return {"n_common_active": len(ids), "n_active_rep": len(a), "n_active_pub": len(b),
            "spearman_corr": _spearman([a[i][0] for i in ids], [b[i][0] for i in ids]),
            "spearman_firing_rate": _spearman([a[i][1] for i in ids], [b[i][1] for i in ids])}


def to_markdown(cmp_, pf=None):
    L = ["| condition | ppl (rep / pub) | ppl delta % (rep / pub) | survived % (rep / pub) | d pp | median r (rep / pub) | active (rep / pub) |", "|---|---|---|---|---|---|---|"]
    fmt = lambda v, p=2: "-" if v is None else f"{v:.{p}f}"
    for x in cmp_["rows"]:
        L.append(f"| {x['condition']} | {fmt(x['perplexity_rep'], 1)} / {fmt(x['perplexity_pub'], 1)} | {fmt(x['ppl_delta_pct_rep'])} / {fmt(x['ppl_delta_pct_pub'])} | "
                 f"{fmt(x['survived_rep'])} / {fmt(x['survived_pub'])} | {fmt(x['d_survived_pp'])} | {fmt(x['median_corr_rep'], 4)} / {fmt(x['median_corr_pub'], 4)} | "
                 f"{fmt(x['n_active_rep'], 0)} / {fmt(x['n_active_pub'], 0)} |")
    L += ["", "Bands (pre-committed): " + ", ".join(f"{k}={'n/a' if v is None else v}" for k, v in cmp_["bands"].items()),
          f"**Q1 (published numbers reproduced): {'NOT EVALUABLE (missing conditions)' if cmp_['q1_pass'] is None else cmp_['q1_pass']}**"]
    if pf:
        L.append(f"Per-feature agreement at INT6 over {pf['n_common_active']} common active features: Spearman(r)={pf['spearman_corr']}, Spearman(firing rate)={pf['spearman_firing_rate']}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True); ap.add_argument("--published", required=True)
    ap.add_argument("--per-feature"); ap.add_argument("--published-per-feature")
    ap.add_argument("--out"); ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    run = Path(a.run_dir)
    rep_rows = _read(run / "phase3_summary_final.csv")
    cmp_ = compare_rows(rep_rows, _read(a.published))
    pf = per_feature_agreement(a.per_feature, a.published_per_feature) if a.per_feature and a.published_per_feature else None
    md = to_markdown(cmp_, pf)
    out = Path(a.out) if a.out else run / "baseline_report.md"
    out.write_text(md); (out.with_suffix(".json")).write_text(json.dumps({"comparison": cmp_, "per_feature_int6": pf}, indent=2))
    print(md)
    if a.wandb:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from g0_wandb import G0Logger
        import os
        lg = G0Logger.start(project=os.environ.get("WANDB_PROJECT", "anlp-project"), entity=os.environ.get("WANDB_ENTITY") or None,
                            name=os.environ.get("G0_WANDB_NAME", f"baseline-{run.name}"), config={"run_dir": str(run), "kind": "baseline-replication"},
                            out_dir=run, tags=["g0", "baseline", "duan03-faithful"])
        for r_ in rep_rows:
            lg.log_condition({k: (_f(v) if _f(v) is not None else v) for k, v in r_.items()})
        lg.log_table(cmp_["rows"], key="comparison_vs_published")
        if lg.active:
            lg.run.summary["q1_pass"] = cmp_["q1_pass"]
            for k, v in cmp_["bands"].items():
                lg.run.summary[k] = v
            if pf:
                lg.run.summary.update({f"per_feature_int6/{k}": v for k, v in pf.items()})
        lg.finish(run)


if __name__ == "__main__":
    main()
