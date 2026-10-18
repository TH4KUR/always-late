"""N2: sensitivity of the binary 'selective' split to FIXED S thresholds (proposal section 4.1 asks to report threshold sensitivity; decision D-019).
A labelled deviation: the proposal sets the threshold at percentiles of the shuffled-label null of S, but the S2 runs saved only the 95th-percentile cutoff, not the
null samples, so other percentiles cannot be recomputed from the files; fixed S thresholds are the closest quantity that can (a tagging-only rerun that saves the null
is the faithful alternative, DECISIONS_FOR_REVIEW.md Q6).
Nothing new is estimated: the selection rule is phase1_run.selectivity_report's (`supported & (stable >= need) & (S > cutoff)`) with the cutoff replaced by tau, the
controls are drawn by the same matched_sample (same strata, same seed), and the analysis is the unchanged phase1_analysis.run_analysis (D-016, D-017)."""
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch

import phase1_analysis as an
import phase1_lib as p1


def alternative_selection(selp, tau, seed=0):
    """Copy of the saved selectivity dict with `selective`, `controls` and `control_strata_short` recomputed for the fixed threshold tau (strict S > tau, as in the run)."""
    need = int(selp["min_stable_needed"]); supported = selp["supported"].bool()
    selective = supported & (selp["stable_count"] >= need) & (selp["S"] > float(tau))
    controls, short = p1.matched_sample(selp["strata"], selective, supported & ~selective, seed)
    out = dict(selp); out.update(selective=selective, controls=controls, control_strata_short=short, tau=float(tau)); return out


def _row(label, tau, alt, run_dir, roster_path, out_dir, n_perm, n_boot, seed, **kw):
    row = {"label": label, "tau": float(tau), "n_selective": int(alt["selective"].sum()), "n_controls": int(len(alt["controls"])), "control_strata_short": list(alt["control_strata_short"])}
    if row["n_selective"] == 0 or row["n_controls"] == 0:
        row["undefined"] = f"{row['n_selective']} selective and {row['n_controls']} control features at this threshold: the contrasts have no groups"; return row
    derived = Path(tempfile.mkdtemp(prefix="_derived_", dir=out_dir))
    try:
        for f in ("survival.pt", "meta.json"):
            os.symlink((Path(run_dir) / f).resolve(), derived / f)
        torch.save(alt, derived / "selectivity.pt")
        an.run_analysis(derived, roster_path, Path(out_dir) / f"tau_{label}", n_perm=n_perm, n_boot=n_boot, seed=seed, **kw)
    finally:
        shutil.rmtree(derived, ignore_errors=True)
    full = json.loads((Path(out_dir) / f"tau_{label}" / "analysis.json").read_text())
    row.update(tests=full["tests"], holm=full["holm"]); return row


def run_thresholds(run_dir, roster_path, out_dir, taus, n_perm=10000, n_boot=2000, seed=0, **kw):
    """One row for the run's own null cutoff (it must reproduce the original analysis) and one per fixed tau; undefined rows (no selective or no control feature) are
    recorded, not raised. Writes thresholds.json / thresholds.md and, per tau, the unchanged analysis report in tau_<label>/."""
    taus = [float(t) for t in taus]
    if len(set(taus)) != len(taus) or any(t <= 0 for t in taus):
        raise ValueError("thresholds must be positive and unique")
    run_dir, out_dir = Path(run_dir), Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    selp = torch.load(run_dir / "selectivity.pt"); meta = json.loads((run_dir / "meta.json").read_text())
    plan = [("null", float(selp["null_cutoff"]))] + [(str(t), t) for t in taus]
    rows = [_row(label, tau, alternative_selection(selp, tau, seed), run_dir, roster_path, out_dir, n_perm, n_boot, seed, **kw) for label, tau in plan]
    res = {"decision": "D-019", "run": str(run_dir), "regime": meta.get("regime"), "stream": meta.get("stream"), "seed": seed, "n_perm": n_perm, "n_boot": n_boot,
           "null_cutoff": float(selp["null_cutoff"]), "min_stable_needed": int(selp["min_stable_needed"]), "rows": rows}
    (out_dir / "thresholds.json").write_text(json.dumps(res, indent=1) + "\n"); (out_dir / "thresholds.md").write_text(render_md(res)); return res


def render_md(res):
    L = [f"# Fixed-threshold sensitivity of the selective split, regime `{res['regime']}`, stream `{res['stream']}` (decision D-019)", "",
         "**Labelled deviation from the proposal (section 4.1):** the proposal sets the binary threshold at percentiles of the shuffled-label null of S; the runs saved only the 95th-percentile "
         f"cutoff ({res['null_cutoff']:.4f}), not the null samples, so fixed S thresholds are reported instead (DECISIONS_FOR_REVIEW.md Q6). The rule is the run's own: selective = supported, peak "
         f"language stable in at least {res['min_stable_needed']} of the bootstrap draws, and S > tau (strict); controls = firing-rate-stratum-matched universal features drawn by the same "
         f"`matched_sample`; the estimators are the unchanged D-014 / D-016 analysis ({res['n_perm']} permutations, {res['n_boot']} bootstrap draws, seed {res['seed']}). "
         "The `null` row is the run's own cutoff and must equal the original analysis.", "",
         "| tau | selective | controls | strata short | T1 INT7 selective - control (95% CI), p, Holm | T2 INT6 selective - control (95% CI), p, Holm | T3 tier slope, p, Holm |", "|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        head = f"| {r['label']} ({r['tau']:.4g}) | {r['n_selective']} | {r['n_controls']} | {len(r['control_strata_short'])} | "
        if "undefined" in r:
            L.append(head + f"undefined: {r['undefined']} | | |"); continue
        cell = lambda k, hk: (f"{r['tests'][k]['delta'] * 100:+.2f} pp [{r['tests'][k]['ci'][0] * 100:+.2f}, {r['tests'][k]['ci'][1] * 100:+.2f}], p {r['tests'][k]['p']:.3g}, Holm {r['holm'][hk]:.3g}")
        t3 = r["tests"]["T3_tier"]; slope = "n/a" if t3.get("slope") is None else f"{t3['slope']:+.4f}"
        L.append(head + cell("T1_int7", "T1_int7") + " | " + cell("T2_int6", "T2_int6") + f" | {slope}, p {t3['p']:.3g}, Holm {r['holm']['T3_tier']:.3g} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=run_thresholds.__doc__)
    ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--roster", default=str(Path(__file__).resolve().parents[2] / "reports/g0/phase1_languages.json"))
    ap.add_argument("--taus", type=float, nargs="*", default=[2.0, 3.0, 5.0]); ap.add_argument("--n-perm", type=int, default=10000); ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv); res = run_thresholds(a.run, a.roster, a.out, a.taus, n_perm=a.n_perm, n_boot=a.n_boot, seed=a.seed)
    for r in res["rows"]:
        print(f"tau {r['label']}: {r['n_selective']} selective, {r['n_controls']} controls" + (" -> undefined" if "undefined" in r else
              "; " + "; ".join(f"{k} delta {r['tests'][k]['delta'] * 100:+.2f} pp p {r['tests'][k]['p']:.3g} Holm {r['holm'][k]:.3g}" for k in ("T1_int7", "T2_int6")) + f"; T3 p {r['tests']['T3_tier']['p']:.3g}"))
    print("wrote", Path(a.out) / "thresholds.md")


if __name__ == "__main__":
    main()
