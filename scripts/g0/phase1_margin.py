"""N11 step 2 (decision D-025; pre-specified in handoff 007 section 4): does the margin of an activation above the SAE's learned JumpReLU threshold (Gemma Scope, Lieberum et al. 2024,
arXiv:2408.05147: a feature is active when its pre-activation exceeds a learned threshold) carry more than the raw maximum activation? margin ratio = max_activation / theta_f.
The margin is introduced by re-expressing `max_activation` (and `mean_activation`) in a DERIVED survival file, so that the existing, tested estimators run unchanged with 'magnitude' meaning
'margin': (a) N6 (controls matched on firing-rate stratum x bins of the margin, `phase1_magnitude.run_magnitude`), (b) N7 (the mixed model with log10 of the margin as covariate,
`phase1_mixed_variants.run_variants`: its `log_m_all` variant), (c) N9 (the margin scale as covariate of the language-level tier test, `phase1_language_level.run_language_level`).
Reading rules (fixed beforehand): (i) the A-035 residual is 'explained by margin' if the margin-matched INT6 contrast at S > 5 (the finest resolution keeping at least half of the selective features)
is not significant or within +-1 pp (the 'explained by magnitude' class of D-020), otherwise it 'remains'; (ii) margin 'carries more than raw magnitude' at the language level if with the margin
scale alone the INT5 tier slope has p >= 0.05 while it is < 0.05 with the max-activation scale of N9, otherwise 'no more than raw activation'."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

import extract_sae_thresholds as ex
import phase1_language_level as ll
import phase1_magnitude as mg
import phase1_mixed_variants as mv

ALPHA = 0.05
NOTE = ("> **N11 (D-025): in this report every activation is divided by its feature's learned JumpReLU threshold (the margin ratio; thresholds in `reports/phase1/sae_thresholds.csv`); "
        "read 'maximum activation' as 'margin ratio' and 'magnitude' as 'margin' throughout.**\n\n")


def margin_survival(survival, thresholds):
    """Copy of the survival dict with max_activation and mean_activation divided by the feature's threshold along the feature axis; everything else is shared unchanged."""
    th = torch.as_tensor(np.asarray(thresholds), dtype=torch.float32); F = survival["max_activation"].shape[1]
    if len(th) != F:
        raise ValueError(f"{len(th)} thresholds for {F} features")
    out = dict(survival); out["max_activation"] = survival["max_activation"] / th[None, :]; out["mean_activation"] = survival["mean_activation"] / th[None, :]; return out


def make_margin_run(run_dir, thresholds_csv, out_dir):
    """A run directory whose survival.pt carries the margin ratios; selectivity.pt and meta.json are byte copies; the originals are never touched."""
    run_dir, out_dir = Path(run_dir), Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    theta = ex.read_thresholds(thresholds_csv); torch.save(margin_survival(torch.load(run_dir / "survival.pt"), theta), out_dir / "survival.pt")
    for f in ("selectivity.pt", "meta.json"):
        shutil.copyfile(run_dir / f, out_dir / f)
    (out_dir / "PROVENANCE.txt").write_text(f"derived from {run_dir} (D-025): max_activation and mean_activation divided by the thresholds of {thresholds_csv} (sha256 {hashlib.sha256(Path(thresholds_csv).read_bytes()).hexdigest()})\n"); return out_dir


def _relabel(md_path):
    md_path = Path(md_path); md_path.write_text(NOTE + md_path.read_text())


def run_feature_level(run_dir, thresholds_csv, roster_path, out_dir, taus, bins=(3, 5, 10), n_perm=10000, n_boot=2000, seed=0, rate_only_json=None):
    out_dir = Path(out_dir); run = make_margin_run(run_dir, thresholds_csv, out_dir / "run")
    res = mg.run_magnitude(run, roster_path, out_dir, taus, bins=bins, n_perm=n_perm, n_boot=n_boot, seed=seed, rate_only_json=rate_only_json); _relabel(out_dir / "magnitude.md"); return res


def run_mixed_margin(run_dir, thresholds_csv, roster_path, out_dir, tau=5.0, seed=0):
    out_dir = Path(out_dir); run = make_margin_run(run_dir, thresholds_csv, out_dir / "run")
    res = mv.run_variants(run, roster_path, out_dir, tau=tau, seed=seed); _relabel(out_dir / "mixed_variants.md"); return res


def run_language_level_margin(run_dir, thresholds_csv, roster_path, out_dir, per_language_csv, meta_json, stats_json, n_perm=10000, seed=0):
    out_dir = Path(out_dir); run = make_margin_run(run_dir, thresholds_csv, out_dir / "run")
    res = ll.run_language_level(per_language_csv, meta_json, stats_json, roster_path, out_dir, n_perm=n_perm, seed=seed, survival_pt=run / "survival.pt"); _relabel(out_dir / "language_level.md"); return res


def reading_i(final_magnitude_reading):
    """Map the D-020 reading of the margin-matched INT6 contrast at S > 5 to the N11 rule (i)."""
    if final_magnitude_reading == "explained by magnitude":
        return "explained by margin"
    if final_magnitude_reading in ("reversed", "supported", "partly explained"):
        return "remains"
    raise ValueError(f"unknown D-020 reading {final_magnitude_reading!r}")


def reading_ii(p_margin_int5, p_raw_int5, alpha=ALPHA):
    return "margin carries more than raw magnitude" if (p_margin_int5 >= alpha and p_raw_int5 < alpha) else "no more than raw activation"


def reading_inputs(mag, lev, n9):
    """The numbers that feed the readings: the S > 5 headline of the margin-matched contrast, the margin scale's INT5 p, and N9's max-activation INT5 p."""
    fin = mag["readings"]["5.0"]["final"]
    if fin is None:
        raise ValueError("no margin-matched reading at S > 5")
    return fin, lev["activation_scale"]["widths"]["5"]["p_scale"], n9["activation_scale"]["widths"]["5"]["p_scale"]


def run_margin(run_dir, thresholds_csv, roster_path, out_dir, per_language_csv, meta_json, stats_json, n2_json, n9_json, taus=(3.0, 5.0), bins=(3, 5, 10), n_perm=10000, n_boot=2000, seed=0):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    mag = run_feature_level(run_dir, thresholds_csv, roster_path, out_dir / "magnitude", taus, bins, n_perm, n_boot, seed, rate_only_json=n2_json)
    mix = run_mixed_margin(run_dir, thresholds_csv, roster_path, out_dir / "mixed_variants", 5.0, seed)
    lev = run_language_level_margin(run_dir, thresholds_csv, roster_path, out_dir / "language_level", per_language_csv, meta_json, stats_json, n_perm, seed)
    fin, mar_p, raw_p = reading_inputs(mag, lev, json.loads(Path(n9_json).read_text()))
    res = {"decision": "D-025", "readings": {"i": reading_i(fin["reading"]), "ii": reading_ii(mar_p, raw_p)}, "feature_level": {"final": fin, "by_bins": mag["readings"]["5.0"]["by_bins"]},
           "mixed": {k: mix["variants"][k]["S_rank"] for k in ("full_all", "log_m_all")}, "mixed_log_margin": dict(zip(mix["variants"]["log_m_all"]["names"], zip(mix["variants"]["log_m_all"]["coef"], mix["variants"]["log_m_all"]["p"]))),
           "language_level": {"activation_scale": lev["activation_scale"], "raw_p_int5": raw_p, "margin_p_int5": mar_p}}
    (out_dir / "margin_summary.json").write_text(json.dumps(res, indent=1, default=float) + "\n"); (out_dir / "margin_summary.md").write_text(render_md(res)); return res


def render_md(res):
    a = res["language_level"]["activation_scale"]; L = ["# JumpReLU threshold margin as a matching variable and covariate (decision D-025)", "",
         "The thresholds theta_f are the learned JumpReLU thresholds of the SAE (Gemma Scope, Lieberum et al. 2024, arXiv:2408.05147: a feature is active when its pre-activation exceeds theta_f), read with Duan et al.'s own loader "
         "(`reports/phase1/sae_thresholds.csv`). Margin ratio = maximum activation / theta_f. The existing estimators of N6, N7 and N9 are run unchanged on a survival file in which the activations are divided by the thresholds.", "",
         "**Reading rule fixed beforehand.** (i) The positive residual of A-035 is *explained by margin* if the margin-matched INT6 contrast at S > 5 (finest resolution keeping at least half of the selective features) is not significant "
         "or within +-1 pp, otherwise it *remains*. (ii) Margin *carries more than raw magnitude* at the language level if with the margin scale alone the INT5 tier slope has p >= 0.05 while it is < 0.05 with the "
         "max-activation scale of N9, otherwise *no more than raw activation*.", "",
         f"## (a) Margin-matched INT6 contrast at S > 5", "", "| bins | matched contrast | Holm | kept | D-020 reading |", "|---|---|---|---|---|"]
    for n, v in res["feature_level"]["by_bins"].items():
        L.append(f"| {n} | {v['matched_delta'] * 100:+.2f} pp | {v['matched_holm']:.3g} | {v['retained'] * 100:.0f} % | {v['reading']} |")
    f = res["feature_level"]["final"]; L += ["", f"Headline ({f['bins']} bins, {f['retained'] * 100:.0f} % kept): **{f['reading']}**.", "", "## (b) Mixed model with log10 margin added",
        "", f"S~ coefficient in the full model {res['mixed']['full_all']['coef']:+.3f} (p {res['mixed']['full_all']['p']:.3g}); with log10 margin added {res['mixed']['log_m_all']['coef']:+.3f} (p {res['mixed']['log_m_all']['p']:.3g}).", "",
        "## (c) Language-level tier test with the margin scale", "", f"Spearman(tier, margin scale) {a['spearman_tier_scale']:+.3f}.", "", "| width | slope with margin scale (p) | with fertility and margin scale (p) | Spearman(margin scale, survival) |", "|---|---|---|---|"]
    for b, w in a["widths"].items():
        L.append(f"| INT{b} | {w['slope_scale'] * 100:+.2f} pp (p {w['p_scale']:.3g}) | {w['slope_both'] * 100:+.2f} pp (p {w['p_both']:.3g}) | {w['spearman_scale_survival']:+.3f} |")
    L += ["", f"INT5 p with the max-activation scale (N9): {res['language_level']['raw_p_int5']:.3g}; with the margin scale: {res['language_level']['margin_p_int5']:.3g}.", "",
          f"**Reading (i): {res['readings']['i']}. Reading (ii): {res['readings']['ii']}.**"]
    return "\n".join(L) + "\n"


def main(argv=None):
    import argparse
    repo = Path(__file__).resolve().parents[2]; P = repo / "reports" / "phase1"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory with survival.pt, selectivity.pt, meta.json (e.g. the cached p1-mask/eq)"); ap.add_argument("--stream-dir", required=True, help="reports/phase1/<regime>/eq (per_language.csv, thresholds/, language_level/)")
    ap.add_argument("--out", required=True); ap.add_argument("--thresholds", default=str(P / "sae_thresholds.csv")); ap.add_argument("--stats", default=str(P / "flores_text_stats.json"))
    ap.add_argument("--roster", default=str(repo / "reports/g0/phase1_languages.json")); ap.add_argument("--n-perm", type=int, default=10000); ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv); sd = Path(a.stream_dir)
    r = run_margin(a.run, a.thresholds, a.roster, a.out, sd / "per_language.csv", sd / "meta.json", a.stats, sd / "thresholds" / "thresholds.json", sd / "language_level" / "language_level.json", n_perm=a.n_perm, n_boot=a.n_boot, seed=a.seed)
    print("reading (i):", r["readings"]["i"], "| reading (ii):", r["readings"]["ii"]); print("wrote", Path(a.out) / "margin_summary.md")


if __name__ == "__main__":
    main()
