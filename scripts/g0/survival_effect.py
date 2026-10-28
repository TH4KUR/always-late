"""Stage 3 of the survival-and-intervention plan (freeze: reports/survival_effect_rescue/stage3_analysis_freeze_2026-10-28.md; post hoc, exploratory): does the continuous activation correlation R1 of a feature predict
its effect similarity W1 beyond simple baselines? Per arm (legacy RTN, matched RTN, GPTQ) and bit width (INT6, INT4) on the 1,487 features of the F5 sample, held out by language.

The functions are pure numpy (tested in tests/test_survival_effect.py); `load_*` and `main` read the saved F5 files (torch is imported only there).

  python3 scripts/g0/survival_effect.py --control-only      the reproduction control: the committed Spearman correlations of R1 with W1 are reproduced from the saved arrays (exit 1 on a mismatch)
  python3 scripts/g0/survival_effect.py --out DIR            the full analysis (the control runs first and must pass)"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_func_f5_metrics as met  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
N_BOOT, SEED, CONTROL_SEED = 2000, 20261028, 0
TARGET = 0.10
TAU = 0.94888916015625                      # the committed numerical floor of F5 (INT16 versus FP, 5th percentile)
ARMS = ("legacy RTN", "matched RTN", "GPTQ")
STATES = ("INT6", "INT4")
BASE_NAMES = ("n_active", "firing_rate_native", "mean_activation_native", "abs_dnll_fp", "n_pos_fp")


# ----------------------------------------------------------------------------- pure functions
def quantile_transform(train, x):
    """Empirical CDF of the training values at x, ties at their mid-rank: (#train < x + 0.5 #train == x) / n."""
    train = np.sort(np.asarray(train, dtype=np.float64)); x = np.asarray(x, dtype=np.float64)
    return (np.searchsorted(train, x, side="left") + np.searchsorted(train, x, side="right")) / (2.0 * len(train))


def prepare(train, test):
    """Column-wise: NaN -> median of the defined training values, then the quantile transform of the training fold applied to both."""
    train = np.array(train, dtype=np.float64, copy=True); test = np.array(test, dtype=np.float64, copy=True)
    for j in range(train.shape[1]):
        med = np.nanmedian(train[:, j]) if np.any(~np.isnan(train[:, j])) else 0.0
        train[np.isnan(train[:, j]), j] = med; test[np.isnan(test[:, j]), j] = med
        train[:, j], test[:, j] = quantile_transform(train[:, j], train[:, j]), quantile_transform(train[:, j], test[:, j])
    return train, test


def partial_spearman(x, y, Z):
    """Spearman correlation of x and y after removing the linear effect (with intercept) of the RANKS of the columns of Z from the ranks of x and y; NaN without variation."""
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64); Z = np.asarray(Z, dtype=np.float64).reshape(len(x), -1)
    d = ~(np.isnan(x) | np.isnan(y) | np.isnan(Z).any(axis=1))
    if d.sum() < 4:
        return float("nan")
    rx, ry = met._ranks(x[d]), met._ranks(y[d])
    R = np.column_stack([np.ones(int(d.sum()))] + [met._ranks(Z[d, j]) for j in range(Z.shape[1])])
    ex = rx - R @ np.linalg.lstsq(R, rx, rcond=None)[0]; ey = ry - R @ np.linalg.lstsq(R, ry, rcond=None)[0]
    if np.allclose(ex, 0.0, atol=1e-9) or np.allclose(ey, 0.0, atol=1e-9) or ex.std() == 0 or ey.std() == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def ridge_fit(X, y, lam):
    xm = X.mean(axis=0); ym = float(np.mean(y)); Xc = X - xm
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ (np.asarray(y, dtype=np.float64) - ym))
    return w, xm, ym


def ridge_predict(model, X):
    w, xm, ym = model
    return (X - xm) @ w + ym


def fit_predict(Xtr, ytr, Xte, lam):
    a, b = prepare(Xtr, Xte)
    return ridge_predict(ridge_fit(a, ytr, lam), b)


def grouped_folds(groups, k):
    """Fold of every row: the j-th group (sorted) goes to fold j mod k, so a group is never split."""
    groups = np.asarray(groups); u = np.unique(groups); pos = {g: i % k for i, g in enumerate(u.tolist())}
    return np.array([pos[g] for g in groups.tolist()])


def choose_lambda(X, y, groups, grid=GRID, k=5):
    """Grouped k-fold cross-validation inside the training set; minimum pooled MAE; a tie goes to the larger penalty."""
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.float64); f = grouped_folds(groups, k); best, best_mae = grid[0], None
    for lam in sorted(grid):
        err = np.empty(len(y))
        for j in range(k):
            te = f == j
            if te.any() and (~te).any():
                err[te] = np.abs(y[te] - fit_predict(X[~te], y[~te], X[te], lam))
        mae = float(np.mean(err))
        if best_mae is None or mae <= best_mae:
            best, best_mae = lam, mae
    return best


def loo_language(X, y, lang, grid=GRID, k=5):
    """Leave-one-language-out predictions (every language is predicted by a model fitted on the others, lambda chosen inside the training languages); also the chosen lambdas."""
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.float64); lang = np.asarray(lang); pred = np.empty(len(y)); chosen = {}
    for g in np.unique(lang).tolist():
        te = lang == g; tr = ~te
        lam = choose_lambda(X[tr], y[tr], lang[tr], grid, k); chosen[g] = lam
        pred[te] = fit_predict(X[tr], y[tr], X[te], lam)
    return pred, chosen


def loo_null(y, lang):
    """The training-fold median of the other languages."""
    y = np.asarray(y, dtype=np.float64); lang = np.asarray(lang); out = np.empty(len(y))
    for g in np.unique(lang).tolist():
        out[lang == g] = np.median(y[lang != g])
    return out


def improvement(err_a, err_b, lang, n_boot=N_BOOT, seed=SEED):
    """1 - sum(err_b) / sum(err_a) (relative reduction of the MAE of b against a on the same rows) with a language-cluster percentile bootstrap (whole languages resampled)."""
    a = np.asarray(err_a, dtype=np.float64); b = np.asarray(err_b, dtype=np.float64)
    fn = lambda idx: 1.0 - float(b[idx].sum()) / float(a[idx].sum()) if a[idx].sum() > 0 else float("nan")
    obs, lo, hi = met.cluster_bootstrap(np.asarray(lang), fn, n_boot=n_boot, seed=seed)
    return {"improvement": float(obs), "lo": float(lo), "hi": float(hi)}


def cell_met(imp, lo, target=TARGET):
    """The decision rule of one cell: improvement of B over A at least 10 % and the lower end of its interval above 0."""
    return bool(imp == imp and lo == lo and imp >= target and lo > 0)


def overall_statement(cells):
    n = sum(bool(c) for c in cells)
    if n == len(cells):
        return "adds predictive value"
    return "no clear predictive value beyond baselines" if n == 0 else "mixed"


def reliable_mask(cos16, tau=TAU):
    return np.asarray(cos16, dtype=np.float64) >= tau


def loss_mask(dnll_fp, eps):
    return np.abs(np.asarray(dnll_fp, dtype=np.float64)) >= eps


def reference_noise(d_fp, d16, q=95):
    return float(np.percentile(np.abs(np.asarray(d16, dtype=np.float64) - np.asarray(d_fp, dtype=np.float64)), q))


def design(base, r1, jaccard, scale_dev):
    base = np.asarray(base, dtype=np.float64)
    return {"A": base, "B": np.column_stack([base, r1]), "C": np.column_stack([base, r1, jaccard, scale_dev]), "R1": np.asarray(r1, dtype=np.float64)[:, None]}


def analyse_cell(Xs, w1, lang, mask, tau=TAU, n_boot=N_BOOT, seed=SEED):
    """All statistics of one (arm, state, population): association, prediction, improvements, the dichotomy; `mask` selects the population (leave-language-out inside it)."""
    m = np.asarray(mask, dtype=bool); y = np.asarray(w1, dtype=np.float64)[m]; g = np.asarray(lang)[m]; X = {k: v[m] for k, v in Xs.items()}
    r1 = X["R1"][:, 0]; out = {"n": int(m.sum()), "n_languages": int(len(np.unique(g)))}
    boot = lambda fn: met.cluster_bootstrap(g, fn, n_boot=n_boot, seed=seed)
    rho, lo, hi = boot(lambda idx: met.spearman(r1[idx], y[idx])["rho"]); out["spearman"] = {"rho": float(rho), "lo": float(lo), "hi": float(hi)}
    A = X["A"]; prho, plo, phi = boot(lambda idx: partial_spearman(r1[idx], y[idx], A[idx])); out["partial_spearman_given_A"] = {"rho": float(prho), "lo": float(plo), "hi": float(phi)}
    errs, chosen = {}, {}
    for name in ("A", "B", "C", "R1"):
        pred, ch = loo_language(X[name], y, g); errs[name] = np.abs(y - pred); chosen[name] = ch
    errs["null"] = np.abs(y - loo_null(y, g))
    out["mae"] = {k: float(v.mean()) for k, v in errs.items()}
    out["improvement"] = {"B_over_A": improvement(errs["A"], errs["B"], g, n_boot, seed), "C_over_B": improvement(errs["B"], errs["C"], g, n_boot, seed),
                          "A_over_null": improvement(errs["null"], errs["A"], g, n_boot, seed), "R1_over_null": improvement(errs["null"], errs["R1"], g, n_boot, seed)}
    out["median_chosen_lambda"] = {k: float(np.median(list(v.values()))) for k, v in chosen.items()}
    ev = y >= tau; hi_r = r1 > 0.9
    out["dichotomy"] = {"survives_event": int((hi_r & ev).sum()), "survives_no_event": int((hi_r & ~ev).sum()), "fails_event": int((~hi_r & ev).sum()), "fails_no_event": int((~hi_r & ~ev).sum())}
    if ev.any() and (~ev).any():
        a = met.auroc(r1, ev); alo, ahi = boot(lambda idx: met.auroc(r1[idx], ev[idx])["auroc"])[1:]
        out["auroc_R1_for_event"] = {"auroc": float(a["auroc"]), "lo": float(alo), "hi": float(ahi), "n_event": int(ev.sum()), "n_no_event": int((~ev).sum())}
    else:
        out["auroc_R1_for_event"] = {"auroc": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_event": int(ev.sum()), "n_no_event": int((~ev).sum())}
    b = out["improvement"]["B_over_A"]; out["cell_met"] = cell_met(b["improvement"], b["lo"])
    return out


# ----------------------------------------------------------------------------- inputs (torch is imported only here)
def load_arm(func_inputs, arm):
    import phase1_func_f5_analysis as fa
    fi = Path(func_inputs)
    dirs = {"legacy RTN": [fi / "legacy_f5_a", fi / "legacy_f5_b"], "matched RTN": [fi / "legacy_f5_a", fi / "jobs/rtnm-f5-int64"], "GPTQ": [fi / "legacy_f5_a", fi / "jobs/gptq-f5-int64"]}[arm]
    sa, states, meta = fa.load_runs([str(d) for d in dirs])
    return sa, states


def feature_table(sa, states, sv, languages):
    """Per feature (rows of the write side): the baselines of model A, the strata inputs and, per state, R1, jaccard, scale deviation and W1."""
    rows = np.asarray(sa["write_rows"]).astype(int); ids = np.asarray(sa["feat_ids"]).astype(int)[rows]; fl = np.asarray(sa["feat_lang"]).astype(int)[rows]
    if list(sv["languages"]) != list(languages):
        raise ValueError("the F5 language list and the Phase 1 language list differ")
    import torch
    pi = (torch.tensor(fl), torch.tensor(ids))
    fr = sv["firing_rate"][pi].numpy().astype(np.float64); ma = sv["mean_activation"][pi].numpy().astype(np.float64)
    fpw = states["FP"]["write"]
    base = np.column_stack([np.asarray(sa["n_active"], dtype=np.float64)[rows], fr, ma, np.abs(np.asarray(fpw["dnll"], dtype=np.float64)), np.asarray(fpw["n_pos"], dtype=np.float64)])
    T = {"ids": ids, "lang": fl, "base": base, "dnll_fp": np.asarray(fpw["dnll"], dtype=np.float64), "cos16": np.asarray(states["INT16"]["write"]["cos"], dtype=np.float64), "dnll16": np.asarray(states["INT16"]["write"]["dnll"], dtype=np.float64)}
    for s in STATES:
        r, w = states[s]["read"], states[s]["write"]
        T[s] = {"r1": np.asarray(r["corr"], dtype=np.float64)[rows], "jaccard": np.asarray(r["jaccard"], dtype=np.float64)[rows], "scale_dev": np.abs(np.asarray(r["scale"], dtype=np.float64)[rows] - 1.0), "w1": np.asarray(w["cos"], dtype=np.float64)}
    return T


def committed_spearman(arm):
    path = {"legacy RTN": ROOT / "reports/functional_validation/results/f5/f5_results.json", "matched RTN": ROOT / "reports/gptq_extension/results/f5/rtnm/f5_results.json", "GPTQ": ROOT / "reports/gptq_extension/results/f5/gptq/f5_results.json"}[arm]
    return json.loads(path.read_text())["validity"]["spearman"]


def control(T, arm, tol=1e-9):
    """Reproduce the committed Spearman correlation of R1 with W1 and its language-cluster interval (committed code and seed) for INT6 and INT4; returns (ok, rows)."""
    com = committed_spearman(arm); rows = []; ok = True
    for s in STATES:
        x, y, lang = T[s]["r1"], T[s]["w1"], T["lang"]
        sp = met.spearman(x, y); _, lo, hi = met.cluster_bootstrap(lang, lambda idx: met.spearman(x[idx], y[idx])["rho"], n_boot=2000, seed=CONTROL_SEED)
        c = com[s]; same = abs(sp["rho"] - c["rho"]) <= tol and abs(lo - c["lo"]) <= tol and abs(hi - c["hi"]) <= tol and sp["n"] == c["n"]
        ok &= bool(same); rows.append({"arm": arm, "state": s, "rho": sp["rho"], "lo": float(lo), "hi": float(hi), "n": sp["n"], "committed_rho": c["rho"], "committed_lo": c["lo"], "committed_hi": c["hi"], "reproduced": bool(same)})
    return ok, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--func-inputs", default="/home/karma/Documents/ANLP/project/transfer_runs/func_inputs"); ap.add_argument("--phase1", default=os.path.expanduser("~/.cache/anlp-always-late/phase1_runs/p1-mask/eq/survival.pt"))
    ap.add_argument("--out"); ap.add_argument("--control-only", action="store_true"); ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    import torch
    sv = torch.load(a.phase1, map_location="cpu", weights_only=False)
    data = {}; control_rows = []; all_ok = True
    for arm in ARMS:
        sa, states = load_arm(a.func_inputs, arm); meta_langs = json.loads((Path(a.func_inputs) / "jobs/gptq-f5-int64/meta_f5.json").read_text())["languages"]
        T = feature_table(sa, states, sv, meta_langs); data[arm] = T
        ok, rows = control(T, arm); all_ok &= ok; control_rows += rows
        print(f"control {arm}: {'reproduced' if ok else 'MISMATCH'}", *[f"{r['state']} rho {r['rho']:.4f} [{r['lo']:.3f}, {r['hi']:.3f}]" for r in rows])
    if not all_ok:
        print("the reproduction control failed: stopping"); return 1
    if a.control_only or not a.out:
        return 0
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    legacy = data["legacy RTN"]; eps = reference_noise(legacy["dnll_fp"], legacy["dnll16"]); rel = reliable_mask(legacy["cos16"]); loss = loss_mask(legacy["dnll_fp"], eps)
    pops = {"primary": rel, "noisy_stratum": ~rel, "S_all": np.ones(len(rel), dtype=bool), "S_loss": loss}
    res = {"freeze": "reports/survival_effect_rescue/stage3_analysis_freeze_2026-10-28.md", "tau": TAU, "eps_dnll": eps, "n_features": int(len(rel)), "population_counts": {k: int(v.sum()) for k, v in pops.items()},
           "n_boot": a.n_boot, "seed": SEED, "base_columns": list(BASE_NAMES), "cells": {}, "control": control_rows}
    for arm in ARMS:
        T = data[arm]
        for s in STATES:
            Xs = design(T["base"], T[s]["r1"], T[s]["jaccard"], T[s]["scale_dev"])
            for name, mask in pops.items():
                res["cells"][f"{arm}|{s}|{name}"] = analyse_cell(Xs, T[s]["w1"], T["lang"], mask, n_boot=a.n_boot)
                print("done", arm, s, name, flush=True)
    prim = [res["cells"][f"{arm}|{s}|primary"]["cell_met"] for arm in ARMS for s in STATES]
    res["overall_statement"] = overall_statement(prim); res["cells_met_primary"] = int(sum(prim))
    import phase1_func_f5_analysis as fa
    (out / "survival_effect_results.json").write_text(json.dumps(fa.to_jsonable(res), indent=1, allow_nan=False)); print("overall:", res["overall_statement"], f"({res['cells_met_primary']} of 6 cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
