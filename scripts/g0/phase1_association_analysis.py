"""D-035 Task 4 (plan: docs/superpowers/plans/2026-10-21-language-association-retention.md; design: reports/survival_method_reassessment.md section 7): analysis of the association
sidecar written by phase1_association_collect. Loads the per-sentence shards (CSR per condition and language), maps the frozen original cohort by language CODE, computes the status of
every original selective feature in every condition and the pooled and per-language summaries, and attaches paired-bootstrap intervals.

Resampling (register M01 Koehn 2004 section 6, M02 Berg-Kirkpatrick et al. 2012, M04 for clusters): each draw resamples the aligned FLORES sentence ids with replacement and applies the SAME
multiplicities to every language and every condition (FP, INTb, controls), recomputes the token-weighted counts, statuses and summaries, and keeps the dev-selected feature ids, their original
peak languages and the token-selection masks fixed. Intervals are pointwise 2.5 / 97.5 percentiles of the draws, descriptive and conditional on this corpus, these sampled positions and the frozen
cohort; no hypothesis test and no equivalence claim. Denominators are fixed: every original selective feature is counted whatever its status. Distances that are undefined (an all-zero profile)
are excluded from medians and counted, never set to zero."""
import csv, hashlib, io, json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

import phase1_association_collect as ac
import phase1_association_metrics as am

STATUSES = ("retained", "ambiguous_peak", "different_peak", "low_support", "inactive")


class AnalysisError(ValueError):
    pass


def _totals(counts_t, sums_t, ntok, W):
    cnt = np.stack([np.rint(counts_t[l] @ W).T for l in range(len(counts_t))], axis=1).astype(np.int64)
    sm = np.stack([(sums_t[l] @ W).T for l in range(len(sums_t))], axis=1)
    nt = np.rint(np.stack([ntok[l] @ W for l in range(len(ntok))], axis=1)).astype(np.int64)
    return cnt, sm, nt


class Sidecar:
    """Per-sentence data of one stream. Conditions are read from disk on demand; FP (the baseline of every comparison) and at most one other condition stay in memory."""
    def __init__(self, languages, conditions, feature_ids, sentence_ids, loader, n_features_total, stream, keep=("FP",), aggregate_only=False):
        self.aggregate_only = aggregate_only                           # True: one pseudo-sentence holding the stream's aggregate profile (no sentence shards were saved)
        self.languages, self.conditions, self.feature_ids, self.sentence_ids = languages, conditions, feature_ids, sentence_ids
        self._loader, self._keep, self._cache, self.n_features_total, self.stream = loader, tuple(keep), {}, n_features_total, stream
        self.n_sentences, self.n_features = len(sentence_ids), len(feature_ids)

    def _get(self, cond):
        if cond not in self._cache:
            for c in [c for c in self._cache if c not in self._keep]:
                del self._cache[c]
            self._cache[cond] = self._loader(cond)
        return self._cache[cond]

    def loaded_conditions(self):
        return list(self._cache)

    def totals(self, cond, W):
        """Token-weighted totals for resampling weights W [sentences, draws]: counts [draws, L, K] int64, sums [draws, L, K] float64, tokens [draws, L] int64."""
        W = np.asarray(W, dtype=np.float64)
        if W.ndim != 2 or W.shape[0] != self.n_sentences:
            raise AnalysisError(f"weights must have shape [{self.n_sentences}, draws], got {W.shape}")
        return _totals(*self._get(cond), W)


class Cohort:
    def __init__(self, languages, feature_ids, selective, controls, peak, peak_base=None):
        self.languages, self.feature_ids, self.selective, self.controls, self.peak, self.peak_base = languages, feature_ids, selective, controls, peak, peak_base
        self.n_selective = int(selective.sum())


def _read_condition(run_dir, stream, cond, langs):
    ids = feats = None; counts_t, sums_t, ntok, all_ids, all_feats = [], [], [], [], []
    for l in langs:
        sh = ac.read_shard(run_dir, stream, cond, l)
        S, K = len(sh["sentence_ids"]), len(sh["feature_ids"]); row_ptr, col = sh["row_ptr"].numpy(), sh["col"].numpy()
        counts_t.append(sp.csr_matrix((sh["counts"].numpy(), col, row_ptr), shape=(S, K)).T.tocsr())                                  # int16 counts: memory
        sums_t.append(sp.csr_matrix((sh["sums"].numpy().astype(np.float64), col, row_ptr), shape=(S, K)).T.tocsr())
        ntok.append(sh["n_tokens"].numpy().astype(np.float64)); all_ids.append(sh["sentence_ids"]); all_feats.append(sh["feature_ids"])
    return counts_t, sums_t, ntok, all_ids, all_feats


def load_sidecar(run_dir, stream):
    root = Path(run_dir) / "association"
    manifest = json.loads((root / "manifest.json").read_text())
    ac.validate_run(run_dir, manifest["conditions"], manifest["languages"])
    if stream not in manifest["shards"]:
        raise AnalysisError(f"no sentence shards were saved for stream {stream!r} (saved: {sorted(manifest['shards'])})")
    langs, conds = manifest["languages"], manifest["conditions"]
    prof = ac.read_profiles(run_dir, stream)["conditions"]
    ids = feats = fp_ntok = None
    for c in conds:                                                    # one validation pass; nothing but the identifiers is kept
        counts_t, sums_t, ntok, all_ids, all_feats = _read_condition(run_dir, stream, c, langs)
        if ids is None:
            ids, feats = all_ids[0], all_feats[0]
        for l, (i_l, f_l) in enumerate(zip(all_ids, all_feats)):
            if not torch.equal(i_l, ids):
                raise AnalysisError(f"sentence ids of {c}/{langs[l]} differ from the other languages or conditions: translations must be aligned and complete")
            if not torch.equal(f_l, feats):
                raise AnalysisError(f"feature subset of {c}/{langs[l]} differs")
        if not bool((ids[1:] > ids[:-1]).all()):
            raise AnalysisError("sentence ids must be ascending")
        if fp_ntok is None:
            fp_ntok = ntok
        for l in range(len(langs)):
            if not np.array_equal(ntok[l], fp_ntok[l]):
                raise AnalysisError(f"token counts of {c}/{langs[l]} differ from FP: every condition must see the same positions")
        cnt, sm, nt = _totals(counts_t, sums_t, ntok, np.ones((len(ids), 1)))
        if not (np.array_equal(cnt[0], prof[c]["positive_counts"].numpy()[:, feats.numpy()]) and np.array_equal(nt[0], prof[c]["n_tokens"].numpy())
                and np.allclose(sm[0], prof[c]["activation_sums"].numpy()[:, feats.numpy()], rtol=1e-5, atol=1e-9)):
            raise AnalysisError(f"the saved aggregate profile of {c} disagrees with the totals of its sentence shards")
    return Sidecar(langs, conds, feats.numpy(), ids.numpy(), lambda cond: _read_condition(run_dir, stream, cond, langs)[:3], int(manifest["n_features"]), stream)


def load_aggregate_sidecar(run_dir, stream):
    """A Sidecar made of the saved AGGREGATE profile of a stream (one pseudo-sentence): point estimates only, no resampling. For streams without sentence shards (the all-token stream of the full run)."""
    manifest = json.loads((Path(run_dir) / "association" / "manifest.json").read_text())
    ac.validate_run(run_dir, manifest["conditions"], manifest["languages"])
    prof = ac.read_profiles(run_dir, stream)["conditions"]; langs, conds, F = manifest["languages"], manifest["conditions"], int(manifest["n_features"])
    def loader(c):
        p = prof[c]
        counts_t = [sp.csr_matrix(p["positive_counts"][l].numpy().reshape(F, 1)) for l in range(len(langs))]
        sums_t = [sp.csr_matrix(p["activation_sums"][l].numpy().astype(np.float64).reshape(F, 1)) for l in range(len(langs))]
        return counts_t, sums_t, [np.array([float(p["n_tokens"][l])]) for l in range(len(langs))]
    return Sidecar(langs, conds, np.arange(F), np.array([0]), loader, F, stream, aggregate_only=True)


def load_cohort(baseline_dir, side, allow_subset=False):
    """The frozen original cohort (selective features, controls, dev peak languages) restricted to the sentence feature subset and mapped to the run's language order BY CODE.
    With allow_subset (a smoke run on fewer languages) only features whose ORIGINAL peak language is among the run's languages stay in the cohort; the result then checks the machinery,
    not the 27-language question."""
    sel = torch.load(Path(baseline_dir) / "selectivity.pt", map_location="cpu", weights_only=True)
    base = list(sel["languages"])
    same = len(base) == len(side.languages) and set(base) == set(side.languages)
    if not same and not (allow_subset and set(side.languages) <= set(base)):
        raise AnalysisError(f"baseline languages {base} differ from the run's languages {side.languages}")
    if int(sel["selective"].shape[0]) != side.n_features_total:
        raise AnalysisError(f"baseline has {int(sel['selective'].shape[0])} features, the run {side.n_features_total} (feature count)")
    to_run = np.array([side.languages.index(c) if c in side.languages else -1 for c in base])
    peak_all = to_run[sel["peak"].numpy()]                                                           # -1: the original language is not in the run
    in_run = peak_all >= 0
    selective_all = sel["selective"].numpy().astype(bool) & in_run; controls_all = np.zeros_like(selective_all); controls_all[sel["controls"].numpy()] = True; controls_all &= in_run
    f = side.feature_ids
    selective, controls, peak = selective_all[f], controls_all[f], peak_all[f]
    if int(selective.sum()) != int(selective_all.sum()):
        raise AnalysisError("some selective features of the baseline are outside the sentence feature subset of the sidecar")
    return Cohort(side.languages, f, selective, controls, np.where(peak >= 0, peak, 0), sel["peak"].numpy()[f])


def draw_weights(n_sentences, n_draws, seed, start=0):
    """Multiplicities [sentences, draws] of bootstrap draws start .. start + n_draws - 1: sentences drawn with replacement; draw d uses its own generator seeded (seed, d), so results do
    not depend on how the draws are batched."""
    p = np.full(n_sentences, 1.0 / n_sentences)
    return np.stack([np.random.default_rng([int(seed), int(start) + d]).multinomial(n_sentences, p) for d in range(n_draws)], axis=1).astype(np.float64)


def percentile_interval(x, lo=2.5, hi=97.5):
    x = np.asarray(x, dtype=np.float64); x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    a, b = np.percentile(x, [lo, hi]); return float(a), float(b)


def _rates(cnt, nt):
    return cnt / nt.astype(np.float64)[:, None]


def statistics_for_weights(side, coh, W, min_positives=am.MIN_POSITIVES, min_panel_rate=am.MIN_PANEL_RATE, group="selective", conditions=None):
    """{condition: {statistic: array over the draws of W}} for every condition except FP. FP is evaluated once per draw (baseline establishment), conditions are compared with it."""
    if group not in ("selective", "controls"):
        raise AnalysisError("group must be 'selective' or 'controls'")
    W = np.asarray(W, dtype=np.float64); nb = W.shape[1]; L = len(side.languages); sel, peak = (coh.selective if group == "selective" else coh.controls), coh.peak
    fp_cnt, _, fp_nt = side.totals("FP", W)
    if (fp_nt <= 0).any():
        raise AnalysisError("a draw selected no token for some language (FP)")
    fp_stat = [am.association_status(fp_cnt[i], fp_nt[i], peak, min_positives, min_panel_rate) for i in range(nb)]
    fp_met = [am.association_metrics(_rates(fp_cnt[i], fp_nt[i]), _rates(fp_cnt[i], fp_nt[i]), peak) for i in range(nb)]
    out = {}
    for cond in (conditions if conditions is not None else [c for c in side.conditions if c != "FP"]):
        q_cnt, _, q_nt = side.totals(cond, W)                                                  # token counts equal FP's (checked at loading), so FP's emptiness check covers every condition
        rows = {k: [] for k in ("n_selective", "fp_established", "retained_of_established", "lost_of_established", "loss_fraction", "retention_fraction", "conditional_fraction", "fp_establishment_fraction", "median_tv", "n_tv_undefined",
                                "median_delta_share", "median_delta_margin")}
        sc, fsc, bl_n, bl_e, bl_r = [], [], [], [], []
        for i in range(nb):
            qs = am.association_status(q_cnt[i], q_nt[i], peak, min_positives, min_panel_rate)
            fr, qr = _rates(fp_cnt[i], fp_nt[i]), _rates(q_cnt[i], q_nt[i]); qm = am.association_metrics(qr, qr, peak); cmp = am.compare_profiles(fr, qr)
            est = fp_stat[i]["retained"]; e = est & sel; ret = e & qs["retained"]; n_sel, n_e, n_ret = int(sel.sum()), int(e.sum()), int(ret.sum())
            defined = e & cmp["defined"]
            ds = np.where(e & np.isfinite(qm["share_rate"]) & np.isfinite(fp_met[i]["share_rate"]), qm["share_rate"] - fp_met[i]["share_rate"], np.nan)
            dm = np.where(e, qm["margin_rate"] - fp_met[i]["margin_rate"], np.nan)
            nanmed = lambda v: float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")
            rows["n_selective"].append(n_sel); rows["fp_established"].append(n_e); rows["retained_of_established"].append(n_ret)
            rows["lost_of_established"].append(n_e - n_ret); rows["loss_fraction"].append((n_e - n_ret) / n_sel if n_sel else float("nan"))
            rows["retention_fraction"].append(n_ret / n_sel if n_sel else float("nan")); rows["conditional_fraction"].append(n_ret / n_e if n_e else float("nan"))
            rows["fp_establishment_fraction"].append(n_e / n_sel if n_sel else float("nan"))
            rows["median_tv"].append(nanmed(np.where(defined, cmp["tv_distance"], np.nan))); rows["n_tv_undefined"].append(int((e & ~cmp["defined"]).sum()))
            rows["median_delta_share"].append(nanmed(ds)); rows["median_delta_margin"].append(nanmed(dm))
            sc.append([int(((qs["status"] == s) & e).sum()) for s in STATUSES]); fsc.append([int(((fp_stat[i]["status"] == s) & sel).sum()) for s in STATUSES])
            bl_n.append([int((sel & (peak == l)).sum()) for l in range(L)]); bl_e.append([int((sel & (peak == l) & est).sum()) for l in range(L)])
            bl_r.append([int((sel & (peak == l) & ret).sum()) for l in range(L)])
        res = {k: np.array(v, dtype=np.float64 if k.endswith(("fraction", "tv", "share", "margin")) else np.int64) for k, v in rows.items()}
        res["status_counts"] = np.array(sc, dtype=np.int64); res["fp_status_counts"] = np.array(fsc, dtype=np.int64)
        res["by_language_n"], res["by_language_established"], res["by_language_retained"] = (np.array(x, dtype=np.int64) for x in (bl_n, bl_e, bl_r))
        out[cond] = res
    return out


def point_estimate(side, coh, min_positives=am.MIN_POSITIVES, min_panel_rate=am.MIN_PANEL_RATE, group="selective", conditions=None):
    """Statistics of the observed data (unit weights), as plain numbers per condition."""
    st = statistics_for_weights(side, coh, np.ones((side.n_sentences, 1)), min_positives, min_panel_rate, group, conditions); out = {}
    for cond, r in st.items():
        d = {k: (int(v[0]) if v.dtype.kind in "iu" else float(v[0])) for k, v in r.items() if v.ndim == 1}
        d["status_counts"] = {s: int(r["status_counts"][0][j]) for j, s in enumerate(STATUSES)}
        d["fp_status_counts"] = {s: int(r["fp_status_counts"][0][j]) for j, s in enumerate(STATUSES)}
        d["by_language"] = {}
        for l, code in enumerate(side.languages):
            n, e, rt = int(r["by_language_n"][0][l]), int(r["by_language_established"][0][l]), int(r["by_language_retained"][0][l])
            d["by_language"][code] = {"n_selective": n, "fp_established": e, "retained": rt, "retention_fraction": rt / n if n else float("nan")}
        out[cond] = d
    return out


def bootstrap(side, coh, n_boot=1000, seed=0, min_positives=am.MIN_POSITIVES, min_panel_rate=am.MIN_PANEL_RATE, batch=50, group="selective", conditions=None):
    """Paired cluster bootstrap over the aligned sentence ids: {condition: {statistic: array [n_boot, ...]}}. The same multiplicities apply to every language and condition. Conditions are
    processed one after the other (each is read from disk once), draws in batches."""
    out = {}
    for cond in (conditions if conditions is not None else [c for c in side.conditions if c != "FP"]):
        parts = []
        for start in range(0, n_boot, batch):
            nb = min(batch, n_boot - start)
            parts.append(statistics_for_weights(side, coh, draw_weights(side.n_sentences, nb, seed, start), min_positives, min_panel_rate, group, [cond])[cond])
        out[cond] = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
    return out


# ---------------------------------------------------------------- outputs
SCALARS = ("retention_fraction", "loss_fraction", "conditional_fraction", "fp_establishment_fraction", "median_tv", "median_delta_share", "median_delta_margin")
CONDITION_ORDER_NOTE = "conditions are listed in the order of the sidecar manifest"


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    return o


def _fmt(x):
    return "" if x is None or not np.isfinite(x) else format(float(x), ".10g")


def _md(x, digits=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{float(x):.{digits}f}"


def _nan_interval():
    return [float("nan"), float("nan")]


def _summarise(side, point, boot):
    """{condition: {"point": ..., "interval": {...}, "by_language": {code: {..., "interval": [lo, hi]}}, "pooled" and "macro" entries}} for one group."""
    out = {}
    L = len(side.languages)
    for cond, pt in point.items():
        b = boot[cond]
        if b is None:                                                   # no resampling possible: point estimates only
            n0 = len(side.languages); empty = {k: np.zeros(0) for k in SCALARS}; empty.update({"status_counts": np.zeros((0, len(STATUSES))), "by_language_retained": np.zeros((0, n0)), "by_language_established": np.zeros((0, n0))}); b = empty
        iv = {k: list(percentile_interval(b[k])) for k in SCALARS}
        iv["status_counts"] = {s: list(percentile_interval(b["status_counts"][:, j])) for j, s in enumerate(STATUSES)}
        diag = {}
        for k in SCALARS:
            x = b[k][~np.isnan(b[k])]; lo, hi = iv[k]
            diag[k] = {"bootstrap_mean": float(x.mean()) if x.size else float("nan"), "point_in_interval": bool(np.isfinite(pt[k]) and np.isfinite(lo) and lo <= pt[k] <= hi)}
        n_l = np.array([pt["by_language"][c]["n_selective"] for c in side.languages], dtype=np.float64)
        frac = np.where(n_l > 0, b["by_language_retained"] / np.where(n_l > 0, n_l, 1), np.nan)
        by = {}
        lossfrac = np.where(n_l > 0, (b["by_language_established"] - b["by_language_retained"]) / np.where(n_l > 0, n_l, 1), np.nan)
        for l, code in enumerate(side.languages):
            by[code] = {**pt["by_language"][code], "interval": list(percentile_interval(frac[:, l])), "loss_interval": list(percentile_interval(lossfrac[:, l]))}
        used = n_l > 0
        macro_draws = np.nanmean(frac[:, used], axis=1) if used.any() else np.full(len(frac), np.nan)
        pf = np.array([pt["by_language"][c]["retention_fraction"] for c in side.languages], dtype=np.float64)
        out[cond] = {"point": {k: v for k, v in pt.items() if k != "by_language"}, "interval": iv, "diagnostics": diag, "by_language": by,
                     "macro": {"retention_fraction": float(np.mean(pf[used])) if used.any() else float("nan"), "n_languages": int(used.sum()), "interval": list(percentile_interval(macro_draws))}}
    return out


def _feature_rows(side, coh, bl_surv, min_positives, min_panel_rate):
    ones = np.ones((side.n_sentences, 1)); fp_cnt, fp_sm, fp_nt = (a[0] for a in side.totals("FP", ones)); peak = coh.peak
    fp_stat = am.association_status(fp_cnt, fp_nt, peak, min_positives, min_panel_rate); fr = _rates(fp_cnt, fp_nt)
    members = np.nonzero(coh.selective | coh.controls)[0]; rows = []
    for cond in side.conditions:
        cnt, sm, nt = (a[0] for a in side.totals(cond, ones)); st = am.association_status(cnt, nt, peak, min_positives, min_panel_rate)
        rates = _rates(cnt, nt); means = sm / nt.astype(np.float64)[:, None]; met = am.association_metrics(rates, means, peak)
        cmp = am.compare_profiles(fr, rates) if cond != "FP" else None
        bits = int(cond[3:]) if cond.startswith("INT") else None
        corr = None if bl_surv is None or bits is None or bits not in bl_surv["corr"] else bl_surv["corr"][bits].numpy()
        for k in members:
            peak_set = ";".join(side.languages[l] for l in np.nonzero(met["peak_set_rate"][:, k])[0])
            rows.append({"feature_id": int(side.feature_ids[k]), "group": "selective" if coh.selective[k] else "controls", "original_language": side.languages[peak[k]], "condition": cond,
                         "fp_status": str(fp_stat["status"][k]), "status": str(st["status"][k]), "peak_set": peak_set, "peak_count": int(st["peak_count"][k]),
                         "positives_at_peak": int(st["positives_at_peak"][k]), "positives_at_target": int(st["positives_at_target"][k]), "target_rate": _fmt(met["target_rate"][k]),
                         "target_share": _fmt(met["share_rate"][k]), "margin": _fmt(met["margin_rate"][k]),
                         "tv_distance": "" if cmp is None or not cmp["defined"][k] else _fmt(cmp["tv_distance"][k]), "target_mean": _fmt(met["target_mean"][k]), "nu_target": _fmt(met["nu_target"][k]),
                         "historical_corr": "" if corr is None else _fmt(corr[coh.peak_base[k], int(side.feature_ids[k])])})
    return sorted(rows, key=lambda r: (r["group"] != "selective", r["feature_id"], side.conditions.index(r["condition"])))


def _csv_text(rows, fields):
    buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n"); w.writeheader(); w.writerows(rows); return buf.getvalue()


def _language_rows(side, summaries):
    rows = []
    for group, per_cond in summaries.items():
        for cond, s in per_cond.items():
            def row(lang, n, e, r, frac, iv, cond_frac, n_lang="", loss_iv=None):
                lost = e - r if e != "" else ""
                return {"group": group, "condition": cond, "language": lang, "n_selective": n, "fp_established": e, "retained": r, "lost": lost, "retention_fraction": _fmt(frac),
                        "retention_lo": _fmt(iv[0]) if iv else "", "retention_hi": _fmt(iv[1]) if iv else "", "loss_fraction": _fmt(lost / n) if (n not in ("", 0) and lost != "") else "",
                        "loss_lo": _fmt(loss_iv[0]) if loss_iv else "", "loss_hi": _fmt(loss_iv[1]) if loss_iv else "", "conditional_fraction": _fmt(cond_frac), "n_languages": n_lang}
            for code in side.languages:
                b = s["by_language"][code]
                rows.append(row(code, b["n_selective"], b["fp_established"], b["retained"], b["retention_fraction"], b["interval"], b["retained"] / b["fp_established"] if b["fp_established"] else None, "", b["loss_interval"]))
            pt, iv = s["point"], s["interval"]
            rows.append(row("POOLED", pt["n_selective"], pt["fp_established"], pt["retained_of_established"], pt["retention_fraction"], iv["retention_fraction"], pt["conditional_fraction"], "", iv["loss_fraction"]))
            rows.append(row("MACRO_LANGUAGE_MEAN", "", "", "", s["macro"]["retention_fraction"], s["macro"]["interval"], None, s["macro"]["n_languages"]))
    return rows


def _report(config, prov, side, coh, summaries):
    sel = summaries["selective"]; ctl = summaries["controls"]; lines = []
    lines += ["# Language association retention (D-035)", "",
              "Generated by `scripts/g0/phase1_association_analysis.py` from the saved sidecar; the numbers below are produced from the files and the static prose states no results. "
              "This is an exploratory measurement on already examined FLORES data (dev selected the cohort, devtest is evaluated; both splits were seen before), not a confirmatory test.", "",
              "## Configuration and provenance", "",
              (f"- Stream `{config['stream']}`; aggregate profile only, no resampling." if config.get("aggregate_only") else f"- Stream `{config['stream']}`; {config['n_boot']} paired bootstrap draws over aligned sentence ids, seed {config['seed']}; pointwise {config['interval']} intervals."),
              f"- Support rules: at least {config['min_positives']} positive tokens at the peak and a panel-mean rate of at least {config['min_panel_rate']:g}.",
              f"- {len(side.languages)} languages ({', '.join(side.languages)}); {coh.n_selective} original selective features, {int(coh.controls.sum())} matched controls, {side.n_features} features with sentence rows.",
              *(["- No sentence-level shards were saved for this stream: the values are point estimates from the aggregate profiles and carry no intervals."] if config.get("aggregate_only") else []),
              f"- Sidecar manifest sha256 `{prov['run_manifest_sha256'][:12]}`; baseline selectivity sha256 `{prov['baseline_selectivity_sha256'][:12]}`.", "",
              "## Original selective features: headline per condition", "",
              "Every original selective feature is in the fixed denominator whatever its status. Intervals are pointwise percentile intervals over sentence resamples, conditional on this corpus, these sampled positions and the frozen cohort.", "",
              "| condition | selective | FP baseline established | retained, fixed denominator | lost (established in FP, not retained), fixed denominator | retained among established | median change of target share | median change of margin | median total variation | undefined distances |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    cell = lambda p, iv: f"{_md(p)} [{_md(iv[0])}, {_md(iv[1])}]"
    for cond, s in sel.items():
        pt, iv = s["point"], s["interval"]
        lines.append(f"| {cond} | {pt['n_selective']} | {pt['fp_established']} ({cell(pt['fp_establishment_fraction'], iv['fp_establishment_fraction'])}) | {pt['retained_of_established']} ({cell(pt['retention_fraction'], iv['retention_fraction'])}) | "
                     f"{pt['lost_of_established']} ({cell(pt['loss_fraction'], iv['loss_fraction'])}) | {cell(pt['conditional_fraction'], iv['conditional_fraction'])} | {cell(pt['median_delta_share'], iv['median_delta_share'])} | {cell(pt['median_delta_margin'], iv['median_delta_margin'])} | "
                     f"{cell(pt['median_tv'], iv['median_tv'])} | {pt['n_tv_undefined']} |")
    lines += ["", "## Status of the features established in full precision", "", "| condition | " + " | ".join(STATUSES) + " |", "|---|" + "---|" * len(STATUSES)]
    for cond, s in sel.items():
        lines.append(f"| {cond} | " + " | ".join(str(s["point"]["status_counts"][x]) for x in STATUSES) + " |")
    lines += ["", "## Retention fraction by original language (fixed denominators; blank where the language has no selective feature)", "", "| language | selective | " + " | ".join(sel) + " |", "|---|---|" + "---|" * len(sel)]
    for code in side.languages:
        first = next(iter(sel.values()))["by_language"][code]
        lines.append(f"| {code} | {first['n_selective']} | " + " | ".join("" if not s["by_language"][code]["n_selective"] else _md(s["by_language"][code]["retention_fraction"]) for s in sel.values()) + " |")
    lines += ["", "## Matched controls (descriptive; their baseline labels are their saved peak labels, not a claim of selectivity)", "", "| condition | controls | FP baseline established | retained, fixed denominator | median total variation |", "|---|---|---|---|---|"]
    for cond, s in ctl.items():
        pt, iv = s["point"], s["interval"]
        lines.append(f"| {cond} | {pt['n_selective']} | {pt['fp_established']} | {pt['retained_of_established']} ({cell(pt['retention_fraction'], iv['retention_fraction'])}) | {cell(pt['median_tv'], iv['median_tv'])} |")
    lines += ["", "## How to read this", "",
              "- A retained peak is not unchanged specificity: the target share, the margin and the total variation are reported next to it, and a common rescaling leaves shares and distances unchanged while amplitudes change.",
              "- The restore-only control and the high-precision controls characterise the numerical and sampling floor; they are measured, not assumed to preserve every association.",
              "- Percentile intervals of fractions that depend on thresholds and on the uniqueness of a maximum are pulled toward the middle by resampling, so the observed value can lie outside its interval (`analysis.json` records the bootstrap mean and whether the observed value is inside). The loss fraction (features established in full precision that a condition does not retain) is paired draw by draw, so it is less affected by the threshold noise of the baseline and is the statistic to read for the effect of a condition, but it can also fall outside its interval; the restore-only control has zero loss in every draw by construction.",
              "- Intervals describe sentence-sampling variation only. They do not cover other corpora, other dictionaries or the choice of thresholds, and an interval that includes no change is not evidence that function or meaning is unchanged.",
              "- Per-feature rows are in `feature_status.csv`, per-language rows with intervals in `language_summary.csv`, everything numeric in `analysis.json`; no hypothesis test is run.", ""]
    return "\n".join(lines)


def analyse_association(run_dir, baseline_dir, out_dir, stream="eq", n_boot=1000, seed=0, min_positives=am.MIN_POSITIVES, min_panel_rate=am.MIN_PANEL_RATE, batch=50, allow_subset=False):
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise AnalysisError(f"{out} is not empty: reports are generated into a fresh directory")
    has_shards = stream in json.loads((Path(run_dir) / "association" / "manifest.json").read_text())["shards"]
    side = load_sidecar(run_dir, stream) if has_shards else load_aggregate_sidecar(run_dir, stream)
    if not has_shards:
        n_boot = 0
    coh = load_cohort(baseline_dir, side, allow_subset)
    groups = ("selective", "controls"); pts = {g: {} for g in groups}; boots = {g: {} for g in groups}
    for cond in [c for c in side.conditions if c != "FP"]:                 # each condition is read once and used for both groups
        for g in groups:
            pts[g].update(point_estimate(side, coh, min_positives, min_panel_rate, g, [cond]))
            boots[g].update(bootstrap(side, coh, n_boot, seed, min_positives, min_panel_rate, batch, g, [cond]) if n_boot else {cond: None})
    summaries = {g: _summarise(side, pts[g], boots[g]) for g in groups}
    bl = Path(baseline_dir) / "survival.pt"; bl_surv = torch.load(bl, map_location="cpu", weights_only=True) if bl.exists() else None
    manifest_bytes = (Path(run_dir) / "association" / "manifest.json").read_bytes()
    prov = {"run_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "baseline_selectivity_sha256": hashlib.sha256((Path(baseline_dir) / "selectivity.pt").read_bytes()).hexdigest(),
            "languages": side.languages, "conditions": side.conditions, "n_sentences": side.n_sentences, "n_features_with_rows": side.n_features}
    config = {"stream": stream, "n_boot": int(n_boot), "seed": int(seed), "min_positives": int(min_positives), "min_panel_rate": float(min_panel_rate), "batch": int(batch),
              "subset": bool(allow_subset), "aggregate_only": not has_shards, "interval": "percentile 2.5 / 97.5 (pointwise, descriptive)", "resampling": "paired bootstrap of aligned sentence ids, same multiplicities for every language and condition"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(_clean({"schema_version": 1, "config": config, "provenance": prov, "groups": summaries}), indent=1, sort_keys=True, allow_nan=False) + "\n")
    fields = ["feature_id", "group", "original_language", "condition", "fp_status", "status", "peak_set", "peak_count", "positives_at_peak", "positives_at_target", "target_rate", "target_share",
              "margin", "tv_distance", "target_mean", "nu_target", "historical_corr"]
    (out / "feature_status.csv").write_text(_csv_text(_feature_rows(side, coh, bl_surv, min_positives, min_panel_rate), fields))
    lf = ["group", "condition", "language", "n_selective", "fp_established", "retained", "lost", "retention_fraction", "retention_lo", "retention_hi", "loss_fraction", "loss_lo", "loss_hi", "conditional_fraction", "n_languages"]
    (out / "language_summary.csv").write_text(_csv_text(_language_rows(side, summaries), lf))
    (out / "association_retention.md").write_text(_report(config, prov, side, coh, summaries))
    return {"config": config, "provenance": prov, "groups": summaries}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="D-035 association-retention analysis of an association sidecar")
    ap.add_argument("--run", required=True, help="run directory holding association/ (the sidecar)"); ap.add_argument("--baseline", required=True, help="the original mask/eq directory (selectivity.pt, survival.pt)")
    ap.add_argument("--out", required=True); ap.add_argument("--stream", default="eq", choices=["eq", "all"]); ap.add_argument("--n-boot", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-positives", type=int, default=am.MIN_POSITIVES); ap.add_argument("--min-panel-rate", type=float, default=am.MIN_PANEL_RATE); ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--allow-subset", action="store_true", help="smoke runs on fewer languages: keep only the features whose original language is in the run (checks the machinery, not the question)")
    a = ap.parse_args(argv)
    analyse_association(a.run, a.baseline, a.out, a.stream, a.n_boot, a.seed, a.min_positives, a.min_panel_rate, a.batch, a.allow_subset)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
