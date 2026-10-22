"""Metrics of the F5 study (plan Task 11; design docs/superpowers/specs/2026-10-23-feature-function-equivalence-design.md): does an SAE feature of the full-precision (FP) model do the same thing in a quantized model, and
is the survival metric therefore sensible? Pure numpy; no model, no GPU. The runner (Task 13) feeds it sufficient statistics and fingerprints, the analysis (Task 14) feeds it the saved statistics.

Read side (R): per-feature agreement of the activations in FP and in the quantized model on the same positions, from streaming sufficient statistics (ReadAccumulator): token-level Pearson correlation (the baseline
survival basis), firing-set Jaccard, scale ratio over the positions active in both.
Write side (W): comparison of the effect fingerprints (the centred logit change of a single-feature ablation): cosine (no effect in the quantized model = similarity 0 and flagged, no effect in FP = undefined),
top-k promoted-token overlap, magnitude ratio, and retrieval of the feature's own identity among all candidates (ties resolved AGAINST the true feature, counted).
Validity: numerical floor percentile, share above the floor, AUROC (ties count half), Spearman correlation, a seeded cluster bootstrap over languages and the paired median difference by cluster.
Undefined values are NaN and are counted, never zero."""
import numpy as np


class ReadAccumulator:
    """Streaming sufficient statistics of one precision against the FP activations of the same positions. add(x, y): x = FP activations (positions, features), y = quantized activations; features are the columns."""
    def __init__(self, n_features):
        self.F = int(n_features); z = lambda: np.zeros(self.F, dtype=np.float64)
        self.n = 0; self.sx, self.sy, self.sxx, self.syy, self.sxy = z(), z(), z(), z(), z()
        self.n_x, self.n_y, self.n_both, self.sx_both, self.sy_both = z(), z(), z(), z(), z()

    def add(self, x, y):
        x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or x.shape != y.shape:
            raise ValueError(f"x and y must be 2-d arrays of the same shape, got {x.shape} and {y.shape}")
        if x.shape[1] != self.F:
            raise ValueError(f"expected {self.F} features (columns), got {x.shape[1]}")
        ax, ay = x > 0, y > 0; both = ax & ay
        self.n += x.shape[0]; self.sx += x.sum(0); self.sy += y.sum(0); self.sxx += (x * x).sum(0); self.syy += (y * y).sum(0); self.sxy += (x * y).sum(0)
        self.n_x += ax.sum(0); self.n_y += ay.sum(0); self.n_both += both.sum(0); self.sx_both += np.where(both, x, 0.0).sum(0); self.sy_both += np.where(both, y, 0.0).sum(0)

    def finish(self):
        n = float(self.n)
        with np.errstate(divide="ignore", invalid="ignore"):
            mx, my = self.sx / n, self.sy / n
            vx, vy = self.sxx / n - mx * mx, self.syy / n - my * my
            cov = self.sxy / n - mx * my
            corr = np.where((vx > 1e-15 * (self.sxx / n + 1e-300)) & (vy > 1e-15 * (self.syy / n + 1e-300)), cov / np.sqrt(vx * vy), np.nan)
            union = self.n_x + self.n_y - self.n_both
            jaccard = np.where(union > 0, self.n_both / union, np.nan)
            scale = np.where((self.n_both > 0) & (self.sx_both > 0), self.sy_both / self.sx_both, np.nan)
        return {"n": self.n, "corr": corr, "jaccard": jaccard, "scale": scale, "n_active_fp": self.n_x.astype(int), "n_active_q": self.n_y.astype(int), "n_both": self.n_both.astype(int)}


def cosine_rows(U, V):
    """Row-wise cosine similarity of fingerprints U (full precision) and V (quantized). A zero V row (the feature does nothing in the quantized model) gives similarity 0 and is flagged; a zero U row (nothing to
    compare with) gives NaN. Returns (similarity, {"zero_fp", "zero_q"})."""
    U = np.asarray(U, dtype=np.float64); V = np.asarray(V, dtype=np.float64)
    nu, nv = np.linalg.norm(U, axis=1), np.linalg.norm(V, axis=1); zero_fp, zero_q = nu == 0, nv == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = (U * V).sum(1) / (nu * nv)
    sim = np.where(zero_fp, np.nan, np.where(zero_q, 0.0, sim))
    return sim, {"zero_fp": zero_fp, "zero_q": zero_q}


def _top_indices(row, k):
    return set(np.argsort(-row, kind="stable")[:k].tolist())                                         # the largest values, ties by index order


def topk_overlap(U, V, k):
    """Share of the k most PROMOTED tokens (largest entries, not largest magnitudes) of each U row that are also among the k most promoted of the same V row; NaN where either row is zero."""
    U = np.asarray(U, dtype=np.float64); V = np.asarray(V, dtype=np.float64)
    if k > U.shape[1] or k < 1:
        raise ValueError(f"k = {k} does not fit fingerprints of length {U.shape[1]}")
    out = np.full(U.shape[0], np.nan)
    for i in range(U.shape[0]):
        if np.any(U[i]) and np.any(V[i]):
            out[i] = len(_top_indices(U[i], k) & _top_indices(V[i], k)) / k
    return out


def magnitude_ratio(U, V):
    """Norm of the quantized fingerprint over the norm of the full-precision one; NaN where the full-precision fingerprint is zero."""
    nu = np.linalg.norm(np.asarray(U, dtype=np.float64), axis=1); nv = np.linalg.norm(np.asarray(V, dtype=np.float64), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(nu > 0, nv / nu, np.nan)


def _retrieve(S, true, allowed):
    """Ranks of the true candidate per query; allowed(i) -> boolean mask of the candidates that compete with query i (never the true one). Ties count against the true candidate."""
    ranks, ties, undefined = [], 0, 0
    for i in range(S.shape[0]):
        st = S[i, true[i]]
        if np.isnan(st):
            undefined += 1; ranks.append(np.nan); continue
        mask = allowed(i); mask[true[i]] = False
        others = S[i][mask]; others = others[~np.isnan(others)]
        better, tied = int((others > st).sum()), int((others == st).sum())
        ties += tied > 0; ranks.append(1 + better + tied)
    return np.array(ranks, dtype=float), ties, undefined


def retrieval(S, true=None, groups=None):
    """Functional identity: is the feature itself the best match for its own full-precision fingerprint? S[i, j] = similarity of the full-precision fingerprint of query i and the quantized fingerprint of candidate j;
    true[i] = the index of the true candidate (default: the diagonal of a square matrix). Rank = 1 + the candidates at or above the true one's similarity (ties against the true one, counted). groups (per candidate) adds
    the same numbers when only the candidates of the true candidate's group compete (queries whose group has no other member are excluded and counted), with the chance level (mean of 1 / group size)."""
    S = np.asarray(S, dtype=np.float64)
    if true is None:
        if S.shape[0] != S.shape[1]:
            raise ValueError("a non-square similarity matrix needs the indices of the true candidates")
        true = np.arange(S.shape[0])
    true = np.asarray(true, dtype=int)
    if true.shape != (S.shape[0],) or true.min() < 0 or true.max() >= S.shape[1]:
        raise ValueError("true must give, per query, a candidate index inside the matrix")
    ranks, ties, undefined = _retrieve(S, true, lambda i: np.ones(S.shape[1], dtype=bool))
    ok = ~np.isnan(ranks)
    out = {"top1": float((ranks[ok] == 1).mean()) if ok.any() else float("nan"), "mrr": float((1.0 / ranks[ok]).mean()) if ok.any() else float("nan"), "n_queries": int(ok.sum()), "n_undefined": int(undefined),
           "n_ties": int(ties), "ranks": ranks}
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != (S.shape[1],):
            raise ValueError("groups must give one label per candidate")
        keep = np.array([int((groups == groups[true[i]]).sum()) > 1 for i in range(S.shape[0])]); single = int((~keep).sum())
        sub = np.flatnonzero(keep)
        r, t_, u_ = _retrieve(S[sub], true[sub], lambda i: groups == groups[true[sub[i]]]) if len(sub) else (np.array([]), 0, 0)
        good = ~np.isnan(r)
        sizes = np.array([int((groups == groups[true[i]]).sum()) for i in sub], dtype=float)
        out["within"] = {"top1": float((r[good] == 1).mean()) if good.any() else float("nan"), "mrr": float((1.0 / r[good]).mean()) if good.any() else float("nan"), "n_queries": int(good.sum()),
                         "n_undefined": int(u_), "n_singletons": single, "n_ties": int(t_), "chance": float((1.0 / sizes).mean()) if len(sizes) else float("nan")}
    return out


def floor_threshold(reference, q=0.05):
    """The q-quantile (default the 5th percentile) of a reference sample (for example W1 at INT8), undefined values ignored."""
    ref = np.asarray(reference, dtype=np.float64); ref = ref[~np.isnan(ref)]
    if not len(ref):
        raise ValueError("the reference sample has no defined value")
    return float(np.percentile(ref, 100 * q))


def share_above(values, threshold):
    """Share of the defined values at or above the threshold, with the number of defined and of all values."""
    v = np.asarray(values, dtype=np.float64); d = ~np.isnan(v)
    return {"share": float((v[d] >= threshold).mean()) if d.any() else float("nan"), "n_defined": int(d.sum()), "n_total": int(v.size)}


def _ranks(x):
    u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    start = np.cumsum(cnt) - cnt
    return (start + (cnt + 1) / 2.0)[inv]                                                              # average rank, 1-based


def auroc(score, event):
    """Probability that a random event row has a higher score than a random non-event row, ties counting half (Mann-Whitney). Rows with an undefined score are excluded and counted; NaN without both classes."""
    s = np.asarray(score, dtype=np.float64); e = np.asarray(event).astype(bool); d = ~np.isnan(s)
    pos, neg = s[d & e], s[d & ~e]
    out = {"n_pos": int(len(pos)), "n_neg": int(len(neg)), "n_undefined": int((~d).sum()), "auroc": float("nan")}
    if len(pos) and len(neg):
        r = _ranks(np.concatenate([pos, neg]))
        out["auroc"] = float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))
    return out


def spearman(x, y):
    """Spearman rank correlation over the pairs where both values are defined (average ranks for ties); NaN for fewer than 3 pairs or no variation."""
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64); d = ~(np.isnan(x) | np.isnan(y)); n = int(d.sum())
    if n < 3:
        return {"rho": float("nan"), "n": n}
    rx, ry = _ranks(x[d]), _ranks(y[d])
    if rx.std() == 0 or ry.std() == 0:
        return {"rho": float("nan"), "n": n}
    return {"rho": float(np.corrcoef(rx, ry)[0, 1]), "n": n}


def cluster_bootstrap(clusters, fn, n_boot=2000, seed=0, level=0.95, return_counts=False):
    """Seeded percentile bootstrap that resamples WHOLE clusters (languages), never rows: fn(row_indices) -> statistic; returns (observed value, lower, upper), or a dict with the counts of draws and of draws in which the
    statistic was undefined (NaN; they are left out of the interval). At least two clusters are needed."""
    clusters = np.asarray(clusters); labels = np.unique(clusters)
    if len(labels) < 2:
        raise ValueError("a cluster bootstrap needs at least two clusters")
    by = [np.flatnonzero(clusters == c) for c in labels]; rng = np.random.default_rng(seed)
    point = float(fn(np.arange(len(clusters)))); vals = []
    for _ in range(n_boot):
        ids = rng.integers(0, len(by), len(by))
        vals.append(float(fn(np.concatenate([by[i] for i in ids]))))
    v = np.array(vals); ok = ~np.isnan(v)
    lo, hi = (np.percentile(v[ok], [100 * (1 - level) / 2, 100 * (1 + level) / 2]) if ok.any() else (float("nan"), float("nan")))
    if return_counts:
        return {"point": point, "lo": float(lo), "hi": float(hi), "n_boot": int(n_boot), "n_undefined": int((~ok).sum())}
    return point, float(lo), float(hi)


def paired_median_difference(a, clusters_a, b, clusters_b):
    """Median of the values a minus median of the values b within each cluster (the language), then the median over clusters. The two sets may differ in size (lost against matched retained features); a cluster that is
    missing on one side or has no defined value on one side is left out."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64); ca = np.asarray(clusters_a); cb = np.asarray(clusters_b); by = {}
    for c in sorted(set(ca.tolist()) & set(cb.tolist())):
        va, vb = a[ca == c], b[cb == c]
        if np.isnan(va).all() or np.isnan(vb).all():
            continue
        by[c] = float(np.nanmedian(va) - np.nanmedian(vb))
    return {"by_cluster": by, "median": float(np.median(list(by.values()))) if by else float("nan")}
