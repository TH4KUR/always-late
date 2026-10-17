"""Phase 1 library (labelled: new code built ON TOP of Duan et al.'s setup; their functions are reused verbatim by phase1_run.py).
Pure logic, tested on CPU in scripts/g0/tests_torch. Decisions: reports/decisions_and_anomalies.md (D-001, D-003, D-004, D-010).

S1: batching with BOS regimes, per-language streaming Pearson (their float64 sufficient statistics), sentence-level tagging statistics."""
import torch

REGIMES = ("mask", "none")     # mask: BOS prepended, position 0 excluded from every statistic (Gemma Scope's own protocol); none: Duan's regime


def budget_for(data, max_len=256):
    """Equal token budget (decision D-015): the smallest per-language number of content tokens (sentences truncated to max_len; BOS and padding never count)."""
    return min(sum(min(len(s), max_len) for s in ss) for ss in data.values())


def keep_masks(data, budget, seed, split_offset=0, max_len=256):
    """{language: [bool tensor per sentence, length = truncated sentence length]} with EXACTLY min(budget, n_tokens) True per language, drawn uniformly without replacement.
    The generator is seeded by (seed, split_offset, language) only, so a language's draw does not depend on the other languages, on batching, or on the quantization rung."""
    out = {}
    for lang, ss in data.items():
        lens = [min(len(x), max_len) for x in ss]; n = sum(lens)
        g = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(split_offset) * 10_007 + int(lang))
        flat = torch.zeros(n, dtype=torch.bool); flat[torch.randperm(n, generator=g)[:min(budget, n)]] = True
        out[lang] = list(torch.split(flat, lens))
    return out


def make_batches(data, bos_id, regime, pad_id=0, max_len=256, batch_size=32, max_tokens=None, keep=None):
    """data: {lang_index: [token-id lists per sentence, no special tokens]} -> list of batches, each single-language and length-sorted.
    tokens [B, L] right-padded (a causal model cannot be influenced by later padding); valid [B, L] marks real content positions only
    (never BOS under `mask`, never padding); sent = sentence indices within the language; content longer than max_len is truncated;
    a batch holds at most batch_size sentences and, if max_tokens is given, at most max_tokens padded positions (B x L).
    eq [B, L] marks the positions of the equal-budget stream: valid AND selected by `keep` (from keep_masks); without `keep` it equals valid."""
    if regime not in REGIMES:
        raise ValueError(f"unknown BOS regime {regime!r}; choose from {REGIMES}")
    off, batches = (1 if regime == "mask" else 0), []
    for lang in sorted(data):
        order = sorted(range(len(data[lang])), key=lambda i: (len(data[lang][i]), i))
        groups, cur = [], []
        for i in order:
            new_L = min(len(data[lang][i]), max_len) + off
            if cur and (len(cur) + 1 > batch_size or (max_tokens is not None and (len(cur) + 1) * new_L > max_tokens)):
                groups.append(cur); cur = []
            cur.append(i)
        if cur:
            groups.append(cur)
        for idx in groups:
            seqs, trunc = [], 0
            for i in idx:
                s_ = data[lang][i]
                if len(s_) > max_len:
                    s_, trunc = s_[:max_len], trunc + 1
                seqs.append(s_)
            L = max(len(s_) for s_ in seqs) + off
            tokens = torch.full((len(seqs), L), pad_id, dtype=torch.long)
            valid = torch.zeros((len(seqs), L), dtype=torch.bool); eq = torch.zeros((len(seqs), L), dtype=torch.bool)
            for r, s_ in enumerate(seqs):
                if off:
                    tokens[r, 0] = bos_id
                if s_:
                    tokens[r, off:off + len(s_)] = torch.tensor(s_, dtype=torch.long)
                    valid[r, off:off + len(s_)] = True
                    eq[r, off:off + len(s_)] = True if keep is None else keep[lang][idx[r]]
            batches.append({"lang": lang, "tokens": tokens, "valid": valid, "eq": eq, "sent": torch.tensor(idx, dtype=torch.long), "n_truncated": trunc})
    return batches


class LangPearson:
    """Their streaming_feature_metrics math (float64 running sums, corr clamped to [-1, 1], denominator clamped at sqrt(1e-12)), one set of sums per language."""
    def __init__(self, n_lang, d, device):
        z = lambda: torch.zeros(n_lang, d, dtype=torch.float64, device=device)
        self.sx, self.sy, self.sx2, self.sy2, self.sxy, self.fire, self.maxx = z(), z(), z(), z(), z(), z(), z()
        self.n, self.d, self.n_lang = [0] * n_lang, d, n_lang

    def update(self, lang, x, y):
        if x.shape[0] == 0:                                   # a batch of the equal-budget stream can hold no selected position
            return
        x, y = x.to(torch.float64), y.to(torch.float64)
        self.sx[lang] += x.sum(0); self.sy[lang] += y.sum(0); self.sx2[lang] += (x ** 2).sum(0); self.sy2[lang] += (y ** 2).sum(0)
        self.sxy[lang] += (x * y).sum(0); self.fire[lang] += (x > 0).sum(0); self.maxx[lang] = torch.maximum(self.maxx[lang], x.max(0).values)
        self.n[lang] += x.shape[0]

    @staticmethod
    def _finish(sx, sy, sx2, sy2, sxy, fire, maxx, n):
        if n == 0:
            z = torch.zeros(sx.shape[0], dtype=torch.float64)
            return {"n": 0, "corr": z, "firing_rate": z.clone(), "mean_activation": z.clone(), "max_activation": z.clone()}
        num = sxy - sx * sy / n
        den = torch.sqrt(torch.clamp((sx2 - sx ** 2 / n) * (sy2 - sy ** 2 / n), min=1e-12))
        return {"n": n, "corr": torch.clamp(num / den, -1.0, 1.0).cpu(), "firing_rate": (fire / n).cpu(), "mean_activation": (sx / n).cpu(), "max_activation": maxx.cpu()}

    def result(self, lang):
        g = lambda t: t[lang]
        return self._finish(g(self.sx), g(self.sy), g(self.sx2), g(self.sy2), g(self.sxy), g(self.fire), g(self.maxx), self.n[lang])

    def pooled(self):
        s = lambda t: t.sum(0)
        return self._finish(s(self.sx), s(self.sy), s(self.sx2), s(self.sy2), s(self.sxy), s(self.fire), self.maxx.max(0).values, sum(self.n))


class TagStats:
    """Per-sentence tagging statistics for the dev split: number of positive tokens and sum of activations per feature, token count (host memory)."""
    def __init__(self, n_sentences, d):
        self.counts = torch.zeros(n_sentences, d, dtype=torch.int16)
        self.sums = torch.zeros(n_sentences, d, dtype=torch.float32)
        self.ntok = torch.zeros(n_sentences, dtype=torch.long)
        self.eq = None                                       # optional TagStats of the equal-budget stream (decision D-015), set by run_tagging when `keep` is given

    def update(self, feats, row_sent):
        """feats [N, d] features of the VALID rows of a batch, row_sent [N] global sentence index of each row."""
        if feats.shape[0] == 0:
            return
        uniq, inv = torch.unique(row_sent, return_inverse=True)
        d = feats.shape[1]
        c = torch.zeros(len(uniq), d, dtype=torch.int32, device=feats.device).index_add_(0, inv, (feats > 0).to(torch.int32))
        s = torch.zeros(len(uniq), d, dtype=torch.float32, device=feats.device).index_add_(0, inv, feats.float())
        n = torch.bincount(inv, minlength=len(uniq))
        u = uniq.cpu()
        self.counts[u] += c.cpu().to(torch.int16); self.sums[u] += s.cpu(); self.ntok[u] += n.cpu()



def chunked_nll(logits, tokens, valid, chunk=256, cast_float=True):
    """Per-row sum (float64) of -log p(token_t | tokens_<t) over positions t >= 1 whose position is valid, and the number of such positions.
    cast_float=True: float32 log-softmax per row chunk (accurate; a 256k-vocabulary logits tensor is never converted whole).
    cast_float=False: log-softmax in the logits' own dtype, which is what TransformerLens' lm_cross_entropy_loss does and therefore what their published perplexity is."""
    sums, counts = [], []
    for i in range(0, logits.shape[0], chunk):
        x = logits[i:i + chunk, :-1]
        lp = torch.log_softmax(x.float() if cast_float else x, dim=-1).gather(-1, tokens[i:i + chunk, 1:, None])[..., 0].double()
        m = valid[i:i + chunk, 1:]
        sums.append(-(lp * m).sum(1)); counts.append(m.sum(1))
    return torch.cat(sums), torch.cat(counts)


class SparseRows:
    """JumpReLU features are about 0.5% non-zero: keep the full-precision features of every batch sparsely on the host."""
    def __init__(self):
        self._b = []

    def add(self, dense):
        r, c = dense.nonzero(as_tuple=True)
        self._b.append((tuple(dense.shape), dense.dtype, r.to(torch.int32).cpu(), c.to(torch.int32).cpu(), dense[r, c].cpu()))

    def get(self, i, device):
        shape, dt, r, c, v = self._b[i]
        out = torch.zeros(shape, dtype=dt, device=device)
        out[r.to(device).long(), c.to(device).long()] = v.to(device)
        return out

# ------------------------------------------------------------------ S2: selectivity (proposal section 4.1) and controls
import numpy as np


def language_totals(x, lang, n_lang, weights=None, chunk=4096):
    """Sum the rows of x [S, d] per language -> [n_lang, d] float64 (optional 0/1 row weights; chunked to bound memory)."""
    out = torch.zeros(n_lang, x.shape[1], dtype=torch.float64)
    for i in range(0, x.shape[0], chunk):
        rows = x[i:i + chunk].double()
        if weights is not None:
            rows = rows * weights[i:i + chunk].double()[:, None]
        out.index_add_(0, lang[i:i + chunk], rows)
    return out


def tokens_per_language(ntok, lang, n_lang, weights=None):
    n = ntok.double() if weights is None else ntok.double() * weights.double()
    return torch.zeros(n_lang, dtype=torch.float64).index_add_(0, lang, n)


def firing_rates(counts, ntok, lang, n_lang):
    """r_l(f): mean firing rate of feature f on language l (positive tokens over tokens)."""
    return language_totals(counts, lang, n_lang) / tokens_per_language(ntok, lang, n_lang)[:, None]


def mean_activations(sums, ntok, lang, n_lang):
    """mu_l(f): mean activation (zeros included) on language l, the input of Deng et al.'s nu."""
    return language_totals(sums, lang, n_lang) / tokens_per_language(ntok, lang, n_lang)[:, None]


def selectivity_S(rates):
    """Proposal section 4.1: S(f) = max_l r_l(f) / mean_l r_l(f); 0 for features that never fire."""
    mean = rates.mean(0)
    return torch.where(mean > 0, rates.max(0).values / mean.clamp_min(1e-300), torch.zeros_like(mean))


def peak_language(rates):
    return rates.argmax(0)


def supported_mask(rates, totals, min_panel_rate=1e-5, min_pos=50):
    """Support rule of the proposal: panel mean firing rate above a numerical floor AND at least `min_pos` positive activations at the peak language
    (`totals` = language_totals of the positive-token counts). The floor value is not given in the proposal (provisional, decision D-003)."""
    peak = rates.argmax(0)
    pos = totals[peak, torch.arange(rates.shape[1])]
    return (rates.mean(0) >= min_panel_rate) & (pos >= min_pos)


def bootstrap_peak_stability(counts, ntok, lang, n_lang, peak, n_boot=100, frac=0.9, seed=0):
    """Number of draws (out of n_boot) in which the peak language is recovered after resampling `frac` of the sentences of every language."""
    g = torch.Generator().manual_seed(seed)
    S, stable = counts.shape[0], torch.zeros(counts.shape[1], dtype=torch.long)
    members = [(lang == l).nonzero().squeeze(1) for l in range(n_lang)]
    for _ in range(n_boot):
        keep = torch.zeros(S)
        for idx in members:
            k = int(round(frac * len(idx)))
            keep[idx[torch.randperm(len(idx), generator=g)[:k]]] = 1.0
        rates = language_totals(counts, lang, n_lang, weights=keep) / tokens_per_language(ntok, lang, n_lang, weights=keep)[:, None]
        stable += (rates.argmax(0) == peak).long()
    return stable


def shuffled_null_cutoff(counts, ntok, lang, n_lang, supported, n_perm=20, q=0.95, seed=0):
    """q-quantile of S(f) over supported features after shuffling the sentence-language assignment (the proposal's null for the binary split)."""
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(n_perm):
        perm = lang[torch.randperm(len(lang), generator=g)]
        S = selectivity_S(language_totals(counts, perm, n_lang) / tokens_per_language(ntok, perm, n_lang)[:, None])
        vals.append(S[supported])
    return float(np.quantile(torch.cat(vals).numpy(), q))


def deng_nu(means):
    """Deng et al. (2025) Eq. 3: nu_l(f) = mu_l(f) - mean over the other languages of mu_l'(f); means [n_lang, d]."""
    n = means.shape[0]
    return means - (means.sum(0, keepdim=True) - means) / (n - 1)


def strata_ids(values, n=5):
    """Equal-count quantile strata (0..n-1) of a 1-D tensor (ties broken by order)."""
    ranks = values.argsort().argsort()
    return (ranks * n) // len(values)


def matched_sample(strata, selective, pool, seed=0):
    """For every stratum draw, without replacement, as many features from `pool` as `selective` has in that stratum.
    Returns (sorted indices, list of strata where the pool was too small and all candidates were taken)."""
    g = torch.Generator().manual_seed(seed)
    chosen, short = [], []
    for s in strata.unique().tolist():
        need = int(((strata == s) & selective).sum())
        cand = ((strata == s) & pool).nonzero().squeeze(1)
        if len(cand) < need:
            short.append(s); chosen.append(cand)
        else:
            chosen.append(cand[torch.randperm(len(cand), generator=g)[:need]])
    return torch.cat(chosen).sort().values, short
