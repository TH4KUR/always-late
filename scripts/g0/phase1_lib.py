"""Phase 1 library (labelled: new code built ON TOP of Duan et al.'s setup; their functions are reused verbatim by phase1_run.py).
Pure logic, tested on CPU in scripts/g0/tests_torch. Decisions: reports/decisions_and_anomalies.md (D-001, D-003, D-004, D-010).

S1: batching with BOS regimes, per-language streaming Pearson (their float64 sufficient statistics), sentence-level tagging statistics."""
import torch

REGIMES = ("mask", "none")     # mask: BOS prepended, position 0 excluded from every statistic (Gemma Scope's own protocol); none: Duan's regime


def make_batches(data, bos_id, regime, pad_id=0, max_len=256, batch_size=32):
    """data: {lang_index: [token-id list per sentence, no special tokens]} -> list of batches, each single-language and length-sorted.
    tokens [B, L] right-padded (a causal model cannot be influenced by later padding); valid [B, L] marks real content positions only
    (never BOS under `mask`, never padding); sent = sentence indices within the language; content longer than max_len is truncated."""
    if regime not in REGIMES:
        raise ValueError(f"unknown BOS regime {regime!r}; choose from {REGIMES}")
    off, batches = (1 if regime == "mask" else 0), []
    for lang in sorted(data):
        order = sorted(range(len(data[lang])), key=lambda i: (len(data[lang][i]), i))
        for k in range(0, len(order), batch_size):
            idx = order[k:k + batch_size]
            seqs, trunc = [], 0
            for i in idx:
                s = data[lang][i]
                if len(s) > max_len:
                    s, trunc = s[:max_len], trunc + 1
                seqs.append(s)
            L = max(len(s) for s in seqs) + off
            tokens = torch.full((len(seqs), L), pad_id, dtype=torch.long)
            valid = torch.zeros((len(seqs), L), dtype=torch.bool)
            for r, s in enumerate(seqs):
                if off:
                    tokens[r, 0] = bos_id
                if s:
                    tokens[r, off:off + len(s)] = torch.tensor(s, dtype=torch.long)
                    valid[r, off:off + len(s)] = True
            batches.append({"lang": lang, "tokens": tokens, "valid": valid, "sent": torch.tensor(idx, dtype=torch.long), "n_truncated": trunc})
    return batches


class LangPearson:
    """Their streaming_feature_metrics math (float64 running sums, corr clamped to [-1, 1], denominator clamped at sqrt(1e-12)), one set of sums per language."""
    def __init__(self, n_lang, d, device):
        z = lambda: torch.zeros(n_lang, d, dtype=torch.float64, device=device)
        self.sx, self.sy, self.sx2, self.sy2, self.sxy, self.fire, self.maxx = z(), z(), z(), z(), z(), z(), z()
        self.n, self.d, self.n_lang = [0] * n_lang, d, n_lang

    def update(self, lang, x, y):
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

    def update(self, feats, row_sent):
        """feats [N, d] features of the VALID rows of a batch, row_sent [N] global sentence index of each row."""
        uniq, inv = torch.unique(row_sent, return_inverse=True)
        d = feats.shape[1]
        c = torch.zeros(len(uniq), d, dtype=torch.int32, device=feats.device).index_add_(0, inv, (feats > 0).to(torch.int32))
        s = torch.zeros(len(uniq), d, dtype=torch.float32, device=feats.device).index_add_(0, inv, feats.float())
        n = torch.bincount(inv, minlength=len(uniq))
        u = uniq.cpu()
        self.counts[u] += c.cpu().to(torch.int16); self.sums[u] += s.cpu(); self.ntok[u] += n.cpu()
