"""D-035 independent check: re-derives the headline numbers (established count and lost counts per condition, equal-budget stream) from the raw aggregate profile and the original selectivity.pt with a
separate, deliberately plain implementation that imports none of the phase1_association_* modules. Run on Ada (paths below) after a full run: python env/bin/python check_association_headline_independent.py.
Result of 2026-10-21: established 6,663 of 8,086; lost 124 / 168 / 356 / 726 / 1,945 at INT8 / INT7 / INT6 / INT5 / INT4, 56 at INT16 and INT12, 0 at INT0: identical to the analysis report."""
import torch
run = "/home2/karma/duan-verify/out/phase1/association-full"; base = "/home2/karma/duan-verify/out/phase1/p1-mask/eq"
prof = torch.load(run + "/association/eq/profiles.pt", weights_only=True); sel = torch.load(base + "/selectivity.pt", weights_only=True)
assert list(prof["languages"]) == list(sel["languages"])
n = prof["conditions"]["FP"]["n_tokens"]; assert int(n.min()) == int(n.max())          # equal budget: counts are proportional to rates
peak = sel["peak"].long(); selective = sel["selective"].bool(); F = peak.shape[0]
def status(counts):
    """returns bool [F]: the original language is the strict unique maximum of the counts, has >= 50 positives, and the panel mean rate is >= 1e-5"""
    c = counts.long(); target = c[peak, torch.arange(F)]
    others = c.clone(); others[peak, torch.arange(F)] = -1
    strict = target > others.max(0).values
    rate_panel = (c.double() / n.double()[:, None]).mean(0)
    return strict & (target >= 50) & (rate_panel >= 1e-5)
fp = status(prof["conditions"]["FP"]["positive_counts"]) & selective
print("established:", int(fp.sum()), "of", int(selective.sum()))
for cond in ["INT8", "INT7", "INT6", "INT5", "INT4", "INT16", "INT12", "INT0"]:
    q = status(prof["conditions"][cond]["positive_counts"])
    lost = int((fp & ~q).sum())
    print(f"{cond}: lost {lost} ({lost / int(selective.sum()):.4f} of cohort), retained among established {int((fp & q).sum()) / int(fp.sum()):.4f}")
