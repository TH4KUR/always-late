# Phase 1 results: do language-selective SAE features, and low-resource languages, lose more under RTN quantization?

Written 2026-10-18 from the S6 runs (Slurm 25745 `p1-mask`, 25746 `p1-none`, runner commit `618496f`, about 3 h 40 min each on one 2080 Ti) and the S7 analysis (`scripts/g0/phase1_analysis.py`). **Provisional**: every choice behind it is a D-entry in `decisions_and_anomalies.md` that the team may override (D-001, D-003, D-004, D-006, D-010, D-011 adopted provisionally; V-10, whether the OSF registration was frozen, is still open and only the user knows). The paper text is the team's to write; this file only states what was measured.

Setting (fixed by the proposal and Duan et al. 2026): Gemma-2-2B, layer-12 residual stream, Gemma Scope 16k SAE (`layer_12/width_16k/canonical`), RTN per-output-channel quantization with **their functions** (extracted verbatim, hashes in `reports/g0/their_functions.json`), 27 FLORES-200 languages (tagging on `dev`, survival on `devtest`), per-language per-feature Pearson r between full-precision and quantized SAE activations on identical tokens, survived = r > 0.9 and active = full-precision firing rate > 0.001 (their definitions). The pipeline reproduces their own smoke run in every digit (S5 anchor).

**Primary stream: BOS masked, equal token budget (27,036 devtest tokens per language).** Sensitivities: BOS-free input as in their pipeline (`none`), and all tokens instead of equal budgets (`all`). Which regime is primary is decision D-001; the reasons are in `audit_our_vs_duan.md`.

## 1. Pooled survival (their metric, all 27 languages pooled, equal budget)

| width | regime mask: survived > 0.9 | median r | perplexity change | regime none: survived > 0.9 | median r | perplexity change |
|---|---|---|---|---|---|---|
| FP baseline | 100 % | 1.000 | (ppl 19.46) | 100 % | 1.000 | (ppl 48,864) |
| INT16 (noise floor) | 100.00 % | 0.9957 | +0.003 % | 99.98 % | 0.9823 | -0.15 % |
| INT12 (noise floor) | 100.00 % | 0.9957 | +0.035 % | 100.00 % | 0.9823 | -0.26 % |
| INT8 | 100.00 % | 0.9906 | +0.08 % | 97.98 % | 0.9618 | -0.88 % |
| INT7 | 99.97 % | 0.9829 | +0.40 % | 64.18 % | 0.9292 | -10.95 % |
| INT6 | 94.94 % | 0.9629 | +2.67 % | 38.60 % | 0.8502 | +3.87 % |
| INT5 | 59.00 % | 0.9143 | +11.5 % | 15.77 % | 0.6851 | +122.5 % |
| INT4 | 13.75 % | 0.7495 | +103.9 % | 5.93 % | 0.3209 | +197.5 % |
| INT0 (restore only) | 100 % | 1.000 | 0 | 100 % | 1.000 | 0 |

(9,500 active features in regime mask, 4,350 in regime none.) The restore-only control equals full precision exactly. In regime mask the instrument sits at its ceiling at INT16/INT12 (median r 0.9957, above the 0.99 gate); in regime none it does not (0.9823) and the perplexity is 48,864 with INT7 "improving" it by 11 %: that regime is not a usable instrument on short FLORES sentences (**A-029**). The collapse between INT6 and INT4 is the same shape as in their table, at much higher survival at INT7/INT6 than their 81 % / 51 % because their windows have no BOS (audit sections 6.2-6.7); this is a different corpus, so it is a consistency check and not a replication.

## 2. Confirmatory family (Holm over three tests)

Selective features (S(f) above the shuffled-label 95th percentile, supported, stable peak) are measured at their peak language and compared with firing-rate-quintile-matched universal controls (stratum-standardised difference, D-016; permutation within strata and cluster bootstrap). The tier test is the language-level slope of the per-language gap (median r of selective minus controls, mean of INT7 and INT6) on tier (low 0, mid 1, high 2), one-sided in the direction "selective features of low-resource languages are less robust", all 27 languages used. Estimates are differences in survival rate (1.0 = 100 pp).

| stream | T1: INT7 selective - control | Holm p | T2: INT6 selective - control | Holm p | T3: tier slope | p |
|---|---|---|---|---|---|---|
| **mask, equal budget (primary)** | **-0.28 pp** [-0.63, +0.13], p 0.257 | **0.514** | **-1.44 pp** [-2.65, -0.13], p 0.0427 | **0.128** | **-0.0048** (High minus Low -0.0082) | **0.990** |
| mask, all tokens | -0.39 pp [-0.74, +0.02], p 0.165 | 0.330 | -1.93 pp [-3.32, -0.46], p 0.0183 | 0.055 | -0.0035 | 0.930 |
| none, equal budget | +9.16 pp [+5.18, +13.23] | 0.0003 | +7.71 pp [+3.95, +11.37] | 0.001 | +0.0026 (High minus Low +0.0212) | 0.378 |
| none, all tokens | +8.74 pp | 0.0003 | +4.08 pp, p 0.0368 | 0.074 | +0.0080 | 0.236 |

TOST (90 % bootstrap interval against +-0.02 in median r): primary stream, tier slope -0.0075..-0.0016 and High-minus-Low -0.0129..-0.0031, **both inside the bound, i.e. equivalent to zero effect within +-0.02**; regime none, slope inside, High-minus-Low not.

Group sizes in the primary stream: 8,062 selective and 2,843 control features evaluable at their peak language (of 10,946 supported, 8,086 selective, 2,860 controls; null cutoff of S 1.83). Raw survival rates: INT7 99.35 % vs 99.82 %, INT6 91.45 % vs 94.20 % (raw gap -2.75 pp; the stratum-standardised gap -1.44 pp is half of it, the rest is firing-rate rarity, **A-031**).

### What this supports

1. **In the primary regime none of the three pre-registered confirmatory tests rejects its null after Holm correction.** The INT6 contrast is nominally significant (p 0.043) in the direction "selective features survive slightly less" and is not after correction (0.128); it is at most about 1.4 pp (standardised) and the one-sided tier test points the other way: the selective features of low-resource languages are, if anything, slightly more robust than their matched controls (per-language gaps in `reports/phase1/p1-mask/eq/analysis.md`: the median-r gap is positive for 9 of 9 low-resource, 7 of 9 mid-resource (not tur, pol) and 7 of 9 high-resource languages (not eng, jpn); the gaps are +0.001 to +0.020, largest for the low-resource languages).
2. **Any effect is small next to the effect of width**: pooled survival goes from 94.9 % at INT6 to 59 % at INT5 and 13.7 % at INT4, while the selective-versus-control difference stays within a few points at every width (INT5 -2.26 pp, uncorrected p 0.045, exploratory; INT4 +0.42 pp, p 0.63).
3. **The equivalence tests support "no tier effect larger than 0.02 in median r"** in the primary stream (the proposal's own interpretation branch for a null on RQ2).

### What this does not support

- It does not show that language-selective features are *more* fragile: the sign in the primary stream is consistent with that at INT6 only weakly and non-significantly, and **reverses in regime none (A-030, unexplained)**, where the instrument is also broken (A-029).
- INT7 cannot discriminate (ceiling, A-031); the confirmatory evidence rests on INT6.
- "Selective" is a broad class here: 74 % of supported features pass the pre-registered binary rule at 27 languages (A-027), so the contrast is "most features" against a minority of "universal" ones, not "one-language" features against the rest. The continuous S~ is the intended moderator; the exploratory mixed-effects logistic model in `analysis.md` is labelled exploratory and shows survival rising with S~ (+1.19, p 0.0004) and with tier within strata (+1.45), again the opposite of the "more fragile" direction. The threshold sensitivity named in proposal section 4.1 (several percentiles of the null) is **not yet run**.
- One layer, one model, one SAE width, RTN only (the proposal's scope); 27 languages, one tokenizer.

## 3. How the numbers were checked (D-017)

- The analysis ran once per stream with the pre-set defaults (10,000 permutations, 2,000 bootstrap draws, seed 0); a rerun of the primary stream gave a byte-identical `analysis.json`.
- Group survival rates and counts were recomputed from the raw `survival.pt` / `selectivity.pt` with a separate short script (not the analysis code): INT7 0.9935 vs 0.9982, INT6 0.9145 vs 0.9420, INT5 0.6080 vs 0.6423, INT4 0.1601 vs 0.1463, INT16 0.9994 vs 1.0000 (n 8,062 + 2,843), matching the report.
- Code at HEAD: 122 torch and 119 stdlib tests pass; the key behaviours were mutation-checked when the code was committed (30 of 30 analysis mutants killed, commit `af3af10`; not repeated today).

## 4. Anomalies and open items

A-029 (regime none breaks the model on short sentences), A-030 (sign change between regimes, unexplained), A-031 (INT7 ceiling, small noise-floor asymmetry, rarity half of the raw gap), A-027 (binary split makes most features selective), A-026 (the first launch ignored equal budgets, fixed before any result), A-024 (their bf16 loss quantisation; production loss is the exact float64 sum), A-022 / A-020 (script families and tier boundaries). Open: V-10 (registration frozen?); team decisions on provisional D-entries; threshold sensitivity of the binary split; a test of A-030 hypothesis (a) by running regime none with longer concatenated windows (not planned, would be a labelled diagnostic); optional R3 and their script 06.

## 5. Where everything is

- Small outputs per run and stream (summary, per-language, metadata, analysis report and JSON): `reports/phase1/{p1-mask,p1-none}/{eq,all}/`.
- Large outputs (`survival.pt`, `selectivity.pt`, per-feature data): Hugging Face private dataset `karma-skz/always-late-artifacts`, `runs/phase1/` (index in `reports/artifacts.md`).
- Reproduce: `bash scripts/g0/fetch_phase1.sh p1-mask <dest>`; `python3 scripts/g0/phase1_analysis.py --run <dest>/p1-mask/eq --out <out> --seed 0` (torch, numpy, scipy, pandas, statsmodels).
