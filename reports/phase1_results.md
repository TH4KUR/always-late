# Phase 1 results: do language-selective SAE features, and low-resource languages, lose more under RTN quantization?

**Interpretation update, 2026-10-21 (Q13-Q15, D-032/D-033/D-034):** preserve the measurements below, but treat the magnitude-adjusted/unadjusted contrasts as associations, not identified direct/total effects. The tier-gap test and the overall language-level tests address distinct questions; their coexistence is not a contradiction. Attenuation after magnitude adjustment does not prove mediation, and a p-value above 0.05 does not identify an upstream cause. The user's confidence/diffuse-activity explanation is an investigation hypothesis. [Verified literature, source limits and exact saved-result synthesis](research_sources.md). Older causal or “explained” wording below records the earlier reading rules and is superseded by this interpretation; no numerical output is changed.

**Measurement interpretation update, 2026-10-21 (Q3/Q4, D-030/D-031):** the user authorized reassessing survival detection and confirmed including INT5/INT4. The existing correlation-threshold results concern a cohort identified as language-selective before quantization; they do not establish preserved language identity, preserved causal function or erasure of a mechanism. The new [source-grounded method review](survival_method_reassessment.md) documents the code limitation, accepted literature and proposed validation. Historical measurements, thresholds and statistical labels below are preserved; a replacement method has not been selected.

**Scope update, 2026-10-21 (Q1, D-028):** `mask` is the sole ongoing analysis regime. The completed `none` results in this report are retained as historical evidence supporting that choice; older references to them as a continuing sensitivity are superseded. No further `none` work or numerical comparisons with Duan are planned. [Closed audit and direct result links](audit_our_vs_duan.md#current-disposition----closed-2026-10-21-q1-d-028). Saved measurements and the dated analyses below are unchanged.

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

**Update 2026-10-18 (N2, section 7): the null below holds at the pre-registered cutoff but depends on the threshold: with stricter S thresholds selective features survive significantly less than matched controls (post hoc, not yet controlled for activation magnitude). Read sections 7 to 12 before quoting item 1 (section 12: the JumpReLU threshold margin adds nothing beyond raw magnitude, the thresholds being almost constant; section 11: activation scale carries the whole tier effect at INT7 and INT6; sections 9 and 10: survival depends strongly on the resource tier, 69 / 91 / 95 % of a language's active features at INT6 for low / mid / high tier, for selective and universal features alike, which the selective-minus-control tier test cannot see): section 8 shows that activation magnitude accounts for that gap (matched on magnitude the contrast is not negative).**

1. **In the primary regime none of the three pre-registered confirmatory tests rejects its null after Holm correction.** The INT6 contrast is nominally significant (p 0.043) in the direction "selective features survive slightly less" and is not after correction (0.128); it is at most about 1.4 pp (standardised) and the one-sided tier test points the other way: the selective features of low-resource languages are, if anything, slightly more robust than their matched controls (per-language gaps in `reports/phase1/mask/eq/analysis.md`: the median-r gap is positive for 9 of 9 low-resource, 7 of 9 mid-resource (not tur, pol) and 7 of 9 high-resource languages (not eng, jpn); the gaps are +0.001 to +0.020, largest for the low-resource languages).
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
- Code at HEAD (after N1): the torch and stdlib suites pass (counts in the newest handoff; 122 and 119 when the analysis ran); the key behaviours were mutation-checked when the code was committed (30 of 30 analysis mutants killed, commit `af3af10`; not repeated today).

## 4. Anomalies and open items

A-040 (threshold margin adds nothing beyond raw magnitude), A-039 (activation scale carries the tier effect), A-038 (language-level tier effect), A-037 (tier deficit in survival, not selectivity-specific; two RQ2 analyses disagree), A-036 (thread-count caveat), A-035 (magnitude explains the N2 gap; confounder or mediator), A-034 (threshold dependence of the primary-stream null), A-033 (TOST edge case), A-032 (behavioural link: rank association, selective damage not separable from overall damage), A-029 (regime none breaks the model on short sentences), A-030 (sign change between regimes, unexplained), A-031 (INT7 ceiling, small noise-floor asymmetry, rarity half of the raw gap), A-027 (binary split makes most features selective), A-026 (the first launch ignored equal budgets, fixed before any result), A-024 (their bf16 loss quantisation; production loss is the exact float64 sum), A-022 / A-020 (script families and tier boundaries). Open: V-10 (registration frozen?); team decisions on provisional D-entries; threshold sensitivity of the binary split; a test of A-030 hypothesis (a) by running regime none with longer concatenated windows (not planned, would be a labelled diagnostic); optional R3 and their script 06.

## 6. Behavioural link (N1, secondary; decision D-018, anomaly A-032, source check V-11)

The team's evaluation-plan table (`paper/main.tex`, row "Behavioral Link") asks whether the selective-feature damage fraction predicts the next-token loss increase per language, with Spearman rho and a multivariable OLS (covariates: overall feature damage and token fertility). Definitions, all taken from sources (D-018): damage = 1 - survival at t_s = 0.9 over the selective features peaking in the language (the proposal's own usage in G0 (b)), 'damaged' (r < 0.5, Duan et al.) as sensitivity; loss increase = per-token mean loss of the quantized minus the full-precision model on the same tokens; fertility = Srivastava's (2026, arXiv:2607.24276) tax multiplier relative to English on parallel FLORES-200 text, in character fertility (their own fallback; five of our scripts have no word spaces; V-11). One observation per language (27). Secondary analysis, outside the Holm family, p-values uncorrected. Full tables: `reports/phase1/<regime>/<stream>/behavior.md`.

Primary stream (BOS masked, equal budget), damage = 1 - survival:

| width | Spearman rho | p (one-sided) | OLS slope of selective damage (p) | OLS slope of overall damage (p) | fertility (p) | VIF selective / overall | R2 |
|---|---|---|---|---|---|---|---|
| INT7 | +0.539 | 0.0019 | -0.052 (0.55) | +0.109 (0.050) | -0.00004 (0.93) | 5.5 / 5.6 | 0.36 |
| INT6 | +0.458 | 0.0082 | +0.056 (0.31) | +0.012 (0.78) | -0.0008 (0.71) | 7.0 / 9.5 | 0.32 |
| INT5 (descriptive) | +0.692 | 3.2e-5 | +0.111 (0.015) | +0.113 (0.024) | +0.004 (0.46) | 2.3 / 3.4 | 0.76 |

With the stricter 'damaged' definition (r < 0.5): INT7 is undefined (no selective feature is damaged in any language, so the fraction is constant), INT6 rho +0.172 (p 0.20), INT5 rho +0.631 (p 0.0002) with the overall-damage slope carrying it (p 0.0002 against 0.39 for selective damage). The sensitivity stream (all tokens) agrees (INT7 +0.537, INT6 +0.455, INT5 +0.703). The BOS-free regime has the same sign at INT7 and INT6 (rho +0.439 and +0.531, one-sided p 0.011 and 0.002) and none at INT5 (+0.066); its instrument is not valid (A-029).

**Reading.** Languages whose selective features are damaged more also lose more next-token accuracy (a rank association across 27 languages), but the regression cannot attribute this to selective features: selective and overall damage are strongly collinear (VIF up to 9.5), the selective-damage slope is not distinguishable from zero at INT7 and INT6, fertility is never significant, and the association disappears at INT6 under the stricter damage definition. In the table's own terms this is the "partial" branch (dissociation between features and loss), at best weak support for "positive". A shared language-level driver (how well the model handles the language) would produce the same pattern; a causal ablation (proposal Phase 2, an extension, Q8) is the test that could separate them and has not been run. Limits: n = 27, five defined cells in the primary stream are uncorrected (a post-hoc Bonferroni threshold over 5 is 0.01: four pass, INT6 'damaged' does not), INT7 loss increases are tiny (+0.4% perplexity).

## 7. Fixed-threshold sensitivity of the selective split (N2; D-019, A-033, A-034)

Proposal 4.1 asks to report how the binary "selective" split depends on its threshold. The runs saved only the 95th-percentile cutoff of the shuffled-label null (S > 1.83), not the null samples, so other percentiles cannot be recomputed; fixed S thresholds 2, 3 and 5 are reported instead (**a labelled deviation, and the three values are my choice**, D-019; the faithful alternative is a tagging-only rerun that saves the null, Q6 (a)). The rule is the run's own (supported, peak stable in at least 80 of 100 bootstrap draws, strict S > tau), controls come from the run's own `matched_sample`, and the estimators are the unchanged D-014 / D-016 analysis with the original settings. Validation: the row at the run's own cutoff is identical to the committed analysis on all four streams, and the primary-stream `analysis.json` is byte-identical after the one code fix this step needed (A-033). Tables: `reports/phase1/<regime>/<stream>/thresholds/thresholds.md`.

Primary stream (BOS masked, equal budget); effects are selective minus matched-universal survival, stratum-standardised; Holm over the three tests of each row:

| threshold | selective | controls | T1 INT7 (95% CI), p, Holm | T2 INT6 (95% CI), p, Holm | INT6 survival selective / control | T3 tier slope (p) |
|---|---|---|---|---|---|---|
| S > 1.83 (pre-registered, the run's cutoff) | 8,086 | 2,860 | -0.28 pp, 0.257, 0.514 | -1.44 pp, 0.043, 0.128 | 91.5 % / 94.2 % | -0.0048 (0.99) |
| S > 2 | 7,470 | 3,476 | -0.27 pp, 0.268, 0.537 | -1.02 pp, 0.157, 0.472 | 91.3 % / 94.0 % | -0.0041 (0.99) |
| S > 3 | 3,922 | 3,148 | -0.42 pp [-0.88, +0.07], 0.129, 0.259 | **-2.14 pp** [-3.52, -0.74], 0.0055, **0.0165** | 88.9 % / 92.3 % | +0.0020 (0.14) |
| S > 5 | 1,677 | 1,677 | **-0.93 pp** [-1.74, -0.17], 0.019, **0.038** | **-7.16 pp** [-9.14, -5.20], 0.0001, **0.0003** | 84.1 % / 91.3 % | +0.0033 (0.13) |

The all-token stream agrees in sign and is larger (S > 5: -1.74 pp at INT7, -8.24 pp at INT6, both Holm 0.0003). In the BOS-free regime the positive gap of A-030 also grows with the threshold (equal budget: +9.2 pp at the cutoff, +10.9 at S > 2, +17.3 at S > 3, +19.2 at S > 5 at INT7; INT6 +7.7 to +17.5 pp), all Holm at most 0.001.

**Reading.**
1. The primary-stream null at the pre-registered cutoff is partly a dilution effect: that cutoff labels 74 % of supported features selective (A-027). The more selective the features, the larger the gap, monotonically (INT6: -1.0, -2.1, -7.2 pp), and from S > 3 it survives Holm correction. The direction is that language-selective features survive *less* than firing-rate-matched controls, the direction the hypothesis predicts, not the reverse.
2. The language-level tier test never rejects at any threshold (primary stream p 0.13 to 0.99; all four streams p 0.10 to 0.99): the slope changes sign (negative at the cutoff and S > 2, positive at S > 3 and S > 5) but is never significant. There is no support for the resource-tier hypothesis (RQ2) at any threshold.
3. **What this does not show.** (a) The thresholds are post hoc and mine; Holm covers the three tests within a row, not the choice among thresholds (12 further families over four streams), so the S > 3 INT6 result in particular is not a confirmatory finding. (b) The controls are matched on firing-rate strata only. Duan's own released per-feature data show survival tracks maximum activation (Spearman 0.78) and not firing rate (-0.085) (`issues.md` C6), and highly language-selective features may differ in activation magnitude; this is **not yet controlled** and is the next step (N6). (c) The exploratory mixed-effects model in `analysis.md` has a positive S~ coefficient (+1.19: higher selectivity rank, higher survival within strata), the opposite direction from this dose-response; the populations and estimands differ (all supported features with a random intercept versus the matched contrast) and the two are not reconciled (A-034). (d) The BOS-free regime reverses the sign at every threshold in an instrument that fails its noise-floor check (A-029, A-030).

## 8. Activation-magnitude check of the N2 dose-response (N6; D-020, A-035)

N2 found that selective features survive less than firing-rate-matched controls and that the gap grows with the S threshold (section 7). The controls are matched on firing rate only, while Duan's released data show survival follows maximum activation (`issues.md` C6). N6 asks whether magnitude explains the gap. It was pre-specified before any magnitude number was looked at (handoff 002, D-020), including the reading rule, with one amendment made before any real number (matching at 3, 5 and 10 bins, because a synthetic planted magnitude-only effect of -0.52 left -0.08 after terciles, -0.03 after 5 bins and -0.007 after 10 bins). No new estimator: the D-014 / D-016 analysis with firing-rate stratum x magnitude-bin strata. m = maximum activation of the feature at its peak language (full-precision stream). Files: `reports/phase1/<regime>/<stream>/magnitude/magnitude.md`.

**(a) Threshold-free dose-response, primary stream (mask, equal budget), inside firing-rate strata.** Spearman(S, survival) is -0.017 (INT7, p 0.12), -0.028 (INT6, p 0.004) and +0.019 (INT5, p 0.047): across the whole range of S, selectivity is essentially unrelated to survival. Stratum-standardised INT6 survival by decile of S is flat at 0.93 to 0.95 for deciles 1 to 8 and falls only in the top two (0.920, 0.902; raw 0.806 in the top decile, mean S 11.6): the N2 gap lives in the extreme tail of S. Magnitude, by contrast, predicts survival (Spearman(m, survival) +0.09 at INT7, +0.35 at INT6, +0.54 at INT5, the same direction as Duan's data, where Spearman(r, maximum activation) is 0.78 at INT6), and the more selective a feature, the lower its magnitude (Spearman(S, m) -0.18 within strata).

**(b) Selective features have lower activations than their controls** (median maximum activation, selective / control): 13.9 / 16.0 at the cutoff, 12.1 / 14.8 at S > 3, 9.8 / 14.2 at S > 5; standardised difference of log10 m -0.048, -0.065, -0.119 (all p 0.0001).

**(c) INT6 contrast with magnitude-matched controls** (primary stream; effects are selective minus control survival, stratum-standardised; Holm over the three tests of each row; all selective features kept in every cell):

| threshold | firing rate only (N2) | + 3 magnitude bins | + 5 bins | + 10 bins | pre-specified reading (10 bins) |
|---|---|---|---|---|---|
| pre-registered cutoff | -1.44 pp (Holm 0.128) | +1.35 pp (0.242) | +2.69 pp (0.0039) | +3.12 pp (0.0003) | reversed |
| S > 3 | -2.14 pp (0.0165) | +1.79 pp (0.111) | +3.90 pp (0.0003) | +4.26 pp (0.0003) | reversed |
| S > 5 | **-7.16 pp (0.0003)** | +0.80 pp (1.0) | +4.13 pp (0.0048) | +4.58 pp (0.0012) | reversed |

At INT7 the matched contrast is within +-0.7 pp and never significant; the tier test (T3) stays null everywhere (p 0.10 to 1.0 over the four streams). The all-token stream agrees (S > 5: +3.5, +5.1, +5.6 pp at 3, 5, 10 bins, Holm 0.03 to 0.001; at the cutoff +1.1 to +2.1 pp, not significant). In the BOS-free regime the positive gap of A-030 is halved by magnitude matching but not removed (equal budget at S > 5: +17.5 pp rate-only, +9.1 / +9.2 / +9.3 pp matched), so magnitude does not explain that sign (A-029 still applies).

**Reading (by the rule fixed beforehand).** The selectivity-specific reading (a) is **not supported**: with magnitude matched the INT6 gap is not negative, it is modestly positive, +3 to +5 pp at 5 and 10 bins and not significant at terciles (the synthetic calibration says terciles under-correct, and the residual they leave is in the direction of the N2 gap). In the rule's terms the headline is "reversed" at every threshold, while the resolutions disagree (terciles say "explained by magnitude"). So the N2 gap is accounted for by activation magnitude, and nothing in the primary regime indicates that language-selective features are more fragile than universal ones of the same firing rate and magnitude.

**Two things this does not settle (A-035).** (1) *Confounder or mediator.* If low activation is simply a property of language-selective features (S and m are negatively related), then magnitude is a mediator: selective features **do** survive less overall (the N2 total effect, up to -7 pp at S > 5), and they do so **because** their activations are lower, not because they are language-specific. Matching on magnitude estimates the remaining direct effect. Which of the two answers the research question is a design choice for the team (`DECISIONS_FOR_REVIEW.md` Q13); both numbers are reported. (2) *The positive residual* (+3 to +5 pp) is unexplained; untested ideas are that maximum activation is measured at the peak language where selective features are by construction most active while a control's peak language is arbitrary, or that maximum activation is an imperfect proxy for the margin above the JumpReLU threshold. The exploratory mixed-effects model's positive S~ coefficient (A-034 (c)) is not yet reconciled (N7). All thresholds remain post hoc (D-019).

## 9. Why the exploratory mixed model disagrees with the matched contrasts (N7; D-021, A-036, A-037)

The mixed-effects logistic model of the proposal's section 7 (survival ~ bits + firing-rate stratum + S~ + tier + S~ x tier, random intercept per feature) has a positive S~ coefficient (+1.19 in the primary stream) while the matched contrasts of N2 are negative and the threshold-free within-stratum Spearman of N6 is about zero. The proposal's section 7 names "the interaction of continuous S~(f) with resource tier, via mixed-effects logistic regression" as part of the primary endpoint, while its section 6 defines the Holm family as the INT7 and INT6 contrasts and the language-level tier test; D-014 follows section 6 and reports the mixed model as a labelled exploratory block. That classification is provisional (`DECISIONS_FOR_REVIEW.md` Q14). N7 refits the same model on the same data in variants pre-specified in handoff 003 (D-021); variant (i) reproduces the saved coefficients exactly. Files: `reports/phase1/mask/<stream>/mixed_variants/mixed_variants.md`.

Primary stream (mask, equal budget), 10,905 supported evaluable features unless stated; coefficients on the logit scale (p):

| variant | S~ | tier | S~ x tier | log10 max activation |
|---|---|---|---|---|
| (i) full model, as saved | +1.194 (0.0004) | +1.455 (9e-13) | +0.943 (0.0013) | |
| (ii) without the tier terms | **-0.812 (0.004)** | | | |
| (iii) full model + log10 m | +3.001 (2e-14) | +0.861 (3e-6) | **-0.007 (0.98)** | +16.4 (5e-37) |
| (iv-a) matched sets of N2 at S > 5 (3,327 features) | +0.271 (0.66) | +1.081 (0.009) | +1.694 (0.003) | |
| (iv-b) matched sets of N6, 10 bins (3,181 features) | +2.508 (2e-5) | +1.180 (0.0009) | +1.372 (0.005) | |

**Reading by the rule fixed beforehand: explained by the tier terms.** Without them the S~ coefficient is significantly negative (-0.81), in line with N2 and with the near-zero threshold-free correlation (the population-averaged GEE fit agrees, -0.53); it turns positive only when tier and S~ x tier are in the model. Adding magnitude does not remove the positive coefficient, it enlarges it (+3.0), the same direction as the positive magnitude-matched contrast of N6. The all-token stream agrees on every point (S~ -1.04 without tier, +3.41 with magnitude, S~ x tier +0.06 with magnitude).

**Why the tier terms matter (post hoc, one-off descriptive check, not part of the pre-specified variants; command in the progress log).** Supported evaluable features of the primary stream by tier of their peak language, INT6:

| tier of the peak language | features | mean S~ | survival | median max activation | survival, selective (cutoff set) | survival, matched universal controls |
|---|---|---|---|---|---|---|
| low | 2,932 | 0.68 | 80.2 % | 10.1 | 80.5 % (n 2,629) | 77.9 % (n 303) |
| mid | 2,582 | 0.45 | 96.1 % | 15.7 | 96.3 % (1,766) | 95.7 % (816) |
| high | 5,391 | 0.43 | 96.8 % | 16.0 | 97.0 % (3,667) | 96.3 % (1,724) |

Spearman(tier, S~) = -0.33 and Spearman(tier, survival) = +0.23: the most selective features peak in low-resource languages, and features measured on low-resource-language tokens survive much less, so pooling tiers makes selectivity look harmful (a Simpson-type pattern); within a tier the S~-survival correlation is about zero (-0.08, +0.06, -0.01).

**What this means for RQ2 (post hoc reading, A-037).**
1. **There is a large tier effect on survival**: features peaking in a low-resource language survive about 16 pp less at INT6 than those of mid- and high-resource languages (80 % against 96 to 97 %). It is the largest effect in the data; the tier main effect of the mixed model is significant in every variant (p at most 0.02, as small as 7e-13), is reduced but not removed by magnitude in the primary stream (+1.46 to +0.86) and not at all in the all-token stream (+1.44 to +1.43).
2. **It is not specific to language-selective features**: matched universal controls show the same deficit (77.9 / 95.7 / 96.3 %), which is why the pre-registered tier test (selective minus matched-control gap per language, section 2) is null and equivalent to zero. The two RQ2 analyses of the proposal measure different things: the permutation test asks whether the selectivity penalty depends on tier (no), the mixed-model tier term asks whether survival depends on tier (yes, strongly).
3. **The S~ x tier interaction, the proposal's primary-endpoint term (betaSt), is +0.94 (p 0.001) in the full model and in the matched sets, but -0.007 (p 0.98) once log magnitude is in the model**: it is a magnitude effect (low-tier features have lower activations, 10 against 16).
4. In the proposal's own interpretation table (RQ2: "Positive: low-resource features more vulnerable. Null: tier gap localizes upstream.") the pattern is a tier gap in absolute survival that shows no selectivity-specific component, which is the shape the "localizes upstream" branch describes; whether it is tokenization, script, activation magnitude or the model's treatment of those languages is not tested here (the behavioural link of section 6 found token fertility not significant for the loss increase).

Limits: the two descriptive tables are post hoc; the "peak language" of a universal control is nearly arbitrary, so the tier of a control says little about the feature and a lot about the text it was measured on; the tier effect is entangled with script and language-specific token statistics (the language-level analysis of N8 addresses this); the mixed-model classification (exploratory in D-014, primary-endpoint term in the proposal's section 7) is open (Q14).

## 10. Language-level survival by resource tier (N8; D-022, A-038)

N7 (section 9) found a large tier effect on feature survival that the selective-minus-control gap test cannot see. N8 tests it at the language level, without the selectivity machinery and without the peak-language device. **Status:** an additional RQ2 analysis, motivated by a post hoc observation (N7's descriptive tables) but pre-specified in handoff 004 before it was run and before the per-language values were looked at (only the file header had been read); it is outside the proposal's Holm family (D-014) and its two-width Holm correction is within this analysis only (Q14). Outcome per language and width: the fraction of the language's active features (full-precision firing rate in that language above 0.001) with r > 0.9, unweighted over the 27 languages. Statistic: slope on the ordinal tier (low 0, mid 1, high 2), one-sided (higher tier, higher survival), the existing Freedman-Lane permutation with script-family dummies (10,000 draws, seed 0); Holm over INT7 and INT6; the same test with the character-fertility multiplier (Srivastava 2026, arXiv:2607.24276; N1) as a further covariate. Files: `reports/phase1/<regime>/<stream>/language_level/language_level.md`.

Primary stream (mask, equal budget):

| width | mean survival, low / mid / high tier | slope per tier step (p, Holm) | High minus Low | slope with fertility (p) | Joshi-class slope (p) | median-r slope (p) |
|---|---|---|---|---|---|---|
| INT7 | 96.5 / 99.6 / 99.8 % | +1.65 pp (0.0001, 0.0002) | +3.4 pp | +0.56 pp (0.071) | +0.87 pp (0.0006) | +0.0085 (0.0001) |
| **INT6** | **69.4 / 90.6 / 95.1 %** | **+12.3 pp (0.0001, 0.0002)** | **+25.7 pp** | **+6.3 pp (0.003)** | +6.8 pp (0.0001) | +0.019 (0.0001) |
| INT5 (descriptive) | 27.5 / 52.4 / 66.2 % | +19.0 pp (0.0001) | +38.7 pp | +14.6 pp (0.0002) | +10.4 pp (0.0001) | +0.044 (0.0001) |

(0.0001 is the floor of 10,000 permutations, 1/10,001.) **Reading by the rule fixed beforehand: "depends on tier; beyond fertility".** The all-token stream gives the same (INT6 slope +12.6 pp, with fertility +6.1 pp, p 0.004). In the BOS-free regime the slopes are positive and significant at every width but smaller (equal budget, INT6: 25.5 / 34.5 / 38.6 %, slope +6.7 pp, with fertility +4.0 pp, p 0.0002), and that regime's instrument is not valid (A-029).

**What this says (A-038).**
1. **Feature survival depends strongly on the resource tier of the language**: at INT6 a low-resource language keeps 26 pp fewer of its active features than a high-resource one (69 % against 95 %), robustly to script family, in every stream, with the same sign for the selective-feature view of N7 (80 % against 97 %).
2. **Character fertility accounts for about half of the slope** (12.3 -> 6.3 pp at INT6; at INT7 it removes two thirds of the slope and the remainder is not significant, p 0.071) **but not all**: mid- and high-resource languages have almost the same fertility (mean multiplier 1.7 and 1.8, low-resource 3.1) yet differ in survival (90.6 against 95.1 %).
3. **It is not a property of language-selective features** (A-037): matched universal controls show the same deficit, and the pre-registered gap test is null. In the proposal's own table (RQ2: "Positive: low-resource features more vulnerable. Null: tier gap localizes upstream.") the data show the positive pattern for survival in general and no selectivity-specific component.

**Limits.** Correlational with 27 languages and three tiers; tier is entangled with script, corpus size, tokenization and the model's competence in the language (the script-family covariate absorbs only the first, fertility only part of the third); fertility and tier are strongly collinear (low tier 3.1 against 1.7 to 1.8), so "beyond fertility" means p < 0.05 given fertility, not that tokenization is irrelevant; the outcome is over the features active in each language, which differ between languages; the analysis is additional and post hoc-motivated, not part of the proposal's confirmatory family (Q14); the BOS-free regime is shown only as a sensitivity.

## 11. Activation scale carries the language-level tier effect (N9; D-023, A-039)

N8 (section 10) left about half of the language-level tier slope unexplained by character fertility. N9 asks whether per-language activation scale carries it (pre-specified in handoff 005 before it was run; D-023). Scale of a language = log10 of the median, over its active features (full-precision firing rate in that language above 0.001), of the feature's maximum activation in that language (Duan's own predictor aggregated per language; secondary: mean activation). The same one-sided Freedman-Lane test as N8 with script-family dummies and (A) the scale alone, (B) character fertility and the scale together. Same status as N8: an additional, correlational analysis outside the proposal's Holm family (Q14). Files: `reports/phase1/<regime>/<stream>/language_level/language_level.md` (the N8 part is unchanged, verified on all four streams).

Primary stream (mask, equal budget), slope per tier step in survival (pp), p in brackets (the permutation floor is 0.0001):

| width | no covariate | with fertility (N8) | with scale | with fertility and scale | with mean-activation scale | Spearman(scale, survival) |
|---|---|---|---|---|---|---|
| INT7 | +1.65 (0.0001) | +0.56 (0.071) | **-0.74 (0.977)** | -0.65 (0.971) | -0.43 (0.70) | +0.957 |
| **INT6** | **+12.3 (0.0001)** | +6.3 (0.003) | **-0.81 (0.667)** | **-0.30 (0.577)** | -2.0 (0.71) | **+0.966** |
| INT5 (descriptive) | +19.0 (0.0001) | +14.6 (0.0002) | +5.8 (0.002) | +5.7 (0.005) | +1.0 (0.30) | +0.978 |

Spearman(tier, scale) across the 27 languages is **+0.83**. **Reading by the rule fixed beforehand: "explained jointly by tokenization cost and activation scale; activation scale alone explains it: True"** (the INT6 slope with both covariates has p 0.58; with the scale alone p 0.67). The all-token stream agrees (INT6 with scale -1.1 pp, p 0.69); so does the BOS-free regime (equal budget, INT6 with scale +1.0 pp, p 0.27; there Spearman(scale, survival) is +0.55 at INT7, +0.91 at INT6, +0.99 at INT5, and that regime's instrument is not valid, A-029).

**What this says (A-039).**
1. **A language's activation scale rank-orders its survival almost perfectly** (Spearman +0.96 to +0.98 over 27 languages at all three widths in the primary stream), and low-resource languages have lower scale (+0.83 with tier).
2. **Given the scale, tier adds nothing at the confirmatory widths** (INT7 and INT6 slopes about -0.7 pp, p 0.98 and 0.67); at INT5 a residual of +5.8 pp remains with the maximum-activation scale and vanishes with the mean-activation scale (+1.0 pp, p 0.30).
3. **In the proposal's own interpretation table** (RQ2: "Null: tier gap localizes upstream") the tier gap in survival sits in activation scale, a property of how the SAE features respond to a language's tokens, and not in the erasure of language-selective features (sections 2, 8, 9). It is consistent with Duan et al.'s own finding that survival follows maximum activation (feature level, Spearman 0.78) and with N6 and N7.

**Limits.** The tight link between scale and survival (Spearman about 0.97) is partly mechanical: fixed-size quantization noise matters less for large activations, so "scale explains survival" is close to restating that low-activation features are fragile; the substantive content is that tier adds nothing beyond scale and that low-tier languages have lower scale, which identifies no cause. **Why low-resource languages have lower activation scale is not tested here**: character fertility accounts for about half of the survival slope (section 10), and candidates untested are the residual-stream norm of those tokens (upstream of the SAE), the model's competence in the language, and the SAE's training distribution (`DECISIONS_FOR_REVIEW.md` Q15 proposes a GPU diagnostic). Correlational, 27 languages, scale and survival computed from the same activations; additional to the Holm family (Q14).

## 12. JumpReLU threshold margin adds nothing beyond raw magnitude (N11; D-025, A-040)

The thresholds theta_f of the SAE's 16,384 JumpReLU features (Gemma Scope, Lieberum et al. 2024, arXiv:2408.05147: a feature is active when its pre-activation exceeds a learned threshold) were read once with Duan et al.'s own loader on the Ada login node (no Slurm job; `reports/phase1/sae_thresholds.csv`, sha256 of the vector in its header). **The thresholds are almost constant: min 2.59, 10th to 90th percentile 3.00 to 3.19, median 3.10, max 16.76**, so the margin ratio (max activation / theta) is the raw magnitude divided by a number between 2.6 and 3.2 for 90 % of the features (80 % within 3.00 to 3.19). Pre-specified in handoff 007 before the file existed (D-025); the existing estimators of N6, N7 and N9 were run unchanged on a survival file in which the activations are divided by the thresholds. Files: `reports/phase1/<regime>/<stream>/margin/`.

Primary stream (mask, equal budget):

| analysis | with the raw magnitude (N6, N7, N9) | with the margin |
|---|---|---|
| INT6 contrast at S > 5, 10 bins (Holm) | +4.58 pp (0.0012) | +5.32 pp (0.0006) |
| INT6 contrast at S > 5, 5 bins | +4.13 pp (0.0048) | +3.88 pp (0.0108) |
| INT6 contrast at S > 5, 3 bins | +0.80 pp (1.0) | +0.95 pp (1.0) |
| mixed model, S~ coefficient with the covariate added | +3.001 (2e-14) | +3.002 (1.5e-14) |
| language-level tier slope with the scale, INT7 / INT6 / INT5 (p) | -0.74 (0.977) / -0.81 (0.667) / +5.8 (0.002) pp | -0.72 (0.975) / -0.67 (0.643) / +6.0 (0.002) pp |
| Spearman(scale, survival), INT7 / INT6 / INT5 | +0.957 / +0.966 / +0.978 | +0.957 / +0.966 / +0.978 |

**Readings by the rules fixed beforehand: (i) the positive residual of A-035 *remains* (margin-matched INT6 contrast at S > 5, 10 bins, 100 % of selective features kept: +5.3 pp, Holm 0.0006); (ii) at the language level margin carries *no more than raw activation* (INT5 p 0.0020 with the margin scale against 0.0022 with the max-activation scale).** The same two readings hold in all four streams (equal budget and all tokens, BOS masked and BOS-free; at S > 5 and 10 bins the BOS-free regime gives +9.4 pp against +9.3 pp with raw magnitude).

**What this says (A-040).** The margin above the learned threshold is not a better explanation of survival than the raw magnitude, because the thresholds hardly vary; so the two open residuals, the selective-feature advantage of A-035 (selective features survive 4 to 6 pp more than controls matched on magnitude or margin, at the finest matching) and the INT5 tier residual of N9, are not explained by anything the SAE's own threshold structure offers. **What remains untested:** why maximum activation at the peak language matters for survival the way it does, whether the maximum at a selective feature's peak language (where it is by construction most active) is comparable to that of a control at its arbitrary peak language, and the upstream question of why low-resource languages have lower scale (Q15, a GPU diagnostic).

## 13. Clean-room reproduction of every committed result file (N13; D-026)

Every committed result file under `reports/phase1/` was regenerated from the cached run files (the Ada outputs fetched with `fetch_phase1.sh`) by the committed scripts at commit `0a71b35`, in a fresh directory outside the repository, at `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4`, all four streams in parallel (23 minutes). The comparison rule was fixed before the run (D-026): a regenerated file is reproduced if it equals the committed one byte for byte; JSON is compared after dropping filesystem-path keys, numbers exactly. **Verdict: REPRODUCED, 89 of 89 compared files identical** (23 files for each BOS-masked stream, 21 for each BOS-free stream, which has no mixed-model step, plus the check that the generated summary is up to date); the path-valued `run` keys were even equal, so the ignore rule was not needed. The report is `reports/phase1_reproduction.md` (generated by `scripts/g0/reproduce_phase1.py`; 23 tests, 24 of 24 mutants killed). Independent spot checks (`cmp`, file times inside the run window, two different streams differing) agree.

**What this establishes:** no committed Phase 1 number depends on a past state of the code, and `reports/phase1_summary.md` is reproducible from the Ada outputs with the committed scripts. **What it does not establish:** the Ada runs themselves (S6, 3 h 40 min of GPU per regime) were not re-run (their fidelity rests on the S5 anchor, which reproduced their own smoke run in every digit); the Ada-side extractions (`flores_text_stats.json`, `sae_thresholds.csv`) are inputs with provenance hashes, not regenerated; the check was made on one machine at 4 BLAS threads (A-036: other thread counts move the mixed-model coefficients at about 1e-10).

## 5. Where everything is

- Small outputs per run and stream (summary, per-language, metadata, analysis report and JSON, behavioural link `behavior.md` / `behavior.json`; FLORES text statistics `reports/phase1/flores_text_stats.json`): `reports/phase1/{mask,none}/{eq,all}/`.
- Large outputs (`survival.pt`, `selectivity.pt`, per-feature data): Hugging Face private dataset `karma-skz/always-late-artifacts`, `runs/phase1/` (index in `reports/artifacts.md`).
- Reproduce: `bash scripts/g0/fetch_phase1.sh p1-mask <dest>`; `python3 scripts/g0/phase1_analysis.py --run <dest>/p1-mask/eq --out <out> --seed 0` (torch, numpy, scipy, pandas, statsmodels).
