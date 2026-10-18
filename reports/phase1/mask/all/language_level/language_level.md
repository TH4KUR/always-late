# Language-level survival by resource tier, regime `mask`, stream `all` (decision D-022)

This analysis was pre-specified (handoff 004, written when only the header of the per-language files had been read). Question: does the survival of ALL active features of a language depend on the language's resource tier (proposal RQ2; paper RQ2 row), and is it explained by character fertility (Srivastava 2026, arXiv:2607.24276: tax multiplier = tokens per character relative to English, FLORES-200 parallel text)? Outcome per language: the fraction of its active features with r > 0.9 (unweighted over 27 languages). Statistic: slope on the ordinal tier (low 0, mid 1, high 2), one-sided (higher tier, higher survival), Freedman-Lane permutation (10000 draws, seed 0) with script-family dummies as nuisance; Holm over the two confirmatory widths INT7 and INT6; INT5 descriptive; the same test with the fertility multiplier added as a nuisance covariate. **Reading rule fixed beforehand:** survival depends on tier if the INT6 slope is positive with Holm < 0.05; if so, a fertility-adjusted p >= 0.05 (strict) means the effect is explained by tokenization cost, otherwise it is beyond fertility; if the first condition fails nothing is claimed and no equivalence is asserted.

| width | mean survival low / mid / high tier | slope (p, Holm) | High - Low | slope with fertility (p) | Joshi-class slope (p) | median-r slope (p) |
|---|---|---|---|---|---|---|
| INT7 | 97.4 / 99.7 / 99.9 % | +0.0128 (p 0.0002, Holm 0.0002) | +2.4 pp | +0.0050 (p 0.0743) | +0.0066 (p 0.0021) | +0.0089 (p 0.0001) |
| INT6 | 68.9 / 90.9 / 95.2 % | +0.1262 (p 0.0001, Holm 0.0002) | +26.4 pp | +0.0614 (p 0.0041) | +0.0698 (p 0.0001) | +0.0199 (p 0.0001) |
| INT5 | 26.8 / 52.4 / 66.2 % | +0.1936 (p 0.0001 (descriptive)) | +39.4 pp | +0.1461 (p 0.0002) | +0.1065 (p 0.0001) | +0.0446 (p 0.0001) |

**Reading (by the rule fixed beforehand): depends on tier; beyond fertility.**
