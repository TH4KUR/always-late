# Language-level survival by resource tier, regime `mask`, stream `all` (decision D-022)

This analysis was pre-specified (handoff 004, written when only the header of the per-language files had been read). Question: does the survival of ALL active features of a language depend on the language's resource tier (proposal RQ2; paper RQ2 row), and is it explained by character fertility (Srivastava 2026, arXiv:2607.24276: tax multiplier = tokens per character relative to English, FLORES-200 parallel text)? Outcome per language: the fraction of its active features with r > 0.9 (unweighted over 27 languages). Statistic: slope on the ordinal tier (low 0, mid 1, high 2), one-sided (higher tier, higher survival), Freedman-Lane permutation (10000 draws, seed 0) with script-family dummies as nuisance; Holm over the two confirmatory widths INT7 and INT6; INT5 descriptive; the same test with the fertility multiplier added as a nuisance covariate. **Reading rule fixed beforehand:** survival depends on tier if the INT6 slope is positive with Holm < 0.05; if so, a fertility-adjusted p >= 0.05 (strict) means the effect is explained by tokenization cost, otherwise it is beyond fertility; if the first condition fails nothing is claimed and no equivalence is asserted.

| width | mean survival low / mid / high tier | slope (p, Holm) | High - Low | slope with fertility (p) | Joshi-class slope (p) | median-r slope (p) |
|---|---|---|---|---|---|---|
| INT7 | 97.4 / 99.7 / 99.9 % | +0.0128 (p 0.0002, Holm 0.0002) | +2.4 pp | +0.0050 (p 0.0743) | +0.0066 (p 0.0021) | +0.0089 (p 0.0001) |
| INT6 | 68.9 / 90.9 / 95.2 % | +0.1262 (p 0.0001, Holm 0.0002) | +26.4 pp | +0.0614 (p 0.0041) | +0.0698 (p 0.0001) | +0.0199 (p 0.0001) |
| INT5 | 26.8 / 52.4 / 66.2 % | +0.1936 (p 0.0001 (descriptive)) | +39.4 pp | +0.1461 (p 0.0002) | +0.1065 (p 0.0001) | +0.0446 (p 0.0001) |

**Reading (by the rule fixed beforehand): depends on tier; beyond fertility.**

## Activation scale as a covariate (N9, decision D-023)

Does per-language activation scale carry the remaining tier effect? Source: `issues.md` C6 / A-035 (survival follows activation magnitude in Duan et al.'s released data and in ours). Scale of a language = log10 of the median, over its active features (full-precision firing rate in that language above 0.001), of the feature's maximum activation in that language (secondary: mean activation). The same Freedman-Lane test as above with (A) the scale alone and (B) character fertility and the scale together as nuisance covariates. **Reading rule fixed beforehand:** with both covariates the INT6 slope p >= 0.05 (strict, p exactly 0.05 is not significant) -> explained jointly by tokenization cost and activation scale; p < 0.05 -> a tier effect remains beyond fertility and activation scale; the scale alone with p >= 0.05 -> activation scale alone explains it.

Spearman(tier, scale) across languages: +0.815.

| width | slope, no covariate | slope with scale (p) | slope with fertility and scale (p) | Spearman(scale, survival) | secondary (mean activation): with scale (p), with both (p) |
|---|---|---|---|---|---|
| INT7 | +0.0128 (p 0.0002) | -0.0065 (p 0.976) | -0.0065 (p 0.984) | +0.976 | -0.0022 (p 0.624), -0.0026 (p 0.675) |
| INT6 | +0.1262 (p 0.0001) | -0.0111 (p 0.687) | -0.0111 (p 0.765) | +0.972 | -0.0171 (p 0.664), -0.0206 (p 0.785) |
| INT5 | +0.1936 (p 0.0001) | +0.0549 (p 0.0092) | +0.0549 (p 0.0087) | +0.966 | +0.0103 (p 0.319), +0.0088 (p 0.303) |

**Reading (by the rule fixed beforehand): explained jointly by tokenization cost and activation scale; activation scale alone explains it: True.**
