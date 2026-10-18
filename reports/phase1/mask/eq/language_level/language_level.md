# Language-level survival by resource tier, regime `mask`, stream `eq` (decision D-022)

This analysis was pre-specified (handoff 004, written when only the header of the per-language files had been read). Question: does the survival of ALL active features of a language depend on the language's resource tier (proposal RQ2; paper RQ2 row), and is it explained by character fertility (Srivastava 2026, arXiv:2607.24276: tax multiplier = tokens per character relative to English, FLORES-200 parallel text)? Outcome per language: the fraction of its active features with r > 0.9 (unweighted over 27 languages). Statistic: slope on the ordinal tier (low 0, mid 1, high 2), one-sided (higher tier, higher survival), Freedman-Lane permutation (10000 draws, seed 0) with script-family dummies as nuisance; Holm over the two confirmatory widths INT7 and INT6; INT5 descriptive; the same test with the fertility multiplier added as a nuisance covariate. **Reading rule fixed beforehand:** survival depends on tier if the INT6 slope is positive with Holm < 0.05; if so, a fertility-adjusted p >= 0.05 (strict) means the effect is explained by tokenization cost, otherwise it is beyond fertility; if the first condition fails nothing is claimed and no equivalence is asserted.

| width | mean survival low / mid / high tier | slope (p, Holm) | High - Low | slope with fertility (p) | Joshi-class slope (p) | median-r slope (p) |
|---|---|---|---|---|---|---|
| INT7 | 96.5 / 99.6 / 99.8 % | +0.0165 (p 0.0001, Holm 0.0002) | +3.4 pp | +0.0056 (p 0.0713) | +0.0087 (p 0.0006) | +0.0085 (p 0.0001) |
| INT6 | 69.4 / 90.6 / 95.1 % | +0.1234 (p 0.0001, Holm 0.0002) | +25.7 pp | +0.0626 (p 0.0032) | +0.0682 (p 0.0001) | +0.0193 (p 0.0001) |
| INT5 | 27.5 / 52.4 / 66.2 % | +0.1903 (p 0.0001 (descriptive)) | +38.7 pp | +0.1457 (p 0.0002) | +0.1042 (p 0.0001) | +0.0438 (p 0.0001) |

**Reading (by the rule fixed beforehand): depends on tier; beyond fertility.**

## Activation scale as a covariate (N9, decision D-023)

Does per-language activation scale carry the remaining tier effect? Source: `issues.md` C6 / A-035 (survival follows activation magnitude in Duan et al.'s released data and in ours). Scale of a language = log10 of the median, over its active features (full-precision firing rate in that language above 0.001), of the feature's maximum activation in that language (secondary: mean activation). The same Freedman-Lane test as above with (A) the scale alone and (B) character fertility and the scale together as nuisance covariates. **Reading rule fixed beforehand:** with both covariates the INT6 slope p >= 0.05 (strict, p exactly 0.05 is not significant) -> explained jointly by tokenization cost and activation scale; p < 0.05 -> a tier effect remains beyond fertility and activation scale; the scale alone with p >= 0.05 -> activation scale alone explains it.

Spearman(tier, scale) across languages: +0.833.

| width | slope, no covariate | slope with scale (p) | slope with fertility and scale (p) | Spearman(scale, survival) | secondary (mean activation): with scale (p), with both (p) |
|---|---|---|---|---|---|
| INT7 | +0.0165 (p 0.0001) | -0.0074 (p 0.977) | -0.0065 (p 0.971) | +0.957 | -0.0043 (p 0.699), -0.0048 (p 0.786) |
| INT6 | +0.1234 (p 0.0001) | -0.0081 (p 0.667) | -0.0030 (p 0.577) | +0.966 | -0.0201 (p 0.709), -0.0224 (p 0.823) |
| INT5 | +0.1903 (p 0.0001) | +0.0582 (p 0.0022) | +0.0569 (p 0.0045) | +0.978 | +0.0102 (p 0.296), +0.0093 (p 0.279) |

**Reading (by the rule fixed beforehand): explained jointly by tokenization cost and activation scale; activation scale alone explains it: True.**
