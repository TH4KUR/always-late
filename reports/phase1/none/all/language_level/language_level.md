# Language-level survival by resource tier, regime `none`, stream `all` (decision D-022)

This analysis was pre-specified (handoff 004, written when only the header of the per-language files had been read). Question: does the survival of ALL active features of a language depend on the language's resource tier (proposal RQ2; paper RQ2 row), and is it explained by character fertility (Srivastava 2026, arXiv:2607.24276: tax multiplier = tokens per character relative to English, FLORES-200 parallel text)? Outcome per language: the fraction of its active features with r > 0.9 (unweighted over 27 languages). Statistic: slope on the ordinal tier (low 0, mid 1, high 2), one-sided (higher tier, higher survival), Freedman-Lane permutation (10000 draws, seed 0) with script-family dummies as nuisance; Holm over the two confirmatory widths INT7 and INT6; INT5 descriptive; the same test with the fertility multiplier added as a nuisance covariate. **Reading rule fixed beforehand:** survival depends on tier if the INT6 slope is positive with Holm < 0.05; if so, a fertility-adjusted p >= 0.05 (strict) means the effect is explained by tokenization cost, otherwise it is beyond fertility; if the first condition fails nothing is claimed and no equivalence is asserted.

| width | mean survival low / mid / high tier | slope (p, Holm) | High - Low | slope with fertility (p) | Joshi-class slope (p) | median-r slope (p) |
|---|---|---|---|---|---|---|
| INT7 | 57.2 / 63.9 / 65.5 % | +0.0534 (p 0.0001, Holm 0.0002) | +8.3 pp | +0.0419 (p 0.0002) | +0.0315 (p 0.0001) | +0.0119 (p 0.0001) |
| INT6 | 25.2 / 34.5 / 38.5 % | +0.0661 (p 0.0001, Holm 0.0002) | +13.3 pp | +0.0382 (p 0.0002) | +0.0360 (p 0.0001) | +0.0251 (p 0.0001) |
| INT5 | 8.1 / 13.1 / 19.3 % | +0.0525 (p 0.0001 (descriptive)) | +11.2 pp | +0.0445 (p 0.0001) | +0.0275 (p 0.0001) | +0.0400 (p 0.0001) |

**Reading (by the rule fixed beforehand): depends on tier; beyond fertility.**

## Activation scale as a covariate (N9, decision D-023)

Does per-language activation scale carry the remaining tier effect? Source: `issues.md` C6 / A-035 (survival follows activation magnitude in Duan et al.'s released data and in ours). Scale of a language = log10 of the median, over its active features (full-precision firing rate in that language above 0.001), of the feature's maximum activation in that language (secondary: mean activation). The same Freedman-Lane test as above with (A) the scale alone and (B) character fertility and the scale together as nuisance covariates. **Reading rule fixed beforehand:** with both covariates the INT6 slope p >= 0.05 (strict, p exactly 0.05 is not significant) -> explained jointly by tokenization cost and activation scale; p < 0.05 -> a tier effect remains beyond fertility and activation scale; the scale alone with p >= 0.05 -> activation scale alone explains it.

Spearman(tier, scale) across languages: +0.798.

| width | slope, no covariate | slope with scale (p) | slope with fertility and scale (p) | Spearman(scale, survival) | secondary (mean activation): with scale (p), with both (p) |
|---|---|---|---|---|---|
| INT7 | +0.0534 (p 0.0001) | +0.0321 (p 0.0368) | +0.0207 (p 0.112) | +0.570 | +0.0379 (p 0.0008), +0.0380 (p 0.0012) |
| INT6 | +0.0661 (p 0.0001) | +0.0290 (p 0.1) | +0.0014 (p 0.451) | +0.883 | +0.0312 (p 0.0003), +0.0311 (p 0.0004) |
| INT5 | +0.0525 (p 0.0001) | +0.0048 (p 0.322) | -0.0031 (p 0.631) | +0.966 | +0.0423 (p 0.0001), +0.0422 (p 0.0002) |

**Reading (by the rule fixed beforehand): explained jointly by tokenization cost and activation scale; activation scale alone explains it: True.**
