# Behavioural link, stream `eq` (regime `mask`; decision D-018)

Per-language selective-feature damage fraction against the next-token loss increase, as in the evaluation-plan table of `paper/main.tex` (row 'Behavioral Link'). Damage fraction = 1 - survival (r not above 0.9, the proposal's own usage in G0 (b)); 'damaged' = r below 0.5 as sensitivity (Duan et al.). Token fertility = Srivastava (2026, arXiv:2607.24276) tax multiplier relative to English on parallel FLORES-200 devtest text, in character fertility (tokens per character; their fallback for scripts without word spaces, five of our scripts have none). Spearman rho with a one-sided test in the table's 'positive' direction (more selective damage, more loss increase); OLS on raw predictors; each language is one observation.

## INT7

| damage definition | n languages | Spearman rho | p (one-sided) | p (two-sided) | OLS slope of selective damage | SE | p | VIF (selective / overall / fertility) |
|---|---|---|---|---|---|---|---|---|
| 1 - survival (r <= 0.9) | 27 | +0.539 | 0.00186 | 0.00371 | -0.0519 | 0.0855 | 0.55 | 5.5 / 5.6 / 1.9 |
| damaged (r < 0.5) | undefined (constant selective damage or loss increase: rank correlation undefined) | | | | | | | |

## INT6

| damage definition | n languages | Spearman rho | p (one-sided) | p (two-sided) | OLS slope of selective damage | SE | p | VIF (selective / overall / fertility) |
|---|---|---|---|---|---|---|---|---|
| 1 - survival (r <= 0.9) | 27 | +0.458 | 0.00816 | 0.0163 | +0.05638 | 0.0539 | 0.306 | 7.0 / 9.5 / 2.3 |
| damaged (r < 0.5) | 27 | +0.172 | 0.196 | 0.391 | +1.73 | 1.24 | 0.176 | 2.2 / 3.2 / 1.9 |

## INT5

| damage definition | n languages | Spearman rho | p (one-sided) | p (two-sided) | OLS slope of selective damage | SE | p | VIF (selective / overall / fertility) |
|---|---|---|---|---|---|---|---|---|
| 1 - survival (r <= 0.9) | 27 | +0.692 | 3.22e-05 | 6.44e-05 | +0.1114 | 0.0424 | 0.0149 | 2.3 / 3.4 / 2.0 |
| damaged (r < 0.5) | 27 | +0.631 | 0.000209 | 0.000418 | +0.9244 | 1.05 | 0.386 | 2.0 / 2.3 / 1.9 |

