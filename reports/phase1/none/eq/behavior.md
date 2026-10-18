# Behavioural link, stream `eq` (regime `none`; decision D-018)

Per-language selective-feature damage fraction against the next-token loss increase, as in the evaluation-plan table of `paper/main.tex` (row 'Behavioral Link'). Damage fraction = 1 - survival (r not above 0.9, the proposal's own usage in G0 (b)); 'damaged' = r below 0.5 as sensitivity (Duan et al.). Token fertility = Srivastava (2026, arXiv:2607.24276) tax multiplier relative to English on parallel FLORES-200 devtest text, in character fertility (tokens per character; their fallback for scripts without word spaces, five of our scripts have none). Spearman rho with a one-sided test in the table's 'positive' direction (more selective damage, more loss increase); OLS on raw predictors; each language is one observation.

## INT7

| damage definition | n languages | Spearman rho | p (one-sided) | p (two-sided) | OLS slope of selective damage | SE | p | VIF (selective / overall / fertility) |
|---|---|---|---|---|---|---|---|---|
| 1 - survival (r <= 0.9) | 27 | +0.439 | 0.011 | 0.022 | +0.293 | 0.156 | 0.0734 | 1.4 / 1.5 / 1.3 |
| damaged (r < 0.5) | 27 | +0.561 | 0.00116 | 0.00233 | +2.672 | 2.6 | 0.314 | 1.4 / 1.4 / 1.2 |

## INT6

| damage definition | n languages | Spearman rho | p (one-sided) | p (two-sided) | OLS slope of selective damage | SE | p | VIF (selective / overall / fertility) |
|---|---|---|---|---|---|---|---|---|
| 1 - survival (r <= 0.9) | 27 | +0.531 | 0.00221 | 0.00442 | +0.333 | 0.187 | 0.0882 | 1.3 / 2.3 / 1.8 |
| damaged (r < 0.5) | 27 | +0.257 | 0.098 | 0.196 | +0.5492 | 1.06 | 0.609 | 1.2 / 1.0 / 1.2 |

## INT5

| damage definition | n languages | Spearman rho | p (one-sided) | p (two-sided) | OLS slope of selective damage | SE | p | VIF (selective / overall / fertility) |
|---|---|---|---|---|---|---|---|---|
| 1 - survival (r <= 0.9) | 27 | +0.066 | 0.372 | 0.744 | -0.6801 | 1.26 | 0.595 | 1.9 / 2.3 / 2.1 |
| damaged (r < 0.5) | 27 | -0.434 | 0.988 | 0.0236 | -1.565 | 1.45 | 0.293 | 1.8 / 1.5 / 1.3 |

