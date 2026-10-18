# Behavioural link, stream `all` (regime `mask`; decision D-018)

Per-language selective-feature damage fraction against the next-token loss increase, as in the evaluation-plan table of `paper/main.tex` (row 'Behavioral Link'). Damage fraction = 1 - survival (r not above 0.9, the proposal's own usage in G0 (b)); 'damaged' = r below 0.5 as sensitivity (Duan et al.). Token fertility = Srivastava (2026, arXiv:2607.24276) tax multiplier relative to English on parallel FLORES-200 devtest text, in character fertility (tokens per character; their fallback for scripts without word spaces, five of our scripts have none). Spearman rho with a one-sided test in the table's 'positive' direction (more selective damage, more loss increase); OLS on raw predictors; each language is one observation.

## INT7

| damage definition | n languages | Spearman rho | p (one-sided) | p (two-sided) | OLS slope of selective damage | SE | p | VIF (selective / overall / fertility) |
|---|---|---|---|---|---|---|---|---|
| 1 - survival (r <= 0.9) | 27 | +0.537 | 0.00195 | 0.00389 | +0.08836 | 0.075 | 0.251 | 5.4 / 5.9 / 1.6 |
| damaged (r < 0.5) | undefined (constant selective damage or loss increase: rank correlation undefined) | | | | | | | |

## INT6

| damage definition | n languages | Spearman rho | p (one-sided) | p (two-sided) | OLS slope of selective damage | SE | p | VIF (selective / overall / fertility) |
|---|---|---|---|---|---|---|---|---|
| 1 - survival (r <= 0.9) | 27 | +0.455 | 0.00852 | 0.017 | +0.03742 | 0.0489 | 0.452 | 8.0 / 11.0 / 2.4 |
| damaged (r < 0.5) | 27 | +0.004 | 0.493 | 0.985 | -0.6836 | 0.984 | 0.494 | 2.3 / 3.1 / 1.6 |

## INT5

| damage definition | n languages | Spearman rho | p (one-sided) | p (two-sided) | OLS slope of selective damage | SE | p | VIF (selective / overall / fertility) |
|---|---|---|---|---|---|---|---|---|
| 1 - survival (r <= 0.9) | 27 | +0.703 | 2.19e-05 | 4.38e-05 | +0.1164 | 0.0405 | 0.00858 | 2.5 / 3.4 / 2.0 |
| damaged (r < 0.5) | 27 | +0.524 | 0.00251 | 0.00503 | -0.3227 | 1.26 | 0.8 | 3.5 / 3.8 / 1.8 |

