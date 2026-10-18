# Behavioural link, stream `all` (regime `none`; decision D-018)

Per-language selective-feature damage fraction against the next-token loss increase, as in the evaluation-plan table of `paper/main.tex` (row 'Behavioral Link'). Damage fraction = 1 - survival (r not above 0.9, the proposal's own usage in G0 (b)); 'damaged' = r below 0.5 as sensitivity (Duan et al.). Token fertility = Srivastava (2026, arXiv:2607.24276) tax multiplier relative to English on parallel FLORES-200 devtest text, in character fertility (tokens per character; their fallback for scripts without word spaces, five of our scripts have none). Spearman rho with a one-sided test in the table's 'positive' direction (more selective damage, more loss increase); OLS on raw predictors; each language is one observation.

## INT7

| damage definition | n languages | Spearman rho | p (one-sided) | p (two-sided) | OLS slope of selective damage | SE | p | VIF (selective / overall / fertility) |
|---|---|---|---|---|---|---|---|---|
| 1 - survival (r <= 0.9) | 27 | +0.510 | 0.00326 | 0.00653 | +0.3185 | 0.152 | 0.0479 | 1.6 / 1.8 / 1.2 |
| damaged (r < 0.5) | 27 | +0.398 | 0.0199 | 0.0398 | +1.033 | 3.95 | 0.796 | 1.6 / 1.7 / 1.1 |

## INT6

| damage definition | n languages | Spearman rho | p (one-sided) | p (two-sided) | OLS slope of selective damage | SE | p | VIF (selective / overall / fertility) |
|---|---|---|---|---|---|---|---|---|
| 1 - survival (r <= 0.9) | 27 | +0.524 | 0.00249 | 0.00498 | +0.3042 | 0.171 | 0.0883 | 1.4 / 2.3 / 1.8 |
| damaged (r < 0.5) | 27 | +0.269 | 0.0872 | 0.174 | +0.3667 | 1.13 | 0.749 | 1.2 / 1.1 / 1.1 |

## INT5

| damage definition | n languages | Spearman rho | p (one-sided) | p (two-sided) | OLS slope of selective damage | SE | p | VIF (selective / overall / fertility) |
|---|---|---|---|---|---|---|---|---|
| 1 - survival (r <= 0.9) | 27 | +0.093 | 0.322 | 0.643 | -0.3741 | 1.2 | 0.759 | 1.9 / 2.3 / 2.1 |
| damaged (r < 0.5) | 27 | -0.273 | 0.916 | 0.169 | -0.6911 | 1.42 | 0.632 | 1.8 / 1.7 / 1.1 |

