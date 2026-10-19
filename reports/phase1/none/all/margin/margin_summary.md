# JumpReLU threshold margin as a matching variable and covariate (decision D-025)

The thresholds theta_f are the learned JumpReLU thresholds of the SAE (Gemma Scope, Lieberum et al. 2024, arXiv:2408.05147: a feature is active when its pre-activation exceeds theta_f), read with Duan et al.'s own loader (`reports/phase1/sae_thresholds.csv`). Margin ratio = maximum activation / theta_f. The existing estimators of N6, N7 and N9 are run unchanged on a survival file in which the activations are divided by the thresholds.

**Reading rule fixed beforehand.** (i) The positive residual of A-035 is *explained by margin* if the margin-matched INT6 contrast at S > 5 (finest resolution keeping at least half of the selective features) is not significant or within +-1 pp, otherwise it *remains*. (ii) Margin *carries more than raw magnitude* at the language level if with the margin scale alone the INT5 tier slope has p >= 0.05 while it is < 0.05 with the max-activation scale of N9, otherwise *no more than raw activation*.

## (a) Margin-matched INT6 contrast at S > 5

| bins | matched contrast | Holm | kept | D-020 reading |
|---|---|---|---|---|
| 3 | +7.14 pp | 0.0003 | 100 % | reversed |
| 5 | +8.04 pp | 0.0003 | 100 % | reversed |
| 10 | +7.71 pp | 0.0003 | 100 % | reversed |

Headline (10 bins, 100 % kept): **reversed**.

## (b) Mixed model with log10 margin added

S~ coefficient in the full model +5.393 (p 3e-21); with log10 margin added +3.909 (p 1.78e-22).

## (c) Language-level tier test with the margin scale

Spearman(tier, margin scale) +0.780.

| width | slope with margin scale (p) | with fertility and margin scale (p) | Spearman(margin scale, survival) |
|---|---|---|---|
| INT7 | +3.41 pp (p 0.0275) | +2.15 pp (p 0.105) | +0.562 |
| INT6 | +3.23 pp (p 0.0771) | +0.21 pp (p 0.431) | +0.873 |
| INT5 | +0.68 pp (p 0.251) | -0.24 pp (p 0.6) | +0.952 |

INT5 p with the max-activation scale (N9): 0.322; with the margin scale: 0.251.

**Reading (i): remains. Reading (ii): no more than raw activation.**
