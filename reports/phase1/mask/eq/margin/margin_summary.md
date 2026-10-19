# JumpReLU threshold margin as a matching variable and covariate (decision D-025)

The thresholds theta_f are the learned JumpReLU thresholds of the SAE (Gemma Scope, Lieberum et al. 2024, arXiv:2408.05147: a feature is active when its pre-activation exceeds theta_f), read with Duan et al.'s own loader (`reports/phase1/sae_thresholds.csv`). Margin ratio = maximum activation / theta_f. The existing estimators of N6, N7 and N9 are run unchanged on a survival file in which the activations are divided by the thresholds.

**Reading rule fixed beforehand.** (i) The positive residual of A-035 is *explained by margin* if the margin-matched INT6 contrast at S > 5 (finest resolution keeping at least half of the selective features) is not significant or within +-1 pp, otherwise it *remains*. (ii) Margin *carries more than raw magnitude* at the language level if with the margin scale alone the INT5 tier slope has p >= 0.05 while it is < 0.05 with the max-activation scale of N9, otherwise *no more than raw activation*.

## (a) Margin-matched INT6 contrast at S > 5

| bins | matched contrast | Holm | kept | D-020 reading |
|---|---|---|---|---|
| 3 | +0.95 pp | 1 | 100 % | explained by magnitude |
| 5 | +3.88 pp | 0.0108 | 100 % | reversed |
| 10 | +5.32 pp | 0.0006 | 100 % | reversed |

Headline (10 bins, 100 % kept): **reversed**.

## (b) Mixed model with log10 margin added

S~ coefficient in the full model +1.194 (p 0.000434); with log10 margin added +3.002 (p 1.48e-14).

## (c) Language-level tier test with the margin scale

Spearman(tier, margin scale) +0.833.

| width | slope with margin scale (p) | with fertility and margin scale (p) | Spearman(margin scale, survival) |
|---|---|---|---|
| INT7 | -0.72 pp (p 0.975) | -0.64 pp (p 0.971) | +0.957 |
| INT6 | -0.67 pp (p 0.643) | -0.20 pp (p 0.555) | +0.966 |
| INT5 | +5.99 pp (p 0.002) | +5.88 pp (p 0.004) | +0.978 |

INT5 p with the max-activation scale (N9): 0.0022; with the margin scale: 0.002.

**Reading (i): remains. Reading (ii): no more than raw activation.**
