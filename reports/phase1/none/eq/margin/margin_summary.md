# JumpReLU threshold margin as a matching variable and covariate (decision D-025)

The thresholds theta_f are the learned JumpReLU thresholds of the SAE (Gemma Scope, Lieberum et al. 2024, arXiv:2408.05147: a feature is active when its pre-activation exceeds theta_f), read with Duan et al.'s own loader (`reports/phase1/sae_thresholds.csv`). Margin ratio = maximum activation / theta_f. The existing estimators of N6, N7 and N9 are run unchanged on a survival file in which the activations are divided by the thresholds.

**Reading rule fixed beforehand.** (i) The positive residual of A-035 is *explained by margin* if the margin-matched INT6 contrast at S > 5 (finest resolution keeping at least half of the selective features) is not significant or within +-1 pp, otherwise it *remains*. (ii) Margin *carries more than raw magnitude* at the language level if with the margin scale alone the INT5 tier slope has p >= 0.05 while it is < 0.05 with the max-activation scale of N9, otherwise *no more than raw activation*.

## (a) Margin-matched INT6 contrast at S > 5

| bins | matched contrast | Holm | kept | D-020 reading |
|---|---|---|---|---|
| 3 | +9.24 pp | 0.0003 | 100 % | reversed |
| 5 | +9.94 pp | 0.0003 | 100 % | reversed |
| 10 | +9.37 pp | 0.0003 | 100 % | reversed |

Headline (10 bins, 100 % kept): **reversed**.

## (b) Mixed model with log10 margin added

S~ coefficient in the full model +5.536 (p 2.94e-20); with log10 margin added +3.933 (p 4.35e-21).

## (c) Language-level tier test with the margin scale

Spearman(tier, margin scale) +0.815.

| width | slope with margin scale (p) | with fertility and margin scale (p) | Spearman(margin scale, survival) |
|---|---|---|---|
| INT7 | +2.39 pp (p 0.0643) | +2.30 pp (p 0.0721) | +0.564 |
| INT6 | +1.03 pp (p 0.255) | +0.75 pp (p 0.257) | +0.916 |
| INT5 | -0.03 pp (p 0.522) | -0.01 pp (p 0.511) | +0.982 |

INT5 p with the max-activation scale (N9): 0.58; with the margin scale: 0.522.

**Reading (i): remains. Reading (ii): no more than raw activation.**
