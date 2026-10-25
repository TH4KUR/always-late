# Task 17 part (b): prompt cross-check of the Belebele pipeline

Exploratory; changes no frozen statement. Full precision, no edit, calibration items of each language (250 per language), the letter format scored on the next token among ' A' to ' D'. **Frozen** = the prompt of the frozen task module ('Passage:', 'Question:', 'A.' to 'D.'); **harness** = the lm-evaluation-harness Belebele prompt ('P:', 'Q:', 'A:' to 'D:', source S08 of `reports/research_sources.md`); the same items, the same model and scoring. Rule (fixed before the run): margin 0.05; 'equivalent' when the 95 % interval of the accuracy difference (harness minus frozen) lies within the margin, 'harness prompt better' when its lower end is above the margin, 'ours better' when its upper end is below minus the margin, otherwise 'inconclusive'.

## Verdict

| quantity | harness minus frozen accuracy | 95 % interval | label |
|---|---|---|---|
| primary: eng_Latn (250 items) | -0.0800 (0.516 against 0.596) | [-0.1321, -0.0240] (bootstrap over items, 2000 draws) | **inconclusive** |
| mean over 18 languages | -0.1000 | [-0.1096, -0.0904] (bootstrap over languages, 5000 draws) | **ours better** |

## Reproduction of the stored calibration (checked before the reading): **exact**

The frozen variant is run again by the same pipeline and must give the accuracies of the stored calibration record (`belebele_calibration.json`, letter format, full precision): exact on the same GPU model; one item off in a language is flagged for a hardware explanation (D-079); more is an anomaly and stops the analysis.

Items off per language: eng_Latn 0, deu_Latn 0, spa_Latn 0, fra_Latn 0, por_Latn 0, rus_Cyrl 0, zho_Hans 0, jpn_Jpan 0, kor_Hang 0, hin_Deva 0, ben_Beng 0, tur_Latn 0, vie_Latn 0, ind_Latn 0, tha_Thai 0, pol_Latn 0, ukr_Cyrl 0, kan_Knda 0.

## Per language

| language | items | accuracy frozen | accuracy harness | difference | 95 % interval | frozen right, harness wrong | frozen wrong, harness right | McNemar p | agreement | label |
|---|---|---|---|---|---|---|---|---|---|---|
| eng_Latn | 250 | 0.596 | 0.516 | -0.080 | [-0.132, -0.024] | 37 | 17 | 0.00907 | 0.672 | inconclusive |
| deu_Latn | 250 | 0.484 | 0.396 | -0.088 | [-0.156, -0.024] | 47 | 25 | 0.0128 | 0.608 | inconclusive |
| spa_Latn | 250 | 0.484 | 0.372 | -0.112 | [-0.184, -0.044] | 54 | 26 | 0.00232 | 0.532 | inconclusive |
| fra_Latn | 250 | 0.464 | 0.380 | -0.084 | [-0.152, -0.020] | 45 | 24 | 0.0154 | 0.536 | inconclusive |
| por_Latn | 250 | 0.484 | 0.372 | -0.112 | [-0.180, -0.048] | 50 | 22 | 0.00129 | 0.532 | inconclusive |
| rus_Cyrl | 250 | 0.488 | 0.348 | -0.140 | [-0.208, -0.072] | 59 | 24 | 0.000155 | 0.472 | ours better |
| zho_Hans | 250 | 0.484 | 0.360 | -0.124 | [-0.188, -0.056] | 52 | 21 | 0.000371 | 0.504 | ours better |
| jpn_Jpan | 250 | 0.412 | 0.304 | -0.108 | [-0.172, -0.044] | 47 | 20 | 0.00131 | 0.556 | inconclusive |
| kor_Hang | 250 | 0.432 | 0.340 | -0.092 | [-0.156, -0.028] | 51 | 28 | 0.0128 | 0.448 | inconclusive |
| hin_Deva | 250 | 0.376 | 0.280 | -0.096 | [-0.152, -0.040] | 39 | 15 | 0.0015 | 0.640 | inconclusive |
| ben_Beng | 250 | 0.380 | 0.324 | -0.056 | [-0.124, +0.008] | 44 | 30 | 0.13 | 0.532 | inconclusive |
| tur_Latn | 250 | 0.412 | 0.304 | -0.108 | [-0.164, -0.048] | 45 | 18 | 0.000898 | 0.540 | inconclusive |
| vie_Latn | 250 | 0.448 | 0.364 | -0.084 | [-0.156, -0.016] | 52 | 31 | 0.0275 | 0.464 | inconclusive |
| ind_Latn | 250 | 0.460 | 0.356 | -0.104 | [-0.168, -0.044] | 47 | 21 | 0.00219 | 0.552 | inconclusive |
| tha_Thai | 250 | 0.460 | 0.320 | -0.140 | [-0.200, -0.076] | 54 | 19 | 5.06e-05 | 0.504 | ours better |
| pol_Latn | 250 | 0.424 | 0.332 | -0.092 | [-0.164, -0.020] | 53 | 30 | 0.0152 | 0.464 | inconclusive |
| ukr_Cyrl | 250 | 0.444 | 0.356 | -0.088 | [-0.160, -0.016] | 53 | 31 | 0.0214 | 0.496 | inconclusive |
| kan_Knda | 250 | 0.364 | 0.272 | -0.092 | [-0.156, -0.028] | 44 | 21 | 0.00592 | 0.560 | inconclusive |

## Reading

- equivalent: our prompt is not an unusual choice, and the absolute accuracies of the frozen Belebele statement do not depend on it at this margin.
- harness prompt better: the frozen format is prompt-sensitive. The frozen choice stays (a change would be a D-entry for the user), the null of the group edits (D-050, D-062) is stated for the frozen prompt only, and a rerun of the K = all edit with the harness prompt becomes a question for the user (default: do not run it).
- ours better: reported, nothing else. inconclusive: reported with the intervals.

## Limits

- Exploratory; one model (Gemma-2-2B, full precision), 18 languages, 250 calibration items each; the interval of a single language is wide (the difference of two accuracies of 250 items).
- The harness prompt is the lm-evaluation-harness one (mutable upstream source, fetched 2026-10-25), not the prompt of the original Belebele paper; the loader's whitespace normalisation is applied to both variants.
- A label 'equivalent' says that the accuracy level does not move by more than the margin; it does not say that the two prompts give the same answers item by item (see the agreement column).
