# Figure index of the functional study and Study G, for the paper team

Written 2026-10-26 (Claude). A map of the finished figures: where each file is, the question its title asks, which claim of [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md) it supports, and the status and limit that must travel with it. **No number in this page is new**: the titles are the figures' own (read from the figure code), the claim numbers refer to the claim table of RESULTS_SUMMARY, and every figure carries its status in its own subtitle. Every figure exists as `svg` (live text), `png`, in a `light` and a `dark` theme (same name with `_light` or `_dark`); the links below point to the light PNG. The wording of the paper and the choice of figures are the team's.

Common limits (state them once in the paper, the figure subtitles repeat them): one model (Gemma-2-2B), one layer (12), one SAE (Gemma Scope 16k), round-to-nearest per-output-channel weight quantization; devtest sentences examined before (exploratory) unless the row says confirmatory; 27 FLORES-200 languages (Belebele: the 18 covered). Known presentation limit of the palette (seen in the Study G figures, which share the precision ramp): in the dark theme the full-precision line (the darkest step of the ramp) is faint against the dark background; use the light theme for print.

## 1. Functional study (T-LM loss task, frozen setting `evaluation_freeze_lm.json`)

| figure | question it asks (its title) | claim | status and limit |
|---|---|---|---|
| [`figures/eval/fig1_effect_per_language`](figures/eval/fig1_effect_per_language_light.png) | Removing a language's feature group: harm beyond matched controls (full precision, 27 languages) | 1 | exploratory frozen setting; the effect E = group harm minus matched-control harm, nats per token, on the language's own text |
| [`figures/eval/fig2_retention`](figures/eval/fig2_retention_light.png) | Effect of the language feature groups, relative to full precision (ratio of E at each precision to E at full precision) | 2 | rounding-only controls (INT16, INT12) sit at 1.00; INT4 is far from the others (2.14), see claim 2 for the bands |
| [`figures/eval/fig3_group_size`](figures/eval/fig3_group_size_light.png) | Effect of removing the language group, by group size and precision (16, 64 and all selective features) | 1, 2 | median over the 27 languages; the groups are the frozen full-precision selections |
| [`figures/rescue/fig4_rescue`](figures/rescue/fig4_rescue_light.png) | Share of the quantization loss recovered by patching full-precision activations | 4 | exploratory; at most about 3 % of the damage sits in the language groups, 30 to 40 % in the layer-12 residual |
| [`figures/eval/fig5_association_vs_function`](figures/eval/fig5_association_vs_function_light.png) | Association retention and functional retention across precision | 8 to 10 | the association measure of the earlier study (D-035) against the group effect |
| [`figures/eval/fig6_sae_fidelity`](figures/eval/fig6_sae_fidelity_light.png) | SAE fidelity at layer 12 across precision and language (fraction of variance unexplained) | setting | context for every SAE-based figure: lower is better, languages sorted by the full-precision value |
| [`figures/eval/fig8_association_by_rank`](figures/eval/fig8_association_by_rank_light.png) | Association retention by rank of the feature within its language | 8 | pooled over languages, Wilson 95 % intervals; among the features established in full precision |
| [`figures/eval/fig9_f4_INT4`](figures/eval/fig9_f4_INT4_light.png), [`INT5`](figures/eval/fig9_f4_INT5_light.png), [`INT6`](figures/eval/fig9_f4_INT6_light.png) | Ablating features by their association status, matched on native activity | F4 row of the question table | exploratory; no systematic difference between features whose association was lost and activity-matched features that kept it |

## 2. Downstream task (Belebele, the only confirmatory family)

| figure | question it asks (its title) | claim | status and limit |
|---|---|---|---|
| [`figures/eval/fig7_belebele_coverage`](figures/eval/fig7_belebele_coverage_light.png) | Which languages the Belebele task can speak for | 3 | accuracy of the chosen format on the calibration items with its Wilson 95 % interval, chance and the coverage threshold (chance + margin 0.05) as reference lines; 18 of 27 languages covered; the prompt cross-check (claim 16, [`promptcheck/report.md`](promptcheck/report.md)) found the frozen prompt scores higher than the harness prompt in 18 of 18 languages |
| [`figures/mc_eval/fig10_macro_effect`](figures/mc_eval/fig10_macro_effect_light.png) | Macro effect of removing the language groups, by precision | 3 | **confirmatory statement not met**: zeroing the group does not change Belebele accuracy beyond matched controls |
| [`figures/mc_eval/fig1_effect_per_language`](figures/mc_eval/fig1_effect_per_language_light.png) | Removing a language's feature group: harm beyond matched controls, per language (the same figure function as the loss task, here on the Belebele items; the unit is on the axis) | 3 | same status as fig10 |

## 3. Does a single feature do the same thing after quantization (F5)

| figure | question it asks (its title) | claim | status and limit |
|---|---|---|---|
| [`figures/f5/f5_fig1_w1_distribution`](figures/f5/f5_fig1_w1_distribution_light.png) | Does switching a feature off still do the same thing? (W1 per tested feature and state, with the numerical floor) | 5 | exploratory; the rule was fixed before the first number; 1,487 features of all 27 languages |
| [`figures/f5/f5_fig2_read_vs_write`](figures/f5/f5_fig2_read_vs_write_light.png) | Do the features that survive by the activation cutoff still do the same thing? (one dot per tested feature: token correlation with full precision on the read side against W1 on the write side; dotted: the survival cutoff 0.9; dashed: the numerical floor; coloured by the D-035 status) | 7 | the activation-based survival criterion is only partly informative about function; panels INT6, INT5, INT4 |
| [`figures/f5/f5_fig3_statements`](figures/f5/f5_fig3_statements_light.png) | Is the survival metric sensible? Three statements fixed in advance | 7 | statements S1 to S3 as pre-written |
| [`figures/f5/f5_fig4_retrieval`](figures/f5/f5_fig4_retrieval_light.png) | Is a quantized feature still the same feature? (nearest effect fingerprint) | 5 | retrieval 0.997 to 1.000 |
| [`figures/f5/f5_fig5_median_ladder`](figures/f5/f5_fig5_median_ladder_light.png) | Does the activation (survival) basis track the function? (median read-side correlation and median write-side W1 by state; dashed: numerical floor) | 5, 7 | the median ladder INT16 to INT4 against the numerical floor |
| [`figures/f5/f5_fig6_paired`](figures/f5/f5_fig6_paired_light.png) | Do retained features keep their function better than lost ones? (F4 pairs, one dot per language) | 10 and the F4 row | activity-matched pairs; no systematic difference |
| [`figures/f5/ladder_control`](figures/f5/ladder_control_light.png) | Does a feature keep its effect better than another direction? (features against swapped directions) | 6 | the change is generic: a different direction at the same sites loses about as much similarity |

## 4. Study G (graded gain), exploratory

| figure | question it asks (its title) | claim | status and limit |
|---|---|---|---|
| [`gain_study/figures/fig_g1_language_curves`](gain_study/figures/fig_g1_language_curves_light.png) | Net harm of scaling a language's selective features, full precision (27 panels) | 12 | net harm (group minus matched controls) as a fraction of each language's own zero-ablation harm; symmetric log scale beyond 1 |
| [`gain_study/figures/fig_g2_macro`](gain_study/figures/fig_g2_macro_light.png) | Dose-response of the language-selective group across precisions (median over languages) | 12 | the precision dependence is a threshold at INT4; six states, 300 sentences, three control draws |
| [`gain_study/figures/fig_g3_heatmap`](gain_study/figures/fig_g3_heatmap_light.png) | Net harm by language and precision, as a fraction of the full-precision zero-ablation net harm | 12 | consistency across languages |
| [`gain_study/figures/fig_g4_repair`](gain_study/figures/fig_g4_repair_light.png) | Does turning the group up repair the quantized model? (fine ratios 1.1 and 1.2; optimal gain against shrinkage) | 12 | a suggestion at 300 sentences; **superseded by figure `gc1` below for the repair statement** |
| [`gain_study/figures_extension_c/fig_gc1_repair`](gain_study/figures_extension_c/fig_gc1_repair_light.png) | Does turning the language-selective group up slightly lower the loss? (median net harm with its 97.5 % interval, languages with a negative net harm) | 13 | 562 fresh sentences, five control draws; the repair is met at ratio 1.1 only (interval below 0), size about a thousandth of a nat per token; the interval endpoints are bootstrap percentiles over 27 languages and move with the language order: read them from `extension_c/bands.csv`, never recompute |
| [`gain_study/figures_extension_d/fig_gd1_alpha`](gain_study/figures_extension_d/fig_gd1_alpha_light.png) | Is the optimal gain of a language set by how far its group shrank at INT4? (and its split-half check) | 14 | post hoc in origin; not supported (rank correlation -0.17); the estimator is reliable (0.92), the relation is absent |
| [`gain_study/figures_state_defined/fig_gb1_state_defined`](gain_study/figures_state_defined/fig_gb1_state_defined_light.png) | Does selecting the group in the quantized model change its dose-response? (INT6 and INT4 panels) | 15 | the two selections' dose-response curves coincide; H-Gb2 not met with the opposite sign |

## 5. Tables

The table files sit next to the figures (`table_*.md` and `.csv` in `figures/eval/`, `figures/mc_eval/`, `figures/rescue/`, `figures/f5/`) and in the result folders (`results/`, `gain_study/results/`, `gain_study/extension_c/`, `gain_study/extension_d/`, `gain_study/state_defined/results/`, `promptcheck/`). Every analysis in this study can be rerun from its input files and compared byte for byte: [`REPRODUCE.md`](REPRODUCE.md) (45 outputs of Study G and the prompt cross-check identical on the last fresh run, 2026-10-26).

## 6. Not figured (by design)

The Study G extension B figure shows the median only; the per-language differences of H-Gb2 are in `gain_study/state_defined/results/analysis.json` (`rules.H-Gb2.INT4.per_language`). The Belebele prompt cross-check has a table (`promptcheck/report.md`, `languages.csv`) and no figure: 18 differences of the same sign fit a table.
