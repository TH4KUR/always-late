# Phase 1: state of the evidence (a results summary, not paper prose)

This file is a results summary for the team. It is **not paper prose** (course policy: the team writes the prose). Every number below sits in the generated block and is produced from the committed result files by `scripts/g0/phase1_summary.py`; a test fails if the committed file and the files disagree. The text around the block states no results.

**Current interpretation:** survival detection is under reassessment. The generated results describe fixed-SAE activation correlations for a cohort selected before quantization, not demonstrated retention of language identity or causal function. Completed BOS-free results are historical; ongoing work uses the BOS-masked regime only. See [accepted evidence and proposed validation](survival_method_reassessment.md) and [current decisions](../DECISIONS_FOR_REVIEW.md). The generated block is preserved.

Setting: Gemma-2-2B, layer-12 residual stream, Gemma Scope 16k SAE, round-to-nearest weight quantization with Duan et al.'s own functions, 27 FLORES-200 languages in three resource tiers. The primary stream masks the BOS token and uses equal token budgets per language; the BOS-free stream is a sensitivity analysis whose instrument fails its own noise-floor check (A-029).

Where the details are: the full account is `reports/phase1_results.md`, the decisions and anomalies are in `reports/decisions_and_anomalies.md`, the questions that change how the evidence is headlined are in `DECISIONS_FOR_REVIEW.md` (Q1, Q2, Q13, Q14, Q15).

<!-- BEGIN GENERATED (scripts/g0/phase1_summary.py; do not edit by hand) -->
## 1. Pooled survival (their metric, all languages pooled, equal budget)

| width | primary: survived > 0.9 | median r | perplexity change | BOS-free: survived > 0.9 | median r | perplexity change |
|---|---|---|---|---|---|---|
| INT8 | 100.00 % | 0.9906 | +0.08 % | 97.98 % | 0.9618 | -0.88 % |
| INT7 | 99.97 % | 0.9829 | +0.40 % | 64.18 % | 0.9292 | -10.95 % |
| INT6 | 94.94 % | 0.9629 | +2.67 % | 38.60 % | 0.8502 | +3.87 % |
| INT5 | 59.00 % | 0.9143 | +11.47 % | 15.77 % | 0.6851 | +122.52 % |
| INT4 | 13.75 % | 0.7495 | +103.88 % | 5.93 % | 0.3209 | +197.46 % |
| INT16 (noise floor) | 100.00 % | 0.9957 | +0.00 % | 99.98 % | 0.9823 | -0.15 % |
| INT12 (noise floor) | 100.00 % | 0.9957 | +0.04 % | 100.00 % | 0.9823 | -0.26 % |
| INT0 (restore only) | 100.00 % | 1.0000 | +0.00 % | 100.00 % | 1.0000 | +0.00 % |

## 2. Confirmatory family (Holm over three tests)

Status of every row: confirmatory (proposal section 6; D-014).

| test | primary stream (BOS masked, equal budget) | BOS-free sensitivity |
|---|---|---|
| T1 INT7 selective minus matched universal survival | -0.28 pp [-0.63, +0.13], p 0.257, Holm 0.514 | +9.16 pp [+5.18, +13.23], p 0.0001, Holm 0.0003 |
| T2 INT6 selective minus matched universal survival | -1.44 pp [-2.65, -0.13], p 0.0427, Holm 0.128 | +7.71 pp [+3.95, +11.37], p 0.0005, Holm 0.001 |
| T3 language-level tier slope of the selective-minus-control gap (one-sided) | slope -0.0048, p 0.99, Holm 0.99; equivalent within +-0.02: yes; High minus Low -0.0082, equivalent within +-0.02: yes | slope +0.0026, p 0.378, Holm 0.378; equivalent within +-0.02: yes; High minus Low +0.0212, equivalent within +-0.02: no |

## 3. Additional analyses (not part of the Holm family)

| id | analysis | status | primary stream (BOS masked, equal budget) | BOS-free sensitivity | register and file |
|---|---|---|---|---|---|
| N1 | Behavioural link: per-language selective damage against loss increase | planned in the team's table, definitions provisional (D-018) | INT6 Spearman(selective damage, loss increase) +0.46 (one-sided p 0.00816) | INT6 Spearman(selective damage, loss increase) +0.53 (one-sided p 0.00221) | D-018, A-032, V-11; reports/phase1/mask/eq/behavior.json |
| N2 | Fixed S thresholds for the selective split (deviation from the proposal's null percentiles) | additional, post hoc thresholds (D-019) | INT6 selective minus control at S > 5: -7.16 pp (Holm 0.0003) | INT6 selective minus control at S > 5: +17.46 pp (Holm 0.0003) | D-019, A-034; reports/phase1/mask/eq/thresholds/thresholds.json |
| N6 | Activation-magnitude-matched controls | additional, pre-specified before running (D-020) | INT6 at S > 5, controls matched on firing rate x 10 magnitude bins: +4.58 pp (Holm 0.0012); headline reading reversed | INT6 at S > 5, controls matched on firing rate x 10 magnitude bins: +9.29 pp (Holm 0.0003); headline reading reversed | D-020, A-035; reports/phase1/mask/eq/magnitude/magnitude.json |
| N7 | Exploratory mixed model refitted in variants | exploratory (D-014); variants pre-specified (D-021) | S~ coefficient without the tier terms -0.81 (p 0.0044); in the full model +1.19; reading: explained by the tier terms | not run | D-021, A-034, A-036, A-037; reports/phase1/mask/eq/mixed_variants/mixed_variants.json |
| N8 | Language-level survival by resource tier | additional, motivated post hoc and pre-specified before running (D-022) | INT6 mean survival low / mid / high tier 69.4 / 90.6 / 95.1 %; slope +12.3 pp per tier step (Holm 0.0002); with fertility +6.3 pp (p 0.0032) | INT6 mean survival low / mid / high tier 25.5 / 34.5 / 38.6 %; slope +6.7 pp per tier step (Holm 0.0002); with fertility +4.0 pp (p 0.0002) | D-022, A-037, A-038; reports/phase1/mask/eq/language_level/language_level.json |
| N9 | Activation scale as a covariate of the language-level tier test | additional, pre-specified before running (D-023) | INT6 slope with activation scale -0.8 pp (p 0.667); Spearman(scale, survival) +0.97; explained jointly by tokenization cost and activation scale | INT6 slope with activation scale +1.0 pp (p 0.266); Spearman(scale, survival) +0.91; explained jointly by tokenization cost and activation scale | D-023, A-039; reports/phase1/mask/eq/language_level/language_level.json |
| N11 | JumpReLU threshold margin (activation over the SAE's learned threshold) as a matching variable and covariate | additional, pre-specified before running (D-025) | INT6 at S > 5, controls matched on firing rate x 10 margin bins: +5.32 pp (Holm 0.0006); (i) remains; (ii) no more than raw activation | INT6 at S > 5, controls matched on firing rate x 10 margin bins: +9.37 pp (Holm 0.0003); (i) remains; (ii) no more than raw activation | D-025, A-040; reports/phase1/mask/eq/margin/margin_summary.json |

## 4. Status labels

| label | meaning |
|---|---|
| confirmatory | the proposal's pre-registered family (section 6): the INT7 and INT6 selective-versus-matched contrasts and the language-level tier test, Holm over three |
| planned in the team's table | named in the team's evaluation-plan table but with undefined terms; definitions are provisional choices |
| additional, pre-specified before running | not in the proposal's family; the analysis and its reading rule were written down before it was run |
| additional, motivated post hoc | as above, but motivated by an unplanned observation made on the results |
| additional, post hoc thresholds | the threshold values were chosen after seeing the data |
| exploratory | reported and labelled as such; p-values uncorrected |
<!-- END GENERATED -->

Reading notes (qualitative; read with the caveats in the register):

- The confirmatory family is the proposal's pre-registered family; its results are in the second table. Everything in the third table was added after the confirmatory results were known; each row says how it was specified, and none is part of the Holm family.
- The tier and activation-scale analyses are correlational (27 languages, three tiers); the activation scale and the survival are computed from the same activations and are partly mechanically related.
- Which analysis is the confirmatory test of the resource-tier question is an open decision (Q14).
- The BOS-free column is a sensitivity analysis only.
