# Evidence map for the research manuscript

Revision: 30 October 2026. Root is the `always-late` repository. The manuscript uses completed evidence only. Final RQs are retrospective synthesis, not replacement preregistrations. No new model runs or hypothesis tests are introduced.

| Manuscript result | Committed source | Population, uncertainty and interpretation |
|---|---|---|
| Initial survival contrasts; 69.4/90.6/95.1 tier survival and scale adjustment | `reports/phase1_summary.md`, `reports/phase1_results.md` | Original BOS-masked analysis; Holm-adjusted tests; nonrejection is not equivalence. Adjustment is associational. |
| GPTQ INT4 +17.83 pp and −0.461 NLL versus legacy; +11.86 pp and −0.255 versus matched | `reports/gptq_extension/results/stage1/stage1_comparison.md`, `stage1_summary.csv` | Mean of 27 paired language differences; language-bootstrap CIs; all three-arm contrasts exploratory. |
| FP mean 0.704 [0.684,0.723], median .262; 27/27 positive; 1.03/1.54 median retention | `reports/functional_validation/results/eval/report.md`, `reports/functional_validation/evaluation_freeze_lm.json` | Three K64 evaluation control draws; manifest stores up to five. 300 sentences per language; sentence-cluster bootstrap with languages/groups fixed. Ratio aggregations kept distinct. |
| Ordered graded suppression .022/.090/.169/.262 | `reports/functional_validation/gain_study/results/report.md`, `curves.csv` | Three matched control draws; original Study G; exploratory. |
| GPTQ median INT4 group ratio 1.22 [1.14,1.24] | `reports/gptq_extension/results/stage3/analysis_gptq_eval/report.md` | Within-arm sentence bootstrap; not a CI on a paired quantizer contrast. |
| 9B .270 FP harm; 1.586/2.479/1.474 INT4 ratios; 26/27 >1; paired model difference | `reports/functional_validation/gemma9b/REVIEWED_SUMMARY.md`, `review/` | 27-language percentile bootstrap; raw files hash-checked, unchanged analysis reproduced byte-identically; inference not independently rerun. Sign error and circular criterion retained. |
| Llama FP .048 [.018,.173], 21/27 positive and inconsistent verdict | `reports/functional_validation/llama/report.md` | Different model, SAE architecture, depth, tokenizer and matching behavior; no causal isolation of those factors. |
| Belebele −.0001 [−.0035,+.0031] | `reports/functional_validation/results/mc_eval/report.md` | Correct-option loss in nats/item, not accuracy percentage points; 18 languages, 385 items/language, passage bootstrap. Only fully confirmatory family; positive-FP prerequisite unmet. |
| Positive controls; above-chance GPTQ INT4 task null | `reports/functional_validation/results/mc_posctrl/summary.md`, `reports/gptq_extension/RESULTS.md` | Task/condition sensitivity does not establish zero effect for every edit or task. |
| Legacy 2.6% and whole-residual 39.3% rescue; GPTQ 2.0% and 41.4% | `reports/functional_validation/results/rescue/report.md`, `reports/gptq_extension/results/stage3/analysis_gptq_rescue/report.md` | Recovery under specified aligned patches; not mediation fractions or additive causal shares. |
| Tiny ×1.1 legacy gain −.00135 [−.0020,−.0003], 20/27; original repair rules unmet | `reports/functional_validation/gain_study/extension_c/report.md`, `bands.csv`, `analysis.json`; `reports/gptq_extension/results/gain/` | 562 fresh sentences; 97.5% interval; two primary ratios with Holm correction; H-Gc1/H-Gc2 met. Small benefit absent under GPTQ. Does not establish substantial repair. |
| F5 13.2% R1 survival, .839 cosine, 99.7% identity retrieval | `reports/functional_validation/results/f5/report.md`, `f5_tables.csv` | Same 1,487-feature sample, language-bootstrap intervals; .9489 reference is not the threshold for existence of function. |
| Alternate-direction control; GPTQ .923 cosine, matched .875 | `reports/functional_validation/figures/f5/ladder_control_table.csv`, `reports/gptq_extension/RESULTS.md` | Generic drift and feature-specific contribution can coexist; labels inherited from legacy association study. |
| R1 prediction gains 3.1–4.4%, all positive intervals | `reports/survival_effect_rescue/stage4_gemma_report_2026-10-28.md`, `results/survival_effect_results.json` | 1,412 reliable features; leave-language-out ridge, language-bootstrap CIs; misses 10% practical convention without erasing positive gain. |
| Llama R1 INT4 4.2% [1.6,6.7] | `reports/survival_effect_rescue/llama/llama_vs_gemma_report_2026-10-28.md` | Limited additional system check; INT6 unresolved. |
| Exact gain/noise identity; 214,506 cells; scale correlations; g/nu prediction intervals | `reports/survival_measurement/stage3/stage3_results.json`, `stage3_results.md`, `README.md` | Same FP-active cells and counts across arms; degenerate cases counted; all new-feature incremental CIs include zero. g/nu diagnosis is distinct from predictive utility. |
| nu definition, gain and flip measurement | `reports/survival_measurement/stage2_design.md` | Encoder pre-activation RMS difference conditional on FP-active tokens divided by feature threshold. |
| Main crossed transfer .848 vs .263 group loss, C/W/I, native .433 | `reports/intervention_transfer/results/main/group_per_language.csv`, `claim_status.json` | Fixed FP-amplitude versus native edits; median group loss terms differ from norms of feature logit effects. |
| .768/.822/.877 transfer cosine, .664/.572/.476 relative error | `reports/gptq_extension/results/stage2/transfer_feature_level.csv` | Identical eligible real items; within-arm crossed-bootstrap CIs. Norms of C/W/I are not additive shares. |
| Generic predictor 21.5 [12.7,29.4], 10.9 [−1.2,21.4]; averaging gap | `reports/intervention_transfer/results/main/predictive_validation.csv`, `averaged_fingerprints.csv` | Feature-held-out split with source-sentence overlap; uncertainty crossed by feature/sentence except averaging gap's feature bootstrap. |
| External-panel cosines and utility differences | `reports/gptq_extension/RESULTS.md`, `reports/intervention_transfer/results/external/` | Six languages, fresh documents; Wikipedia domain shared with GPTQ calibration. Domain and compensation not separately identified. |

## Literature and figures

Primary literature was checked against official ACL, ICLR, PMLR and arXiv records; see `reports/research_sources.md` and `reports/publication_context_2026-10-30.md`. Preprints are labeled in the bibliography. No priority or exhaustive literature coverage is claimed. `make_figures.py` reads committed estimates and intervals only; `figures/sources.json` records the two plot inputs.

## Reporting boundary

The fixed model/dictionary defines the paired estimand. Coverage limits external validity; the original 9B prediction-definition defects and exploratory data reuse are separate validity/inference issues. The 9B GPTQ extension is pending; no missing result is filled by assumption. No public-data availability claim or publication acceptance claim is made.
