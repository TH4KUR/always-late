# Final research questions and answers

Readability revision, 30 October 2026. These questions organize the completed work. They were chosen after the investigation and do not replace the original hypotheses or analysis records. The [research paper](../paper/research-paper/main.tex) and [full report](FULL_PROJECT_REPORT_DRAFT.md) give methods and further results.

## A problem we found before the main experiments

We reproduced Duan's Gemma survival results using their released code, then identified a problem in its input protocol: it omitted the beginning-of-sequence (BOS) token. On the same WikiText content windows, adding BOS and excluding it from feature statistics reduced full-precision perplexity from 458.4 to 16.1 and increased six-bit feature survival from 51.15% to 97.52%. Lower perplexity means better next-token prediction. Each setting defines its own active-feature set, so the survival rates do not follow exactly the same population.

We corrected our subsequent input protocol. This establishes a large BOS effect on the reported Gemma measurements, without showing that every conclusion in Duan's paper is wrong. The [BOS audit](audit_our_vs_duan.md) retains the reproduction and controlled comparisons.

## RQ1. Which language features retain their activation patterns after quantization, and how does the quantizer affect this?

**Answer.** The quantizer and the size of a feature's activations matter. We did not establish that language-selective features consistently survive worse than controls across methods. In the original RTN tests, their six-bit survival was 1.44 percentage points lower, but the difference did not pass correction for multiple comparisons (adjusted p = 0.128). A small difference remains possible. Matched RTN changes the significance of a small contrast; GPTQ does not show that disadvantage.

GPTQ improves four-bit survival by 17.83 percentage points [16.29, 19.38] over legacy RTN and 11.86 [11.18, 12.50] over matched RTN. These compare mean survival rates across paired languages. Under legacy RTN, low-, mid- and high-resource languages have six-bit survival rates of 69.4%, 90.6% and 95.1%. Accounting for activation size reduces the initial tier difference. Other measures retain some tier patterns, including at the small INT16 reference change; GPTQ's adjusted pattern depends on the adjustment. These observations do not by themselves identify the cause of language-level performance damage.

Sources: [`phase1_summary.md`](phase1_summary.md), [`gptq_extension/results/stage1/stage1_comparison.md`](gptq_extension/results/stage1/stage1_comparison.md), [`survival_measurement/stage3/README.md`](survival_measurement/stage3/README.md).

## RQ2. How does editing language features affect text prediction and comprehension before and after quantization?

**Answer.** Removing a language's selected group harms its next-token prediction more than removing matched controls in all 27 languages of Gemma-2-2B. Each group contains its 64 selective features with the highest mean development-set activation (K64). Median extra loss is 0.262 nats/token in full precision (FP). Lower loss is better. Gradually weakening the group increases harm in all 27 languages.

At six bits the removal effect stays near FP. At four bits its median ratio to FP is 1.54 under legacy RTN and 1.22 under GPTQ. A ratio of 1.54 means 54% greater harm from removal; it does not mean improved language ability. Restoring FP values to K64 repairs only 2.6% of the legacy four-bit loss gap, compared with 39.3% when restoring the whole layer residual. These are benefits of particular patches, not shares of the total cause of damage.

The original strengthening experiment fails its repair rules. A follow-up on 562 fresh sentences finds a tiny legacy-RTN benefit at multiplier 1.1: −0.00135 nats/token [−0.0020, −0.0003] (97.5% interval; 20/27 languages improve). The primary multiplier tests use Holm correction. GPTQ does not repeat the trend on the main 300-sentence sample; the fresh-sentence follow-up was legacy-only.

Gemma-2-9B shows the same broad RTN pattern: positive FP harm in all 27 languages and a four-bit median ratio of 1.586 [1.238, 2.264], with intervals resampling languages. Its mean ratio is 2.479 [1.781, 3.368], while its ratio of mean effects is 1.474 [1.263, 1.937]. Those are different summaries. Paired comparisons do not resolve a 2B–9B size effect. Two defective criteria in the original 9B checklist are documented; they are not counted as independent confirmations. Llama does not meet its separate Gemma-consistency rule.

The comprehension result differs. On held-out Belebele, extra correct-option loss is −0.0001 nats/item [−0.0035, +0.0031] in FP, so no clear K64 effect is established. Baseline accuracy is about 42.6%, a separate measure. Larger control edits have detectable effects, but small K64 effects remain possible. The text-prediction finding therefore does not automatically extend to comprehension.

Sources: [`functional_validation/results/eval/report.md`](functional_validation/results/eval/report.md), [`gptq_extension/RESULTS.md`](gptq_extension/RESULTS.md), [`functional_validation/gemma9b/REVIEWED_SUMMARY.md`](functional_validation/gemma9b/REVIEWED_SUMMARY.md), [`functional_validation/llama/report.md`](functional_validation/llama/report.md), [`functional_validation/results/mc_eval/report.md`](functional_validation/results/mc_eval/report.md).

## RQ3. How well do activation measurements predict whether a feature's output effect is preserved?

**Answer.** Activation correlation provides a small positive prediction gain, but a survival label gives an incomplete picture. At legacy four-bit precision, only 13.2% of the 1,487-feature sample pass correlation 0.9. Yet 99.7% can still be identified by matching their output effects to their own FP feature, and median effect cosine is 0.839 [0.829, 0.848]. Cosine compares the direction of output changes; one means identical direction. The effects have changed, but remain distinctive.

For 1,412 features with reliable reference effects, adding activation correlation to activity and effect-size predictors reduces prediction error by 3.1–4.4% across the six main quantizer/precision conditions. Each test holds out one language at a time; every language-bootstrap interval is above zero. The gain is below the project's 10% practical target, but it is still positive. Simply keeping the same preferred-language label is less informative in the tested classifications.

We also separate activation shrinkage, unexplained residual change, switching off, and change relative to a feature's activation threshold. This audit covers 214,506 active language–feature pairs, reproduces earlier correlations within 3e−8, and checks the gain/noise identity to about 1e−14. Adding gain and threshold-scaled change after correlation gives further prediction improvements of −0.4% to +1.8%, with all intervals including zero. The extra measures help explain the changes without establishing an added prediction benefit here.

Sources: [`functional_validation/results/f5/report.md`](functional_validation/results/f5/report.md), [`survival_effect_rescue/stage4_gemma_report_2026-10-28.md`](survival_effect_rescue/stage4_gemma_report_2026-10-28.md), [`survival_measurement/stage3/stage3_results.md`](survival_measurement/stage3/stage3_results.md).

## RQ4. How much does the same feature edit change when the model is quantized, and what explains that change?

**Answer.** The edit changes more at four bits, and GPTQ preserves it more closely. On the same eligible feature–context pairs, four-bit output-effect cosines are 0.768, 0.822 and 0.877 for legacy RTN, matched RTN and GPTQ. Relative errors are 0.664, 0.572 and 0.476; lower is better.

We keep the edit fixed while changing the earlier layers, later weights, or both. This separates the changed context, changed downstream response and their interaction. Their vectors add exactly; their lengths and medians do not. For the legacy four-bit group test, median extra harm is 0.848 nats/token with the fixed FP-strength edit, versus 0.263 in FP. Using the quantized model's own feature strength reduces it to 0.433. Context and later-weight terms are 0.310 and 0.104 nats/token in median absolute size. For individual output vectors, the later-weight term is larger. The answer therefore depends on the edit and outcome measured.

Other control directions also change substantially. Their sensitivity improves prediction of legacy four-bit transfer error by 21.5% [12.7, 29.4]; the six-bit interval includes zero, so the original rule requiring utility at both widths is not met. Fresh Wikipedia articles repeat the four-bit quantizer ordering. On that panel, the utility rule holds at both widths for RTN, but only at four bits for GPTQ. GPTQ's six-bit error is near the reference scale. Calibration uses separate Wikipedia articles, so shared domain may contribute to the comparison. Averaging effects across contexts also makes preservation look stronger than comparing individual contexts.

Sources: [`intervention_transfer/results/main/README.md`](intervention_transfer/results/main/README.md), [`intervention_transfer/results/external/README.md`](intervention_transfer/results/external/README.md), [`gptq_extension/results/stage2/transfer_comparison.md`](gptq_extension/results/stage2/transfer_comparison.md), [`gptq_extension/RESULTS.md`](gptq_extension/RESULTS.md).

## What the study establishes

The controlled experiments show that selected language features affect text prediction, that quantizer choice changes their preservation, and that activation changes can coexist with recognizable output effects. The BOS investigation also identifies and corrects a problem in the earlier Gemma input protocol.

SAE dictionaries can mix concepts, split them across features and leave information unexplained. These are shared problems in current mechanistic interpretability. Keeping the dictionary fixed and comparing precise edits with controls still supports the reported effects. More models and dictionaries would test how widely the findings apply.

Most analyses are exploratory; the original held-out Belebele family is confirmatory. The completed evidence is the present study. **Gemma-2-9B GPTQ remains pending and contributes no result.** Historical results and decisions remain available in their original records.
