# Steering feasibility and literature assessment

28 October 2026, IST. Research/advice only: no model experiments, deployment, or queue changes. This is a targeted scoping review, not an exhaustive systematic review or a priority certificate. Existing working-tree edits were left untouched.

## Decision

A bounded diagnostic pilot is feasible in two days if the current model infrastructure and GPUs are available. A publication-ready new steering method or a general causal account is not a credible two-day promise. Prefer controlled cross-language activation patching before investing in a steering-vector sweep. Do not replace the existing project with a broad language-steering story.

The useful question is whether quantization-induced errors on non-English inputs can be repaired by task-matched English representations beyond same-answer unrelated donors and generic representation repair. This is an extension of existing cross-language patching work, not a new intervention method.

## Project evidence inspected

- `CURRENT_STATUS.md` is partly stale. The GPTQ ledger's latest inspected entry is 28 October 12:05 IST. Phase 1, transfer, functional evaluation/rescue, Belebele and F5 have results; GPTQ Study G and external-panel analysis remain pending in that snapshot. No live cluster status was checked.
- Original Study G already performs graded down/up interventions. Its small loss improvement at INT4 does not establish useful task repair; activation shrinkage did not explain the selected gains.
- GPTQ INT4 mean Belebele accuracy is 0.31847 versus FP 0.42569, so this is a more useful repair regime than legacy INT4 near chance. GPTQ INT6 is 0.42266, leaving little aggregate repair headroom.
- In GPTQ, language-group ablation changes macro accuracy by -0.289 percentage points at INT4, versus -0.014 at FP and +0.216 at INT6. These are macro numbers, not bounds on every language. The group-minus-control correct-option loss effect is also near zero on average.
- The current GPTQ summary describes group rescue as carrying only 2–3% of damage. That is too strong causally. The measured fact is that this particular group patch recovers about 2% of the loss gap (legacy about 2.6%), whereas the whole-residual patch recovers about 41% under GPTQ. These effects are neither additive mediation fractions nor upper bounds on causal responsibility. `PROJECT_RESULTS.md` already contains the more careful interpretation for legacy runs.
- Current scope is principally Gemma-2-2B, layer 12, one SAE, simulated weight quantization with floating-point execution. The Llama check did not meet its consistency rule. The paired arithmetic audit is built but unrun. Adding a vector does not resolve those limitations.

Source files: `PROJECT_RESULTS.md`; `gptq_extension/RESULTS.md`; `gptq_extension/STATUS.md`; `gptq_extension/results/stage3/analysis_gptq_mc/{accuracy.csv,report.md}`; `novelty_assessment_2026-10-26.md`; `../scripts/g0/phase1_func_{run,intervene,tasks}.py`.

### The English premise

Read directly from the GPTQ accuracy CSV, 385 evaluation items per language:

| Language | FP accuracy | GPTQ INT4 | Absolute loss, percentage points |
|---|---:|---:|---:|
| English | 54.81% | 36.36% | 18.44 |
| Spanish | 47.53% | 29.35% | 18.18 |
| Hindi | 37.14% | 29.87% | 7.27 |
| Bengali | 37.92% | 32.21% | 5.71 |
| French | 46.23% | 37.14% | 9.09 |

These are descriptive point estimates, not significance tests. English outperforming a language after quantization does not imply English was better preserved. Per-item English-correct/non-English-wrong donor availability must be counted separately. Cross-language token NLL is also affected by tokenization and should not replace aligned task comparisons.

## Literature evidence

| Primary source | Relevant evidence and limit |
|---|---|
| [ReCoVeR, Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.1056/) | Parallel-corpus language vectors, fixed and trainable steering, 18 languages and three benchmarks. Primarily language-confusion control while retaining task quality; not evidence of quantization repair. |
| [Language Steering for Multilingual In-Context Learning, February 2026 preprint](https://arxiv.org/html/2602.02326v1) | Direct precedent for mean target-minus-source activation vectors. Three instruction-tuned models, 19 languages, MGSM/MSVAMP/XNLI. Uses separate compute/validation/test splits and a substantial layer/strength/position search; only validation-improving configurations reach test. Positive results do not guarantee a single layer on Gemma-2-2B. |
| [Can you map it to English?, EACL 2026](https://aclanthology.org/2026.eacl-long.225/) | Parallel English activation patches correct non-English failures on NLU tasks including Belebele. Crucial control: unrelated English donors with the same correct answer symbol. Middle-layer semantic effects must be distinguished from copying an answer-token bias. Most directly reusable experimental precedent. |
| [Enhancing Low-Resource Language Reasoning via High-Resource Language Feature Transfer, August 2026 preprint](https://arxiv.org/html/2608.30462v1) | SAE-based transfer of task-related features from successful high-resource reasoning into lower-resource inference; source suppression and random/non-task controls. Close overlap with a generic English-feature-transfer proposal. Different models/tasks; no demonstrated result for this project's quantization setting. |
| [DFKI-MLT SemEval-2026 system paper, inspected arXiv version](https://arxiv.org/html/2605.23069v1) | FLORES-derived language steering gives heterogeneous, layer-sensitive cultural-task changes, including degradation. Random controls do not show reliably positive language-vector gains. Useful counterweight to optimistic abstracts. |
| [Steering Under Compression, September 2026 preprint](https://arxiv.org/html/2609.06473v1) | Direct quantization/steering precedent: four 7–9B models, INT8/NF4, sentiment and reasoning length. Separates baseline capability, dose-response and capability cost at matched behavioral effect. Sentiment results are descriptive, INT8 equivalent under their rule and NF4 inconclusive; length has collapse boundaries. Neither inevitable steering failure nor guaranteed repair follows. |
| [Recurrence Is Not Enough, September 2026; arXiv reports BlackboxNLP acceptance](https://arxiv.org/abs/2609.04808) | Most recurring multilingual SAE translation features have weak/inconsistent interventions; one transfers robustly. Relevant negative evidence against equating language association with a task mechanism. |
| [Language-Conditional Dequantization, August 2026 preprint](https://arxiv.org/html/2608.11786v1) | Per-language low-rank corrections for English-calibrated INT3 GPTQ; reports stronger perplexity recovery than accuracy recovery. Already occupies broad multilingual quantization-repair territory. Our multilingual calibration, INT4 setting and fixed-vector intervention would differ; these differences alone do not establish significance or novelty. |

No exact completion of a multilingual quantization-specific English-donor patch factorial was identified in this search. This is a bounded search outcome, not proof of absence. The broad constituent ideas are already well explored.

## Interpreting the proposed vector

At a specified layer/position, estimate v(en→L) = mean[h_FP(x_L) - h_FP(x_en)] on parallel training examples. Adding it to h_Q(x_en) is a valid intervention, but only approximates a language-dependent displacement. It need not recreate the target sentence's state or preserve the English computation. A language-identity direction is not automatically a competence direction.

Three different experiments must stay separate:

1. English input plus en→L vector: may change output language or internal language context; does not by itself test understanding of L inputs.
2. L input plus an L→English vector: may encourage English-associated processing, but a mean shift is not equivalent to semantic alignment or translation.
3. L input patched with the matching English item's activation: an instance-specific donor experiment, not a universal steering vector or deployable translation-free repair.

The existing literature supports trying these as distinct interventions, not inferring their equivalence. Success is causal evidence for the specified edit's effect; it does not alone show that natural quantization damage acts through that direction.

## Bounded pilot recommendation (not a frozen implementation plan)

First count baseline errors and available donors. Use FP and GPTQ INT4, with INT6 as a near-preserved control if affordable. Start with English and two target languages selected for usable baseline competence and adequate paired errors, using only a development partition for selection. Do not select languages from final patching results.

Use parallel multiple-choice prompts with the same answer alphabet and mapped item IDs. At layer 12 and a consistently defined final prompt position, compare: no edit/sham; FP same-language donor; quantized English donor; FP English donor; unrelated English donors matched for answer symbol and donor correctness; and perturbation-scale controls. An adjacent layer can be a prespecified development check; a single-layer failure does not refute all steering.

The current donor code assumes identical token rows in donor and recipient. Cross-language all-token replacement is therefore not a drop-in change. Extracting one shared prompt position avoids token-by-token alignment but still requires explicit item alignment and tests. Same-answer donor controls and alternative answer-order evaluation help detect direct answer copying.

Report accuracy and correct-answer margin on all locked evaluation examples; separately report repair and newly broken answers. Analyze FP-correct→Q-wrong examples as a named diagnostic stratum, not the only endpoint. Compare patch benefits between FP and Q recipients. That interaction distinguishes quantization-dependent responsiveness from ordinary cross-language help, but is not on its own a formal mediation decomposition.

If genuine English donor content helps beyond controls, a second-stage fixed vector is motivated. Estimate it on separate parallel data, freeze the layer/strength using development data, and test both signs and several random directions. Compare the same FP vector in FP/Q and a re-estimated Q vector in Q; distinguish changed direction from changed scale. If steering helps FP and Q equally, report general steering benefit, not demonstrated quantization repair.

Use passage-level disjoint extraction/development/test splits across languages; bootstrap by passage with aligned resampling across conditions. Previously inspected evaluation data remain exploratory. Aim for a few hundred genuinely unused items per language for a large-effect pilot; modest changes can remain unresolved. If no untouched comparable data are available, acquiring/validating them is an additional schedule risk.

Stop this branch if matched English donors do not beat the same-answer unrelated controls at the frozen settings, if apparent gains only change output language, or if no adequate quantization-induced error stratum exists. A null here limits this pilot, not the entire class of language interventions. If generic same-language FP repair works but English-specific repair does not, prefer the generic representation-damage interpretation.

## Cost and timing

The expensive part is valid evaluation, not adding a vector. Existing GPTQ Belebele jobs took about 2 h 13 min to 2 h 28 min per state for 18 languages and their intervention configurations; Study G takes about three hours per state. Those jobs include setup and cannot be linearly scaled as a guaranteed runtime.

Engineering estimate, not a benchmark: reserve roughly 8–20 GPU-hours for a two-target-language pilot with a small control matrix, donor extraction, setup and rerun allowance, plus one working day for implementation/validation and the remaining day for runs and analysis. Two available 11 GB GPUs make that plausible, subject to the current queue and clean data availability. A large layer × strength × language × precision × model sweep is incompatible with this estimate. No dollar cost is quoted because the cluster charging model is unknown. Dry-run throughput must replace this estimate before committing compute.

## Search protocol and log

Scope: primary multilingual steering, English activation patching, quantization/steering interaction, multilingual quantization repair, and contrary intervention evidence. Search date 28 October 2026. No lower date restriction; inspected sources principally 2025–2026. Discovery: web search across arXiv/ACL and alphaXiv. Secondary summaries/repositories used only to locate primary papers; unrelated audio, antenna steering, VQ-autoencoder-only and social-media claims excluded from substantive evidence.

Exact query families used:

- `language steering vectors quantization multilingual activation steering English language vectors`
- `multilingual steering English latent language activation patching language vectors reasoning`
- `quantization activation steering vectors robustness quantized models steering paper`
- `"language steering" negative results multilingual`
- `"steering" "quantization" "vectors" arxiv multilingual`
- `"activation steering" "quantization" robustness paper`
- `"steering vectors" "quantized" language models paper`
- `"quantization" "cross-lingual" "steering"`
- `"quantization" "English" "activation patching" multilingual`
- Title searches for the feature-transfer, SemEval and Language-Conditional Dequantization papers.
- alphaXiv question: has English-to-other-language activation steering or cross-lingual patching been tested as a causal explanation/repair for multilingual performance loss under weight quantization? Keywords: steering, quantization, multilingual, English, activation patching. Fourteen leads returned; relevant sources deduplicated by arXiv ID/title against web results.

Inspected primary abstracts and relevant methods/results sections; the EACL patching PDF's answer-symbol controls were inspected directly. Not every paper received a full replication or statistical audit. This record supports a research decision, not an exhaustive novelty claim.

## Current reporting context, 30 October 2026

This dated assessment is retained as a historical record; its old pending-work and publication-priority statements are not the current project status. [Current synthesis](FINAL_RESEARCH_QUESTIONS.md), [research paper](../paper/research-paper/main.tex), [current status](CURRENT_STATUS.md). The completed paper reports a controlled evaluation across three Gemma-2-2B quantizers, task/control checks and scoped scale/architecture extensions. It credits direct precedents and makes no first-ever or acceptance claim. A limited model/layer/SAE scope bounds generalization rather than invalidating controlled effects. Positive associations, supported intervention contrasts and informative nulls all belong in the final account. Original predictions and their outcomes are unchanged.

The 9B original-RTN analysis was reproduced from raw files and reviewed (D-127); GPTQ remains pending, matched RTN is unrun. Belebele's primary interval is in correct-option NLL, not accuracy points. Continuous survival adds about 3–4% predictive improvement beyond baselines, below the 10% convention but not zero. F5 effect changes can coexist with high own-identity retrieval. These later clarifications govern current interpretation where the dated prose differs.
