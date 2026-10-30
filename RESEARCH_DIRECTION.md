# Final research direction

Updated 30 October 2026; evidence through 29 October. The completed work supports a controlled empirical paper on **language-feature intervention preservation under weight quantization**. The paper connects three measurements that should not be treated as interchangeable: activation correlation, language association and the effects of defined interventions.

The [final research questions and answers](reports/FINAL_RESEARCH_QUESTIONS.md) organize the evidence around four questions:

1. How do selectivity and language-resource patterns change after accounting for feature activity and the quantization recipe?
2. What causal effects do language-selective group edits have, and how do those effects depend on task and precision?
3. How well do activation survival and association labels describe preservation of intervention effects?
4. How do activation context, downstream weights and generic sensitivity contribute to intervention drift, and what changes across quantizers?

These are retrospective synthesis questions. They do not replace the original hypotheses or turn exploratory results into confirmatory ones. The failed predictions, nulls, threshold-sensitive outcomes and post hoc analyses are reported with their original status.

The central evidence is substantive: group removal harms next-token prediction in all 27 Gemma-2-2B languages under the specified controls; intervention effects change much less at INT6 than INT4; GPTQ reduces several low-bit changes; activation correlation adds a modest 3–4% to effect-similarity prediction beyond the specified baselines; association labels and additional gain/perturbation metrics provide little additional predictive separation in the tested analyses. The reading-comprehension null and the limited 2.0–2.6% repair by the specified group patch place useful boundaries on the causal interpretation. None establishes that a language group exhausts the model's language mechanism.

The main evaluation uses Gemma-2-2B, its layer-12 Gemma Scope 16k SAE, 27 languages, two corpora for transfer and three quantizers (two RTN recipes and GPTQ). Reviewed original-RTN Gemma-2-9B results extend the scale comparison; Llama-3.2-1B supplies a limited, partly consistent cross-architecture check. Their coverage is reported by experiment rather than implying that every analysis was repeated everywhere. A fixed dictionary and intervention site define the estimand and limit generalization; they do not invalidate controlled comparisons.

Mechanistic interpretability still has unresolved measurement problems: SAE features need not exhaust a computation, activity or labels alone do not certify function, and intervention strength, context and redundancy affect what an edit reveals. The project's matched controls, gain sweeps, crossed states, numerical references and quantizer comparison address several of these problems directly. They support an empirical evaluation of the chosen interventions, with natural-mechanism completeness left open. The [updated source register](reports/research_sources.md) documents work on SAE nonuniqueness, task-grounded validation and divergence between proxy and practical evaluations.

The paper should claim the completed controlled evaluation and its quantitative findings. It should credit direct precedents for survival/function distinctions, fingerprints and patching, and avoid an unsupported first-ever or venue-tier claim. The [dated novelty searches](reports/novelty_assessment_2026-10-26.md) remain useful records, but the current paper is based on the complete evidence available on 29 October rather than an unfinished rescue plan. Publication acceptance is an editorial judgment, not a result this repository can establish.

**Deliverables:** [research paper](paper/research-paper/main.tex), [full report](reports/FULL_PROJECT_REPORT_DRAFT.md), [results overview](reports/PROJECT_RESULTS.md), [current status](reports/CURRENT_STATUS.md). **Pending extension:** Gemma-2-9B GPTQ; no result is assumed. Other quantizers, layers, SAE dictionaries and the unrun arithmetic audit are future work rather than prerequisites for reporting the present findings.

The pre-revision text is preserved in [`reports/archive/reporting_revision_2026-10-29/RESEARCH_DIRECTION.md.txt`](reports/archive/reporting_revision_2026-10-29/RESEARCH_DIRECTION.md.txt). Original plans and decisions remain unchanged.
