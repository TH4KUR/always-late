# Does compression forget language? What we found, in plain words

**Final synthesis update, 30 October 2026; evidence through 29 October.** Read the [final research questions and answers](FINAL_RESEARCH_QUESTIONS.md) and [research paper](../paper/research-paper/main.tex) for the completed framing. The work supports controlled intervention findings and a measurement-validity contribution. A finite model/SAE scope limits where the findings can be generalized; it does not make the measured effects untrustworthy. The original hypotheses, nulls and failed criteria are retained. The questions used to organize the final paper are retrospective.

**Evidence base.** Gemma-2-2B with a layer-12 Gemma Scope dictionary of 16,384 features; 27 FLORES-200 languages; original RTN, matched RTN and GPTQ; a six-language Wikipedia transfer panel; reviewed Gemma-2-9B original-RTN scale results; and a limited Llama-3.2-1B check. Each analysis below states its own coverage. The original sections and appendix A1–A10 describe legacy RTN unless stated otherwise; [the GPTQ extension](gptq_extension/RESULTS.md) and appendices A11–A16 compare the three quantizers. Quantization is simulated weight-only rounding with floating-point execution. Gemma-2-9B GPTQ remains pending and supplies no evidence yet.

**Reading the results.** This page summarizes existing result files; it introduces no new experiment or statistical analysis. Generated appendix tables are retained. Definitions, frozen inputs, decisions, logs and dated plans remain the record of what was specified and when. The pre-revision page is [preserved](archive/reporting_revision_2026-10-29/reports/PROJECT_RESULTS.md.txt).

---


> **Update 2026-10-28 10:32 IST (GPTQ extension):** the same experiments were rerun with two other ways of rounding the weights (matched RTN, GPTQ) and compared with the results below: [`gptq_extension/RESULTS.md`](gptq_extension/RESULTS.md). The text below is unchanged and describes the original (legacy RTN) runs.
>
> **Update 2026-10-28 22:57 IST (survival versus effect):** a post hoc analysis asked whether the continuous activation correlation of a feature predicts how well the effect of switching it off survives rounding, beyond simple baselines: on Gemma-2-2B (three ways of rounding) and on Llama-3.2-1B the answer is "only weakly, below the pre-set 10 % target": [`survival_effect_rescue/README.md`](survival_effect_rescue/README.md) (D-110 to D-114).

## 1. The short answers

### What does the completed study establish?

The results support four connected answers. They are narrower than a complete explanation of multilingual computation, but they are reportable empirical findings.

1. **Activity and quantization recipe matter for survival.** The original corrected tests do not establish a robust selective-feature penalty. Activation-scale matching substantially changes the selective/control contrast, and GPTQ greatly reduces the INT6 resource-tier gap. Resource tier is observational, so these analyses do not identify an effect of changing training resources.
2. **Language-selective groups have measurable causal effects under the specified edits.** Removing the top-64 group harms the language's next-token loss beyond the chosen controls in all 27 Gemma-2-2B languages. The reading-comprehension test does not establish a corresponding group-specific effect on its primary correct-option NLL endpoint; the accompanying accuracy changes are small. The task distinction is a finding; it does not imply that the group is either the entire language mechanism or irrelevant.
3. **Preservation of activity and labels is an incomplete guide to preservation of intervention effects.** Continuous activation correlation is informative: it adds about 3–4% to the specified held-out prediction beyond baselines, with positive intervals. That gain falls below the project's 10% practical target. Binary labels and the added gain/perturbation metrics do not provide a reliable substitute for measuring effects in these analyses.
4. **Intervention drift depends on context, weights, edit strength and generic sensitivity.** The crossed experiment estimates these contrasts directly. At INT4, control directions share much of the drift, and GPTQ reduces its size. The context/weights comparison depends on whether the edit uses fixed full-precision or native amplitudes and on whether the endpoint is scalar loss or an effect vector.

Mechanistic interpretability has unresolved measurement problems that also matter here: a fixed SAE may split, merge or omit relevant computations; labels do not certify causal function; and redundancy or an unusually strong edit can change the reading of an ablation. Our matched edits, numerical references, gain sweeps and crossed states directly address parts of this problem. They do not certify a complete natural circuit. Broader model and dictionary studies would test transfer of the findings rather than determine whether the present controlled contrasts exist.

**Evidence status.** Belebele is the designated confirmatory family in the course report. Phase 1 has proposal-specified corrected tests, but external registration was not verified (V-10, D-029). Later protocols often fixed predictions before their runs while using data examined earlier; they remain exploratory. Survival/effect recovery and measurement-validity work are post hoc relative to the earlier project, with their later analysis rules frozen before the corresponding new outputs. A retrospective final question does not change those labels.

### What do the null results tell us?

A non-significant result means the test did not establish the specified effect. It does **not** by itself establish zero or an upper bound. Report the effect estimate and interval. A useful bound on practical importance requires an appropriate interval relative to a stated meaningful-effect threshold, or an equivalence test. These conclusions concern the tested intervention and population, not every hidden mechanism.

| what we tested | what came out | what this supports | what it does *not* rule out |
|---|---|---|---|
| Do language-specific features survive 6–7 bit rounding worse than matched ordinary features? (Phase 1, fixed in advance) | INT7: −0.28 points; INT6: −1.44 points (range −2.65 to −0.13); after adjusting for running three tests, p = 0.51 and 0.13 | No robust selective penalty under the prespecified corrected tests; the reported intervals describe the tested contrasts, not a universal bound | A small gap. Also: with stricter "how language-specific" cut-offs the gap grows (−7 points) but turns into +4.6 points once features are matched on how strongly they fire. The contrast is strongly affected by activation scale; this matching analysis does not identify the causal origin of the scale differences |
| Does the gap between language-specific and matched features get worse for lower-resource languages? (Phase 1, fixed in advance) | Slope across tiers −0.005; equivalence test passes (no tier effect bigger than 0.02 in correlation) | Evidence against a tier effect on the selective-versus-matched gap larger than the stated equivalence margin | That low-resource languages are hurt more *overall*: at INT6, 69 % / 91 % / 95 % of a language's active features survive for low / mid / high. The observed gap attenuates after accounting for activity scale; lower-resource languages have weaker measured activations in this setup |
| Does removing a language's feature group hurt a reading-comprehension test (Belebele)? (fixed in advance) | Primary correct-option NLL effect −0.0001 [−0.0035, +0.0031] nats/item; accompanying accuracy 0.4257 untouched, 0.4255 group removed, 0.4251 controls (18 languages, 385 questions each) | No established positive group-specific NLL effect under this edit. The NLL interval is not an accuracy bound. Large random or other-language control sets reduce accuracy by 1.2 to 6.0 points (the random 2,860-feature pool: 2.2 points), so the edit path registers damage; the pre-set positive-control criterion was not fully met. Even removing up to 1,719 features of the language group (everything we had) gives 0.4257 → 0.4209 | Small accuracy effects or other interventions/tasks. A random set of 256 features costs about 0.7–0.8 accuracy points; the NLL test does not establish accuracy equivalence |
| Does the language tag of a feature ("this feature is a Yoruba feature") predict that the feature matters for function? | Association status gives AUROC 0.51 (0.5 = chance discrimination for this event) | Little predictive separation from this tag in the measured population; not a universal impossibility result | That features keep their function (we show the opposite is hard to claim too) |
| Is the INT4 "repair" (turning the group up a little) explained by the group shrinking? (Study G extension D) | Correlation −0.17, p = 0.80 | No evidence for the proposed shrinkage–optimal-gain relationship in this analysis; p = 0.80 alone does not disprove the mechanism | Other mechanisms, or a relationship this analysis cannot resolve |
| Does a second model show the same INT4 blow-up? (Llama) | The ratio is 0.86, not 1.54 as on Gemma | The evaluated Llama setup does not show the same amplification; universal amplification is not established | That it is real on Gemma: it still is |

Together these results do not support a general selective-destruction account. They identify activity scale, baseline language margin and quantizer choice as relevant to the observed survival patterns, while leaving the exact causal origin of those scale differences open.

### What causal claims do the experiments establish?

**We can identify effects of specified interventions in this model.** Repeating the
study with every SAE, layer, model or quantizer is not a prerequisite for that.
Those repetitions test generalization. Different claims require different evidence:

- **Feature-group edits:** compare the same input with and without the defined
  K64 removal while keeping the model, arithmetic and other settings fixed. The
  increase in next-token loss is a causal effect of that edit. The excess over
  the specified control edit is a difference between two intervention effects.
  It does not establish that these are the only features the model uses for a
  language, or that every part of the effect is specific to language.
- **Weight quantization:** applying RTN to the selected weights and rerunning the
  same inputs identifies the effect of that weight transformation under the
  specified execution settings. Changing many weights at once does not prevent
  estimating the total effect of that controlled transformation.
- **Prefix/suffix hybrids:** the crossed experiment explicitly holds one part
  fixed while changing the other. Its contrasts identify effects in those
  engineered model states, including interactions, for the stated edit and
  endpoint. We should not say that holding the other parts fixed is impossible.
- **Feature mediation:** whether the total quantization effect occurs *through*
  a natural language mechanism is a stronger question. Our FP-donor group patch
  recovers 2.6% of the INT4 loss gap under that patching operation. This measures
  the repair achieved by the patch. It is not an upper bound on the group's
  causal contribution, because intervention definition and interactions matter.
- **Language resource tier:** this is an observational language characteristic,
  not an experimentally manipulated training-resource treatment. Its association
  with outcomes does not identify the effect of changing training resources.
- **Null interventions:** a small or non-significant average change concerns the
  tested edit, metric and inputs. Only sufficiently precise intervals or a
  suitable equivalence test support a bound; it does not prove that the language
  or computation has no causal mechanism.

One model and one SAE restrict the scope of these statements. They do not erase
the causal contrasts. A causal effect, an explanation of the model's natural
computation, and generalization to other systems are separate achievements.

### What did the second-model check say?

**Llama-3.2-1B with its own dictionary (a TopK SAE, 12 of 16 layers deep): the pre-set verdict was "not consistent".** (`reports/functional_validation/llama/report.md`, `llama_robust/report.md`; D-087, D-090.)

- Removing a language's group *does* hurt that language, but much less than on Gemma: median extra error 0.048 against 0.262 on Gemma. It is above zero in 21 of 27 languages; the rule asked for 24. (23 of 27 if the median of the control draws is used.)
- Flat through INT6 as on Gemma: retention 1.010 at INT8 and 1.024 at INT6 (met).
- **The INT4 blow-up is not found.** Gemma's median ratio at INT4 is 1.54; Llama's is 0.86 [0.74, 1.03]. (The pre-set measure, a mean, gives −3.57 because one language has a tiny full-precision effect; the median is the fairer number and says the same thing.)
- Graded dose ordering holds in 20 of 27 languages at full precision and 15 of 27 at INT4 (Gemma: all 27).

What this check can and cannot say: the model family, the dictionary type, the dictionary training, the depth and the tokenizer all differ, and this dictionary calls 91 % of its features language-specific (Gemma Scope: about 75 %) with a smaller control pool (898 against 2,860). So "not consistent" does **not** mean "the Gemma findings are wrong", and it is not a replication. It means we cannot claim the Gemma picture holds on another model.

**Gemma-2-9B with Gemma Scope** supplies a same-family scale extension. Its original-RTN analysis was reproduced from raw per-state files and reviewed (D-127): positive group-removal effects in all 27 languages; retention about 1.00 at INT8/INT6; INT4 median language ratio 1.586 [1.238, 2.264]. The corresponding 2B/9B paired comparison does not resolve a scale difference. The imported P3a inequality has the wrong sign, and P3b cannot fail; these prediction defects are reported rather than counted as confirmation. See [the reviewed summary](functional_validation/gemma9b/REVIEWED_SUMMARY.md). GPTQ on 9B is pending; matched RTN and the full measurement study were not run on 9B.

### What about the downstream task?

The only confirmatory downstream test was **Belebele**, a multiple-choice reading test in 122 languages. We used it in 18 of our 27 languages (the other 9: they did not meet the pre-set baseline-accuracy coverage criterion). Each language: 385 questions. The primary frozen question was whether removing the language's group increases the correct-option negative log-likelihood (NLL) more than matched controls; accuracy is an additional endpoint. (`reports/functional_validation/results/mc_eval/`, `mc_ksens/`, `mc_posctrl/`, `promptcheck/`; D-047, D-050, D-062, D-069, D-072, D-086.)

- **The primary frozen criterion is not met.** The excess correct-option NLL is −0.0001 [−0.0035, +0.0031] **nats per item**. This is not an accuracy interval. Accuracy is 0.4257 untouched, 0.4255 group removed and 0.4251 controls. Because no language has a clearly positive full-precision NLL effect under the rule, retention statements S1–S5 are undefined. Exploratory legacy NLL effects at INT5 and INT4 have intervals excluding zero in opposite directions (−0.0066 [−0.0102, −0.0031] and +0.0042 [+0.0003, +0.0081]); neither rescues the failed FP criterion or defines its retention ratio.
- **The edit path registers damage.** Large random or other-language control sets lower accuracy by 1.2–6.0 points (the full random pool: 2.2 points); the pre-set positive-control target was not fully met. A do-nothing edit changes it by exactly 0 (`results/mc_posctrl/`).
- **The whole-group sensitivity does not establish the primary effect either.** Removing up to 1,719 selective features gives a similarly small NLL contrast (`results/mc_ksens/`): 0.4257 → 0.4209, controls 0.4226.
- **Not because the question wording was weak.** Our prompt scores *higher* than the standard lm-evaluation-harness prompt in all 18 languages (`promptcheck/`). (Absolute numbers depend on the prompt by about ten points, so do not compare with published Belebele numbers.)
- **The legacy INT4 reading result is limited by its baseline.** Accuracy is 0.236 (chance 0.25), versus 0.387 at INT5 and about 0.42 at INT6/INT7/INT8. Matched RTN and GPTQ improve INT4 accuracy to 0.291 and 0.318; the small group-edit accuracy effect also persists there, so the cross-quantizer null cannot be attributed solely to the legacy model being at chance.
- **Meanwhile the same removal raises next-word error in 27 of 27 languages.** This establishes a task-dependent contrast: clear loss effects under the tested group edit, without a resolved group-specific reading-accuracy effect at this sample size.

*Why* the two differ is **not explained**. Guesses, none tested: (a) reading questions can be answered through many routes, so losing 64 features at one layer is absorbed; (b) language identity is easy to recover from the context; (c) a test at 43 % accuracy with 385 questions is a blunt instrument. We did not run the experiments that would separate these.

### What remains unresolved?

The controlled edits establish their effects under the evaluated execution settings. They do not uniquely identify a natural language circuit or partition the total quantization loss among features. Small transfer differences close to the INT16 reference are harder to interpret because that reference itself includes a small weight perturbation and floating-point effects. The unrun precision audit is relevant to such comparisons. Model, layer and SAE coverage bounds generalization; three quantizers, two transfer corpora and the 9B/Llama checks provide evidence within their stated coverage rather than a universal guarantee.

---

## 2. Words used (simple)

| word | what it means here |
|---|---|
| **feature** | one of the 16,384 directions of the dictionary (SAE) that reads layer 12. Each fires on some words and stays quiet on others |
| **language-selective feature** | a feature that fires much more on one language than on others (we pick the top 64 per language for the group tests, "K64") |
| **matched control** | features chosen to fire as often (and often as strongly) as the group, but not language-specific; the fair comparison |
| **INT8 … INT4** | weights rounded to 8 … 4 bits. INT16, INT12 = almost no change ("rounding only"). INT0 = put the original weights back (a check that nothing else changed) |
| **survival** | a feature "survives" if its activation, over many words, still correlates at least 0.9 with its full-precision self |
| **association retention** | does the feature still count as "belonging" to its language after compression? (a different question from survival) |
| **zero-ablation** | set a feature to zero and see what happens |
| **nats per token** | how surprised the model is by each word, on average. Lower is better. For scale, its normal value on this text is about 3. A harm of 0.26 is roughly 9 % worse |
| **net harm / effect** | harm from removing the group minus harm from removing matched controls. "How much more the language features matter than ordinary ones" |
| **retention (function)** | the effect in the quantized model divided by the effect in full precision. 1.0 = same; above 1 = bigger; below 1 = smaller |
| **numerical reference / floor** | the INT16 RTN comparison: a small actual weight perturbation plus execution effects. Differences close to it need cautious interpretation; it is not a pure arithmetic-noise measurement |
| **confidence interval** (the numbers in brackets) | an uncertainty interval under the stated resampling or model assumptions; its units and sampling level matter |
| **pre-set / confirmatory** | pre-set means specified before the relevant outcome; confirmatory status additionally depends on prior data use and the analysis family. Exploratory effects remain evidence with their stated status |
| **cosine** | how similar two directions are: 1.0 = same, 0 = unrelated |
| **context (C), weights (W), interaction (I)** | the transfer study's three parts: the activations reaching the edit changed (C), the layers above the edit changed (W), and both changed beyond the sum of the two (I) |
| **equivalence test** | a test that shows an effect is *smaller than a stated size*, instead of only "not found" |
| **dose-response** | how the effect changes as we change the amount: remove 1/3 of the group, 2/3, all of it, or turn it up |
| **BOS** | the special "start of text" token. Without it the model reads the text badly; we add it and leave it out of every statistic ("masked") |
| **perplexity** | the usual measure of how well a language model predicts text (lower is better); the loss in "nats per token" is its logarithm |
| **Holm** | an adjustment of p-values because three tests were run at once |
| **AUROC** | how well one number separates two groups: 0.5 = not at all, 1.0 = perfectly |

---

## 3. The story in order

| when (2026) | what we did | what we found, in one line |
|---|---|---|
| 28 Sep | Built the first pipeline: Gemma-2-2B + Gemma Scope dictionary, rounding, survival statistic; first sanity check (Kannada against English); first bigger run ran out of memory (another process, not a code bug) | The tools work; nothing scientific yet |
| 29 Sep – 2 Oct | Team plan; re-run at 20,000 words per language on a big GPU | First numbers, later retired |
| 15 – 16 Oct | Audit of our code and of the earlier paper's code (Duan et al.). Our first pipeline disagreed with theirs, so we used their own code | **Their published table is reproduced with their code** (job 23472). Our first pipeline was retired and its outputs are not used |
| 16 – 19 Oct | Phase 1: 27 languages on the university cluster ("Ada"), the three pre-set tests, many follow-ups, a clean-room recomputation | **No pre-set test rejects.** The result depends on whether a start token is added to the text (section 4.1). Recomputation: 89 of 89 files identical |
| 21 Oct | **Association retention:** how many "language-tagged" features keep the tag | 98 % at INT8, 71 % at INT4; mostly features that were borderline to begin with |
| 21 – 23 Oct | **Functional study F1–F5:** do the features matter, does it survive quantization, where is the damage, does the tag predict it, is a feature's effect the same after compression? | Removing a group hurts in 27 of 27 languages; the dependence is kept to INT5 and increases at INT4; the group patch repairs little of the loss gap; much INT4 drift is shared with controls |
| 22 – 23 Oct | **Belebele** (downstream, confirmatory) | **Null** |
| 24 – 26 Oct | **Study G:** turn group features down and up, per language, across the precision ladder | Graded, specific and consistent; a clear step at INT4; a tiny "repair" at INT4 |
| 26 Oct | **Llama-3.2-1B check** (second model) | "Not consistent" |
| 26 – 27 Oct | **Intervention transfer:** a fixed edit, quantized in two parts (below and above the edit) | INT6 effect almost kept (cosine 0.98); INT4 effect rotated (0.77); mostly a generic reaction |
| 27 Oct | Four extra analyses of the transfer study (recovery analyses R1–R4); a last test ("precision audit") was written and tested but **never run** | Details in 4.11. User put the project on hold at 09:50 on 27 Oct to finalise |

---

## 4. Study by study

Each block: the question, how we tested it, what came out, what it means, and what it does not mean. Numbers in brackets are 95 % ranges.

### 4.1 Copying the earlier paper, and a hidden catch (Phase 1 start)

**Question.** The earlier paper (Duan 2026) showed on English text that after INT6 rounding the model's perplexity barely moves while only about 51 % of the active features keep their activation pattern (a "blind spot" of ordinary quality checks). Our project asked whether that blind spot hits language-specific features, and low-resource languages, harder (proposal questions RQ1–RQ3 in `anlp_revised.txt`).

**What we did.** We first ran our own pipeline and found it disagreed with theirs, so we used *their vendored code*. It reproduced their published table (INT7: 81.19 % against their 81.26 %; INT6: 51.15 % against 51.30 %), on WikiText.

**The catch.** The model is run on text *without* a special start token (BOS). Without it, the model is confused: perplexity on that text is 458. With a start token added and ignored in the statistics ("masked"), perplexity is about 16 and INT6 survival on WikiText is about **97.5 %** (not 51 %). So the published survival numbers depend on the input format. We therefore use the masked regime as the main one and keep the other as history. (`reports/phase1_reproduction.md`, `issues.md`, anomalies A-029, A-030.)

**What it means.** The survival numbers in that style of paper are sensitive to a technical detail of the input. That is a warning about how to read them, not a claim that anyone's result is wrong.

### 4.2 Phase 1: do language-specific features break first? (three pre-set tests)

(`reports/phase1_summary.md`, `reports/phase1_results.md`, `reports/phase1/`.)

**Pooled survival** (masked, equal words per language): INT8 100 %, INT7 99.97 %, INT6 94.9 %, INT5 59.0 %, INT4 13.75 %. INT16: 100 %. Perplexity changes: +0.08 %, +0.40 %, +2.67 %, +11.5 %, +104 %.

**The three tests, fixed before the run** (language-selective features against matched universal ones; Holm-adjusted):

| test | result | adjusted p |
|---|---|---|
| T1: INT7, selective minus control survival | −0.28 points [−0.63, +0.13] | 0.514 |
| T2: INT6, same | −1.44 points [−2.65, −0.13] | 0.128 |
| T3: gap as a function of resource tier | slope −0.0048 (high minus low −0.008) | 0.990 |

None rejects. An equivalence test says any tier effect is inside ±0.02 correlation.

**What happened when we looked closer (all after the fact, so exploratory):**
- *Stricter definitions of "selective".* The gap grows: INT6 −1.0, −2.1, −7.2 points at thresholds S > 2, 3, 5 (the last two survive adjustment).
- *Then we matched on how strongly the feature fires.* Selective features fire more weakly than the controls. With strength matched the gap **reverses**: +3 to +5 points at INT6. So "selective features die more" is really "weakly firing features die more".
- *Resource tier.* At INT6, 69.4 % / 90.6 % / 95.1 % of active features survive for low / mid / high. This is a real gap, but when we account for activation size the slope across tiers goes from +12.3 points to −0.8 (p = 0.67). It is correlation only.
- *Does it hurt the language?* Across the 27 languages, languages whose selective features are damaged more also lose more next-word accuracy (rank correlation +0.46 at INT6, +0.54 at INT7). But overall damage and selective damage move together, so we cannot say the selective part is responsible.
- *Broken regime.* Without the start token, the model's own error is huge and the sign of the gap flips (+9 points at INT7). That instrument is not valid (A-029).
- *Checked.* A clean-room recomputation of all files: 89 of 89 identical (`reports/phase1_reproduction.md`).

**What it means.** The headline idea, "compression selectively destroys language-specific features, hitting low-resource languages hardest", is **not supported** in the main regime. The size of a feature's activation explains most of what looks like selectivity.
**What it does not mean.** That language features are safe: the language-specific class used here was broad (74 % of supported features count as selective at the pre-set cut), so this tests "most features against a minority", not "one-language features against the rest".

### 4.3 Do language tags survive? (association retention)

(`reports/association_retention/`; D-035, D-066, D-067.) Question: among the 6,663 features that carry a language tag in full precision, how many still carry it after rounding?

Retained: **98.1 % (INT8), 97.5 % (INT7), 94.7 % (INT6), 89.1 % (INT5), 70.8 % (INT4)**; the rounding-only floor (INT16) is 99.2 %. Lost features: 124, 168, 356, 726, 1,945.

**Why they are lost.** Mostly because they were borderline: how far the feature already sat from the tagging line in full precision predicts the loss with AUROC 0.90 (INT6), 0.81 (INT5), 0.75 (INT4). Only at INT4 do clearly tagged features lose their tag too. But that margin hardly predicts whether the feature's *function* changes (AUROC 0.58 at INT6). **So the tag changes mostly because of a technical boundary, not because the feature's job changed.** Per-language numbers: appendix A3.

### 4.4 Do the features matter, and does that survive compression? (functional study F1, F2)

(`reports/functional_validation/results/eval/`; D-041, D-050, D-068.) For each language: the 64 most selective features set to zero, compared with 64 matched controls, on 300 sentences, measured as extra next-word error.

- **F1.** The group matters in **27 of 27** languages. Average extra error 0.704 [0.684, 0.723] nats per token; the median language 0.262 (a few languages have large values and pull the average up: Mandarin 7.6, Lao 1.9, Yoruba 1.6). On the 18 Belebele languages: 0.713 (18 of 18).
- **F2: after rounding.** Retention of the effect: 1.01 to 1.13 through INT5 (rounding-only controls 1.00) and **2.14 at INT4** (median language 1.54). So the dependence is kept, then larger at INT4. At INT4 the quantized model is already damaged, so this inflation is not a "stronger dependence" in a simple sense.

These edits establish that the selected group contributes to next-token loss under the specified intervention and matched-control design. They do not establish an exclusive language circuit; the contrast can also depend on feature size and remaining control imbalance. The feature selection and the test sentences come from the same source family (FLORES), and the test sentences had been seen before, so this is exploratory. Per-language table: appendix A4.

### 4.5 How much does the specified patch repair? (F3)

(`results/rescue/`; D-058.) We took the INT4 model and put back the full-precision activations of the group. This patch recovers **2.6%** of the extra loss at INT4; the whole-residual patch recovers about 39%. These are repair effects of different interventions, not additive shares of the damage or upper bounds on causal responsibility. The small group rescue shows that this patch is insufficient to repair most of the loss gap. It does not prove that the group mediates at most 2.6% of quantization harm. Limit: one layer and the specified donor/patch construction. Per-language table: appendix A5.

### 4.6 Does the tag predict function? Is a feature's effect the same after compression? (F4, F5)

(`results/f4/`, `results/f5/`, `results/f5_sensitivity/`; D-059 to D-064, D-066, D-067.)
- **F5.** The effect of one feature on the output words stays within rounding noise through INT8: median cosine with full precision **0.997 (INT8), 0.988 (INT6), 0.961 (INT5), 0.839 (INT4)**; the rounding floor is 0.949. Each feature is still individually identifiable (retrieval 0.997 to 1.000). 1,487 features, stratified sample.
- **But the change is generic.** If we remove the direction of a *different* feature at the same places and strengths, the similarity drops almost as much (0.003 to 0.014 less). Fixing the activation amplitudes changes almost nothing. This control addresses generic model response; the final paper credits related controls rather than claiming priority.
- **F4/F5.** F4 finds small lost-versus-retained group differences at matched activity. F5's descriptive association-status AUROC on the F4 pairs is 0.506 [0.469, 0.543], offering little discrimination for the chosen effect-similarity event.
- **The survival threshold and effect criterion disagree:** about 31% of features meeting the effect-preservation criterion at INT5 fail the activation-correlation threshold. At INT4 about 13% meet the activation threshold, while almost none meet the stringent effect-reference criterion.

The activation threshold and language tag do not replace direct measurement of the intervention effect. The direction controls show that much of the drift is shared with the model's response to other edits; they do not imply that feature effects have become meaningless.

**Functional change does not mean erased identity.** In the legacy F5 INT4 sample, only 13.2% pass the activation-correlation survival threshold, yet the median effect cosine is 0.839 [0.829, 0.848] and own-identity retrieval is 99.7% (matched-control retrieval 99.9%). Many effects change relative to the stringent numerical reference while remaining identifiable. Retrieval and cosine answer different questions; neither alone certifies complete preservation.

### 4.7 The downstream task: Belebele (the only confirmatory family)

Covered in section 1. All numbers: appendix A6 and `reports/functional_validation/results/mc_eval/`, `statements.md`.

### 4.8 Study G: turning the group up and down (exploratory)

(`reports/functional_validation/gain_study/README.md`; D-070 to D-085.) Idea: not only remove the group; scale it to 1/32, 1/16, 1/8 … 0 (down) and up to 1.1, 1.2, 1.4, 1.7, 2.8, 5.7, 11.3 times, in each language, at six precisions (FP, INT8 … INT4), always against matched controls. 27 languages × 6 states × 300 sentences × 3 control draws.

| claim | verdict |
|---|---|
| Down is graded: every step bigger than the one before (27 of 27 languages ordered). Median harm 0.022 / 0.090 / 0.169 / 0.262 at ratio 0.35 / 0.18 / 0.088 / zero | **met** |
| Down is specific to the group (26 of 27 beat controls) | **met** |
| Up is graded and costs (27 and 24 of 27) | **met** |
| Same pattern in every language (27 of 27, rank correlation FP vs INT4 0.94) | **met** |
| Precision changes the dose-response (H-G4) | **narrowly not met** by the pre-set rule (a fall of 0.012 from FP to INT8 is bigger than one error); met with a looser error rule, chosen after the fact |
| Repair at INT4, the pre-set rules (H-G5) | **not met**, under-powered |
| Baseline check (H-G7) | not met |

What the curves say:
1. **A step at INT4.** Zero-ablation net harm across FP, INT8, INT7, INT6, INT5, INT4: 0.262, 0.268, 0.266, 0.271, 0.281, **0.432**. Flat until INT5, then 1.6 times larger (1.9 times at the 1/11 ratio). The activations themselves also shrink by about 15 % at INT4 (1 to 1.5 % up to INT5).
2. **Turning it up costs less at INT4**: 0.81 against 1.21 at 11.3 times.
3. **A tiny repair at INT4 only.** Turn the group up by 10 % and the loss goes *down* by about 0.001 nats per token (about 0.05 % of the loss); at FP, INT5, INT6 the same turn-up costs a little. A second set of 562 fresh sentences per language with five control draws (extension C) agreed at ×1.1 (median −0.00135 [−0.00197, −0.00032], 20 of 27 languages; at ×1.2 the range includes zero, 18 of 27). A shallow minimum near 1.1 to 1.2, gone by 1.4, a cost at 1.7.
4. **Extension D** (is the repair "turn the group up by its own shrinkage"?): **not supported** (rank correlation −0.17, p = 0.80). The best gain per language is a reliable property (split-half reliability 0.92) but the shrinkage does not explain it; what does is unknown.
5. **Extension B** (does it matter which precision you choose the group in?): the 64 chosen features change (only 49 % overlap with the full-precision selection at INT4, 86 % at INT6) but the effect curves coincide (INT4 zero-ablation 0.452 against 0.477). So choosing the group in full precision does not limit the INT4 results.
6. The up-side effect is not language-independent (the harm at 11.3 times as a multiple of the zero-ablation harm runs from −8.7 to 72.9).

**Means:** the dependence on precision is a **threshold at INT4**, not a gradual slide. **Does not mean:** a usable repair (a thousandth of a nat) or a mechanism. Per-language: appendix A7.

### 4.9 The second model (Llama-3.2-1B) and the bigger Gemma

Covered in section 1. Per-language numbers: appendix A10. **Current 9B state (D-127): original-RTN results have been reproduced and reviewed (section 4.14); the GPTQ run is pending, matched RTN is unrun, and prediction-definition defects remain disclosed.**

### 4.10 Intervention transfer: context or weights? (main matrix + outside text)

(`reports/intervention_transfer/results/main/`, `external/`; D-088, D-094.) **Design.** Take one fixed edit (remove a feature's direction by the full-precision amount) and apply it to four combinations: full-precision or quantized layers below the edit (the "context" = what reaches the edit) × full-precision or quantized layers above it (the "weights"). The change of the edit's effect, in loss, is split into C (context), W (weights) and I (interaction). Also measured at the level of the output-word effect (a vector). Main set: 404 features, 27 languages, FLORES. Outside check: Wikipedia text, six languages.

**Group level (net harm in nats per token, median over languages)** (main):
- Full precision 0.263. INT6: nothing clear: |C| 0.0091 and |W| 0.0092 (rounding alone 0.0005).
- **INT4:** the same edit now harms 0.848; with the model's own scaled activation 0.433. |C| 0.310 [0.258, 0.334], |W| 0.104 [0.094, 0.122], |I| 0.071; C bigger in 19 of 27 languages. **[Note 2026-10-29: the same 19 of 27 occurs at INT16 (independent audit), so the count is not evidence for context dominance; only the interval of the median difference is (D-094).]** The pre-set expectation "context dominates" (A3): supported at INT4, inconclusive at INT6, **overall mixed**.
- Outside text (six languages): INT4 C 0.133, W 0.073 (supported), **INT6 reversed** (W bigger).

**Effect-vector level (2,203 eligible items of 235 features):**
- Error of the effect after quantization: 17 % of the effect at INT6 (cosine 0.984; about two numerical floors), **66 % at INT4** (cosine 0.768): the effect is **rotated, not shrunk** (size ratio 1.09). Almost all the error (97–99 %) points at right angles to the original effect. Using the quantized model's own activation barely helps (97 % of the error remains).
- Here the **weights** part has the bigger norm (0.145 against 0.100 at INT6, 0.560 against 0.359 at INT4): the opposite order from the loss-level result. The two measure different things (a vector length against a signed loss), so both are reported.
- Specificity is weak: the real direction's error is about the same as for swapped or random directions (+0.019 [+0.009, +0.032] at INT4). The error is largely **a generic reaction of the model**.
- The averaged fingerprint of a feature over its contexts overstates how well a single context transfers (in all 404 features at INT4).

**Can we predict the error for features we held out?** **[Note 2026-10-29: "held out" means held out by feature. About 88 % of the held-out documents are also in the calibration data; on 136 items from 65 unseen documents the INT4 gain is 6.73 % (no interval) (`novelty_assessment_2026-10-26.md`).]** The "generic sensitivity" (how much the error of the *other* directions at the same place moves) reduces the held-out error by **21.5 % [12.7, 29.4] at INT4** (rule met) and **10.9 % [−1.2, 21.4] at INT6** (not met), so the pre-set overall verdict on the main set is **not met**. On the outside text both states are met (19.4 % and 23.4 %): overall **met**. But the same predictor also predicts the rounding-only error (23.4 % in the main set, 28.9 % outside), so it measures **how noise-prone a context is**, not something specific to quantization. Survival numbers and gradient attribution add nothing at INT4.

**Means:** at INT6 an intervention's effect is almost kept (about two noise units off); at INT4 it is rotated; most of that is generic. **Does not mean:** a cheap audit; one language-specific mechanism; a result for deployed integer kernels. Per-language: appendices A8 and A9.

### 4.11 The four recovery analyses (27 October, after the main run)

All post hoc: protocol files were written before the code and each recorded what had been seen before (`reports/intervention_transfer/revision_R1..R4*.json`, D-096 to D-099, `results/recovery/`).
- **R1, FLORES-fitted predictor on fresh Wikipedia text** (1,368 eligible items): held-out error falls 28.2 % [21.3, 35.1] at INT16, 18.6 % [4.9, 29.5] at INT6, **19.9 % [7.8, 31.3] at INT4**. Add a "just use the median" null and INT6 becomes 15.8 % [−2.7, 28.7]: **at INT6 the predictor is not clearly better than the median**. INT16 is predicted as well, so still not specific to quantization.
- **R2, gradient scalars against the real loss change:** correlation 0.63 [0.46, 0.80] on FLORES, 0.53 on Wikipedia; the INT4 shift tracks at 0.58 and 0.66; at INT6 weak; at INT16 none. So gradients are not hopeless for the scalar loss; the earlier null was about the vector target.
- **R3, real direction against swapped direction at the *same* effect size** (only 144 FLORES pairs and 107 Wikipedia pairs qualified, about 2 % of the pairs): the real direction's error is ×1.12 (INT6) and ×1.14 (INT4) that of the swapped one (INT16 ×1.00–1.02). **So there is a modest direction-specific part on top of the large generic one.** The earlier "less rotated" look was the real direction's larger effect size.
- **R4, own activation against fixed edit, per language:** at INT4 the model's own-activation harm is smaller than the fixed-edit harm in **27 of 27** languages (median difference −0.351 [−0.358, −0.293]). The leftover above the weights-only effect is **+0.014 in the median language**, but ranges from −0.78 to +0.23 across languages. So "the harm is that of the weights" is true **for the median language only**.

### 4.12 Planned but not run

- **Precision audit (D-100, R5):** a paired check of whether the INT16/INT4 difference in these edits is real arithmetic effect or batching noise. Fully built and tested locally (476 transfer tests). **Never deployed**: the 27 October hold came first.
- **Gemma-2-9B:** reviewed original-RTN scale results are in this repository; GPTQ remains pending. **Current 9B state (D-127): original-RTN results have been reproduced and reviewed (section 4.14); the GPTQ run is pending, matched RTN is unrun, and prediction-definition defects remain disclosed.**
- Future work beyond the completed three-quantizer study: additional quantizers, deployed integer kernels, activation quantization, other intervention layers, steering, and a quantization-specific alternative to the generic-sensitivity predictor.

---

### 4.13 What does the "survival" number measure? (measurement-validity study, 29 October; D-117 to D-124)

**Question.** The share of features whose activation correlation with full precision stays above 0.9 is the headline of the survival studies. Does it measure how hard the quantizer pushes a feature, or something else? **How.** A GPU pass (three quantizers, 27 languages, 214,506 active (language, feature) cells) streamed the sums behind the correlation, so that gain, noise, flips and the encoder's pre-activation change in threshold units (nu) could be computed. The analysis rules were committed before the data ([`analysis_freeze.json`](survival_measurement/analysis_freeze.json)); the pass reproduced the committed survival rates to 3e-8. **Findings (plain words, every number in [`survival_measurement/stage3/README.md`](survival_measurement/stage3/README.md)).** At INT4 under plain rounding the median feature loses half of its active tokens and keeps about 60 % of its gain, which the correlation (blind to gain) does not show; GPTQ loses 27 % and keeps 82 %. The noise-to-signal ratio behind the correlation follows the feature's size (Spearman -0.5 with its spread, -0.7 with its peak); the perturbation in threshold units is nearly free of the spread (|rho| <= 0.11) but not of the peak (0.25 to 0.41). A difference between resource tiers in how hard the quantizer pushes features exists already at INT16 (low tier about 13 % harder than high) and is nearly the same at INT6 and INT4 for both rounding variants: it is not a quantization-specific language effect; for GPTQ it nearly vanishes but reverses when the feature scale is adjusted, so it is not settled. Adding the activation correlation to simple baselines improves the prediction of a feature's effect similarity by 3 to 4 % (intervals above zero); adding gain and nu on top improves it by -0.4 % to +1.8 % with every interval including zero. **Limits.** One model, layer and SAE; simulated weight-only quantizers; 27 languages; post hoc relative to Stage 1; several H4 labels depend on how the slope is standardized.

### 4.14 Gemma-2-9B original-RTN scale extension (review complete, 29 October; D-125 to D-127)

The imported Study G analysis was reproduced byte for byte from raw per-state files and independently reviewed. With language-bootstrap 95% intervals: full-precision group-removal effect mean 0.794 [0.359, 1.394], positive in all 27 languages; INT4 retention macro mean 2.479 [1.781, 3.368], median language ratio 1.586 [1.238, 2.264], and ratio of mean effects 1.474 [1.263, 1.937]. These estimators answer different aggregation questions, so the headline should not select the largest ratio. Retention at INT8 and INT6 is about 1.00. The matching 2B macro mean is 2.136 [1.684, 2.656]; the paired 9B-minus-2B difference +0.343 [-0.488, 1.302] does not resolve a scale difference.

The frozen P3a inequality is reversed relative to the observed dose-response: it holds in 0/27 languages as written, while harm increases as the multiplier falls in 27/27. P3b cannot fail by construction. Neither is counted as a successful frozen confirmation. These defects limit the imported verdict, not the reproduced effect estimates. The results support a scoped same-family scale comparison; they do not establish concentration of quantization damage in the language groups. Only original RTN was run on 9B. **GPTQ is pending and matched RTN is unrun.** Full sources, aggregation definitions and prediction audit: [reviewed summary](functional_validation/gemma9b/REVIEWED_SUMMARY.md), [review ledger](functional_validation/gemma9b/REVIEW_STATUS.md).

## 5. Supported findings and their scope

The [final research questions](FINAL_RESEARCH_QUESTIONS.md) give the paper's complete answers. This table distinguishes a supported contrast from a stronger explanation that the experiment does not identify.

| supported finding | interpretation and boundary |
|---|---|
| Language-selective K64 removal harms next-token loss beyond the specified controls in all 27 Gemma-2-2B languages; graded edits give consistent dose-response behavior | Defined group edits have causal effects. They do not isolate an exclusive or complete natural language mechanism. |
| Group-effect retention is near 1 at INT6; INT4 median ratios are 1.54 / 1.36 / 1.22 for legacy RTN / matched RTN / GPTQ | Amplification depends on quantizer and endpoint; GPTQ's classification under the pre-set practical band is inconclusive rather than amplified. |
| The specified group patch repairs 2.0–2.6% of the INT4 loss gap across quantizers; the whole-residual patch repairs about 35–41% | These are measured repair effects. They are not additive damage shares or upper bounds on mediation. |
| The original corrected tests do not establish a robust selective penalty; scale matching substantially changes the comparison; GPTQ reduces the INT6 tier disparity | Selectivity/resource conclusions require activity controls and the quantization recipe. The analyses do not causally attribute scale differences to resource levels. |
| The Belebele family does not establish a positive group-specific correct-option NLL effect at FP; accuracy changes remain small under GPTQ's above-chance INT4 baseline | Task dependence is a useful result. The NLL interval is not an accuracy-equivalence bound, and it does not prove that language mechanisms never matter for comprehension. |
| Continuous activation correlation adds about 3–4% to effect-similarity prediction beyond baselines; binary association labels and added gain/perturbation measures provide limited additional information | Survival carries some signal and is an incomplete proxy. The 10% utility target is a project convention; missing it does not mean zero association. |
| Much INT4 intervention drift is shared with control directions; crossed experiments reveal dependence on context, downstream weights and edit definition | A generic component is supported. The specific component on the common-support subset is small and post hoc; scalar-loss and effect-vector decompositions are different endpoints. |
| Reproduced 9B original-RTN results show similar high-bit retention and INT4 amplification; the paired scale contrast is unresolved; Llama agrees only partly | Useful supporting checks with their own coverage and prediction defects. They do not establish quantizer-independent or architecture-universal behavior. |

The contribution is the controlled evaluation connecting multilingual feature measurements to interventions and quantizer choice. Natural-mechanism completeness, new model/SAE generalization, deployed integer kernels and activation quantization remain outside the supported claims. They are future research questions, not reasons to suppress the results above.

---

## 6. Where all the data is

Everything below is in the repository unless marked. Appendix tables A1–A10 are the same data per language; A11 to A16 are the GPTQ extension (legacy RTN, matched RTN, GPTQ side by side; see [`gptq_extension/RESULTS.md`](gptq_extension/RESULTS.md)).

| what | where |
|---|---|
| This summary, per-language tables | this file; regenerate with `python3 -B scripts/g0/make_project_results_tables.py --doc reports/PROJECT_RESULTS.md` |
| Phase 1 (survival, the three tests, follow-ups) | `reports/phase1_summary.md` (generated tables), `reports/phase1_results.md` (full account), `reports/phase1/{mask,none}/{eq,all}/` (`per_language.csv`, `summary.csv`, `analysis.md/json`, `behavior.*`, `magnitude/`, `margin/`, `thresholds/`, `language_level/`, `mixed_variants/`); `reports/phase1_reproduction.md` |
| Association retention | `reports/association_retention/results/{eq,all,within-language,margin-function,robustness-eq,crosscheck-eq}/` (`language_summary.csv`, `association_retention.md`), `reports/association_retention/interpretation.md` |
| Function (F1, F2) | `reports/functional_validation/results/eval/` (`effects_by_language.csv`, `families.csv`, `retention.csv`, `fidelity.csv`, `report.md`); pilot: `results/pilot/` |
| Patching (F3) | `results/rescue/` (`rescue.csv`, `report.md`) |
| F4, F5 | `results/f4/{INT4,INT5,INT6}/`, `results/f5/` (`f5_tables.csv`, `f5_results.json`, `report.md`), `results/f5_sensitivity/{m8,m32,direction_swap,...}/`, `results/assoc_rank/` |
| Belebele | `results/mc_eval/` (`accuracy.csv`, `effects_by_language.csv`, `statements.csv/md`), `results/mc_ksens/` (whole group), `results/mc_posctrl/` (random sets), `results/mc_pilot/`, `promptcheck/`, `belebele_calibration.md` |
| Study G | `reports/functional_validation/gain_study/results/` (`curves.csv`, `alpha_star.csv`, `reliability.csv`, `report.md`), `extension_c/`, `extension_d/`, `state_defined/results/`, `figures*/`, `README.md` (the full story) |
| Llama-3.2-1B | `reports/functional_validation/llama/` (`analysis.json`, `report.md`), `llama_robust/` (`languages.csv`, `robust.json`) |
| Intervention transfer | `reports/intervention_transfer/results/{main,external,recovery}/` (`group_per_language.csv`, `component_summary.csv`, `specificity.csv`, `predictive_validation.csv`, `claim_status.json`, `analysis_summary.json`, `cross_corpus_locked.csv`, `gradient_vs_nll.csv`, `support_*.csv`, `native_fixed_*.csv`, `figures/`) |
| Per-claim map of all 18 functional claims | `reports/functional_validation/RESULTS_SUMMARY.md`, figures: `FIGURE_INDEX.md` |
| Pre-set rules, freezes, protocols | `reports/functional_validation/evaluation_freeze_*.json`, `reports/intervention_transfer/*.json`, `docs/superpowers/` |
| How each decision was made | `reports/decisions_and_anomalies.md` (D-, A-, V-entries), `DECISIONS_FOR_REVIEW.md` (open questions), `reports/progress_log.md`, `reports/handoffs/` |
| Rerun everything | `reports/functional_validation/REPRODUCE.md`; checksums of raw outputs `reports/functional_validation/artifact_checksums.md`, `reports/intervention_transfer/artifact_checksums.md` |
| **Outside git** (too big): item-level raw outputs (e.g. `per_context_*.csv`, 26 MB each), run directories | `/home/karma/Documents/ANLP/project/transfer_runs/` and on Ada; also the private Hugging Face dataset `karma-skz/always-late-artifacts` |
| Run logs | Weights & Biases project `karma-skz-/anlp-project` (schema `reports/wandb_schema_v2.md`) |
| Report and slides | `paper/course-report/` (11 pages), `paper/presentation/`; final bundle: `scripts/make_final_archive.sh` |

---

## 7. Per-language tables (generated)

<!-- BEGIN GENERATED TABLES -->

### A1. The 27 languages

Nine languages per resource tier. 'Tier' is the project's own sorting (D-011); the sources are in `reports/g0/phase1_languages.json`.

| code | language | resource tier | script |
|---|---|---|---|
| eng_Latn | english | high | Latn |
| deu_Latn | german | high | Latn |
| spa_Latn | spanish | high | Latn |
| fra_Latn | french | high | Latn |
| por_Latn | portuguese | high | Latn |
| rus_Cyrl | russian | high | Cyrl |
| zho_Hans | mandarin | high | Hans |
| jpn_Jpan | japanese | high | Jpan |
| kor_Hang | korean | high | Hang |
| hin_Deva | hindi | mid | Deva |
| ben_Beng | bengali | mid | Beng |
| tam_Taml | tamil | mid | Taml |
| tur_Latn | turkish | mid | Latn |
| vie_Latn | vietnamese | mid | Latn |
| ind_Latn | indonesian | mid | Latn |
| tha_Thai | thai | mid | Thai |
| pol_Latn | polish | mid | Latn |
| ukr_Cyrl | ukrainian | mid | Cyrl |
| kan_Knda | kannada | low | Knda |
| tel_Telu | telugu | low | Telu |
| yor_Latn | yoruba | low | Latn |
| sun_Latn | sundanese | low | Latn |
| amh_Ethi | amharic | low | Ethi |
| ibo_Latn | igbo | low | Latn |
| hau_Latn | hausa | low | Latn |
| khm_Khmr | khmer | low | Khmr |
| lao_Laoo | lao | low | Laoo |

### A2. Phase 1: share of active features that survive (token correlation with full precision above 0.9), per language

Source: `reports/phase1/mask/eq/per_language.csv` (BOS masked, equal budget). One row per language, in percent. The median row (INT6 90.4 %) differs from the pooled figure of section 4.2 (INT6 94.9 %), which pools all active features of all languages.

| language | INT8 | INT7 | INT6 | INT5 | INT4 |
|---|---|---|---|---|---|
| english (eng, high) | 100.0 | 99.9 | 97.2 | 76.9 | 31.8 |
| german (deu, high) | 100.0 | 99.9 | 95.6 | 66.7 | 20.4 |
| spanish (spa, high) | 100.0 | 99.9 | 96.3 | 70.1 | 24.3 |
| french (fra, high) | 100.0 | 99.9 | 95.6 | 69.4 | 23.0 |
| portuguese (por, high) | 100.0 | 99.9 | 95.7 | 69.3 | 22.5 |
| russian (rus, high) | 100.0 | 99.8 | 95.5 | 67.1 | 20.3 |
| mandarin (zho, high) | 100.0 | 99.8 | 95.8 | 67.8 | 20.6 |
| japanese (jpn, high) | 100.0 | 99.8 | 93.5 | 59.6 | 15.4 |
| korean (kor, high) | 100.0 | 99.7 | 90.4 | 48.9 | 10.0 |
| hindi (hin, mid) | 100.0 | 99.7 | 90.3 | 50.2 | 11.5 |
| bengali (ben, mid) | 99.9 | 99.0 | 83.1 | 37.6 | 5.8 |
| tamil (tam, mid) | 100.0 | 99.0 | 82.4 | 34.6 | 4.6 |
| turkish (tur, mid) | 100.0 | 99.6 | 90.3 | 50.8 | 12.0 |
| vietnamese (vie, mid) | 100.0 | 99.7 | 93.5 | 59.8 | 14.6 |
| indonesian (ind, mid) | 100.0 | 99.8 | 94.8 | 64.2 | 19.1 |
| thai (tha, mid) | 100.0 | 99.7 | 92.2 | 53.1 | 9.1 |
| polish (pol, mid) | 100.0 | 99.8 | 94.6 | 61.9 | 16.2 |
| ukrainian (ukr, mid) | 100.0 | 99.9 | 93.8 | 59.4 | 15.1 |
| kannada (kan, low) | 99.9 | 98.5 | 78.5 | 31.0 | 4.3 |
| telugu (tel, low) | 99.9 | 98.9 | 80.8 | 33.4 | 4.9 |
| yoruba (yor, low) | 99.5 | 94.9 | 67.6 | 27.9 | 6.8 |
| sundanese (sun, low) | 99.9 | 98.5 | 78.0 | 34.6 | 8.4 |
| amharic (amh, low) | 99.5 | 94.9 | 62.8 | 22.3 | 3.8 |
| igbo (ibo, low) | 99.5 | 95.9 | 66.5 | 28.0 | 6.4 |
| hausa (hau, low) | 99.8 | 97.2 | 73.1 | 30.0 | 6.3 |
| khmer (khm, low) | 99.7 | 96.2 | 64.0 | 21.3 | 2.6 |
| lao (lao, low) | 99.1 | 93.2 | 53.0 | 19.3 | 2.8 |
| **median of the 27** | 100.0 | 99.7 | 90.4 | 50.8 | 11.5 |

### A3. Association retention: of the features that carry the language tag in full precision, how many still carry it (%)

Source: `reports/association_retention/results/eq/language_summary.csv`, group 'selective', column `conditional_fraction`. INT16 is the rounding-only floor.

| language | features with the language tag in full precision | INT16 kept % | INT8 kept % | INT7 kept % | INT6 kept % | INT5 kept % | INT4 kept % |
|---|---|---|---|---|---|---|---|
| english (eng, high) | 1447 | 99.3 | 98.1 | 98.9 | 96.1 | 89.8 | 87.8 |
| german (deu, high) | 164 | 97.0 | 97.6 | 96.3 | 92.7 | 88.4 | 73.8 |
| spanish (spa, high) | 139 | 98.6 | 97.8 | 96.4 | 92.1 | 88.5 | 61.9 |
| french (fra, high) | 127 | 99.2 | 97.6 | 95.3 | 92.9 | 91.3 | 78.0 |
| portuguese (por, high) | 39 | 100.0 | 94.9 | 100.0 | 97.4 | 89.7 | 79.5 |
| russian (rus, high) | 46 | 100.0 | 97.8 | 93.5 | 97.8 | 78.3 | 67.4 |
| mandarin (zho, high) | 365 | 98.9 | 97.3 | 96.4 | 95.9 | 88.8 | 75.9 |
| japanese (jpn, high) | 407 | 98.8 | 96.6 | 95.8 | 92.9 | 89.9 | 64.1 |
| korean (kor, high) | 174 | 98.3 | 98.9 | 97.7 | 96.0 | 87.9 | 67.2 |
| hindi (hin, mid) | 123 | 98.4 | 97.6 | 95.1 | 95.9 | 83.7 | 69.9 |
| bengali (ben, mid) | 88 | 97.7 | 96.6 | 95.5 | 90.9 | 87.5 | 61.4 |
| tamil (tam, mid) | 179 | 98.9 | 97.8 | 95.5 | 95.5 | 88.3 | 67.6 |
| turkish (tur, mid) | 244 | 98.8 | 97.5 | 97.5 | 95.5 | 87.7 | 59.0 |
| vietnamese (vie, mid) | 233 | 99.1 | 99.6 | 96.6 | 96.1 | 88.0 | 73.4 |
| indonesian (ind, mid) | 98 | 100.0 | 99.0 | 98.0 | 95.9 | 94.9 | 74.5 |
| thai (tha, mid) | 270 | 100.0 | 98.9 | 98.5 | 96.7 | 88.5 | 76.3 |
| polish (pol, mid) | 87 | 98.9 | 98.9 | 97.7 | 95.4 | 90.8 | 66.7 |
| ukrainian (ukr, mid) | 61 | 100.0 | 100.0 | 96.7 | 85.2 | 90.2 | 73.8 |
| kannada (kan, low) | 138 | 98.6 | 98.6 | 97.8 | 93.5 | 85.5 | 54.3 |
| telugu (tel, low) | 118 | 100.0 | 97.5 | 96.6 | 96.6 | 87.3 | 69.5 |
| yoruba (yor, low) | 424 | 99.3 | 98.8 | 97.4 | 94.8 | 90.6 | 65.1 |
| sundanese (sun, low) | 217 | 100.0 | 99.5 | 98.2 | 96.3 | 87.6 | 71.9 |
| amharic (amh, low) | 174 | 99.4 | 96.6 | 97.1 | 92.0 | 91.4 | 59.8 |
| igbo (ibo, low) | 257 | 99.6 | 98.8 | 98.1 | 95.3 | 89.5 | 64.2 |
| hausa (hau, low) | 214 | 98.6 | 99.1 | 96.7 | 90.2 | 91.6 | 50.9 |
| khmer (khm, low) | 404 | 99.8 | 98.5 | 97.8 | 93.6 | 89.6 | 63.1 |
| lao (lao, low) | 426 | 99.3 | 98.1 | 98.1 | 92.7 | 88.0 | 57.3 |
| **all 27 pooled** | 6663 | 99.2 | 98.1 | 97.5 | 94.7 | 89.1 | 70.8 |

### A4. Function: what zeroing the 64 most language-specific features does to the language's loss, and how much of that is kept when the model is quantized

Source: `reports/functional_validation/results/eval/effects_by_language.csv` and `retention.csv`. Ratio = effect in the quantized model divided by effect in full precision. ▲ = clearly amplified, ? = inconclusive, no mark = within the band of 'kept'.

| language | harm of zeroing the group (nats/token) | harm of zeroing matched controls | effect = group − control [95 % interval] | INT8 ratio | INT7 ratio | INT6 ratio | INT5 ratio | INT4 ratio |
|---|---|---|---|---|---|---|---|---|
| english (eng, high) | 0.209 | 0.036 | 0.173 [0.151, 0.197] | 0.99 | 0.99 | 0.96 | 1.02 | 0.97 |
| german (deu, high) | 1.087 | 0.013 | 1.073 [0.997, 1.161] | 0.99 | 1.04 | 1.09 | 0.91 | 0.72 ? |
| spanish (spa, high) | 0.514 | 0.020 | 0.494 [0.429, 0.560] | 1.05 | 1.02 | 1.04 | 1.17 | 1.28 ? |
| french (fra, high) | 0.609 | 0.026 | 0.584 [0.538, 0.631] | 1.03 | 0.97 | 1.16 | 0.95 | 1.54 ▲ |
| portuguese (por, high) | 1.031 | 0.007 | 1.024 [0.939, 1.107] | 0.96 | 1.01 | 1.09 | 1.22 | 0.95 |
| russian (rus, high) | 0.080 | 0.007 | 0.073 [0.061, 0.087] | 1.05 | 1.08 | 1.08 | 1.02 | 2.92 ▲ |
| mandarin (zho, high) | 7.599 | 0.024 | 7.575 [7.206, 7.937] | 1.02 | 1.03 | 1.06 | 0.96 | 1.08 |
| japanese (jpn, high) | 0.286 | 0.024 | 0.262 [0.240, 0.286] | 1.02 | 1.02 | 1.03 | 1.07 | 1.42 ▲ |
| korean (kor, high) | 0.363 | 0.012 | 0.351 [0.306, 0.405] | 0.89 | 1.24 | 0.88 | 1.04 | 2.63 ▲ |
| hindi (hin, mid) | 0.121 | 0.016 | 0.104 [0.088, 0.122] | 0.97 | 1.10 | 1.36 ? | 0.91 | 2.90 ▲ |
| bengali (ben, mid) | 0.016 | 0.006 | 0.010 [0.007, 0.013] | 0.99 | 1.12 | 0.94 | 1.26 ? | 3.89 ▲ |
| tamil (tam, mid) | 0.041 | 0.013 | 0.028 [0.024, 0.033] | 0.96 | 1.05 | 1.01 | 1.37 ? | 2.63 ▲ |
| turkish (tur, mid) | 0.244 | 0.036 | 0.208 [0.188, 0.229] | 1.00 | 1.03 | 0.96 | 1.09 | 2.75 ▲ |
| vietnamese (vie, mid) | 0.023 | 0.011 | 0.012 [0.004, 0.020] | 1.06 | 1.25 ? | 0.83 ? | 1.26 ? | 5.70 ▲ |
| indonesian (ind, mid) | 0.036 | 0.010 | 0.027 [0.013, 0.041] | 1.04 | 0.96 | 0.99 | 1.38 ? | 1.05 ? |
| thai (tha, mid) | 0.058 | 0.012 | 0.046 [0.040, 0.053] | 0.96 | 1.04 | 1.11 | 1.35 ? | 3.74 ▲ |
| polish (pol, mid) | 0.388 | 0.028 | 0.360 [0.318, 0.403] | 1.05 | 0.87 | 0.92 | 1.15 | 1.58 ▲ |
| ukrainian (ukr, mid) | 0.429 | 0.010 | 0.419 [0.382, 0.453] | 1.04 | 1.13 | 1.15 | 0.89 | 2.30 ▲ |
| kannada (kan, low) | 0.052 | 0.018 | 0.034 [0.028, 0.040] | 1.03 | 1.03 | 1.00 | 1.00 | 1.29 ? |
| telugu (tel, low) | 0.054 | 0.026 | 0.028 [0.023, 0.034] | 1.02 | 1.04 | 1.24 ? | 1.08 | 5.75 ▲ |
| yoruba (yor, low) | 1.640 | 0.062 | 1.578 [1.520, 1.637] | 1.00 | 1.00 | 1.02 | 1.28 | 1.38 ▲ |
| sundanese (sun, low) | 0.972 | 0.099 | 0.873 [0.822, 0.924] | 0.99 | 1.02 | 0.90 | 1.07 | 1.18 |
| amharic (amh, low) | 1.047 | 0.027 | 1.020 [0.998, 1.045] | 0.99 | 1.02 | 1.15 | 1.11 | 2.31 ▲ |
| igbo (ibo, low) | 0.391 | 0.046 | 0.346 [0.322, 0.368] | 1.01 | 0.98 | 0.98 | 1.05 | 1.25 |
| hausa (hau, low) | 0.215 | 0.053 | 0.162 [0.145, 0.179] | 0.99 | 1.01 | 1.03 | 1.11 | 1.29 ? |
| khmer (khm, low) | 0.251 | 0.032 | 0.220 [0.200, 0.241] | 1.01 | 0.99 | 1.03 | 1.28 ? | 1.89 ▲ |
| lao (lao, low) | 1.941 | 0.026 | 1.915 [1.837, 1.994] | 1.04 | 0.92 | 1.00 | 1.49 ▲ | 1.27 |
| **median of the 27** | 0.286 | 0.024 | 0.262 | 1.01 | 1.02 | 1.03 | 1.09 | 1.54 |

### A4b. The same ratio for the control rows (rounding only, and no quantization)

These rows show what 'no change' looks like: INT0 is exactly 1.00; INT16 and INT12 are rounding only.

| precision | median ratio over the 27 languages | meaning |
|---|---|---|
| INT16 | 1.00 | rounding only (control) |
| INT12 | 1.00 | rounding only (control) |
| INT0 | 1.00 | no quantization at all (control) |
| INT8 | 1.01 | quantized |
| INT7 | 1.02 | quantized |
| INT6 | 1.03 | quantized |
| INT5 | 1.09 | quantized |
| INT4 | 1.54 | quantized |

### A5. Patching experiment (F3): how much of the INT4 loss gap is repaired when the group's full-precision activations are put back

Source: `reports/functional_validation/results/rescue/rescue.csv`, part 'group'. A small number is what this particular patch recovers; it is not an upper bound and not a location of the damage (D-110).

| language | loss gap INT4 minus full precision (nats/token) | recovered by putting the group's full-precision activations back | share of the gap [95 % interval] |
|---|---|---|---|
| english (eng, high) | 0.175 | 0.0015 | 0.9 % [-0.7, 2.5] |
| german (deu, high) | 0.399 | -0.0001 | -0.0 % [-0.7, 0.7] |
| spanish (spa, high) | 0.287 | 0.0032 | 1.1 % [0.3, 1.9] |
| french (fra, high) | 0.315 | 0.0068 | 2.2 % [1.1, 3.4] |
| portuguese (por, high) | 0.345 | 0.0094 | 2.7 % [1.7, 3.7] |
| russian (rus, high) | 0.356 | 0.0044 | 1.2 % [0.5, 2.1] |
| mandarin (zho, high) | 0.389 | 0.0034 | 0.9 % [0.0, 1.7] |
| japanese (jpn, high) | 0.418 | 0.0074 | 1.8 % [1.0, 2.5] |
| korean (kor, high) | 0.526 | 0.0153 | 2.9 % [2.4, 3.5] |
| hindi (hin, mid) | 0.495 | 0.0058 | 1.2 % [0.7, 1.7] |
| bengali (ben, mid) | 0.591 | 0.0121 | 2.0 % [1.6, 2.5] |
| tamil (tam, mid) | 0.595 | 0.0049 | 0.8 % [0.5, 1.2] |
| turkish (tur, mid) | 0.548 | 0.0321 | 5.9 % [4.8, 7.1] |
| vietnamese (vie, mid) | 0.439 | 0.0087 | 2.0 % [1.2, 2.8] |
| indonesian (ind, mid) | 0.495 | 0.0018 | 0.4 % [-0.0, 0.8] |
| thai (tha, mid) | 0.499 | 0.0145 | 2.9 % [2.4, 3.5] |
| polish (pol, mid) | 0.517 | 0.0246 | 4.8 % [3.7, 5.9] |
| ukrainian (ukr, mid) | 0.458 | 0.0041 | 0.9 % [0.4, 1.4] |
| kannada (kan, low) | 0.600 | 0.0065 | 1.1 % [0.8, 1.4] |
| telugu (tel, low) | 0.593 | 0.0142 | 2.4 % [2.0, 2.8] |
| yoruba (yor, low) | 0.582 | 0.0389 | 6.7 % [5.7, 7.6] |
| sundanese (sun, low) | 0.806 | 0.0620 | 7.7 % [6.6, 8.8] |
| amharic (amh, low) | 1.482 | 0.0258 | 1.7 % [1.3, 2.2] |
| igbo (ibo, low) | 0.660 | 0.0208 | 3.1 % [2.6, 3.6] |
| hausa (hau, low) | 0.832 | 0.0295 | 3.5 % [3.0, 4.1] |
| khmer (khm, low) | 0.813 | 0.0102 | 1.2 % [0.7, 1.8] |
| lao (lao, low) | 1.740 | 0.1506 | 8.7 % [8.1, 9.2] |

### A6. Downstream task (Belebele reading comprehension), accuracy per language

Source: `reports/functional_validation/results/mc_eval/accuracy.csv`. Chance is 0.25. Nine languages are not covered because the model cannot read them well enough for the task to measure anything.

| language | items | FP: nothing removed | FP: group removed | FP: matched controls removed | no edit, INT8 | no edit, INT7 | no edit, INT6 | no edit, INT5 | no edit, INT4 |
|---|---|---|---|---|---|---|---|---|---|
| english (eng, high) | 385 | 0.548 | 0.558 | 0.555 | 0.556 | 0.545 | 0.545 | 0.494 | 0.278 |
| german (deu, high) | 385 | 0.468 | 0.447 | 0.451 | 0.455 | 0.442 | 0.457 | 0.400 | 0.260 |
| spanish (spa, high) | 385 | 0.475 | 0.470 | 0.472 | 0.481 | 0.468 | 0.455 | 0.397 | 0.234 |
| french (fra, high) | 385 | 0.462 | 0.460 | 0.452 | 0.444 | 0.449 | 0.436 | 0.395 | 0.236 |
| portuguese (por, high) | 385 | 0.416 | 0.423 | 0.424 | 0.416 | 0.429 | 0.423 | 0.410 | 0.229 |
| russian (rus, high) | 385 | 0.460 | 0.460 | 0.461 | 0.468 | 0.421 | 0.455 | 0.408 | 0.231 |
| mandarin (zho, high) | 385 | 0.452 | 0.475 | 0.442 | 0.470 | 0.439 | 0.455 | 0.392 | 0.242 |
| japanese (jpn, high) | 385 | 0.397 | 0.395 | 0.417 | 0.395 | 0.384 | 0.403 | 0.397 | 0.226 |
| korean (kor, high) | 385 | 0.436 | 0.455 | 0.458 | 0.439 | 0.439 | 0.439 | 0.410 | 0.226 |
| hindi (hin, mid) | 385 | 0.371 | 0.369 | 0.371 | 0.369 | 0.374 | 0.364 | 0.348 | 0.223 |
| bengali (ben, mid) | 385 | 0.379 | 0.387 | 0.375 | 0.379 | 0.369 | 0.395 | 0.335 | 0.226 |
| tamil (tam, mid) | not covered |  |  |  |  |  |  |  |  |
| turkish (tur, mid) | 385 | 0.403 | 0.392 | 0.398 | 0.397 | 0.410 | 0.392 | 0.377 | 0.252 |
| vietnamese (vie, mid) | 385 | 0.387 | 0.392 | 0.386 | 0.387 | 0.400 | 0.403 | 0.384 | 0.252 |
| indonesian (ind, mid) | 385 | 0.439 | 0.442 | 0.438 | 0.452 | 0.431 | 0.434 | 0.364 | 0.221 |
| thai (tha, mid) | 385 | 0.387 | 0.369 | 0.376 | 0.392 | 0.379 | 0.374 | 0.338 | 0.223 |
| polish (pol, mid) | 385 | 0.423 | 0.421 | 0.423 | 0.431 | 0.434 | 0.431 | 0.384 | 0.239 |
| ukrainian (ukr, mid) | 385 | 0.421 | 0.410 | 0.411 | 0.405 | 0.400 | 0.436 | 0.390 | 0.223 |
| kannada (kan, low) | 385 | 0.338 | 0.335 | 0.342 | 0.340 | 0.348 | 0.314 | 0.351 | 0.223 |
| telugu (tel, low) | not covered |  |  |  |  |  |  |  |  |
| yoruba (yor, low) | not covered |  |  |  |  |  |  |  |  |
| sundanese (sun, low) | not covered |  |  |  |  |  |  |  |  |
| amharic (amh, low) | not covered |  |  |  |  |  |  |  |  |
| igbo (ibo, low) | not covered |  |  |  |  |  |  |  |  |
| hausa (hau, low) | not covered |  |  |  |  |  |  |  |  |
| khmer (khm, low) | not covered |  |  |  |  |  |  |  |  |
| lao (lao, low) | not covered |  |  |  |  |  |  |  |  |
| **mean of the covered languages** |  | 0.426 | 0.426 | 0.425 | 0.426 | 0.420 | 0.423 | 0.387 | 0.236 |

### A7. Study G (scaling the group up and down): net harm per language in nats per token (group minus matched controls)

Source: `reports/functional_validation/gain_study/results/curves.csv`. First table: group removed (ratio 0). Second table: group scaled UP; negative = the loss got slightly lower (a 'repair').

| language | FP, group removed | INT8, group removed | INT7, group removed | INT6, group removed | INT5, group removed | INT4, group removed |
|---|---|---|---|---|---|---|
| english (eng, high) | +0.173 | +0.171 | +0.170 | +0.165 | +0.176 | +0.167 |
| german (deu, high) | +1.073 | +1.060 | +1.113 | +1.166 | +0.981 | +0.770 |
| spanish (spa, high) | +0.494 | +0.520 | +0.503 | +0.514 | +0.576 | +0.633 |
| french (fra, high) | +0.584 | +0.602 | +0.567 | +0.676 | +0.551 | +0.898 |
| portuguese (por, high) | +1.024 | +0.986 | +1.032 | +1.121 | +1.248 | +0.973 |
| russian (rus, high) | +0.073 | +0.077 | +0.079 | +0.079 | +0.074 | +0.213 |
| mandarin (zho, high) | +7.575 | +7.718 | +7.776 | +7.995 | +7.277 | +8.218 |
| japanese (jpn, high) | +0.262 | +0.268 | +0.266 | +0.271 | +0.280 | +0.373 |
| korean (kor, high) | +0.351 | +0.314 | +0.434 | +0.310 | +0.364 | +0.925 |
| hindi (hin, mid) | +0.104 | +0.101 | +0.115 | +0.142 | +0.095 | +0.302 |
| bengali (ben, mid) | +0.010 | +0.010 | +0.011 | +0.009 | +0.012 | +0.038 |
| tamil (tam, mid) | +0.028 | +0.027 | +0.029 | +0.028 | +0.038 | +0.074 |
| turkish (tur, mid) | +0.208 | +0.209 | +0.213 | +0.199 | +0.227 | +0.572 |
| vietnamese (vie, mid) | +0.012 | +0.013 | +0.015 | +0.010 | +0.015 | +0.067 |
| indonesian (ind, mid) | +0.027 | +0.028 | +0.026 | +0.026 | +0.037 | +0.028 |
| thai (tha, mid) | +0.046 | +0.044 | +0.048 | +0.051 | +0.062 | +0.172 |
| polish (pol, mid) | +0.360 | +0.377 | +0.313 | +0.330 | +0.414 | +0.569 |
| ukrainian (ukr, mid) | +0.419 | +0.436 | +0.472 | +0.482 | +0.372 | +0.966 |
| kannada (kan, low) | +0.034 | +0.035 | +0.035 | +0.034 | +0.034 | +0.044 |
| telugu (tel, low) | +0.028 | +0.029 | +0.029 | +0.035 | +0.031 | +0.163 |
| yoruba (yor, low) | +1.578 | +1.585 | +1.578 | +1.616 | +2.016 | +2.186 |
| sundanese (sun, low) | +0.873 | +0.860 | +0.891 | +0.790 | +0.930 | +1.029 |
| amharic (amh, low) | +1.020 | +1.007 | +1.041 | +1.169 | +1.135 | +2.360 |
| igbo (ibo, low) | +0.346 | +0.348 | +0.339 | +0.340 | +0.364 | +0.432 |
| hausa (hau, low) | +0.162 | +0.161 | +0.163 | +0.168 | +0.181 | +0.209 |
| khmer (khm, low) | +0.220 | +0.221 | +0.216 | +0.226 | +0.281 | +0.416 |
| lao (lao, low) | +1.915 | +1.990 | +1.760 | +1.907 | +2.862 | +2.438 |
| **median of the 27** | +0.262 | +0.268 | +0.266 | +0.271 | +0.281 | +0.432 |

| language | FP ×1.1 | FP ×1.2 | INT6 ×1.1 | INT6 ×1.2 | INT4 ×1.1 | INT4 ×1.2 | FP ×11.3 | INT4 ×11.3 |
|---|---|---|---|---|---|---|---|---|
| english (eng, high) | -0.0007 | -0.0008 | -0.0005 | -0.0010 | -0.0010 | -0.0017 | -0.220 | -0.383 |
| german (deu, high) | +0.0020 | +0.0041 | +0.0026 | +0.0045 | +0.0026 | +0.0041 | +1.529 | +1.312 |
| spanish (spa, high) | +0.0007 | +0.0025 | +0.0010 | +0.0025 | -0.0009 | -0.0009 | +0.721 | +0.436 |
| french (fra, high) | +0.0011 | +0.0020 | +0.0008 | +0.0029 | -0.0022 | -0.0015 | +0.844 | +0.599 |
| portuguese (por, high) | +0.0008 | +0.0022 | +0.0006 | +0.0009 | -0.0006 | -0.0005 | +0.893 | +0.940 |
| russian (rus, high) | +0.0017 | +0.0034 | +0.0020 | +0.0032 | -0.0007 | -0.0005 | +1.235 | +0.525 |
| mandarin (zho, high) | +0.0048 | +0.0121 | +0.0055 | +0.0126 | -0.0005 | +0.0030 | +1.613 | +1.239 |
| japanese (jpn, high) | +0.0035 | +0.0069 | +0.0024 | +0.0066 | -0.0002 | +0.0007 | +0.999 | +0.493 |
| korean (kor, high) | +0.0026 | +0.0052 | +0.0009 | +0.0036 | -0.0049 | -0.0094 | +0.837 | +0.403 |
| hindi (hin, mid) | +0.0017 | +0.0059 | +0.0024 | +0.0064 | +0.0003 | +0.0016 | +1.541 | +1.069 |
| bengali (ben, mid) | +0.0003 | +0.0012 | +0.0009 | +0.0012 | -0.0010 | -0.0011 | +0.635 | +0.451 |
| tamil (tam, mid) | -0.0002 | +0.0009 | -0.0000 | +0.0005 | -0.0019 | -0.0034 | +2.051 | +1.624 |
| turkish (tur, mid) | +0.0002 | -0.0003 | -0.0004 | +0.0007 | -0.0052 | -0.0107 | +0.504 | -0.446 |
| vietnamese (vie, mid) | +0.0034 | +0.0069 | +0.0037 | +0.0067 | -0.0003 | +0.0004 | +0.772 | +0.519 |
| indonesian (ind, mid) | +0.0010 | +0.0018 | +0.0008 | +0.0014 | +0.0010 | +0.0027 | +0.221 | +0.332 |
| thai (tha, mid) | +0.0006 | +0.0019 | +0.0005 | +0.0014 | -0.0040 | -0.0068 | +0.069 | -0.402 |
| polish (pol, mid) | +0.0007 | +0.0029 | +0.0013 | +0.0036 | -0.0011 | -0.0013 | +1.390 | +1.242 |
| ukrainian (ukr, mid) | +0.0024 | +0.0060 | +0.0027 | +0.0053 | -0.0019 | -0.0029 | +2.530 | +1.608 |
| kannada (kan, low) | +0.0002 | +0.0011 | +0.0008 | +0.0013 | +0.0009 | +0.0010 | +1.213 | +1.927 |
| telugu (tel, low) | +0.0007 | +0.0008 | -0.0002 | +0.0002 | -0.0049 | -0.0089 | +1.227 | +0.810 |
| yoruba (yor, low) | -0.0099 | -0.0135 | -0.0107 | -0.0142 | -0.0160 | -0.0229 | -0.015 | -0.268 |
| sundanese (sun, low) | -0.0059 | -0.0119 | -0.0057 | -0.0092 | -0.0223 | -0.0412 | +0.311 | +0.180 |
| amharic (amh, low) | -0.0036 | -0.0027 | -0.0042 | -0.0027 | +0.0064 | +0.0264 | +6.179 | +4.147 |
| igbo (ibo, low) | -0.0011 | -0.0003 | -0.0009 | -0.0006 | -0.0085 | -0.0144 | +1.739 | +1.231 |
| hausa (hau, low) | +0.0008 | +0.0024 | +0.0004 | +0.0023 | -0.0009 | -0.0018 | +1.326 | +1.276 |
| khmer (khm, low) | +0.0003 | +0.0020 | +0.0000 | +0.0016 | +0.0015 | +0.0061 | +5.430 | +2.674 |
| lao (lao, low) | -0.0012 | +0.0010 | -0.0027 | -0.0013 | -0.0181 | -0.0191 | +7.159 | +4.490 |

### A8. Intervention transfer, main matrix (27 languages, FLORES): harm of removing the group, split into a context part and a weights part

Source: `reports/intervention_transfer/results/main/group_per_language.csv`. C = change from the activations (context), W = change from the weights, I = the two together beyond the sum. Net harm in nats per token.

| language | full precision: harm of removing the group | INT6: C | INT6: W | INT6: I | INT4: nothing quantized | INT4: only the layers below the edit quantized | INT4: only the layers above the edit quantized | INT4: both quantized, same edit as before | INT4: both quantized, edit scaled to the model's own activations | INT4: C | INT4: W | INT4: I |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| english (eng, high) | 0.172 | -0.0053 | -0.0027 | -0.0002 | 0.172 | 0.162 | 0.184 | 0.179 | 0.168 | -0.011 | +0.012 | +0.005 |
| german (deu, high) | 1.075 | +0.0507 | +0.0079 | +0.0031 | 1.075 | 2.039 | 0.852 | 1.806 | 0.768 | +0.964 | -0.224 | -0.010 |
| spanish (spa, high) | 0.492 | -0.0510 | +0.0119 | +0.0018 | 0.492 | 1.248 | 0.617 | 1.530 | 0.627 | +0.755 | +0.125 | +0.157 |
| french (fra, high) | 0.583 | -0.0114 | +0.0606 | -0.0036 | 0.583 | 1.260 | 1.093 | 1.947 | 0.884 | +0.677 | +0.509 | +0.178 |
| portuguese (por, high) | 1.022 | -0.0010 | +0.0144 | -0.0051 | 1.022 | 0.686 | 1.728 | 1.156 | 0.944 | -0.336 | +0.706 | -0.235 |
| russian (rus, high) | 0.073 | +0.0010 | +0.0061 | +0.0007 | 0.073 | 0.279 | 0.169 | 0.564 | 0.210 | +0.206 | +0.096 | +0.190 |
| mandarin (zho, high) | 7.582 | +0.0973 | +0.1161 | -0.0010 | 7.582 | 9.013 | 8.159 | 9.543 | 8.218 | +1.431 | +0.577 | -0.047 |
| japanese (jpn, high) | 0.263 | +0.0083 | +0.0092 | +0.0007 | 0.263 | 0.684 | 0.366 | 0.848 | 0.373 | +0.421 | +0.104 | +0.061 |
| korean (kor, high) | 0.350 | -0.0238 | -0.0152 | +0.0066 | 0.350 | 0.825 | 0.809 | 1.481 | 0.905 | +0.475 | +0.459 | +0.197 |
| hindi (hin, mid) | 0.104 | +0.0133 | +0.0287 | +0.0055 | 0.104 | 0.414 | 0.228 | 0.651 | 0.300 | +0.310 | +0.125 | +0.113 |
| bengali (ben, mid) | 0.010 | +0.0002 | -0.0002 | -0.0001 | 0.010 | 0.029 | 0.032 | 0.088 | 0.038 | +0.019 | +0.022 | +0.037 |
| tamil (tam, mid) | 0.028 | +0.0006 | +0.0002 | -0.0001 | 0.028 | 0.057 | 0.053 | 0.097 | 0.074 | +0.029 | +0.025 | +0.015 |
| turkish (tur, mid) | 0.208 | +0.0332 | -0.0255 | -0.0036 | 0.208 | 0.720 | 0.537 | 1.582 | 0.569 | +0.513 | +0.330 | +0.532 |
| vietnamese (vie, mid) | 0.012 | +0.0033 | -0.0053 | -0.0009 | 0.012 | 0.087 | 0.053 | 0.127 | 0.067 | +0.076 | +0.042 | -0.002 |
| indonesian (ind, mid) | 0.027 | +0.0007 | -0.0005 | -0.0000 | 0.027 | 0.029 | 0.026 | 0.028 | 0.028 | +0.002 | -0.001 | +0.000 |
| thai (tha, mid) | 0.046 | +0.0038 | +0.0028 | +0.0003 | 0.046 | 0.133 | 0.137 | 0.329 | 0.171 | +0.087 | +0.091 | +0.105 |
| polish (pol, mid) | 0.360 | +0.0091 | -0.0184 | +0.0011 | 0.360 | 0.697 | 0.449 | 0.915 | 0.564 | +0.337 | +0.089 | +0.129 |
| ukrainian (ukr, mid) | 0.419 | +0.0129 | +0.0701 | +0.0031 | 0.419 | 1.054 | 0.857 | 1.789 | 0.955 | +0.636 | +0.438 | +0.297 |
| kannada (kan, low) | 0.034 | -0.0006 | +0.0008 | -0.0002 | 0.034 | 0.043 | 0.042 | 0.051 | 0.045 | +0.009 | +0.008 | -0.000 |
| telugu (tel, low) | 0.029 | +0.0039 | +0.0047 | +0.0000 | 0.029 | 0.102 | 0.109 | 0.254 | 0.163 | +0.073 | +0.081 | +0.071 |
| yoruba (yor, low) | 1.578 | +0.0263 | +0.0355 | -0.0028 | 1.578 | 2.255 | 2.072 | 2.781 | 2.183 | +0.676 | +0.494 | +0.032 |
| sundanese (sun, low) | 0.872 | +0.0168 | -0.1060 | -0.0030 | 0.872 | 0.956 | 1.097 | 1.103 | 1.028 | +0.084 | +0.225 | -0.078 |
| amharic (amh, low) | 1.020 | +0.0037 | +0.1127 | -0.0008 | 1.020 | 1.429 | 2.133 | 2.715 | 2.358 | +0.409 | +1.113 | +0.173 |
| igbo (ibo, low) | 0.347 | +0.0008 | -0.0023 | +0.0002 | 0.347 | 0.441 | 0.428 | 0.505 | 0.433 | +0.094 | +0.082 | -0.018 |
| hausa (hau, low) | 0.161 | +0.0108 | +0.0051 | -0.0000 | 0.161 | 0.262 | 0.195 | 0.274 | 0.209 | +0.101 | +0.034 | -0.023 |
| khmer (khm, low) | 0.220 | +0.0106 | +0.0003 | -0.0014 | 0.220 | 0.374 | 0.291 | 0.511 | 0.416 | +0.154 | +0.071 | +0.066 |
| lao (lao, low) | 1.911 | -0.2082 | +0.1781 | +0.0090 | 1.911 | 2.803 | 3.107 | 3.302 | 2.440 | +0.892 | +1.196 | -0.697 |

### A9. Intervention transfer, external panel (six languages, Wikipedia text)

Source: `reports/intervention_transfer/results/external/group_per_language.csv`. Same columns as A8, on fresh text.

| language | full precision: harm of removing the group | INT6: C | INT6: W | INT6: I | INT4: nothing quantized | INT4: only the layers below the edit quantized | INT4: only the layers above the edit quantized | INT4: both quantized, same edit as before | INT4: both quantized, edit scaled to the model's own activations | INT4: C | INT4: W | INT4: I |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| english (eng, high) | 0.106 | -0.0007 | -0.0016 | -0.0002 | 0.106 | 0.107 | 0.111 | 0.113 | 0.102 | +0.001 | +0.005 | +0.001 |
| spanish (spa, high) | 0.259 | -0.0308 | +0.0143 | -0.0018 | 0.259 | 0.793 | 0.377 | 1.195 | 0.470 | +0.534 | +0.118 | +0.284 |
| russian (rus, high) | 0.069 | +0.0016 | +0.0070 | +0.0004 | 0.069 | 0.202 | 0.134 | 0.353 | 0.143 | +0.133 | +0.064 | +0.086 |
| mandarin (zho, high) | 7.884 | -0.0294 | +0.1134 | +0.0026 | 7.884 | 8.517 | 8.914 | 9.368 | 8.409 | +0.633 | +1.030 | -0.179 |
| hindi (hin, mid) | 0.056 | +0.0076 | +0.0124 | +0.0013 | 0.056 | 0.181 | 0.129 | 0.306 | 0.149 | +0.126 | +0.073 | +0.052 |
| yoruba (yor, low) | 1.195 | +0.0543 | +0.0274 | -0.0073 | 1.195 | 1.774 | 1.674 | 2.169 | 1.684 | +0.579 | +0.479 | -0.084 |

### A10. Second model check (Llama-3.2-1B): the group effect in full precision, next to Gemma

Source: `reports/functional_validation/llama_robust/languages.csv` and the Gemma table above. The pre-set rule needed 24 of 27 languages above zero.

| language | Gemma-2-2B effect (group − control) | Llama-3.2-1B effect, mean of 3 control draws | Llama effect, median of the draws | group beats all draws? |
|---|---|---|---|---|
| english (eng, high) | 0.173 | 0.057 | 0.062 | True |
| german (deu, high) | 1.073 | 1.163 | 1.222 | True |
| spanish (spa, high) | 0.494 | 0.266 | 0.256 | True |
| french (fra, high) | 0.584 | 0.185 | 0.166 | True |
| portuguese (por, high) | 1.024 | 0.239 | 0.239 | True |
| russian (rus, high) | 0.073 | -0.082 | -0.069 | False |
| mandarin (zho, high) | 7.575 | -0.012 | -0.011 | False |
| japanese (jpn, high) | 0.262 | -1.234 | -0.103 | False |
| korean (kor, high) | 0.351 | -0.592 | 0.004 | False |
| hindi (hin, mid) | 0.104 | -0.085 | -0.019 | False |
| bengali (ben, mid) | 0.010 | 0.017 | 0.017 | True |
| tamil (tam, mid) | 0.028 | 0.003 | 0.015 | False |
| turkish (tur, mid) | 0.208 | 0.048 | 0.092 | False |
| vietnamese (vie, mid) | 0.012 | 0.297 | 0.308 | True |
| indonesian (ind, mid) | 0.027 | 0.173 | 0.254 | False |
| thai (tha, mid) | 0.046 | 0.121 | 0.128 | True |
| polish (pol, mid) | 0.360 | 0.233 | 0.179 | True |
| ukrainian (ukr, mid) | 0.419 | 0.030 | 0.030 | True |
| kannada (kan, low) | 0.034 | -0.002 | 0.019 | False |
| telugu (tel, low) | 0.028 | 0.018 | 0.018 | True |
| yoruba (yor, low) | 1.578 | 1.496 | 1.496 | True |
| sundanese (sun, low) | 0.873 | 0.032 | 0.021 | True |
| amharic (amh, low) | 1.020 | 1.210 | 1.209 | True |
| igbo (ibo, low) | 0.346 | 0.059 | 0.061 | True |
| hausa (hau, low) | 0.162 | 0.091 | 0.086 | True |
| khmer (khm, low) | 0.220 | 0.025 | 0.026 | True |
| lao (lao, low) | 1.915 | 0.037 | 0.104 | False |
| **median of the 27** | 0.262 | 0.048 | 0.062 |  |

### A11. GPTQ extension: share of active features that survive, per language, legacy RTN / matched RTN / GPTQ (6 and 4 bits)

Source: `reports/gptq_extension/results/stage1/stage1_paired_per_language.csv` (D-104; BOS masked, equal budget; percent). The median row is the median of the 27 languages, not the pooled figure of the text (INT4 12.7 / 18.6 / 30.5 %).

| language | INT6, legacy RTN | INT6, matched RTN | INT6, GPTQ | INT4, legacy RTN | INT4, matched RTN | INT4, GPTQ |
|---|---|---|---|---|---|---|
| english (eng, high) | 97.2 | 98.6 | 99.6 | 31.8 | 41.8 | 52.9 |
| german (deu, high) | 95.6 | 97.6 | 99.4 | 20.4 | 28.8 | 42.1 |
| spanish (spa, high) | 96.3 | 97.8 | 99.5 | 24.3 | 33.3 | 47.0 |
| french (fra, high) | 95.6 | 97.6 | 99.5 | 23.0 | 31.7 | 45.2 |
| portuguese (por, high) | 95.7 | 97.5 | 99.4 | 22.5 | 31.4 | 45.1 |
| russian (rus, high) | 95.5 | 97.3 | 99.4 | 20.3 | 29.0 | 42.5 |
| mandarin (zho, high) | 95.8 | 97.6 | 99.4 | 20.6 | 29.7 | 44.1 |
| japanese (jpn, high) | 93.5 | 96.4 | 99.1 | 15.4 | 23.3 | 37.1 |
| korean (kor, high) | 90.4 | 94.4 | 98.7 | 10.0 | 16.1 | 28.2 |
| hindi (hin, mid) | 90.3 | 94.2 | 98.6 | 11.5 | 18.1 | 30.1 |
| bengali (ben, mid) | 83.1 | 89.4 | 97.4 | 5.8 | 9.8 | 21.2 |
| tamil (tam, mid) | 82.4 | 88.8 | 97.3 | 4.6 | 7.9 | 19.3 |
| turkish (tur, mid) | 90.3 | 94.6 | 98.6 | 12.0 | 17.7 | 29.1 |
| vietnamese (vie, mid) | 93.5 | 96.2 | 98.9 | 14.6 | 23.0 | 35.4 |
| indonesian (ind, mid) | 94.8 | 96.9 | 99.2 | 19.1 | 27.9 | 41.1 |
| thai (tha, mid) | 92.2 | 95.3 | 99.0 | 9.1 | 16.2 | 29.9 |
| polish (pol, mid) | 94.6 | 96.7 | 99.1 | 16.2 | 23.8 | 36.8 |
| ukrainian (ukr, mid) | 93.8 | 96.1 | 99.1 | 15.1 | 22.2 | 34.8 |
| kannada (kan, low) | 78.5 | 86.2 | 96.8 | 4.3 | 7.4 | 19.4 |
| telugu (tel, low) | 80.8 | 87.3 | 97.0 | 4.9 | 8.3 | 18.2 |
| yoruba (yor, low) | 67.6 | 73.7 | 92.7 | 6.8 | 9.5 | 19.5 |
| sundanese (sun, low) | 78.0 | 85.5 | 95.8 | 8.4 | 12.3 | 20.0 |
| amharic (amh, low) | 62.8 | 72.1 | 94.2 | 3.8 | 6.4 | 17.4 |
| igbo (ibo, low) | 66.5 | 74.3 | 92.7 | 6.4 | 9.0 | 18.5 |
| hausa (hau, low) | 73.1 | 79.2 | 93.1 | 6.3 | 8.9 | 16.3 |
| khmer (khm, low) | 64.0 | 74.5 | 95.0 | 2.6 | 5.3 | 15.5 |
| lao (lao, low) | 53.0 | 66.7 | 93.5 | 2.8 | 4.9 | 17.1 |
| **median of the 27** | 90.4 | 94.6 | 98.7 | 11.5 | 17.7 | 29.9 |

### A12. GPTQ extension: loss increase on the same sentences (nats per token, quantized minus full precision), per language

Source: the same file (D-104). Lower is better.

| language | INT6, legacy RTN | INT6, matched RTN | INT6, GPTQ | INT4, legacy RTN | INT4, matched RTN | INT4, GPTQ |
|---|---|---|---|---|---|---|
| english (eng, high) | 0.006 | 0.005 | 0.002 | 0.161 | 0.099 | 0.066 |
| german (deu, high) | 0.021 | 0.013 | 0.004 | 0.387 | 0.281 | 0.151 |
| spanish (spa, high) | 0.018 | 0.017 | 0.004 | 0.288 | 0.204 | 0.110 |
| french (fra, high) | 0.014 | 0.012 | 0.005 | 0.318 | 0.226 | 0.117 |
| portuguese (por, high) | 0.019 | 0.013 | 0.004 | 0.353 | 0.239 | 0.112 |
| russian (rus, high) | 0.020 | 0.013 | 0.004 | 0.355 | 0.248 | 0.129 |
| mandarin (zho, high) | 0.024 | 0.023 | 0.009 | 0.385 | 0.264 | 0.138 |
| japanese (jpn, high) | 0.023 | 0.020 | 0.008 | 0.402 | 0.292 | 0.141 |
| korean (kor, high) | 0.026 | 0.022 | 0.007 | 0.519 | 0.367 | 0.145 |
| hindi (hin, mid) | 0.018 | 0.017 | 0.004 | 0.478 | 0.344 | 0.120 |
| bengali (ben, mid) | 0.022 | 0.018 | 0.006 | 0.600 | 0.387 | 0.119 |
| tamil (tam, mid) | 0.028 | 0.017 | 0.007 | 0.589 | 0.406 | 0.125 |
| turkish (tur, mid) | 0.033 | 0.021 | 0.007 | 0.547 | 0.380 | 0.181 |
| vietnamese (vie, mid) | 0.023 | 0.018 | 0.007 | 0.434 | 0.274 | 0.120 |
| indonesian (ind, mid) | 0.029 | 0.015 | 0.005 | 0.487 | 0.339 | 0.140 |
| thai (tha, mid) | 0.018 | 0.017 | 0.006 | 0.495 | 0.322 | 0.128 |
| polish (pol, mid) | 0.024 | 0.016 | 0.005 | 0.517 | 0.331 | 0.163 |
| ukrainian (ukr, mid) | 0.027 | 0.015 | 0.007 | 0.467 | 0.323 | 0.142 |
| kannada (kan, low) | 0.021 | 0.014 | 0.005 | 0.594 | 0.411 | 0.105 |
| telugu (tel, low) | 0.023 | 0.018 | 0.005 | 0.598 | 0.394 | 0.120 |
| yoruba (yor, low) | 0.023 | 0.018 | 0.001 | 0.573 | 0.372 | 0.088 |
| sundanese (sun, low) | 0.052 | 0.047 | 0.010 | 0.829 | 0.535 | 0.189 |
| amharic (amh, low) | 0.049 | 0.040 | 0.010 | 1.495 | 0.849 | 0.120 |
| igbo (ibo, low) | 0.026 | 0.023 | 0.004 | 0.662 | 0.451 | 0.134 |
| hausa (hau, low) | 0.045 | 0.021 | 0.009 | 0.833 | 0.562 | 0.193 |
| khmer (khm, low) | 0.026 | 0.024 | 0.005 | 0.829 | 0.499 | 0.109 |
| lao (lao, low) | 0.030 | 0.034 | 0.006 | 1.747 | 0.992 | 0.095 |
| **median of the 27** | 0.023 | 0.018 | 0.005 | 0.517 | 0.344 | 0.125 |

### A13. GPTQ extension: function of the language group (zeroing the 64 features), ratio of the effect to its full-precision value, per language

Source: `reports/functional_validation/results/eval/retention.csv` (legacy RTN) and `reports/gptq_extension/results/stage3/analysis_{rtnm,gptq}_eval/retention.csv` (D-107). ▲ = clearly amplified, ▽ = clearly attenuated, ? = inconclusive, no mark = within the band of no change.

| language | INT6, legacy RTN | INT6, matched RTN | INT6, GPTQ | INT4, legacy RTN | INT4, matched RTN | INT4, GPTQ |
|---|---|---|---|---|---|---|
| english (eng, high) | 0.96 | 0.96 | 0.98 | 0.97 | 1.03 | 0.89 |
| german (deu, high) | 1.09 | 1.07 | 0.94 | 0.72 ? | 0.91 | 1.01 |
| spanish (spa, high) | 1.04 | 1.08 | 0.91 | 1.28 ? | 2.13 ▲ | 1.26 ? |
| french (fra, high) | 1.16 | 1.13 | 1.01 | 1.54 ▲ | 1.10 | 1.01 |
| portuguese (por, high) | 1.09 | 1.10 | 0.97 | 0.95 | 0.66 ▽ | 0.92 |
| russian (rus, high) | 1.08 | 1.13 | 0.94 | 2.92 ▲ | 2.45 ▲ | 1.54 ▲ |
| mandarin (zho, high) | 1.06 | 1.01 | 0.98 | 1.08 | 1.36 ? | 1.07 |
| japanese (jpn, high) | 1.03 | 0.97 | 1.00 | 1.42 ▲ | 1.98 ▲ | 1.27 |
| korean (kor, high) | 0.88 | 0.85 | 0.97 | 2.63 ▲ | 1.38 ? | 1.70 ▲ |
| hindi (hin, mid) | 1.36 ? | 1.31 ? | 1.01 | 2.90 ▲ | 2.08 ▲ | 1.57 ▲ |
| bengali (ben, mid) | 0.94 | 0.92 | 1.09 | 3.89 ▲ | 2.73 ▲ | 1.42 ? |
| tamil (tam, mid) | 1.01 | 1.04 | 1.05 | 2.63 ▲ | 2.11 ▲ | 1.22 ? |
| turkish (tur, mid) | 0.96 | 1.03 | 0.98 | 2.75 ▲ | 1.95 ▲ | 1.33 ? |
| vietnamese (vie, mid) | 0.83 ? | 1.50 ? | 1.04 | 5.70 ▲ | 3.90 ▲ | 3.45 ▲ |
| indonesian (ind, mid) | 0.99 | 1.03 | 0.93 | 1.05 ? | 1.58 ? | 1.28 ? |
| thai (tha, mid) | 1.11 | 1.17 | 1.01 | 3.74 ▲ | 3.21 ▲ | 1.64 ▲ |
| polish (pol, mid) | 0.92 | 1.07 | 0.98 | 1.58 ▲ | 0.97 | 0.94 |
| ukrainian (ukr, mid) | 1.15 | 1.06 | 0.99 | 2.30 ▲ | 1.90 ▲ | 1.49 ▲ |
| kannada (kan, low) | 1.00 | 0.96 | 1.01 | 1.29 ? | 1.17 | 1.04 |
| telugu (tel, low) | 1.24 ? | 1.18 | 1.02 | 5.75 ▲ | 4.34 ▲ | 1.79 ▲ |
| yoruba (yor, low) | 1.02 | 1.02 | 1.00 | 1.38 ▲ | 1.15 | 1.13 |
| sundanese (sun, low) | 0.90 | 0.92 | 1.00 | 1.18 | 0.81 | 0.90 |
| amharic (amh, low) | 1.15 | 1.07 | 1.01 | 2.31 ▲ | 1.35 ? | 1.23 |
| igbo (ibo, low) | 0.98 | 0.98 | 1.00 | 1.25 | 1.14 | 1.02 |
| hausa (hau, low) | 1.03 | 1.04 | 1.00 | 1.29 ? | 1.20 | 1.06 |
| khmer (khm, low) | 1.03 | 1.02 | 0.99 | 1.89 ▲ | 1.35 ? | 1.09 |
| lao (lao, low) | 1.00 | 1.02 | 0.98 | 1.27 | 1.19 | 1.15 |
| **median of the 27** | 1.03 | 1.04 | 1.00 | 1.54 | 1.36 | 1.22 |

### A14. GPTQ extension: intervention transfer, main matrix, INT4 context part (C) and weights part (W) of the group's harm, per language

Source: `reports/intervention_transfer/results/main/group_per_language.csv` (legacy RTN) and `reports/gptq_extension/results/stage2/analysis_{rtnm,gptq}/group_per_language.csv` (D-105). Same quantities as A8, nats per token.

| language | INT4 C, legacy RTN | INT4 W, legacy RTN | INT4 C, matched RTN | INT4 W, matched RTN | INT4 C, GPTQ | INT4 W, GPTQ |
|---|---|---|---|---|---|---|
| english (eng, high) | -0.011 | +0.012 | +0.000 | -0.000 | -0.008 | -0.008 |
| german (deu, high) | +0.964 | -0.224 | +0.728 | +0.050 | +0.248 | +0.037 |
| spanish (spa, high) | +0.755 | +0.125 | +0.885 | +0.302 | +0.288 | +0.110 |
| french (fra, high) | +0.677 | +0.509 | +0.912 | +0.148 | +0.093 | +0.038 |
| portuguese (por, high) | -0.336 | +0.706 | +0.180 | +0.029 | -0.046 | +0.134 |
| russian (rus, high) | +0.206 | +0.096 | +0.167 | +0.063 | +0.052 | +0.014 |
| mandarin (zho, high) | +1.431 | +0.577 | +1.408 | +1.494 | +0.543 | +0.575 |
| japanese (jpn, high) | +0.421 | +0.104 | +0.316 | +0.182 | +0.060 | +0.044 |
| korean (kor, high) | +0.475 | +0.459 | +0.569 | +0.034 | +0.400 | +0.094 |
| hindi (hin, mid) | +0.310 | +0.125 | +0.175 | +0.113 | +0.075 | +0.039 |
| bengali (ben, mid) | +0.019 | +0.022 | +0.003 | +0.013 | +0.001 | +0.003 |
| tamil (tam, mid) | +0.029 | +0.025 | +0.013 | +0.024 | +0.003 | +0.005 |
| turkish (tur, mid) | +0.513 | +0.330 | +0.446 | +0.263 | +0.156 | +0.058 |
| vietnamese (vie, mid) | +0.076 | +0.042 | +0.036 | +0.041 | +0.017 | +0.022 |
| indonesian (ind, mid) | +0.002 | -0.001 | -0.000 | +0.011 | -0.003 | +0.011 |
| thai (tha, mid) | +0.087 | +0.091 | +0.050 | +0.110 | +0.040 | +0.012 |
| polish (pol, mid) | +0.337 | +0.089 | +0.328 | +0.048 | +0.090 | +0.026 |
| ukrainian (ukr, mid) | +0.636 | +0.438 | +0.414 | +0.581 | +0.244 | +0.147 |
| kannada (kan, low) | +0.009 | +0.008 | +0.005 | +0.003 | +0.004 | -0.002 |
| telugu (tel, low) | +0.073 | +0.081 | +0.041 | +0.084 | +0.025 | +0.012 |
| yoruba (yor, low) | +0.676 | +0.494 | +0.419 | +0.350 | +0.138 | +0.206 |
| sundanese (sun, low) | +0.084 | +0.225 | -0.052 | +0.111 | -0.033 | -0.031 |
| amharic (amh, low) | +0.409 | +1.113 | +0.177 | +0.344 | +0.127 | +0.127 |
| igbo (ibo, low) | +0.094 | +0.082 | +0.058 | +0.046 | +0.035 | +0.000 |
| hausa (hau, low) | +0.101 | +0.034 | +0.043 | +0.034 | +0.025 | +0.005 |
| khmer (khm, low) | +0.154 | +0.071 | +0.035 | +0.110 | +0.021 | +0.012 |
| lao (lao, low) | +0.892 | +1.196 | +0.035 | +1.125 | +0.042 | +0.340 |
| **median of the 27** | +0.206 | +0.096 | +0.167 | +0.084 | +0.042 | +0.026 |

### A15. GPTQ extension: Study G, net harm per language (group minus matched controls) when the group is scaled down to ×0.088 and up to ×11.3, full precision and INT4

Source: `reports/functional_validation/gain_study/results/curves.csv` (legacy RTN and full precision; the full-precision file is the same in every arm) and `reports/gptq_extension/results/gain/{rtnm,gptq}/curves.csv` (D-109). Nats per token.

| language | FP, group ×0.088 | INT4 ×0.088, legacy RTN | INT4 ×0.088, matched RTN | INT4 ×0.088, GPTQ | FP, group ×11.3 | INT4 ×11.3, legacy RTN | INT4 ×11.3, matched RTN | INT4 ×11.3, GPTQ |
|---|---|---|---|---|---|---|---|---|
| english (eng, high) | +0.107 | +0.113 | +0.119 | +0.099 | -0.220 | -0.383 | -0.253 | -0.218 |
| german (deu, high) | +0.422 | +0.438 | +0.490 | +0.481 | +1.529 | +1.312 | +1.466 | +1.565 |
| spanish (spa, high) | +0.216 | +0.320 | +0.562 | +0.306 | +0.721 | +0.436 | +0.555 | +0.674 |
| french (fra, high) | +0.307 | +0.586 | +0.393 | +0.333 | +0.844 | +0.599 | +0.582 | +0.866 |
| portuguese (por, high) | +0.563 | +0.619 | +0.405 | +0.555 | +0.893 | +0.940 | +0.853 | +0.993 |
| russian (rus, high) | +0.032 | +0.143 | +0.105 | +0.058 | +1.235 | +0.525 | +0.854 | +1.146 |
| mandarin (zho, high) | +5.460 | +6.442 | +8.547 | +6.208 | +1.613 | +1.239 | +1.763 | +1.580 |
| japanese (jpn, high) | +0.157 | +0.256 | +0.331 | +0.207 | +0.999 | +0.493 | +0.737 | +0.981 |
| korean (kor, high) | +0.193 | +0.708 | +0.347 | +0.393 | +0.837 | +0.403 | +0.573 | +0.831 |
| hindi (hin, mid) | +0.052 | +0.198 | +0.131 | +0.089 | +1.541 | +1.069 | +1.707 | +1.626 |
| bengali (ben, mid) | +0.008 | +0.031 | +0.022 | +0.012 | +0.635 | +0.451 | +0.712 | +0.666 |
| tamil (tam, mid) | +0.022 | +0.063 | +0.050 | +0.027 | +2.051 | +1.624 | +1.748 | +2.409 |
| turkish (tur, mid) | +0.126 | +0.398 | +0.281 | +0.176 | +0.504 | -0.446 | -0.083 | +0.276 |
| vietnamese (vie, mid) | +0.001 | +0.051 | +0.030 | +0.020 | +0.772 | +0.519 | +0.548 | +0.755 |
| indonesian (ind, mid) | +0.018 | +0.022 | +0.036 | +0.027 | +0.221 | +0.332 | +0.311 | +0.271 |
| thai (tha, mid) | +0.031 | +0.128 | +0.107 | +0.052 | +0.069 | -0.402 | -0.053 | +0.041 |
| polish (pol, mid) | +0.179 | +0.387 | +0.225 | +0.187 | +1.390 | +1.242 | +1.194 | +1.368 |
| ukrainian (ukr, mid) | +0.205 | +0.642 | +0.473 | +0.350 | +2.530 | +1.608 | +1.799 | +2.359 |
| kannada (kan, low) | +0.026 | +0.037 | +0.033 | +0.028 | +1.213 | +1.927 | +1.860 | +1.505 |
| telugu (tel, low) | +0.020 | +0.130 | +0.097 | +0.039 | +1.227 | +0.810 | +1.334 | +1.312 |
| yoruba (yor, low) | +1.146 | +1.714 | +1.390 | +1.303 | -0.015 | -0.268 | -0.131 | +0.076 |
| sundanese (sun, low) | +0.677 | +0.887 | +0.566 | +0.608 | +0.311 | +0.180 | +0.339 | +0.372 |
| amharic (amh, low) | +0.768 | +1.765 | +1.021 | +0.909 | +6.179 | +4.147 | +6.148 | +5.972 |
| igbo (ibo, low) | +0.278 | +0.374 | +0.332 | +0.286 | +1.739 | +1.231 | +1.371 | +2.098 |
| hausa (hau, low) | +0.129 | +0.174 | +0.162 | +0.139 | +1.326 | +1.276 | +1.214 | +1.408 |
| khmer (khm, low) | +0.169 | +0.323 | +0.238 | +0.181 | +5.430 | +2.674 | +5.313 | +5.635 |
| lao (lao, low) | +1.169 | +1.997 | +1.841 | +1.449 | +7.159 | +4.490 | +6.022 | +7.334 |
| **median of the 27** | +0.169 | +0.323 | +0.281 | +0.187 | +1.213 | +0.810 | +0.854 | +1.146 |

### A16. GPTQ extension: intervention transfer, external panel (six languages, Wikipedia text), INT4 context part (C) and weights part (W) of the group's harm, per language

Source: `reports/intervention_transfer/results/external/group_per_language.csv` (legacy RTN) and `reports/gptq_extension/results/external/analysis_{rtnm,gptq}/group_per_language.csv` (D-111). Same quantities as A9 and A14, nats per token. GPTQ's calibration text is Wikipedia too (different articles), so part of its edge here may be domain match.

| language | INT4 C, legacy RTN | INT4 W, legacy RTN | INT4 C, matched RTN | INT4 W, matched RTN | INT4 C, GPTQ | INT4 W, GPTQ |
|---|---|---|---|---|---|---|
| english (eng, high) | +0.001 | +0.005 | +0.002 | -0.004 | -0.002 | -0.002 |
| spanish (spa, high) | +0.534 | +0.118 | +0.498 | +0.202 | +0.069 | +0.018 |
| russian (rus, high) | +0.133 | +0.064 | +0.097 | +0.064 | +0.028 | +0.007 |
| mandarin (zho, high) | +0.633 | +1.030 | +0.466 | +1.577 | +0.069 | +0.254 |
| hindi (hin, mid) | +0.126 | +0.073 | +0.056 | +0.062 | +0.023 | +0.022 |
| yoruba (yor, low) | +0.579 | +0.479 | +0.174 | +0.317 | +0.140 | +0.121 |
| **median of the 6** | +0.333 | +0.096 | +0.135 | +0.133 | +0.048 | +0.020 |

<!-- END GENERATED TABLES -->
