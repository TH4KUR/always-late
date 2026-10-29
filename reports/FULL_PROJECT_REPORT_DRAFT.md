# Does compressing a language model damage its language features? The full story of the "Always Late" project (draft for the team)

**Status of this document.** Draft, written on 29 October 2026, for the team to use when writing the final course report. It covers the whole project, from the first pipeline at the end of September 2026 to 29 October 2026. It adds no new experiment. Every number comes from a result file in this repository; the file is named next to the number. Numbers that I recomputed myself from the result files are listed in Appendix C, together with the few I could only re-read or could not check. Where two project documents disagree, I say so and say which one the underlying result file supports.

**Main sources.** The plain-language results page ([`PROJECT_RESULTS.md`](PROJECT_RESULTS.md)), the independent audit ([`analysis/OPUS_DETAILED_ANALYSIS.md`](analysis/OPUS_DETAILED_ANALYSIS.md)), the decision log ([`decisions_and_anomalies.md`](decisions_and_anomalies.md), entries D-001 to D-125 and anomalies A-001 to A-042), the open-questions file ([`../DECISIONS_FOR_REVIEW.md`](../DECISIONS_FOR_REVIEW.md)), the original proposal ([`../anlp_revised.txt`](../anlp_revised.txt)), the dated log ([`progress_log.md`](progress_log.md)) and the numbered handoffs ([`handoffs/`](handoffs/)).

**How to read the evidence labels.** Every finding in this document carries one label.

| label | what it means here |
|---|---|
| **ESTABLISHED** | Large, consistent and controlled within our setting (one model, one layer, one feature dictionary, simulated weight rounding, 27 languages). It held up under the checks we ran. It can still be "exploratory" in the statistical sense (see below). |
| **SUGGESTIVE** | A signal that looks real but is small, decided after seeing data, under-powered, sensitive to analysis choices, or not repeated. |
| **NULL** | Tested with an adequate design (a positive control or a stated bound) and no effect was found. |
| **INCONCLUSIVE** | The data cannot decide. Too little power, an unclear measure, or the answer changes with a reasonable analysis choice. |
| **NOT SUPPORTED** | Tested, and the data argue against the claim or fail the rule we set. |
| **NOT TESTED** | No experiment in the project addresses it. |
| **UNVERIFIED** | Results produced elsewhere and imported, not yet reproduced by us. (The Gemma-2-9B tables were such results until the evening of 29 October; we then reproduced them from the raw files, see section 5.15.) |

Two more words are used all through. **Confirmatory** means the test and its pass rule were written down and committed before the data were scored, and the data were fresh. **Exploratory** means everything else: still worth reporting, but not proof. Most of this project is exploratory, because later studies reused test sentences that earlier studies had already looked at.

---

## Abstract

**What we asked.** Large language models are often made smaller by *quantization*: storing each weight with fewer bits (for example 4 bits instead of 16). Behaviour tests in earlier work suggest that this hurts some languages, especially low-resource ones, more than others. We asked whether the damage shows up inside the model, in the "language features" that a *sparse autoencoder* (a tool that splits the model's internal state into many readable parts) finds. Our first hypothesis was: quantization destroys language-specific features more than other features, and more for low-resource languages.

**What we did.** We used one model (Gemma-2-2B), one sparse autoencoder (Gemma Scope, 16,384 features, at layer 12 of 26), and 27 languages from the FLORES-200 parallel corpus, nine each in three resource tiers. We rounded the model's weights to 8, 7, 6, 5 and 4 bits with three different methods (the project's original "legacy" round-to-nearest, a standard "matched" round-to-nearest, and GPTQ, a smarter method that corrects rounding errors). We then ran about fifteen studies: feature "survival" (does a feature's activity pattern stay the same?), language-tag retention, removal and scaling of each language's 64 most language-specific features, patching, a reading-comprehension test (Belebele), a check on a second model (Llama-3.2-1B), an "intervention transfer" study, and a study of what the survival number really measures.

**What we found (with honest labels).**
1. **The starting hypothesis is not supported.** In the three tests we fixed in advance, language-specific features did not survive rounding clearly worse than matched ordinary features (largest gap 1.44 percentage points at 6 bits, corrected p = 0.13). The apparent gap is explained by how strongly features fire. The null is also fragile: under a standard round-to-nearest the same tiny gap crosses the significance line, and under GPTQ it disappears. *NOT SUPPORTED as a language-specific effect.*
2. **Low-resource languages do lose more features at 6 bits (69 % against 95 % for high-resource), but this is not a language-specific effect of quantization.** It tracks the size of the features' activations, the underlying sensitivity is already visible at 16 bits (where the weights barely change) in threshold-free measures, and GPTQ removes most of the gap. *ESTABLISHED as a description; NOT SUPPORTED as language damage.*
3. **Each language's 64 most selective features matter for predicting that language's text.** Switching them off raises the language's next-word loss far more than switching off matched control features, in 27 of 27 languages (median extra loss 0.262 nats per token). This dependence stays within about 10 % of its full-precision size down to 5 bits and grows at 4 bits (median ratio 1.54 with legacy rounding, 1.22 with GPTQ). *ESTABLISHED in this setting.*
4. **The same features do not matter measurably for reading comprehension.** On Belebele (18 languages, 385 questions each), removing them changes accuracy by +0.04 points relative to matched controls (my rough 95 % range −0.43 to +0.57), while a positive control shows the test can detect changes of 1 to 6 points. This was our only fully confirmatory test. *NULL.*
5. **Survival numbers and language tags are poor guides to whether a feature still does the same job.** Six different analyses agree: the language tag predicts functional change no better than chance (AUROC 0.506), the activation correlation adds only 3 to 4 % to predicting functional change (target was 10 %), and most change at 4 bits is shared with random control directions. The survival rate mostly measures feature size and a quantizer-wide shift. *ESTABLISHED as a measurement finding; this is the project's strongest result.*
6. **Putting the full-precision language features back into a 4-bit model repairs only 2.6 % of its extra loss** (the whole layer-12 state repairs 39 %). This tells us what this patch can repair, not where the damage "sits". *Measurement ESTABLISHED; location claim NOT TESTED / not identifiable.*
7. **Almost every effect size depends strongly on the quantizer**, the direction mostly does not. GPTQ beats legacy rounding in 27 of 27 languages at 4 and 6 bits. *ESTABLISHED.*
8. **The picture did not carry over to Llama-3.2-1B** (pre-set verdict "not consistent"; many confounds). Gemma-2-9B results produced by another person report "consistent" for legacy rounding only; they are imported; on 29 October we **reproduced their analysis tables byte for byte from the raw files**, but two of their prediction texts have problems and their wording goes further than the data. *Llama: NOT SUPPORTED / INCONCLUSIVE; 9B: computation REPRODUCED, meaning SUGGESTIVE.*

**What we could not show.** That quantization hurts low-resource languages *because* their language features break. Where in the model the quantization damage arises. Why loss and reading comprehension disagree. Anything about real integer kernels, activation quantization, other layers or other dictionaries. A planned arithmetic-precision audit was built but never run, and a GPTQ run on Gemma-2-9B was prepared but never run.

---

## 1. Introduction and background (in simple words)

### 1.1 Language models and "loss"

A **language model** reads text one piece at a time and guesses the next piece. The pieces are called **tokens**; a token is a word or part of a word. The model gives a probability to every possible next token.

The **loss** (also "next-token loss" or "negative log-likelihood") measures how surprised the model is by the real next token. It is measured in **nats per token**. Lower is better. On our test text the full model's loss is about 3 nats per token, so a change of 0.26 nats per token is roughly 9 % worse. **Perplexity** is the same thing on another scale (perplexity = e to the power of the loss).

A model is built from **layers** stacked on top of each other. Between layers it keeps a long list of numbers for each token, called the **residual stream** or "hidden state". Gemma-2-2B has 26 layers; we look at the state after layer 12, in the middle.

### 1.2 Features and sparse autoencoders

The residual stream is hard to read: thousands of numbers, all mixed. A **sparse autoencoder (SAE)** is a separate small network trained to rewrite that state as a sum of a few "directions" taken from a large dictionary. Each direction is called a **feature**. "Sparse" means only a few features are active for any token. Many features are easy to interpret: one fires on French text, one on numbers, and so on.

We used **Gemma Scope** (Lieberum et al., BlackboxNLP 2024, P03 as registered in [`research_sources.md`](research_sources.md)), a public set of SAEs trained by Google for Gemma 2 models. Ours has 16,384 features and reads layer 12. Its features use a **JumpReLU** rule: a feature is "on" only when its input passes a learned **threshold**. We did not train any SAE. We used it as a fixed measuring tool, both for the original model and for the compressed one.

A feature **fires** (is **active**) on a token when its value is above zero. Its **firing rate** is how often it fires. Its **activation** is how strongly it fires.

### 1.3 Quantization

A model's weights are normally stored as 16-bit numbers. **Quantization** stores them with fewer bits. "INT8" means 8-bit whole numbers; "INT4" means only 16 possible values per weight group. Fewer bits make the model smaller and faster but rougher.

We studied three ways of choosing the rounded weights:

- **RTN (round-to-nearest).** Each weight is rounded to the closest allowed value. This is the simplest method. The project's original version ("**legacy RTN**") came from the earlier paper we replicated. It had an unusual detail: it shared one scale across the attention heads of a layer.
- **Matched RTN.** The same rounding done the standard way: one scale per output channel, on the same grid as GPTQ. We added it to separate "a better implementation of rounding" from "a smarter method".
- **GPTQ** (Frantar et al., arXiv 2210.17323, as registered in [`quantization_gptq_audit_2026-10-27.md`](quantization_gptq_audit_2026-10-27.md)). It rounds weights one at a time and adjusts the not-yet-rounded weights to cancel the error, using some example text called **calibration text**.

All our quantization is **simulated** ("fake quantization"): we round the weights, store them back as ordinary numbers and run the model as usual. We never used real integer computer kernels, and we never rounded activations, only weights. We also used three control settings: **INT16** and **INT12** (rounding so fine that the weights barely change; the "rounding only" baseline) and **INT0** (put the original weights back; an identity check that nothing else changed).

### 1.4 Why anyone would worry about languages

Behaviour studies have found that quantization hurts some languages more than others, often lower-resource languages and languages not written in Latin script (Marchisio et al., Findings of EMNLP 2024, P23; Marie and Fujita, arXiv 2508.20893, P25; other work disagrees, e.g. UniComp, arXiv 2602.09130, P24; all as registered in [`research_sources.md`](research_sources.md)). Inside models, earlier work found SAE features that are specific to one language (Deng et al., ACL 2025, P02), and found that low- and mid-resource languages get weaker SAE activations in Gemma-2-2B (Sin Jing Xuan et al., arXiv 2507.18918, P17). So a natural question is: do quantization's language disparities run *through* the loss of language-specific features?

The **resource tier** of a language is a rough label for how much text exists for it. We used the proposal's three tiers (high, mid, low), assigned from Joshi et al.'s (2020) language classes as cited in the proposal, cross-checked against training-corpus size (decision D-011). Tier is an observed property of a language. It is not something we can change in an experiment, so it can only show associations, not causes.

### 1.5 The measures we used, explained once

| measure | plain meaning | why we used it |
|---|---|---|
| **survival** | A feature "survives" if, over many tokens, its activations in the compressed model still correlate at r ≥ 0.9 with the full-precision model (Pearson correlation). This is the measure from the paper we replicated. | It was the field's existing yardstick. |
| **language-selective feature** | A feature that fires much more often on one language than on the others. Our score S(f) is the highest per-language firing rate divided by the average rate. A feature counts as selective if S(f) is above the 95th percentile of the same score with shuffled language labels (the cutoff came out at 1.83). | To find the "language features". |
| **matched control features** | Ordinary features chosen to fire about as often (and, in later studies, about as strongly) as the language features, but not language-specific. | Rare and weak features behave differently; a fair comparison must hold that fixed. |
| **association retention** | Whether a feature still has its original language as its clear top language after compression. | A second, different meaning of "the feature still belongs to its language". |
| **ablation** | Switching features off. **Zero-ablation** sets them to zero. Our edits remove each chosen feature's contribution from the residual stream and keep the SAE's reconstruction error, so only those features change. | To test whether the features *matter*. |
| **net harm, effect E** | Extra loss from removing the language's group minus extra loss from removing matched controls. In nats per token. | "How much more these features matter than ordinary ones." |
| **retention (function)** | Net harm in the compressed model divided by net harm in the full model. 1 = unchanged, above 1 = bigger. | To see if the dependence on the features survives compression. |
| **gain scaling** (Study G) | Multiplying a group's activations by a factor: below 1 turns it down, above 1 turns it up. | A dose-response test, finer than on/off. |
| **patching / rescue** | Copying the full-precision values of chosen parts back into the compressed model, then measuring how much of its extra loss goes away. | To see what a repair of those parts achieves. |
| **effect fingerprint, W1** | How switching one feature off changes the output probabilities; W1 is the cosine similarity between that change in the full model and in the compressed model (1 = same effect). | To ask "does the feature still do the same thing?" |
| **numerical floor** | How much a measure moves from tiny rounding alone (INT16 or re-batching). | Effects near the floor cannot be trusted. |

### 1.6 The setting in one table

| element | value |
|---|---|
| model | Gemma-2-2B (weights run in bfloat16) |
| SAE | Gemma Scope, `layer_12/width_16k/canonical` (16,384 features, JumpReLU), reading the residual stream after layer 12 |
| quantizers | legacy RTN; matched RTN; GPTQ (calibrated on 8,192 Wikipedia tokens in each of 27 languages, 221,184 in all) |
| bit widths | INT8, INT7, INT6, INT5, INT4; controls INT16, INT12, INT0 |
| text | FLORES-200: "dev" split (997 sentences per language) to choose features, "devtest" split (1,012) to evaluate |
| languages | 27, nine per tier (Appendix A1 of [`PROJECT_RESULTS.md`](PROJECT_RESULTS.md)) |
| start-of-text token | one start token (BOS) added to every sentence and left out of every statistic (the "mask" regime, D-001, D-028) |
| second models | Llama-3.2-1B with a third-party TopK SAE (exploratory); Gemma-2-9B with Gemma Scope at layer 20 (imported, unverified, legacy RTN only) |
| compute | the university cluster "Ada" (mostly RTX 2080 Ti GPUs); one rented GPU for the 9B run by another person |

---

## 2. Related work, and what is new here

We only cite sources the project registered, with the identifiers used in [`research_sources.md`](research_sources.md), [`functional_validation/prior_art.md`](functional_validation/prior_art.md) and the prior-art section of the measurement-validity plan ([`../docs/superpowers/plans/2026-10-29-survival-measurement-validity-study.md`](../docs/superpowers/plans/2026-10-29-survival-measurement-validity-study.md)). Venue statements are as recorded there; several sources are preprints.

### 2.1 The paper we started from

**Duan (2026), arXiv 2606.03002 (P01, preprint; v2 is titled "Perplexity Can Miss SAE Feature Damage Under Quantization").** Same model, same SAE and layer, same RTN bit widths. It reported that at 6 bits the model's perplexity barely changes while only about 51 % of active SAE features "survive" (r ≥ 0.9), on English WikiText. It treated all features as one group and had no languages. Our project began as an extension of this study to 27 languages. Duan names causal validation as future work.

### 2.2 Language features in SAEs

- **Deng et al., ACL 2025 (P02).** Found language-specific SAE features in Gemma-2-2B with Gemma Scope on FLORES. Removing them raises that language's loss. Full precision only, no compression.
- **Sin Jing Xuan, Huseynov and Zhang, arXiv 2507.18918 (P17, preprint).** Reported weaker SAE activations for medium- and low-resource languages in Gemma-2-2B. This is close to our finding that activation scale explains the tier gap (A-019).
- **Nguyen et al., arXiv 2609.04808 (P44); Andrylie et al., arXiv 2507.11230 (P34); Chou et al., arXiv 2507.13410 (P35); Rahmanisa et al., arXiv 2507.22581 (P46).** Language features or neurons can be ablated, amplified or used for steering at full precision. Rahmanisa et al. is the closest work to our "turn up" direction (neurons, not SAE features, no quantization).

### 2.3 Feature survival under compression

- **Borobia, Seguí-Mas and Tormo-Carbó: arXiv 2603.25325 (P31, preprint) and the Neurocomputing 2026 article (P19, DOI 10.1016/j.neucom.2026.134826).** Pruning, not quantization (in the version read), on Gemma-2-2B and others. They report that geometric survival of features does not predict their causal importance. This is the closest precedent for our "survival is not function" finding. By 26 October their revised code repository also contained INT8/INT4 pilots ([`novelty_assessment_2026-10-26.md`](novelty_assessment_2026-10-26.md)), so "pruning only" can no longer be used to claim novelty.
- **arXiv 2607.12166** (as registered in the prior-art table of the measurement-validity plan, with the one-line summary "correlational recovery overstates causal validity"; the register gives no title or authors). It already states the general form of our claim that correlation-based recovery overstates causal validity.
- **Gupte, Chhabra and Khalili, arXiv 2507.15977 (P20).** Full-precision SAEs still work on a pruned Gemma-2-2B at the reconstruction level.
- **Bhandari and Wade, arXiv 2609.06473 (P28).** Steering-vector directions stay highly similar under INT8/NF4 quantization. No SAE, no languages.
- **Sathyanarayanan, arXiv 2609.23125 (P30).** Restoring chosen SAE features to full precision after *activation* quantization. Our patching study (F3) is not new as a method.
- **Chang, Zhang, Thomason and Jia, EMNLP 2025 (P11).** Inputs with small residual norms break more under low-bit quantization; patching full-precision module outputs into 3-bit models. The closest precedent for our context-versus-weights question, at the loss level.
- **Hoang et al., arXiv 2607.24645 (P43, "FEGA").** Measures the cloud of output changes when one feature is removed across contexts. Same kind of object as our effect fingerprint (F5).
- **Cho et al., EMNLP 2026 (P41).** Single-feature ablation; necessity depends on the SAE family. Relevant to why a Llama result need not match a Gemma one.

### 2.4 Quantization and languages, and quantization methods

- Behaviour: Marchisio et al. (P23), Marie and Fujita (P25), UniComp (P24, counter-evidence), Chimoto et al., EACL 2026 (P27, calibration language matters for GPTQ/AWQ).
- Methods: AWQ (Lin et al., MLSys 2024, P10), GPTQ (arXiv 2210.17323).

### 2.5 Method references we leaned on

Bootstrap intervals over sentences (Koehn, EMNLP 2004, M01; Berg-Kirkpatrick et al., EMNLP-CoNLL 2012, M02); resampling whole clusters such as languages (Field and Welsh, 2007, M04); good practice for activation patching (Zhang and Nanda, ICLR 2024, M06; Heimersheim and Nanda, arXiv 2404.15255, M07); keeping the SAE error term in feature edits and matching controls on activity (Marks et al., ICLR 2025, and Makelov et al., ICLR 2025, P36, P08); mediation needs explicit assumptions (Imai, Keele and Yamamoto, Statistical Science 2010, P12); the start token acts as an "attention sink" (Owen et al., arXiv 2503.22329, P04; Barbero et al., arXiv 2504.02732, P05); SAEs are not canonical units (Leask et al., ICLR 2025, P06; Paulo and Belrose, arXiv 2501.16615, P47).

### 2.6 What is new here, stated carefully

The project's own dated novelty assessment ([`novelty_assessment_2026-10-26.md`](novelty_assessment_2026-10-26.md)) concludes that **the broad novelty claim must be withdrawn**: "survival is not function", "patching full-precision activations", "fingerprints of feature effects" and "context matters more than late weights" all have precedents. What we did not find in the (bounded, not exhaustive) searches:

- a 27-language, tier-balanced study of language-selective SAE features under weight quantization with activity-matched controls;
- the same experiments under three quantizers on the same features, sentences and controls;
- a swapped-direction control for single-feature effects (another feature's direction removed at the same places and strengths);
- a decomposition of the survival correlation into gain, noise and on/off flips, and a perturbation measure in units of each feature's own threshold.

"Not found" is not "first". Any paper must repeat the search before submission.

---

## 3. The research questions

### 3.1 The original questions (the proposal)

The proposal ([`../anlp_revised.txt`](../anlp_revised.txt), "Does Compression Forget Language? Language-Specific Feature Survival Under Quantization and Pruning in Multilingual LLMs") asked three questions:

- **RQ1.** Do language-specific SAE features survive compression (quantization *and pruning*) at different rates than shared features?
- **RQ2.** Does the survival rate of language-specific features depend on the resource level of their language, across a script-balanced panel of 27 languages?
- **RQ3.** Does geometric feature survival predict functional or causal importance after compression? (Phase 2, an optional extension.)

It also planned: a Phase 0 replication of Duan (2026) with their code; layers 3 and 20 as exploratory extras; a Llama-3.2-1B extension; magnitude and Wanda pruning at matched perplexity as an extension; and a "behavioural link" between feature damage and loss.

**What happened to the planned extras.** Pruning was **never started** (NOT TESTED). Layers 3 and 20 were **never run** (NOT TESTED). The Llama-3.2-1B extension was run later, in a reduced form (section 5.10). Phase 2 (RQ3) grew into most of the project.

### 3.2 How the questions changed, and why

| when | change | why (decision entries) |
|---|---|---|
| 15–16 Oct | Replicate Duan with *their own* code; retire our first pipeline | Our pipeline disagreed with theirs; the audit showed the cause was a setting difference (the start token), not a bug; their code reproduced their table (D-002, D-009) |
| 17–21 Oct | Use only the "mask" regime (start token present but ignored in statistics) | Without a start token the model is badly degraded on short sentences and the noise-floor check fails (A-029); the SAE was trained this way (D-001, D-027, D-028) |
| 18–19 Oct | Phase 1 tests null; follow-ups show feature size explains the gap | Results (A-034, A-035, A-039) |
| 21 Oct | Publishability becomes the main goal; questions may change with recorded reasons | User decision (D-029) |
| 21 Oct | Reassess what "survival" means; include 5 and 4 bits | Phase 1 was null and survival at 7 bits sat at the ceiling (D-030, D-031) |
| 21 Oct | Report magnitude matching as association only; separate "overall tier gap" from "extra gap for selective features" | Causal language had been too loose (D-032, D-033, D-034) |
| 21 Oct | New study: does a feature keep its language tag? | User's choice among options (D-035) |
| 21–23 Oct | Functional studies F1–F5 and a reading test (Belebele) as the only confirmatory family | Tag retention is not function; a causal check was needed (D-039, D-040, D-043, D-053) |
| 24 Oct | Study G (graded gain) made "the major study" | User's instruction (D-070) |
| 26 Oct | Second model: Llama-3.2-1B, then Gemma-2-9B as the proper confirmation | Llama mixes model, SAE type, recipe and depth (D-087, D-089) |
| 26–27 Oct | Intervention transfer (does an edit made in the full model work the same in the compressed one?) | Novelty search showed the broad claims were covered; a design by Codex was checked and adopted (D-088) |
| 27 Oct | HOLD to finalise; then lifted for one purpose: re-run everything with GPTQ and matched RTN | User wanted a fair comparison of quantizers (D-101) |
| 28 Oct | Does survival predict whether an intervention keeps its effect? | Post hoc "publication rescue" plan, validity-checked (D-110 to D-114) |
| 29 Oct | Paper direction: what does the survival number measure? | Survival tracked size and quantizer, not function (D-117 to D-124) |
| 29 Oct | Gemma-2-9B legacy-RTN results imported for review | User request (D-125) |

**Reading of the drift.** The questions moved from "do language features survive worse?" (answer: no, not in a robust way), to "do language features matter, and does that survive compression?" (they matter for loss; mostly yes down to 5 bits), to "what do survival metrics actually measure?" (mostly feature size and a quantizer-wide shift). Every change has a recorded reason, and the earlier results were kept. The cost is that almost everything after Phase 1 is exploratory, because it reused FLORES test sentences that had already been examined.

### 3.3 The final list of questions we explored

| id | question | short answer | label |
|---|---|---|---|
| Q-A | Is our measuring setup valid against the paper we replicated? | Yes with their code; their survival numbers depend strongly on the start token | ESTABLISHED |
| Q-B (RQ1) | Do language-selective features survive rounding worse than matched features? | No robust difference; tiny and quantizer-dependent | NOT SUPPORTED (null under the pre-set tests; fragile) |
| Q-C (RQ2) | Is the selective-versus-control gap larger for low-resource languages? | No; equivalent within ±0.02 correlation | NULL (with an equivalence test) |
| Q-D | Do low-resource languages lose more features overall? | Yes at 6 bits under legacy RTN, but it follows activation size, the underlying tier pattern in threshold-free measures is already there at INT16, and GPTQ removes most of it | ESTABLISHED as description; NOT SUPPORTED as language damage |
| Q-E | Do features keep their language tag? | Mostly yes to 5 bits (89 %); losses are mostly borderline features | ESTABLISHED |
| Q-F | Do language features matter for the language (loss)? | Yes, 27 of 27 | ESTABLISHED |
| Q-G | Does that dependence survive quantization? | Kept to 5 bits; inflated at 4 bits, much less with GPTQ | ESTABLISHED (size depends on quantizer) |
| Q-H | Do language features matter for reading comprehension? | Not measurably | NULL |
| Q-I (RQ3) | Do survival or tag status predict whether a feature keeps its function? | Barely (3–4 % gain) or not at all (AUROC about 0.5) | NOT SUPPORTED as a useful predictor; ESTABLISHED as a negative finding |
| Q-J | Where does the quantization damage arise? | Not identifiable with our patches | NOT TESTED / not identifiable |
| Q-K | Does turning the group up repair a 4-bit model? | A tiny effect with legacy RTN only; gone with GPTQ | SUGGESTIVE (legacy) / NOT SUPPORTED (general) |
| Q-L | Does the picture hold on other models? | Llama: not consistent; 9B: reported consistent; tables reproduced from raw files, verdict wording problems | NOT SUPPORTED (Llama); 9B: REPRODUCED computation, SUGGESTIVE meaning |
| Q-M | Does an edit made in the full model keep its effect in the compressed model, and is the change due to the layers before or after it? | Near-kept at 6 bits, rotated at 4 bits, mostly generic; "before versus after" depends on how it is measured | ESTABLISHED (first part); INCONCLUSIVE (second) |
| Q-N | How much do results depend on the quantizer? | Sizes strongly, directions mostly not | ESTABLISHED |
| Q-O | What does the survival rate measure? | Feature size plus a quantizer-wide shift, magnified by the 0.9 cutoff | ESTABLISHED |

---

## 4. How the project went, from start to 29 October 2026

### 4.1 Timeline

| dates (2026) | phase | what was done | key outcome or turning point |
|---|---|---|---|
| 28 Sep – 2 Oct | Phase −1: first pipeline | Fixed the laptop GPU driver, worked around disk space (a bf16 copy of the model), got FLORES from the official archive, built our own pipeline in Hugging Face Transformers (own SAE loader, own RTN), a first 27-language run at 1,500 and then 20,000 tokens per language for a mid-term submission ([`progress_log.md`](progress_log.md) 28 Sep to 2 Oct) | First numbers. **Later retired**: they must not be cited ([`legacy_pipeline/README.md`](legacy_pipeline/README.md)) |
| 15 Oct | Audit | Line-by-line review of our code and of Duan's released code ([`../issues.md`](../issues.md)) | Our pipeline had design flaws (below); our numbers (98.3 % / 94.3 % survival at 7 / 6 bits) disagreed with Duan's (81.3 % / 51.3 %) |
| 16 Oct | Phase 0: replication | Ran Duan's own vendored code on Ada (job 23472); measured why the two pipelines differ ([`audit_our_vs_duan.md`](audit_our_vs_duan.md)) | **Their table reproduced within 0.15 points.** The difference was the start token, not a bug. **Our first pipeline was discarded** (tag `legacy-pipeline-2026-10-16`) |
| 17 Oct | Planning | Primary-source checks (Gemma Scope trains with the start token masked; Deng et al.'s code does the same); decision log started; Phase 1 built as a small, tested change to Duan's script | Mask regime chosen (D-001) |
| 17–19 Oct | Phase 1 | 27 languages, survival on equal token budgets, three tests fixed beforehand, then follow-ups N1–N13 and a clean-room recomputation | **All three tests null.** Follow-ups: activation size explains the apparent gap; a large tier gap explained by activation scale |
| 21 Oct | Re-orientation | User answers Q1–Q4 and Q13–Q15: mask only, close the Duan comparison, publishability first, reassess "survival", include 5 and 4 bits, causal wording tightened (D-027 to D-034) | The project's question widens beyond the proposal |
| 21 Oct | Association retention | Does each feature keep its language tag? Run, analysed, reproduced byte for byte the same day (D-035 to D-038) | 98 % kept at 8 bits, 71 % at 4 bits; losses are mostly borderline features |
| 21–23 Oct | Functional studies F1–F5 | Remove each language's top-64 features; precision ladder; patching; tag versus function; single-feature fingerprints with numerical floors and swapped-direction controls (D-039 to D-064) | Groups matter in 27/27 languages; kept to 5 bits; most 4-bit change is generic |
| 22–23 Oct | Belebele (confirmatory) | Reading-comprehension test with a freeze committed before scoring (D-043, D-044, D-050); then K = all, positive control, prompt check (D-062, D-069, D-072, D-086) | **Null**, with a working positive control |
| 24–26 Oct | Study G | Turn each language's group down and up by graded factors at six precisions; extensions B, C, D (D-070 to D-085) | Clean dose-response; a step at 4 bits; a tiny 4-bit "repair" |
| 26 Oct | Second models | Llama-3.2-1B check (D-087, D-090); Gemma-2-9B handed to another person/team (D-089, D-092; [handoff 040](handoffs/040-2026-10-26-gemma9b-confirmation-external-cluster.md)) | Llama: "not consistent" |
| 26 Oct | Novelty reassessment | Dated search after the first transfer results ([`novelty_assessment_2026-10-26.md`](novelty_assessment_2026-10-26.md)) | **Broad novelty claim withdrawn** |
| 26–27 Oct | Intervention transfer | Crossed design: compressed or full layers before the edit × after the edit; main (FLORES, 27 languages) and external (Wikipedia, 6 languages) panels; recovery analyses R1–R4 (D-088, D-094 to D-099) | Pre-set utility rule: main not met, external met; precision audit R5 built but **never run** (D-100) |
| 27 Oct 09:50 | HOLD | User: stop and finalise; course report and slides finished; cron jobs stopped; documentation consolidated ([handoff 047](handoffs/047-2026-10-27-cron-stopped-doc-cleanup-project-results.md)) | |
| 27 Oct 17:34 – 28 Oct | GPTQ extension | HOLD lifted for one purpose: re-run with matched RTN and GPTQ, everything else identical (D-101 to D-111) | Sizes depend strongly on quantizer; one old null becomes fragile |
| 28 Oct | Survival versus effect | Post hoc: does activation survival predict effect similarity beyond baselines? Gemma (3 quantizers) and Llama (D-110 to D-114) | Only 3–4 % improvement (target 10 %) |
| 29 Oct | Measurement-validity study | Stage 1 on existing files; Stage 2 GPU pass streaming gain, noise, flips and threshold-unit perturbation; Stage 3 frozen analysis (D-117 to D-124) | Survival = size + quantizer shift; the new metrics add nothing for function. **All project compute finished** (D-122) |
| 29 Oct | Gemma-2-9B | Legacy-RTN results imported (D-125); source audit done; raw files fetched and the analysis reproduced byte for byte; a GPTQ-only recovery run prepared and tested on CPU, **never run** ([`functional_validation/gemma9b/REVIEW_STATUS.md`](functional_validation/gemma9b/REVIEW_STATUS.md)) | UNVERIFIED |

### 4.2 Turning points

1. **The start-token discovery (16 Oct).** Duan's survival numbers came from text fed without a start token. With one start token added and ignored in statistics, the same windows give 97.5 % survival at 6 bits instead of 51 %, and the model's perplexity drops from 458 to 16 ([`audit_our_vs_duan.md`](audit_our_vs_duan.md)). This changed what our "survival" baseline meant: at 7 bits, survival sat at the ceiling (99.97 %), so the pre-set 7-bit test had almost no room to find anything.
2. **The Phase 1 null and the magnitude explanation (18–19 Oct).** It killed the original story and moved the project to function.
3. **The functional studies (21–23 Oct).** They gave the project's clearest positive finding (the groups matter) and its clearest negative finding (tags and survival do not predict function).
4. **The Belebele null (23 Oct).** Loss and reading comprehension disagree, and we could not explain why.
5. **The GPTQ extension (27–28 Oct).** It showed that many of our effect sizes were properties of the legacy quantizer.
6. **The measurement-validity reframing (29 Oct).** It turned the many nulls into one coherent statement about what survival numbers measure.

### 4.3 Mistakes caught along the way (details in section 8)

- Our first pipeline mixed several design flaws, including synthetic figures and hard-coded fallback numbers; it was discarded (16 Oct).
- A first Phase 1 launch used all tokens instead of the pre-set equal budgets; cancelled before any output (A-026).
- A rescue job was submitted with settings that could not have been analysed; cancelled after four minutes (A-041).
- A design assumption that INT16 equals full precision was wrong; corrected before any F5 number (A-042, D-055).
- One Study G job ran on a different GPU type, which shifts results by about the size of the effect being measured; it was rerun (D-079).
- Several wordings over-stated results and were corrected (repair shares "where damage sits", D-110; "5 % of items clear the floor" was really 34 %, D-094 update).

### 4.4 How the work was carried out

- **Compute.** Almost all GPU work ran on the university cluster "Ada" through the Slurm job scheduler, mostly on RTX 2080 Ti cards. Large raw outputs went to a private Hugging Face dataset (D-012) and run summaries to Weights & Biases (D-008, D-091, D-115). The 9B legacy-RTN run used one rented RTX PRO 6000 GPU.
- **Who did the work.** The team set the questions and made the key decisions. Much of the coding, running and analysis was done by AI coding assistants under the user's direction: Claude (through scheduled sessions that fired at fixed intervals, hourly at first and later every 20 or 40 minutes, and wrote a numbered handoff after each step) and Codex (another assistant that designed the intervention-transfer study, the "publication rescue" plan and the measurement-validity direction, and imported the 9B results). The course rules allow AI use for code; on 26 October they were changed to allow AI-written reports as well (D-093). The final report should state this use, as the course requires.
- **Rules the work followed.** (1) Replicate a source with its own code; any deviation is a labelled diagnostic (D-002). (2) Write analysis rules into a committed "freeze" file before scoring data whenever possible. (3) Check every new analysis module by "mutation testing": deliberately break the code and confirm that the tests catch it. (4) Never delete progress records; cleanups only move text verbatim into an archive ([`INDEX.md`](INDEX.md)). (5) Do not reduce sample sizes in Study G (D-075). (6) Stop and ask the user on unclear choices; keep a file of open questions with defaults ([`../DECISIONS_FOR_REVIEW.md`](../DECISIONS_FOR_REVIEW.md)).

### 4.5 The checks we used across studies (how careful the work was)

Many answers in this project are "no effect found" or "cannot tell". A reader should know that this is not because the checks were loose. The table lists the safeguards, where they were used, and where they were missing.

| safeguard | what it means in plain words | where it was used | gaps |
|---|---|---|---|
| **Freeze before scoring** | The analysis rules are written in a file and committed to git before the data are scored, so they cannot be bent to fit the result | Belebele (freeze committed 22 Oct 15:43, statements 23 Oct 01:17); loss task (freeze 07:41, report 09:02 on 22 Oct); Study G (design 24 Oct 07:01, report 25 Oct); transfer claim rules (amendment A14); survival measurement Stage 3 (freeze 29 Oct 06:12, results 12:37); 9B predictions (01:42) before the verdict (06:31) | Commit times are set by the author's machine, so they show order, not proof. Most frozen studies still ran on reused test text, so they are exploratory |
| **Identity gates** | Checks that must give an exact known answer, or the run is rejected | INT0 (original weights restored) reproduces full precision exactly; a sham edit has harm exactly 0; "split equals full", "scale one is identity" and "donor self identity" gates in every Study G and 9B job; Stage 3 identity to 1e-14 | None found |
| **Reproduction controls** | Re-running old code on old data must give the old answer before new analysis starts | Phase 1 clean-room recomputation (89 of 89 files identical); old analysis code reproduced old outputs before every GPTQ analysis; Stage 1 reproduced 486 values exactly; committed correlations reproduced before survival-versus-effect | The clean-room check needs cached raw files that are not in git |
| **Mutation testing** | Deliberately inserting bugs into analysis code to make sure the tests catch them | Almost every analysis module (e.g. survival-versus-effect 16 of 16 bugs caught; the Stage 2 sidecar 58 of 59, the one survivor being harmless; Stage 3 generator with 53 tests) | Very thorough for a course project |
| **Multiple-comparison correction** | Adjusting p-values when several tests are run | Holm over the three Phase 1 tests; Holm in Study G extension C | No correction across the many exploratory families or across the three quantizers |
| **Cluster bootstrap** | Intervals that resample whole sentences, passages or languages, because items inside them are related | Sentence bootstrap for group effects; passage bootstrap for Belebele; language-cluster bootstrap in survival-versus-effect and Stage 3; feature × sentence bootstrap in transfer | Group-level transfer intervals cover sentences only; groups and control draws are fixed |
| **Fresh or held-out data** | Testing on text that was not used to choose or tune anything | Study G extension C (562 reserve sentences); the six-language Wikipedia panel; R1 locked FLORES-to-Wikipedia scoring; Belebele items | Transfer prediction is held out by feature only, with about 88 % document overlap; survival-versus-effect is held out by language, not document |
| **Equivalence logic** | Showing an effect is *smaller than* a stated size, not just "not found" | Phase 1 T3 (±0.02); Stage 3 H4 ("absent" if the 90 % interval is inside ±0.20) | Used in only two places; most nulls are "interval includes zero" without a stated margin |
| **Positive controls** | Showing the test *can* detect an effect when one exists | Belebele random-feature removal (1 to 6 points); Study G dose-response; F5 swapped direction | No positive control for the Phase 1 survival contrast |
| **Rounding-only floors** | Measuring how much a number moves when the weights barely change (INT16, INT12) or when only the batching changes | Every functional study; F5 floor; transfer "floors" | INT16 is a real small change, not pure arithmetic noise; the paired precision audit that would separate them was not run |
| **Hardware check** | Making sure GPU type does not shift results | D-079 rerun after a 0.0018-nat shift between GPU types | Not checked systematically for every job |
| **Never delete the record** | Every decision, wrong turn and old text is kept | [`INDEX.md`](INDEX.md), archive folders, 56 numbered handoffs | — |

---

## 5. The studies, one by one

Each block follows the same shape: **question**, **why we ran it**, **what we did** (data, sizes, controls), **result** (with uncertainty and sample size), **label**, **how rigorous the check was**, and **what it cannot tell us**. Numbers in square brackets are 95 % intervals unless stated otherwise.

### 5.1 Phase 0: reproducing the earlier paper, and the start-token issue

**Question.** Is our measuring setup valid compared with Duan (2026), who reported 51 % survival at 6 bits with almost no perplexity change?

**Why.** Our first pipeline gave 98.3 % / 94.3 % survival at 7 / 6 bits, far from Duan's 81.3 % / 51.3 %. Before building anything on survival, we had to know who was measuring what.

**What we did.** We ran Duan's own vendored code, changed only through a small whitelist that a script checks (D-002), on Ada (job 23472). Then we varied one setting at a time to find what causes the difference ([`audit_our_vs_duan.md`](audit_our_vs_duan.md), [`duan_codebase_audit.md`](duan_codebase_audit.md)). We also ran "noise-floor" checks (INT16 and INT12) to see whether their pipeline can tell tiny weight changes from real ones.

**Result.**
- Their code reproduced their table: 7 bits 81.19 % against their 81.26 %; 6 bits 51.15 % against 51.30 % ([`duan_codebase_audit.md`](duan_codebase_audit.md); re-read).
- The difference came from the **start token (BOS)**. Their windows had no start token. With one start token added and left out of the statistics, on the same WikiText windows, perplexity is 16.1 instead of 458.4 and survival is 99.97 % (7 bits) and 97.52 % (6 bits) ([`audit_our_vs_duan.md`](audit_our_vs_duan.md)).
- The start token alone moved survival by +18.6 / +46.2 points (7 / 6 bits). Their shared attention scale moved it by +7.7 / +3.6 points, but only without the start token ([`progress_log.md`](progress_log.md), 16 Oct post-mortem).
- Duan's surprising "7 bits *improves* perplexity by 5.6 %" is an artefact of the missing start token (A-002).
- In the no-start-token setting their pipeline's numbers were "noise-dominated" by the pre-committed closure rule (D-009).

**Label.** Reproduction: **ESTABLISHED.** "Survival numbers in this setup depend strongly on start-token handling": **ESTABLISHED** in two independent software stacks. *Why* the start token matters so much (the model uses the first token as an "attention sink"; P04, P05) is a **hypothesis we never tested** (A-001).

**How rigorous.** A pre-committed closure protocol (D-009), one-factor-at-a-time diagnostic runs, two software stacks (Hugging Face and TransformerLens), and a vendoring checker so that no unintended edit entered their code.

**What it cannot tell us.** Whether Duan's conclusions are wrong. It says their numbers belong to one input setting. The user closed further comparison with Duan on 21 October (D-028), so no later cross-check exists.

### 5.2 Phase 1: do language-selective features survive worse? (the three tests fixed in advance)

**Question.** RQ1 and RQ2: do language-selective features survive rounding worse than matched ordinary features, and is that gap bigger for lower-resource languages?

**Why.** This was the proposal's central hypothesis.

**What we did.**
- 27 languages; for each, the same number of devtest tokens (27,036) so that no language gets more statistical weight (D-015). Features were chosen on the dev split and tested on devtest.
- 9,500 features active in the pooled data. Selective features: S(f) above the shuffled-label 95th percentile (cutoff 1.83). 8,086 of 10,946 supported features passed (74 %, A-027). The tests compared 8,062 selective features with 2,843 controls matched within firing-rate groups ("strata").
- Survival = correlation r ≥ 0.9, read at the feature's peak language.
- Three tests (D-014, D-016): T1 at 7 bits, T2 at 6 bits (selective minus control survival), T3 (does the gap change with resource tier?). 10,000 label permutations, 2,000 bootstrap draws, seed 0, **Holm correction** over the three tests (a standard way to adjust p-values when several tests are run together). An **equivalence test (TOST)** for T3 with a margin of ±0.02 in correlation: it asks whether the effect is *smaller than* a stated size, not just "not found".
- The analysis was run once with the pre-set settings; nothing was changed after seeing numbers (D-017).

**Result** ([`phase1/mask/eq/analysis.md`](phase1/mask/eq/analysis.md), [`phase1/mask/eq/summary.csv`](phase1/mask/eq/summary.csv)).

| test | estimate [95 %] | p | Holm-corrected p |
|---|---|---|---|
| T1, 7 bits: selective minus control survival | −0.28 points [−0.63, +0.13] (99.35 % against 99.82 %) | 0.257 | 0.514 |
| T2, 6 bits | −1.44 points [−2.65, −0.13] (91.45 % against 94.20 %) | 0.043 | 0.128 |
| T3, tier slope of the gap | −0.0048 per tier step | 0.990 | 0.990 |

T3's 90 % interval [−0.0075, −0.0016] lies inside ±0.02, so any tier effect on the gap is smaller than the margin.

Pooled survival over all active features (verified): 8 bits 100 %, 7 bits 99.97 %, 6 bits 94.94 %, 5 bits 59.0 %, 4 bits 13.75 %; INT16 and INT12 100 %. Perplexity change: +0.08 %, +0.40 %, +2.67 %, +11.5 %, +103.9 %.

**Follow-ups (all decided after seeing the data, so exploratory).**
- *Stricter selectivity cutoffs* (A-034, D-019). The 6-bit gap grows to −1.02, −2.14 and −7.16 points at S > 2, 3 and 5; the last two pass Holm.
- *Matching on activation size* (A-035, D-020). Selective features fire more weakly than controls (median peak activation 9.8 against 14.2 at S > 5). When controls are also matched on maximum activation, the S > 5 gap turns from −7.16 points into +0.80 (3 size bins, not significant), +4.13 (5 bins) and +4.58 (10 bins). At the pre-set cutoff it becomes +1.35 / +2.69 / +3.12. So "selective features die more" is mostly "weakly firing features die more". A leftover of +3 to +5 points in the other direction is unexplained.
- *The SAE threshold margin* added nothing beyond raw activation size, because the thresholds are nearly constant (3.00–3.19 for 80 % of features; A-040).
- *Language-level tier gap* (A-038, A-039; verified). At 6 bits, 69.4 % / 90.6 % / 95.1 % of a language's active features survive for low / mid / high tiers. The tier slope is +12.3 points per step (Holm p 0.0002). With the language's activation scale as a covariate it becomes −0.81 (p 0.667). Activation scale and survival correlate at Spearman +0.966 across languages. This is a correlation, partly built into the measure.
- *Behavioural link* (A-032). Languages with more damaged selective features also lose more loss (Spearman +0.46 at 6 bits, +0.54 at 7 bits, n = 27 languages), but selective damage and overall damage move together (variance inflation up to 9.5), so they cannot be separated.
- *Clean-room recomputation* (D-026). Every committed Phase 1 result file was recomputed from cached run files into a fresh folder: 89 of 89 identical ([`phase1_reproduction.md`](phase1_reproduction.md); I could not re-run this, it needs the cached raw files).

**Label.**
- RQ1 (selective features survive worse): **NULL** under the pre-set tests; the largest estimate is −1.44 points at 6 bits. Later shown to be fragile across quantizers (section 5.11). Overall: **NOT SUPPORTED** as a language-specific effect.
- RQ2 (the gap grows for low-resource languages): **NULL, with an equivalence test**.
- Overall tier gap in survival: **ESTABLISHED as a description**; **NOT SUPPORTED** as a language-specific quantization effect (explained by activation scale; and see sections 5.11 and 5.14).
- "The selectivity penalty is an activation-size effect": **ESTABLISHED in direction** (it reverses when size is matched finely), with an unexplained leftover.

**How rigorous.** Analysis estimands fixed before the run (D-014 to D-017); Holm; an equivalence test; noise-floor controls (INT16, INT12, INT0); every analysis module mutation-tested; 89 of 89 files reproduced in a clean room.

**What it cannot tell us.**
- The "selective" class is broad (74 % of supported features), so the test is "most features against a minority", not "one-language features against the rest".
- At 7 bits survival is at the ceiling (99.35 % against 99.82 %), so T1 had almost no room (A-031).
- **"Pre-registered" is not established.** The proposal speaks of an OSF pre-registration, but whether it was ever frozen is recorded as unverified (V-10, Q2, D-029). The tests come from the proposal; the exact estimands were the analyst's choices made before the run. Several project documents still call Phase 1 "pre-registered" or "confirmatory" (section 8.3).

### 5.3 Association retention: do features keep their language tag?

**Question.** Of the features that clearly belong to one language in full precision, how many still have that language as their unique, clear top language after rounding?

**Why.** "Survival" asks whether the activity pattern is the same. A different, more language-focused question is whether the feature still *belongs to its language*. The user chose this study on 21 October (D-035).

**What we did.** 8,086 selective features; 6,663 of them (82.4 %) reproduce their dev peak language on devtest at full precision ("established"). For each bit width we checked whether the feature keeps its original language as the unique, supported peak, with sentence-level bootstrap intervals. Run 31640; analyses reproduced byte for byte by two further jobs (31919, 31920). The plan was checked for factual errors and prior art before any code (D-036, [`association_retention/plan_verification.md`](association_retention/plan_verification.md)).

**Result** ([`association_retention/results/eq/language_summary.csv`](association_retention/results/eq/language_summary.csv), verified).

| | INT16 | INT8 | INT7 | INT6 | INT5 | INT4 |
|---|---|---|---|---|---|---|
| tag kept | 99.2 % | 98.1 % | 97.5 % | 94.7 % | 89.1 % | 70.8 % |
| tags lost (of 6,663) | 56 | 124 | 168 | 356 | 726 | 1,945 |

Follow-ups A1–A3 (D-045, D-066, D-067): within a language, how far the feature's top language was ahead of the runner-up in full precision (its "margin") predicts tag loss with AUROC 0.897 at 6 bits, 0.81 at 5 bits and 0.75 at 4 bits. **AUROC** is how well one number separates two groups: 0.5 = not at all, 1 = perfectly. But the same margin predicts a *functional* change only weakly (AUROC 0.577 at 6 bits, 0.525 at 5 bits), against 0.864 for the tag loss on the same features.

**Label.** "Tag loss is mostly a borderline effect": **ESTABLISHED.** "Tag loss means function loss": **NOT SUPPORTED.**

**How rigorous.** Plan verified before implementation; byte-for-byte re-runs; fixed denominators; sentence bootstrap; the INT16 rounding-only floor as reference.

**What it cannot tell us.** Margin and tag status come from the same devtest text, so part of the margin effect is regression to the mean. The study was **not repeated for GPTQ or matched RTN**: the per-arm data file (2.76 GB) did not fit on the cluster disk (2.4 GB free) ([`gptq_extension/RESULTS.md`](gptq_extension/RESULTS.md) §6).

### 5.4 F1 and F2: do the language features matter, and does that survive quantization?

**Question.** F1: does switching off a language's most selective features hurt that language's next-word prediction more than switching off matched controls? F2: is that dependence kept after rounding?

**Why.** Survival and tags describe activity. They do not show that the features *do* anything. This is the proposal's Phase 2 causal check, made concrete.

**What we did** (design [`../docs/superpowers/specs/2026-10-21-functional-validation-design.md`](../docs/superpowers/specs/2026-10-21-functional-validation-design.md); D-040 to D-043).
- Per language, the 64 most selective features ("K64", chosen on dev) were zero-ablated. Comparison: the mean of 3 control draws, each 64 features matched on activity.
- 300 devtest sentences per language (a pilot of 150 other sentences was used only to set up the design).
- Outcome: next-token loss on the language's own text.
- Nine precision states: full precision, INT8 to INT4, and the controls INT16, INT12, INT0.
- A freeze file was committed before the report was generated ([`functional_validation/evaluation_freeze_lm.json`](functional_validation/evaluation_freeze_lm.json)). Still exploratory, because devtest had been examined before.
- "Macro retention" = the mean of per-language ratios (D-041). This rule was fixed before any quantized number, because the pilot showed effects from 0.005 to 7.3 nats per token, so a few languages (Mandarin above all) would dominate a ratio of means; in the main run Mandarin's effect (7.58) is about 29 times the median language's (0.262).

**Result** ([`functional_validation/results/eval/effects_by_language.csv`](functional_validation/results/eval/effects_by_language.csv), [`retention.csv`](functional_validation/results/eval/retention.csv); verified).
- **F1.** Mean net harm 0.704 [0.684, 0.723] nats per token; median 0.262. The interval is above zero in **27 of 27 languages**. The median control harm is only 0.024. Range: Bengali 0.010 to Mandarin 7.58. On the 18 Belebele languages the mean is 0.713 (18 of 18 positive).
- **F2.**

| | INT16 | INT8 | INT7 | INT6 | INT5 | INT4 |
|---|---|---|---|---|---|---|
| mean of per-language ratios | 1.00 | 1.01 | 1.03 | 1.04 [1.02, 1.05] | 1.13 [1.11, 1.18] | 2.14 [2.04, 2.50] |
| median ratio | 1.00 | 1.01 | 1.02 | 1.03 | 1.09 | 1.54 |

- The F1 and F2 conclusions are the same on the 18 languages the model can read well and on the 9 it cannot (D-068).

**Label.** F1: **ESTABLISHED** (large and universal in this setting). F2, "kept to 5 bits, inflated at 4 bits": **ESTABLISHED for legacy RTN**; the inflation is smaller with better quantizers (median 1.36 matched RTN, 1.22 GPTQ; section 5.11).

**How rigorous.** Activity-matched controls with three draws; a do-nothing "sham" edit gives exactly zero; rounding-only controls give ratios of 1.00; the identity control INT0 gives exactly 1.00; a freeze before the report; per-language intervals.

**What it cannot tell us.**
- Zero-ablation shows that these features are *needed* for part of the prediction, not that they are *enough*, and not that they are the language's only carrier.
- Controls are matched on activity, but not perfectly on size.
- At 4 bits the model is already damaged (its loss is about 0.6 nats per token higher), so a bigger removal effect there is not simply "stronger dependence".

### 5.5 F3: patching full-precision activations back into the 4-bit model

**Question.** If we put the full-precision values of the language group back into the quantized model at layer 12, how much of the quantized model's extra loss goes away?

**Why.** To see whether repairing the language features repairs the language.

**What we did.** At layer 12 we replaced the quantized values of (a) the K64 group, (b) the matched controls, (c) all SAE features, or (d) the whole residual stream with their full-precision values on the same tokens (D-058). Outcome: the share of the quantized-minus-full loss gap that disappears. Macro mean over languages with a clearly positive gap (27 at INT4 and INT5, 12 at INT6).

**Result** ([`functional_validation/results/rescue/rescue.csv`](functional_validation/results/rescue/rescue.csv), verified). At INT4: group 2.62 % (range across languages −0.0 to 8.7 %), controls 1.22 %, all SAE features 17.4 % (−10.3 to 25.9 %), whole residual 39.3 % (28.0 to 49.9 %). At INT5 the group repairs about 0.1 %. Under GPTQ at INT4: group 2.0 %, all features 10.8 %, whole residual 41.4 %.

**Label.** The measurement ("this patch repairs little"): **ESTABLISHED.** The interpretation "the damage does not sit in the language features": **NOT SUPPORTED as a causal claim** (not identifiable with this design).

**How rigorous.** Same tokens, same frozen settings; controls patched the same way; three quantizers.

**What it cannot tell us.** A repair share is not a causal share and not an upper bound on how much of the damage "goes through" these features: the effects of patches are not additive, and some partial patches make the loss *worse*, which shows the parts interact. This was corrected in the project's wording on 28 October (D-110), but the old functional summary still says the groups "account for at most about 3 %" (section 8.3). One layer and one patch construction only; nothing was measured above layer 12.

### 5.6 F4: does losing the language tag predict losing function, at equal activity?

**Question.** Among features of the same activity level, do those that lost their tag lose more function than those that kept it?

**Why.** A direct test of RQ3 using the tag from section 5.3.

**What we did.** At INT6, INT5 and INT4, features that lost their tag were matched to features of equal activity that kept it; the harm of removing each set was compared (D-046, D-048, D-049, D-051). Languages with enough features: 26 at INT4 (Portuguese not analysed), 24 at INT5, 18 at INT6. Paired sentence bootstrap.

**Result** ([`functional_validation/results/f4/`](functional_validation/results/f4/)). Median lost-minus-kept difference in group harm: INT4 +0.0006 [−0.0001, +0.0012], INT5 +0.0007, INT6 +0.0002 nats per token. The INT4 mean is dominated by one language (Lao). A robustness summary added after the first result (post hoc): without Lao, the INT4 mean is +0.0032 [+0.0026, +0.0038], but the same subsets already differ by +0.0014 [+0.0010, +0.0019] at full precision. The change from full precision to INT4 has median −0.0005 [−0.0012, +0.0001].

**Label.** **NULL** for any meaningful difference (effects of a thousandth of a nat, partly present before quantization). The phrase "no systematic difference" used in some documents is slightly too strong, since the mean without Lao is formally above zero.

**How rigorous.** Matching on native activity; paired bootstrap; the same code run unchanged on every bit width; the post hoc robustness summary labelled as such.

**What it cannot tell us.** The feature sets come from the legacy quantizer's tag losses, so F4 was not repeated for GPTQ.

### 5.7 F5: does a single feature still do the same thing after quantization?

**Question.** Taking one feature at a time, is its effect on the model's output the same before and after rounding?

**Why.** The user asked directly whether feature survival means anything if features no longer do the same job (D-053).

**What we did** (design [`../docs/superpowers/specs/2026-10-23-feature-function-equivalence-design.md`](../docs/superpowers/specs/2026-10-23-feature-function-equivalence-design.md)).
- 1,487 features from all 27 languages, a stratified sample (chosen to cover groups of interest; not random) because the full 4,382 would have cost about 13 GPU-hours per state (D-056).
- **Effect fingerprint**: the change of the output logits when the single feature is switched off, over M = 16 contexts. **W1** = cosine between the full-precision and quantized fingerprints.
- **Numerical floor**: the 5th percentile of W1 at INT16 (0.949). A design error (assuming INT16 equals full precision) was caught and fixed before any F5 number (A-042, D-055).
- Controls: a **swapped-direction control** (the decoder direction of a different, matched feature removed at the same positions and strengths), fixed-amplitude variants (removing the read-side change), and M = 8 / 32 sensitivity (D-060 to D-064).
- Pass rules S1–S3 were fixed before the first number, with minimum counts added after a smoke test and before the full run (D-057).

**Result** ([`functional_validation/results/f5/report.md`](functional_validation/results/f5/report.md), re-read).

| state | median W1 [95 %] | share at or above the floor | median activation correlation | identity retrieval |
|---|---|---|---|---|
| INT8 | 0.997 [0.996, 0.997] | 94.0 % | 0.990 | 1.000 |
| INT6 | 0.988 [0.986, 0.989] | 89.7 % | 0.961 | 1.000 |
| INT5 | 0.961 [0.958, 0.964] | 66.5 % | 0.909 | 1.000 |
| INT4 | 0.839 [0.829, 0.848] | 0.3 % | 0.738 | 0.997 |

- **Swapped-direction control.** At INT4 the swapped direction has median cosine 0.828 against 0.842 for the real features; with full-precision amplitudes everywhere 0.838 against 0.846. At INT8 / INT6 / INT5 the control is only 0.004 / 0.006 / 0.007 below the real features. So most of the loss of similarity is shared by arbitrary directions: it is a **generic** reaction of the compressed model, not something specific to each feature.
- **Survival misclassifies function.** About 31 % of features whose effect is intact at INT5 are counted as "lost" by the r ≥ 0.9 survival rule.
- **Tag status.** S1 met (0.902) but uninformative (87 % of "lost" features also pass). S3 met (0.824). S2 (does tag status predict function at INT4?) undefined: only 2 events. The descriptive AUROC of W1 for the tag on the F4 pairs is 0.506 [0.469, 0.543].
- Under GPTQ: median W1 at INT4 is 0.839 / 0.875 / 0.923 (legacy / matched / GPTQ); share at the floor 0.3 / 0.7 / 11.7 %. S2 under GPTQ: AUROC 0.490 [0.431, 0.560], 79 events: **not met**.

**Label.** "Features stay identifiable and their effects are near-preserved to 6 bits": **ESTABLISHED.** "Most drift is generic": **ESTABLISHED in direction**; the small feature-specific margin (0.003 to 0.014) has no intervals, so at most **SUGGESTIVE**. "Tag status predicts function": **NOT SUPPORTED** (AUROC 0.49 to 0.51).

**How rigorous.** Two noise-only states define the floor; swapped-direction and fixed-amplitude controls; context-count sensitivity; rules fixed before data; identity retrieval checks.

**What it cannot tell us.** The sample is stratified, so shares are not population shares. The swap control used one control per feature (404 features) without intervals for the difference. Because the floor is the 5th percentile of INT16, the "within noise" share cannot exceed about 95 % by construction. Fixed amplitudes still use each state's own background, so the control does not separate downstream weights from context (this motivated section 5.12).

### 5.8 Belebele: the only fully confirmatory downstream test

**Question.** Does removing a language's feature group lower reading-comprehension accuracy more than removing matched controls?

**Why.** Next-word loss is a technical measure. A real task tells whether the features matter for something people care about. **Belebele** is a multiple-choice reading-comprehension benchmark built on FLORES passages, covering 122 languages; each question has four answers, so chance is 25 %. (The original Belebele paper is not in the source register; the prompt reference we checked is the lm-evaluation-harness template, S08.)

**What we did.**
- Freeze committed before any test item was scored ([`functional_validation/evaluation_freeze_mc.json`](functional_validation/evaluation_freeze_mc.json); D-043, D-044).
- 18 of 27 languages are "covered": the model's accuracy is clearly above chance. The other 9 cannot be measured meaningfully.
- 385 devtest-origin items per language (all of them; D-047). Zero-shot, answer by letter.
- K64 group against 3 matched control draws. Passage-cluster bootstrap with 2,000 draws.
- Statement S0: at full precision, the average effect E (in nats per item of the correct answer's log-probability) is above zero, one-sided 95 %. S1–S5 (retention at each bit width) only if S0 holds.

**Result** ([`functional_validation/results/mc_eval/`](functional_validation/results/mc_eval/), verified).
- Accuracy at full precision: no edit 0.4257, group removed 0.4255, controls removed 0.4251.
- S0: E = −0.0001 [−0.0035, +0.0031] nats per item; 10 of 18 languages above zero. **Not met.** S1–S5 therefore undefined (D-050).
- My own rough bound on the accuracy difference "group minus controls", bootstrapping the 18 languages: +0.04 points [−0.43, +0.57]. (This is not a project analysis; it supports the informal statement "at most about half a point could hide".)
- Accuracy without any edit: INT8 0.426, INT7 0.420, INT6 0.423, INT5 0.387, **INT4 0.236 (below chance)** under legacy RTN.

**Checks that the null is not an artefact (all exploratory).**
- *Whole group* (up to 1,719 features; D-062): 0.4257 → 0.4209, controls 0.4226. My rough bound: −0.17 points [−0.97, +0.82]. Also null.
- *Positive control* (D-069, D-072; [`functional_validation/results/mc_posctrl/summary.md`](functional_validation/results/mc_posctrl/summary.md)): removing random non-selective features lowers accuracy by −0.76 points [−1.24, −0.26] (256 features), −1.20 (1,024) and −2.19 (all 2,860). Removing random selective features of *other* languages lowers it by up to −6.01 points. A sham edit changes it by exactly 0. So the test can see damage. But the pre-written prediction P1 (at least 5 points from the whole non-selective pool) was **not met**, and P4 was reversed: other languages' selective features (which include English prompt-scaffold features) hurt most.
- *Prompt check* (D-086): our prompt scores higher than the lm-evaluation-harness prompt in 18 of 18 languages (mean difference −0.100 [−0.110, −0.090]).
- *Above-chance 4-bit model*: under GPTQ, INT4 accuracy is 0.318; removing the group changes it by −0.29 points against no edit and −0.44 against controls ([`gptq_extension/results/stage3/analysis_gptq_mc/accuracy.csv`](gptq_extension/results/stage3/analysis_gptq_mc/accuracy.csv)).

**Not mentioned in the plain-language summary.** In the exploratory context table, macro E at INT5 is −0.0066 [−0.0102, −0.0031] and at INT4 +0.0042 [+0.0003, +0.0081] ([`mc_eval/statements.md`](functional_validation/results/mc_eval/statements.md)). Both intervals exclude zero, in opposite directions. The INT4 model is at chance, so these are probably noise; they should still be reported.

**Label.** **NULL** (confirmatory), with demonstrated sensitivity to edits of 1 to 6 points. *Why* loss and comprehension disagree: **NOT TESTED**. Three guesses were written down and none was tested: (a) reading questions can be answered by many routes, so 64 features at one layer are not missed; (b) language identity is easy to recover from the context; (c) a test at 43 % accuracy with 385 questions per language is blunt.

**How rigorous.** Freeze before scoring; held-out items; positive control; sham; prompt check; whole-group sensitivity; three quantizers.

**What it cannot tell us.** No formal equivalence margin for accuracy was set, so "no effect larger than half a point" is informal. 18 languages, one task, one layer, zero-ablation only; absolute accuracies depend on the prompt by about 10 points.

### 5.9 Study G: turning each language's group down and up (graded gain)

**Question.** If we scale a language's group by graded factors, does the harm change smoothly with the factor ("dose-response")? Is it specific to the group? Does the curve change with precision? Can turning the group *up* repair a compressed model?

**Why.** The user made this "the major study" on 24 October (D-070). On/off removal is coarse; a dose-response is stronger evidence that an effect is real, and turning features up tests a possible repair.

**What we did** (design [`../docs/superpowers/specs/2026-10-24-graded-gain-study-design.md`](../docs/superpowers/specs/2026-10-24-graded-gain-study-design.md), committed 24 Oct 07:01 before the data).
- 27 languages × 6 states (full precision, INT8 to INT4) × 300 sentences × 3 control draws. The user set this as a floor that later runs may only add to (D-075).
- Factors: down 0.354, 0.177, 0.088 and 0 (zero-ablation); up 2.83, 5.66, 11.3; fine 1.1 and 1.2. The down factors were calibrated by a pre-set rule on 100 pilot sentences (D-071).
- Hypotheses H-G1 to H-G7 with pass rules written into the design before data.
- Exploratory, because devtest had been used before.

**Result** ([`functional_validation/gain_study/results/curves.csv`](functional_validation/gain_study/results/curves.csv), verified).
- **Dose, down (H-G1): met.** Full-precision median net harm 0.022 / 0.090 / 0.169 / 0.262 at factors 0.354 / 0.177 / 0.088 / 0; strictly ordered in 27 of 27 languages.
- **Specific (H-G2): met.** 26 of 27 have a lower bound above zero at 0.088.
- **Dose, up (H-G3): met.** Increasing in 27; net harm positive at ×11.3 in 24 of 27.
- **Consistency (H-G6): met.** Positive in 27 of 27 at every state; Spearman between full-precision and INT4 language rankings 0.936.
- **Precision ordering (H-G4): narrowly not met.** The median at the deepest down factor falls by slightly more than one bootstrap error from full precision to INT8. With two errors it would pass; that looser rule was chosen after the fact.
- **Repair hypotheses (H-G5a/b): not met**, under-powered (12 of 27 against 14 required). H-G5 (best factor against inverse shrinkage): not met. **Baseline (H-G7): not met.**
- **A step at INT4.** Zero-ablation net harm by state (median): 0.262, 0.268, 0.266, 0.271, 0.281, **0.432**. Flat to INT5, then a jump. At ×0.088: INT4 0.323 against full precision 0.169 (1.9 times). At ×11.3 turning up costs less at INT4 (0.810) than at full precision (1.213).
- **A tiny 4-bit "repair".** At INT4, turning the group up by ×1.1 lowers loss slightly: median net −0.0010 (21 of 27 negative); ×1.2: −0.0013 (18 of 27). At full precision the same turn-up costs a little (+0.0007, +0.0020). The best factor is above 1 at INT4 in 21 of 27 languages, against 7 of 27 at full precision and INT6.

**Extensions.**
- **C, fresh sentences** (D-082; [`gain_study/extension_c/cells.csv`](functional_validation/gain_study/extension_c/cells.csv), verified). 562 reserve devtest sentences never used before, 5 control draws, Holm. INT4 ×1.1: −0.00135 [−0.0020, −0.0003], 20 of 27 negative (sign test p 0.0096): **met**. ×1.2: −0.0017 [−0.0031, +0.0003], 18 of 27: not met. Shape: a shallow minimum near 1.1–1.2, gone by 1.4 (−0.0007), a cost at 1.7 (+0.0046). Controls −0.0002 [−0.0004, 0.0000], so "group-specific" holds by the rule, narrowly. The first INT4 run was on a different GPU type; it was rerun (D-079).
- **D, mechanism** (D-083, D-084; rules committed before the module and the run). Is the best factor explained by how much the group shrank? Spearman −0.170, one-sided p 0.80. The best factor is a reliable property per language (split-half 0.846, Spearman-Brown 0.917), so this null is not noise. **NOT SUPPORTED.**
- **B, groups chosen in the quantized model** (D-081, D-085). Groups re-chosen at INT6 / INT4 overlap the full-precision group with Jaccard 0.855 / 0.488 (27 of 27). They are not more consequential: INT4 zero-ablation 0.452 (state-chosen) against 0.477 (full-precision group); positive in 11 of 27. H-Gb2 not met. So choosing the group in full precision does not limit the INT4 results.
- **Under GPTQ** (D-109, verified): INT4 ×0.088 is 0.323 / 0.281 / 0.187 (legacy / matched / GPTQ), so the 1.9-fold amplification shrinks to 1.1-fold. Best factor above 1 at INT4: 21 / 13 / 7 of 27 (GPTQ equals full precision). ×11.3 at INT4: 0.810 / 0.854 / 1.146.

**Labels.**
- Graded, specific, consistent dose-response: **ESTABLISHED.**
- "A threshold at INT4 rather than a slide": **ESTABLISHED for legacy RTN**, much weaker under GPTQ. It reflects the quantizer's damage level, not a universal threshold.
- INT4 fine-gain "repair": **SUGGESTIVE for legacy RTN only.** It replicated on fresh sentences, but it is about 0.04 % of the loss and of the same order as the GPU-type shift (0.0018 nats, D-079). It disappears under GPTQ, so **NOT SUPPORTED as a general phenomenon**.
- Shrinkage mechanism: **NOT SUPPORTED.**

**How rigorous.** Design and pass rules committed before the data; the user's sample-size floor; identity gates in every job ("split equals full", "scale one is identity", "donor self identity", sham); a fresh-sentence replication with Holm; a reliability check for the extension D null; a hardware rerun.

**What it cannot tell us.** No mechanism for the step or the small repair. The repair is too small to be useful. The calibration of factors was done on full precision only.

### 5.10 Second model: Llama-3.2-1B

**Question.** Do three Gemma findings (the group matters; flat through 6 bits then a change at 4; graded dose-response) also hold on another model?

**Why.** One model is not enough for any general claim. Llama-3.2-1B was the proposal's planned extension and had a public SAE.

**What we did** (plan [`../docs/superpowers/plans/2026-10-26-llama-consistency-check.md`](../docs/superpowers/plans/2026-10-26-llama-consistency-check.md); D-087, D-090). Llama-3.2-1B with the third-party TopK SAE `seonglae/Llama-3.2-1B-sae` (k = 48, 14,336 features) at layer 12 of 16. Same languages, sentences and control design; states full precision, INT8, INT6, INT5, INT4. Predictions P1–P3b and the verdict function committed before any Llama number.

**Result** ([`functional_validation/llama/analysis.json`](functional_validation/llama/analysis.json), verified).
- P1 (group matters, ≥ 24 of 27 positive): median effect 0.048 [0.018, 0.173] against Gemma's 0.262; 21 of 27 positive. **Not met.** (23 of 27 with the median of control draws, post hoc.)
- P2a (flat through 6 bits): 1.010 at INT8, 1.024 at INT6. **Met.**
- P2b (rise at 4 bits): the pre-set mean ratio is −3.57, driven by a language with a tiny full-precision effect. Post hoc median 0.856 [0.743, 1.029] over 21 languages: no amplification. **Not met.**
- P3a (dose ordering, ≥ 20 of 27): 20 of 27 at full precision, 15 of 27 at INT4. **Not met.**
- P3b (INT4 over full-precision harm at ×0.088 larger in ≥ 20 of 27): 14 of 27, median 1.17. **Not met.**
- **Verdict: "not consistent".**

**Label.** Generalisation of the Gemma picture to this pair: **NOT SUPPORTED.** Why it differs: **INCONCLUSIVE**, because model family, SAE type and training recipe, depth and tokenizer all differ. This SAE marks 91 % of its features as language-selective (Gemma about 75 %), and its control pool is small (898 against 2,860).

**How rigorous.** Predictions and verdict code committed before data; same sentence ids.

**What it cannot tell us.** It is not a replication, and "not consistent" does not mean the Gemma results are wrong. These confounds are why Gemma-2-9B (same family, same SAE type) was chosen as the proper second model (D-089).

### 5.11 The GPTQ and matched-RTN extension

**Question.** Which of our results are properties of the old legacy quantizer, and which hold for better quantizers?

**Why.** The legacy RTN had a non-standard shared attention scale. Practitioners use GPTQ or similar methods. The user asked for a fair comparison on 27 October (D-101).

**What we did** (plan [`../docs/superpowers/plans/2026-10-27-gptq-matched-comparison.md`](../docs/superpowers/plans/2026-10-27-gptq-matched-comparison.md); D-101 to D-111).
- Same model, SAE, sentences, groups, controls, batching and analysis code. Only the weights differ. The old code files were not edited; a wrapper swaps the weights from outside.
- GPTQ calibration: 221,184 Wikipedia tokens, disjoint from FLORES and from the external panel's articles.
- Gates: the weight-set hash was equal on more than ten cluster machines; every full-precision number was bit-identical to the old runs (Phase 1 exact; transfer study 259,000 items, 0 differences; Belebele 0.4257); old analysis code reproduced old outputs exactly before every new analysis.
- INT6 and INT4 for most studies; the whole ladder for Phase 1, because its tests are at 7 and 6 bits (D-102).

**Result** (all verified from [`gptq_extension/results/`](gptq_extension/results/)).

| quantity | legacy RTN | matched RTN | GPTQ |
|---|---|---|---|
| survival INT6 (mean of 27 languages) | 85.0 % | 89.7 % | 97.5 % |
| survival INT4 | 12.7 % | 18.6 % | 30.5 % |
| loss increase INT6 / INT4 (nats/token) | 0.0254 / 0.590 | 0.0197 / 0.385 | 0.0058 / 0.130 |
| Phase 1 T1 / T2, Holm p | 0.514 / 0.128 | **0.042 / 0.038** | 1 / 1 |
| group-removal median ratio at INT4 | 1.54 | 1.36 | 1.22 |
| mean effect E at INT4 | 0.935 | 0.900 | 0.779 |
| patch repair at INT4: group / all features / residual | 2.6 / 17.4 / 39.3 % | 2.3 / 13.7 / 35.2 % | 2.0 / 10.8 / 41.4 % |
| Belebele INT4 accuracy, no edit | 0.236 | 0.291 | 0.318 |
| F5 median W1 at INT4 | 0.839 | 0.875 | 0.923 |
| transfer normalized error INT4 / INT6 | 0.664 / 0.165 | 0.572 / 0.153 | 0.476 / 0.132 |

- GPTQ beats legacy RTN in **27 of 27 languages** on survival and loss at INT4 and INT6.
- At INT6, GPTQ's survival gain over legacy is +25.2 points for low-resource languages, +8.0 mid, +4.3 high. **Most of the "low-resource languages lose more features at 6 bits" pattern belongs to the legacy quantizer.** At INT4 the order reverses (+21.7 / +18.9 / +12.9 for high / mid / low). The lowest language at INT6 keeps 53.0 % under legacy and 92.7 % under GPTQ.
- **The Phase 1 null is fragile.** Under matched RTN, T1 (−0.48 points) and T2 (−1.53 points) cross Holm 0.05; under GPTQ nothing is left (+0.16 points at INT6). With a factor of 2 for the two extra quantizer families they would not cross (0.084, 0.077). The effect is at most 1.5 points out of about 95.

**Labels.** "Effect sizes depend strongly on the quantizer; directions mostly do not": **ESTABLISHED.** "The 6-bit low-resource gap is mostly a legacy-quantizer property": **ESTABLISHED as a description.** "Phase 1 null is robust": **NOT SUPPORTED**; better called **INCONCLUSIVE / fragile** for a tiny effect. "GPTQ is better because of its error compensation": **NOT TESTED** (calibration and compensation are mixed; GPTQ's calibration text is Wikipedia, like the external panel).

**How rigorous.** The strongest integration gates of the project: identical full-precision numbers, identical weight hashes across machines, reproduction of old outputs before each new analysis, calibration text disjoint from test text.

**What it cannot tell us.** Mostly INT6 and INT4 only. Association retention and F4 were not repeated (disk). Belebele S1–S5 not rerun. No multiple-comparison rule covers the three arms.

### 5.12 Intervention transfer (context versus weights) and the recovery analyses R1–R4

**Question.** If we make an edit (remove a feature's direction by its full-precision amount) in the full model, does it have the same effect in the compressed model? And is any change caused by the compressed layers *before* the edit (the "context" that reaches it, C) or the compressed layers *after* it (the "weights" that read it, W), or by their interaction (I)?

**Why.** F5 could not separate these two. Novelty checks showed the broad claims were covered, and Codex proposed this crossed design as a more specific contribution; it was verified and adopted (D-088; [`intervention_transfer/verification_2026-10-26.md`](intervention_transfer/verification_2026-10-26.md)).

**What we did** (design [`../docs/superpowers/specs/2026-10-26-intervention-transfer-research-design.md`](../docs/superpowers/specs/2026-10-26-intervention-transfer-research-design.md)).
- A 2 × 2 crossed design: full or quantized layers below the edit × full or quantized layers above it.
- Group level: K64 groups, 27 languages, 300 sentences, 3 controls.
- Feature level: 404 features, 6,464 real-direction items; 2,203 items from 235 features (34.1 %) are "eligible" (their full-precision effect clears the numerical floor). Controls: a matched swapped direction and 3 random directions.
- A held-out prediction test: can a "generic sensitivity" score (how much other directions' effects move at the same place) predict the transfer error for held-out features, at least 10 % better than the strongest simple baseline? Rules fixed before outputs (amendment A14). Crossed feature × sentence bootstrap, 2,000 replicates.
- External panel: Wikipedia text in 6 languages (English, Hindi, Russian, Spanish, Yoruba, Mandarin).

**Group-level result** ([`intervention_transfer/results/main/group_per_language.csv`](intervention_transfer/results/main/group_per_language.csv), verified; medians over 27 languages, nats per token).

| state | full-precision net harm | both quantized, fixed edit | own (native) amount | weights only | \|C\| | \|W\| | \|I\| | C > W |
|---|---|---|---|---|---|---|---|---|
| INT16 | 0.263 | 0.262 | 0.262 | 0.262 | 0.0005 | 0.0003 | 0.0002 | 19 of 27 |
| INT6 | 0.263 | 0.281 | 0.270 | 0.272 | 0.0091 | 0.0092 | 0.0010 | 12 of 27 |
| INT4 | 0.263 | 0.848 | 0.433 | 0.428 | 0.310 | 0.104 | 0.071 | 19 of 27 |

Important: "C bigger in 19 of 27" also happens at INT16, where nothing meaningful is quantized. So that count is no evidence; only the interval of the median difference is (D-094 update). When the edit uses the compressed model's own activation amount, the INT4 harm (0.433) equals the weights-only cell (0.428): the large context term belongs to the *fixed* edit's amplitude mismatch.

**Feature-level result** ([`main/analysis_summary.json`](intervention_transfer/results/main/analysis_summary.json), n = 2,203 eligible items). INT6: cosine 0.984, normalized error 0.165 (about 1.9 numerical floors). INT4: cosine 0.768, normalized error 0.664, size ratio 1.09: the effect is **rotated, not shrunk**. At this level the weights part is larger (0.560 against 0.359 at INT4), the opposite of the loss-level result, because the two measure different things. The real direction's error is only +0.019 [+0.009, +0.032] above the controls at INT4: mostly generic.

**Prediction test** ([`main/predictive_validation.csv`](intervention_transfer/results/main/predictive_validation.csv), verified).

| state | improvement over strongest baseline [95 %] | rule (≥ 10 %, lower bound > 0) |
|---|---|---|
| INT16 | 23.4 % [15.1, 30.8] | met (a rounding-only state) |
| INT6 | 10.9 % [−1.2, 21.4] | not met |
| INT4 | 21.5 % [12.7, 29.4] | met |

Overall utility verdict: **not met** on the main panel; **met** on the external panel (INT4 19.4 %, INT6 23.4 %). But the predictor works just as well at INT16, so it measures how noise-prone a context is, not something specific to quantization. And "held out" is by feature only: about 88 % of the held-out features' documents also appear in the calibration half. On 136 items from 65 unseen documents the gains were 6.7 % (INT4), 0.5 % (INT6) and 29.4 % (INT16), without intervals ([`novelty_assessment_2026-10-26.md`](novelty_assessment_2026-10-26.md)).

**Recovery analyses** (post hoc, each protocol hashed before its code; [`intervention_transfer/results/recovery/`](intervention_transfer/results/recovery/), verified).
- **R1, FLORES-fitted predictor scored on Wikipedia** (1,368 items): INT16 28.2 % [21.3, 35.1]; INT6 18.6 % [4.9, 29.5], but only 15.8 % [−2.7, 28.7] against a simple median predictor; INT4 19.9 % [7.8, 31.3].
- **R2, gradient estimates against real loss changes:** correlation 0.63 [0.46, 0.80] on FLORES. Gradients are not useless for the scalar loss.
- **R3, real against swapped direction at equal effect size:** the real direction's error is ×1.12 (INT6) and ×1.14 (INT4) the swapped one's, but only 144 FLORES pairs (about 2 %) qualified.
- **R4, own amount against fixed edit:** at INT4 the own-amount harm is below the fixed-edit harm in 27 of 27 languages (median −0.351 [−0.358, −0.293]). The leftover above weights-only is +0.014 in the median language, ranging −0.78 to +0.23.

**External group level: a documents disagreement.** The external README and [`PROJECT_RESULTS.md`](PROJECT_RESULTS.md) §4.10 report INT4 |C| 0.133 and |W| 0.073 and say INT6 is "reversed". Appendix A16 of the same file gives medians +0.333 and +0.096. I recomputed both from [`intervention_transfer/results/external/group_per_language.csv`](intervention_transfer/results/external/group_per_language.csv): with six languages, the analysis code takes the *lower* of the two middle values (0.133, 0.073), while the table generator averages them (0.333, 0.096). At INT6 the two conventions even give opposite orders (lower median C 0.0076 < W 0.0124; usual median C 0.0185 > W 0.0134). C > W holds in 4 of 6 languages at INT4 and 2 of 6 at INT6. Both numbers are "correct" under their convention; with six languages, neither supports a firm statement.

**Labels.** "Effect near-kept at INT6, rotated at INT4, largely generic": **ESTABLISHED.** Direction-specific extra part (×1.12–1.14): **SUGGESTIVE** (2 % of pairs, post hoc). "Context versus weights": **INCONCLUSIVE** (the answer depends on loss versus vector, fixed versus own amount, the corpus and, for six languages, the median convention). Generic-sensitivity predictor: **ESTABLISHED as an association; NOT SUPPORTED** as quantization-specific or as a cheap audit (it needs four extra interventions per context).

**How rigorous.** Crossed design; swapped and random controls; claim rules fixed before outputs; feature × sentence bootstrap; a fresh-text panel; post hoc analyses with hashed protocols and mutation-tested code.

**What it cannot tell us.** Group-level intervals cover sentence sampling only (groups and control draws are fixed). Document leakage in the held-out test. A paired arithmetic-precision audit (R5, D-100) was built and tested (476 tests) but **never run**, first because of the user's HOLD, then because of the deadline. So we do not know how much of the small INT16/INT6 differences is arithmetic noise.

### 5.13 Survival versus effect: does activation survival predict whether a feature's effect survives?

**Question.** Does the continuous activation correlation (R1) predict effect similarity (W1) beyond simple facts about the feature (how often and how strongly it fires, how big its effect is)?

**Why.** If survival is a useful stand-in for function, it should predict W1. This was the first step of a post hoc "publication rescue" plan, checked for validity before running (D-110, D-112).

**What we did.** Unit: the F5 feature; 1,412 "reliable" of 1,487. Leave-one-language-out ridge regression; language-cluster bootstrap (resampling whole languages, 2,000 replicates). Pre-set rule: R1 must improve held-out error by ≥ 10 % over baselines with lower bound > 0. Analysis frozen before running; 19 tests, 16 of 16 mutants killed; committed correlation values reproduced first (D-113). Repeated on Llama-3.2-1B (643 features, 610 reliable; D-114).

**Result** ([`survival_effect_rescue/results/survival_effect_results.json`](survival_effect_rescue/results/survival_effect_results.json), verified).
- Spearman R1–W1: 0.30 to 0.52; partial (given baselines) 0.24 to 0.40.
- Improvement of R1 over baselines: legacy INT6 4.4 % [2.6, 6.2], INT4 3.7 % [2.3, 5.2]; matched 3.8 % / 3.1 %; GPTQ 4.1 % / 3.5 %. **0 of 6 cells meet the 10 % rule.**
- The baselines themselves beat a plain median by only 0.2 to 3.2 %: W1 is hard to predict at all.
- Secondary: in the small "noisy effect" group (n = 75), two cells exceed 10 % (legacy INT4 15.6 % [8.5, 24.1], GPTQ INT4 15.0 % [7.1, 23.8]). This is a secondary stratum by the freeze and is not in the plain-language summary.
- Llama: INT6 −0.7 % [−2.0, 0.4] (association not resolvable); INT4 4.2 % [1.6, 6.7], partial correlation 0.252 (Gemma 0.265).

**Label.** A weak positive association: **ESTABLISHED.** Useful predictive value at the 10 % convention: **NULL**. The user noted that the 10 % target is a convention, not a claim (D-121): the honest statement is "a real but small improvement of 3 to 4 %".

**How rigorous.** Frozen analysis; language-held-out; cluster bootstrap; mutation tests; reproduction control; a second model.

**What it cannot tell us.** R1 and W1 come from the same perturbed model. Held out by language, not by document. Stratified sample.

### 5.14 The survival measurement-validity study

**Question.** What does the survival rate actually measure? How hard the quantizer pushes a feature, or something else?

**Why.** All lines of evidence said survival is a weak guide to function. A paper needs to say *why* and *what to report instead* (D-117). Two new metrics were proposed: **metric 1**, splitting the survival correlation into **gain** (how much of the activation size is kept), **noise** and **on/off flips**, summarised as **λ** (the log noise-to-signal ratio); **metric 2**, **ν** ("nu"), the change of each feature's encoder input measured in units of its own threshold.

**What we did** (plan [`../docs/superpowers/plans/2026-10-29-survival-measurement-validity-study.md`](../docs/superpowers/plans/2026-10-29-survival-measurement-validity-study.md)).
- *Stage 1* (CPU, existing files, rules frozen first, post hoc): λ from stored sums. Control: 486 committed per-language values reproduced with largest difference 0.
- *Stage 2* (one GPU pass per quantizer): a "sidecar" streamed 21 statistics for each of 214,506 full-precision-active (language, feature) cells, 27,036 tokens per language; legacy, matched, GPTQ at INT6 and INT4, plus legacy INT16.
- *Stage 3*: analysis frozen before any Stage 2 output was read (D-120). Gates: survival reproduced to 3e-8; full-precision side bitwise identical across jobs; an algebraic identity held to 1e-14. Four documented generator changes after the freeze but before any Stage 3 number (D-123). The user's reporting rule: thresholds are conventions; report every improvement with its interval (D-121).

**Results** ([`survival_measurement/stage1/README.md`](survival_measurement/stage1/README.md), [`stage3/README.md`](survival_measurement/stage3/README.md), [`stage3/stage3_results.md`](survival_measurement/stage3/stage3_results.md)).

| median over cells | legacy INT16 | INT6: legacy / matched / GPTQ | INT4: legacy / matched / GPTQ |
|---|---|---|---|
| gain g | 0.997 | 0.958 / 0.967 / 0.976 | 0.594 / 0.672 / 0.823 |
| active tokens switched off | 1.1 % | 9.2 / 7.7 / 5.9 % | 50.0 / 42.9 / 27.0 % |
| ν (share of the threshold) | 0.018 | 0.130 / 0.113 / 0.086 | 0.706 / 0.562 / 0.378 |

- At INT4 under legacy rounding the median feature loses half its active tokens and keeps about 60 % of its size. The survival correlation is blind to size loss, so it cannot show this.
- *Same features are fragile under every quantizer* (Stage 1 H1): median Spearman of λ across quantizers 0.91 to 0.96.
- *One global shift explains most quantizer differences* (H2): one number per quantizer and width reproduces matched-RTN survival rates within 0.6 / 1.4 points (INT6 / INT4); for GPTQ it removes most but not all of the gap (4.7 points left at INT4). The 0.9 cutoff magnifies small global shifts.
- *Survival tracks feature size* (H3): λ correlates with log feature spread at −0.51 to −0.53 (the pre-set convention |ρ| > 0.5 is only just met) and with log peak size at −0.72 to −0.75. ν is nearly free of the spread (|ρ| ≤ 0.11) but not of the peak (0.25 to 0.41).
- *Tier pattern* (H4): median ν by tier (low / mid / high) at INT16 is 0.0192 / 0.0179 / 0.0170: low-resource features are pushed about 13 % harder **already at INT16**, where weights barely change. For both RTN versions the tier gradient is about the same at INT16, INT6 and INT4, so it is **not quantization-specific**. Under GPTQ the gradient almost vanishes, but it reverses when feature scale is adjusted: not settled.
- *Stage 1 nuance*: the tier slope of λ stays positive (0.09 to 0.18, p ≤ 0.017) even after the activation-scale covariate, and is the same at INT16. This contradicts the Phase 1 reading that "scale carries the whole tier effect", for the threshold-free measure.
- *Function* (H5): adding g and ν on top of R1 changes the prediction of W1 by −0.4 % to +1.8 %, every interval including zero.
- *Selectivity* (descriptive): language-selective cells have 14 to 19 % lower ν; they are pushed *less* hard (no size adjustment).

**Labels.** "Survival rates mix the quantizer's overall strength and feature size, and a fixed cutoff magnifies a global shift": **ESTABLISHED.** "ν removes feature size": **partly** (spread yes, peak no). "The tier pattern is quantization-specific": **NOT SUPPORTED** under RTN; **INCONCLUSIVE** under GPTQ. "Gain and ν improve prediction of function": **NULL.**

**How rigorous.** Frozen analysis with hash; three hard gates that all held; documented pre-result changes; intervals for every improvement; the analysis was run twice and gave byte-identical results.

**What it cannot tell us.** One model, layer and SAE. About 23 effective languages for tier slopes (four script families have one language each). Several H4 labels depend on how the slope is standardised. Post hoc relative to Stage 1.

### 5.15 Gemma-2-9B: imported legacy-RTN results (reproduced from raw files; meaning unsettled)

**Question.** Do the Gemma-2-2B Study G findings hold on the larger sibling, with the same SAE family?

**Why.** Llama mixed too many differences. Gemma-2-9B has the same architecture family and a Gemma Scope SAE (D-089).

**What was done, and by whom.** The run was done by another person on a rented GPU (one RTX PRO 6000), on branch `gemma9b-confirmation`, and imported to main on 29 October (D-125; [`functional_validation/gemma9b/`](functional_validation/gemma9b/)). Model `google/gemma-2-9b`; Gemma Scope `layer_20/width_16k/canonical`. Same 300 sentence ids and corpus as 2B; 6 states; 3 control draws; K64; Study G factors copied from 2B (not recalibrated). Selective features: 5,218 of 16,384 (31.8 %; on 2B 49.4 %). **Legacy RTN only.** The predictions were frozen (commit `d27a6a0`, 01:42) before the verdict (`0c3d3c7`, 06:31), on the other branch.

**What our review has done so far** ([`REVIEW_STATUS.md`](functional_validation/gemma9b/REVIEW_STATUS.md)). Task 1, the source and instrument audit, is **done**: all ten copied files match their recorded hashes; the frozen code hashes equal main's; model, SAE and hook are as stated; the 300 sentence ids and the corpus are identical to 2B; all four run gates were true. **Task 2, the CPU reproduction, is also done** (29 October evening): the six raw per-state files and the selectivity file were fetched at the pinned dataset revision (through the university cluster, outside git); their hashes equal those recorded in the analysis and in the recovery amendment; each file carries its own state label, the evaluation split, 27 languages and the same 300 sentence ids as the freeze; the unchanged analysis code with the original settings (2,000 bootstrap resamples, seed 0) then gave **byte-identical** `curves.csv`, `reliability.csv`, `alpha_star.csv` and `report.md`, and equal verdicts. The earlier gap (the metadata lists only INT4) is explained: all six states share one metadata file, written last by the INT4 job. What is still open is meaning, not arithmetic: the three problems listed below.

**Reported result** ([`verdict.md`](functional_validation/gemma9b/verdict.md)): P1, P2a, P2b, P3a and P3b all "met"; verdict "consistent". H-G5b repair 8 of 27 (14 needed).

**What I recomputed from the imported derived table** ([`results/analysis/curves.csv`](functional_validation/gemma9b/results/analysis/curves.csv); a derived table, not raw data).
- Full-precision zero-ablation net harm positive with lower bound above zero in 27 of 27 languages; median 0.270, mean 0.794.
- Mean of per-language ratios: INT8 1.003, INT7 1.047, INT6 1.006, INT5 1.205, INT4 2.479 (matches the verdict).
- **The median INT4 ratio is 1.586**, which the verdict does not report. The mean of 2.479 is driven by Kannada (9.9×, from a tiny full-precision effect of 0.010) and Korean (7.6×). The median is the fairer comparison with 2B (1.54), and it makes 9B look *closer* to 2B than the reported mean does.
- **P3a as frozen is written backwards.** The freeze says E(0.354) > E(0.177) > E(0.088). Read literally, 0 of 27 languages meet it. Harm actually increases as the factor falls in 27 of 27. The verdict silently uses the intended direction.
- **P3b changed meaning.** In the Llama plan, P3b is the INT4-over-full-precision harm ratio at ×0.088. The 9B freeze says "median selectivity ratio above 1.3". In the Llama sense it would also be met (26 of 27 larger, median 1.69).
- The P1 confidence interval required by the freeze is not reported.
- INT4 ×1.1: median net −0.0005, 17 of 27 negative: the same sign as 2B legacy, smaller.
- The verdict's sentence that this is "consistent with the 2B observation that quantization damage concentrates on language-selective circuits" contradicts the 2B patching result (2.6 % repair) and is not supported by amplified ablation harm alone. A consolidation note on the file already says so.

**The prepared GPTQ recovery run (prepared, not run).** The 9B GPTQ and matched-RTN arms, and "tier 2" (Phase 1 survival and loss on 9B), were **never run**. On 29 October a GPTQ-only recovery run was prepared for the roughly 3 GPU-hours left on the rented machine ([`RECOVERY_PLAN.md`](../../gemma9b-v2/reports/functional_validation/gemma9b/RECOVERY_PLAN.md) in the worktree of branch `gemma9b-confirmation`, folder `/home/karma/Documents/ANLP/project/gemma9b-v2/`). What exists:
- code for a GPTQ INT4 Study G unit (INT6 only if the budget allows), reusing the legacy run's feature groups (hash-checked);
- a budget planner (estimates 1.3 / 2.0 / 3.75 hours at low / mid / high; only the 1,768-second unit pass is measured, the GPTQ build time is a guess of 7 to 35 minutes) and a budget guard that stops before the unit if the measured projection exceeds the budget;
- a partial-evidence report with predictions R0–R3 written before any 9B GPTQ number (amendment A2); by design it **never gives the full three-arm verdict**;
- tests on CPU (planner 16, amendment tool 10, report 16 with 12 of 12 mutants killed, driver dry runs and a stub rehearsal 16, bundle 5).

It has **never been run on a GPU**. Its own plan lists what is unverified: any real execution, the GPTQ build time and memory on 9B, the software stack on that GPU, the gate on the real model.

**Label.** The 9B results: **computation REPRODUCED; meaning SUGGESTIVE.** The tables follow from the raw files and are consistent with the verdict on P1, P2a, P2b and the intended direction of P3a. With language-bootstrap intervals (my computation over the 27 languages, 5,000 resamples): the INT4-over-full-precision ratio has mean 2.479 [1.781, 3.368] and median 1.586 [1.238, 2.264], above 1 in 26 of 27 languages and above 1.3 in 18 of 27; the full-precision zero-ablation effect has mean 0.794 [0.359, 1.394] and median 0.270 [0.197, 0.608], positive in 27 of 27; INT6 over full precision is 1.006 [0.957, 1.050] (mean) and 1.019 [0.978, 1.045] (median). The freeze asked for the interval of the mean effect; the verdict does not report it. This is one run of one model on legacy rounding; it does not say the 2B findings 'replicate' in every respect, and it does not show that quantization damage sits in language features. No 9B GPTQ or matched-RTN result exists.

### 5.16 Things planned or proposed but never run

| item | status | where recorded |
|---|---|---|
| Pruning (magnitude and Wanda) | proposed in the proposal; never started | [`../anlp_revised.txt`](../anlp_revised.txt); Q8 |
| Layers 3 and 20 of Gemma-2-2B | proposed; never run | proposal §5 |
| Paired arithmetic-precision audit (R5) | built, 476 tests, never deployed | D-100, Q29 |
| Real integer kernels; activation quantization | not studied | [`PROJECT_RESULTS.md`](PROJECT_RESULTS.md) §4.12 |
| Other SAEs on the same model | not studied | |
| A GPU diagnostic of why low-resource languages have lower activation scale (Q15: confidence, diffuse activity, SAE fidelity) | approved as a question; job never designed or run | D-034; [`INDEX.md`](INDEX.md) §3 |
| Random-weight-noise controls; all-token intervals (Q16 options D, E) | not selected | D-039 |
| The full 4,382-feature F5 sample | not run (cost) | Q24 |
| GPTQ without or with other calibration text | not run | [`gptq_extension/RESULTS.md`](gptq_extension/RESULTS.md) §6 |
| Association retention and F4 under GPTQ | not run (disk) | same |
| 9B GPTQ / matched-RTN arms; 9B survival study | not run; GPTQ-only recovery prepared | D-125; RECOVERY_PLAN |
| Steering / cross-language patching repair pilot | judged feasible, not adopted | [`steering_feasibility_review_2026-10-28.md`](steering_feasibility_review_2026-10-28.md) |

---

## 6. Results at a glance, and what they add up to

### 6.1 One table of all questions

| question | study | main number [interval], sample | label |
|---|---|---|---|
| Is our setup valid against Duan? | Phase 0 | their table reproduced within 0.15 points; start token moves 6-bit survival 51 % → 97.5 % | ESTABLISHED |
| Do selective features survive worse? (RQ1) | Phase 1 | 6 bits: −1.44 points [−2.65, −0.13], Holm p 0.128; 27 languages, 8,062 vs 2,843 features | NULL under pre-set tests; NOT SUPPORTED as language effect; fragile across quantizers |
| Is the gap larger for low-resource languages? (RQ2) | Phase 1 | slope −0.0048, Holm p 0.99; 90 % interval inside ±0.02 | NULL (equivalence) |
| Do low-resource languages lose more features overall? | Phase 1, GPTQ, Stage 1/3 | 6 bits: 69.4 / 90.6 / 95.1 % (low / mid / high); slope +12.3 → −0.81 with scale covariate; GPTQ gains +25 points for low tier; tier pattern in λ and ν already present at INT16 | ESTABLISHED as description; NOT SUPPORTED as quantization-specific language damage |
| Do features keep their language tag? | Association retention | 98.1 / 94.7 / 89.1 / 70.8 % at 8 / 6 / 5 / 4 bits (6,663 features); margin AUROC 0.90 at 6 bits | ESTABLISHED |
| Does tag loss mean function loss? | A3, F4, F5 | margin AUROC for function 0.58; F4 median +0.0006 nats; F5 AUROC 0.506 [0.469, 0.543] | NOT SUPPORTED |
| Do language features matter for loss? | F1, Study G | median net harm 0.262, mean 0.704 [0.684, 0.723]; 27 of 27; 300 sentences per language | ESTABLISHED |
| Is that dependence kept after rounding? | F2, Study G | median ratio 1.03 at 6 bits, 1.09 at 5 bits, 1.54 at 4 bits (legacy); 1.22 at 4 bits (GPTQ) | ESTABLISHED (size depends on quantizer) |
| Do they matter for reading comprehension? | Belebele | accuracy 0.4257 / 0.4255 / 0.4251 (none / group / controls); group − controls +0.04 points [−0.43, +0.57]; 18 languages × 385 items | NULL (confirmatory, positive control works) |
| What does patching the group repair? | F3 | 2.6 % of the INT4 loss gap (residual 39 %) | measurement ESTABLISHED; location NOT TESTED / not identifiable |
| Does a feature keep doing the same thing? | F5 | median W1 0.988 at 6 bits, 0.839 at 4 bits; swap control 0.828 at 4 bits; 1,487 features | ESTABLISHED (near-kept to 6 bits; drift mostly generic) |
| Is the graded dose-response real? | Study G | 27 of 27 ordered; harm 0.022 → 0.262 as factor falls 0.354 → 0 | ESTABLISHED |
| Does turning the group up repair a 4-bit model? | Study G, ext. C, GPTQ | −0.00135 [−0.0020, −0.0003] nats, 20 of 27 (legacy); gone under GPTQ | SUGGESTIVE (legacy only); NOT SUPPORTED in general |
| Is that repair explained by shrinkage? | Ext. D | Spearman −0.17, p 0.80 (reliability 0.92) | NOT SUPPORTED |
| Same picture on Llama-3.2-1B? | Llama check | median effect 0.048 vs 0.262; INT4 median ratio 0.86 [0.74, 1.03] | NOT SUPPORTED (confounded) |
| Same picture on Gemma-2-9B? | imported, reproduced from raw files | 27 of 27 positive; INT4 ratio mean 2.48 [1.78, 3.37], median 1.59 [1.24, 2.26] | REPRODUCED (computation); SUGGESTIVE (meaning) |
| Does an edit transfer, and is change from context or weights? | Transfer | INT6 cosine 0.984; INT4 0.768; C vs W depends on estimand | ESTABLISHED (transfer); INCONCLUSIVE (C vs W) |
| Can a generic score predict transfer error? | Transfer | INT4 21.5 % [12.7, 29.4], INT6 10.9 % [−1.2, 21.4]; also predicts INT16 | NOT SUPPORTED as quantization-specific |
| How much do results depend on the quantizer? | GPTQ extension | INT4 survival 12.7 / 18.6 / 30.5 %; loss +0.590 / +0.385 / +0.130 | ESTABLISHED |
| Does survival predict effect beyond baselines? | Survival vs effect | +3.1 to +4.4 % (intervals above 0), 0 of 6 cells reach 10 %; Llama the same | weak association ESTABLISHED; useful prediction NULL |
| What does the survival rate measure? | Stage 1, Stage 3 | λ vs spread −0.51 to −0.53, vs peak −0.72 to −0.75; one global shift explains matched-RTN rates within 0.6 / 1.4 points | ESTABLISHED |
| Do gain and ν add predictive value? | Stage 3 | −0.4 % to +1.8 %, every interval includes 0 | NULL |

### 6.2 What the whole body of evidence supports

The central question was: *does compressing the model damage the language-specific features of the SAE, and does that hurt the language?*

1. **Survival of language features.** Under weight-only rounding to 6 bits, language-selective features do not survive worse than matched features in any robust or large way. In threshold units they are even pushed somewhat *less* (14 to 19 %, without a size adjustment). The apparent disadvantage of low-resource languages follows feature size and a perturbation sensitivity that is already measurable at INT16, and a better quantizer removes most of it.
2. **Function of language features.** The groups are needed for own-language next-word prediction in this model: removing them is a defined intervention with a large, specific, graded effect. That effect is kept through 6 bits (and mostly 5 bits). At 4 bits it is inflated, and the size of the inflation depends on the quantizer.
3. **Does compression hurt the language through these features? Not shown.** Patching the group repairs only 2–3 % of the 4-bit loss gap, while the whole residual repairs about 40 %. Inflated removal harm at 4 bits is a change in the effect of an edit inside an already damaged model; it does not locate the quantization damage. Most 4-bit drift is shared with arbitrary directions. On the reading task the features do not matter measurably at all.
4. **Measurement.** The most coherent result of the project is negative and methodological: survival numbers and language tags are poor stand-ins for whether a feature still does the same job. Six analyses agree (F4, F5, A3, survival-versus-effect on two models, Stage 3, GPTQ F5), across three quantizers.

### 6.3 Causal claims the data do NOT support

1. "Quantization damage concentrates in language-selective features or circuits." (Stated in the 9B verdict and, in older wording, in the functional summary.) Patch-repair shares are neither causal shares nor upper bounds.
2. "Low-resource languages are hurt *because* their language features break." Tier is observed, not manipulated; the gap follows activation scale and exists without meaningful quantization.
3. "The 4-bit turn-up repair compensates for activation shrinkage." Extension D says no.
4. "GPTQ is better because of its error compensation." Calibration and compensation are mixed together.
5. "The generic-sensitivity predictor measures quantization damage." It predicts INT16 equally well.
6. "Language features matter for comprehension." Belebele is null.
7. Anything about real integer kernels or activation quantization. Not tested.

### 6.4 Why so many answers are inconclusive, and how hard we tried anyway

The user asked us to be clear about studies that did not give a conclusive answer. This table says, for each one, how much was done and what stopped a clear answer.

| study | how much was done | why no clear answer |
|---|---|---|
| Phase 1 (RQ1, RQ2) | 27 languages, equal budgets, 10,000 permutations, Holm, an equivalence test, 13 follow-ups (thresholds, magnitude matching, threshold margin, mixed models, language level, scale covariate, behavioural link), a clean-room recomputation, then a re-run under two more quantizers | The true effect is small (≤ 1.5 points) and sits near the ceiling at 7 bits; the "selective" class is 74 % of features; activation size explains the gap; the verdict word flips with the quantizer |
| Behavioural link (selective damage → loss) | Per-language ranks and regressions with fertility and overall damage | Only 27 data points, and selective and overall damage move together (variance inflation up to 9.5) |
| Belebele | Freeze before scoring, 18 × 385 items, three control draws, whole-group sensitivity, a three-dose positive control, a prompt check, an above-chance 4-bit model under GPTQ | The null itself is clear; what is unclear is *why* it differs from the loss result. Baseline accuracy is only 43 %, the 4-bit legacy model is at chance, and no experiment separated the three guesses |
| F4 | Activity-matched pairs at three bit widths, paired bootstrap, robust summaries | Effects of a thousandth of a nat, partly present already at full precision |
| F5 rule S2 | 1,487 features, two noise states, swap controls | Floor effect: almost no feature stays inside the noise floor at 4 bits under RTN (2 events), so the rule is undefined; under GPTQ it fails (0.49) |
| Study G repair hypotheses | 27 × 6 × 300 × 3 design, a fresh-sentence replication with five control draws, a reliability-checked mechanism test | The effect is about 0.001 nats, the same size as GPU-type differences; it vanishes under GPTQ |
| Llama check | Predictions frozen; same sentences and control design | Model family, SAE type, recipe, depth and tokenizer all differ, so a difference cannot be attributed |
| Context versus weights | A 2 × 2 crossed design, two corpora, native and fixed edits, feature and group levels, four recovery analyses | The answer depends on what is measured (loss versus vector length), on fixed versus own-amount edits, and for six languages even on the median convention |
| Transfer prediction ("utility") | Held-out ridge models, crossed bootstrap, a second corpus, a locked cross-corpus test | At 6 bits the error is near the floor (1.9 floors), leaving little to predict; the predictor works equally at INT16, so it is not about quantization; document overlap |
| Survival versus effect | 1,412 features, three quantizers, two models, language-held-out, cluster bootstrap | The outcome (W1) is hard to predict at all (baselines beat a median by only 0.2 to 3.2 %) |
| Tier pattern in ν (Stage 3 H4) | 214,506 cells, three quantizers, pre-set equivalence bound | About 23 effective languages; the GPTQ verdict reverses with the scale adjustment |
| Precision audit (R5) | Built, 476 tests | Never run (HOLD, then deadline) |
| 9B | Source audit done; GPTQ recovery prepared and tested | Not reproduced; GPTQ not run; budget about 3 GPU-hours |

The common reasons are: **small true effects** near a ceiling or a numerical floor; **measures that mix several things** (survival mixes size, noise and the cutoff); **one model and one layer**, so nothing can be separated from model specifics; **reused test text**, which keeps most results exploratory; **few independent units** (27 languages, about 23 effective) for anything about tiers; and **time**, which stopped the precision audit and the 9B arms.

### 6.5 A worked example: three languages through the studies

To make the numbers concrete, here is one language from each tier. All values are from the files named in section 5 (recomputed by me). "Legacy" and "GPTQ" are the quantizers.

| measure | English (high) | Hindi (mid) | Yoruba (low) |
|---|---|---|---|
| share of active features surviving at 6 bits, legacy / GPTQ | 97.2 % / 99.6 % | 90.3 % / 98.6 % | 67.6 % / 92.7 % |
| same at 4 bits, legacy / GPTQ | 31.8 % / 52.9 % | 11.5 % / 30.1 % | 6.8 % / 19.5 % |
| loss increase at 4 bits (nats/token), legacy / GPTQ | 0.161 / 0.066 | 0.478 / 0.120 | 0.573 / 0.088 |
| language tag kept at 6 / 4 bits (legacy) | 96.1 % / 87.8 % | 95.9 % / 69.9 % | 94.8 % / 65.1 % |
| net harm of removing the K64 group, full precision [95 %] | 0.173 [0.151, 0.197] | 0.104 [0.088, 0.122] | 1.578 [1.520, 1.637] |
| ratio of that harm at 4 bits, legacy / matched / GPTQ | 0.97 / 1.03 / 0.89 | 2.90 / 2.08 / 1.57 | 1.38 / 1.15 / 1.13 |
| 4-bit loss gap repaired by patching the group / the whole residual (legacy) | 0.9 % / 41.9 % | 1.2 % / 40.6 % | 6.7 % / 29.1 % |
| Belebele accuracy: no edit / group removed / controls removed | 0.548 / 0.558 / 0.555 | 0.371 / 0.369 / 0.371 | not covered (model near chance) |

How to read it. Yoruba loses far more features at 6 bits under legacy rounding, but GPTQ closes most of that gap (67.6 % → 92.7 %), which is why we do not read the gap as language damage. Yoruba's group matters a lot for its loss (1.58 nats), yet its tag is kept about as well as English's at 6 bits. Among these three, the 4-bit inflation of the group's effect is largest for Hindi, not Yoruba, and it shrinks with better quantizers for both. In no language does patching the group repair much of the 4-bit loss gap. For English and Hindi, removing the group does not lower reading accuracy.

---

## 7. The major decisions, and why we took them

| decision | what it was, in plain words | why | how it turned out |
|---|---|---|---|
| D-002, D-009 | Replicate Duan with *their* vendored code, edits only through a checked whitelist; a closure rule fixed in advance | Two pipelines disagreed; re-implementing had caused confusion | Right. It found the start-token cause within a day |
| Discard our first pipeline (16 Oct) | Remove our own code (kept under a git tag); build on Duan's | It had real flaws (start token counted in statistics, token drift, no noise floor, synthetic figures, hard-coded fallbacks) | Right. Its mid-term numbers must not be cited |
| D-001, D-027, D-028 | "Mask" regime: one start token added and ignored in statistics; later, mask only and stop comparing numbers with Duan | The SAE was trained this way; without the token the model is badly degraded (perplexity 48,864 on FLORES sentences) and the noise-floor check fails | Right for validity. Cost: 7-bit survival at the ceiling, so T1 had little power |
| D-010, D-015 | Same sentences for every language; same number of tokens per language (27,036) by seeded sampling | So that no language gets more statistical weight; it was in the proposal | Right. A first launch forgot it and was cancelled (A-026) |
| D-003, D-011 | Selectivity score S(f) as the proposal defined it; tiers from the proposal, with Joshi class as a sensitivity | Follow the proposal | S(f) made 74 % of features "selective", which weakened the Phase 1 contrast (A-027). In hindsight a stricter definition was needed |
| D-014, D-016, D-017 | Holm correction over three tests; equivalence test for the tier slope; run once with pre-set settings | Several tests at once inflate false positives; "not significant" is not "no effect" | Right. Equivalence was used in only two places in the whole project |
| D-020, D-032, D-033 | Add magnitude-matched controls as a labelled sensitivity; report both comparisons as associations, not causes | Duan's own data show survival follows activation size | Right. It explained the gap |
| D-029 | Publishability is the main goal; questions may change with recorded reasons | User decision after the null | Led to the functional studies. Cost: everything later is exploratory |
| D-030, D-031 | Reassess survival; include 5 and 4 bits | 7 bits at the ceiling; 5 and 4 bits are where things change | Right |
| D-041 | Macro retention = mean of per-language ratios | Mandarin's effect (7.58) is about 29 times the median language's and would dominate a ratio of means | Sensible, but a mean of ratios is itself fragile when a language has a tiny effect (Llama −3.57; 9B Kannada 9.9×). The median should have been reported next to it from the start |
| D-043, D-044 | Belebele as the only confirmatory family; loss task exploratory; freeze before scoring | Loss-task sentences had been seen; Belebele items had not | Right. It gave the one clean confirmatory result (a null) |
| D-055 | F5 numerical floor from INT16 and re-batching, not from "INT16 = full precision" | INT16 is a real (tiny) weight change (A-042) | A correction made before any number. Right |
| D-061, D-063 | Swapped-direction and fixed-amplitude controls in F5 | To tell feature-specific change from generic drift | The most informative control of the project |
| D-070, D-075 | Study G as "the major study"; a sample-size floor (27 × 6 × 300 × 3) | User's instruction; avoid cutting samples for cost | Clean dose-response. The floor meant every follow-up only added data |
| D-077 | Queue all cluster jobs at once | GPUs had idled for hours between scheduled checks | Saved time; no scientific effect |
| D-079 | Rerun a job that landed on a different GPU type | A 0.0018-nat shift, the size of the effect being measured | Right. A useful warning for every effect near 1e-3 nats |
| D-087, D-089, D-092 | Llama check, then Gemma-2-9B as the proper second model, handed to another team | Llama confounds model, SAE type, recipe and depth | Llama gave an uninterpretable "not consistent". 9B was only partly run and not by us |
| D-088 | Adopt the intervention-transfer study (Codex's design) after verification | The broad novelty was gone; a more specific design was needed | Produced solid descriptive results; the headline prediction rule failed on the main panel |
| D-100 | Build the precision audit in bounded steps | To learn whether tiny INT16/INT6 effects are arithmetic noise | Built and tested, never run (HOLD, then deadline). An important gap |
| HOLD (27 Oct 09:50) and D-101 | Stop and finalise; then lift the hold only for the GPTQ extension | Deadline; the user wanted a fair quantizer comparison | The GPTQ extension was the most informative late addition |
| D-102 | Run the whole Phase 1 ladder for both new quantizers | The pre-set tests need 7 and 6 bits | Right. It exposed the fragility of the Phase 1 null |
| D-110 | Authorise the "publication rescue" plan; correct repair-share wording | A validity check found over-statements | Right |
| D-117, D-120, D-121 | Measurement-validity study; freeze Stage 3 before data; report every improvement with its interval, thresholds are conventions | Survival tracked size and quantizer, not function | Gave the project's best-supported framing |
| D-122 | Finish all compute on 29 October | Course deadline (31 October) | All jobs completed; precision audit left unrun |
| D-125 | Import the 9B legacy-RTN results for review | User request | Imported; reproduced from raw files on 29 October; wording problems recorded as anomaly A-043 |
| D-126 | Approve the 9B changes, prepare the 9B GPTQ-only run (code and tests, not run), have the existing results analysed and the report written by Opus 5.5 high-effort agents, clean up the documentation by archiving and dated notes (nothing of the progress record deleted) | User (29 October) | Done as described; the 9B run itself waits for the user |

**The 10 % improvement target was only a convention.** Several studies used "beats the strongest baseline by at least 10 %, with a lower bound above zero" as a pass rule (transfer, survival versus effect, Stage 3). On 29 October the user stated plainly that this number "isn't backed by any actual claims" (D-121). So the honest reading of "not met" is the improvement and its interval (for example 3–4 % for survival), not a yes/no.

**Stop-and-ask rules.** Open choices went into [`../DECISIONS_FOR_REVIEW.md`](../DECISIONS_FOR_REVIEW.md) with a default, so work did not block. Several questions are still open there (for example Q6, Q7, Q11, Q12 on the Phase 1 sensitivity analyses, and Q29 on the precision audit).

**Decisions that turned out wrong or were revised.** The assumption that INT16 equals full precision (A-042); the all-token first launch (A-026); the repair-share wording "the damage sits in…" (D-058, revised by D-110); counting "C > W in 19 of 27" as evidence (revised in the D-094 update); a deck line that only 5 % of items clear the floor (34 % in fact); the 9B P3a inequality written in the wrong direction; reliance on a mean of ratios without a median.

---

## 8. Mistakes, corrections and honesty notes

### 8.1 Bugs and process errors found and fixed

| what happened | effect | fix | record |
|---|---|---|---|
| First pipeline: start token counted as a normal token in every statistic; decode-then-re-encode changed tokens in 0–32 % of chunks; no noise floor; synthetic figures and hard-coded fallback numbers; panel not content-matched; controls matched on firing rate only | Its numbers were misleading as a "replication" | Pipeline discarded 16 Oct; outputs moved to [`legacy_pipeline/`](legacy_pipeline/) as "hypotheses, not findings" | [`../issues.md`](../issues.md); progress log 16 Oct |
| Three defects in the first Phase 1 jobs' main routine (no CPU test possible) | Jobs failed | Fixed before results | A-025 |
| First Phase 1 launch used all tokens, not equal budgets | Would have broken the pre-set design | Cancelled after 25 minutes, no outputs | A-026, D-015 |
| Test temp files filled local disk | Harness incident | Cleaned; environments moved | A-028 |
| Equivalence test crashed when few features were selective | None on results | Fixed | A-033 |
| Mixed-model results differ with the number of computer threads | Tiny | Thread count fixed | A-036 |
| Rescue job submitted with an unanalysable setting | 4 minutes lost | Cancelled, resubmitted | A-041 |
| F5 design assumed INT16 = full precision | Would have mis-set the floor | Corrected before any number | A-042, D-055 |
| Study G job ran on an RTX 4070 instead of a 2080 Ti | Shift of 0.0018 nats, as large as the effect | Rerun on the right hardware | D-079 |
| Stage 3 comparison group "contaminated" in first coding | Would have biased a descriptive result | Fixed before the run | D-123 |
| Stage 3 JSON too large (saved masks) | None | Saved counts; rerun byte-identical | Stage 3 README |
| `issues.md` B1 said TransformerLens computes RMSNorm in bf16 | Wrong statement (it is fp32) | Corrected in the audit | [`audit_our_vs_duan.md`](audit_our_vs_duan.md) |

### 8.2 Numbers corrected

- Matched-RTN INT4 survival: 18.7 % → **18.6 %** (mean 18.649; verified).
- GPTQ INT4 Belebele accuracy: 0.319 → **0.318** (0.3185; verified).
- A slide said only about 5 % of transfer items clear the numerical floor; the true share is **34.1 %** (2,203 of 6,464).
- "Preserved at INT6" for the transfer study softened to "changes little" (cosine 0.984, about 1.9 floors).

### 8.3 Statements in project documents that say more (or less) than the evidence

These come from the independent audit; I checked each against the files.

**Over-statements.**
1. **The 9B verdict** says the 9B results "replicate the Gemma-2-2B findings" and are "consistent with the 2B observation that quantization damage concentrates on language-selective circuits". Not reproduced; P3a written backwards; P3b redefined; the second clause contradicts the 2B patching result.
2. **The functional claim map** ([`functional_validation/RESULTS_SUMMARY.md`](functional_validation/RESULTS_SUMMARY.md), F3 row and claim 4) still says the language groups "account for at most about 3 %" of the quantization damage and "at most 30 to 40 % sits in the layer-12 residual". D-110 corrected this: repair shares are not upper bounds or locations. The old text remains under routing banners.
3. **"Pre-registered" / "confirmatory" Phase 1.** [`PROJECT_RESULTS.md`](PROJECT_RESULTS.md) §1 says "two sets of tests were confirmatory"; [`phase1_results.md`](phase1_results.md) says "pre-registered confirmatory tests". The registration status is recorded as unverified (V-10, Q2, D-029), while [`functional_validation/RESULTS_SUMMARY.md`](functional_validation/RESULTS_SUMMARY.md) calls Belebele "the only confirmatory family". The fair wording: Phase 1 tests were specified in the proposal and their estimands fixed before the run; registration is unverified.
4. **"C bigger in 19 of 27 languages"** (PROJECT_RESULTS §4.10) is presented as support for "context dominates", but the same count occurs at INT16.
5. **"The tier gap shrinks to nothing once you account for how strongly features fire"** (PROJECT_RESULTS §1) is true for the thresholded survival rate, but Stage 1 shows a leftover tier slope in the threshold-free λ after the same covariate, present at INT16.
6. **Belebele positive control "1.2 to 6.0 points from random sets"** (PROJECT_RESULTS §1): the 6.0 comes from control sets of *other languages' selective features*; the full random pool gives 2.2; and the pre-written P1 (≥ 5 points) was not met.
7. **"With strength matched the gap reverses: +3 to +5 points"** holds at S > 5 with 5 or 10 bins; with 3 bins it is +0.80 and not significant.
8. **"Reduces the held-out error by 21.5 %"** for the transfer predictor omits that held-out is by feature only, with about 88 % document overlap.
9. **"At most ~0.5 point could hide" on Belebele** rests on no formal equivalence analysis (my rough bound agrees for K64, not for K = all, where it is closer to one point).

**Under-statements (things given less weight than they deserve).**
1. The measurement-validity finding (six consistent analyses, three quantizers, two models for one analysis) is the project's strongest result but is spread across five documents, while the plain summary leads with "why the findings are not conclusive".
2. F4's "no systematic difference": the INT4 mean without Lao is formally above zero (+0.0032 [+0.0026, +0.0038]), though tiny and partly present at full precision.
3. The survival-versus-effect "noisy" stratum, where two cells exceed 10 % (n = 75), is not mentioned in the plain summary.
4. The Belebele INT5 and INT4 context intervals that exclude zero are not mentioned.
5. The 9B median INT4 ratio (1.586) is not reported; it is closer to 2B's 1.54 than the reported mean.

### 8.4 Places where documents disagree or are out of date

| documents | disagreement | which one the result files support |
|---|---|---|
| [`PROJECT_RESULTS.md`](PROJECT_RESULTS.md) §4.10 and the external README vs Appendix A16 of the same file | External INT4 |C| 0.133 / |W| 0.073 vs 0.333 / 0.096 | Both are right under different median conventions for six languages (lower median in the analysis code; average of the two middle values in the table generator). I recomputed both. With n = 6 neither supports a firm claim |
| [`gptq_extension/RESULTS.md`](gptq_extension/RESULTS.md) header vs its body | "Study G, F5 and the external panel are still running" vs reported as done | The body; all three are done (D-108, D-109, D-111) |
| [`PROJECT_RESULTS.md`](PROJECT_RESULTS.md) §1, §4.9, §4.12 vs its own banner | "No Gemma-2-9B results here" vs imported 9B results | The banner (D-125); the old text was not updated |
| [`functional_validation/RESULTS_SUMMARY.md`](functional_validation/RESULTS_SUMMARY.md) | "Llama … has not been run" vs the finished Llama check (claim 17) | Llama is done (D-090) |
| Phase 1 status across documents | "confirmatory / pre-registered" vs "Belebele is the only confirmatory family" | Registration unverified; treat Phase 1 as "specified in advance, registration unverified" |
| Three INT6 survival figures | 94.9 % (pooled features), 90.4 % (median language), 85.0 % (mean language) | All correct; different aggregates. Always say which |
| [`PROJECT_RESULTS.md`](PROJECT_RESULTS.md) §4.6 | Attributes AUROC 0.506 to F4 | It is F5's descriptive statistic computed on the F4 pairs |
| 9B freeze vs 9B verdict | P3a inequality direction; P3b meaning | The derived table: harm grows as the factor falls (27 of 27) |
| [`CURRENT_STATUS.md`](CURRENT_STATUS.md) header date vs body | "Updated 27 Oct" vs updates to 29 Oct | Body |
| Measurement-validity plan, text vs its own prior-art table | The sentence "(Borobia, Neurocomputing 2026; arXiv 2607.12166)" can be read as one paper; the table lists arXiv 2607.12166 as a separate work ("correlational recovery overstates causal validity") | The table; I cite the two separately. Check the identity of arXiv 2607.12166 before citing it in the paper |

---

## 9. Limitations (all in one place)

1. **One main model, one layer, one SAE.** Gemma-2-2B, layer 12, Gemma Scope 16k. The only second-model evidence is Llama (confounded, not consistent) and 9B (same family, unverified, legacy RTN only).
2. **Simulated, weight-only quantization.** Rounded weights run in ordinary arithmetic. No integer kernels, no activation or memory-cache quantization. INT16 is a real small perturbation, not a pure arithmetic-noise reference. The paired precision audit that would have measured arithmetic noise was not run.
3. **Quantizer dependence.** Most sizes, and several verdict words (Phase 1, H-G4, the external utility rule, the F2 class), change with the quantizer. The legacy RTN had a non-standard shared scale.
4. **27 languages, three tiers, about 23 effective clusters** for tier slopes. Tier is observed and tangled with script, tokenization, corpus size and how well the model knows the language.
5. **Reuse of FLORES.** Features chosen on dev, tested on devtest, which earlier studies had already examined. Fresh text exists only for Study G extension C (reserve devtest), the six-language Wikipedia panel and R1.
6. **Registration status.** Phase 1's registration is unverified. Belebele is the only family with a documented freeze before scoring that every document calls confirmatory.
7. **Multiplicity.** No correction across the many exploratory families or across the three quantizer arms.
8. **Held-out validation is weak.** Transfer prediction is held out by feature with about 88 % document overlap. Survival-versus-effect is held out by language, not by document. No F5 on the external panel.
9. **Control quality.** Activity-matched controls are not perfectly size-matched. Transfer controls are not filtered for eligibility. The F5 swap control uses one control per feature, without intervals.
10. **Ceiling and floor effects.** 7-bit survival at the ceiling; 4-bit Belebele at chance under legacy RTN; the F5 floor share capped near 95 %; 6-bit transfer error near the floor.
11. **One downstream task.** Zero-shot, letter format, 18 languages, 385 items each, 43 % baseline accuracy.
12. **Hardware sensitivity** at the 0.001-nat level, comparable to the 4-bit fine-gain effects.
13. **Sampling design.** F5 is stratified, not random. Groups are fixed at K = 64. Selectivity marks 74 % of supported features as selective.
14. **Time.** Most studies were designed, run and analysed within days, by scheduled sessions, under a fixed course deadline. Several planned controls were never run (section 5.16).

---

## 10. Future work, and the path to a publishable result

### 10.1 What is publishable now (candid view)

**A measurement and negative-results paper**, for a workshop or a findings-level venue: *"SAE feature survival under weight quantization measures feature size and a quantizer-wide shift, not whether a feature still does the same job."* Supporting evidence: the λ decomposition and ν (Stage 1 and 3); survival versus effect (3–4 % gain, two models); the F5 swapped-direction control; tag AUROC about 0.5; three quantizers; the Belebele null with a positive control.

Its weaknesses: one model (plus a confounded Llama check), reuse of FLORES devtest, and simulated quantization. Its ingredients overlap with Duan, Borobia, arXiv 2607.12166, FEGA and Chang et al., so it must be positioned as a **controlled evaluation**, not a discovery.

**Not publishable as is:** the original selective-destruction hypothesis (null); the 4-bit "repair" (tiny, legacy-only); the context-versus-weights split as a mechanism (depends on the estimand); the generic-sensitivity predictor as an audit (not quantization-specific; document leakage); any 9B claim before reproduction.

### 10.2 What would turn it into a strong, confirmatory result

| need | concrete step | rough size |
|---|---|---|
| Fresh, document-held-out data | Freeze the main hypotheses (λ follows size; R1 adds less than X % beyond baselines, with an equivalence margin) and test on text never examined (new Wikipedia sample, FLORES+ or NTREX), held out by document | 27+ languages, ≥ 300 new documents per language |
| Second model, same SAE family | The imported 9B legacy-RTN results are now reproduced from raw files (review task 2); next run the GPTQ arm (prepared), then matched RTN, then tier 2 | 3 arms × INT6 / INT4 |
| Second SAE and layer on the same model | Other Gemma Scope widths or sparsity levels; layers 6 and 18 (or the proposal's 3 and 20) | 2–3 SAEs × 2–3 layers |
| Real kernels | GPTQ / AWQ INT4 with real integer kernels against fake quantization on the same weights; run the precision audit R5 | small, paired |
| Activation quantization | W8A8 or W4A8 | one extra arm |
| The tier question | 60–100 languages; a design that states its causal question in advance (D-032), with scale measured per feature | for power on a tier slope |
| Reading task | State an equivalence margin (e.g. ±1 point) in advance; more items and languages; a second task (e.g. XNLI or generation scored with chrF) | ≥ 4× items to halve the interval |
| Why GPTQ is better | GPTQ with off-language, small or random calibration text | 3–4 calibration variants |
| Multiplicity | Pre-specify the quantizer family and correct across arms | — |
| Why loss and comprehension disagree | Test the three written guesses (redundancy across layers, context recovery of language identity, task bluntness), e.g. multi-layer removal or a harder task | — |

### 10.3 What to do with Gemma-2-9B and the remaining budget

1. **First, finish the review of what exists**, on CPU. The reproduction is done (raw files fetched, unchanged analysis rerun, every table identical). What remains is the claim audit: settle P1's interval (computed: mean 0.794 [0.359, 1.394]), the retention aggregate (median 1.586 [1.238, 2.264] next to the mean 2.479), P3a's direction and P3b's definition. Record corrections as addenda; do not edit the imported files.
2. **Then decide on the GPTQ recovery run.** With about 3 rented GPU-hours, the prepared GPTQ-only INT4 unit fits the mid estimate (2.0 hours) but not the high one (3.75 hours). The plan's safeguards are sensible: stop after the gate job, read the measured GPTQ build time, continue only if the budget guard allows. Its result can only be a partial reading (GPTQ against legacy at INT4); it cannot give the three-arm verdict and says nothing about matched RTN or Phase 1 on 9B.
3. **If the budget is not enough**, report 9B as "legacy-RTN results produced by another team member, source-audited, raw reproduction pending (or done), GPTQ and matched-RTN arms not run".

### 10.4 Lessons for the next project

1. **Validate the instrument before testing the hypothesis.** The start-token issue changed survival by up to 46 points. A noise-floor check and a reproduction of the source should come first. Here they came only in mid-October, after a first pipeline had been built on an unchecked instrument.
2. **Pick a measure with room to move.** At 7 bits survival was at 99.97 %, so the pre-set 7-bit test could not find anything.
3. **Define "language-specific" strictly enough.** A rule that marks 74 % of features as selective tests "most against few".
4. **Always match controls on size, not only on frequency.** Most apparent selectivity effects were size effects.
5. **Report a median next to every mean of ratios.** Single languages with tiny effects dominated the means (Llama −3.57; 9B Kannada 9.9×).
6. **Keep fresh text in reserve from the start.** Only extension C and the Wikipedia panel had it; everything else became exploratory.
7. **Set equivalence margins in advance.** Without them a null can only say "not found", not "smaller than X".
8. **Run more than one quantizer early.** Several of our "findings" were properties of one non-standard quantizer.

### 10.5 Smaller open items

The open questions in [`../DECISIONS_FOR_REVIEW.md`](../DECISIONS_FOR_REVIEW.md) (Q6, Q7, Q11, Q12 on how to report the Phase 1 sensitivities; Q29 on the precision audit); the Q15 question of *why* low-resource languages have smaller activations (confidence, diffuse activity or SAE fidelity), which was approved but never designed; making the Hugging Face dataset public (the user allowed it; it is still private).

---

## 11. Conclusion

We set out to show that quantization selectively destroys language-specific SAE features, especially for low-resource languages. It does not, in any way we could find. The apparent effects follow how strongly features fire, exist even when the weights barely change, and largely disappear with a better quantizer.

Along the way we established three things that hold in our setting. First, each language's most selective features really matter for predicting that language's text, and this dependence survives rounding to 5 or 6 bits. Second, those same features do not matter measurably for a reading-comprehension test, which we could not explain. Third, and most useful for others, the standard "survival" number and the language tag are poor guides to whether a feature still does the same job: survival mostly measures feature size and the quantizer's overall strength.

Much of the work produced nulls or inconclusive answers. That was not for lack of care: tests were fixed in advance where possible, every analysis was mutation-tested, results were reproduced byte for byte, controls were added for every alternative explanation we could think of, and three quantizers were compared on identical inputs. The nulls come from real limits: small true effects near the ceiling, a broad definition of "selective", a blunt downstream test, one model and one layer, reused test text, and simulated quantization. A publishable paper is within reach as a careful measurement study, if it adds fresh document-held-out data, a reproduced second model and a real-kernel check.

---

## Appendix A. Glossary (simple words)

| term | meaning |
|---|---|
| ablation, zero-ablation | Switching features off; zero-ablation sets them to zero |
| activation | How strongly a feature fires on a token |
| AUROC | How well a number separates two groups: 0.5 = not at all, 1 = perfectly |
| Belebele | A multiple-choice reading-comprehension test built on FLORES passages; four answers, chance 25 % |
| BOS (start token) | The special "beginning of text" token |
| bootstrap | Resampling the data many times to get an interval for an estimate |
| cluster bootstrap | Resampling whole groups (sentences, passages or languages) because items inside a group are related |
| calibration text | Example text GPTQ uses to decide how to round |
| confirmatory / exploratory | Fixed in advance on fresh data / everything else |
| context (C), weights (W), interaction (I) | In the transfer study: change due to compressed layers before the edit, after the edit, and both together |
| control features (matched) | Ordinary features chosen to fire about as often and as strongly as the language group |
| cosine | Similarity of two directions: 1 = same, 0 = unrelated |
| effect fingerprint, W1 | The change in output probabilities when one feature is switched off; W1 compares it before and after rounding |
| equivalence test (TOST) | A test that shows an effect is smaller than a stated margin, not just "not found" |
| feature | One of the SAE's 16,384 directions; it fires on some tokens |
| fixed vs native (own) edit | Remove a feature by its full-precision amount vs by the amount it actually has in the compressed model |
| flips | Tokens where a feature switches on or off because of rounding |
| FLORES-200 | A set of the same sentences professionally translated into 200 languages ("dev" 997 sentences, "devtest" 1,012) |
| freeze | A committed file fixing the analysis rules before the data are scored |
| gain g | Share of a feature's activation size kept after rounding (1 = all) |
| gain scaling (Study G) | Multiplying a group's activations by a factor below 1 (down) or above 1 (up) |
| GPTQ | A quantizer that rounds weights one by one and adjusts the rest to cancel the error |
| Holm correction | Adjusting p-values when several tests are run together |
| INT8 … INT4 | Bits per weight; INT16 and INT12 are near-no-change controls; INT0 restores the original weights |
| JumpReLU | The SAE rule: a feature is on only if its input passes a learned threshold |
| K64 | The 64 most language-selective features of a language |
| λ (lambda) | Log noise-to-signal ratio of a feature after rounding; it combines size loss, noise and flips |
| loss (nats per token) | How surprised the model is by the next token; lower is better |
| margin | How far a feature's top language was ahead of the runner-up |
| mask regime | Start token added to the input but excluded from all statistics |
| mutation test | Deliberately breaking analysis code to check that the tests catch it |
| net harm, effect E | Loss increase from removing the group minus that from removing matched controls |
| ν (nu) | How far rounding pushes a feature's encoder input, in units of its own threshold |
| numerical floor | How much a measure moves from tiny rounding alone; effects near it cannot be trusted |
| patching, rescue | Putting full-precision values of chosen parts back into the compressed model |
| perplexity | e to the power of the loss |
| quantization | Storing weights with fewer bits |
| residual stream | The model's running internal state between layers |
| resource tier | High, mid or low: how much text exists for a language |
| retention (function) | Effect in the compressed model divided by effect in full precision |
| RTN (legacy, matched) | Round to nearest; legacy = the project's original version with a shared attention scale; matched = the standard version |
| SAE (sparse autoencoder) | A tool that rewrites the model's internal state as a sum of a few features from a large dictionary |
| selectivity S(f) | A feature's highest per-language firing rate divided by its average rate |
| survival | A feature's activations still correlate at r ≥ 0.9 with full precision |
| swapped-direction control | Removing a *different* feature's direction at the same places and strengths |
| token | A word or part of a word |

## Appendix B. Map from each study to its decisions and key files

| study | decision / anomaly entries | key result files | plan or design written before |
|---|---|---|---|
| Phase 0, start token | D-001, D-002, D-009, D-027, D-028; A-001–A-024, A-029, A-030 | [`audit_our_vs_duan.md`](audit_our_vs_duan.md), [`duan_codebase_audit.md`](duan_codebase_audit.md), [`g0/`](g0/) | proposal §4.2 |
| Phase 1 | D-003, D-010–D-026, D-032–D-034; A-025–A-040; V-10 | [`phase1/mask/eq/`](phase1/mask/eq/), [`phase1_results.md`](phase1_results.md), [`phase1_summary.md`](phase1_summary.md), [`phase1_reproduction.md`](phase1_reproduction.md) | [`../anlp_revised.txt`](../anlp_revised.txt) |
| Association retention, A1–A3 | D-035–D-038, D-045, D-066, D-067 | [`association_retention/results/`](association_retention/results/) | [`../docs/superpowers/plans/2026-10-21-language-association-retention.md`](../docs/superpowers/plans/2026-10-21-language-association-retention.md) |
| F1 / F2 | D-039–D-043, D-068 | [`functional_validation/results/eval/`](functional_validation/results/eval/) | [`../docs/superpowers/specs/2026-10-21-functional-validation-design.md`](../docs/superpowers/specs/2026-10-21-functional-validation-design.md) |
| F3 | D-058; A-041 | [`functional_validation/results/rescue/`](functional_validation/results/rescue/) | same |
| F4 | D-046, D-048, D-049, D-051 | [`functional_validation/results/f4/`](functional_validation/results/f4/) | same |
| F5 | D-053–D-057, D-059–D-064; A-042 | [`functional_validation/results/f5/`](functional_validation/results/f5/), `f5_sensitivity/` | [`../docs/superpowers/specs/2026-10-23-feature-function-equivalence-design.md`](../docs/superpowers/specs/2026-10-23-feature-function-equivalence-design.md) |
| Belebele | D-043, D-044, D-047, D-050, D-062, D-069, D-072, D-086 | `functional_validation/results/{mc_eval, mc_ksens, mc_posctrl}/`, [`evaluation_freeze_mc.json`](functional_validation/evaluation_freeze_mc.json) | functional-validation design |
| Study G + B / C / D | D-070–D-085 | [`functional_validation/gain_study/`](functional_validation/gain_study/) | [`../docs/superpowers/specs/2026-10-24-graded-gain-study-design.md`](../docs/superpowers/specs/2026-10-24-graded-gain-study-design.md) |
| Llama-3.2-1B | D-087, D-089, D-090 | [`functional_validation/llama/`](functional_validation/llama/), `llama_robust/` | [`../docs/superpowers/plans/2026-10-26-llama-consistency-check.md`](../docs/superpowers/plans/2026-10-26-llama-consistency-check.md) |
| Intervention transfer, R1–R4, (R5 unrun) | D-088, D-094–D-100 | [`intervention_transfer/results/`](intervention_transfer/results/) | [`../docs/superpowers/specs/2026-10-26-intervention-transfer-research-design.md`](../docs/superpowers/specs/2026-10-26-intervention-transfer-research-design.md) |
| GPTQ / matched-RTN extension | D-101–D-111, D-115, D-116 | [`gptq_extension/results/`](gptq_extension/results/), [`gptq_extension/RESULTS.md`](gptq_extension/RESULTS.md) | [`../docs/superpowers/plans/2026-10-27-gptq-matched-comparison.md`](../docs/superpowers/plans/2026-10-27-gptq-matched-comparison.md) |
| Survival versus effect | D-110, D-112–D-114 | [`survival_effect_rescue/`](survival_effect_rescue/) | [`../docs/superpowers/plans/2026-10-28-survival-and-intervention-publication-rescue.md`](../docs/superpowers/plans/2026-10-28-survival-and-intervention-publication-rescue.md) |
| Survival measurement validity | D-117–D-124 | [`survival_measurement/`](survival_measurement/) | [`../docs/superpowers/plans/2026-10-29-survival-measurement-validity-study.md`](../docs/superpowers/plans/2026-10-29-survival-measurement-validity-study.md) |
| Gemma-2-9B (imported) and recovery run | D-089, D-092, D-125 | [`functional_validation/gemma9b/`](functional_validation/gemma9b/); recovery plan in the `gemma9b-confirmation` worktree | [`../docs/superpowers/plans/2026-10-26-gemma9b-confirmation.md`](../docs/superpowers/plans/2026-10-26-gemma9b-confirmation.md) |
| Novelty and publication assessments | D-040, D-065, D-088 | [`novelty_assessment_2026-10-26.md`](novelty_assessment_2026-10-26.md), [`publication_reassessment_2026-10-26/`](publication_reassessment_2026-10-26/), [`steering_feasibility_review_2026-10-28.md`](steering_feasibility_review_2026-10-28.md) | [`../docs/superpowers/plans/2026-10-26-publication-recovery-after-results.md`](../docs/superpowers/plans/2026-10-26-publication-recovery-after-results.md) |

## Appendix C. Which numbers I checked myself

**Recomputed by me on 29 October 2026 from committed result files** (CPU, scratch scripts outside the repository):

- Phase 1 pooled survival (100 / 99.97 / 94.94 / 59.0 / 13.75 %), perplexity changes, 9,500 active features, 27,036 tokens per language, median language INT6 90.4 %, mean language 85.0 % ([`phase1/mask/eq/summary.csv`](phase1/mask/eq/summary.csv), `per_language.csv`).
- Association retention 99.2 / 98.1 / 97.5 / 94.7 / 89.1 / 70.8 % and lost counts 56 / 124 / 168 / 356 / 726 / 1,945 of 6,663.
- F1 mean 0.7037, median 0.2624, 27 of 27 with lower bound above zero, control median 0.0235, range 0.0098 (Bengali) to 7.575 (Mandarin); F2 mean-of-ratios 1.006 / 1.034 / 1.038 / 1.129 / 2.136, INT4 median 1.538.
- F3 rescue at INT4: group 2.62 %, controls 1.22 %, all 17.4 %, residual 39.3 %.
- Belebele accuracies (0.4257 / 0.4255 / 0.4251; INT4 0.2358, INT5 0.3874), K = all 0.4209 / 0.4226; my language-bootstrap bounds +0.04 [−0.43, +0.57] and −0.17 [−0.97, +0.82] points.
- Study G full-precision and INT4 medians at every factor (0.0223 / 0.0902 / 0.1687 / 0.2624; INT4 0.4316, ×0.088 0.3233, ×11.3 1.2128 vs 0.8095); zero-ablation medians by state; INT4 ×1.1 / ×1.2 counts (21, 18 of 27); best factor above 1 (FP 7, INT6 7, INT4 21).
- Extension C INT4 medians −0.00135 (20 of 27) and −0.00169 (18 of 27), ×1.4 −0.00072, ×1.7 +0.00458; extension D ρ −0.170, p 0.804, split-half 0.846, Spearman-Brown 0.917.
- GPTQ arms: best factor above 1 at INT4 21 / 13 / 7; INT4 ×0.088 0.323 / 0.281 / 0.187; ×11.3 0.810 / 0.854 / 1.146.
- Transfer group medians (all rows of the table in 5.12), C > W counts (19 / 12 / 19), predictive validation (23.4 / 10.9 / 21.5 % with intervals); external medians under both conventions (0.333 / 0.096 vs 0.133 / 0.073; INT6 0.0185 / 0.0134 vs 0.0076 / 0.0124).
- GPTQ stage 1: survival 85.0 / 89.7 / 97.5 % (INT6), 12.68 / 18.65 / 30.51 % (INT4); loss 0.0254 / 0.0197 / 0.0058 and 0.5904 / 0.3848 / 0.1296; tier gains +25.2 / +8.0 / +4.3 (INT6), +21.7 / +18.9 / +12.9 (INT4, high / mid / low); lowest language 53.0 % vs 92.66 %; 27 of 27 languages better.
- GPTQ stage 3: median INT4 ratio 1.355 / 1.223; mean E at INT4 0.900 / 0.779; rescue group 2.3 / 2.0 %, all 13.7 / 10.8 %, residual 35.2 / 41.4 %; Belebele INT4 0.2912 / 0.3185.
- Survival versus effect: all six primary cells (3.1 to 4.4 %), the noisy-stratum cells (15.6 %, 15.0 %), the baselines over the median null (0.2 to 3.2 %), Llama INT6 −0.7 % and INT4 4.2 % (partial 0.252).
- Llama P1–P3b values from `analysis.json`.
- Gemma-2-9B from the imported derived `curves.csv`: 27 of 27 positive, median 0.270, mean 0.794; mean ratios 1.003 / 1.047 / 1.006 / 1.205 / 2.479; median INT4 1.586; Kannada 9.93× (FP 0.010), Korean 7.62×; literal P3a 0 of 27, harm increasing 27 of 27; Llama-sense P3b 26 of 27, median 1.69; INT4 ×1.1 median −0.0005, 17 of 27.

**Re-read from generated report files, not recomputed:** Phase 1 test p-values and intervals (`analysis.md`); the TOST interval; Duan reproduction numbers; F5 table and S1–S3; F4 robust summaries; Belebele S0 and the INT5/INT4 context intervals; the positive-control table; extension C intervals; transfer feature-level numbers and R1–R4 (from the audit's verification, spot-checked); Stage 1 and Stage 3 statistics (I spot-checked gain and ν rows in `stage3_results.md`); the hardware difference of D-079; the magnitude-matching gaps (+0.80 / +4.13 / +4.58 points, from `phase1/mask/eq/magnitude/magnitude.md`); the Phase 1 group sizes (8,062 + 2,843; 8,086 of 10,946 supported); the 88 % document overlap and the 136-item / 65-document check (from `novelty_assessment_2026-10-26.md`).

**Could not check** (they need raw files that are outside the repository): the clean-room recomputation 89 of 89; Stage 3's reproduction to 3e-8 and identity to 1e-14 (the sums files are outside the repository; the numbers are in the committed ledger); anything from the discarded first pipeline (deliberately not used).

## Appendix D. Sources cited (as registered in the project)

All identifiers are those in [`research_sources.md`](research_sources.md) (P, M, S numbers), [`functional_validation/prior_art.md`](functional_validation/prior_art.md), [`quantization_gptq_audit_2026-10-27.md`](quantization_gptq_audit_2026-10-27.md) and the measurement-validity plan's prior-art section. Venue status is as recorded there; preprints are marked.

- P01 Duan, arXiv 2606.03002 (preprint; v2 "Perplexity Can Miss SAE Feature Damage Under Quantization").
- P02 Deng et al., ACL 2025. P03 Lieberum et al., Gemma Scope, BlackboxNLP 2024. P04 Owen et al., arXiv 2503.22329. P05 Barbero et al., arXiv 2504.02732. P06 Leask et al., ICLR 2025. P08 Makelov et al., ICLR 2025. P09 Karvonen et al., SAEBench, ICML 2025. P10 Lin et al., AWQ, MLSys 2024. P11 Chang, Zhang, Thomason and Jia, EMNLP 2025. P12 Imai, Keele and Yamamoto, Statistical Science 2010.
- P17 Sin Jing Xuan, Huseynov and Zhang, arXiv 2507.18918 (preprint). P18 Srivastava, "The Tokenizer Tax", arXiv 2607.24276 (preprint; source of the character-fertility covariate used in the Phase 1 behavioural link). P19 / P31 Borobia, Seguí-Mas and Tormo-Carbó, Neurocomputing 2026 (DOI 10.1016/j.neucom.2026.134826) and arXiv 2603.25325.
- P20 Gupte, Chhabra and Khalili, arXiv 2507.15977. P23 Marchisio et al., Findings of EMNLP 2024. P24 UniComp, arXiv 2602.09130. P25 Marie and Fujita, arXiv 2508.20893. P27 Chimoto, Elhoushi and Bassett, EACL 2026. P28 Bhandari and Wade, arXiv 2609.06473. P30 Sathyanarayanan, arXiv 2609.23125. P34 Andrylie et al., arXiv 2507.11230. P35 Chou et al., arXiv 2507.13410. P36 Marks et al., Sparse Feature Circuits, ICLR 2025. P41 Cho et al., EMNLP 2026. P43 Hoang et al., FEGA, arXiv 2607.24645. P44 Nguyen et al., arXiv 2609.04808. P46 Rahmanisa et al., arXiv 2507.22581. P47 Paulo and Belrose, arXiv 2501.16615.
- arXiv 2607.12166 (registered in the measurement-validity plan as "correlational recovery overstates causal validity"; no title or authors recorded). JumpReLU, arXiv 2407.14435 (same plan). GPTQ, arXiv 2210.17323 (Frantar et al.; registered in the GPTQ audit).
- M01 Koehn, EMNLP 2004. M02 Berg-Kirkpatrick, Burkett and Klein, EMNLP-CoNLL 2012. M04 Field and Welsh, JRSS-B 2007. M05 NLLB Team, arXiv 2207.04672 (FLORES-200). M06 Zhang and Nanda, ICLR 2024. M07 Heimersheim and Nanda, arXiv 2404.15255. S08 lm-evaluation-harness Belebele template.

**Shape of the final report.** This draft follows the usual order of papers in this area: abstract, introduction, background, method, experiments, discussion, limitations. The final report should keep that order, add a short statement of AI use as the course requires, and keep the evidence labels (or an equivalent wording) so that nulls and exploratory results are not read as findings.

**Added after writing, 29 October evening (verified by the orchestrator, not by the writing agent):** the 9B raw-file reproduction described in section 5.15 (hashes, headers, byte-identical tables), the interval computations for the 9B ratios, and the repository cleanup notes (dated corrections added to the older result pages; two stale documents were left in place after checking that a reproduction script still reads them).
