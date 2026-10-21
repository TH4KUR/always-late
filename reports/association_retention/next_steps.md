# Next steps after the D-035 study: options for the review

Written 2026-10-21 while the user was away. Nothing below is started or approved; each option names what it would show, its cost (my estimates, unmeasured unless stated), the sources it rests on in [the register](../research_sources.md), and what the results of D-035 suggest about it. The submission date is 2026-10-31. Results and limits of the study itself: [interpretation](interpretation.md).

## What the study leaves open

1. The loss of association could reflect the model, the frozen dictionary, or a generic sensitivity of near-ties to any perturbation. The data point to the last at the high widths (loss concentrates in features with a small baseline margin; INT16/INT12 already lose 0.7 %), and to scale effects at INT4 (activation scale and firing rate relate to loss).
2. Nothing was measured about function: whether lost features still do what they did (ablation, steering), or whether retained ones still do.
3. The frozen SAE's own fidelity under quantization, by language and width, was not measured (reconstruction error, active-feature counts, residual scale): candidates M2 and Q15 (D-034).
4. Results are exploratory on already examined FLORES data; the cohort is English-heavy (1,719 of 8,086 features).

## Options

| # | option | what it would show | cost and needs | sources and notes |
|---|---|---|---|---|
| A | **CPU-only analyses of the existing files**: per-language descriptive models of loss with baseline margin, share and scale as covariates and language as a grouping (exploratory, no new test family), the margin finding quantified against the INT16/INT12 floor, per-language cohorts without English | whether the near-tie mechanism and the INT4 scale effect hold within languages; how English weighting shapes the pooled numbers | hours; `feature_status.csv`, profiles and cohort already exist; tested code first | Q13/Q14 and D-032 (association, no causal labels; define the causal question first); register P10 to P13 for the magnitude background |
| B | **SAE fidelity by language and width** (M2/Q15): per language and condition the normalised reconstruction error, active-feature counts and concentration, and residual-stream norm at layer 12 | whether a moving language peak reflects the dictionary failing on quantized activations (instrument) or the representation changing; tests the user's hypothesis of weaker, more diffuse activity in lower-resource languages as separate predictions | one GPU pass like the full run (3.5 h without tagging; the sink can be extended by a few accumulators); about one day with tests; the SAE decoder is needed | register P03 (Gemma Scope fidelity worst on multilingual text, 65K SAE, delta loss; bf16 and SAE-quantization notes), P09 (SAEBench dimensions), P14 to P16 (magnitude, confidence, uncertainty: do not equate them); close preprint P17 |
| C | **Functional check** (M3): ablate or clamp lost and retained features (matched cohorts, matched controls) in full precision and in a quantized model and compare the cross-entropy effect by language | whether association retention predicts functional retention; the strongest claim a paper could make and the most expensive | GPU, per-language loss with hooks over several groups and conditions: 1 to 2 days with tests and review; needs a justified subset of languages and features | register P02 (decoder-direction projection scored by cross-entropy, English dropped, Gemma-2-2B and FLORES), P08 (full-distribution SAEs are weaker on necessity and sufficiency; templatic tasks only), P06 (SAEs are not canonical units) |
| D | **Perturbation control**: the same measurement under random weight noise of matched relative size instead of RTN | whether the near-tie flips are generic to any perturbation of that size or specific to quantization | one GPU pass (about 3.5 h per noise level) plus the existing analysis; moderate | register P11 (small residuals and large errors in English text), P13; our INT16/INT12 floor is the in-study analogue and may already answer most of it |
| E | **All-token intervals**: sentence shards for the all-token stream | intervals for the sensitivity stream | one more full run (3.5 h; several GB more disk, which `/home2` cannot hold without freeing space) | low value: the two streams already agree closely |
| F | **Writing support**: tables and figures from the generated files for the team's paper (retention by width, loss by margin decile, per-language rows), the closest-work paragraph from [the register](../research_sources.md) (P20 to P29 and the audit in [plan_verification](plan_verification.md)) | nothing new; makes the result usable | hours; prose is the team's (course policy) | D-029: publishability is the objective; no novelty is claimed here, a dated closest-work check must precede any claim |

## Suggested order (a recommendation, to be reviewed)

1. **A** first (cheap, uses what exists, sharpens what B and C would test).
2. **B** next: it addresses the largest open caveat (instrument versus representation) and answers the D-034 hypothesis; one GPU pass.
3. **C** only if time allows and if A and B leave the function question as the main gap; otherwise state it as the limit.
4. **F** in parallel with whichever is running. **D** and **E** are optional.

## Decisions needed from the user

- Which of A to F to run (the plan for each would be written test first, with a source check, before any job).
- Whether the Hugging Face dataset is made public and under which licence for our own statistics (`license: other` for now); the upload of the association run is in progress and the dataset stays private.
- The remaining open review items Q6, Q7, Q8, Q11 and Q12 in `DECISIONS_FOR_REVIEW.md` (not blocking).
