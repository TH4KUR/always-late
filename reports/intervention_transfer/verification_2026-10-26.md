# Verification of Codex's intervention-transfer direction (26 October 2026, Claude)

Requested by the user on 2026-10-26 ~07:40 IST: check that the new directions are worthwhile and not already done in credible literature; if so, hand the implementation to the 40-minute cron without losing the running Llama check. Inputs (Codex's, uncommitted, read-only for Claude): [`RESEARCH_DIRECTION.md`](../../RESEARCH_DIRECTION.md), [design](../../docs/superpowers/specs/2026-10-26-intervention-transfer-research-design.md), [plan](../../docs/superpowers/plans/2026-10-26-intervention-transfer-main-and-contingencies.md), [novelty register](../publication_strategy_2026-10-26/future_work_and_novelty.md), [memo](../publication_direction_2026-10-26.md).

## Verdict: worthwhile, go, with the corrections below

1. **Our own evidence supports the motivation.** Checked in code: the F5 fixed-amplitude variant (D-063) runs each quantized state on its OWN cached residual (`scripts/g0/phase1_func_f5_run.py:316`, `cache_state` under the Q prefix). So D-063's sentence "a property of the quantized downstream map" is not identified: context and downstream weights changed together. The crossed 2x2 (prefix FP/Q x suffix FP/Q) is the experiment that identifies it, and it reuses the F5 machinery (`lower`, `upper_final`, `fingerprint_feature(..., amplitudes=...)`). Correction appended as D-088.
2. **The factorial test values in the plan are correct** (F_s(h)=w_s h^2, h 1/2, v .1, w 1/2: effects -.19, -.39, -.38, -.78; C -.20, W -.19, I -.20, total -.59; recomputed by hand).
3. **Literature (subagent search, 17 queries; the two most important items re-checked by Claude in the primary text):** all 10 cited sources exist and match, with three citation corrections. No paper does the crossed context x weights decomposition of a fixed SAE-feature intervention under quantization, or a held-out diagnostic of per-feature transfer error. Branch A (layerwise propagation under compression) and Branch B (output-language steering vs comprehension under compression) were not found either. Absence in a bounded search is not proof.

## Required corrections before or during Task 1 (freeze)

- **Chang et al., EMNLP 2025, §6.1-6.2 (verified in the PDF, p. 7-8):** "leaving late layers unquantized has little effect on NLLs, implying that the cumulative error in residual states, not the precision of upper-layer weights, is the primary cause"; restoring W_gate weights to 16 bit is ineffective while patching gate inputs works. This is the LOSS-level context-vs-weights split (non-factorial, no interaction, no interventions). Consequences: (a) cite it as the loss-level precedent; (b) pre-register the expectation that C dominates W for interventions too. Confirming C > W alone is NOT a contribution. The contribution is the intervention target, the interaction term, the specificity contrast vs matched/random directions, and the held-out diagnostic.
- **Bhandari & Wade, arXiv:2609.06473 (6 Sep 2026, verified abstract), "Steering Under Compression":** CAA steering under INT8/NF4 on 7-9B models; steering vectors stay collinear with FP16 ones; no SAE and no context/weight split. Add to the novelty register as the closest steering-under-quantization work.
- **Guo et al., arXiv:2610.15850 (gradient intervention attribution):** a REQUIRED baseline of the diagnostic comparison (the plan has it as "only if budget fits"; make it mandatory, since it is cheap: one backward pass per context).
- Citation fixes: Sathyanarayanan 2609.23125's title is "Perplexity Cost Understates What Activation Quantisation Breaks"; Borobia 2603.25325's arXiv-listed repo (`hborobia/...`) 404s, the live one is `github.com/hecboar/sae-pruning-paper` (its quant pilot reports only fixed-dictionary Jaccard 0.830/0.603 on Gemma 3 1B); Duan v2 is dated 4 June 2026, single author.

## Risks to carry into the freeze (not blockers)

- **Generic drift is the likely outcome** (D-061, D-064: swapped directions drift like features). The specificity gate will then say "generic"; the paper value must come from the decomposition plus the held-out diagnostic, which is plausible precisely because the drift is generic.
- **Llama:** interim FP check gives E > 0 in 21/27 languages (P1 not met) and heavy-tailed controls. Expect fewer eligible features (FP effect norm > 5x floor) in Llama. Report coverage; do not relax the eligibility rule after seeing it.
- **States:** the plan uses FP, INT16 (floor), INT6, INT4. This is a new study, not Study G, so the D-075 floor of 6 states does not bind; the group panel keeps 27 languages x 300 sentences x 3 draws. Adding INT8/INT5 cells is an option (Q28, default: plan as written).
- **Hybrid validity:** same-state hybrid must equal the native forward exactly (gate); report unedited hybrid loss for Q-prefix/FP-suffix and FP-prefix/Q-suffix.
- **Doc conflict:** the memo says Arabic in the external panel and 96 features; the plan (which supersedes the memo) says Yoruba and 404. Follow the plan.

## Hand-off to the cron (no run lost)

The running Llama job `llama-lm-a` 39721 (INT6 in progress at 07:44) and the Llama comparison and robustness analysis stay FIRST. The new study starts only after that comparison is written up, and it never deploys to Ada while 39721 runs (cron rule 2c). Codex's design and plan files are untracked and hers: the cron reads them and never edits or commits them. New files of this study go under `reports/intervention_transfer/` and the `scripts/g0/phase1_transfer_*` paths named in the plan.
