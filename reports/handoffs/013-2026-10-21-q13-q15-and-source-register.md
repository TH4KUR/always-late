# Handoff 013: Q13-Q15 adopted and research sources retained (2026-10-21)

## User instruction and applied decisions

The user requested published literature for Q13/Q15 and the results of existing Q14 tests, then asked for results and exact proposed answers before document edits. After reviewing those results/answers, the user said “works” and requested keeping track of all discussed papers and other sources alongside the earlier literature record.

- D-032 / Q13: keep both matching comparisons as associations; investigate whether language specificity predicts quantization vulnerability beyond baseline magnitude and what role magnitude plays. Do not label them identified direct/total causal effects. Baseline magnitude predates quantization, so cannot mediate that later intervention. Exact causal design and novelty remain open.
- D-033 / Q14: report the completed overall-tier and extra-selective-disparity tests as distinct questions. Preserve chronology and analysis status; no duplicate GPU run or retrospective promotion to confirmation. Carry the distinction into Q3 across the full width ladder.
- D-034 / Q15: investigate confidence, diffuse activity, feature count and magnitude as separate predictions against residual/normalization-scale and SAE-fidelity alternatives. Begin with available artifacts. The old N12 job, INT8/7/6-only scope and 1-2 h estimate are superseded, not approved unchanged. Precise new measurements, compute cost, unseen evaluation and job remain to be designed.

`DECISIONS_FOR_REVIEW.md` marks Q13-Q15 APPLIED for these decisions. This is not a claim that their empirical questions have been answered. Q1-Q4 policy remains: mask only, no further none or Duan numerical comparison, publishability with checked novelty, survival-method reassessment, INT8/7/6/5/4. Q6/Q7/Q8/Q11/Q12 are unanswered; Q10 has an earlier separate answer requiring its own follow-through. Do not expand this documentation step into those tasks.

## Sources and evidence

Canonical register: [research_sources.md](../research_sources.md). It consolidates 19 paper entries plus seven official-source/guidance groups, with published versus preprint/unverified status, exact primary links, inspected sections, supported claims and limitations. Earlier BOS, Gemma Scope, language-specific SAE, canonicality, identifiability and functional-evaluation sources remain; the Q13-Q15 additions are integrated into the original [method reassessment](../survival_method_reassessment.md) as well. `paper/` bibliography/prose is untouched.

Key additions: AWQ (MLSys 2024), Chang et al. (EMNLP 2025), Imai et al. (Statistical Science 2010), Silva et al. (PROPOR 2026), Olmo et al. (Findings NAACL 2025), Huang et al. (EACL 2026 short), Ferrando et al. (ICLR 2025); Gemma Scope's BlackboxNLP 2024 publication and Appendix D.3 are explicit. Sin Jing Xuan et al.'s close multilingual activation preprint is recorded as prior art with publication unverified, not ignored for novelty or promoted to accepted evidence. General magnitude-dependent quantization is already explored; the exact causal comparison was not settled by the verified papers located. Targeted search limits and queries are retained, not an absence or novelty proof.

Existing mask/eq Q14 results, read rather than rerun: T3 directional median-r-gap Holm p 0.9902; survival-rate-gap sensitivity p 0.7370. N7 interaction +0.9431 (p 0.00130) -> -0.00662 (p 0.98174) with magnitude; the adjusted feature-level tier main term remains +0.861 (p 0.00000303). N8 INT6 low/mid/high means 69.4/90.6/95.1%, slope +12.3 pp per step (Holm p 0.0002). N9 language-level adjusted slope -0.81 pp (p 0.667); not the same outcome/population as N7 and not causal mediation. Source links and INT5 caveats are in the register.

## Communication, ownership and next step

Claude previously confirmed receipt of Q1-Q4 and the one-writer protocol. [New message](../../communication/messages/2026-10-21-codex-to-claude-q13-q15.md) communicates the approved Q13-Q15 answers and source register; acknowledgment of that new message is pending. No automatic watcher. Codex retains shared review-document ownership; Claude corrections should come through messages until a transfer is agreed.

Next technical work is the bounded artifact-feasibility inventory and a concrete source-grounded measurement proposal using these questions. Existing caches lack quantized response profiles/sentence counts needed for full reassessment; do not mislabel reference statistics as quantized outputs. Some FP summaries are available for CPU checks; identify exactly what can be derived before proposing new collection. Both existing FLORES splits have been examined; a new plan does not make their outcomes unseen. No new pass, code change, model training, measurements, commit or push occurred in this documentation step.

## Verification

Check local links/anchors, decision status/ID consistency, source-entry coverage and unchanged numerical-result files. Preserve earlier progress-log entries and Claude's reply files. Do not rerun historical numerical/GPU workloads for documentation-only changes. The next writer must inspect current files and new messages before edits.
