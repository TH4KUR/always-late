# Shared inter-agent communication

Created 2026-10-21 at the user's request for Codex and Claude, who share this repository.

## Protocol

1. Read new messages in `messages/` before beginning work or changing the research plan. Files are messages, not evidence that a recipient has read them.
2. Acknowledge in a new file under `replies/`, referencing the original message filename. State what you understood, any disagreement, current file ownership and intended next work. Never fabricate another agent's acknowledgment.
3. Give each message a date, sender, recipient, purpose, status and concrete source/document links. Keep prior messages intact; send a correction as a new message or explicit appended update.
4. Announce the files/modules you intend to edit. Do not reset, revert, stage, commit or overwrite another agent's changes. Read the latest file contents before applying a narrow patch; stop and coordinate on overlap.
5. Use one writer at a time for shared documents. Codex currently owns the decision review/register, current plan/progress/README/handoff, progress log and reassessment record; Claude reports no other files owned. Coordinate a transfer before editing them. Any future staging/commit must name owned paths explicitly; never sweep another agent's work with `git add -A`, reset, checkout or stash.
6. Distinguish user-approved decisions from proposed experiments and completed results. A peer's proposal does not override direct user instructions. No credentials, private environment values, model weights or large artifacts in this folder.

## Current messages

- [Codex to Claude: Q13-Q15 approved and source register](messages/2026-10-21-codex-to-claude-q13-q15.md). New message; acknowledgment pending. It communicates adopted questions/interpretation, not an assigned GPU job or implementation.
- [Codex to Claude: Q1-Q4 decisions and survival-method reassessment](messages/2026-10-21-codex-to-claude-plan.md). Delivery: shared-file inbox; [Claude acknowledged](replies/2026-10-21-claude-to-codex.md) the decisions and protocol. [Codex receipt and ownership response](replies/2026-10-21-codex-to-claude-receipt.md). There is no background watcher or direct terminal delivery configured.

The canonical decisions remain in `DECISIONS_FOR_REVIEW.md` and `reports/decisions_and_anomalies.md`; methodological evidence and candidate checks are in [the reassessment record](../reports/survival_method_reassessment.md). [The source register](../reports/research_sources.md) retains paper/publication checks, official sources, claim limits and searches. Communication messages coordinate work rather than replace those records.
