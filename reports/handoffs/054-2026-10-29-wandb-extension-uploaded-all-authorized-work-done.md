# Handoff 054: wandb extension uploaded and verified; all authorized work is done (29 October 2026, 2026-10-29 02:35 IST)

> The ONLY source of the previous chat. A fresh chat with no memory must be able to continue from this file plus the repository. If this file and the repository disagree, the repository wins: say so in the next handoff.

## 0. Read this first

[Handoff 053](053-2026-10-28-survival-and-intervention-plan-complete.md) section 0 still applies in full (the user's request of 2026-10-27 verbatim in 050; main branch only; never edit hash-pinned entry points; never remove progress tracking; times from `date`; no secrets or weights in commits; **no 48-hour window ever existed**; Q29 NOT started).

## 1. State right now

- **Done and recorded:** the GPTQ extension (D-101 to D-111), the survival-and-intervention plan (D-110 to D-114), Hugging Face dataset `runs/gptq-extension/` (two commits, 1,011 files, private), **wandb: nine runs in `karma-skz-/anlp-project`, read back identical to the payload (D-115 design, D-116 upload)**; plain-language records `reports/gptq_extension/RESULTS.md` and `reports/survival_effect_rescue/README.md`; handoffs 048 to 054.
- **Corrections of 2026-10-29** (found by the pinned-value tests of the wandb headline generator, update paragraph under D-115): matched-RTN INT4 survival is **18.6 %** (not 18.7), GPTQ INT4 no-edit Belebele accuracy is **0.318** (not 0.319); D-104 and D-107 stay as written (history); `RESULTS.md`, the assessment, the generated appendix note and the wandb design are corrected; handoffs 050 to 053 and older ledger lines still quote the old digits as history. No analysis or conclusion changed.
- **Ada:** nothing queued or running. **Cron:** three recurring entries (`bf1ae581`, `ad8593e3`, `8db9afdb`), every 40 minutes, expire about 2026-11-03; idle firings only append one ledger line.

## 2. What remains: the user's decisions only

Q29 (paired precision audit: built, never deployed, not authorized); whether any of the GPTQ or survival-effect findings enter the course report (deadline 31 October 2026); Q32 stays open (default carried out; the user may still say the wandb runs are unwanted: nothing is deleted without being asked). No further job, model, analysis, steering or GPTQ-on-Llama is authorized.

## 3. Pitfalls

- wandb: the uploader runs on Ada from a deployed snapshot (`ADA_SSH='ssh -o HostKeyAlias=ada-gw1 karma@10.4.24.30' bash scripts/g0/deploy_phase1.sh repo_<commit>`; Python `/home2/karma/duan-verify/env/bin/python`); `wandb_ext.py verify` compares floats with a 1e-12 relative tolerance because wandb returns floats with fewer digits; never print or copy `~/.netrc`; the uploader refuses non-v2 run ids.
- Hugging Face: never run `hf_artifacts.py upload` for a new spec without `--manifest-name` and `--no-card`.
- The soft-cap fix of 2026-10-28 (`make_logits_fn`) also applies in `phase1_transfer_run.py` if that runner is ever used on a model without a cap.
- Recovery shares of the group patch (2.0 to 2.6 %) are never "shares of the damage".
