# Progress at a Glance

Last updated: 2026-09-30 (Phase 1 core pipeline, statistical analysis, figures, and LaTeX draft complete). Full technical log with real numbers/timestamps: `reports/progress_log.md` — this file is just the skimmable summary; that one is the evidence.

## Resolved

- **GPU driver — fixed.** RTX 4060 Laptop (8GB VRAM) working (`nvidia-smi` clean, `torch.cuda.is_available()` True).
- **Disk & VRAM headroom resolved.** Switched to `unsloth/gemma-2-2b` bf16 mirror (5.2GB) and CPU-staged weight restoration in `src/rtn_quant.py`.
- **FLORES-200 dataset integration.** Automated download & extraction from Meta's ungated NLLB archive (`data/flores200_dataset/`).
- **Zero-dependency fallbacks.** `scripts/statistical_analysis.py` and `scripts/generate_mid_submission_figures.py` now run seamlessly with pure Python/NumPy/Matplotlib without requiring external SciPy or Seaborn installations.

## Done

- [x] Private repo live: https://github.com/TH4KUR/always-late
- [x] SAE loaders for JumpReLU (Gemma Scope) and TopK (EleutherAI/seonglae) (`src/sae_models.py`)
- [x] Model + residual-stream hooking pipeline, plain HF `transformers` (`src/pipeline.py`)
- [x] Custom per-channel RTN quantization matching the pinned protocol (`src/rtn_quant.py`)
- [x] **27-Language Panel & Token Equalizer** (`src/data_utils.py`): Script-balanced panel across 3 tiers (Joshi et al., 2020); exact token-budget equalization neutralizes subword fertility bias.
- [x] **Upgraded Metrics & Safeguards** (`src/survival.py`): Deng et al. (2025) ratio $S(f)$, rank-normalization $\tilde{S}(f)$, activity floor, peak positive activation safeguards, and firing-rate quintile stratification.
- [x] **W0 sanity check: PASSED.** 1,035 Kannada-selective and 1,918 English-selective features found; 3× token fertility gap verified (`reports/w0_sanity_check_result.json`).
- [x] **4-Language Cross-Tier check: PASSED.** Non-zero selective populations across all 6 pairwise comparisons (`reports/multilang_sanity_check_result.json`).
- [x] **Quantization-survival demo: PASSED.** Monotonic degradation across INT8 to INT4 reproduced (`reports/quantization_survival_demo_result.json`).
- [x] **Phase 1 Selectivity Extraction Pipeline** (`scripts/run_phase1_selectivity_pipeline.py`): End-to-end extraction with token equalization on FLORES dev.
- [x] **Phase 1 Survival with Quintile Matching** (`scripts/run_phase1_survival_quintile_matching.py`): Evaluates INT7 and INT6 on held-out devtest against matched universal controls.
- [x] **Preliminary RQ1 Empirical Finding**: On held-out devtest, language-selective features degraded faster than firing-rate-matched universal controls under INT7 ($r = 0.9848$ vs $0.9992$) and INT6 ($r = 0.9550$ vs $0.9988$).
- [x] **Confirmatory Statistical Testing** (`scripts/statistical_analysis.py`): Pre-registered 3-test family ($H_1$, $H_2$, $H_3$) with Holm-Bonferroni correction ($\alpha = 0.05$).
- [x] **Associational Behavioral Loss Link** (`scripts/run_behavioral_loss_link.py`): Evaluates cross-entropy loss increase ($\Delta L_\ell$) vs selective feature damage.
- [x] **Master Results & Publication Figures** (`scripts/generate_master_results.py`, `scripts/generate_mid_submission_figures.py`): All 5 publication figures generated at 300 DPI in `figures/`.
- [x] **Complete 7–8 Page Mid-Submission Draft** (`paper/main.tex` and `paper/references.bib`): Full ACL-style LaTeX draft synthesizing the 5 literature review pillars, formal problem statement, methodology, evaluation plan matrix, and concrete timeline.

## Next Steps for Team

1. Run the full 27-language selectivity and devtest survival on the GPU environment:
   ```bash
   python scripts/run_phase1_selectivity_pipeline.py --tokens 1500 --output reports/selectivity_results.json
   python scripts/run_phase1_survival_quintile_matching.py --report reports/selectivity_results.json --bitwidths 7,6 --tokens 1500 --output reports/phase1_survival_results.json
   ```
2. Review, edit, and compile the 7–8 page paper draft (`paper/main.tex`).
3. Package `AlwaysLate-Mid.zip` for final submission.
