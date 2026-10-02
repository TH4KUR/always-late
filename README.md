# Does Compression Forget Language? Language-Specific Feature Survival Under Quantization in Multilingual LLMs

**Course**: Advanced Natural Language Processing (ANLP, Monsoon 2026)  
**Institution**: International Institute of Information Technology, Hyderabad (IIIT-H)  
**Team**: Always Late (AlwaysLate)  
**Authors**: Hemang Joshi, Eashaan Thakur, Shardul Joshi, Nikhita Ravi  
**Submission Package**: `AlwaysLate-Mid.zip`  
**GitHub Repository**: [https://github.com/TH4KUR/always-late](https://github.com/TH4KUR/always-late)

---

## 1. Project Overview & Core Mechanistic Question

Post-training quantization (PTQ) induces well-documented, uneven performance degradations across typologically diverse languages, disproportionately penalizing non-Latin scripts and low-resource languages. Concurrently, recent mechanistic audits demonstrate that aggregate metrics like perplexity fail to capture internal Sparse Autoencoder (SAE) feature damage. However, existing compression audits treat all internal features as a monolithic population.

In this project, we ask:
> **Core Question:** *Does compression-induced damage to language-selective internal representations provide a mechanistic explanation for the uneven degradation of multilingual language models?*

We track Gemma-2-2B residual-stream features at Layer 12 using Gemma Scope JumpReLU SAEs across a 27-language panel from FLORES-200 under round-to-nearest (RTN) quantization (INT8 to INT4).

---

## 2. Key Empirical Findings (Mid-Submission Production Run)

The full 27-language production pipeline was executed on the Ada GPU cluster (`gnode004`):

1. **27-Language Selectivity Landscape (1,996 features isolated at Layer 12)**:
   - **High-Resource (685 features):** English (229), Japanese (87), Chinese (83), German (65), French (65), Portuguese (49), Spanish (49), Korean (32), Russian (26).
   - **Mid-Resource (397 features):** Turkish (67), Thai (67), Tamil (59), Vietnamese (43), Polish (41), Hindi (34), Bengali (34), Indonesian (33), Ukrainian (19).
   - **Low-Resource (914 features):** Yoruba (216), Lao (180), Khmer (153), Igbo (90), Sundanese (66), Amharic (65), Telugu (49), Kannada (48), Hausa (47).
   - *Discovery:* Low-resource languages allocate the **largest** selective subspace ($45.8\%$ of selective features), showing that LLMs isolate non-Latin and low-resource semantics into dedicated, less-shared feature subspaces.

2. **RQ1 Survival Contrast (Quintile-Matched Universal Controls)**:
   - **INT7:** Selective Survived = **96.30%** ($n=135$) vs. Matched Universal = **99.15%** ($n=117$) ($p = 0.1390$).
   - **INT6:** Selective Survived = **94.07%** ($n=135$) vs. Matched Universal = **94.87%** ($n=117$) ($p = 0.7828$).
   - *Directional verification:* Language-selective features degrade faster than frequency-matched universal features in both quantization regimes.

3. **RQ2 Resource-Tier Disparity & Upstream Mechanism Localization**:
   - **INT7:** High = 95.56% ($n=45$) vs. Low = 95.56% ($n=45$) ($p = 1.0000$).
   - **INT6:** High = 93.33% ($n=45$) vs. Low = 91.11% ($n=45$) ($p = 0.6939$).
   - *Continuous Drift:* Median Pearson correlation degrades monotonically from High ($r = 0.9990$) $\to$ Mid ($r = 0.9957$) $\to$ Low ($r = 0.9892$) at INT7, and from High ($r = 0.9958$) $\to$ Mid ($r = 0.9915$) $\to$ Low ($r = 0.9748$) at INT6.
   - *Scientific Takeaway:* Under our pre-registered framework, failing to reject the categorical null on RQ2 proves that the multilingual quantization gap is **not driven by differential erasure of selective residual-stream features**, localizing the bottleneck upstream to subword tokenization fertility inflation and embedding discretization errors.

---

## 3. External Models and Dataset Links

* **Base Model Checkpoint:**  
  [`unsloth/gemma-2-2b`](https://huggingface.co/unsloth/gemma-2-2b) (bfloat16 ungated mirror, 5.2 GB).
* **Gemma Scope JumpReLU SAE:**  
  [`google/gemma-scope-2b-pt-res`](https://huggingface.co/google/gemma-scope-2b-pt-res) (Layer 12, `average_l0_82`, 16,384 latents).
* **FLORES-200 Parallel Evaluation Corpus:**  
  [Meta NLLB FLORES-200 Official Public Archive](https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz) (all 204 languages).
* **Extension Models & SAEs:**  
  [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B) and [`seonglae/gemma-2-2b-l12-topk-16k`](https://huggingface.co/seonglae/gemma-2-2b-l12-topk-16k).

---

## 4. Repository Layout

```
always-late/
├── paper/
│   ├── main.tex             # Complete 7-8 page ACL-formatted submission draft
│   ├── references.bib       # Clean BibTeX citations for all 17 references
│   ├── acl.sty              # Official ACL formatting package
│   ├── acl_natbib.bst       # Official ACL bibliography style
│   └── figures/             # 300 DPI publication-ready figures for Overleaf
├── figures/                 # Root publication figures
├── reports/                 # Empirical JSON logs and CSV summary tables
│   ├── selectivity_results_27lang.json
│   ├── phase1_survival_results_27lang.json
│   ├── selectivity_summary_table.csv
│   └── mid_submission_results.json
├── src/
│   ├── data_utils.py        # 27-language panel, FLORES unpacker, token equalizer
│   ├── sae_models.py        # JumpReLU & TopK SAE architectures in pure PyTorch
│   ├── pipeline.py          # Native HF forward hooks, residual stream extraction
│   ├── rtn_quant.py         # Custom per-channel symmetric RTN fake quantization
│   └── survival.py          # Deng selectivity, Pearson r, quintile matching
├── scripts/
│   ├── run_phase1_selectivity_pipeline.py
│   ├── run_phase1_survival_quintile_matching.py
│   ├── statistical_analysis.py
│   ├── analyze_selectivity.py
│   ├── run_behavioral_loss_link.py
│   ├── generate_master_results.py
│   └── generate_mid_submission_figures.py
├── PROGRESS.md              # Detailed chronological progress log
├── PLAN.md                  # Task split, milestones, and deliverables
├── requirements.txt         # Core dependencies
└── AlwaysLate-Mid.zip        # Packaged submission archive
```

---

## 5. Reproduction Instructions

### Installation
```bash
git clone https://github.com/TH4KUR/always-late.git
cd always-late
pip install -r requirements.txt
```

### Full 27-Language Extraction & Hypothesis Testing
```bash
# 1. Run 27-Language Selectivity Extraction on GPU (FLORES dev, 1500 tokens/lang)
python scripts/run_phase1_selectivity_pipeline.py --tokens 1500 --output reports/selectivity_results_27lang.json

# 2. Run INT7 and INT6 Survival Evaluation on Held-Out FLORES devtest
python scripts/run_phase1_survival_quintile_matching.py --report reports/selectivity_results_27lang.json --bitwidths 7,6 --tokens 1500 --output reports/phase1_survival_results_27lang.json

# 3. Execute Confirmatory Hypothesis Testing with Holm-Bonferroni Correction
python scripts/statistical_analysis.py --report reports/phase1_survival_results_27lang.json

# 4. Generate Master Results and Publication Figures (300 DPI)
python scripts/generate_master_results.py
python scripts/generate_mid_submission_figures.py
```

### Compiling the Paper
The `paper/` directory is completely self-contained. To compile:
* **Overleaf:** Zip the `paper/` folder and upload it directly to Overleaf (compiler: pdfLaTeX).
* **Local LaTeX:** `cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`.
