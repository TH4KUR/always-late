"""Master Results Aggregator for Mid-Submission 1.

Implements STEP 7 of the Mid-Submission priority plan:
- Automatically reads completed experiment outputs from `reports/`:
    - W0 & multilang sanity checks
    - Quantization sweep demo (`reports/quantization_survival_demo_result.json`)
    - Selectivity extraction (`reports/selectivity_results.json`)
    - Phase 1 survival evaluation (`reports/phase1_survival_results.json`)
    - Behavioral loss link (`reports/behavioral_loss_link_results.json`)
- Combines metadata, model/SAE specifications, primary RQ1/RQ2 contrasts,
  confirmatory statistical tests, and Holm-Bonferroni corrections into a
  single unified master file: `reports/mid_submission_results.json`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"


def generate_master_results(output_file: str = "reports/mid_submission_results.json"):
    master = {
        "project_title": "Does Compression Forget Language? Language-Specific Feature Survival Under Quantization and Pruning in Multilingual LLMs",
        "institution": "IIIT Hyderabad",
        "submission_stage": "Mid-Submission 1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_and_environment": {
            "primary_model": "Gemma-2-2B (unsloth/gemma-2-2b bf16 mirror)",
            "sae_suite": "Gemma Scope (google/gemma-scope-2b-pt-res)",
            "target_layer": 12,
            "sae_width": "16k (16,384 features)",
            "sae_config": "average_l0_82",
            "gating_type": "JumpReLU",
            "compression_method": "Per-channel symmetric RTN fake quantization",
            "quantized_modules": "Attention (q, k, v, o_proj) and MLP (gate, up, down_proj) - 182 Linear modules total",
            "excluded_modules": "Embedding layers, RMSNorms, lm_head",
        },
        "dataset_metadata": {
            "source": "FLORES-200 (Meta NLLB ungated release)",
            "panel_size": 27,
            "resource_tiers": "9 High, 9 Mid, 9 Low (Joshi et al., 2020 taxonomy)",
            "script_families": "Latin, Devanagari, Dravidian, Cyrillic, CJK, Hangul, Ethiopic, Khmer, Lao",
            "split_tagging": "FLORES dev (content-matched parallel text)",
            "split_evaluation": "FLORES devtest (strictly held out)",
            "control_mechanism": "Exact token-budget equalization across languages to eliminate subword fertility bias",
        },
        "experiments": {},
    }

    # 1. Ingest W0 Sanity Check
    w0_path = REPORTS_DIR / "w0_sanity_check_result.json"
    if w0_path.exists():
        master["experiments"]["w0_sanity_check"] = json.loads(w0_path.read_text(encoding="utf-8"))

    # 2. Ingest Multilingual Sanity Check
    multi_path = REPORTS_DIR / "multilang_sanity_check_result.json"
    if multi_path.exists():
        master["experiments"]["multilang_sanity_check"] = json.loads(multi_path.read_text(encoding="utf-8"))

    # 3. Ingest Quantization Survival Demo
    quant_demo_path = REPORTS_DIR / "quantization_survival_demo_result.json"
    if quant_demo_path.exists():
        master["experiments"]["quantization_survival_demo"] = json.loads(quant_demo_path.read_text(encoding="utf-8"))

    # 4. Ingest Phase 1 Selectivity Results
    sel_path = REPORTS_DIR / "selectivity_results_27lang.json"
    if sel_path.exists():
        master["experiments"]["phase1_selectivity"] = json.loads(sel_path.read_text(encoding="utf-8"))

    # 5. Ingest Phase 1 Survival Results (RQ1 & RQ2)
    surv_path = REPORTS_DIR / "phase1_survival_results_27lang_gpu.json"
    if surv_path.exists():
        master["experiments"]["phase1_survival_quintile_matching"] = json.loads(surv_path.read_text(encoding="utf-8"))

    # 6. Ingest Behavioral Loss Link
    behavior_path = REPORTS_DIR / "behavioral_loss_link_results.json"
    if behavior_path.exists():
        master["experiments"]["behavioral_loss_link"] = json.loads(behavior_path.read_text(encoding="utf-8"))

    # 7. Pre-registered Confirmatory Hypotheses Registry
    master["confirmatory_hypothesis_framework"] = {
        "correction_method": "Step-down Holm-Bonferroni across 3 confirmatory tests (alpha = 0.05)",
        "family": [
            {
                "id": "H1",
                "statement": "Language-selective features survive INT7 RTN quantization at a significantly different rate than quintile-matched universal features.",
                "dependent_variable": "Pearson correlation r(f) >= 0.90 (survived)",
                "control": "Baseline firing-rate quintile matching",
            },
            {
                "id": "H2",
                "statement": "Language-selective features survive INT6 RTN quantization at a significantly different rate than quintile-matched universal features.",
                "dependent_variable": "Pearson correlation r(f) >= 0.90 (survived)",
                "control": "Baseline firing-rate quintile matching",
            },
            {
                "id": "H3",
                "statement": "Language-selective features tied to Low-resource languages exhibit lower median survival than High-resource languages.",
                "dependent_variable": "Per-language median Pearson correlation r(f)",
                "test": "One-sided permutation test across the 27-language panel, controlling for script family",
            },
        ],
    }

    out_file = REPO_ROOT / output_file
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(master, indent=2))
    print(f"Master Mid-Submission results generated and saved to: {out_file}")
    return master


if __name__ == "__main__":
    generate_master_results()
