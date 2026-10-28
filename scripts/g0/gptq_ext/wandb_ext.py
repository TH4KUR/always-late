"""Extension of the wandb schema-v2 post-hoc upload by nine compact runs (design reports/wandb_schema_v2_gptq_extension.md, D-115): seven for the GPTQ extension (three arms: legacy RTN, matched RTN, GPTQ) and two for the
survival-versus-effect analyses (Gemma, Llama). The v2 builder and uploader (`scripts/g0/wandb_upload_v2.py`) are used UNCHANGED: this module only supplies the family list, the read-back verification and a command line.

  python3 scripts/g0/gptq_ext/wandb_ext.py build --out DIR [--dry-run] [--allow-dirty]        offline: payload.json and files.tar (every number is read from a committed file)
  python  scripts/g0/gptq_ext/wandb_ext.py upload --payload DIR/payload.json --root EXTRACTED_DIR   needs wandb and credentials in the environment of the machine it runs on (Ada)
  python  scripts/g0/gptq_ext/wandb_ext.py verify --payload DIR/payload.json                  reads every run back through the API and lists every difference (empty = identical)"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wandb_upload_v2 as wu  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
HL = "reports/gptq_extension/wandb_headlines.json"
GX = "reports/gptq_extension/results/"
FV = "reports/functional_validation/"
SE = "reports/survival_effect_rescue/"
FREEZE_EXT = "reports/gptq_extension/freeze.json"
CODE_REF = "snapshot repo_88cfa49 (commit 88cfa491645b8b9ce4d86fbd3211e97d66685f89)"
QUANT = ("three arms, weights only, simulated by dequantized weights: legacy RTN (the original implementation: one scale shared across the Q/K/V heads), matched RTN (per-output-channel symmetric grid), "
         "GPTQ (error compensation, 221,184 Wikipedia calibration tokens in the 27 languages)")
GEMMA = {"model": wu.GEMMA["model"], "sae": wu.GEMMA["sae"], "layer": wu.GEMMA["layer"], "quantizer": QUANT}
LLAMA = {"model": "meta-llama/Llama-3.2-1B", "sae": "seonglae/Llama-3.2-1B-sae, FineWeb TopK (k = 48, 14,336 features), hook blocks.12.hook_resid_pre (= blocks.11.hook_resid_post)", "layer": 12,
         "quantizer": "legacy per-output-channel RTN only (the shared method of both systems), weights only, simulated"}
COMMON_LIMITS = ("exploratory (D-101): no multiplicity rule across the three arms; each arm's intervals are its own unless a contrast is stated as paired; the arms differ in more than one factor (legacy to matched: grid "
                 "and granularity; matched to GPTQ: error compensation AND the calibration text, which RTN never saw); one model, one layer, one SAE, weights-only simulated quantization; FLORES sentences examined in earlier work")
ARMS = (("legacy", "legacy RTN"), ("matched", "matched RTN"), ("gptq", "GPTQ"))


def _hl_metrics(slug):
    keys = json.loads((ROOT / HL).read_text())[slug]
    return [wu.M(f"headline/{k}", HL, slug, k) for k in sorted(keys)]


def _fam(slug, name, group, decisions, title, states, jobs, metrics, tables, files, limits, model=GEMMA, freeze=FREEZE_EXT, code_ref=CODE_REF):
    return {"slug": slug, "name": name, "group": group, "status": "exploratory", "status_text": "exploratory", "decisions": decisions, "title": title, **model, "states": states, "jobs": jobs, "code_ref": code_ref,
            "freeze": freeze, "limits": limits, "metrics": metrics, "tables": tables, "images": [], "files": files, "history": None}


def _se_metrics():
    R = SE + "results/survival_effect_results.json"
    m = [wu.M("headline/overall_statement", R, "overall_statement"), wu.M("headline/cells_met", R, "cells_met_primary"), wu.M("headline/n_primary", R, "population_counts", "primary")]
    for arm, key in ARMS:
        for st in ("INT6", "INT4"):
            c = ["cells", f"{key}|{st}|primary"]
            m += [wu.M(f"headline/improvement_B_over_A_{st}_{arm}", R, *c, "improvement", "B_over_A", "improvement"), wu.M(f"headline/partial_spearman_{st}_{arm}", R, *c, "partial_spearman_given_A", "rho"),
                  wu.M(f"headline/spearman_{st}_{arm}", R, *c, "spearman", "rho")]
    return m


def _llama_metrics():
    R = SE + "llama/results/llama_survival_effect_results.json"
    m = [wu.M("headline/gate_ok", R, "gate", "ok"), wu.M("headline/gate_floor", R, "gate", "floor"), wu.M("headline/gate_median_cos16", R, "gate", "median_cos16"), wu.M("headline/n_reliable", R, "gate", "n_reliable"),
         wu.M("headline/overall_statement", R, "overall_statement"), wu.M("headline/cells_met", R, "cells_met_primary")]
    for st in ("INT6", "INT4"):
        c = ["cells", f"{st}|primary"]
        m += [wu.M(f"headline/improvement_B_over_A_{st}", R, *c, "improvement", "B_over_A", "improvement"), wu.M(f"headline/partial_spearman_{st}", R, *c, "partial_spearman_given_A", "rho"), wu.M(f"headline/spearman_{st}", R, *c, "spearman", "rho")]
    return m


def _arm_tables(stem, paths):
    return [(f"{a}_{stem}", paths[a]) for a, _ in ARMS]


FAMILIES_EXT = [
    _fam("gptq-ext-stage1-survival-loss", "gptq-extension/stage1-survival-loss", "gptq-extension", ["D-104", "D-106"],
         "Phase 1 on the 27 languages: feature survival, next-token loss and the pre-set tests T1 to T3, legacy RTN against matched RTN against GPTQ", ["FP", "INT8", "INT7", "INT6", "INT5", "INT4"],
         {"gptq-p1-mask": 42951, "rtnm-p1-mask": 42952}, _hl_metrics("gptq-ext-stage1-survival-loss"), [("stage1_summary", GX + "stage1/stage1_summary.csv"), ("stage1_paired_per_language", GX + "stage1/stage1_paired_per_language.csv")],
         [GX + "stage1/README.md", GX + "stage1/stage1_comparison.md", GX + "stage1/bridge.json", GX + "stage1/analysis_gptq/analysis.json", GX + "stage1/analysis_gptq/analysis.md", GX + "stage1/analysis_rtnm/analysis.json", GX + "stage1/analysis_rtnm/analysis.md"],
         COMMON_LIMITS + "; the pre-set tests are confirmatory for the legacy arm only, the two new arms are two more families (D-106); headline means are over the 27 languages, not pooled"),
    _fam("gptq-ext-transfer-main", "gptq-extension/transfer-main", "gptq-extension", ["D-105"],
         "Intervention transfer, main matrix (27 languages, FLORES): the pre-set claim verdicts and the size of the transfer error per arm", ["INT6", "INT4"],
         {"transfer feature and group, INT4 and INT6, both arms": "42953 to 42960"}, _hl_metrics("gptq-ext-transfer-main"),
         [("transfer_claims", GX + "stage2/transfer_claims.csv"), ("transfer_feature_level", GX + "stage2/transfer_feature_level.csv"), ("transfer_group_summary", GX + "stage2/transfer_group_summary.csv"), ("transfer_group_paired", GX + "stage2/transfer_group_paired.csv")],
         [GX + "stage2/README.md", GX + "stage2/transfer_comparison.md", GX + "stage2/analysis_gptq/claim_status.json", GX + "stage2/analysis_rtnm/claim_status.json"],
         COMMON_LIMITS + "; feature-level intervals are each arm's own (not paired); group-level contrasts are paired by language; the INT16 floors are the legacy rounding in every arm"),
    _fam("gptq-ext-transfer-external", "gptq-extension/transfer-external", "gptq-extension", ["D-111"],
         "Intervention transfer on fresh Wikipedia text in six languages (the external panel): claim verdicts, transfer error and held-out prediction per arm", ["INT6", "INT4"],
         {"external group and feature level, both arms": "44209 to 44216"}, _hl_metrics("gptq-ext-transfer-external"),
         [("transfer_claims", GX + "external/transfer_claims.csv"), ("transfer_feature_level", GX + "external/transfer_feature_level.csv"), ("transfer_group_summary", GX + "external/transfer_group_summary.csv"), ("transfer_group_paired", GX + "external/transfer_group_paired.csv")],
         [GX + "external/README.md", GX + "external/transfer_comparison.md", GX + "external/analysis_gptq/claim_status.json", GX + "external/analysis_rtnm/claim_status.json"],
         COMMON_LIMITS + "; six languages only (the paired group contrasts are very coarse); the GPTQ calibration text is Wikipedia like the panel, so domain match cannot be separated from error compensation"),
    _fam("gptq-ext-loss-task-rescue", "gptq-extension/loss-task-rescue", "gptq-extension", ["D-107"],
         "Zeroing the 64 most language-selective features: the harm to the language's loss kept at 6 and 4 bits (E4) and the share of the loss gap recovered by patching full-precision activations (E5), per arm", ["FP", "INT6", "INT4"],
         {"E4 and E5, both arms, INT4 and INT6": "43862 to 43869"}, _hl_metrics("gptq-ext-loss-task-rescue"),
         _arm_tables("retention", {"legacy": FV + "results/eval/retention.csv", "matched": GX + "stage3/analysis_rtnm_eval/retention.csv", "gptq": GX + "stage3/analysis_gptq_eval/retention.csv"})
         + _arm_tables("rescue", {"legacy": FV + "results/rescue/rescue.csv", "matched": GX + "stage3/analysis_rtnm_rescue/rescue.csv", "gptq": GX + "stage3/analysis_gptq_rescue/rescue.csv"}),
         [GX + "stage3/analysis_rtnm_eval/report.md", GX + "stage3/analysis_gptq_eval/report.md", GX + "stage3/analysis_rtnm_rescue/report.md", GX + "stage3/analysis_gptq_rescue/report.md"],
         COMMON_LIMITS + "; a recovery share is the repair this particular patch achieves, not a share of causal responsibility and not an upper bound; groups and controls are the old ones (defined from full-precision data)"),
    _fam("gptq-ext-belebele", "gptq-extension/belebele", "gptq-extension", ["D-107"],
         "Belebele reading comprehension (18 covered languages) with the language group removed: accuracy without any edit and the group-removal effect per arm (full-precision accuracy 0.4257 in all arms)", ["FP", "INT6", "INT4"],
         {"E7, both arms, INT4 and INT6": "43893 to 43896"}, _hl_metrics("gptq-ext-belebele"),
         _arm_tables("accuracy", {"legacy": FV + "results/mc_eval/accuracy.csv", "matched": GX + "stage3/analysis_rtnm_mc/accuracy.csv", "gptq": GX + "stage3/analysis_gptq_mc/accuracy.csv"}),
         [GX + "stage3/analysis_rtnm_mc/report.md", GX + "stage3/analysis_gptq_mc/report.md"],
         COMMON_LIMITS + "; the pre-set Belebele statements S1 to S5 need all six bit widths and were not rerun; the legacy 4-bit model is at chance, the other two are above it"),
    _fam("gptq-ext-f5", "gptq-extension/f5", "gptq-extension", ["D-108"],
         "F5: does a single SAE feature do the same thing after rounding? Effect similarity W1 on the 1,487-feature sample and the pre-set statements, per arm", ["FP", "INT6", "INT4"],
         {"F5, both arms": "44072, 44073"}, _hl_metrics("gptq-ext-f5"), _arm_tables("f5_tables", {"legacy": FV + "results/f5/f5_tables.csv", "matched": GX + "f5/rtnm/f5_tables.csv", "gptq": GX + "f5/gptq/f5_tables.csv"}),
         [GX + "f5/README.md", GX + "f5/rtnm/statements.md", GX + "f5/rtnm/report.md", GX + "f5/gptq/statements.md", GX + "f5/gptq/report.md"],
         COMMON_LIMITS + "; stratified sample of 1,487 features; the lost/retained labels and the F4 pairs are the legacy arm's (association retention was not repeated); INT5 and INT8 were not run for the new arms"),
    _fam("gptq-ext-study-g", "gptq-extension/study-g", "gptq-extension", ["D-109"],
         "Study G: net harm per language when the language group is scaled down and up, full precision, 6 and 4 bits, and the pre-set verdicts, per arm", ["FP", "INT6", "INT4"],
         {"Study G INT4 and INT6, both arms": "44074 to 44077"}, _hl_metrics("gptq-ext-study-g"),
         _arm_tables("curves", {"legacy": FV + "gain_study/results/curves.csv", "matched": GX + "gain/rtnm/curves.csv", "gptq": GX + "gain/gptq/curves.csv"})
         + _arm_tables("alpha_star", {"legacy": FV + "gain_study/results/alpha_star.csv", "matched": GX + "gain/rtnm/alpha_star.csv", "gptq": GX + "gain/gptq/alpha_star.csv"}),
         [GX + "gain/README.md", GX + "gain/rtnm/report.md", GX + "gain/gptq/report.md"],
         COMMON_LIMITS + "; three states only (FP, INT6, INT4) for the new arms, the legacy tables hold six; groups and matched controls are the old ones; the best gain alpha* is read on a coarse grid"),
    _fam("survival-effect-gemma", "survival-effect/gemma", "survival-effect", ["D-112", "D-113"],
         "Does the continuous activation correlation R1 of a feature predict the effect similarity W1 of its ablation beyond simple baselines? Gemma-2-2B, 1,487 features, three arms, INT6 and INT4, held out by language (post hoc)", ["INT6", "INT4"],
         {"F5 inputs": "44072, 44073 and the old f5 jobs"}, _se_metrics(), [],
         [SE + "stage4_gemma_report_2026-10-28.md", SE + "stage3_analysis_freeze_2026-10-28.md", SE + "results/survival_effect_results.json", SE + "all_results_assessment_2026-10-28.md"],
         "post hoc and exploratory (the committed F5 tables already held the correlations); R1 and W1 come from the same perturbed model; held out by language, not by document; stratified sample; baselines limited to five; the 10 % target is the project's, not universal; one model, layer, SAE",
         freeze=SE + "stage3_analysis_freeze_2026-10-28.md", code_ref="scripts/g0/survival_effect.py (commit e9bb2f3)"),
    _fam("survival-effect-llama", "survival-effect/llama", "survival-effect", ["D-114"],
         "The same question on Llama-3.2-1B (FineWeb TopK SAE, 643 features, legacy RTN, INT6 and INT4) with the instrument gate on Llama's own noise floor (post hoc)", ["FP", "INT16", "INT6", "INT4"],
         {"llama-f5": 45221}, _llama_metrics(), [],
         [SE + "llama/llama_vs_gemma_report_2026-10-28.md", SE + "llama/llama_f5_addendum_2026-10-28.md", SE + "llama/results/llama_survival_effect_results.json"],
         "post hoc and exploratory; two fixed model-plus-SAE setups, no population claim, overlap of intervals is not equivalence; 643 features with two roles; the evaluation sentences were examined earlier; the raw Llama files are not uploaded (weights licence caution)",
         model=LLAMA, freeze=SE + "llama/llama_f5_addendum_2026-10-28.md", code_ref="snapshot repo_c0ccaf0 (commit c0ccaf0a9d60abfd653a4ebd907ea8f0bbde24ae)"),
]


def same(a, b, rel_tol=1e-12):
    """Equality of a payload value and what wandb returns: booleans, strings and integers exactly; floats up to a relative 1e-12 (wandb stores floats with fewer digits than Python's round-trip form: on the real
    upload 0.042195780421957804 came back as 0.04219578042195781); a NaN is never equal."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            return a == b
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=0.0)
    return a == b


def verify_runs(payload, api):
    """Read every run of the payload back and list every difference: a run that is missing, a summary value that differs or is missing, the config `files_sha256`, a table that is not in the summary."""
    problems = []
    for r in payload["runs"]:
        path = f"{payload['entity']}/{payload['project']}/{r['id']}"
        try:
            run = api.run(path)
        except Exception as e:  # noqa: BLE001 (any API failure is a finding, not a crash)
            problems.append(f"{r['id']}: the run could not be read ({type(e).__name__})"); continue
        for k, v in r["summary"].items():
            if k not in run.summary:
                problems.append(f"{r['id']}: summary key {k} is missing")
            elif not same(run.summary[k], v):
                problems.append(f"{r['id']}: summary {k} is {run.summary[k]!r}, the payload says {v!r}")
        if run.config.get("files_sha256") != r["config"]["files_sha256"]:
            problems.append(f"{r['id']}: config files_sha256 differs from the payload")
        for k in r["tables"]:
            if f"table/{k}" not in run.summary:
                problems.append(f"{r['id']}: table {k} is not in the run")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--root", default=str(ROOT)); b.add_argument("--out", required=True); b.add_argument("--only", nargs="*"); b.add_argument("--dry-run", action="store_true"); b.add_argument("--allow-dirty", action="store_true"); b.add_argument("--commit")
    u = sub.add_parser("upload"); u.add_argument("--payload", required=True); u.add_argument("--root", required=True); u.add_argument("--only", nargs="*"); u.add_argument("--force-media", action="store_true")
    v = sub.add_parser("verify"); v.add_argument("--payload", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        payload = wu.build_all(a.root, FAMILIES_EXT, a.only, a.commit or wu._git_head(a.root)); print(wu.dry_run_text(payload))
        dirty = wu.dirty_files(a.root, wu.shipped_files(payload))
        if dirty and not a.allow_dirty:
            raise wu.SpecError(f"files that differ from the committed version (commit or pass --allow-dirty): {dirty}")
        if not a.dry_run:
            wu.write_payload(a.root, payload, a.out); print(f"wrote {a.out}/payload.json and files.tar")
        return 0
    import os
    os.environ.setdefault("WANDB_SILENT", "true")
    import wandb
    payload = json.loads(Path(a.payload).read_text())
    if a.cmd == "upload":
        for _ in wu.upload(payload, a.root, wandb, a.only, force_media=a.force_media):
            pass
        return 0
    problems = verify_runs(payload, wandb.Api())
    print("\n".join(problems) if problems else f"verified: {len(payload['runs'])} runs identical to the payload")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
