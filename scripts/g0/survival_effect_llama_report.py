"""Report of the Llama-3.2-1B F5 check against Gemma (Stage 5 of the survival-and-intervention plan; addendum reports/survival_effect_rescue/llama/llama_f5_addendum_2026-10-28.md), generated from the two result
JSON files (`survival_effect_llama.py` for Llama, `survival_effect.py` for Gemma). Every number is read from the JSON; the labels are the ones defined in the addendum.

  python3 scripts/g0/survival_effect_llama_report.py --llama <llama json> --gemma reports/survival_effect_rescue/results/survival_effect_results.json --out <md>"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import survival_effect_llama as sl  # noqa: E402
from survival_effect_report import ci, f3, pci  # noqa: E402

STATES = ("INT6", "INT4")


def render(ll, gm):
    g = ll["gate"]; L = []
    L += ["# Llama-3.2-1B F5 check, side by side with Gemma-2-2B (Stage 5 of the survival-and-intervention plan)", "",
          "**Status: post hoc and exploratory.** Frozen before any Llama outcome: [`llama_f5_addendum_2026-10-28.md`](llama_f5_addendum_2026-10-28.md); Gemma analysis it repeats: [`../stage4_gemma_report_2026-10-28.md`](../stage4_gemma_report_2026-10-28.md) (D-113). All numbers are read from the result files by `scripts/g0/survival_effect_llama_report.py`. Model, SAE family, SAE training data, tokenizer, size and layer depth all differ from Gemma, so a difference cannot be attributed to any one of them; no feature identities are compared across the two SAEs; with two systems no population claim is made and overlapping intervals are not equivalence.", "",
          "## 1. Instrument gate (Llama's own floor)", "",
          f"Median W1 between full precision and the INT16 repeat: {f3(g['median_cos16'])} (the F5 design rule needs at least {sl.ADEQUATE_MEDIAN}); this system's floor (5th percentile of the INT16 cosine): {f3(g['floor'])}; reliable features: {g['n_reliable']} of {g['n_features']} (at least {sl.MIN_RELIABLE} needed). Gate: **{'passed' if g['ok'] else 'FAILED'}**." + ("" if g["ok"] else " Reasons: " + "; ".join(g["why"]) + "."), ""]
    if not g["ok"]:
        L += ["**Result: instrument inadequate; no relation is reported** (addendum gate 5). The Gemma result stands alone.", ""]
        return "\n".join(L) + "\n"
    c0 = ll["population_counts"]
    L += [f"## 2. Result", "", f"**Pre-set rule (identical to Gemma: improvement of model B over model A of at least 10 % with the lower end of the language-cluster interval above 0), Llama: {ll['overall_statement']}** ({ll['cells_met_primary']} of 2 cells).", "",
          f"Populations (fixed from the INT16 reference only): primary {c0['primary']} features (INT16 cosine at least {ll['tau']:.3f}), noisy-effect stratum {c0['noisy_stratum']}, S-all {c0['S_all']}, S-loss {c0['S_loss']}.", "",
          "## 3. Side by side (legacy RTN on both systems, same states, same rule)", "",
          "| system | state | reliable n | floor | Spearman R1-W1 | partial Spearman given baselines | association | B over A (interval, points) | value |", "|---|---|---|---|---|---|---|---|---|"]
    for s in STATES:
        gc = gm["cells"][f"legacy RTN|{s}|primary"]; lc = ll["cells"][f"{s}|primary"]
        for name, c, n, fl, lab_a, lab_v in (("Gemma-2-2B", gc, gc["n"], gm["tau"], "repeats" if gc["partial_spearman_given_A"]["lo"] > 0 else "cannot be resolved", sl.value_label(gc["improvement"]["B_over_A"]["improvement"] >= 0.1 and gc["improvement"]["B_over_A"]["lo"] > 0)),
                                             ("Llama-3.2-1B", lc, lc["n"], ll["tau"], sl.association_label(lc["partial_spearman_given_A"]["lo"], lc["partial_spearman_given_A"]["hi"]), sl.value_label(lc["cell_met"]))):
            L.append(f"| {name} | {s} | {n} | {fl:.3f} | {f3(c['spearman']['rho'])} {ci(c['spearman']['lo'], c['spearman']['hi'])} | {f3(c['partial_spearman_given_A']['rho'])} {ci(c['partial_spearman_given_A']['lo'], c['partial_spearman_given_A']['hi'])} | {lab_a} | {pci(c['improvement']['B_over_A'])} | {'**met**' if (name == 'Llama-3.2-1B' and c['cell_met']) else ('met' if lab_v == 'repeats' else 'not met')} |")
    L += ["", "Labels (defined before the run): the association **repeats** if the partial Spearman interval lies above 0, **cannot be resolved** if it contains 0, **reverses** if it lies below 0; predictive value repeats only if the cell meets the 10 % rule (no Gemma cell did). Each system's estimates are relative to its own noise floor.", "",
          "## 4. Llama cells in detail", "", "| state | n | MAE null / A / B | C over B | A over null |", "|---|---|---|---|---|"]
    for s in STATES:
        c = ll["cells"][f"{s}|primary"]; m = c["mae"]
        L.append(f"| {s} | {c['n']} | {m['null']:.4f} / {m['A']:.4f} / {m['B']:.4f} | {pci(c['improvement']['C_over_B'])} | {pci(c['improvement']['A_over_null'])} |")
    L += ["", "## 5. Sensitivity populations (cannot change the statement)", "", "| population | state | n | Spearman R1-W1 | partial Spearman | B over A |", "|---|---|---|---|---|---|"]
    for p in ("S_all", "S_loss", "noisy_stratum"):
        for s in STATES:
            c = ll["cells"][f"{s}|{p}"]
            L.append(f"| {p} | {s} | {c['n']} | {f3(c['spearman']['rho'])} {ci(c['spearman']['lo'], c['spearman']['hi'])} | {f3(c['partial_spearman_given_A']['rho'])} {ci(c['partial_spearman_given_A']['lo'], c['partial_spearman_given_A']['hi'])} | {pci(c['improvement']['B_over_A'])} |")
    L += ["", "## 6. Limits", "",
          "- Two fixed systems: the result concerns these two model-plus-SAE setups only; Llama's sample (643 features, two roles) is smaller and differently composed than Gemma's (1,487 features, more roles).",
          "- R1 and W1 come from the same perturbed model; held out by language, not by document; the evaluation sentences were examined in earlier studies (post hoc); the same shared RTN method on both systems, no GPTQ on Llama.",
          "- Llama's earlier group-level check was not consistent with Gemma (D-090); nothing here explains why the systems differ."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--llama", required=True); ap.add_argument("--gemma", required=True); ap.add_argument("--out", required=True); a = ap.parse_args(argv)
    Path(a.out).write_text(render(json.loads(Path(a.llama).read_text()), json.loads(Path(a.gemma).read_text()))); print("wrote", a.out)


if __name__ == "__main__":
    main()
