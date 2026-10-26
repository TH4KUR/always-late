"""Exploratory synthesis of committed Study G tables; no new model runs or tests."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "reports/functional_validation"


def read(name):
    with (BASE / name).open() as handle:
        return list(csv.DictReader(handle))


def main():
    curves_name = "gain_study/results/curves.csv"
    fidelity_name = "results/rescue/fidelity.csv"
    curves = read(curves_name)
    index = {(r["state"], r["language"], float(r["ratio"])): r for r in curves}
    assert len(index) == len(curves), "duplicate curve keys"
    langs = sorted({r["language"] for r in curves})
    down = sorted({float(r["ratio"]) for r in curves if float(r["ratio"]) < 1})
    assert len(langs) == 27 and len(down) == 4
    fidelity = {(r["precision"], r["language"]): r for r in read(fidelity_name)}
    rows, summary = [], {}
    for state in ("INT6", "INT5", "INT4"):
        state_rows = []
        for lang in langs:
            x = np.array([float(index["FP", lang, a]["net"]) for a in down])
            y = np.array([float(index[state, lang, a]["net"]) for a in down])
            cx = np.array([float(index["FP", lang, a]["control_harm"]) for a in down])
            cy = np.array([float(index[state, lang, a]["control_harm"]) for a in down])
            assert x @ x > 0 and cx @ cx > 0 and y @ y > 0
            k = float(x @ y / (x @ x))
            kc = float(cx @ cy / (cx @ cx))
            # A descriptive proportional fit on four deep-attenuation doses.
            # It is neither a local derivative nor a causal sensitivity estimate.
            row = dict(state=state, language=lang, net_scale=k, control_scale=kc,
                       net_relative_residual=float(np.linalg.norm(y-k*x)/np.linalg.norm(y)),
                       zero_net_ratio=float(y[0]/x[0]))
            if (state, lang) in fidelity:
                row["residual_norm_ratio"] = float(fidelity[state, lang]["rms_norm"])/float(fidelity["FP", lang]["rms_norm"])
                row["fvu_ratio"] = float(fidelity[state, lang]["fvu"])/float(fidelity["FP", lang]["fvu"])
            state_rows.append(row)
            rows.append(row)
        summary[state] = {
            "n_languages": len(state_rows),
            "median_net_scale": float(np.median([r["net_scale"] for r in state_rows])),
            "median_control_scale": float(np.median([r["control_scale"] for r in state_rows])),
            "median_relative_residual": float(np.median([r["net_relative_residual"] for r in state_rows])),
            "max_relative_residual": float(max(r["net_relative_residual"] for r in state_rows)),
            "net_control_scale_spearman": float(spearmanr([r["net_scale"] for r in state_rows], [r["control_scale"] for r in state_rows]).statistic),
        }
    inputs = {n: hashlib.sha256((BASE/n).read_bytes()).hexdigest()
              for n in (curves_name, fidelity_name)}
    result = {"status": "post hoc descriptive, no inferential tests or new GPU data",
              "doses": down, "definition": "k=(FP_net dot Q_net)/(FP_net dot FP_net)",
              "inputs_sha256": inputs, "states": summary,
              "limits": ["Four doses are far from identity; not curvature estimates.",
                         "Similar medians do not imply per-language agreement.",
                         "No raw-data uncertainty propagated into these fitted summaries.",
                         "Means, medians, and mean retention ratios are different estimands."]}
    with (HERE/"existing_curve_patterns.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (HERE/"existing_curve_patterns.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

