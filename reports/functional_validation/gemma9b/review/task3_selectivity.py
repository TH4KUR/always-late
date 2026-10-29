"""9B review task 3 (D-125): what the frozen P3b ("median selectivity ratio above 1.3", at least 20 of 27 languages) can and cannot show.

The freeze does not define the per-language "selectivity ratio". The only selectivity ratio in the code is S(f) = max_l r_l(f) / mean_l r_l(f) (`phase1_lib.selectivity_S`), a per-feature quantity; a language's median is taken here over the features whose
peak language it is, in two ways: the selective features (the ones the study uses; their definition already requires S above the shuffled-null cutoff) and all supported features.

Input: the raw `eq/selectivity.pt` of the imported run (sha 622af41f..., fetched outside git into `gemma9b_raw/`, see REVIEW_STATUS.md). Needs torch (`~/.venvs/anlp-torch/bin/python -I`). Output: printed, and saved as `task3_selectivity.txt` next to this file.

  ~/.venvs/anlp-torch/bin/python -I task3_selectivity.py /home/karma/Documents/ANLP/project/gemma9b_raw/eq/selectivity.pt
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def main(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    S, peak, sel, sup, langs = d["S"], d["peak"], d["selective"].bool(), d["supported"].bool(), d["languages"]
    lines = [f"selective features {int(sel.sum())} of {len(S)}, supported {int(sup.sum())}, shuffled-null cutoff {d['null_cutoff']:.4f}, smallest S among the selective features {float(S[sel].min()):.4f}"]
    rows = []
    for i, lang in enumerate(langs):
        a, b = sel & (peak == i), sup & (peak == i)
        rows.append((lang, int(a.sum()), float(S[a].median()) if a.any() else float("nan"), int(b.sum()), float(S[b].median()) if b.any() else float("nan")))
    lines.append(f"languages with at least one selective peak feature: {sum(r[1] > 0 for r in rows)} of {len(rows)} (fewest: {min((r[1], r[0]) for r in rows)})")
    lines.append(f"median S over a language's selective peak features above 1.3 in {sum(r[2] > 1.3 for r in rows)} of {len(rows)} languages (smallest {min(r[2] for r in rows):.3f}, largest {max(r[2] for r in rows):.3f})")
    lines.append(f"median S over a language's supported peak features above 1.3 in {sum(r[4] > 1.3 for r in rows)} of {len(rows)} languages (smallest {min(r[4] for r in rows):.3f}, largest {max(r[4] for r in rows):.3f})")
    lines.append("language, selective peak features, median S (selective), supported peak features, median S (supported)")
    lines += [f"{r[0]}, {r[1]}, {r[2]:.3f}, {r[3]}, {r[4]:.3f}" for r in rows]
    text = "\n".join(lines)
    print(text)
    (HERE / "task3_selectivity.txt").write_text(text + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
