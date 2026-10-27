"""The FP bridge of the transfer stage (freeze gate G_fp_bridge). A transfer job's chunk files hold, per item, quantities that depend on the quantized weights and quantities that do not. The latter come from the full-precision
model on identical inputs and must therefore equal the old job's EXACTLY (the FP path is unchanged and was bit-for-bit reproducible, A-017): the full-precision edit and residual norms and amplitudes, the numbers of rows and
positions, the direction norm (feature level), the full-precision gradient scalars and the FP-prefix FP-suffix cell 'FP/FP' with its loss quantities (index 0 of `dloss` and `loss_base`; `cells` holds the labels). The chunk keys (a hash that binds the
freeze, the core and the state) must be equal too. A difference is reported per field and never merged silently.

  python3 scripts/g0/gptq_ext/bridge_transfer.py --old <old job dir> --new <new job dir> --prefix transfer_INT4     (group level: --prefix group_INT4)"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCALAR_FIELDS = ("edit_norm_fp", "resid_norm_fp", "amp_fp_mean", "g_pred_fp", "g_norm_fp", "n_rows", "n_target_rows", "n_positions", "direction_norm")
LIST_FIELDS = ("amp_fp",)
CELL0_FIELDS = ("dloss", "loss_base")
FIELDS = SCALAR_FIELDS + LIST_FIELDS + tuple(f"{f}[0]" for f in CELL0_FIELDS) + ("cells",)


def _equal(a, b):
    return bool(np.array_equal(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), equal_nan=True))


def bridge_dirs(old_dir, new_dir, prefix):
    old, new = sorted(Path(old_dir).glob(f"{prefix}_c*.pt")), sorted(Path(new_dir).glob(f"{prefix}_c*.pt"))
    report = {"ok": False, "chunks": 0, "items": 0, "fields_differing": {f: 0 for f in FIELDS}, "problems": []}
    if not old or not new:
        report["problems"].append(f"no chunk files ({len(old)} old, {len(new)} new)"); return report
    if [p.name for p in old] != [p.name for p in new]:
        report["problems"].append("the two directories hold different chunk files"); return report
    for po, pn in zip(old, new):
        a, b = torch.load(po, weights_only=False), torch.load(pn, weights_only=False)
        report["chunks"] += 1
        if a["key"] != b["key"]:
            report["problems"].append(f"{po.name}: chunk keys differ")
        ia, ib = a["payload"]["items"], b["payload"]["items"]
        if set(ia) != set(ib):
            report["problems"].append(f"{po.name}: item sets differ ({len(ia)} old, {len(ib)} new)"); continue
        for k in ia:
            report["items"] += 1; x, y = ia[k], ib[k]
            for f in SCALAR_FIELDS + LIST_FIELDS:
                if f in x or f in y:
                    if f not in x or f not in y or not _equal(x[f], y[f]):
                        report["fields_differing"][f] += 1
            for f in CELL0_FIELDS:
                if not _equal(x[f][0], y[f][0]):
                    report["fields_differing"][f"{f}[0]"] += 1
            if list(x["cells"]) != list(y["cells"]):                          # the cell labels ('FP/FP', 'Q/FP', ...): the order every index above refers to
                report["fields_differing"]["cells"] += 1
    report["ok"] = not report["problems"] and not any(report["fields_differing"].values())
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old", required=True); ap.add_argument("--new", required=True); ap.add_argument("--prefix", required=True)
    a = ap.parse_args(argv)
    r = bridge_dirs(a.old, a.new, a.prefix)
    print(json.dumps(r, indent=1)); return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
