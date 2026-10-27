"""Cross-mode core of the paired arithmetic-precision audit (protocol revision_R5_precision_audit.json; implementation plan docs/superpowers/plans/2026-10-27-precision-audit-task2-implementation.md, step 4a; POST HOC).

An observer on `phase1_transfer_core.effect_many` rebuilds, from the raw logits, the effect vector of every item and cell (the same centering and mean over the edited rows as `effect_many`) so that the cells of
the two execution modes (bf16 suffix, fp32 suffix) of the SAME item can be compared; the frozen runner is not changed. The comparison is Gram algebra: the ten cell vectors of an item (five cells in each mode) give
a 10 x 10 Gram matrix, and every norm, cosine and ratio of the protocol is a quadratic form of linear combinations of the ten vectors. Nothing is subtracted in quadrature and no vector is rescaled.
"""
import numpy as np
import torch

import phase1_transfer_core as tc

CELLS = tuple(tc.CELLS)
MODES = ("bf16", "fp32")
NAMES = tuple(f"{m}:{c}" for m in MODES for c in CELLS)
_SIMPLE = ("FP/FP", "Q/FP", "FP/Q")
_COMBINED = "Q/Q+Q/Q-native"
ELIGIBILITY_MULTIPLE = 5


class VectorObserver:
    """observer(tag, z, rows_list, targets) for `tc.effect_many`: stores the effect vector (float32, host) of every (item key, cell) in `vectors`. The tag is (cell, item keys); the combined quantized call
    (`Q/Q+Q/Q-native`) holds the K fixed-edit edits first and the K native edits after them. A repeated (item, cell), an unknown tag or a wrong number of edits is a ValueError."""
    def __init__(self, center=True):
        self.center = center; self.vectors = {}

    def __call__(self, tag, z, rows_list, targets):
        cell, keys = tag; K = len(keys)
        if cell in _SIMPLE:
            cells = [cell] * K
        elif cell == _COMBINED:
            cells = ["Q/Q"] * K + ["Q/Q-native"] * K
        else:
            raise ValueError(f"unknown tag {cell!r}: the audit has no gradient cells")
        if len(rows_list) != len(cells):
            raise ValueError(f"{len(rows_list)} edits for {len(keys)} items in the call {cell!r}: expected {len(cells)} edits")
        zb = z[0]
        for k, (rows, c) in enumerate(zip(rows_list, cells)):
            r = torch.as_tensor(rows, dtype=torch.long).to(z.device); diff = z[k + 1][r] - zb[r]
            if self.center:
                diff = diff - diff.mean(-1, keepdim=True)
            key = keys[k % K]
            if (key, c) in self.vectors:
                raise ValueError(f"the cell {c} of item {key} occurs twice")
            self.vectors[(key, c)] = diff.mean(0).detach().float().cpu()


def cross_gram(vec_by_mode, key):
    """The 10 x 10 Gram matrix (float64, order NAMES) of the ten cell vectors of the item `key`; vec_by_mode: {"bf16": {(key, cell): vector}, "fp32": {...}}. A missing cell is a KeyError naming it."""
    rows = []
    for m in MODES:
        for c in CELLS:
            if (key, c) not in vec_by_mode[m]:
                raise KeyError(f"no vector of the cell {c} of item {key} in the mode {m}")
            rows.append(vec_by_mode[m][(key, c)].double())
    V = torch.stack(rows); return (V @ V.T).numpy()


def comb(terms):
    """The coefficient vector over the ten cell vectors of a linear combination: terms = [((mode, cell), weight), ...] (repeated cells add)."""
    c = np.zeros(len(NAMES))
    for (mode, cell), w in terms:
        c[MODES.index(mode) * len(CELLS) + CELLS.index(cell)] += w
    return c


def combo_norm(G, c):
    return float(np.sqrt(max(float(c @ G @ c), 0.0)))


def combo_dot(G, c1, c2):
    return float(c1 @ G @ c2)


def combo_cosine(G, c1, c2):
    """The cosine of two combinations, None if either has norm zero."""
    n1, n2 = combo_norm(G, c1), combo_norm(G, c2)
    return None if n1 == 0.0 or n2 == 0.0 else combo_dot(G, c1, c2) / (n1 * n2)


def _ratio(a, b):
    return None if b == 0.0 else a / b


def item_metrics(G):
    """The per-item quantities of the protocol from the cross-mode Gram: per mode m the FP effect norm of the cell FP/FP (`fp_norm_m`), the norm of Q/Q, the transfer error err_m = |D_m[Q/Q] - D_m[FP/FP]| and the
    cosine cos_m between D_m[Q/Q] and D_m[FP/FP]; across modes the arithmetic change of the FP/FP cell (`arith_ff` = |D_fp32[FF] - D_bf16[FF]|) and of Q/Q (`arith_qq`), their ratios to the transfer error of each
    mode and to the FP effect norm, and the cosine between the arithmetic change of Q/Q and the bf16 transfer error. A ratio or a cosine with a zero denominator is None."""
    ff = {m: comb([((m, "FP/FP"), 1.0)]) for m in MODES}; qq = {m: comb([((m, "Q/Q"), 1.0)]) for m in MODES}; err = {m: qq[m] - ff[m] for m in MODES}; out = {}
    for m in MODES:
        out.update({f"fp_norm_{m}": combo_norm(G, ff[m]), f"qq_norm_{m}": combo_norm(G, qq[m]), f"err_{m}": combo_norm(G, err[m]), f"cos_{m}": combo_cosine(G, qq[m], ff[m])})
    a_ff, a_qq = ff["fp32"] - ff["bf16"], qq["fp32"] - qq["bf16"]; out["arith_ff"] = combo_norm(G, a_ff); out["arith_qq"] = combo_norm(G, a_qq)
    out["arith_qq_over_err_bf16"] = _ratio(out["arith_qq"], out["err_bf16"]); out["arith_qq_over_err_fp32"] = _ratio(out["arith_qq"], out["err_fp32"]); out["arith_ff_over_fp_norm_bf16"] = _ratio(out["arith_ff"], out["fp_norm_bf16"])
    out["cos_arith_vs_err"] = combo_cosine(G, a_qq, err["bf16"]); return out


def mode_floors(rows):
    """The mode-specific numerical floor per direction kind and mode: the 95th percentile of the INT16 transfer error over the items of the kind. rows: dicts with `kind`, `err_bf16`, `err_fp32` (the INT16 items)."""
    out = {}
    for kind in sorted({r["kind"] for r in rows}):
        for m in MODES:
            out[(kind, m)] = float(np.percentile([r[f"err_{m}"] for r in rows if r["kind"] == kind], 95))
    return out


def with_floors(row, old_floors, mode_floors_):
    """The row with, per mode, the normalized error `nerr_*` = err / (fp_norm + floor), the error in floors `eof_*` = err / floor and the eligibility (FP effect norm above five floors) for the OLD fixed floors of the
    kind (`*_old_*`) and the mode-specific floors (`*_mode_*`)."""
    out = dict(row)
    for m in MODES:
        for tag, floor in (("old", old_floors[row["kind"]]), ("mode", mode_floors_[(row["kind"], m)])):
            out[f"nerr_{tag}_{m}"] = row[f"err_{m}"] / (row[f"fp_norm_{m}"] + floor); out[f"eof_{tag}_{m}"] = row[f"err_{m}"] / floor; out[f"eligible_{tag}_{m}"] = bool(row[f"fp_norm_{m}"] > ELIGIBILITY_MULTIPLE * floor)
    return out


def batching_difference(vectors_a, vectors_b, keys):
    """Control C1: per item and cell the norm of the difference between the effect vectors of two executions of the same state with different batch composition. {key: {cell: norm}}; a cell missing from the
    second execution is a KeyError."""
    out = {}
    for key in keys:
        out[key] = {}
        for c in CELLS:
            if (key, c) in vectors_a:
                if (key, c) not in vectors_b:
                    raise KeyError(f"the cell {c} of item {key} is missing from the second execution")
                out[key][c] = float((vectors_a[(key, c)].double() - vectors_b[(key, c)].double()).norm())
    return out
