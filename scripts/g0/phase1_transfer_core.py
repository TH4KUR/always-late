"""Core of the intervention-transfer study (D-088; ledger reports/intervention_transfer/STATUS.md; plan Task 2): the effect of a fixed SAE-feature edit in the cells of the crossed design, and its exact decomposition.

For a context (one sentence) let h_p be the cached layer-12 residual of the prefix state p (FP or Q), F_s the suffix (the layers above and the readout) of the state s, and v the edit (the FP decoder direction times
the FP activation at the edited rows). The effect of the edit in the cell p/s is D[p,s] = center(F_s(h_p - v) - F_s(h_p)) averaged over the edited rows, each cell against its OWN unedited run:
  C = D[Q,FP] - D[FP,FP]   (context contrast at the FP suffix)
  W = D[FP,Q] - D[FP,FP]   (suffix-weight contrast at the FP context)
  I = D[Q,Q] - D[Q,FP] - D[FP,Q] + D[FP,FP]   (interaction)      total = D[Q,Q] - D[FP,FP] = C + W + I exactly, before any aggregation.
A fifth cell, Q/Q-native, subtracts the quantized state's own activations instead of the FP ones. Norms and cosines of C, W, I are not additive shares; they are computed from the 5 x 5 Gram matrix of the five cell
vectors, which is what the run stores (the vectors have the size of the vocabulary). The edit is applied to a float32 residual (a bfloat16 residual would round the edit); the readout is float32.
`suffix` is a callable [B, L, d] -> logits [B, L, V]: the unedited and the edited residual go through it in ONE batch so that both take the same computation (a zero edit is then exactly zero: the sham gate)."""
import torch

CELLS = ("FP/FP", "Q/FP", "FP/Q", "Q/Q", "Q/Q-native")
_BASE = ("FP/FP", "Q/FP", "FP/Q", "Q/Q")
_COEF = {"C": (-1, 1, 0, 0, 0), "W": (-1, 0, 1, 0, 0), "I": (1, -1, -1, 1, 0), "total": (-1, 0, 0, 1, 0), "native_minus_fixed": (0, 0, 0, -1, 1)}


def effect(residual, rows, edit, suffix, targets=None, center=True):
    """One context, one edit, one cell. residual [L, d] float32; rows: the edited row indices (distinct, inside the context); edit [len(rows), d]: the vector subtracted from those rows; targets [L] long (-1: no
    target at that row), optional. -> {"vector": [V] mean over the edited rows of the (centered) logit change, "dloss": mean over the rows with a target of the NLL change or None, "loss_base": the unedited
    mean NLL or None, "n_rows": number of edited rows}."""
    if residual.dtype != torch.float32:
        raise ValueError(f"the residual must be float32 (got {residual.dtype}): a lower precision would round the edit")
    rows = torch.as_tensor(rows, dtype=torch.long); L, d = residual.shape
    if rows.numel() == 0:
        raise ValueError("no edited row")
    if int(rows.min()) < 0 or int(rows.max()) >= L:
        raise ValueError(f"edited rows {rows.tolist()} outside the context of {L} rows")
    if len(torch.unique(rows)) != len(rows):
        raise ValueError("a row is edited twice")
    if tuple(edit.shape) != (len(rows), d):
        raise ValueError(f"the edit has shape {tuple(edit.shape)}, expected {(len(rows), d)}")
    edited = residual.clone(); edited[rows] = residual[rows] - edit.to(torch.float32)
    z = suffix(torch.stack([residual, edited])).float()
    if not bool(torch.isfinite(z).all()):
        raise ValueError("the suffix produced a non-finite value")
    dev = z.device; zb, ze = z[0], z[1]; diff = ze[rows.to(dev)] - zb[rows.to(dev)]
    if center:
        diff = diff - diff.mean(-1, keepdim=True)
    out = {"vector": diff.mean(0), "dloss": None, "loss_base": None, "n_rows": int(len(rows))}
    if targets is not None:
        valid = targets >= 0
        if not bool(valid.any()):
            raise ValueError("no row of the context has a target")
        ar = torch.arange(L, device=dev); tg = targets.clamp(min=0).to(dev); valid = valid.to(dev)
        nb = -torch.log_softmax(zb, -1)[ar, tg]; ne = -torch.log_softmax(ze, -1)[ar, tg]
        out["dloss"] = float((ne - nb)[valid].mean()); out["loss_base"] = float(nb[valid].mean())
    return out


def factorial_effects(residuals, suffixes, rows, fp_edit, native_edit=None, targets=None, center=True):
    """The cells FP/FP, Q/FP, FP/Q, Q/Q (the FP edit in every cell) and, if `native_edit` is given, Q/Q-native. residuals and suffixes: {"FP": ..., "Q": ...}."""
    for name, d in (("residuals", residuals), ("suffixes", suffixes)):
        missing = sorted({"FP", "Q"} - set(d))
        if missing:
            raise ValueError(f"{name} lack the state(s) {missing}")
    if residuals["FP"].shape != residuals["Q"].shape:
        raise ValueError(f"the FP and Q residuals have different shapes {tuple(residuals['FP'].shape)} and {tuple(residuals['Q'].shape)}: the contexts are not the same")
    cells = {f"{p}/{s}": effect(residuals[p], rows, fp_edit, suffixes[s], targets, center) for p, s in (("FP", "FP"), ("Q", "FP"), ("FP", "Q"), ("Q", "Q"))}
    if native_edit is not None:
        cells["Q/Q-native"] = effect(residuals["Q"], rows, native_edit, suffixes["Q"], targets, center)
    return cells


def _combine(cells, names, coefs):
    vec = sum(c * cells[n]["vector"] for n, c in zip(names, coefs) if c)
    losses = [cells[n]["dloss"] for n, c in zip(names, coefs) if c]
    return {"vector": vec, "dloss": None if any(x is None for x in losses) else float(sum(c * cells[n]["dloss"] for n, c in zip(names, coefs) if c))}


def decompose(cells):
    """C, W, I, total (and native_minus_fixed when the native cell is present), each {"vector", "dloss"}; the identity total = C + W + I holds exactly."""
    missing = [n for n in _BASE if n not in cells]
    if missing:
        raise ValueError(f"missing cell(s) {missing}")
    out = {k: _combine(cells, CELLS[:4], _COEF[k][:4]) for k in ("C", "W", "I", "total")}
    if "Q/Q-native" in cells:
        out["native_minus_fixed"] = _combine(cells, CELLS, _COEF["native_minus_fixed"])
    return out


def decompose_contexts(per_context):
    """decompose() for every context, keeping the keys; a context with a missing cell is an error that names it (it is never dropped)."""
    out = {}
    for key, cells in per_context.items():
        try:
            out[key] = decompose(cells)
        except ValueError as e:
            raise ValueError(f"context {key}: {e}") from None
    return out


def gram(cells):
    """The 5 x 5 Gram matrix (float64) of the effect vectors of the cells in the order of CELLS: what the run stores per context."""
    missing = [n for n in CELLS if n not in cells]
    if missing:
        raise ValueError(f"missing cell(s) {missing}")
    V = torch.stack([cells[n]["vector"].double() for n in CELLS])
    return V @ V.T


def decomposition_norms(G):
    """The norms of C, W, I, total and native_minus_fixed from the Gram matrix."""
    out = {}
    for k, c in _COEF.items():
        c = torch.tensor(c, dtype=torch.float64); out[k] = float(torch.sqrt(torch.clamp(c @ G @ c, min=0.0)))
    return out


def transfer_error(G, floor):
    """Normalized per-context transfer error ||D[Q,Q] - D[FP,FP]|| / (||D[FP,FP]|| + floor) and its parts, from the Gram matrix; cosine and norm ratio are None when the FP effect is exactly zero."""
    ff, qq, fq = float(G[0, 0]), float(G[3, 3]), float(G[0, 3])
    err = max(qq - 2 * fq + ff, 0.0) ** 0.5; fp = ff ** 0.5
    if fp + floor <= 0:
        raise ValueError("the FP effect norm plus the floor is zero: the normalized error is undefined")
    return {"error_norm": err, "normalized_error": err / (fp + floor), "fp_norm": fp, "q_norm": qq ** 0.5,
            "norm_ratio": (qq ** 0.5) / fp if fp > 0 else None, "cosine": fq / ((qq ** 0.5) * fp) if fp > 0 and qq > 0 else None}


def pool_contexts(effects, key="vector"):
    """The averaged fingerprint of a feature: the mean of the context vectors weighted by their number of edited rows (as the F5 fingerprint, which averages over all edited positions)."""
    if not effects:
        raise ValueError("no context to pool")
    total = sum(e["n_rows"] for e in effects)
    return sum(e["n_rows"] * e[key] for e in effects) / total


def check_same_state(hybrid_logits, native_logits, atol):
    """Gate: a hybrid forward whose prefix and suffix are in the same state must equal the native full forward."""
    if hybrid_logits.shape != native_logits.shape:
        raise ValueError(f"shapes {tuple(hybrid_logits.shape)} and {tuple(native_logits.shape)} differ")
    worst = float((hybrid_logits - native_logits).abs().max())
    if worst > atol:
        raise ValueError(f"the same-state hybrid differs from the native forward by {worst} (tolerance {atol})")
    return worst
