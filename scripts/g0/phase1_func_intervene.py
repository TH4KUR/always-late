"""Feature-group interventions at one residual-stream layer through a frozen SAE (functional-validation study, D-039/D-040; design section 5). Pure torch, tested on CPU.

Every edit changes ONLY the decoder span of the chosen feature group and keeps the SAE reconstruction error of the recipient: with a = encode(x) (the recipient's own activations),
x' = x + sum_{f in G} (a'_f - a_f) d_f. `a'` is zero (ablation), alpha * a (scaling), a random pool row (resampling), fixed per-feature values (mean ablation), the donor's activations (patching, e.g. from the full-precision
model at the same tokens) or a itself (sham: the same arithmetic with no intended change). Edits apply to the VALID positions only (content tokens; never BOS, never padding)."""
import torch

KINDS = ("ablate", "scale", "resample", "mean", "donor", "donor_error", "donor_resid", "sham")


def encode_valid(resid, valid, encode):
    """SAE activations [N, F] (float32) of the valid rows of resid [B, L, d] (row-major order of `valid`)."""
    return encode(resid[valid]).float()


def resample_values(pool, n, generator):
    """n rows drawn with replacement from pool [P, K] (whole group vectors of single positions, so the co-activation of the group is kept)."""
    rows = torch.randint(0, pool.shape[0], (n,), generator=generator)
    return pool[rows.to(pool.device)]


def _check(resid, valid, a, W_dec, idx):
    idx = torch.as_tensor(idx, dtype=torch.long)
    if idx.ndim != 1 or idx.numel() == 0:
        raise ValueError("idx must be a non-empty 1-D index tensor")
    if int(idx.min()) < 0 or int(idx.max()) >= W_dec.shape[0]:
        raise ValueError("feature index out of range")
    if idx.unique().numel() != idx.numel():
        raise ValueError("duplicate feature indices in the group")
    n = int(valid.sum())
    if a.shape[0] != n or a.shape[1] != W_dec.shape[0]:
        raise ValueError(f"activations {tuple(a.shape)} do not match {n} valid rows and {W_dec.shape[0]} features")
    return idx


def apply_edit(resid, valid, a, W_dec, idx, kind, alpha=None, pool=None, generator=None, a_donor=None, mean_vals=None, x_donor=None):
    """-> (resid', stats). resid [B, L, d] (any float dtype, returned in the same dtype, never modified in place), valid [B, L] bool, a [N, F] = encode_valid(...), W_dec [F, d], idx group.
    stats: sum_norm_ratio (sum over edited rows of ||x' - x|| / ||x||), sum_active (group features active before the edit, summed over rows), rows (N): sums, so that batches aggregate exactly.
    Donor kinds (the donor is the full-precision model run on the SAME tokens, so rows align exactly): `donor` injects the donor's activations of the group; `donor_resid` replaces the whole residual by
    the donor's (x_donor [N, d]); `donor_error` keeps the recipient's SAE reconstruction and takes the donor's error term, x' = x_donor - (a_donor - a) @ W_dec (use with the group of ALL features:
    `donor` with all features plus `donor_error` add up to the donor residual, x_all + x_err - x = x_donor)."""
    if kind not in KINDS:
        raise ValueError(f"unknown intervention {kind!r}; choose from {KINDS}")
    idx = _check(resid, valid, a, W_dec, idx)
    x = resid[valid]
    x32 = x.float()
    Wg = W_dec[idx].float().to(x32.device)
    ag = a[:, idx.to(a.device)].float()
    if kind == "sham":
        d = ag @ Wg
        x_new = (x32 + d) - d
    else:
        if kind == "ablate":
            a_new = torch.zeros_like(ag)
        elif kind == "scale":
            if alpha is None:
                raise ValueError("scale needs alpha")
            a_new = ag * float(alpha)
        elif kind == "resample":
            if pool is None or generator is None:
                raise ValueError("resample needs a pool and a seeded generator")
            if pool.shape[1] != idx.numel():
                raise ValueError("pool columns must equal the group size")
            a_new = resample_values(pool.float(), ag.shape[0], generator).to(ag.device)
        elif kind == "mean":
            if mean_vals is None or mean_vals.shape != (idx.numel(),):
                raise ValueError("mean needs one value per group feature")
            a_new = mean_vals.float().to(ag.device)[None, :].expand_as(ag)
        elif kind in ("donor", "donor_error"):
            if a_donor is None:
                raise ValueError("donor needs the donor activations")
            if a_donor.shape != a.shape:
                raise ValueError("donor activations must have the shape of the recipient's")
            a_new = a_donor[:, idx.to(a_donor.device)].float().to(ag.device)
        if kind in ("donor_error", "donor_resid"):
            if x_donor is None or tuple(x_donor.shape) != tuple(x.shape):
                raise ValueError("this donor kind needs the donor residual rows with the shape of the recipient's valid rows")
            xd = x_donor.float().to(x32.device)
        if kind == "donor_resid":
            x_new = xd                                                  # whole-residual patch: the donor's layer-12 residual replaces the recipient's
        elif kind == "donor_error":
            x_new = xd - (a_new - ag) @ Wg                              # recipient's reconstruction plus the donor's error term (b_dec cancels)
        else:
            x_new = x32 + (a_new - ag) @ Wg
    out = resid.clone()
    out[valid] = x_new.to(resid.dtype)
    ratio = (x_new - x32).norm(dim=-1) / x32.norm(dim=-1).clamp_min(1e-12)
    return out, {"sum_norm_ratio": float(ratio.sum()), "sum_active": float((ag > 0).sum()), "rows": int(ag.shape[0])}
