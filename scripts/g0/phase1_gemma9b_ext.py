"""Entry point of the three-arm Gemma-2-9B confirmation (plan docs/superpowers/plans/2026-10-28-gemma9b-three-arm-confirmation.md, Task 4). The pinned adapter `phase1_gemma9b_run.py` is not edited; this module puts its
9B setting (`gemma9b_setting`: model, Gemma Scope SAE, layer 20) around two more frozen entry points and routes the third task to the adapter itself:

    lm    the adapter's frozen Study G run (tier 1), arguments passed through unchanged
    p1    `phase1_run.main`: Phase 1 survival and loss, mask regime (tier 2)
    gate  `gptq_ext.gate_run.main`: the build-and-gate job (artifact hashes, gates)

Through `gptq_ext/run_with_weights.py --entry phase1_gemma9b_ext:main --weights gptq|rtn-matched ... -- <task> <arguments>` the entry points obtain GPTQ or matched-RTN weight sets; with the default legacy rounding nothing
changes. `p1` and `gate` run on one device (the Jarvis card holds the model); the multi-device loader of the adapter is for `lm` only.

    python phase1_gemma9b_ext.py p1 --corpus flores --regime mask --bits 7 6 4 16 12 0 --out <dir> --flores-root <dir> --commit <c>"""
import argparse

import phase1_gemma9b_run as g9
import phase1_run as pr


def main(argv=None):
    ap = argparse.ArgumentParser(description="Gemma-2-9B three-arm confirmation: frozen entry points under the 9B setting (remaining arguments go to the entry point).", allow_abbrev=False)
    ap.add_argument("task", choices=["lm", "p1", "gate"])
    ap.add_argument("--n-devices", type=int, default=g9.default_n_devices(), help=f"GPUs to split the model over (lm only; default ${g9.N_DEVICES_ENV} or 1)")
    a, rest = ap.parse_known_args(argv)
    if a.task == "lm":
        return g9.main(["lm", "--n-devices", str(a.n_devices), *rest])
    if a.n_devices != 1:
        ap.error(f"{a.task} runs on one device (got --n-devices {a.n_devices})")
    with g9.gemma9b_setting(1):
        if a.task == "p1":
            return pr.main(rest)
        from gptq_ext import gate_run
        return gate_run.main(rest)


if __name__ == "__main__":
    main()
