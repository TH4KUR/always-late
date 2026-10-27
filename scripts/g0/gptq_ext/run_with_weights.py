"""Run an existing experiment entry point with another weight method WITHOUT editing it (plan Task 6): several entry points are hash-pinned in frozen evaluation freezes (`evaluation_freeze_lm.json` pins
`phase1_func_run.py` and `phase1_lib.py`), and the old studies' code must stay byte for byte as it was. The entry points build their `env` with `their_code.build_namespace` and quantize through
`env["quantize_rtn_per_output_channel_tl"](model, bits=...)`; this wrapper patches `build_namespace` so that, for `--weights gptq` or `--weights rtn-matched`, that callable is a `states.LazyQuantizer`, then calls the
entry point with its own arguments unchanged. With `--weights rtn` (the default) nothing is patched and the entry point runs exactly as before.

  python gptq_ext/run_with_weights.py --entry phase1_transfer_run:main --weights gptq --gptq-freeze F --calibration-freeze C --calibration-text D -- <the entry point's own arguments, including --out>"""
import argparse
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gptq_ext import states  # noqa: E402


def split_argv(argv):
    if "--" not in argv:
        raise SystemExit("usage: run_with_weights.py --entry module:function [weights options] -- <entry point arguments>")
    i = argv.index("--")
    return argv[:i], argv[i + 1:]


def out_dir_of(entry_argv):
    p = argparse.ArgumentParser(add_help=False); p.add_argument("--out")
    return p.parse_known_args(entry_argv)[0].out


def bits_hint(entry_argv):
    """The integer bit widths an entry point will use, read from its own arguments (`--bits`, `--precisions`, `--states`; labels such as FP, FP_rebatched are ignored): [] when it names none (then the builds stay lazy)."""
    p = argparse.ArgumentParser(add_help=False)
    for opt in ("--bits", "--precisions", "--states"):
        p.add_argument(opt, nargs="*", default=[])
    ns = p.parse_known_args(entry_argv)[0]
    return states.bits_in(ns.bits + ns.precisions + ns.states)


def patch_loader(args, out_dir, log, hint=()):
    """Make `their_code.build_namespace` leave a LazyQuantizer in `env`, and make `env['clone_state_to_cpu']` (which every entry point calls right after loading the model, while it is still full precision) build
    the artifacts named by `hint` first. -> the original function (to restore it)."""
    import their_code
    original = their_code.build_namespace

    def patched(extracted, env):
        env = original(extracted, env)
        lazy = states.LazyQuantizer(env, args, log, out_dir)
        env["quantize_rtn_per_output_channel_tl"] = lazy
        if hint and "clone_state_to_cpu" in env:
            clone = env["clone_state_to_cpu"]

            def clone_and_prebuild(model):
                state = clone(model)
                lazy.prebuild(model, hint, state)
                return state
            env["clone_state_to_cpu"] = clone_and_prebuild
        return env
    their_code.build_namespace = patched
    return original


def main(argv=None):
    mine, theirs = split_argv(list(sys.argv[1:] if argv is None else argv))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--entry", required=True, help="module:function, e.g. phase1_func_run:main"); states.add_weights_args(ap)
    a = ap.parse_args(mine)
    mod, func = a.entry.split(":")
    t0 = time.time(); log = lambda *m: print(f"[weights {time.time() - t0:7.0f}s]", *m, flush=True)
    out = out_dir_of(theirs)
    if a.weights != "rtn":
        if out is None:
            raise SystemExit("the entry point has no --out: the weights manifest needs an output directory")
        if not (a.calibration_freeze and a.calibration_text):
            raise SystemExit("--calibration-freeze and --calibration-text are required for --weights " + a.weights)
        hint = bits_hint(theirs)
        patch_loader(a, out, log, hint)
        log("weights:", a.weights, "entry:", a.entry, "out:", out, "artifact bits built before the job starts:", [b for b in hint if b in states.ARTIFACT_BITS] or "none (lazy)")
    return getattr(importlib.import_module(mod), func)(theirs)


if __name__ == "__main__":
    main()
