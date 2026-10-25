"""Study G: print the exact `sbatch` command of a calibration or main job (so that a firing never types the ratios by hand). The ratios of a main job come from the decision file
`reports/functional_validation/gain_study/calibration/ratios.json` (written by `phase1_gain_ratios.py`); those of a calibration job are the 18 ratios of `phase1_gain_ratios.calibration_ratios()`.
Nothing is submitted: the command is for `ssh karma@10.4.24.30` (see the plan's operational commands).

    python3 scripts/g0/phase1_gain_commands.py main --state INT4 --ratios-json <ratios.json> --repo repo_7fba511
    python3 scripts/g0/phase1_gain_commands.py calibration --state FP --repo repo_7fba511
    python3 scripts/g0/phase1_gain_commands.py extc --state INT4 --repo repo_<sha>      (extension C, spec section 11)"""
import argparse
import json
from pathlib import Path

import phase1_gain_ratios as rt

STATES = {"FP": "FP", "INT8": "8", "INT7": "7", "INT6": "6", "INT5": "5", "INT4": "4"}
BASE = "/home2/karma/duan-verify/phase1"
EXCLUDED_NODES = ("gnode068",)                                       # labelled 2080ti but holds an RTX 4070 (D-079): a job there is not comparable with the others


def _state(s):
    if s not in STATES:
        raise ValueError(f"state {s!r} is not one of {sorted(STATES)}")
    return STATES[s]


def _sbatch(name, hours, repo, args):
    r = f"{BASE}/{repo}"
    return f"sbatch --exclude={','.join(EXCLUDED_NODES)} -t {hours} -J {name} --export=ALL,REPO={r} {r}/scripts/g0/run_gain.sbatch {name} {args}"


def main_command(state, ratios_json, repo, hours="04:00:00", n_controls=3):
    """The command of the main job of one state (all 27 languages, 300 evaluation sentences, three control draws)."""
    short = _state(state)
    dec = json.loads(Path(ratios_json).read_text())
    if len(dec.get("down", [])) != 3 or len(dec.get("up", [])) != 3 or not dec.get("fine_up"):
        raise ValueError("the ratios file must hold three down ratios, three up ratios and the fine up ratios")
    ratios = " ".join(repr(float(r)) for r in list(dec["down"]) + list(dec["up"]) + list(dec["fine_up"]))
    return _sbatch(f"gain-{short}", hours, repo, f"--ratios {ratios} --n-controls {n_controls} --precisions {short} --split eval --n-eval 300 --panel")


def calibration_command(state, repo, hours="03:00:00", max_sentences=100, n_controls=2):
    """The command of the calibration job of FP or INT4 (18 ratios, the first `max_sentences` pilot sentences, two control draws)."""
    if state not in ("FP", "INT4"):
        raise ValueError("calibration runs at FP and INT4")
    short = _state(state)
    ratios = " ".join(repr(r) for r in rt.calibration_ratios())
    return _sbatch(f"gain-cal-{state}", hours, repo, f"--ratios {ratios} --n-controls {n_controls} --precisions {short} --split pilot --max-sentences {max_sentences} --panel")


EXTC_RATIOS = (1.1, 1.2, 1.4, 1.7)


def extc_command(state, repo, hours="05:00:00", n_controls=5):
    """The command of an extension C job (spec section 11): the repair near ratio 1 on the 562 reserve sentences (`--n-pilot 450 --n-eval 562 --split eval` leaves exactly the sentences that no earlier run used),
    all five control draws, only the scaled sets (no zero-ablation). A larger sample than the main study in every respect."""
    short = _state(state)
    ratios = " ".join(repr(r) for r in EXTC_RATIOS)
    return _sbatch(f"gainc-{short}", hours, repo, f"--ratios {ratios} --n-controls {n_controls} --no-zero --precisions {short} --n-pilot 450 --n-eval 562 --split eval --panel")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Print the sbatch command of a Study G job.")
    sub = ap.add_subparsers(dest="kind", required=True)
    m = sub.add_parser("main"); m.add_argument("--state", required=True); m.add_argument("--ratios-json", required=True); m.add_argument("--repo", required=True); m.add_argument("--hours", default="04:00:00")
    c = sub.add_parser("calibration"); c.add_argument("--state", required=True); c.add_argument("--repo", required=True); c.add_argument("--hours", default="03:00:00")
    e = sub.add_parser("extc"); e.add_argument("--state", required=True); e.add_argument("--repo", required=True); e.add_argument("--hours", default="05:00:00")
    a = ap.parse_args(argv)
    print(main_command(a.state, a.ratios_json, a.repo, a.hours) if a.kind == "main" else calibration_command(a.state, a.repo, a.hours) if a.kind == "calibration" else extc_command(a.state, a.repo, a.hours))


if __name__ == "__main__":
    main()
