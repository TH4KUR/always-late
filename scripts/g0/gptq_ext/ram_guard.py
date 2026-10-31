"""Pre-job RAM estimate for the Gemma-2-9B weight-set jobs (plan docs/superpowers/plans/2026-10-28-gemma9b-three-arm-confirmation.md, Task 3). A job keeps in host memory: the full-precision copy of the model
(`clone_state_to_cpu`, 18.5 GB for Gemma-2-9B in bf16), and, when it builds weight sets, the calibrator's host copy of the original target matrices (17.6 GB, released after each build) plus one artifact per bit width
(the target matrices in bf16, 17.6 GB each). A legacy-RTN job (no artifacts) needs the first term only. The estimate carries a margin of 10 % for activations and Python overhead.

  python -m gptq_ext.ram_guard --model-gb 18.5 --target-gb 17.6 --n-artifacts 2 [--ram-gb 160]      (exit 4 when the estimate exceeds the RAM; without --ram-gb the machine's MemTotal is used)"""
import argparse
import sys


class RamShortfall(RuntimeError):
    """The estimated host RAM of a job exceeds the RAM that is there (the message carries both numbers)."""


def estimate_gb(model_gb, target_gb, n_artifacts, margin=1.1):
    """Estimated peak host RAM in GB: (model + [calibrator copy of the targets + n_artifacts artifacts, only when n_artifacts > 0]) x margin."""
    extra = target_gb * (1 + n_artifacts) if n_artifacts > 0 else 0.0
    return (model_gb + extra) * margin


def check(ram_gb, model_gb, target_gb, n_artifacts, margin=1.1):
    """-> the estimate; raises RamShortfall when it exceeds `ram_gb`."""
    need = estimate_gb(model_gb, target_gb, n_artifacts, margin)
    if need > ram_gb:
        raise RamShortfall(f"the job needs about {need:.1f} GB of host RAM ({n_artifacts} artifact(s) of {target_gb} GB, model copy {model_gb} GB, margin {margin}) but only {ram_gb:.1f} GB are available")
    return need


def machine_ram_gb():
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1048576
    raise RuntimeError("MemTotal not found in /proc/meminfo")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Estimate the host RAM of a weight-set job and refuse when it does not fit.")
    ap.add_argument("--model-gb", type=float, required=True); ap.add_argument("--target-gb", type=float, required=True); ap.add_argument("--n-artifacts", type=int, required=True)
    ap.add_argument("--ram-gb", type=float); ap.add_argument("--margin", type=float, default=1.1)
    a = ap.parse_args(argv)
    ram = a.ram_gb if a.ram_gb is not None else machine_ram_gb()
    try:
        need = check(ram, a.model_gb, a.target_gb, a.n_artifacts, a.margin)
    except RamShortfall as e:
        print("RAM SHORTFALL:", e, file=sys.stderr)
        return 4
    print(f"estimated host RAM {need:.1f} GB of {ram:.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
