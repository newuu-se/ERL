"""
Measure the constant-notch energy/time frontier on the CURRENT physics + data.

Run after any physics or input-data change. The frontier is the reference every
result is compared against, so it must be re-measured whenever the simulator or
the route data changes.

Output is written to results/baselines_<trip>.json — a machine-readable file that
rl/make_plots.py reads directly. Nothing is copy-pasted between scripts, so the
plots can never silently disagree with the measurement.

Usage:
    source venv/bin/activate
    python rl/run_baselines.py [notches...]                 # A→B trip, default: 2 3 4 5 6 8
    python rl/run_baselines.py --return-trip [notches...]   # B→A trip (train_return.dat)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.train_env import (NeTrainSimEnv, TRAINS_FILE, TRAINS_FILE_RETURN,
                          LINKS_FILE)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_REPO, "results")
DEFAULT_NOTCHES = [2, 3, 4, 5, 6, 8]


def baselines_path(trip: str) -> str:
    """Canonical location of the measured frontier for a trip."""
    return os.path.join(RESULTS_DIR, f"baselines_{trip}.json")


def load_baselines(trip: str) -> dict:
    """Read a measured frontier. Raises if it was never measured — a missing
    frontier is a real error, not something to silently paper over with stale
    constants."""
    path = baselines_path(trip)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No measured baseline at {path}\n"
            f"Run: python rl/run_baselines.py"
            f"{' --return-trip' if trip == 'return' else ''}"
        )
    with open(path) as f:
        return json.load(f)


def run_constant_notch(notch: int, trains_file: str = TRAINS_FILE):
    env = NeTrainSimEnv(trains_file=trains_file)
    env.reset()
    terminated = truncated = False
    max_step_energy = 0.0
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(notch)
        max_step_energy = max(max_step_energy,
                              float(env._last_state["energy_kwh"]))
    steps, energy = env._step_count, env._cum_energy_kwh
    env.close()
    return steps, energy, max_step_energy, terminated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--return-trip", action="store_true",
                        help="measure the B→A return trip (train_return.dat)")
    parser.add_argument("notches", nargs="*", type=int, default=DEFAULT_NOTCHES)
    args = parser.parse_args()

    trip = "return" if args.return_trip else "forward"
    trains_file = TRAINS_FILE_RETURN if args.return_trip else TRAINS_FILE

    print(f"trip: {'B→A return' if args.return_trip else 'A→B forward'}  "
          f"({os.path.basename(trains_file)})")
    print(f"{'notch':>5}  {'steps(s)':>9}  {'energy(kWh)':>12}  "
          f"{'maxE/step':>9}  {'arrived':>7}")

    records = []
    for n in sorted(args.notches):
        steps, energy, max_e, arrived = run_constant_notch(n, trains_file)
        records.append({
            "notch": n,
            "steps": steps,
            "energy_kwh": round(energy, 2),
            "max_step_energy_kwh": round(max_e, 3),
            "arrived": bool(arrived),
        })
        print(f"{n:>5}  {steps:>9,}  {energy:>12.2f}  {max_e:>9.3f}  "
              f"{str(arrived):>7}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "trip": trip,
        "trains_file": os.path.basename(trains_file),
        "links_file": os.path.basename(LINKS_FILE),
        "physics": "GOST resistance, regenerative braking removed",
        "records": records,
    }
    out = baselines_path(trip)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"\nwrote {os.path.relpath(out, _REPO)}  "
          f"— rl/make_plots.py reads this directly (no manual copy step)")


if __name__ == "__main__":
    main()
