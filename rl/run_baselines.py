"""
Measure the constant-target energy/time frontier on the CURRENT physics + data.

A constant-target policy holds one action for the whole trip, e.g. 70 % of the
speed limit everywhere. Coasting alone (action 0) never leaves the start, so it
is not measured.

Run after any physics or input-data change. The frontier is the reference every
result is compared against, so it must be re-measured whenever the simulator or
the route data changes.

Output is written to results/baselines_<trip>.json — a machine-readable file that
rl/make_plots.py reads directly. Nothing is copy-pasted between scripts, so the
plots can never silently disagree with the measurement.

Usage:
    source venv/bin/activate
    python rl/run_baselines.py [actions...]                 # A→B trip, default: 1-6 (50-100 %)
    python rl/run_baselines.py --return-trip [actions...]   # B→A trip (train_return.dat)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.train_env import (
    NeTrainSimEnv,
    TRAINS_FILE,
    TRAINS_FILE_RETURN,
    LINKS_FILE,
    N_ACTIONS,
    TARGET_FRACTIONS,
    action_label,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_REPO, "results")
DEFAULT_ACTIONS = list(range(1, N_ACTIONS))


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


def run_constant_action(action: int, trains_file: str = TRAINS_FILE):
    env = NeTrainSimEnv(trains_file=trains_file)
    env.reset()
    terminated = truncated = False
    max_step_energy = 0.0
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(action)
        max_step_energy = max(max_step_energy, env._last_step_energy_kwh)
    # energy = energy consumed, the quantity the RL optimises. Regenerated and net
    # energy are recorded for reporting only.
    steps, energy = env._step_count, env._cum_energy_kwh
    regenerated = float(env._last_state["energy_regenerated_kwh_cum"])
    net = float(env._last_state["energy_net_kwh_cum"])
    env.close()
    return steps, energy, regenerated, net, max_step_energy, terminated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--return-trip",
        action="store_true",
        help="measure the B→A return trip (train_return.dat)",
    )
    parser.add_argument("actions", nargs="*", type=int, default=DEFAULT_ACTIONS)
    args = parser.parse_args()

    trip = "return" if args.return_trip else "forward"
    trains_file = TRAINS_FILE_RETURN if args.return_trip else TRAINS_FILE

    print(
        f"trip: {'B→A return' if args.return_trip else 'A→B forward'}  "
        f"({os.path.basename(trains_file)})"
    )
    # consumed is what the RL uses; regenerated and net (*) are reported only
    print(
        f"{'target':>6}  {'steps(s)':>9}  {'consumed(kWh)':>13}  {'regen*':>7}  "
        f"{'net*':>7}  {'maxE/step':>9}  {'arrived':>7}"
    )

    records = []
    for a in sorted(args.actions):
        steps, energy, regenerated, net, max_e, arrived = run_constant_action(
            a, trains_file
        )
        records.append(
            {
                "action": a,
                "target_frac": TARGET_FRACTIONS[a],
                "steps": steps,
                "energy_kwh": round(energy, 2),  # consumed
                # reported only, not used by the RL or the plots
                "energy_regenerated_kwh": round(regenerated, 2),
                "energy_net_kwh": round(net, 2),
                "max_step_energy_kwh": round(max_e, 3),
                "arrived": bool(arrived),
            }
        )
        print(
            f"{action_label(a):>6}  {steps:>9,}  {energy:>13.2f}  {regenerated:>7.2f}  "
            f"{net:>7.2f}  {max_e:>9.3f}  {str(arrived):>7}"
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "trip": trip,
        "trains_file": os.path.basename(trains_file),
        "links_file": os.path.basename(LINKS_FILE),
        "physics": (
            "newuu-se/NeTrainSim main + speed control: Grebenyuk (1987) rolling/air "
            "resistance + grade + curve resistance 700/R N/kN (curvature in % = 100/R); "
            "regenerative braking simulated but not credited: energy_kwh is energy "
            "consumed, regenerated/net are reported only"
        ),
        "records": records,
    }
    out = baselines_path(trip)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(
        f"\nwrote {os.path.relpath(out, _REPO)}  "
        f"— rl/make_plots.py reads this directly (no manual copy step)"
    )


if __name__ == "__main__":
    main()
