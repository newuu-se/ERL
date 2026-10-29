"""
Compare NeTrainSim's own driver with the trained RL policy on one trip.

NeTrainSim's normal mode (no -I) drives the train by itself: full power up to the
speed limit, hold it, brake for the next lower limit or the stop. This script runs
that mode once, reads its per-second trajectory, and puts it next to the RL
policy's per-second profile from rl/evaluate.py.

Energy: the trajectory gives the NET energy of each second. Seconds with positive
energy are consumption and negative ones are regeneration, the same split that
NeTrainSim's summary file reports. The comparison uses energy CONSUMED, like the RL.

Inputs:  the simulator and route files from .env
         results[/return]/action_profile_stochastic.csv  (rl/evaluate.py --stochastic)
         results[/return]/action_profile.csv             (rl/evaluate.py, argmax)
Outputs: $NETRAINSIM_OUTPUT_DIR/driver_<trip>_trajectory.csv and _summary.txt
         results/plots/compare_speed_<trip>.png, compare_energy_<trip>.png (+ .pdf)
         results[/return]/compare_driver.json

Usage:
    source .venv/bin/activate
    python rl/compare_driver.py                 # forward trip, charts with sampled RL
    python rl/compare_driver.py --argmax        # charts with the argmax RL run
    python rl/compare_driver.py --return-trip
"""

import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.make_plots import (
    BLUE,
    GREY,
    RESULTS,
    TRIP_NAMES,
    hours_minutes,
    new_chart,
    results_dir,
    save,
    thousands,
)
from rl.train_env import (
    DEADLINE_STEPS,
    LINKS_FILE,
    NODES_FILE,
    SIMULATOR_BIN,
    SIMULATOR_OUTPUT_DIR,
    TRAINS_FILE,
    TRAINS_FILE_RETURN,
)

ORANGE = "#eb6834"  # NeTrainSim's own driver


def run_driver(trip: str) -> str:
    """Run NeTrainSim's normal mode once; return the trajectory file path."""
    outdir = SIMULATOR_OUTPUT_DIR or os.path.join(RESULTS, "netrainsim")
    os.makedirs(outdir, exist_ok=True)
    trains = TRAINS_FILE_RETURN if trip == "return" else TRAINS_FILE
    trajectory = f"driver_{trip}_trajectory.csv"
    subprocess.run(
        [
            SIMULATOR_BIN,
            "-n",
            NODES_FILE,
            "-l",
            LINKS_FILE,
            "-t",
            trains,
            "-o",
            outdir,
            "-s",
            f"driver_{trip}_summary.txt",
            "-e",
            "true",
            "-i",
            trajectory,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return os.path.join(outdir, trajectory)


def load_driver(path: str) -> dict:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    energy = np.array([float(r["EnergyConsumption_KWH"]) for r in rows])
    return {
        "time_s": np.array([float(r["TStep_s"]) + 1 for r in rows]),
        "km": np.array([float(r["TravelledDistance_m"]) for r in rows]) / 1000.0,
        "speed": np.array([float(r["Speed_mps"]) for r in rows]),
        "limit": np.array([float(r["LinkMaxSpeed_mps"]) for r in rows]),
        "consumed": np.cumsum(np.clip(energy, 0, None)),
        "regenerated": -np.clip(energy, None, 0).sum(),
    }


def load_rl(trip: str, argmax: bool):
    suffix = "" if argmax else "_stochastic"
    path = os.path.join(results_dir(trip), f"action_profile{suffix}.csv")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {
        "time_s": np.array([float(r["time_s"]) for r in rows]),
        "km": np.array([float(r["position_m"]) for r in rows]) / 1000.0,
        "speed": np.array([float(r["speed_mps"]) for r in rows]),
        "consumed": np.array([float(r["energy_consumed_kwh_cum"]) for r in rows]),
        "regenerated": float(rows[-1]["energy_regenerated_kwh_cum"]),
    }


def totals(d: dict) -> dict:
    trip_time = int(d["time_s"][-1])
    consumed = float(d["consumed"][-1])
    return {
        "trip_time_s": trip_time,
        "on_time": trip_time <= DEADLINE_STEPS,
        "energy_consumed_kwh": round(consumed, 2),
        "energy_regenerated_kwh": round(float(d["regenerated"]), 2),  # reported only
        "energy_net_kwh": round(consumed - float(d["regenerated"]), 2),  # reported only
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--return-trip", action="store_true")
    parser.add_argument(
        "--argmax",
        action="store_true",
        help="chart the argmax RL run instead of the sampled one",
    )
    args = parser.parse_args()
    trip = "return" if args.return_trip else "forward"

    rl_label = "argmax" if args.argmax else "sampled"
    rl = load_rl(trip, args.argmax)
    if rl is None:
        flags = ("" if args.argmax else " --stochastic") + (
            " --return-trip" if args.return_trip else ""
        )
        sys.exit(f"No RL profile for this trip; run: python rl/evaluate.py{flags}")
    driver = load_driver(run_driver(trip))

    table = {"NeTrainSim driver": totals(driver)}
    for name, is_argmax in (
        ("RL policy (sampled)", False),
        ("RL policy (argmax)", True),
    ):
        profile = load_rl(trip, is_argmax)
        if profile is not None:
            table[name] = totals(profile)
    with open(os.path.join(results_dir(trip), "compare_driver.json"), "w") as f:
        json.dump(
            {"trip": trip, "deadline_s": DEADLINE_STEPS, "runs": table}, f, indent=2
        )
        f.write("\n")

    d, r = table["NeTrainSim driver"], table[f"RL policy ({rl_label})"]
    print(f"{TRIP_NAMES[trip]}, deadline {DEADLINE_STEPS:,} s")
    print(f"{'':<22}{'trip time':>12}{'consumed':>12}{'regen*':>10}{'net*':>10}")
    for name, t in table.items():
        print(
            f"{name:<22}{t['trip_time_s']:>10,} s{t['energy_consumed_kwh']:>8,.1f} kWh"
            f"{t['energy_regenerated_kwh']:>10.1f}{t['energy_net_kwh']:>10.1f}"
        )
    print("* reported only, not used by the RL")

    # chart 1: speed along the trip
    title = (
        f"Speed, {TRIP_NAMES[trip]}: NeTrainSim driver {hours_minutes(d['trip_time_s'])}, "
        f"RL policy {hours_minutes(r['trip_time_s'])}"
    )
    fig, ax = new_chart(title, "distance from the start (km)", "speed (m/s)")
    ax.step(
        driver["km"],
        driver["limit"],
        where="post",
        color=GREY,
        lw=1.0,
        alpha=0.6,
        label="speed limit",
    )
    ax.plot(
        driver["km"], driver["speed"], color=ORANGE, lw=0.9, label="NeTrainSim driver"
    )
    ax.plot(
        rl["km"],
        rl["speed"],
        color=BLUE,
        lw=0.9,
        label=f"RL policy ({rl_label} actions)",
    )
    ax.set_xlim(0, max(driver["km"][-1], rl["km"][-1]))
    ax.set_ylim(0, driver["limit"].max() * 1.08)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=3)
    speed_png = save(fig, f"compare_speed_{trip}")

    # chart 2: energy consumed along the trip
    change = r["energy_consumed_kwh"] / d["energy_consumed_kwh"] - 1
    title = (
        f"Energy consumed, {TRIP_NAMES[trip]}: RL policy {r['energy_consumed_kwh']:,.0f} kWh, "
        f"NeTrainSim driver {d['energy_consumed_kwh']:,.0f} kWh ({change:+.1%})"
    )
    fig, ax = new_chart(title, "distance from the start (km)", "energy consumed (kWh)")
    ax.plot(driver["km"], driver["consumed"], color=ORANGE, label="NeTrainSim driver")
    ax.plot(
        rl["km"], rl["consumed"], color=BLUE, label=f"RL policy ({rl_label} actions)"
    )
    ax.set_xlim(0, max(driver["km"][-1], rl["km"][-1]))
    ax.yaxis.set_major_formatter(thousands)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2)
    energy_png = save(fig, f"compare_energy_{trip}")

    print(
        f"\ncharts: {os.path.relpath(speed_png)}, {os.path.relpath(energy_png)} (+ .pdf)"
    )


if __name__ == "__main__":
    main()
