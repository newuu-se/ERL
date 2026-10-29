"""
Plots for the train-energy RL study. Every image holds ONE chart.

Training (the training log: logs/train_run_*.log, logs/train_return_*.log):
  training_reward_<trip>      test reward per epoch
  training_energy_<trip>      energy consumed per trip (test trips), per epoch
  training_time_<trip>        trip time per epoch, against the deadline
  training_arrivals_<trip>    share of test trips that reached the end

The policy against constant targets (results/baselines_<trip>.json + eval summary):
  energy_vs_time_<trip>       every constant target as a point, the RL policy on top

One evaluated trip (rl/evaluate.py: results[/return]/action_profile[_stochastic].csv):
  speed_<trip>                train speed and speed limit along the route
  actions_<trip>              the action the agent chose along the route
  action_share_<trip>         how much of the trip time each action was used
  notch_<trip>                the notch the simulator chose along the route
  elevation_<trip>            route height profile, from the grade
  energy_cumulative_<trip>    energy consumed along the route
  energy_per_km_<trip>        energy consumed in each kilometre

Energy is always energy CONSUMED (what the RL optimises). Regenerated and net
energy are reported by rl/evaluate.py only and are not plotted.

Every chart is saved as PNG and PDF under results/plots/. A chart whose input is
missing is skipped with the command that makes it.

Usage:
    source venv/bin/activate
    python rl/make_plots.py
    python rl/make_plots.py --log logs/train_run_20261030_101500.log   # pick a log
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from rl.train_env import DEADLINE_STEPS, N_ACTIONS, action_label

RESULTS = os.path.join(_REPO, "results")
OUTDIR = os.path.join(RESULTS, "plots")

TRIP_NAMES = {"forward": "forward trip A→B", "return": "return trip B→A"}

# One main colour for the policy, grey for context, red only for the deadline.
BLUE = "#2a78d6"
GREY = "#8a8984"
RED = "#d1453b"
INK = "#1b1b19"
FIGSIZE = (7.0, 3.2)

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        # Arial first: the macOS Helvetica lacks the → used in the trip names
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.edgecolor": GREY,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.6,
        "xtick.color": INK,
        "ytick.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.6,
    }
)

thousands = FuncFormatter(lambda v, _pos: f"{v:,.0f}")


def save(fig, name: str) -> str:
    os.makedirs(OUTDIR, exist_ok=True)
    png = os.path.join(OUTDIR, f"{name}.png")
    fig.savefig(png)
    fig.savefig(os.path.join(OUTDIR, f"{name}.pdf"))
    plt.close(fig)
    return png


def new_chart(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title(title, pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def hours_minutes(seconds: float) -> str:
    minutes = round(seconds / 60)
    return f"{minutes // 60} h {minutes % 60:02d} min"


def results_dir(trip: str) -> str:
    return os.path.join(RESULTS, "return") if trip == "return" else RESULTS


# ── data loaders ─────────────────────────────────────────────────────────────


def load_baselines(trip: str):
    """Constant-target trips measured by rl/run_baselines.py (arrived ones only)."""
    path = os.path.join(RESULTS, f"baselines_{trip}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        records = [r for r in json.load(f)["records"] if r.get("arrived", True)]
    return sorted(records, key=lambda r: r["action"]) or None


def best_on_time_baseline(baselines):
    """The lowest-energy constant target that arrives before the deadline."""
    on_time = [r for r in baselines or [] if r["steps"] <= DEADLINE_STEPS]
    return min(on_time, key=lambda r: r["energy_kwh"]) if on_time else None


def load_profile(trip: str):
    """Per-second profile from rl/evaluate.py: the sampled run if present, else argmax.

    Returns (columns, summary, actions_label) or None."""
    for suffix, label in (("_stochastic", "sampled actions"), ("", "argmax actions")):
        path = os.path.join(results_dir(trip), f"action_profile{suffix}.csv")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if not rows or "speed_limit_mps" not in rows[0]:
            print(
                f"    {os.path.relpath(path, _REPO)} is from an older evaluate.py; re-run it"
            )
            return None
        cols = {
            key: np.array([float(r[key]) for r in rows])
            for key in rows[0]
            if key != "target"
        }
        cols["km"] = cols["position_m"] / 1000.0
        summary_path = os.path.join(results_dir(trip), f"eval_summary{suffix}.json")
        summary = None
        if os.path.isfile(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
        return cols, summary, label
    return None


def find_log(trip: str, given: str | None):
    if given:
        return given if os.path.isfile(given) else None
    pattern = "train_return_*.log" if trip == "return" else "train_run_*.log"
    logs = sorted(glob.glob(os.path.join(_REPO, "logs", pattern)), key=os.path.getmtime)
    return logs[-1] if logs else None


EPOCH_RE = re.compile(
    r"Epoch #(\d+):.*?test_reward:\s*(-?[\d.]+).*?best_reward:\s*(-?[\d.]+)"
)
EPISODE_RE = re.compile(
    r"\[(✓ ARRIVED|✗ TIMEOUT)\]\s+ep=\s*\d+\s+([\d,]+) steps.*?energy=\s*([\d.]+) kWh"
)
HEADER_RE = re.compile(r"^\s*Epoch\s+(\d+)/\d+")  # train.py, at each training collect
TEST_MARK = "[test]"  # train.py, right before the test trips of an epoch


def load_training(trip: str, log_path: str | None):
    """Per-epoch numbers from a training log.

    train.py prints "Epoch N/M" while it trains and "[test]" before the test trips;
    trips printed after "[test]" are the test trips of that epoch (epoch 0 = the
    test before any training). Test trips use the current policy, so their
    per-epoch average is a clean learning curve."""
    path = find_log(trip, log_path)
    if path is None:
        return None
    rewards = {}  # epoch -> (test reward, best reward)
    tests = {}  # epoch -> list of (arrived, steps, energy)
    epoch, testing = 0, False
    with open(path, errors="replace") as f:
        for line in f:
            if m := HEADER_RE.search(line):
                epoch, testing = int(m.group(1)), False
            elif TEST_MARK in line:
                testing = True
            elif m := EPOCH_RE.search(line):
                rewards[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
            elif testing and (m := EPISODE_RE.search(line)):
                arrived = m.group(1).endswith("ARRIVED")
                tests.setdefault(epoch, []).append(
                    (arrived, float(m.group(2).replace(",", "")), float(m.group(3)))
                )
    if not rewards and not tests:
        return None

    epochs = sorted(tests)
    energy = []
    for e in epochs:  # energy of arrived trips only: a timed-out trip is partial
        arrived_energy = [en for ok, _s, en in tests[e] if ok]
        energy.append(np.mean(arrived_energy) if arrived_energy else np.nan)
    return {
        "log": os.path.relpath(path, _REPO),
        "reward_epoch": np.array(sorted(rewards)),
        "test_reward": np.array([rewards[e][0] for e in sorted(rewards)]),
        "best_reward": np.array([rewards[e][1] for e in sorted(rewards)]),
        "epoch": np.array(epochs),
        "energy": np.array(energy),
        "time": np.array([np.mean([s for _ok, s, _en in tests[e]]) for e in epochs]),
        "arrived_pct": np.array(
            [100.0 * np.mean([ok for ok, _s, _en in tests[e]]) for e in epochs]
        ),
    }


# ── training ─────────────────────────────────────────────────────────────────


def fig_training_reward(trip, train):
    if train is None or not len(train["reward_epoch"]):
        return None
    fig, ax = new_chart(
        f"Test reward per epoch, {TRIP_NAMES[trip]}", "epoch", "episode reward"
    )
    ax.plot(
        train["reward_epoch"],
        train["best_reward"],
        color=GREY,
        ls="--",
        label="best so far",
    )
    ax.plot(
        train["reward_epoch"],
        train["test_reward"],
        color=BLUE,
        marker="o",
        ms=3,
        label="test reward",
    )
    ax.yaxis.set_major_formatter(thousands)
    ax.legend(loc="lower right")
    return save(fig, f"training_reward_{trip}")


def fig_training_energy(trip, train, baselines):
    if train is None or not len(train["epoch"]):
        return None
    fig, ax = new_chart(
        f"Energy consumed per trip, {TRIP_NAMES[trip]} (test trips)",
        "epoch",
        "energy (kWh)",
    )
    ax.plot(
        train["epoch"], train["energy"], color=BLUE, marker="o", ms=3, label="RL policy"
    )
    best = best_on_time_baseline(baselines)
    if best:
        ax.axhline(
            best["energy_kwh"],
            color=GREY,
            ls="--",
            label=f"best on-time constant target: {action_label(best['action'])} "
            f"({best['energy_kwh']:,.0f} kWh)",
        )
    ax.yaxis.set_major_formatter(thousands)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2)  # below the chart
    return save(fig, f"training_energy_{trip}")


def fig_training_time(trip, train):
    if train is None or not len(train["epoch"]):
        return None
    fig, ax = new_chart(
        f"Trip time per epoch, {TRIP_NAMES[trip]} (test trips)",
        "epoch",
        "trip time (s)",
    )
    ax.plot(
        train["epoch"], train["time"], color=BLUE, marker="o", ms=3, label="RL policy"
    )
    ax.axhline(
        DEADLINE_STEPS,
        color=RED,
        ls="--",
        lw=1.2,
        label=f"deadline {DEADLINE_STEPS:,} s",
    )
    ax.yaxis.set_major_formatter(thousands)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2)  # below the chart
    return save(fig, f"training_time_{trip}")


def fig_training_arrivals(trip, train):
    if train is None or not len(train["epoch"]):
        return None
    fig, ax = new_chart(
        f"Test trips that reached the end, {TRIP_NAMES[trip]}", "epoch", "arrived (%)"
    )
    ax.plot(train["epoch"], train["arrived_pct"], color=BLUE, marker="o", ms=3)
    ax.set_ylim(0, 105)
    return save(fig, f"training_arrivals_{trip}")


# ── the policy against constant targets ──────────────────────────────────────


def fig_energy_vs_time(trip, baselines, profile):
    if baselines is None:
        return None
    fig, ax = new_chart(
        f"Energy vs trip time, {TRIP_NAMES[trip]}",
        "trip time (s)",
        "energy consumed (kWh)",
    )
    times = [r["steps"] for r in baselines]
    energies = [r["energy_kwh"] for r in baselines]
    ax.plot(times, energies, color=GREY, lw=1, zorder=1)
    ax.scatter(
        times,
        energies,
        s=28,
        color="white",
        edgecolors=GREY,
        linewidths=1.2,
        zorder=2,
        label="constant target (whole trip)",
    )
    for r in baselines:
        ax.annotate(
            action_label(r["action"]),
            (r["steps"], r["energy_kwh"]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=GREY,
        )
    ax.axvline(
        DEADLINE_STEPS,
        color=RED,
        ls="--",
        lw=1.2,
        label=f"deadline {DEADLINE_STEPS:,} s",
    )
    if profile is not None and profile[1] is not None:
        s = profile[1]
        ax.scatter(
            [s["trip_time_s"]],
            [s["energy_consumed_kwh"]],
            s=70,
            color=BLUE,
            zorder=3,
            label=f"RL policy ({profile[2]}): {s['energy_consumed_kwh']:,.0f} kWh, "
            f"{s['trip_time_s']:,} s",
        )
    ax.xaxis.set_major_formatter(thousands)
    ax.yaxis.set_major_formatter(thousands)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2)  # below the chart
    return save(fig, f"energy_vs_time_{trip}")


# ── one evaluated trip ───────────────────────────────────────────────────────


def fig_speed(trip, profile):
    if profile is None:
        return None
    p, s, label = profile
    title = f"Speed along the {TRIP_NAMES[trip]} ({label})"
    if s:
        title += f", trip time {hours_minutes(s['trip_time_s'])}"
    fig, ax = new_chart(title, "distance from the start (km)", "speed (m/s)")
    ax.step(
        p["km"],
        p["speed_limit_mps"],
        where="post",
        color=GREY,
        lw=1.2,
        label="speed limit",
    )
    ax.plot(p["km"], p["speed_mps"], color=BLUE, lw=1.1, label="train speed")
    ax.set_xlim(0, p["km"][-1])
    ax.set_ylim(0, max(p["speed_limit_mps"].max(), p["speed_mps"].max()) * 1.08)
    ax.legend(loc="upper right", ncol=2)
    return save(fig, f"speed_{trip}")


def fig_actions(trip, profile):
    if profile is None:
        return None
    p, _s, label = profile
    fig, ax = new_chart(
        f"Action chosen along the {TRIP_NAMES[trip]} ({label})",
        "distance from the start (km)",
        "action",
    )
    ax.step(p["km"], p["action"], where="post", color=BLUE, lw=0.8)
    ax.set_yticks(range(N_ACTIONS))
    ax.set_yticklabels([action_label(a) for a in range(N_ACTIONS)])
    ax.set_ylim(-0.5, N_ACTIONS - 0.5)
    ax.set_xlim(0, p["km"][-1])
    return save(fig, f"actions_{trip}")


def fig_action_share(trip, profile):
    if profile is None:
        return None
    p, _s, label = profile
    seconds = np.bincount(p["action"].astype(int), minlength=N_ACTIONS)
    share = 100.0 * seconds / seconds.sum()
    fig, ax = new_chart(
        f"Share of trip time per action, {TRIP_NAMES[trip]} ({label})",
        "share of trip time (%)",
        "",
    )
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    labels = [action_label(a) for a in range(N_ACTIONS)]
    ax.barh(labels, share, color=BLUE, height=0.6)
    for y, v in enumerate(share):
        ax.text(v + 0.5, y, f"{v:.0f} %", va="center", fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max(share.max() * 1.15, 10))
    return save(fig, f"action_share_{trip}")


def fig_notch(trip, profile):
    if profile is None:
        return None
    p, _s, label = profile
    fig, ax = new_chart(
        f"Notch chosen by the simulator, {TRIP_NAMES[trip]} ({label})",
        "distance from the start (km)",
        "notch",
    )
    ax.step(p["km"], p["notch"], where="post", color=BLUE, lw=0.8)
    ax.set_yticks(range(9))
    ax.set_ylim(-0.3, 8.3)
    ax.set_xlim(0, p["km"][-1])
    return save(fig, f"notch_{trip}")


def fig_elevation(trip, profile):
    """Height along the trip, from the grade the train saw each second."""
    if profile is None:
        return None
    p, _s, _label = profile
    step_m = np.diff(p["position_m"], prepend=p["position_m"][0])
    height = np.cumsum(step_m * p["grade_perc"] / 100.0)
    fig, ax = new_chart(
        f"Route height profile, {TRIP_NAMES[trip]} (net {height[-1]:+,.0f} m)",
        "distance from the start (km)",
        "height above the start (m)",
    )
    ax.fill_between(p["km"], height, height.min(), color=GREY, alpha=0.25, lw=0)
    ax.plot(p["km"], height, color=INK, lw=1.0)
    ax.set_xlim(0, p["km"][-1])
    return save(fig, f"elevation_{trip}")


def fig_energy_cumulative(trip, profile):
    if profile is None:
        return None
    p, _s, label = profile
    km, e = p["km"], p["energy_consumed_kwh_cum"]
    fig, ax = new_chart(
        f"Energy consumed: {e[-1]:,.0f} kWh over {km[-1]:.1f} km, {TRIP_NAMES[trip]} ({label})",
        "distance from the start (km)",
        "energy consumed (kWh)",
    )
    ax.plot(
        [0, km[-1]],
        [0, e[-1]],
        color=GREY,
        ls="--",
        lw=1.0,
        label=f"uniform rate ({e[-1] / km[-1]:.1f} kWh/km)",
    )
    ax.plot(km, e, color=BLUE, label="RL policy")
    ax.set_xlim(0, km[-1])
    ax.yaxis.set_major_formatter(thousands)
    ax.legend(loc="upper left")
    return save(fig, f"energy_cumulative_{trip}")


def fig_energy_per_km(trip, profile):
    if profile is None:
        return None
    p, _s, label = profile
    km, e = p["km"], p["energy_consumed_kwh_cum"]
    edges = np.arange(0.0, np.floor(km[-1]) + 1.0)
    per_km = np.diff(np.interp(edges, km, e))
    fig, ax = new_chart(
        f"Energy consumed in each kilometre, {TRIP_NAMES[trip]} ({label})",
        "distance from the start (km)",
        "energy (kWh per km)",
    )
    ax.bar(edges[:-1] + 0.5, per_km, width=0.8, color=BLUE)
    ax.set_xlim(0, km[-1])
    return save(fig, f"energy_per_km_{trip}")


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        default=None,
        help="forward training log (default: newest logs/train_run_*.log)",
    )
    parser.add_argument(
        "--return-log",
        default=None,
        help="return training log (default: newest logs/train_return_*.log)",
    )
    args = parser.parse_args()

    print(f"writing charts to {os.path.relpath(OUTDIR, _REPO)}/\n")
    made = total = 0
    for trip in ("forward", "return"):
        flag = " --return-trip" if trip == "return" else ""
        train_cmd = (
            "python rl/train_return.py 2>&1 | tee logs/train_return_<time>.log"
            if trip == "return"
            else "python rl/train.py 2>&1 | tee logs/train_run_<time>.log"
        )
        eval_cmd = f"python rl/evaluate.py --stochastic{flag}"
        train = load_training(trip, args.return_log if trip == "return" else args.log)
        baselines = load_baselines(trip)
        profile = load_profile(trip)
        # (chart name, function, its inputs, the command that makes the inputs)
        charts = [
            ("training_reward", fig_training_reward, (train,), train_cmd),
            ("training_energy", fig_training_energy, (train, baselines), train_cmd),
            ("training_time", fig_training_time, (train,), train_cmd),
            ("training_arrivals", fig_training_arrivals, (train,), train_cmd),
            (
                "energy_vs_time",
                fig_energy_vs_time,
                (baselines, profile),
                f"python rl/run_baselines.py{flag}",
            ),
            ("speed", fig_speed, (profile,), eval_cmd),
            ("actions", fig_actions, (profile,), eval_cmd),
            ("action_share", fig_action_share, (profile,), eval_cmd),
            ("notch", fig_notch, (profile,), eval_cmd),
            ("elevation", fig_elevation, (profile,), eval_cmd),
            ("energy_cumulative", fig_energy_cumulative, (profile,), eval_cmd),
            ("energy_per_km", fig_energy_per_km, (profile,), eval_cmd),
        ]
        print(f"{TRIP_NAMES[trip]}" + (f"  (log: {train['log']})" if train else ""))
        for name, make, inputs, howto in charts:
            total += 1
            out = make(trip, *inputs)
            if out:
                made += 1
                print(f"  ✓ {name}_{trip}.png (+ .pdf)")
            else:
                print(f"  – {name}_{trip}: no data, run: {howto}")
        print()
    print(f"{made}/{total} charts written.")


if __name__ == "__main__":
    main()
