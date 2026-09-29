"""
Plot best vs worst energy episode side-by-side to illustrate PG exploration.

Usage:
    python rl/plot_episode_comparison.py                        # auto-pick
    python rl/plot_episode_comparison.py --best 403 --worst 1  # override
    python rl/plot_episode_comparison.py --out custom/path.png  # custom output
"""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_REPO       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_CSV    = os.path.join(_REPO, "logs", "training_log.csv")
_EP_DIR     = os.path.join(_REPO, "logs", "episodes")
_PLOTS_DIR  = os.path.join(_REPO, "logs", "plots")
_DEFAULT_OUT = os.path.join(_PLOTS_DIR, "best_vs_worst_episode.png")


_MIN_EPISODE_LINES = 1000   # files shorter than this are test-env overwrites (only 2 lines)


def _complete_episodes():
    """Return set of episode numbers whose CSV file has full trajectory data."""
    complete = set()
    for fname in os.listdir(_EP_DIR):
        if not fname.startswith("ep_") or not fname.endswith(".csv"):
            continue
        path = os.path.join(_EP_DIR, fname)
        with open(path) as f:
            n = sum(1 for _ in f)
        if n >= _MIN_EPISODE_LINES:
            ep_num = int(fname[3:7])
            complete.add(ep_num)
    return complete


def _read_training_log():
    """Read training log, keep only complete ARRIVED episodes, deduplicate by min energy."""
    complete = _complete_episodes()
    best_per_ep = {}   # ep_num → row with lowest energy (handles dup episode IDs)
    with open(_LOG_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "ARRIVED":
                continue
            ep = int(row["episode"])
            if ep not in complete:
                continue
            energy = float(row["total_energy_kwh"])
            if ep not in best_per_ep or energy < best_per_ep[ep]["energy_kwh"]:
                best_per_ep[ep] = {
                    "episode":    ep,
                    "steps":      int(row["steps"]),
                    "energy_kwh": energy,
                }
    return list(best_per_ep.values())


def _pick_best_worst(rows):
    best  = min(rows, key=lambda r: r["energy_kwh"])
    worst = max(rows, key=lambda r: r["energy_kwh"])
    return best, worst


def _load_episode(ep_num):
    path = os.path.join(_EP_DIR, f"ep_{ep_num:04d}.csv")
    # ndmin=1 prevents 0-d structured array when file has exactly one data row
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None,
                         encoding="utf-8", ndmin=1)
    return data


def _make_plot(best_meta, worst_meta, best_data, worst_data, out_path):
    pos_best  = best_data["position_m"]  / 1000.0   # → km
    pos_worst = worst_data["position_m"] / 1000.0

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(13, 10))
    fig.subplots_adjust(hspace=0.08, top=0.90, bottom=0.07, left=0.08, right=0.97)

    BEST_COLOR  = "#2ca02c"   # green
    WORST_COLOR = "#d62728"   # red
    LIMIT_COLOR = "#555555"

    # ── Panel 1: Notch ────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(pos_worst, worst_data["notch"], color=WORST_COLOR, lw=0.8,
            drawstyle="steps-post", alpha=0.85,
            label=f"Worst  ep {worst_meta['episode']:04d} ({worst_meta['energy_kwh']:.1f} kWh)")
    ax.plot(pos_best,  best_data["notch"],  color=BEST_COLOR,  lw=0.9,
            drawstyle="steps-post", alpha=0.90,
            label=f"Best   ep {best_meta['episode']:04d}  ({best_meta['energy_kwh']:.1f} kWh)")
    ax.set_ylabel("Notch (0–8)", fontsize=9)
    ax.set_yticks(range(0, 9))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_ylim(-0.3, 8.5)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax.set_title(
        f"Best vs Worst Energy Episode  |  "
        f"Δ = {worst_meta['energy_kwh'] - best_meta['energy_kwh']:.2f} kWh "
        f"({100*(worst_meta['energy_kwh'] - best_meta['energy_kwh'])/best_meta['energy_kwh']:.1f}%)",
        fontsize=11, pad=8,
    )

    # ── Panel 2: Speed ────────────────────────────────────────────────────
    ax = axes[1]
    # Speed limit is track-derived — use best episode (same for both)
    ax.plot(pos_best, best_data["max_speed_mps"], color=LIMIT_COLOR, lw=1.2,
            linestyle="--", alpha=0.7, label="Speed limit")
    ax.plot(pos_worst, worst_data["speed_mps"], color=WORST_COLOR, lw=0.8,
            alpha=0.75, label="Worst")
    ax.plot(pos_best,  best_data["speed_mps"],  color=BEST_COLOR,  lw=0.9,
            alpha=0.85, label="Best")
    ax.set_ylabel("Speed (m/s)", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

    # ── Panel 3: Grade ────────────────────────────────────────────────────
    ax = axes[2]
    grade = np.asarray(best_data["grade_pct"], dtype=float)   # same for both (same track)
    ax.plot(pos_best, grade, color="#1f77b4", lw=0.8)
    ax.fill_between(pos_best, grade, 0,
                    where=(grade >= 0), alpha=0.25, color="#d62728", label="Uphill")
    ax.fill_between(pos_best, grade, 0,
                    where=(grade <  0), alpha=0.25, color="#2ca02c", label="Downhill")
    ax.axhline(0, color="#888888", lw=0.6, linestyle=":")
    ax.set_ylabel("Grade (%)", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

    # ── Panel 4: Curvature ───────────────────────────────────────────────
    ax = axes[3]
    ax.plot(pos_best, best_data["curvature_pct"], color="#9467bd", lw=0.8)
    ax.set_ylabel("Curvature", fontsize=9)
    ax.set_xlabel("Position along route (km)", fontsize=9)

    for a in axes:
        a.grid(True, alpha=0.25, linewidth=0.5)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot best vs worst energy episode")
    parser.add_argument("--best",  type=int, default=None, help="Override best episode number")
    parser.add_argument("--worst", type=int, default=None, help="Override worst episode number")
    parser.add_argument("--out",   type=str, default=_DEFAULT_OUT, help="Output PNG path")
    args = parser.parse_args()

    rows = _read_training_log()
    if not rows:
        print("No ARRIVED episodes found in training_log.csv", file=sys.stderr)
        sys.exit(1)

    auto_best, auto_worst = _pick_best_worst(rows)
    ep_idx = {r["episode"]: r for r in rows}

    if args.best is not None:
        if args.best not in ep_idx:
            print(f"Episode {args.best} not in training_log.csv or not ARRIVED", file=sys.stderr)
            sys.exit(1)
        best_meta = ep_idx[args.best]
    else:
        best_meta = auto_best

    if args.worst is not None:
        if args.worst not in ep_idx:
            print(f"Episode {args.worst} not in training_log.csv or not ARRIVED", file=sys.stderr)
            sys.exit(1)
        worst_meta = ep_idx[args.worst]
    else:
        worst_meta = auto_worst

    print(f"Best:  ep_{best_meta['episode']:04d}  energy={best_meta['energy_kwh']:.2f} kWh  steps={best_meta['steps']}")
    print(f"Worst: ep_{worst_meta['episode']:04d}  energy={worst_meta['energy_kwh']:.2f} kWh  steps={worst_meta['steps']}")

    best_data  = _load_episode(best_meta["episode"])
    worst_data = _load_episode(worst_meta["episode"])

    _make_plot(best_meta, worst_meta, best_data, worst_data, args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
