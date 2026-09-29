"""
Figures for the train-energy RL study — for the paper, and for checking that the
model actually learned something sensible.

Four figures, each answering one question:

  1. fig_training_progress  Did training work?
                            Reward, trip energy, trip time and arrival rate over
                            the whole run. If these do not improve, nothing else
                            in this file is worth reading.

  2. fig_energy_time_tradeoff  Is the policy any good?
                            The constant-notch frontier (the thing to beat) with
                            the trained policy placed on it.

  3. fig_speed_profile      What does the policy actually DO?
                            Speed against the limit, the notch it chose, and the
                            terrain that drove the choice, along the route.

  4. fig_energy_usage       Where does the energy GO?
                            Cumulative energy and energy per km against grade —
                            this is where terrain-aware driving shows up or does not.

Every number is read from a file; nothing is pasted in as a constant:

  results/baselines_<trip>.json          rl/run_baselines.py
  results/notch_profile_stochastic.csv   rl/evaluate.py --stochastic
  data/netrainsim_v2/linksFile_v2_clean.dat   route grade + speed limits
  logs/train_run_*.log                   training stdout

A figure whose inputs are missing is skipped with a message saying what to run.

Output: results/plots/<name>.png (300 dpi) + <name>.pdf (vector, for submission).

Usage:
    source venv/bin/activate
    python rl/make_plots.py
"""
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
from matplotlib.ticker import MultipleLocator

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from rl.train_env import LINKS_FILE, DEADLINE_STEPS, TOTAL_ROUTE_LENGTH_M

RESULTS = os.path.join(_REPO, "results")
OUTDIR = os.path.join(RESULTS, "plots")
os.makedirs(OUTDIR, exist_ok=True)

ROUTE_KM = TOTAL_ROUTE_LENGTH_M / 1000.0

# ── figure geometry (journal column widths, inches) ──────────────────────────
COL1, COL2 = 3.46, 7.09

# ── Okabe-Ito colourblind-safe palette ───────────────────────────────────────
BLUE   = "#0072B2"   # the RL policy
ORANGE = "#E69F00"   # energy
GREEN  = "#009E73"   # return trip / success
VERM   = "#D55E00"   # limits, deadlines, warnings
PURPLE = "#CC79A7"   # notch
SKY    = "#56B4E9"
GREY   = "#6E6E6E"
INK    = "#1A1A1A"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#D9D9D9",
    "grid.linewidth": 0.4,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "lines.linewidth": 1.2,
    "lines.solid_capstyle": "round",
    "text.color": INK,
})


def save(fig, name):
    png = os.path.join(OUTDIR, f"{name}.png")
    fig.savefig(png)
    fig.savefig(os.path.join(OUTDIR, f"{name}.pdf"))
    plt.close(fig)
    return png


def panel(ax, text):
    ax.text(-0.13, 1.05, text, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="left")


def rolling(y, w):
    """Centred rolling mean, edge-padded so the output keeps its length."""
    y = np.asarray(y, float)
    if len(y) < w or w < 2:
        return y
    pad = w // 2
    return np.convolve(np.pad(y, pad, mode="edge"), np.ones(w) / w,
                       mode="same")[pad:pad + len(y)]


# ── data loaders ─────────────────────────────────────────────────────────────

def load_baselines(trip="forward"):
    """Constant-notch frontier measured by rl/run_baselines.py."""
    path = os.path.join(RESULTS, f"baselines_{trip}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        recs = sorted(json.load(f)["records"], key=lambda r: r["notch"])
    return {
        "notch":  np.array([r["notch"] for r in recs]),
        "steps":  np.array([r["steps"] for r in recs], float),
        "energy": np.array([r["energy_kwh"] for r in recs], float),
    }


def load_profile(trip="forward"):
    """One evaluated rollout (rl/evaluate.py --stochastic).

    energy_kwh_cum is optional: profiles exported before that column existed
    still load, they just cannot drive the energy figure."""
    sub = ("return",) if trip == "return" else ()
    path = os.path.join(RESULTS, *sub, "notch_profile_stochastic.csv")
    if not os.path.isfile(path):
        return None
    cols = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, []).append(v)
    out = {"pos": np.array([float(v) for v in cols["position_m"]]),
           "speed": np.array([float(v) for v in cols["speed_mps"]]),
           "notch": np.array([int(v) for v in cols["notch"]])}
    for opt, cast in (("energy_kwh_cum", float), ("time_s", float)):
        if opt in cols:
            out[opt.replace("energy_kwh_cum", "energy").replace("time_s", "time")] = \
                np.array([cast(v) for v in cols[opt]])
    return out


def load_links():
    cum, spd, grade = [0.0], [], []
    with open(LINKS_FILE) as f:
        f.readline(); f.readline()
        for line in f:
            p = line.split()
            if len(p) < 8:
                continue
            cum.append(cum[-1] + float(p[3]))
            spd.append(float(p[4])); grade.append(float(p[6]))
    return np.array(cum), np.array(spd), np.array(grade)


def load_training(trip="forward"):
    """Parse a training log into per-epoch rewards and per-episode outcomes.

    Two line kinds are emitted during training:
      Epoch #12: test_reward: -2401.3 ... best_reward: -2323.0
      [✓ ARRIVED]  ep=  87   6,203 steps   74,891m (100.0%)  energy=  819.7 kWh ...
    Episodes interleave across the 8 parallel workers, so they are plotted in log
    order (a training-progress axis, not a per-worker episode index).
    """
    pattern = "train_return_*.log" if trip == "return" else "train_run_*.log"
    logs = sorted(glob.glob(os.path.join(_REPO, "logs", pattern)),
                  key=os.path.getmtime)
    if not logs:
        return None

    ep_re = re.compile(r"Epoch #(\d+):.*?test_reward:\s*(-?[\d.]+).*?best_reward:\s*(-?[\d.]+)")
    ce_re = re.compile(
        r"\[(✓ ARRIVED|✗ TIMEOUT)\]\s+ep=\s*(\d+)\s+([\d,]+) steps.*?"
        r"energy=\s*([\d.]+) kWh\s+reward=\s*([-+]?[\d.]+)")

    epochs, test_r, best_r = [], [], []
    arrived, steps, energy, ep_reward = [], [], [], []
    with open(logs[-1], errors="replace") as f:
        for line in f:
            m = ep_re.search(line)
            if m:
                epochs.append(int(m.group(1)))
                test_r.append(float(m.group(2)))
                best_r.append(float(m.group(3)))
                continue
            m = ce_re.search(line)
            if m:
                arrived.append(m.group(1).endswith("ARRIVED"))
                steps.append(float(m.group(3).replace(",", "")))
                energy.append(float(m.group(4)))
                ep_reward.append(float(m.group(5)))
    if not epochs and not steps:
        return None
    return {
        "log": os.path.basename(logs[-1]),
        "epoch": np.array(epochs), "test": np.array(test_r), "best": np.array(best_r),
        "arrived": np.array(arrived, bool), "steps": np.array(steps),
        "energy": np.array(energy), "ep_reward": np.array(ep_reward),
    }


# ── 1. did training work? ────────────────────────────────────────────────────

def fig_training_progress(trip="forward"):
    d = load_training(trip)
    if d is None:
        return None
    base = load_baselines(trip)

    fig, axes = plt.subplots(2, 2, figsize=(COL2, 4.6))
    (ax_r, ax_e), (ax_t, ax_a) = axes

    # (a) reward per epoch — the optimiser's own view of progress
    if len(d["epoch"]):
        ax_r.plot(d["epoch"], d["test"], color=SKY, lw=0.7, alpha=0.9,
                  label="test reward")
        ax_r.plot(d["epoch"], d["best"], color=BLUE, lw=1.4, label="running best")
        ax_r.set_xlabel("epoch")
        ax_r.legend(loc="lower right")
    ax_r.set_ylabel("episode reward")
    ax_r.set_title("Reward improves", pad=5)
    panel(ax_r, "(a)")

    n = np.arange(len(d["energy"]))
    w = max(5, len(n) // 40)

    # (b) trip energy per training episode — the actual objective
    if len(n):
        ax_e.plot(n, d["energy"], color=ORANGE, lw=0.35, alpha=0.35)
        ax_e.plot(n, rolling(d["energy"], w), color=ORANGE, lw=1.4,
                  label=f"rolling mean ({w})")
        if base is not None:
            eco = int(np.argmin(np.where(base["steps"] <= DEADLINE_STEPS,
                                         base["energy"], np.inf)))
            ax_e.axhline(base["energy"][eco], color=GREY, lw=0.8, ls=(0, (4, 2)),
                         label=f"best on-time constant notch (N{base['notch'][eco]})")
        ax_e.legend(loc="upper right")
    ax_e.set_xlabel("training episode")
    ax_e.set_ylabel("trip energy (kWh)")
    ax_e.set_title("Energy falls toward the frontier", pad=5)
    panel(ax_e, "(b)")

    # (c) trip time against the deadline — the constraint
    if len(n):
        ax_t.plot(n, d["steps"], color=BLUE, lw=0.35, alpha=0.35)
        ax_t.plot(n, rolling(d["steps"], w), color=BLUE, lw=1.4)
        ax_t.axhline(DEADLINE_STEPS, color=VERM, lw=0.9, ls=(0, (4, 2)))
        ax_t.annotate("deadline", (0.99, DEADLINE_STEPS),
                      xycoords=("axes fraction", "data"), xytext=(0, 3),
                      textcoords="offset points", ha="right", va="bottom",
                      fontsize=6.5, color=VERM)
    ax_t.set_xlabel("training episode")
    ax_t.set_ylabel("trip time (s)")
    ax_t.set_title("Trip time settles under the deadline", pad=5)
    panel(ax_t, "(c)")

    # (d) arrival rate — did it ever stop timing out?
    if len(n):
        ax_a.plot(n, 100 * rolling(d["arrived"].astype(float), w),
                  color=GREEN, lw=1.4)
        ax_a.set_ylim(-4, 104)
    ax_a.set_xlabel("training episode")
    ax_a.set_ylabel("arrivals (%, rolling)")
    ax_a.set_title("Episodes reach the terminus", pad=5)
    panel(ax_a, "(d)")

    fig.suptitle(f"Training progress — {d['log']}", fontsize=9.5, y=1.005)
    fig.tight_layout(w_pad=2.2, h_pad=1.8)
    return save(fig, f"fig_training_progress_{trip}")


# ── 2. is the policy any good? ───────────────────────────────────────────────

def fig_energy_time_tradeoff():
    trips = [(t, load_baselines(t)) for t in ("forward", "return")]
    trips = [(t, b) for t, b in trips if b is not None]
    if not trips:
        return None

    fig, axes = plt.subplots(1, len(trips), figsize=(COL2, 2.9))
    axes = np.atleast_1d(axes)
    titles = {"forward": "Toshkent $\\rightarrow$ Ho'jakent (net $+347$ m)",
              "return":  "Ho'jakent $\\rightarrow$ Toshkent (net $-347$ m)"}

    for ax, (trip, base), lab in zip(axes, trips, "ab"):
        order = np.argsort(base["steps"])
        s, e, n = base["steps"][order], base["energy"][order], base["notch"][order]

        ax.axvspan(DEADLINE_STEPS, max(s.max(), DEADLINE_STEPS) * 1.06,
                   color=VERM, alpha=0.06, lw=0)
        ax.axvline(DEADLINE_STEPS, color=VERM, lw=0.8, ls=(0, (4, 2)))
        ax.annotate("deadline", (DEADLINE_STEPS, 0.02),
                    xycoords=("data", "axes fraction"), xytext=(-3, 0),
                    textcoords="offset points", rotation=90, ha="right",
                    va="bottom", fontsize=6.5, color=VERM)

        ax.plot(s, e, "-", color=GREY, lw=1.0, zorder=2,
                label="constant notch")
        ax.scatter(s, e, s=26, color="white", edgecolors=INK, linewidths=0.9,
                   zorder=3)
        # alternate the label side: the frontier bunches at the fast end
        for i, (ni, si, ei) in enumerate(zip(n, s, e)):
            dx, ha = (-7, "right") if i % 2 else (7, "left")
            ax.annotate(f"N{ni}", (si, ei), textcoords="offset points",
                        xytext=(dx, 1), ha=ha, va="center", fontsize=6.5,
                        color=GREY)

        prof = load_profile(trip)
        if prof is not None and "energy" in prof and "time" in prof:
            ax.scatter([prof["time"][-1]], [prof["energy"][-1]], s=44,
                       color=BLUE, marker="o", zorder=5, edgecolors="white",
                       linewidths=0.9,
                       label=f"RL policy ({prof['energy'][-1]:.0f} kWh, "
                             f"{prof['time'][-1]:,.0f} s)")
        ax.legend(loc="upper right")
        ax.set_xlabel("trip time (s)")
        if lab == "a":
            ax.set_ylabel("total trip energy (kWh)")
        ax.set_title(titles.get(trip, trip), pad=6)
        panel(ax, f"({lab})")

    fig.tight_layout(w_pad=2.0)
    return save(fig, "fig_energy_time_tradeoff")


# ── 3. what does the policy do? ──────────────────────────────────────────────

def fig_speed_profile(trip="forward"):
    prof = load_profile(trip)
    if prof is None:
        return None
    cum, lim, grade = load_links()
    km = prof["pos"] / 1000.0
    cum_km = cum[1:] / 1000.0

    fig, axes = plt.subplots(3, 1, figsize=(COL2, 4.4), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1.0, 1.0],
                                          "hspace": 0.12})

    axes[0].step(cum_km, lim, where="post", color=VERM, lw=0.8, ls=(0, (4, 2)),
                 label="speed limit")
    axes[0].plot(km, prof["speed"], color=BLUE, lw=0.9, label="train speed")
    axes[0].set_ylabel("speed (m s$^{-1}$)")
    axes[0].set_ylim(0, max(lim.max(), prof["speed"].max()) * 1.12)
    axes[0].legend(loc="lower right", ncol=2)
    panel(axes[0], "(a)")

    # Raw sampled notch is intentionally noisy (the deployable policy is
    # stochastic); the rolling median exposes the terrain-driven trend.
    axes[1].step(km, prof["notch"], where="post", color=PURPLE, lw=0.5, alpha=0.45)
    axes[1].fill_between(km, prof["notch"], step="post", color=PURPLE,
                         alpha=0.12, lw=0)
    w = 15
    if len(prof["notch"]) >= w:
        pad = w // 2
        padded = np.pad(prof["notch"].astype(float), pad, mode="edge")
        trend = np.array([np.median(padded[i:i + w])
                          for i in range(len(prof["notch"]))])
        axes[1].plot(km, trend, color=PURPLE, lw=1.3,
                     label=f"rolling median ({w} decisions)")
        axes[1].legend(loc="upper right")
    axes[1].set_ylabel("notch")
    axes[1].set_ylim(-0.4, 8.4)
    axes[1].yaxis.set_major_locator(MultipleLocator(4))
    panel(axes[1], "(b)")

    axes[2].fill_between(cum_km, grade, step="mid", color=GREY, alpha=0.5, lw=0)
    axes[2].axhline(0, color=INK, lw=0.5)
    axes[2].set_ylabel("grade (%)")
    axes[2].set_xlabel("distance along route (km)")
    axes[2].set_xlim(0, ROUTE_KM)
    panel(axes[2], "(c)")

    fig.align_ylabels(axes)
    return save(fig, f"fig_speed_profile_{trip}")


# ── 4. where does the energy go? ─────────────────────────────────────────────

def fig_energy_usage(trip="forward"):
    prof = load_profile(trip)
    if prof is None or "energy" not in prof:
        return None
    cum, _lim, grade = load_links()
    km = prof["pos"] / 1000.0
    e = prof["energy"]

    fig, axes = plt.subplots(2, 1, figsize=(COL2, 3.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 1.0],
                                          "hspace": 0.14})

    # (a) cumulative energy, against the constant-rate line. Deviation from the
    # straight line IS the terrain story: above it the policy is paying for a
    # climb, below it it is coasting.
    axes[0].plot(km, e, color=ORANGE, lw=1.4, label="cumulative energy")
    axes[0].fill_between(km, e, color=ORANGE, alpha=0.12, lw=0)
    axes[0].plot([0, km[-1]], [0, e[-1]], color=GREY, lw=0.8, ls=(0, (4, 2)),
                 label=f"uniform rate ({e[-1] / km[-1]:.1f} kWh km$^{{-1}}$)")
    axes[0].set_ylabel("energy (kWh)")
    axes[0].legend(loc="upper left")
    axes[0].set_title(f"Total {e[-1]:.0f} kWh over {km[-1]:.1f} km", pad=5)
    panel(axes[0], "(a)")

    # (b) energy per km against grade — the mechanism behind panel (a)
    bins = np.arange(0, ROUTE_KM + 1.0, 1.0)
    idx = np.clip(np.digitize(km, bins) - 1, 0, len(bins) - 2)
    per_km = np.full(len(bins) - 1, np.nan)
    for b in range(len(bins) - 1):
        sel = np.where(idx == b)[0]
        if len(sel):
            lo = e[sel[0] - 1] if sel[0] > 0 else 0.0
            per_km[b] = e[sel[-1]] - lo
    centres = bins[:-1] + 0.5
    axes[1].bar(centres, per_km, width=0.9, color=ORANGE, alpha=0.75, lw=0,
                label="energy per km")
    axes[1].set_ylabel("kWh km$^{-1}$", color=ORANGE)
    axes[1].tick_params(axis="y", colors=ORANGE)

    ax_g = axes[1].twinx()
    ax_g.grid(False)
    cum_km = cum[1:] / 1000.0
    grade_km = np.array([grade[(cum_km >= b) & (cum_km < b + 1)].mean()
                         if np.any((cum_km >= b) & (cum_km < b + 1)) else np.nan
                         for b in bins[:-1]])
    ax_g.plot(centres, grade_km, color=INK, lw=1.0, label="mean grade")
    ax_g.axhline(0, color=INK, lw=0.4, alpha=0.5)
    ax_g.set_ylabel("mean grade (%)")
    ax_g.spines["right"].set_visible(True)

    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = ax_g.get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, loc="upper left", ncol=2)
    axes[1].set_xlabel("distance along route (km)")
    axes[1].set_xlim(0, ROUTE_KM)
    panel(axes[1], "(b)")

    return save(fig, f"fig_energy_usage_{trip}")


FIGURES = [
    ("training progress (forward)", lambda: fig_training_progress("forward"),
     "python rl/train.py"),
    ("training progress (return)", lambda: fig_training_progress("return"),
     "python rl/train_return.py"),
    ("energy-time frontier", fig_energy_time_tradeoff,
     "python rl/run_baselines.py"),
    ("speed profile (forward)", lambda: fig_speed_profile("forward"),
     "python rl/evaluate.py --stochastic"),
    ("speed profile (return)", lambda: fig_speed_profile("return"),
     "python rl/evaluate.py --stochastic --return-trip"),
    ("energy usage (forward)", lambda: fig_energy_usage("forward"),
     "python rl/evaluate.py --stochastic"),
    ("energy usage (return)", lambda: fig_energy_usage("return"),
     "python rl/evaluate.py --stochastic --return-trip"),
]


def main():
    print(f"writing figures to {os.path.relpath(OUTDIR, _REPO)}/\n")
    made = 0
    for label, fn, howto in FIGURES:
        out = fn()
        if out:
            made += 1
            print(f"  ✓ {label:<28} {os.path.basename(out)} (+ .pdf)")
        else:
            print(f"  – {label:<28} no data — run: {howto}")
    print(f"\n{made}/{len(FIGURES)} figures written.")


if __name__ == "__main__":
    main()
