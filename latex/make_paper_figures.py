#!/usr/bin/env python3
"""
Generate the paper-specific figures into latex/figures/.

Self-contained: READS project files (data/, results/, logs/) but writes ONLY
inside latex/figures/. Deleting the latex/ folder removes this script and its
outputs without touching the project. Re-run from the repo root or from latex/:

    python latex/make_paper_figures.py
"""
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
FIGDIR = os.path.join(_HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

RAW_LINKS = os.path.join(_REPO, "data", "netrainsim_v2", "linksFile_v2_fixed_speed.dat")
CLEAN_LINKS = os.path.join(_REPO, "data", "netrainsim_v2", "linksFile_v2_clean.dat")
RET_PROFILE = os.path.join(_REPO, "results", "return", "notch_profile_stochastic.csv")
# representative campaign runs (100-epoch, w_P=2, seed 0) for the training curves
FWD_LOG = os.path.join(_REPO, "logs", "campaign", "sweep_forward_wp2_s0.log")
RET_LOG = os.path.join(_REPO, "logs", "campaign", "sweep_return_wp2_s0.log")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#fbfbfd",
    "axes.grid": True, "grid.color": "#e3e3ea", "font.size": 10.5,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})
BLUE, RED, ORANGE, GRAY = "#2563eb", "#dc2626", "#f59e0b", "#6b7280"


def read_links(path):
    cum, spd, grd = [0.0], [], []
    with open(path) as f:
        f.readline(); f.readline()
        for line in f:
            p = line.split()
            if len(p) >= 8:
                cum.append(cum[-1] + float(p[3]))
                spd.append(float(p[4])); grd.append(float(p[6]))
    return np.array(cum), np.array(spd), np.array(grd)


def elevation(cum, grd):
    e = [0.0]
    for i in range(len(grd)):
        e.append(e[-1] + grd[i] / 100.0 * (cum[i + 1] - cum[i]))
    return np.array(e)


def fig_data_cleaning():
    cum_r, _, grd_r = read_links(RAW_LINKS)
    cum_c, _, grd_c = read_links(CLEAN_LINKS)
    el_r, el_c = elevation(cum_r, grd_r), elevation(cum_c, grd_c)
    fig, ax = plt.subplots(2, 1, figsize=(8.6, 5.6), sharex=True)
    ax[0].plot(cum_r / 1000, el_r, color=RED, lw=1.0, alpha=0.85, label="raw (DEM)")
    ax[0].plot(cum_c / 1000, el_c, color=BLUE, lw=1.4, label="cleaned")
    ax[0].set_ylabel("relative elevation (m)")
    ax[0].legend(loc="upper left", fontsize=9.5)
    ax[0].set_title("Elevation profile: raw vs. cleaned (net climb 346.6 m preserved)")
    ax[1].plot(cum_r[1:] / 1000, grd_r, color=RED, lw=0.7, alpha=0.8, label="raw grade")
    ax[1].plot(cum_c[1:] / 1000, grd_c, color=BLUE, lw=1.0, label="cleaned grade")
    ax[1].set_ylabel("grade (%)"); ax[1].set_xlabel("distance from Toshkent (km)")
    ax[1].legend(loc="upper left", fontsize=9.5)
    p = os.path.join(FIGDIR, "fig_data_cleaning.png")
    fig.tight_layout(); fig.savefig(p); plt.close(fig); return p


# Measured constant-notch frontiers (rl/run_baselines.py, clean data, 2026-06-11)
FWD = {"n": [8, 6, 5, 4, 3, 2], "t": [5602, 5652, 5721, 5910, 6442, 8492],
       "e": [862.41, 853.46, 849.10, 834.37, 788.63, 758.39]}
RET = {"n": [8, 6, 5, 4, 3, 2, 1], "t": [5605, 5641, 5682, 5768, 5950, 6375, 7681],
       "e": [348.34, 335.79, 327.37, 317.67, 293.93, 242.22, 170.06]}
DEADLINE = 6500


def fig_tradeoff_return():
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.plot(RET["t"], RET["e"], "-", color=BLUE, lw=2, alpha=0.6, zorder=2)
    sc = ax.scatter(RET["t"], RET["e"], c=RET["n"], cmap="viridis", s=150,
                    zorder=3, edgecolors="white", linewidths=1.4)
    for n, t, e in zip(RET["n"], RET["t"], RET["e"]):
        ax.annotate(f"N{n}", (t, e), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=9.5, fontweight="bold", color="#333")
    fig.colorbar(sc, ax=ax, pad=0.015).set_label("throttle notch")
    ax.axvline(DEADLINE, color=GRAY, ls=":", lw=1.5,
               label="schedule deadline (6,500 s)")
    # RL policy: 5-seed mean ± 95% CI at the chosen w_P=2 (aggregate_campaign.py)
    ax.errorbar([6523], [221.2], xerr=[143], yerr=[21.9], color=ORANGE, marker="*",
                markersize=18, zorder=5, capsize=5, markeredgecolor="white",
                markeredgewidth=1.4, elinewidth=1.6,
                label="RL, $w_P{=}2$ (5-seed mean $\\pm$ 95% CI): 221 kWh, 6,523 s")
    ax.set_xlabel("trip time (s)"); ax.set_ylabel("total trip energy (kWh)")
    ax.set_title("Return trip (Ho'jakent$\\rightarrow$Toshkent): frontier and RL outcomes")
    ax.legend(loc="upper right", fontsize=9)
    p = os.path.join(FIGDIR, "fig_tradeoff_return.png")
    fig.savefig(p); plt.close(fig); return p


def fig_profile_return():
    cum, spd, grd = read_links(CLEAN_LINKS)
    L = cum[-1]
    # reverse to return-direction chainage (0 = Ho'jakent), grade sign flips
    rev_bounds = L - cum[::-1]
    rev_spd = spd[::-1]
    rev_grd = -grd[::-1]
    pos, speed, notch = [], [], []
    with open(RET_PROFILE) as f:
        for row in csv.DictReader(f):
            pos.append(float(row["position_m"]))
            speed.append(float(row["speed_mps"]))
            notch.append(int(row["notch"]))
    pos = np.array(pos) / 1000
    fig, ax = plt.subplots(3, 1, figsize=(9.5, 7), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1.2, 1.2]})
    ax[0].plot(pos, speed, color=BLUE, lw=1, label="train speed")
    ax[0].plot(rev_bounds[1:] / 1000, rev_spd, color=RED, lw=1, alpha=.7,
               ls="--", label="speed limit")
    ax[0].set_ylabel("speed (m/s)"); ax[0].legend(loc="upper right", fontsize=9)
    ax[0].set_title("RL policy (sampled) — return trip (169.6 kWh, 7,076 s)")
    ax[1].step(pos, notch, where="post", color=ORANGE, lw=1)
    ax[1].set_ylabel("notch"); ax[1].set_ylim(-0.5, 8.5)
    ax[2].fill_between(rev_bounds[1:] / 1000, rev_grd, color=GRAY, alpha=.5, step="mid")
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_ylabel("grade (%)"); ax[2].set_xlabel("distance from Ho'jakent (km)")
    p = os.path.join(FIGDIR, "fig_profile_return.png")
    fig.tight_layout(); fig.savefig(p); plt.close(fig); return p


def parse_log(path):
    e, t, b = [], [], []
    pat = re.compile(r"Epoch #(\d+): test_reward: (-?[0-9.]+).*best_reward: (-?[0-9.]+)")
    for line in open(path):
        m = pat.search(line)
        if m:
            e.append(int(m.group(1))); t.append(float(m.group(2))); b.append(float(m.group(3)))
    return np.array(e), np.array(t), np.array(b)


def fig_training_curves():
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for a, log, title in ((ax[0], FWD_LOG, "Forward trip (100 epochs, seed 0)"),
                          (ax[1], RET_LOG, "Return trip (100 epochs, seed 0)")):
        ep, te, be = parse_log(log)
        a.plot(ep, te, color=BLUE, lw=0.9, alpha=0.5, label="test reward")
        a.plot(ep, be, color="#059669", lw=2, label="best reward")
        a.axvline(70, color=GRAY, ls=":", lw=1.2)
        a.annotate("entropy $\\rightarrow$ 0", (70, a.get_ylim()[0]), fontsize=8.5,
                   color=GRAY, xytext=(4, 8), textcoords="offset points")
        a.set_xlabel("epoch"); a.set_title(title, fontsize=11)
        a.legend(fontsize=9, loc="lower right")
    ax[0].set_ylabel("episode reward")
    p = os.path.join(FIGDIR, "fig_training_curves.png")
    fig.tight_layout(); fig.savefig(p); plt.close(fig); return p


if __name__ == "__main__":
    for fn in (fig_data_cleaning, fig_tradeoff_return, fig_profile_return,
               fig_training_curves):
        print("wrote", fn())
