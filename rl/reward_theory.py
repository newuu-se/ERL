"""
Closed-form reward analysis of constant-notch policies — the theoretical
justification for the reward coefficients (addresses reviewer comment #1).

A constant-notch policy ignores the reward, so its trajectory x_t, and hence its
episode return under Eq. (reward), is fixed and computable. Writing the return of
constant notch n as a function of the pace weight w_P:

    R(n; w_P) = -W_ENERGY * E(n)            (measured total energy)
                - w_P     * Sigma_beta(n)   (integrated behind-schedule fraction)
                - W_OVERSPEED * OS(n)        (~0, the sim caps speed)
                + ARRIVAL_BONUS  if the train reaches the terminus
                - TIMEOUT_PENALTY otherwise

This is LINEAR in w_P: R(n; w_P) = a(n) - w_P * Sigma_beta(n). The reward-optimal
constant notch is therefore argmax_n R(n; w_P), and there is a threshold w_P* at
which the schedule-feasible eco notch (the slowest notch that still arrives by the
deadline) overtakes the cheaper-but-late notch below it. Computing w_P* tells us,
before any training, the coefficient range for which the reward's optimum is the
operating point we actually want.

We log each constant notch's per-second trajectory through the interactive
simulator (one sim-second per step, no action repeat) and reconstruct the return
with the exact env reward constants.

Usage:
    python rl/reward_theory.py            # both trips → results/theory/ + figure
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.train_env import (
    NeTrainSimEnv, TRAINS_FILE, TRAINS_FILE_RETURN,
    TOTAL_ROUTE_LENGTH_M, DEADLINE_STEPS,
    W_ENERGY, W_OVERSPEED, ARRIVAL_BONUS, TIMEOUT_PENALTY,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(_REPO, "results", "theory")
NOTCHES = [1, 2, 3, 4, 5, 6, 8]   # 0 does not move; 7 omitted to match the baseline table


def log_constant_notch(notch: int, trains_file: str) -> dict:
    """Drive one constant-notch trip at 1-second resolution; return the per-step
    energy and behind-schedule-fraction sums plus trip summary."""
    env = NeTrainSimEnv(trains_file=trains_file)
    env.reset()
    sum_E = float(env._cum_energy_kwh)        # bootstrap step-0 energy
    sum_beta = 0.0
    sum_overspeed = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        # Call the 1-second core directly to get per-second resolution (step()
        # would repeat the notch CONTROL_INTERVAL times before returning).
        _, terminated, truncated, e_t = env._sim_step(notch)
        st = env._last_state
        pos = float(st["position_m"])
        target = (env._step_count / DEADLINE_STEPS) * TOTAL_ROUTE_LENGTH_M
        sum_beta += max(0.0, target - pos) / TOTAL_ROUTE_LENGTH_M
        sum_overspeed += max(0.0, float(st["speed_mps"]) - float(st["link_max_speed_mps"]))
        sum_E += e_t
    steps = env._step_count
    env.close()
    return {
        "notch": notch, "steps": steps, "energy_kwh": round(sum_E, 3),
        "sum_beta": round(sum_beta, 4), "sum_overspeed": round(sum_overspeed, 4),
        "arrived": bool(terminated),
        "on_schedule": bool(terminated and steps <= DEADLINE_STEPS),
    }


def reward_of(rec: dict, w_p: float) -> float:
    term = ARRIVAL_BONUS if rec["arrived"] else -TIMEOUT_PENALTY
    return (-W_ENERGY * rec["energy_kwh"]
            - w_p * rec["sum_beta"]
            - W_OVERSPEED * rec["sum_overspeed"]
            + term)


def analyze_trip(trip: str):
    trains_file = TRAINS_FILE_RETURN if trip == "return" else TRAINS_FILE
    print(f"\n=== {trip.upper()} TRIP — logging constant-notch trajectories ===")
    recs = []
    for n in NOTCHES:
        r = log_constant_notch(n, trains_file)
        recs.append(r)
        print(f"  notch {n}: {r['steps']:>5d} s  {r['energy_kwh']:>7.1f} kWh  "
              f"sum_beta={r['sum_beta']:>7.3f}  on_schedule={r['on_schedule']}")

    # eco target = slowest ON-SCHEDULE notch (lowest energy that still arrives by deadline)
    on_sched = [r for r in recs if r["on_schedule"]]
    eco = min(on_sched, key=lambda r: r["notch"])      # lowest notch that is on-schedule
    # the cheaper-but-late notch just below it (the basin the policy mode falls into)
    below = [r for r in recs if r["notch"] < eco["notch"]]
    rival = max(below, key=lambda r: r["notch"]) if below else None

    # threshold w_P* where R(eco) overtakes R(rival): linear in w_P
    # R(eco;w) - R(rival;w) = (a_eco - a_rival) - w*(beta_eco - beta_rival) >= 0
    wp_star = None
    if rival is not None:
        a_eco = reward_of(eco, 0.0)
        a_riv = reward_of(rival, 0.0)
        d_beta = eco["sum_beta"] - rival["sum_beta"]      # eco lags less → negative
        if d_beta < 0:
            wp_star = (a_riv - a_eco) / (-d_beta)   # > this, eco wins

    # reward-optimal notch as a function of w_P, over a grid
    wp_grid = np.linspace(0, 10, 1001)
    argmax_notch = [max(recs, key=lambda r: reward_of(r, w))["notch"] for w in wp_grid]

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, f"constant_notch_{trip}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader(); w.writerows(recs)

    result = {
        "trip": trip, "eco_notch": eco["notch"],
        "rival_notch": (rival["notch"] if rival else None),
        "wp_star": (round(wp_star, 3) if wp_star is not None else None),
        "records": recs,
        "wp_grid": wp_grid.tolist(), "argmax_notch": argmax_notch,
    }
    print(f"  eco target = notch {eco['notch']} ({eco['energy_kwh']:.1f} kWh, "
          f"{eco['steps']} s);  rival below = notch {rival['notch'] if rival else None}")
    print(f"  --> theoretical threshold  w_P* = {result['wp_star']}  "
          f"(eco notch is reward-optimal for w_P >= w_P*)")
    return result


def plot(results: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wp_max = 8.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, trip in zip(axes, ["forward", "return"]):
        res = results[trip]
        recs = res["records"]
        wp = np.linspace(0, wp_max, 400)
        # y-window: focus on the schedule-feasible notches so the eco-vs-rival
        # crossover near w_P* is visible (the timeout notch runs off the bottom).
        on_sched = [r for r in recs if r["on_schedule"]]
        hi = max(reward_of(r, 0.0) for r in recs)
        lo = min(reward_of(r, wp_max) for r in on_sched)
        span = hi - lo
        for r in recs:
            ys = [reward_of(r, w) for w in wp]
            ax.plot(wp, ys, lw=1.8 if r["notch"] == res["eco_notch"] else 1.2,
                    label=f"N{r['notch']}" + ("" if r["arrived"] else " (timeout)"))
        ax.set_ylim(lo - 0.15 * span, hi + 0.10 * span)
        if res["wp_star"] is not None and res["wp_star"] <= wp_max:
            ax.axvline(res["wp_star"], color="k", ls=":", lw=1.5)
            ax.annotate(f"$w_P^*$={res['wp_star']:.2f}", (res["wp_star"], lo),
                        fontsize=9, xytext=(5, 14), textcoords="offset points")
        ax.axvline(2.0, color="red", ls="--", lw=1.2, alpha=0.8)
        ax.annotate("chosen $w_P{=}2$", (2.0, hi + 0.10 * span), color="red",
                    fontsize=8.5, va="top", ha="center", xytext=(0, -2),
                    textcoords="offset points")
        ax.set_title(f"{trip.capitalize()} trip — eco notch N{res['eco_notch']} "
                     f"optimal for $w_P\\geq{res['wp_star']:.2f}$", fontsize=10.5)
        ax.set_xlabel("pace weight $w_P$"); ax.set_ylabel("constant-notch return $R(n;w_P)$")
        ax.legend(fontsize=8, ncol=2, loc="lower left", framealpha=0.95)
        ax.grid(alpha=0.3)
    fig.suptitle("Closed-form return of constant-notch policies: the eco notch is "
                 "reward-optimal across the whole sensible $w_P$ range", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for d in (os.path.join(_REPO, "results", "plots"),
              os.path.join(_REPO, "latex", "figures")):
        if os.path.isdir(d):
            fig.savefig(os.path.join(d, "fig_reward_theory.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    cache = os.path.join(OUTDIR, "reward_theory.json")
    if "--replot" in sys.argv and os.path.isfile(cache):
        with open(cache) as f:
            results = json.load(f)            # reuse logged trajectories, just redraw
        plot(results)
        print(f"replotted from {cache}")
        return
    results = {t: analyze_trip(t) for t in ["forward", "return"]}
    os.makedirs(OUTDIR, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(results, f, indent=2)
    plot(results)
    print("\n=== SUMMARY ===")
    for t, r in results.items():
        print(f"{t:>8}: eco notch N{r['eco_notch']}, rival N{r['rival_notch']}, "
              f"w_P* = {r['wp_star']}  (current w_P=2)")
    print(f"\nWrote {OUTDIR}/ and fig_reward_theory.png")


if __name__ == "__main__":
    main()
