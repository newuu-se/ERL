"""
Closed-form reward analysis of constant-target policies — the theoretical
justification for the reward coefficients.

A constant-target policy ignores the reward, so its trajectory x_t, and hence its
episode return under Eq. (reward), is fixed and computable. Writing the return of
constant target n as a function of the pace weight w_P:

    R(n; w_P) = -W_ENERGY * E(n)            (total energy, gamma-discounted)
                - w_P     * Sigma_beta(n)   (behind-schedule fraction, discounted)
                - W_OVERSPEED * OS(n)        (~0, the sim caps speed)
                + ARRIVAL_BONUS  if the train reaches the terminus
                - TIMEOUT_PENALTY otherwise

This is LINEAR in w_P: R(n; w_P) = a(n) - w_P * Sigma_beta(n). The reward-optimal
constant target is therefore argmax_n R(n; w_P), and there is a threshold w_P* at
which the schedule-feasible eco target (the slowest target that still arrives by
the deadline) overtakes the cheaper-but-late target below it. Computing w_P* tells us,
before any training, the coefficient range for which the reward's optimum is the
operating point we actually want.

We log each constant target's per-second trajectory through the interactive
simulator and reconstruct the return with the exact env reward constants at the
resolution PPO actually sees: per-second components are summed into
CONTROL_INTERVAL-second decisions and each decision is weighted by gamma**k
(DISCOUNT, read from rl.train so the two cannot drift). A constant-target policy
has no action changes, so the smoothness term is identically zero. Aggregating and
discounting this way matters: PPO sees one discounted return per decision, so an
undiscounted per-second sum would report a w_P* that is not the return it optimises.

Usage:
    python rl/reward_theory.py            # both trips → results/theory/ + charts
    python rl/reward_theory.py --replot   # redraw the charts from results/theory/

Charts: results/plots/reward_theory_forward.png and reward_theory_return.png (+ .pdf):
the constant target the reward prefers at each pace weight, against the targets that
arrive before the deadline, with the current W_PACE marked.
"""

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.train import DISCOUNT
from rl.train_env import (
    NeTrainSimEnv,
    TRAINS_FILE,
    TRAINS_FILE_RETURN,
    TOTAL_ROUTE_LENGTH_M,
    DEADLINE_STEPS,
    CONTROL_INTERVAL,
    W_ENERGY,
    W_OVERSPEED,
    W_PACE,
    ARRIVAL_BONUS,
    TIMEOUT_PENALTY,
    N_ACTIONS,
    TARGET_FRACTIONS,
    action_label,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(_REPO, "results", "theory")
ACTIONS = list(range(1, N_ACTIONS))  # 0 = coast, which never leaves the start


def log_constant_action(action: int, trains_file: str) -> dict:
    """Drive one constant-target trip at 1-second resolution; return the per-step
    energy and behind-schedule-fraction sums plus trip summary."""
    env = NeTrainSimEnv(trains_file=trains_file)
    env.reset()
    sum_E = float(env._cum_energy_kwh)  # bootstrap step-0 energy
    sum_beta = 0.0
    sum_overspeed = 0.0

    # The same totals again, but bucketed into CONTROL_INTERVAL-second PPO
    # decisions and weighted by gamma**k — this is the return PPO maximises.
    disc_E = disc_beta = disc_overspeed = 0.0
    disc = 1.0  # gamma**k for the current decision
    disc_terminal = 1.0  # weight of the arrival/timeout term
    acc_E = acc_beta = acc_overspeed = 0.0
    secs_in_decision = 0

    terminated = truncated = False
    while not (terminated or truncated):
        # Call the 1-second core directly to get per-second resolution (step()
        # would repeat the action CONTROL_INTERVAL times before returning).
        _, terminated, truncated, e_t = env._sim_step(action)
        st = env._last_state
        pos = float(st["position_m"])
        target = (env._step_count / DEADLINE_STEPS) * TOTAL_ROUTE_LENGTH_M
        beta_t = max(0.0, target - pos) / TOTAL_ROUTE_LENGTH_M
        os_t = max(0.0, float(st["speed_mps"]) - float(st["link_max_speed_mps"]))
        sum_E += e_t
        sum_beta += beta_t
        sum_overspeed += os_t
        acc_E += e_t
        acc_beta += beta_t
        acc_overspeed += os_t
        secs_in_decision += 1
        # A decision ends every CONTROL_INTERVAL seconds, or when the episode
        # ends mid-decision (the terminal step still counts as a decision).
        if secs_in_decision == CONTROL_INTERVAL or terminated or truncated:
            disc_E += disc * acc_E
            disc_beta += disc * acc_beta
            disc_overspeed += disc * acc_overspeed
            disc_terminal = disc
            disc *= DISCOUNT
            acc_E = acc_beta = acc_overspeed = 0.0
            secs_in_decision = 0
    steps = env._step_count
    env.close()
    return {
        "action": action,
        "target_frac": TARGET_FRACTIONS[action],
        "steps": steps,
        "energy_kwh": round(sum_E, 3),
        "sum_beta": round(sum_beta, 4),
        "sum_overspeed": round(sum_overspeed, 4),
        "disc_energy_kwh": round(disc_E, 3),
        "disc_sum_beta": round(disc_beta, 4),
        "disc_sum_overspeed": round(disc_overspeed, 4),
        "disc_terminal": round(disc_terminal, 6),
        "arrived": bool(terminated),
        "on_schedule": bool(terminated and steps <= DEADLINE_STEPS),
    }


def reward_of(rec: dict, w_p: float) -> float:
    """Discounted episode return of a constant-target policy, in the units PPO sees."""
    term = ARRIVAL_BONUS if rec["arrived"] else -TIMEOUT_PENALTY
    return (
        -W_ENERGY * rec["disc_energy_kwh"]
        - w_p * rec["disc_sum_beta"]
        - W_OVERSPEED * rec["disc_sum_overspeed"]
        + rec["disc_terminal"] * term
    )


def analyze_trip(trip: str):
    trains_file = TRAINS_FILE_RETURN if trip == "return" else TRAINS_FILE
    print(f"\n=== {trip.upper()} TRIP — logging constant-target trajectories ===")
    recs = []
    for a in ACTIONS:
        r = log_constant_action(a, trains_file)
        recs.append(r)
        print(
            f"  {action_label(a):>5}: {r['steps']:>5d} s  {r['energy_kwh']:>7.1f} kWh  "
            f"sum_beta={r['disc_sum_beta']:>7.3f} (disc)  on_schedule={r['on_schedule']}"
        )

    # eco target = slowest ON-SCHEDULE target (lowest energy that still arrives by deadline)
    on_sched = [r for r in recs if r["on_schedule"]]
    eco = min(on_sched, key=lambda r: r["action"])  # slowest target that is on-schedule
    # the cheaper-but-late target just below it (the basin the policy mode falls into)
    below = [r for r in recs if r["action"] < eco["action"]]
    rival = max(below, key=lambda r: r["action"]) if below else None

    # threshold w_P* where R(eco) overtakes R(rival): linear in w_P
    # R(eco;w) - R(rival;w) = (a_eco - a_rival) - w*(beta_eco - beta_rival) >= 0
    wp_star = None
    if rival is not None:
        a_eco = reward_of(eco, 0.0)
        a_riv = reward_of(rival, 0.0)
        d_beta = (
            eco["disc_sum_beta"] - rival["disc_sum_beta"]
        )  # eco lags less → negative
        if d_beta < 0:
            wp_star = (a_riv - a_eco) / (-d_beta)  # > this, eco wins

    # reward-optimal action as a function of w_P, over a grid
    wp_grid = np.linspace(0, 10, 1001)
    argmax_action = [
        max(recs, key=lambda r: reward_of(r, w))["action"] for w in wp_grid
    ]

    os.makedirs(OUTDIR, exist_ok=True)
    with open(
        os.path.join(OUTDIR, f"constant_target_{trip}.csv"), "w", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)

    result = {
        "trip": trip,
        "eco_action": eco["action"],
        "rival_action": (rival["action"] if rival else None),
        "wp_star": (round(wp_star, 3) if wp_star is not None else None),
        "records": recs,
        "wp_grid": wp_grid.tolist(),
        "argmax_action": argmax_action,
    }
    print(
        f"  eco target = {action_label(eco['action'])} ({eco['energy_kwh']:.1f} kWh, "
        f"{eco['steps']} s);  rival below = {action_label(rival['action']) if rival else None}"
    )
    print(
        f"  --> theoretical threshold  w_P* = {result['wp_star']}  "
        f"(eco target is reward-optimal for w_P >= w_P*)"
    )
    return result


def plot(results: dict):
    """One chart per trip: the constant target the reward prefers at each pace
    weight, against the targets that arrive before the deadline.

    Writes results/plots/reward_theory_<trip>.png (+ .pdf)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, grey, green, ink = "#2a78d6", "#8a8984", "#1baf7a", "#1b1b19"
    trip_names = {"forward": "Forward trip A→B", "return": "Return trip B→A"}
    outdir = os.path.join(_REPO, "results", "plots")
    os.makedirs(outdir, exist_ok=True)

    for trip, res in results.items():
        recs = {r["action"]: r for r in res["records"]}
        wp = np.array(res["wp_grid"])
        best = np.array(res["argmax_action"])
        on_time = sorted(a for a, r in recs.items() if r["on_schedule"])
        eco = res["eco_action"]
        now = int(best[np.argmin(np.abs(wp - W_PACE))])  # what the current weight picks

        # the finding, for the title: the pace-weight range where the eco target wins
        eco_ws = wp[best == eco]
        if not len(eco_ws):
            finding = "the reward never picks an on-time target"
        elif eco_ws.max() < wp.max():
            finding = (
                f"the reward picks the on-time {action_label(eco)} target "
                f"for W_PACE {res['wp_star']:.2f} to {eco_ws.max():.2f}"
            )
        else:
            finding = (
                f"the reward picks the on-time {action_label(eco)} target "
                f"only for W_PACE ≥ {res['wp_star']:.2f}"
            )

        fig, ax = plt.subplots(figsize=(7.0, 3.4))
        ax.axhspan(
            min(on_time) - 0.45,
            max(on_time) + 0.45,
            color=green,
            alpha=0.12,
            lw=0,
            label=f"targets that arrive before the {DEADLINE_STEPS:,} s deadline",
        )
        ax.step(wp, best, where="post", color=blue, lw=2)
        ax.legend(loc="lower right", frameon=False, fontsize=8.5)
        ax.axvline(W_PACE, color=grey, ls="--", lw=1.2)
        ax.plot([W_PACE], [now], "o", color=blue, ms=7, zorder=3)
        status = "on time" if recs[now]["on_schedule"] else "late"
        ax.annotate(
            f"now: W_PACE = {W_PACE:g} → {action_label(now)} ({status})",
            (W_PACE, now),
            xytext=(8, -30),
            textcoords="offset points",
            fontsize=8.5,
            color=ink,
        )
        ax.set_yticks(sorted(recs))
        ax.set_yticklabels([action_label(a) for a in sorted(recs)])
        ax.set_ylim(min(recs) - 0.5, max(recs) + 0.5)
        ax.set_xlim(0, wp.max())
        ax.set_xlabel(
            "pace weight W_PACE (penalty for falling behind the deadline pace)"
        )
        ax.set_ylabel("constant target the\nreward prefers")
        ax.set_title(
            f"{trip_names[trip]}: {finding}",
            loc="left",
            fontsize=10,
            fontweight="bold",
            pad=8,
        )
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", color="#e4e3df", lw=0.6)
        name = os.path.join(outdir, f"reward_theory_{trip}")
        fig.savefig(name + ".png", dpi=200, bbox_inches="tight")
        fig.savefig(name + ".pdf", bbox_inches="tight")
        plt.close(fig)


def main():
    cache = os.path.join(OUTDIR, "reward_theory.json")
    if "--replot" in sys.argv and os.path.isfile(cache):
        with open(cache) as f:
            results = json.load(f)  # reuse logged trajectories, just redraw
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
        print(
            f"{t:>8}: eco target {action_label(r['eco_action'])}, "
            f"rival {action_label(r['rival_action']) if r['rival_action'] is not None else None}, "
            f"w_P* = {r['wp_star']}  (current w_P=2)"
        )
    print(f"\nWrote {OUTDIR}/ and results/plots/reward_theory_<trip>.png")


if __name__ == "__main__":
    main()
