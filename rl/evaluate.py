"""
Evaluate a trained policy on one full trip and save what it did, second by second.

Writes two files (the return trip writes the same names under results/return/):

  results/action_profile[_stochastic].csv   one row per simulated second:
      time, position, x/y, speed, speed limit, the speed the driver followed,
      action (coast / target), the notch the simulator chose, grade, and the
      cumulative energy consumed (used by the RL), regenerated and net (reported only)
  results/eval_summary[_stochastic].json    trip totals: time, energy, arrival,
      deadline, reward, and how many seconds each action was used

rl/make_plots.py reads both.

Usage:
    source venv/bin/activate
    python rl/evaluate.py --stochastic                  # latest checkpoint, sampled actions
    python rl/evaluate.py                               # argmax actions
    python rl/evaluate.py --checkpoint checkpoints/policy_best.pth
    python rl/evaluate.py --stochastic --return-trip    # B→A trip, checkpoints/return/
    python rl/evaluate.py --stochastic --seed 1         # reproducible sampling
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from tianshou.policy import PPOPolicy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.train import DEVICE

# Reuse training's policy factory + device so the eval policy can never drift
# from the trained one (same architecture and hyperparameters).
from rl.train import build_policy as build_ppo_policy
from rl.train_env import (
    DEADLINE_STEPS,
    LINKS_FILE,
    N_ACTIONS,
    NODES_FILE,
    TRAINS_FILE,
    TRAINS_FILE_RETURN,
    NeTrainSimEnv,
    action_label,
)

_BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_BASE, "..", "results")
CHECKPOINT_DIR = os.path.join(_BASE, "..", "checkpoints")
# Return trip (B→A): separate checkpoints + outputs (see rl/train_return.py)
OUTPUT_DIR_RETURN = os.path.join(OUTPUT_DIR, "return")
CHECKPOINT_DIR_RETURN = os.path.join(CHECKPOINT_DIR, "return")

CSV_FIELDS = [
    "time_s",
    "position_m",
    "x_m",
    "y_m",
    "speed_mps",
    "speed_limit_mps",  # the limit the whole train must obey
    "applied_cap_mps",  # the speed the driver followed (target, or the limit when coasting)
    "action",
    "target",  # "coast" or e.g. "70%"
    "notch",  # chosen by the simulator, not by the agent
    "acceleration_mps2",
    "grade_perc",  # in the direction of travel
    "energy_consumed_kwh_cum",  # used by the RL
    "energy_regenerated_kwh_cum",  # reported only
    "energy_net_kwh_cum",  # reported only
]


def load_route_xy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map distance along the route (forward direction, from node 1) to (x, y).

    Returns three parallel arrays (chainage_m, x_m, y_m) for np.interp. The links
    file lists the route's links in order; the node file's second line holds the
    x and y scale factors that convert its coordinates to metres."""
    nodes: dict[int, tuple[float, float]] = {}
    with open(NODES_FILE) as f:
        f.readline()  # description
        _count, x_scale, y_scale = (float(v) for v in f.readline().split()[:3])
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                nodes[int(parts[0])] = (
                    float(parts[1]) * x_scale,
                    float(parts[2]) * y_scale,
                )

    chainage, xs, ys = [0.0], [], []
    with open(LINKS_FILE) as f:
        f.readline()  # description
        f.readline()  # count / scales
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            from_id, to_id, length = int(parts[1]), int(parts[2]), float(parts[3])
            if not xs:
                xs.append(nodes[from_id][0])
                ys.append(nodes[from_id][1])
            chainage.append(chainage[-1] + length)
            xs.append(nodes[to_id][0])
            ys.append(nodes[to_id][1])
    return np.array(chainage), np.array(xs), np.array(ys)


def find_latest_checkpoint(checkpoint_dir: str = CHECKPOINT_DIR) -> str:
    """policy_best.pth, else policy_final.pth, else the highest epoch."""
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"No checkpoints directory found at {checkpoint_dir}")
    files = sorted(f for f in os.listdir(checkpoint_dir) if f.endswith(".pth"))
    if not files:
        raise FileNotFoundError(f"No .pth checkpoint files found in {checkpoint_dir}")
    for name in ("policy_best.pth", "policy_final.pth"):
        if name in files:
            return os.path.join(checkpoint_dir, name)
    return os.path.join(checkpoint_dir, files[-1])


def load_policy(checkpoint_path: str) -> PPOPolicy:
    policy, _actor, _critic, _optim = build_ppo_policy(DEVICE)
    policy.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    policy.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return policy


def choose_action(policy: PPOPolicy, obs: np.ndarray, stochastic: bool) -> int:
    """Argmax of the action probabilities, or a sample from them (--stochastic).

    For a high-entropy policy the argmax can be unrepresentative (it parks on the
    single most likely action); sampling reflects how the policy was trained."""
    obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        probs, _ = policy.actor(obs_t)  # the actor outputs probabilities (softmax)
        probs = probs.squeeze(0)
    if stochastic:
        return int(torch.distributions.Categorical(probs=probs).sample())
    return int(probs.argmax())


def run_episode(
    policy: PPOPolicy,
    trains_file: str = TRAINS_FILE,
    stochastic: bool = False,
    reverse: bool = False,
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """Drive one full trip. Returns (per-second rows, trip summary)."""
    env = NeTrainSimEnv(trains_file=trains_file)
    env.record_trace = True
    obs, _ = env.reset()
    chainage, xs, ys = load_route_xy()

    decisions = 0
    terminated = truncated = False
    if verbose:
        print(
            f"{'Decision':>8}  {'Time(s)':>8}  {'Pos(m)':>9}  {'Speed(m/s)':>10}  "
            f"{'Action':>6}  {'Consumed(kWh)':>13}"
        )
    while not (terminated or truncated):
        action = choose_action(policy, obs, stochastic)
        obs, _, terminated, truncated, _ = env.step(action)
        decisions += 1
        if verbose and decisions % 100 == 0:
            s = env._last_state
            print(
                f"{decisions:>8}  {env._step_count:>8}  {float(s['position_m']):>9.0f}  "
                f"{float(s['speed_mps']):>10.2f}  {action_label(action):>6}  "
                f"{env._cum_energy_kwh:>13.1f}"
            )

    trace = env.trace
    last = env._last_state
    reward = env._episode_reward
    env.close()

    # The trip's own path length (the train may not use every link in the file).
    # On the return trip, position counts from the far end, so it is turned into
    # forward chainage before looking up (x, y).
    path_len = float(trace[0]["position_m"]) + float(trace[0]["remaining_dist_m"])
    rows = []
    for s in trace:
        pos = float(s["position_m"])
        chain = path_len - pos if reverse else pos
        rows.append(
            {
                "time_s": s["time_s"],
                "position_m": round(pos, 2),
                "x_m": round(float(np.interp(chain, chainage, xs)), 2),
                "y_m": round(float(np.interp(chain, chainage, ys)), 2),
                "speed_mps": round(float(s["speed_mps"]), 4),
                "speed_limit_mps": round(float(s["speed_limit_mps"]), 4),
                "applied_cap_mps": round(float(s["applied_cap_mps"]), 4),
                "action": s["action"],
                "target": action_label(s["action"]),
                "notch": int(s["notch"]),
                "acceleration_mps2": round(float(s["acceleration_mps2"]), 4),
                "grade_perc": float(s["grade_perc"]),
                "energy_consumed_kwh_cum": round(
                    float(s["energy_consumed_kwh_cum"]), 5
                ),
                "energy_regenerated_kwh_cum": round(
                    float(s["energy_regenerated_kwh_cum"]), 5
                ),
                "energy_net_kwh_cum": round(float(s["energy_net_kwh_cum"]), 5),
            }
        )

    trip_time = int(rows[-1]["time_s"])
    seconds_per_action = [0] * N_ACTIONS
    for r in rows:
        seconds_per_action[r["action"]] += 1
    summary = {
        "trip": "return" if reverse else "forward",
        "actions": "sampled" if stochastic else "argmax",
        "arrived": bool(terminated),
        "trip_time_s": trip_time,
        "deadline_s": DEADLINE_STEPS,
        "on_time": bool(terminated and trip_time <= DEADLINE_STEPS),
        "late_s": max(0, trip_time - DEADLINE_STEPS),
        "distance_m": round(float(last["position_m"]), 1),
        "decisions": decisions,
        "energy_consumed_kwh": round(float(last["energy_consumed_kwh_cum"]), 3),
        # reported only, never used by the RL
        "energy_regenerated_kwh": round(float(last["energy_regenerated_kwh_cum"]), 3),
        "energy_net_kwh": round(float(last["energy_net_kwh_cum"]), 3),
        "episode_reward": round(reward, 2),
        "seconds_per_action": {
            action_label(a): n for a, n in enumerate(seconds_per_action)
        },
    }
    return rows, summary


def eval_policy(
    policy: PPOPolicy, trains_file: str = TRAINS_FILE, stochastic: bool = False
) -> dict:
    """Score a checkpoint programmatically: the trip summary only, no files."""
    _rows, summary = run_episode(policy, trains_file, stochastic, verbose=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to .pth checkpoint (default: best/final/latest in "
        "checkpoints/, or checkpoints/return/ with --return-trip)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy instead of argmax",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for --stochastic sampling (reproducible runs)",
    )
    parser.add_argument(
        "--return-trip",
        action="store_true",
        help="Evaluate the B→A return trip (TRAIN_RETURN_FILE, "
        "checkpoints/return/, results/return/)",
    )
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    checkpoint_dir = CHECKPOINT_DIR_RETURN if args.return_trip else CHECKPOINT_DIR
    output_dir = OUTPUT_DIR_RETURN if args.return_trip else OUTPUT_DIR
    trains_file = TRAINS_FILE_RETURN if args.return_trip else TRAINS_FILE
    checkpoint = args.checkpoint or find_latest_checkpoint(checkpoint_dir)

    print(
        f"Trip: {'B→A return' if args.return_trip else 'A→B forward'}  "
        f"({'sampled' if args.stochastic else 'argmax'} actions)"
    )
    policy = load_policy(checkpoint)
    rows, summary = run_episode(
        policy, trains_file, args.stochastic, reverse=args.return_trip
    )
    summary["checkpoint"] = os.path.relpath(checkpoint)

    # make_plots.py looks for exactly these names
    suffix = "_stochastic" if args.stochastic else ""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"action_profile{suffix}.csv")
    json_path = os.path.join(output_dir, f"eval_summary{suffix}.json")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    status = "ARRIVED" if summary["arrived"] else "TIMEOUT"
    on_time = "on time" if summary["on_time"] else f"{summary['late_s']:,} s late"
    print(
        f"\n[{status}] {summary['trip_time_s']:,} s ({on_time}, deadline {DEADLINE_STEPS:,} s), "
        f"{summary['decisions']} decisions\n"
        f"  energy consumed    : {summary['energy_consumed_kwh']:>9.2f} kWh   (used by the RL)\n"
        f"  energy regenerated : {summary['energy_regenerated_kwh']:>9.2f} kWh   (reported only)\n"
        f"  net energy         : {summary['energy_net_kwh']:>9.2f} kWh   (reported only)\n"
        f"\nPer-second profile → {os.path.relpath(csv_path)}  ({len(rows):,} rows)\n"
        f"Trip summary       → {os.path.relpath(json_path)}"
    )


if __name__ == "__main__":
    main()
