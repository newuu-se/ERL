"""
Evaluate a trained RL policy on one full A→B trip and export the notch
profile + physics state to a CSV file.

The output CSV can be loaded into NeTrainSim (or plotted) to compare the
RL energy profile against a baseline (e.g. constant notch or human driver).

Usage:
    source venv/bin/activate
    python rl/evaluate.py                                      # uses latest checkpoint
    python rl/evaluate.py --checkpoint checkpoints/policy_epoch050.pth
    python rl/evaluate.py --checkpoint checkpoints/policy_final.pth
"""

import argparse
import csv
import os
import sys

import torch
import numpy as np
from tianshou.policy import PPOPolicy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.train_env import (NeTrainSimEnv, NODES_FILE, LINKS_FILE,
                          TRAINS_FILE, TRAINS_FILE_RETURN, DEADLINE_STEPS)
# Reuse training's policy factory + device so the eval policy can never drift
# from the trained one (same architecture and hyperparameters).
from rl.train import build_policy as build_ppo_policy, DEVICE

_BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_BASE, "..", "results")
CHECKPOINT_DIR = os.path.join(_BASE, "..", "checkpoints")
# Return trip (B→A): separate checkpoints + outputs (see rl/train_return.py)
OUTPUT_DIR_RETURN = os.path.join(_BASE, "..", "results", "return")
CHECKPOINT_DIR_RETURN = os.path.join(_BASE, "..", "checkpoints", "return")


def load_path_coordinates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build cumulative-distance → (x, y) lookup along the route.
    Returns three parallel arrays (cum_pos_m, x_m, y_m) so np.interp can
    map any position-along-path to (x_m, y_m) by linear interpolation.
    """
    nodes: dict[int, tuple[float, float]] = {}
    with open(NODES_FILE) as f:
        f.readline()                # header line
        f.readline()                # count / scales line
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            nid = int(parts[0])
            x = float(parts[1]); y = float(parts[2])
            nodes[nid] = (x, y)

    # Walk links in order; cumulative distance at each visited node.
    cum, xs, ys = [0.0], [], []
    first_id = None
    with open(LINKS_FILE) as f:
        f.readline()                # header
        f.readline()                # count / scales
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            from_id = int(parts[1]); to_id = int(parts[2])
            length  = float(parts[3])
            if first_id is None:
                first_id = from_id
                xs.append(nodes[from_id][0])
                ys.append(nodes[from_id][1])
            cum.append(cum[-1] + length)
            xs.append(nodes[to_id][0])
            ys.append(nodes[to_id][1])
    return np.array(cum), np.array(xs), np.array(ys)


def find_latest_checkpoint(checkpoint_dir: str = CHECKPOINT_DIR) -> str:
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"No checkpoints directory found at {checkpoint_dir}")
    files = sorted(f for f in os.listdir(checkpoint_dir) if f.endswith(".pth"))
    if not files:
        raise FileNotFoundError(f"No .pth checkpoint files found in {checkpoint_dir}")
    if "policy_best.pth" in files:
        return os.path.join(checkpoint_dir, "policy_best.pth")
    if "policy_final.pth" in files:
        return os.path.join(checkpoint_dir, "policy_final.pth")
    return os.path.join(checkpoint_dir, files[-1])


def build_policy(checkpoint_path: str) -> PPOPolicy:
    policy, _actor, _critic, _optim = build_ppo_policy(DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    policy.load_state_dict(state_dict)
    policy.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return policy


def eval_policy(policy: PPOPolicy, trains_file: str = TRAINS_FILE,
                stochastic: bool = False) -> dict:
    """Run one greedy/sampled episode and return summary stats only (no CSV).

    Currently unused by any script in the tree — it was the campaign's scoring
    hook. Kept because it is the cheapest way to score a checkpoint
    programmatically (run_episode writes a CSV; this does not)."""
    env = NeTrainSimEnv(trains_file=trains_file)
    obs, _ = env.reset()
    terminated = truncated = False
    notch_hist = [0] * 9
    while not (terminated or truncated):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            probs, _ = policy.actor(obs_t)
            probs = probs.squeeze(0)
            notch = (int(torch.distributions.Categorical(probs).sample())
                     if stochastic else int(probs.argmax().item()))
        notch_hist[notch] += 1
        obs, _, terminated, truncated, _ = env.step(notch)
    energy = float(env._cum_energy_kwh)
    steps = int(env._step_count)
    env.close()
    return {
        "stochastic": stochastic,
        "energy_kwh": round(energy, 3),
        "steps": steps,
        "arrived": bool(terminated),
        "on_schedule": bool(terminated and steps <= DEADLINE_STEPS),
        "lateness_frac": round(max(0.0, steps - DEADLINE_STEPS) / DEADLINE_STEPS, 4),
        "notch_hist": notch_hist,
    }


def run_episode(policy: PPOPolicy, output_path: str, stochastic: bool = False,
                trains_file: str = TRAINS_FILE, reverse: bool = False) -> None:
    env = NeTrainSimEnv(trains_file=trains_file)
    obs, _ = env.reset()

    cum_pos, xs, ys = load_path_coordinates()
    route_len = float(cum_pos[-1])

    rows = []
    step = 0
    terminated = False
    truncated  = False

    print("Running evaluation episode...")
    print(f"{'Step':>6}  {'Pos(m)':>10}  {'X(m)':>12}  {'Y(m)':>12}  "
          f"{'Speed(m/s)':>10}  {'Notch':>5}  {'CumEnergy':>10}")
    print("-" * 80)

    while not (terminated or truncated):
        # Deterministic (argmax) by default; --stochastic samples from the policy.
        # NOTE: for a high-entropy policy the argmax can be unrepresentative (its mode
        # parks at the lowest-energy notch); --stochastic reflects the trained behavior.
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            probs, _ = policy.actor(obs_tensor)
            probs = probs.squeeze(0)
            notch = (int(torch.distributions.Categorical(probs).sample())
                     if stochastic else int(probs.argmax().item()))

        obs, reward, terminated, truncated, _ = env.step(notch)

        # NOTE: under action repeat (CONTROL_INTERVAL=15) state is the LAST sim
        # second of the decision; per-step sums must come from the env, which
        # accumulates every simulator second (env._cum_energy_kwh).
        state = env._last_state
        position_m = float(state["position_m"])
        speed_mps  = float(state["speed_mps"])

        # Interpolate (x, y) at the train's current position along the path.
        # load_path_coordinates() walks links in FORWARD (node 1→1500) order;
        # on the return trip position_m counts from node 1500, so map it to the
        # forward chainage first.
        route_pos = (route_len - position_m) if reverse else position_m
        x_m = float(np.interp(route_pos, cum_pos, xs))
        y_m = float(np.interp(route_pos, cum_pos, ys))

        rows.append({
            "time_s":         env._step_count,
            "position_m":     round(position_m, 3),
            "x_m":            round(x_m, 3),
            "y_m":            round(y_m, 3),
            "speed_mps":      round(speed_mps, 4),
            "notch":          notch,
            # Cumulative energy is summed by the env over EVERY simulator second,
            # so it stays correct under action repeat (the state above is only the
            # last second of the decision).
            "energy_kwh_cum": round(env._cum_energy_kwh, 5),
        })

        step += 1
        if step % 100 == 0:
            print(f"{step:>6}  {position_m:>10.1f}  {x_m:>12.1f}  {y_m:>12.1f}  "
                  f"{speed_mps:>10.3f}  {notch:>5}  {env._cum_energy_kwh:>10.3f}")

    # Totals from the env (correct under action-repeat: sums every simulator second,
    # not just the last one of each control interval).
    total_energy = env._cum_energy_kwh
    sim_steps = env._step_count
    env.close()

    status = "ARRIVED" if terminated else "TIMEOUT"
    print(f"\n[{status}] decisions={step}  sim_steps={sim_steps}  total_energy={total_energy:.2f} kWh\n")

    # Write CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fieldnames = ["time_s", "position_m", "x_m", "y_m", "speed_mps", "notch",
                  "energy_kwh_cum"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Notch profile saved → {output_path}")
    print(f"Rows: {len(rows)} (control decisions)  |  Total energy: {total_energy:.2f} kWh  |  sim_steps: {sim_steps}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pth checkpoint (default: latest in checkpoints/, "
                             "or checkpoints/return/ with --return-trip)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: results/notch_profile.csv, "
                             "or results/return/notch_profile.csv with --return-trip)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions from the policy instead of argmax "
                             "(representative for a high-entropy policy)")
    parser.add_argument("--return-trip", action="store_true",
                        help="Evaluate the B→A return trip (train_return.dat, "
                             "checkpoints/return/, results/return/)")
    args = parser.parse_args()

    checkpoint_dir = CHECKPOINT_DIR_RETURN if args.return_trip else CHECKPOINT_DIR
    output_dir     = OUTPUT_DIR_RETURN     if args.return_trip else OUTPUT_DIR
    trains_file    = TRAINS_FILE_RETURN    if args.return_trip else TRAINS_FILE

    checkpoint = args.checkpoint or find_latest_checkpoint(checkpoint_dir)
    output_path = args.output or os.path.join(output_dir, "notch_profile.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Trip: {'B→A return' if args.return_trip else 'A→B forward'}")
    policy = build_policy(checkpoint)
    run_episode(policy, output_path, stochastic=args.stochastic,
                trains_file=trains_file, reverse=args.return_trip)


if __name__ == "__main__":
    main()
