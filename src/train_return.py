"""
PPO training for the RETURN trip — Ho'jakent → Toshkent (path 1500→1,
data/netrainsim_v2/train_return.dat).

Identical pipeline/hyperparameters to rl/train.py (shared run_training); only
the trains file and the checkpoint directory differ:
  checkpoints → checkpoints/return/
  evaluation  → python rl/evaluate.py --return-trip   (→ results/return/)

Trip facts (clean data, rl/run_baselines.py --return-trip, 2026-06-11):
  Net elevation is DOWNHILL (−346.6 m), so the frontier sits far below the
  forward trip's:
    n8=5,605 s/348.3 kWh, n6=5,641/335.8, n5=5,682/327.4, n4=5,768/317.7,
    n3=5,950/293.9, n2=6,375/242.2, n1=7,681/170.1
  With the shared DEADLINE_STEPS=6,500: n2 arrives on time, n1 does not →
  eco target = constant n2 ≈ 242 kWh / 6,375 s.

Run:
    source venv/bin/activate
    python rl/train_return.py 2>&1 | tee logs/train_return_<ts>.log
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.train_env import NeTrainSimEnv, TRAINS_FILE_RETURN
from rl.train import run_training

CHECKPOINT_DIR_RETURN = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", "return"
)


def make_return_env():
    return NeTrainSimEnv(trains_file=TRAINS_FILE_RETURN)


def main():
    run_training(
        make_env_fn=make_return_env,
        checkpoint_dir=CHECKPOINT_DIR_RETURN,
        trip_label="B→A return (Ho'jakent→Toshkent, path 1500→1)",
    )


if __name__ == "__main__":
    main()
