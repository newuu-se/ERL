"""
PPO training for the RETURN trip — Ho'jakent → Toshkent (path 1503→1,
data/netrainsim_v2/train_return.dat).

Identical pipeline/hyperparameters to rl/train.py (shared run_training); only
the trains file and the checkpoint directory differ:
  checkpoints → checkpoints/return/
  evaluation  → python rl/evaluate.py --return-trip   (→ results/return/)

Trip facts (real_data, rl/run_baselines.py --return-trip):
  Net elevation is DOWNHILL (−181.4 m), so the frontier sits well below the
  forward trip's:
    n8=8,064 s/536.4 kWh, n7=8,069/535.3, n6=8,074/520.4, n5=8,093/509.8,
    n4=8,142/491.5, n3=8,294/459.5, n2=8,843/355.6, n1=10,822/182.4
  With the shared DEADLINE_STEPS=9,200: n2 arrives on time, n1 does not →
  eco target = constant n2 ≈ 356 kWh / 8,843 s.

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
        trip_label="B→A return (Ho'jakent→Toshkent, path 1503→1)",
    )


if __name__ == "__main__":
    main()
