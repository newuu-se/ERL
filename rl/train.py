"""
PPO training on NeTrainSimEnv using Tianshou 0.5.1.

Improvements over REINFORCE:
  - PPO clips policy updates (stable learning, no entropy collapse)
  - Critic (value network) reduces gradient variance vs pure REINFORCE
  - ent_coef keeps exploration alive throughout training (see ENT_COEF)
  - repeat_per_collect reuses each batch of episodes (more efficient)

Reward (defined in train_env.py): minimize trip energy subject to a schedule
deadline — per-step pace penalty when lagging the deadline pace trades off
against energy. discount=0.9999 (near-undiscounted) so the ~6k-step energy sum
is not distorted and arrival/timeout signals reach the early steps.

Run:
    source venv/bin/activate
    python rl/train.py 2>&1 | tee train_log.txt
"""

import os
import random
import sys
import time

import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.train_env import (NeTrainSimEnv, ARRIVAL_BONUS, DEADLINE_STEPS,
                          W_ENERGY, W_PACE, W_OVERSPEED, W_SMOOTH)

# CPU is faster: network is tiny and inference runs one step at a time,
# so GPU kernel-launch overhead exceeds any compute gain.
DEVICE = "cpu"

NUM_TRAIN_ENVS = 8
NUM_TEST_ENVS  = 1

HIDDEN_SIZES = [256, 128, 64]
LR           = 3e-4
DISCOUNT     = 0.9999  # episodes are ~6k steps (1 step = 1 s); this is a near-undiscounted
                       # min-energy-to-goal problem. At 0.99 terminal signals vanished; at 0.999
                       # the discounting still distorted the long energy SUM and biased toward
                       # finishing sooner (= high notch, high energy — a run plateaued at ~965 kWh).
                       # 0.9999 (effective horizon ~10k > episode) keeps the energy objective
                       # ~undistorted and the arrival/timeout terminal signals visible.
ENT_COEF     = 0.004   # INITIAL value — linearly annealed to 0 by ENT_ANNEAL_END (see train_fn).
                       # A constant coefficient pinned the policy at uniform (run 2026-06-11_1140:
                       # loss/ent stayed at 2.190 ≈ ln 9 for all 100 epochs; sampled histogram flat
                       # 29-58 per notch; argmax parked at notch 0 and timed out). The per-decision
                       # advantage gradient is smaller than the entropy gradient, so exploration
                       # must taper for the policy to commit.
ENT_ANNEAL_END = 140   # epoch at which the entropy bonus reaches 0 (70% of MAX_EPOCH, same
                       # fraction as the 100-epoch runs); the remaining epochs converge the
                       # now-committed policy on the pure objective
MAX_EPOCH    = 200     # doubled from 100: the 100-epoch return run was still improving at the
                       # final epoch (best @ ep100), and the forward run's argmax stopped one
                       # notch short of the eco optimum — both point at training budget

EPISODES_PER_COLLECT = NUM_TRAIN_ENVS   # 8 parallel episodes before each update
EPISODES_PER_TEST    = 1
REPEAT_PER_COLLECT   = 4                # PPO reuses each collected batch 4×
BATCH_SIZE           = 2048
STEP_PER_EPOCH       = 3_000   # env-steps/epoch. With CONTROL_INTERVAL=15 an episode is ~470
                               # env-steps (not ~7k), so 3k ≈ one 8-episode collect per epoch

# PPO / network architecture — single source of truth, also imported by evaluate.py
# so the eval-time policy can never drift from the trained one.
OBS_SHAPE     = (9,)   # 7 physics features + time_frac + behind_frac (see train_env.py)
N_ACTIONS     = 9
EPS_CLIP      = 0.2     # clip ratio — prevents large destructive updates
VF_COEF       = 0.5     # value-loss weight
GAE_LAMBDA    = 0.95    # GAE smoothing for advantage estimates
MAX_GRAD_NORM = 0.5
ADV_NORM      = True    # advantage normalization
REWARD_NORM   = True    # ON — at γ=0.9999 returns are large (~-1000) sums; normalizing the
                        # returns stabilizes the value fn (the prior run diverged with it off)

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")

_train_start = time.time()


def make_env():
    return NeTrainSimEnv()


def set_global_seed(seed: int) -> None:
    """Seed every RNG that affects a run: Python, NumPy (tianshou minibatch
    shuffling), and Torch (policy weight init + action sampling). The vector
    envs are seeded separately in run_training. Same seed → identical run;
    different seeds → the independent replicates the variance study needs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_policy(device: str = DEVICE):
    """Construct the PPO policy (and its actor/critic/optim) from the shared
    hyperparameters above. Used by both training and evaluation so the two can
    never diverge. Returns (policy, actor, critic, optim)."""
    net_actor  = Net(state_shape=OBS_SHAPE, hidden_sizes=HIDDEN_SIZES, device=device)
    net_critic = Net(state_shape=OBS_SHAPE, hidden_sizes=HIDDEN_SIZES, device=device)
    actor  = Actor(net_actor,  action_shape=N_ACTIONS, softmax_output=True, device=device).to(device)
    critic = Critic(net_critic, device=device).to(device)
    optim  = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=LR)
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=torch.distributions.Categorical,
        discount_factor=DISCOUNT,
        eps_clip=EPS_CLIP,
        advantage_normalization=ADV_NORM,
        vf_coef=VF_COEF,
        ent_coef=ENT_COEF,
        gae_lambda=GAE_LAMBDA,
        reward_normalization=REWARD_NORM,
        max_grad_norm=MAX_GRAD_NORM,
    )
    return policy, actor, critic, optim


def run_training(make_env_fn=make_env,
                 checkpoint_dir: str = CHECKPOINT_DIR,
                 trip_label: str = "A→B (Toshkent→Ho'jakent)",
                 seed: int | None = None,
                 max_epoch: int = MAX_EPOCH,
                 ent_anneal_end: int | None = None,
                 constant_entropy: bool = False,
                 save_every: int = 10):
    """Full PPO training loop. Parameterized so every run reuses the exact same
    trainer — only the env factory, checkpoint directory, seed, budget, and
    entropy schedule differ. Used by train.py and train_return.py. (The reward
    sweep and ablation study were driven through the same entry point; their
    launcher scripts are in git history.)

    seed              : reproducibility; also seeds the vector envs (seed+10000 for test).
    max_epoch         : training budget (headline runs use 200).
    ent_anneal_end    : epoch where the entropy bonus reaches 0 (default 0.7*max_epoch).
    constant_entropy  : hold ent_coef fixed (the constant-entropy ABLATION).
    save_every        : per-epoch checkpoint cadence; 0 = only best+final (campaign).
    """
    if ent_anneal_end is None:
        ent_anneal_end = int(0.7 * max_epoch)
    if seed is not None:
        set_global_seed(seed)

    print("━" * 65)
    print("  PPO Training — NeTrainSim Energy Optimization")
    print(f"  Trip         : {trip_label}")
    print("━" * 65)
    print(f"  Device       : {DEVICE}")
    print(f"  Seed         : {seed}")
    print(f"  Train envs   : {NUM_TRAIN_ENVS} parallel C++ simulators")
    print(f"  Epochs       : {max_epoch}  (per-epoch checkpoint every {save_every or '—'})")
    print(f"  Checkpoints  : {os.path.relpath(checkpoint_dir)}")
    _ent_desc = (f"ent_coef={ENT_COEF} (CONSTANT — ablation)" if constant_entropy
                 else f"ent_coef={ENT_COEF}→0 by ep{ent_anneal_end}")
    print(f"  Algorithm    : PPO  (eps_clip=0.2, {_ent_desc}, discount={DISCOUNT}, repeat={REPEAT_PER_COLLECT}×)")
    print(f"  Reward       : -{W_ENERGY:g}·energy  -{W_PACE:g}·pace  -{W_OVERSPEED:g}·overspeed  -{W_SMOOTH:g}·|Δnotch|  +{ARRIVAL_BONUS:.0f} arrival (deadline {DEADLINE_STEPS} steps)")
    print("━" * 65 + "\n")

    train_envs = SubprocVectorEnv([make_env_fn] * NUM_TRAIN_ENVS)
    test_envs  = SubprocVectorEnv([make_env_fn] * NUM_TEST_ENVS)
    if seed is not None:
        train_envs.seed(seed)
        test_envs.seed(seed + 10_000)

    # Actor/critic/optim/policy from the shared factory (see build_policy).
    policy, actor, critic, optim = build_policy(DEVICE)

    def train_fn(epoch: int, env_step: int) -> None:
        # Entropy annealing: linear ENT_COEF → 0 by ent_anneal_end. tianshou
        # 0.5.1 PPOPolicy reads self._weight_ent at every update, so setting it
        # here (epoch start) applies to all of this epoch's gradient steps.
        # constant_entropy=True (ablation) holds the coefficient fixed.
        if not constant_entropy:
            policy._weight_ent = ENT_COEF * max(0.0, 1.0 - epoch / ent_anneal_end)
        elapsed = (time.time() - _train_start) / 60
        print(
            f"\n{'━'*65}\n"
            f"  Epoch {epoch:3d}/{max_epoch}  │  {env_step:>10,} steps  │  {elapsed:.1f} min elapsed"
            f"  │  ent_coef={policy._weight_ent:.5f}\n"
            f"{'━'*65}",
            flush=True,
        )

    def test_fn(epoch: int, env_step: int) -> None:
        # test_fn fires AFTER the epoch's training updates (tianshou calls it at
        # the start of the end-of-epoch test), so checkpoints saved here
        # genuinely contain `epoch` epochs of training. (train_fn fires at epoch
        # START — saving there labeled checkpoints one epoch ahead.)
        print(f"  [test]", flush=True)
        if save_every and epoch % save_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            path = os.path.join(checkpoint_dir, f"policy_epoch{epoch:03d}.pth")
            torch.save(policy.state_dict(), path)
            print(f"  → checkpoint: {path}", flush=True)

    def save_best_fn(pol) -> None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_path = os.path.join(checkpoint_dir, "policy_best.pth")
        torch.save(pol.state_dict(), best_path)
        print(f"  → new best saved: {best_path}", flush=True)

    buffer = VectorReplayBuffer(
        total_size=8_000 * NUM_TRAIN_ENVS,
        buffer_num=NUM_TRAIN_ENVS,
    )
    train_collector = Collector(policy, train_envs, buffer)
    test_collector  = Collector(policy, test_envs)

    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=max_epoch,
        step_per_epoch=STEP_PER_EPOCH,
        repeat_per_collect=REPEAT_PER_COLLECT,
        episode_per_test=EPISODES_PER_TEST,
        batch_size=BATCH_SIZE,
        episode_per_collect=EPISODES_PER_COLLECT,
        train_fn=train_fn,
        test_fn=test_fn,
        save_best_fn=save_best_fn,
        verbose=True,
        show_progress=True,
    )

    result = trainer.run()

    os.makedirs(checkpoint_dir, exist_ok=True)
    final_path = os.path.join(checkpoint_dir, "policy_final.pth")
    torch.save(policy.state_dict(), final_path)

    total_min = (time.time() - _train_start) / 60
    print(f"\n{'━'*65}")
    print(f"  Training complete in {total_min:.1f} min")
    print(f"  Best reward : {result['best_reward']:.3f}")
    print(f"  Final model : {final_path}")
    print(f"{'━'*65}")
    return result


def main():
    run_training()


if __name__ == "__main__":
    main()
