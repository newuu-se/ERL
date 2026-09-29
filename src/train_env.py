"""
Gymnasium environment wrapping NeTrainSim for train energy optimization.

Phase 2 (interactive):
  reset() starts the simulator in interactive mode, then sends notch=0 once to
  get the first timestep state.
  step(action) sends {"notch": N} to simulator stdin and reads the next state
  JSON from stdout (prefixed by "NTS_JSON ").

Observation space (9 floats):
  [speed_mps, position_m, grade_perc, curvature_perc,
   remaining_dist_m, energy_kwh, link_max_speed_mps,
   time_frac, behind_frac]

The last two make the schedule visible: the pace-penalty reward depends on the
step counter, so without a time feature the reward would be a function of
state the agent cannot observe (non-Markov) — the value function could never
predict the pace cost and the policy could never modulate with schedule slack.

Action space: Discrete(9) — notch 0-8 (maps to locomotive currentLocNotch)
"""

import json
import os
import queue
import subprocess
import threading
import time

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import platform as _platform
_BUILD_DIR = "build-linux" if _platform.system() == "Linux" else "build-mac"
SIMULATOR_BIN = os.path.join(
    _REPO, "NeTrainSim-adjusted", _BUILD_DIR,
    "src", "NeTrainSimConsole", "NeTrainSim"
)
# ACTIVE DATASET: data/real_data/ — 1503 nodes / 1502 links, 75,064 m.
# Speed limits here are a near-continuous per-segment profile (960 distinct
# values, 1.42-21.16 m/s), NOT the four regulatory zones of the older v2 data.
NODES_FILE  = os.path.join(_REPO, "data", "real_data", "nodesFile_real_data.dat")
LINKS_FILE  = os.path.join(_REPO, "data", "real_data", "linksFile_real_data.dat")
# NOTE: the trains files still live under netrainsim_v2/ but their paths were
# updated to 1→1503 / 1503→1, matching the real_data node count. They are
# route-definition files for the ACTIVE data despite the folder name.
TRAINS_FILE = os.path.join(_REPO, "data", "netrainsim_v2", "trainsFile_rl.dat")
# Return trip: identical consist, path reversed (1503→1). Forward net elevation
# on real_data is +181.4 m, so the return is DOWNHILL by the same amount and its
# constant-notch frontier differs — measure it with
# `python rl/run_baselines.py --return-trip` (→ results/baselines_return.json).
TRAINS_FILE_RETURN = os.path.join(_REPO, "data", "netrainsim_v2", "train_return.dat")

TOTAL_ROUTE_LENGTH_M = 75_064.00  # sum of all 1502 link lengths (linksFile_real_data.dat)
STATE_PREFIX = "NTS_JSON "
MAX_STEPS      = 18_000  # hard ceiling. Scaled from the speed-limit floor (~7,100 s) by the
                         # same ~2.4x margin the previous dataset used (12,000 over a 5,025 s
                         # floor), so a slow-but-progressing policy is not truncated too early.
DEADLINE_STEPS = 9_200   # schedule deadline (s). PROVISIONAL — set from the physical floor,
                         # NOT from a measured frontier. On real_data the speed limits alone
                         # forbid arriving before ~7,100 s (sum of length/limit; a kinematic
                         # pass with EMU accel/brake limits agrees at ~7,050 s). 9,200 keeps
                         # the previous dataset's ~1.3x floor-to-deadline ratio.
                         # >>> RE-SET THIS after `python rl/run_baselines.py`: pick a value
                         # above the fastest constant notch so at least one constant-notch
                         # policy is on-schedule, else every episode is late by construction
                         # and the pace penalty carries no learnable signal.
CONTROL_INTERVAL = 15    # action repeat: hold each chosen notch this many simulator seconds, so the
                         # agent makes ~470 decisions per trip instead of ~7,000. Shortening the
                         # decision horizon ~15x makes the long-horizon credit assignment tractable —
                         # the prerequisite for learning terrain-aware notch modulation (1-second
                         # control kept collapsing to a single near-constant notch).

# ── Reward weights ──────────────────────────────────────────────────────────
# Goal: minimize trip energy SUBJECT TO arriving by the deadline. The schedule is
# enforced PER-STEP via a pace penalty (not a terminal lump sum): with a strong
# per-step energy reward, a distant terminal late-penalty can't override the
# immediate per-step gain of running the lowest notch over a ~6k-step episode, so
# the deterministic policy crawls (observed: greedy mode = notch 1, 820 kWh but
# 10,700 steps). A per-step pace penalty makes lagging costly immediately.
#   r_t = -W_ENERGY*energy_t                 (the objective — kWh this step)
#         -W_PACE*max(0, behind_fraction)    (per-step: lagging the deadline pace → penalty)
#         -W_OVERSPEED*overspeed_t           (speed cap; rarely active, sim caps speed)
#   terminal(arrived):  +ARRIVAL_BONUS
#   truncated(timeout): -TIMEOUT_PENALTY
# behind_fraction = max(0, step/DEADLINE_STEPS*route_len - position) / route_len, so
# the penalty is ZERO when on/ahead of schedule — the agent coasts freely where it
# has slack and is pushed to keep pace only when lagging.
# The constant-notch frontier for real_data has NOT been measured yet — run
# `python rl/run_baselines.py` before reading anything into absolute energy
# numbers, and re-check DEADLINE_STEPS against it. (For reference, the previous
# v2 dataset gave n8=862 kWh/5602 s … n3=789/6442, eco target n3; real_data is a
# different route profile and those numbers do not carry over.)
# Every reward weight and the time-feature ablation read an optional environment
# override (RL_W_PACE, RL_W_ENERGY, RL_W_SMOOTH, RL_W_OVERSPEED, RL_ABLATE_TIME),
# so the sensitivity sweep and ablation study set configurations WITHOUT editing
# source — the single source of truth for every run is the env it launches with.
def _envf(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None and v != "" else default

W_ENERGY        = _envf("RL_W_ENERGY", 3.0)   # per kWh — amplified so the fast→eco energy gap is a
                           # strong, learnable gradient (at W_ENERGY=1 the eco gain was ~5% of reward)
W_PACE          = _envf("RL_W_PACE", 2.0)     # per-step penalty per (fraction-of-route) BEHIND the
                           # deadline pace; zero when on/ahead → coast freely; lagging → keep pace.
                           # The sweep variable: too low → mode settles one notch below feasible.
W_OVERSPEED     = _envf("RL_W_OVERSPEED", 1.0)   # per (m/s) over the link limit (sim hard-caps speed)
W_SMOOTH        = _envf("RL_W_SMOOTH", 0.15)  # penalty per |Δnotch| between consecutive 15-s
                           # decisions. 0.5 (+low entropy) collapsed to a constant notch; 0.15 is the
                           # middle ground — discourages erratic 0↔8 jumping but allows terrain shifts.
ARRIVAL_BONUS   = 200.0    # one-shot reward at terminus
TIMEOUT_PENALTY = 1500.0   # large enough that giving up is never the best option

# Ablation: when set, zero the two schedule features (time_frac, behind_frac) in
# the observation, recreating the non-Markovian-reward failure mode for study.
ABLATE_TIME_FEATURES = os.environ.get("RL_ABLATE_TIME", "0") == "1"

# No progress-shaping term. With near-undiscounted DISCOUNT (0.9999 in train.py)
# the arrival bonus / timeout penalty propagate back across the full ~6k-step
# episode, so completion is incentivized directly and the reward stays a clean,
# interpretable proxy for the objective (minimize total energy + time). Earlier
# designs added a progress reward to force completion, but under heavier
# discounting it biased the policy toward finishing fast (high notch, high energy);
# a potential-based variant removed that bias but corrupted the logged/selection
# reward via its (1-γ)·Φ accumulation over the long episode.

# Normalisation denominators for _state_to_obs
_SPEED_MAX     = 22.22  # m/s. real_data max limit is 21.16, so this is slightly conservative
                        # (obs tops out at 0.95 rather than 1.0) — harmless, kept for continuity.
_GRADE_MAX     = 8.5    # real_data grade range is [-3.49, +8.49] %.
                        # WAS 4.5 (from linksFile_v2_clean.dat): the four links above 4.5%
                        # — including the +8.49% climb at link 853 — would all clip to obs 1.0,
                        # hiding the steepest terrain on the route from the policy at exactly
                        # the point where notch choice matters most.
_ENERGY_MAX    = 1.5    # per-step energy cap in kWh. Measured max on clean data = 1.393 kWh/s
                        # (constant notch 8, acceleration transient; steady full power ≈ 1.1).
                        # WAS 0.25 (stale) — clipped any heavy power draw to the same obs value.
_MAXSPEED_MAX  = 22.22  # route speed-limit max in m/s (80 km/h)
_TIME_MAX      = 2.0    # time_frac cap: MAX_STEPS/DEADLINE_STEPS = 1.85 < 2.0

_LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")


class NeTrainSimEnv(gym.Env):
    metadata = {"render_modes": []}

    # All 9 features are normalized to roughly [-1, 1] or [0, 1] by _state_to_obs.
    # [speed, position, grade, curvature, remaining, energy, link_max_speed,
    #  time_frac, behind_frac]
    observation_space = Box(
        low =np.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      dtype=np.float32),
        high=np.array([2.0, 1.0,  1.0,  1.0, 1.0, 1.0, 2.0, 2.0, 1.0],
                      dtype=np.float32),
    )
    action_space = Discrete(9)  # notch 0-8

    def __init__(self, trains_file: str = TRAINS_FILE):
        super().__init__()
        # trains_file selects the trip: TRAINS_FILE (A→B) or TRAINS_FILE_RETURN
        # (B→A, path 1500→1). Route length/links are identical either way.
        self._trains_file = trains_file
        self._proc: subprocess.Popen | None = None
        self._out_queue: queue.Queue | None = None  # lines from the reader thread
        self._stderr_log = None   # file handle for simulator stderr
        self._last_state: dict | None = None
        self._cum_energy_kwh: float = 0.0   # running total for logging/reward only
        self._step_count: int = 0
        self._episode_count: int = 0
        self._episode_start: float = 0.0
        self._episode_reward: float = 0.0
        self._prev_notch: int | None = None  # for the notch-change (smoothness) penalty

        if not os.path.isfile(SIMULATOR_BIN):
            raise FileNotFoundError(
                f"Simulator binary not found: {SIMULATOR_BIN}\n"
                "Run: cd NeTrainSim-adjusted && ./build-linux.sh  (or build-mac.sh on macOS)"
            )
        for path in (NODES_FILE, LINKS_FILE, self._trains_file):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"NeTrainSim input file not found: {path}\n"
                    "All input files are checked in under data/netrainsim_v2/ "
                    "and are used as-is — there is no generation step."
                )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        self._step_count = 0
        self._episode_reward = 0.0
        self._cum_energy_kwh = 0.0
        self._prev_notch = None
        self._episode_start = time.time()
        self._episode_count += 1
        self._start_interactive_simulator()
        state = self._send_action_and_read_state(0)
        self._last_state = state
        step_energy = float(state["energy_kwh"])
        self._cum_energy_kwh = step_energy  # include bootstrap step-0 energy
        # Bootstrap consumed one simulator timestep (notch=0 sent to get initial
        # state). Start _step_count at 1 so it tracks actual simulator steps.
        self._step_count = 1
        obs = self._state_to_obs(state, step_energy)
        # Return empty info dict — tianshou 0.5.1 Batch cannot create new keys
        # via index assignment, so any non-empty info dict causes a ValueError.
        return obs, {}

    def step(self, action: int):
        notch = int(action)
        if notch < 0 or notch > 8:
            raise ValueError(f"Action notch must be in [0, 8], got {notch}")

        # Notch-change (smoothness) penalty, once per decision: discourages erratic
        # throttle jumping so the notch profile stays smooth/realistic. Skipped on the
        # first decision of an episode (no previous notch to compare against).
        change_penalty = (W_SMOOTH * abs(notch - self._prev_notch)
                          if self._prev_notch is not None else 0.0)
        self._prev_notch = notch
        self._episode_reward -= change_penalty

        # Coarse control (action repeat): hold the chosen notch for CONTROL_INTERVAL
        # simulator seconds, summing reward, then decide again. Fewer, longer
        # decisions → tractable credit assignment → terrain-aware notch modulation.
        total_reward = -change_penalty
        terminated = truncated = False
        step_energy_kwh = 0.0
        for _ in range(CONTROL_INTERVAL):
            r, terminated, truncated, step_energy_kwh = self._sim_step(notch)
            total_reward += r
            if terminated or truncated:
                break

        obs = self._state_to_obs(self._last_state, step_energy_kwh)
        # Empty info dict — tianshou 0.5.1 Batch rejects new keys via index assignment.
        return obs, float(total_reward), terminated, truncated, {}

    def _sim_step(self, notch: int):
        """Advance the simulator one second at the given notch; return
        (reward, terminated, truncated, step_energy_kwh)."""
        state = self._send_action_and_read_state(notch)
        self._last_state = state
        self._step_count += 1

        # energy_kwh from the simulator is per-step energy (not cumulative); it
        # oscillates ~0–0.2 kWh/step depending on throttle.
        step_energy_kwh = float(state["energy_kwh"])
        self._cum_energy_kwh += step_energy_kwh
        speed_mps = float(state["speed_mps"])
        max_speed = float(state["link_max_speed_mps"])
        position_m = float(state["position_m"])

        # Episode boundaries.
        terminated = bool(state["terminated"]) or position_m >= TOTAL_ROUTE_LENGTH_M
        truncated = self._step_count >= MAX_STEPS and not terminated

        # Per-step pace penalty: penalize lagging the constant pace needed to reach
        # the terminus by DEADLINE_STEPS. Zero when on/ahead of schedule, so the
        # agent coasts freely where it has slack; lagging is penalized immediately.
        target_pos_m = (self._step_count / DEADLINE_STEPS) * TOTAL_ROUTE_LENGTH_M
        behind_frac  = max(0.0, target_pos_m - position_m) / TOTAL_ROUTE_LENGTH_M
        pace_penalty = W_PACE * behind_frac

        # Over-speed penalty: safety net only (sim hard-caps speed at the limit).
        speed_over_penalty = W_OVERSPEED * max(0.0, speed_mps - max_speed)

        reward = -W_ENERGY * step_energy_kwh - pace_penalty - speed_over_penalty
        if terminated:
            reward += ARRIVAL_BONUS
        elif truncated:
            reward -= TIMEOUT_PENALTY

        self._episode_reward += reward

        if terminated or truncated:
            elapsed = time.time() - self._episode_start
            pct = 100.0 * position_m / TOTAL_ROUTE_LENGTH_M
            status = "✓ ARRIVED" if terminated else "✗ TIMEOUT"
            print(
                f"[{status}]  ep={self._episode_count:>4d}"
                f"  {self._step_count:>5,} steps"
                f"  {position_m:>7,.0f}m ({pct:4.1f}%)"
                f"  energy={self._cum_energy_kwh:>7.1f} kWh"
                f"  reward={self._episode_reward:>+8.1f}"
                f"  {elapsed:.1f}s",
                flush=True,
            )
        return reward, terminated, truncated, step_energy_kwh

    def close(self):
        if self._proc is None:
            return
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        finally:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=5)
            self._proc = None
        if self._stderr_log is not None:
            try:
                self._stderr_log.close()
            except Exception:
                pass
            self._stderr_log = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ── Internal helpers ────────────────────────────────────────────────────

    def _start_interactive_simulator(self) -> None:
        os.makedirs(_LOGS_DIR, exist_ok=True)
        # One log file per worker process (8 parallel envs would otherwise clobber
        # a shared file), truncated each reset so it holds only the current
        # episode's stderr instead of growing unbounded across runs.
        stderr_path = os.path.join(_LOGS_DIR, f"netrainsim_stderr_{os.getpid()}.log")
        self._stderr_log = open(stderr_path, "w")
        self._proc = subprocess.Popen(
            [SIMULATOR_BIN,
             "-n", NODES_FILE,
             "-l", LINKS_FILE,
             "-t", self._trains_file,
             "-p", "1.0",
             "-I"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_log,
            text=True,
            bufsize=1,
        )
        # Dedicated reader thread → queue. A select()-on-fd + buffered readline()
        # combo can deadlock: one OS read may pull several lines into Python's
        # buffer, then the next select() blocks on an empty fd while the wanted
        # line sits in the buffer, and the 30 s watchdog kills a healthy episode.
        # The thread does blocking readlines (immune to that race); EOF → None.
        self._out_queue = queue.Queue()
        threading.Thread(
            target=self._drain_stdout,
            args=(self._proc, self._out_queue),
            daemon=True,
        ).start()

    @staticmethod
    def _drain_stdout(proc: subprocess.Popen, out_queue: queue.Queue) -> None:
        try:
            for line in proc.stdout:
                out_queue.put(line)
        except (ValueError, OSError):
            pass  # stream closed during shutdown
        out_queue.put(None)  # EOF sentinel

    def _send_action_and_read_state(self, notch: int) -> dict:
        if self._proc is None or self._proc.stdin is None or self._out_queue is None:
            raise RuntimeError("Interactive simulator process is not running.")

        payload = json.dumps({"notch": int(notch)})
        try:
            self._proc.stdin.write(payload + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError("Simulator stdin pipe is closed.") from exc

        _READ_TIMEOUT = 30.0
        while True:
            try:
                line = self._out_queue.get(timeout=_READ_TIMEOUT)
            except queue.Empty:
                rc = self._proc.poll()
                raise RuntimeError(
                    f"Simulator stalled: no output for {_READ_TIMEOUT}s "
                    f"(returncode={rc}). Check logs/netrainsim_stderr_<pid>.log"
                ) from None
            if line is None:
                rc = self._proc.poll()
                raise RuntimeError(
                    f"Simulator terminated before returning state "
                    f"(returncode={rc}). Check logs/netrainsim_stderr_<pid>.log"
                )
            line = line.strip()
            if not line.startswith(STATE_PREFIX):
                continue
            raw_json = line[len(STATE_PREFIX):]
            try:
                state = json.loads(raw_json)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid simulator state JSON: {raw_json}") from exc
            return state

    def _state_to_obs(self, state: dict, step_energy_kwh: float) -> np.ndarray:
        position = float(state["position_m"])
        remaining = max(0.0, TOTAL_ROUTE_LENGTH_M - position)
        # Schedule features — the pace penalty is a function of the step counter,
        # so the agent must SEE the clock or the reward is non-Markov in its obs:
        # time_frac   = elapsed time as a fraction of the deadline
        # behind_frac = how far (fraction of route) the train lags the deadline
        #               pace; exactly the quantity the pace penalty charges.
        time_frac   = self._step_count / DEADLINE_STEPS
        target_pos  = time_frac * TOTAL_ROUTE_LENGTH_M
        behind_frac = max(0.0, target_pos - position) / TOTAL_ROUTE_LENGTH_M
        if ABLATE_TIME_FEATURES:
            # Non-Markov ablation: hide the clock from the policy (the reward still
            # depends on it). Keeps the 9-dim shape so the network is unchanged.
            time_frac = behind_frac = 0.0
        obs = np.array([
            float(state["speed_mps"])          / _SPEED_MAX,
            position                           / TOTAL_ROUTE_LENGTH_M,
            float(state["grade_perc"])         / _GRADE_MAX,
            float(state["curvature_perc"]),
            remaining                          / TOTAL_ROUTE_LENGTH_M,
            step_energy_kwh                    / _ENERGY_MAX,
            float(state["link_max_speed_mps"]) / _MAXSPEED_MAX,
            min(time_frac, _TIME_MAX),
            behind_frac,
        ], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)
