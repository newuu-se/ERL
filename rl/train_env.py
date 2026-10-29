"""
Gymnasium environment wrapping NeTrainSim for train energy optimization.

Phase 2 (interactive, speed control):
  reset() starts the simulator in interactive mode, then sends {"coast": true}
  once to get the first timestep state (the train is still standing).
  step(action) sends one speed command per simulator second to stdin —
  {"target_frac": f} or {"coast": true} — and reads the next state JSON from
  stdout (prefixed by "NTS_JSON ").

The agent chooses a target speed, not a notch: NeTrainSim's own driver model
picks the throttle needed to reach and hold min(speed limit, target). Coasting
cuts traction to zero and lets the train roll on resistance and grade.

Observation space (23 floats):
  [speed, position, grade, curvature, energy, speed_limit,
   time_frac, behind_frac, acceleration,
   notch (the notch the simulator chose last step, / notch_max),
   grade_ahead x3 (mean over the next 0.5/1/2 km),
   speed_limit_min_ahead x3 (lowest limit over the next 0.5/1/2 km),
   previous action one-hot x7]

time_frac and behind_frac make the schedule visible: the pace-penalty reward
depends on the step counter, so without a time feature the reward would be a
function of state the agent cannot observe (non-Markov) — the value function
could never predict the pace cost and the policy could never modulate with
schedule slack. The previous action is observed for the same reason: the
smoothness penalty depends on it.

Energy: everything the RL uses — the reward, the energy observation and the
episode total — is energy CONSUMED. Regenerated energy is never credited; it is
only reported at the end of a trip, together with net = consumed - regenerated.

Action space: Discrete(7) — 0 = coast, 1-6 = target 50/60/70/80/90/100 % of
the current speed limit (see TARGET_FRACTIONS)
"""

import json
import os
import platform as _platform
import queue
import subprocess
import threading
import time
from typing import Any

import gymnasium as gym
import numpy as np
from dotenv import load_dotenv
from gymnasium.spaces import Box, Discrete

from rl.utils import read_links_file


load_dotenv()

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── External dependencies ───────────────────────────────────────────────────
# This repo holds ONLY the RL layer. The simulator and the route data are
# separate repositories, referenced here rather than vendored:
#
#   simulator   https://github.com/newuu-se/NeTrainSim      (build it first)
#   route data  https://github.com/newuu-se/train-routes    (private)
#
# Point the env vars at those checkouts

_BUILD = "linux" if _platform.system() == "Linux" else "mac"
# The simulator directory is NOT validated at import: only training/eval need the
# binary, so plotting must keep working without a built simulator. _start_sim()
# raises with build instructions if it is missing.
SIMULATOR_BIN = os.environ.get("NETRAINSIM_BIN", "")
# Folder for the simulator's own files (one trainSummary_*.txt per trip). Unset →
# NeTrainSim's default, ~/Documents/NeTrainSim.
SIMULATOR_OUTPUT_DIR = os.environ.get("NETRAINSIM_OUTPUT_DIR", "")

# ACTIVE DATASET: real_data/ — 1503 nodes / 1502 links, 75,064 m.
# Speed limits here are a near-continuous per-segment profile (960 distinct
# values, 1.42-21.16 m/s), NOT the four regulatory zones of the older v2 data.
NODES_FILE = os.getenv("NODE_FILE", "")
LINKS_FILE = os.getenv("LINK_FILE", "")
TRAINS_FILE = os.getenv("TRAIN_FILE", "")
# Return trip: identical consist, path reversed (1503→1). Forward net elevation
# on real_data is +181.4 m, so the return is DOWNHILL by the same amount and its
# constant-target frontier differs — measure it with
# `python rl/run_baselines.py --return-trip` (→ results/baselines_return.json).

TRAINS_FILE_RETURN = os.getenv("TRAIN_RETURN_FILE", "")

links = read_links_file(LINKS_FILE)
TOTAL_ROUTE_LENGTH_M = sum(link["length_m"] for link in links)
STATE_PREFIX = "NTS_JSON "
MAX_STEPS = 18_000  # hard ceiling. Above the slowest action-table trip (constant 50 %
# target: 15,181 s forward, 15,010 s return on oct-2026), so a
# slow-but-progressing policy is not truncated too early.
DEADLINE_STEPS = (
    9_200  # schedule deadline (s), shared by both trips. Checked against the
)
# measured constant-target frontier (oct-2026 data, speed-control simulator):
#   forward: 100 % 8,282 s · 90 % 8,870 s · 80 % 9,792 s
#   return : 100 % 7,880 s · 90 % 8,577 s · 80 % 9,539 s
# The fastest possible trip is 8,282 s forward, so 9,200 s is that +11 %
# (return: +17 %). In both directions 90 % and 100 % arrive on time and 80 % is
# late, so the deadline binds and the pace penalty carries signal. (The speed
# limits alone give a 7,116 s floor; acceleration and braking add ~1,170 s.)
# If the line has a real timetable running time, use that instead.
# >>> RE-CHECK after any physics/data change with `python rl/run_baselines.py`
# (and --return-trip): keep it above the fastest constant target so at least
# one constant-target policy is on-schedule, else every episode is late by
# construction and the pace penalty carries no learnable signal.
CONTROL_INTERVAL = (
    15  # action repeat: hold each chosen action this many simulator seconds, so the
)
# agent makes ~470 decisions per trip instead of ~7,000. Shortening the
# decision horizon ~15x makes the long-horizon credit assignment tractable —
# the prerequisite for learning terrain-aware speed modulation (1-second
# control kept collapsing to a single near-constant notch).

# Action table. Index 0 coasts (zero traction); the others hold a target speed
# at that fraction of the current speed limit, re-read every simulator second so
# the target follows the limit along the route. 1.0 = drive at the limit.
TARGET_FRACTIONS = (None, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
N_ACTIONS = len(TARGET_FRACTIONS)


def action_command(action: int) -> dict:
    """The simulator command for an action index."""
    frac = TARGET_FRACTIONS[action]
    return {"coast": True} if frac is None else {"target_frac": frac}


def action_label(action: int) -> str:
    """Short label for logs and plots: 'coast' or e.g. '70%'."""
    frac = TARGET_FRACTIONS[action]
    return "coast" if frac is None else f"{frac:.0%}"


# ── Reward weights ──────────────────────────────────────────────────────────
# Goal: minimize trip energy SUBJECT TO arriving by the deadline. The schedule is
# enforced PER-STEP via a pace penalty (not a terminal lump sum): with a strong
# per-step energy reward, a distant terminal late-penalty can't override the
# immediate per-step gain of running the lowest notch over a ~6k-step episode, so
# the deterministic policy crawls (observed: greedy mode = notch 1, 820 kWh but
# 10,700 steps). A per-step pace penalty makes lagging costly immediately.
#   r_t = -W_ENERGY*energy_t                 (the objective — kWh consumed this step)
#         -W_PACE*max(0, behind_fraction)    (per-step: lagging the deadline pace → penalty)
#         -W_OVERSPEED*overspeed_t           (speed cap; rarely active, sim caps speed)
#   terminal(arrived):  +ARRIVAL_BONUS
#   truncated(timeout): -TIMEOUT_PENALTY
# behind_fraction = max(0, step/DEADLINE_STEPS*route_len - position) / route_len, so
# the penalty is ZERO when on/ahead of schedule — the agent coasts freely where it
# has slack and is pushed to keep pace only when lagging.
# The constant-target frontier for real_data has NOT been measured yet — run
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
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default


W_ENERGY = _envf(
    "RL_W_ENERGY", 3.0
)  # per kWh — amplified so the fast→eco energy gap is a
# strong, learnable gradient (at W_ENERGY=1 the eco gain was ~5% of reward)
W_PACE = _envf("RL_W_PACE", 2.0)  # per-step penalty per (fraction-of-route) BEHIND the
# deadline pace; zero when on/ahead → coast freely; lagging → keep pace.
# The sweep variable: too low → mode settles one level below feasible.
W_OVERSPEED = _envf(
    "RL_W_OVERSPEED", 1.0
)  # per (m/s) over the link limit (sim hard-caps speed)
W_SMOOTH = _envf("RL_W_SMOOTH", 0.15)  # penalty per |Δaction| between consecutive 15-s
# decisions (one step = 10 % of the limit; coast→50 % counts as one step).
# Tuned on notch control: 0.5 (+low entropy) collapsed to a constant notch; 0.15
# was the middle ground. Re-tune for speed control.
ARRIVAL_BONUS = _envf("RL_W_ARRIVAL_BONUS", 200.0)  # one-shot reward at terminus
TIMEOUT_PENALTY = _envf(
    "RL_W_TIMEOUT_PENALTY", 1500.0
)  # large enough that giving up is never the best option

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
_SPEED_MAX = max(link["speed_limit_mps"] for link in links) / 0.95
# this is slightly conservative (obs tops out at 0.95 rather than 1.0) — harmless, kept for continuity.
_GRADE_MAX = max(
    abs(link["grade"]) for link in links
)  # real_data grade range is [-3.49, +8.49] %.
# WAS 4.5 (from linksFile_v2_clean.dat): the four links above 4.5%
# — including the +8.49% climb at link 853 — would all clip to obs 1.0,
# hiding the steepest terrain on the route from the policy at exactly
# the point where the speed choice matters most.
_ENERGY_MAX = (
    1.5  # per-step energy cap in kWh. Measured max on clean data = 1.393 kWh/s
)
# (constant notch 8, acceleration transient; steady full power ≈ 1.1).
# WAS 0.25 (stale) — clipped any heavy power draw to the same obs value.
_MAXSPEED_MAX = 22.22  # route speed-limit max in m/s (80 km/h)
_TIME_MAX = 2.0  # time_frac cap: MAX_STEPS/DEADLINE_STEPS = 1.85 < 2.0
_ACC_MAX = 2.0  # m/s². NeTrainSim's jerk limit keeps |acceleration| <= 2 per step

# simulator stderr logs go here, one file per worker process
_LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")


class NeTrainSimEnv(gym.Env):
    metadata = {"render_modes": []}

    # All 23 features are normalized to roughly [-1, 1] or [0, 1] by _state_to_obs.
    # [speed, position, grade, curvature, energy, speed_limit, time_frac,
    #  behind_frac, acceleration, notch, grade_ahead x3, speed_limit_min_ahead x3,
    #  previous action one-hot x7]
    # energy is the energy consumed in the last step (never negative).
    observation_space = Box(
        low=np.array(
            [0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]
            + [-1.0] * 3 + [0.0] * 3 + [0.0] * N_ACTIONS,
            dtype=np.float32,
        ),
        high=np.array(
            [2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0]
            + [1.0] * 3 + [2.0] * 3 + [1.0] * N_ACTIONS,
            dtype=np.float32,
        ),
    )
    action_space = Discrete(N_ACTIONS)  # 0 = coast, 1-6 = target fraction of the limit

    def __init__(self, trains_file: str = TRAINS_FILE):
        super().__init__()
        # trains_file selects the trip: TRAINS_FILE (A→B) or TRAINS_FILE_RETURN
        # (B→A, path 1503→1). Route length/links are identical either way.
        self._trains_file = trains_file
        self._proc: subprocess.Popen | None = None
        self._out_queue: queue.Queue | None = None  # lines from the reader thread
        self._stderr_log = None  # file handle for simulator stderr
        self._last_state: dict | None = None
        self._cum_energy_kwh: float = 0.0  # energy consumed so far (no regeneration)
        self._last_step_energy_kwh: float = 0.0  # consumed in the last simulator second
        self._step_count: int = 0
        self._episode_count: int = 0
        self._episode_start: float = 0.0
        self._episode_reward: float = 0.0
        self._prev_action: int | None = None  # for the smoothness penalty and the obs
        # Evaluation only: when True, every simulator second is kept in self.trace
        # (rl/evaluate.py turns it into a per-second profile). Off during training.
        self.record_trace: bool = False
        self.trace: list[dict] = []

        if not os.path.isfile(SIMULATOR_BIN):
            raise FileNotFoundError(
                f"Simulator binary not found: {SIMULATOR_BIN}\n"
                f"The simulator lives in its own repo, not this one:\n"
                f"    git clone https://github.com/newuu-se/NeTrainSim\n"
                f"    cd NeTrainSim && make {_BUILD}\n"
                f"or point NETRAINSIM_BIN at an existing checkout."
            )
        for path in (NODES_FILE, LINKS_FILE, self._trains_file):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"NeTrainSim input file not found: {path}\n"
                    f"The route data lives in its own repo, not this one. Set "
                    f"NODE_FILE, LINK_FILE, TRAIN_FILE, and TRAIN_RETURN_FILE "
                    f"to point to the appropriate files."
                )

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.close()
        self._step_count = 0
        self._episode_reward = 0.0
        self._cum_energy_kwh = 0.0
        self._prev_action = None
        self.trace = []
        self._episode_start = time.time()
        self._episode_count += 1
        self._start_interactive_simulator()
        state = self._send_command_and_read_state({"coast": True})
        self._last_state = state
        # energy consumed in the bootstrap step (the simulator's running total)
        step_energy = float(state["energy_consumed_kwh_cum"])
        self._cum_energy_kwh = step_energy  # include bootstrap step-0 energy
        self._last_step_energy_kwh = step_energy
        # Bootstrap consumed one simulator timestep (coast sent to get initial
        # state). Start _step_count at 1 so it tracks actual simulator steps.
        self._step_count = 1
        obs = self._state_to_obs(state, step_energy)
        # Return empty info dict — tianshou 0.5.1 Batch cannot create new keys
        # via index assignment, so any non-empty info dict causes a ValueError.
        return obs, {}

    def step(self, action: int):
        action = int(action)
        if action < 0 or action >= N_ACTIONS:
            raise ValueError(f"Action must be in [0, {N_ACTIONS - 1}], got {action}")

        # Action-change (smoothness) penalty, once per decision: discourages erratic
        # target jumping so the speed profile stays smooth/realistic. Skipped on the
        # first decision of an episode (no previous action to compare against).
        change_penalty = (
            W_SMOOTH * abs(action - self._prev_action)
            if self._prev_action is not None
            else 0.0
        )
        self._prev_action = action
        self._episode_reward -= change_penalty

        # Coarse control (action repeat): hold the chosen action for CONTROL_INTERVAL
        # simulator seconds, summing reward, then decide again. Fewer, longer
        # decisions → tractable credit assignment → terrain-aware speed modulation.
        total_reward = -change_penalty
        terminated = truncated = False
        step_energy_kwh = 0.0
        for _ in range(CONTROL_INTERVAL):
            r, terminated, truncated, step_energy_kwh = self._sim_step(action)
            total_reward += r
            if terminated or truncated:
                break

        obs = self._state_to_obs(self._last_state, step_energy_kwh)
        # Empty info dict — tianshou 0.5.1 Batch rejects new keys via index assignment.
        return obs, float(total_reward), terminated, truncated, {}

    def _sim_step(self, action: int):
        """Advance the simulator one second under the given action; return
        (reward, terminated, truncated, step_energy_kwh)."""
        state = self._send_command_and_read_state(action_command(action))
        self._last_state = state
        self._step_count += 1
        if self.record_trace:
            self.trace.append({"time_s": self._step_count, "action": action, **state})

        # Energy CONSUMED this second: the change in the simulator's running
        # consumed total. Its energy_kwh field is net of regeneration, so it is
        # not used — regenerated energy is never credited to the agent.
        consumed_cum = float(state["energy_consumed_kwh_cum"])
        step_energy_kwh = consumed_cum - self._cum_energy_kwh
        self._cum_energy_kwh = consumed_cum
        self._last_step_energy_kwh = step_energy_kwh
        speed_mps = float(state["speed_mps"])
        max_speed = float(state["speed_limit_mps"])
        position_m = float(state["position_m"])

        # Episode boundaries.
        terminated = bool(state["terminated"]) or position_m >= TOTAL_ROUTE_LENGTH_M
        truncated = self._step_count >= MAX_STEPS and not terminated

        # Per-step pace penalty: penalize lagging the constant pace needed to reach
        # the terminus by DEADLINE_STEPS. Zero when on/ahead of schedule, so the
        # agent coasts freely where it has slack; lagging is penalized immediately.
        target_pos_m = (self._step_count / DEADLINE_STEPS) * TOTAL_ROUTE_LENGTH_M
        behind_frac = max(0.0, target_pos_m - position_m) / TOTAL_ROUTE_LENGTH_M
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
            # energy= is the energy consumed (what the reward uses). Regenerated and
            # net energy are reported only, at the end: make_plots.py parses
            # "energy=... kWh  reward=...", so they must not go in between.
            print(
                f"[{status}]  ep={self._episode_count:>4d}"
                f"  {self._step_count:>5,} steps"
                f"  {position_m:>7,.0f}m ({pct:4.1f}%)"
                f"  energy={self._cum_energy_kwh:>7.1f} kWh"
                f"  reward={self._episode_reward:>+8.1f}"
                f"  {elapsed:.1f}s"
                f"  (not used: regenerated="
                f"{float(state['energy_regenerated_kwh_cum']):.1f} kWh"
                f"  net={float(state['energy_net_kwh_cum']):.1f} kWh)",
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
        command = [
            SIMULATOR_BIN,
            "-n",
            NODES_FILE,
            "-l",
            LINKS_FILE,
            "-t",
            self._trains_file,
            "-p",
            "1.0",
            "-I",
        ]
        if SIMULATOR_OUTPUT_DIR:
            # NeTrainSim refuses an -o folder that does not exist yet
            os.makedirs(SIMULATOR_OUTPUT_DIR, exist_ok=True)
            command += ["-o", SIMULATOR_OUTPUT_DIR]
        self._proc = subprocess.Popen(
            command,
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
        except ValueError, OSError:
            pass  # stream closed during shutdown
        out_queue.put(None)  # EOF sentinel

    def _send_command_and_read_state(self, command: dict) -> dict:
        if self._proc is None or self._proc.stdin is None or self._out_queue is None:
            raise RuntimeError("Interactive simulator process is not running.")

        payload = json.dumps(command)
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
            raw_json = line[len(STATE_PREFIX) :]
            try:
                state = json.loads(raw_json)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid simulator state JSON: {raw_json}") from exc
            return state

    def _state_to_obs(self, state: dict, step_energy_kwh: float) -> np.ndarray:
        position = float(state["position_m"])
        # Schedule features — the pace penalty is a function of the step counter,
        # so the agent must SEE the clock or the reward is non-Markov in its obs:
        # time_frac   = elapsed time as a fraction of the deadline
        # behind_frac = how far (fraction of route) the train lags the deadline
        #               pace; exactly the quantity the pace penalty charges.
        time_frac = self._step_count / DEADLINE_STEPS
        target_pos = time_frac * TOTAL_ROUTE_LENGTH_M
        behind_frac = max(0.0, target_pos - position) / TOTAL_ROUTE_LENGTH_M
        if ABLATE_TIME_FEATURES:
            # Non-Markov ablation: hide the clock from the policy (the reward still
            # depends on it). Keeps the 23-dim shape so the network is unchanged.
            time_frac = behind_frac = 0.0
        # Look-ahead along the path (the simulator reports 0.5/1/2 km windows):
        # what is coming decides whether coasting or holding speed pays off.
        grade_ahead = [float(g) / _GRADE_MAX for g in state["grade_ahead_perc"]]
        limit_ahead = [
            float(v) / _MAXSPEED_MAX for v in state["speed_limit_min_ahead_mps"]
        ]
        prev_action = np.zeros(N_ACTIONS)
        if self._prev_action is not None:
            prev_action[self._prev_action] = 1.0
        obs = np.concatenate(
            [
                [
                    float(state["speed_mps"]) / _SPEED_MAX,
                    position / TOTAL_ROUTE_LENGTH_M,
                    float(state["grade_perc"]) / _GRADE_MAX,
                    float(state["curvature_perc"]),
                    step_energy_kwh / _ENERGY_MAX,
                    float(state["speed_limit_mps"]) / _MAXSPEED_MAX,
                    min(time_frac, _TIME_MAX),
                    behind_frac,
                    float(state["acceleration_mps2"]) / _ACC_MAX,
                    # the notch the simulator picked for this speed (not chosen
                    # by the agent): shows how hard the train is pulling
                    float(state["notch"]) / max(1.0, float(state["notch_max"])),
                ],
                grade_ahead,
                limit_ahead,
                prev_action,
            ]
        ).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)
