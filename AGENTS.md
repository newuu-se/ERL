# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RL-based train energy optimization: a Tianshou PPO agent controls locomotive throttle
(notch 0–8) each second to minimize total energy consumption over a full A→B trip while
respecting speed limits and arriving within the scheduled time window.

Three components:
1. **NeTrainSim** (`NeTrainSim-adjusted/`) — C++/Qt6 train simulator (GOST resistance,
   regenerative braking removed, interactive RL mode via `-I`)
2. **RL layer** (`rl/`) — Python: Gymnasium env + Tianshou PPO training/eval scripts
3. **Docs** (`gymnasium-docs/`, `tianshou-docs/`) — API reference (do not modify)

## Repository Layout

```
data/
  real_data/                      ACTIVE dataset — 1503 nodes / 1502 links, 75,064 m
    nodesFile_real_data.dat       nodes (terminals 1 and 1503)
    linksFile_real_data.dat       ← what training uses
  netrainsim_v2/                  Mixed: the two trains files are ACTIVE, the rest is dead
    trainsFile_rl.dat             ACTIVE — ER9E EMU (3 locos 1213 kW + 3 cars = 373 t),
                                  path repathed to 1→1503 for real_data
    train_return.dat              ACTIVE — same consist, reversed path (1503→1)
    nodesFile_v2_fixed.dat        superseded route (1500 nodes)
    linksFile_v2_clean.dat        superseded route (1499 links, spike-cleaned grades)
    linksFile_v2_fixed_speed.dat  superseded route (RAW grades, ±17% DEM spikes)

NeTrainSim-adjusted/
  src/
    NeTrainSimConsole/main.cpp    CLI entry; `-I/--interactive` RL JSON loop lives here
    NeTrainSim/
      simulator.h / simulator.cpp Time-step loop, CSV output, runOneTimeStep API
      simulatorapi.h              C++ programmatic API (singleton)
      traindefinition/
        train.h / train.cpp       Physics; per-step energyStat = NEC − NER
        locomotive.h / locomotive.cpp  rlOverrideEnabled/rlOverrideThrottle injection,
                                       GOST resistance, regen removed (returns 0)
        car.cpp                   GOST resistance for cars
      network/
        readwritenetwork.cpp      Parses .dat files; speed in m/s, length in meters
        netlink.cpp               freeFlowSpeed stored and used as m/s
  build-mac.sh / build-linux.sh  Build scripts (macOS / Linux)

rl/
  train_env.py                   NeTrainSimEnv (gymnasium.Env, interactive subprocess)
  train.py                       Tianshou PPO training script (build_policy = source of truth)
  train_return.py                Same trainer, B→A trip → checkpoints/return/
  evaluate.py                    Loads checkpoint, runs greedy/stochastic episode → CSV
  run_baselines.py               Constant-notch frontier → results/baselines_<trip>.json
  make_plots.py                  The 4 figures → results/plots/ (PNG + PDF)
  reward_theory.py               Closed-form reward-coefficient analysis (not in the default flow)

requirements.txt                 THE only dependency file (repo root). Python >= 3.10.
venv/                            Python virtualenv (gymnasium 1.1.1, tianshou 0.5.1, torch 2.x)
```

**`results/` is not committed** — it is produced entirely by `run_baselines.py`,
`evaluate.py` and `make_plots.py`. A fresh clone has none of it; `make_plots.py` skips
each figure whose inputs are missing and prints the command that creates them.

Every number in `make_plots.py` is read from a file (`results/baselines_<trip>.json`,
`results/notch_profile_stochastic.csv`, `logs/train_run_*.log`) — never pasted in as a
constant. Re-measure with `rl/run_baselines.py` and the plots follow automatically.

The four figures: `fig_training_progress_<trip>` (did training work — reward, energy,
trip time, arrival rate), `fig_energy_time_tradeoff` (frontier + policy),
`fig_speed_profile_<trip>` (speed/notch/grade along the route),
`fig_energy_usage_<trip>` (cumulative energy and kWh/km vs grade).

## Build & Run

### NeTrainSim (C++)

Linux (this machine): `./build-linux.sh` from `NeTrainSim-adjusted/` → binary at
`build-linux/src/NeTrainSimConsole/NeTrainSim`.
macOS: `brew install qt cmake`, then `./build-mac.sh`.

Run with the active data + trajectory export (run from inside `NeTrainSim-adjusted/`):
```bash
./build-linux/src/NeTrainSimConsole/NeTrainSim \
  -n ../data/real_data/nodesFile_real_data.dat \
  -l ../data/real_data/linksFile_real_data.dat \
  -t ../data/netrainsim_v2/trainsFile_rl.dat \
  -o res -e true -p 1.0
```

CLI flags: `-n` nodes, `-l` links, `-t` trains, `-o` output dir, `-p` timestep in seconds
(default 1.0), `-z` enable optimizer, **`-I` interactive RL mode** (JSON over stdin/stdout).
**`-e true`** exports the trajectory CSV — `-e` takes a value, NOT a bare flag. `-e` alone uses
default `false` and produces no CSV.

### Data preparation

None — there is no generation step. The two `.dat` files in `data/real_data/`, plus the two
trains files in `data/netrainsim_v2/`, are checked in and used as-is.

`real_data` grades have NOT been spike-cleaned (the old dataset's
`clean_grade_spikes.py` is gone — see git history). They span −3.49…+8.49 %, which is
plausible rail terrain, but the four links above |4 %| are worth eyeballing if the physics
misbehaves: on the old route, uncleaned DEM spikes broke the force balance and tripled
apparent power draw. Re-measure baselines after any data change:
`python rl/run_baselines.py`.

### Python RL layer

```bash
python3.12 -m venv venv              # REQUIRES Python >= 3.10 (PEP 604 unions);
source venv/bin/activate             # macOS `python3` is 3.9 and will fail at import
pip install -r requirements.txt      # the only dependency file
python rl/train.py 2>&1 | tee logs/train_run_$(date +%Y%m%d_%H%M%S).log
```

Validate Gymnasium env compliance before training:
```bash
python -c "
import sys; sys.path.insert(0,'.')
from rl.train_env import NeTrainSimEnv
from gymnasium.utils.env_checker import check_env
check_env(NeTrainSimEnv())
print('OK')
"
```

## Data Files

Active route data lives in `data/real_data/`; the trains files are still in
`data/netrainsim_v2/` (repathed `1→1503`). All files are ASCII, tab-separated, and
hand-derived — there is no generator script. Formats below (unchanged between datasets).

**real_data facts:** 1503 nodes / 1502 links, 75,064 m, segments 34–66 m; speed limits
1.42–21.16 m/s across **960 distinct values** (a near-continuous profile, not the four
regulatory zones of the old data); grades −3.49…+8.49 %, net elevation **+181.4 m**;
curvature 0–0.40; catenary on every link; `directions=2`.
**Speed-limit timing floor ≈ 7,100 s** — no deadline below that is reachable.

**`nodesFile_real_data.dat`** (1503 nodes; the old `nodesFile_v2_fixed.dat` had 1500):
```
This is the node file of route1		
<count>  <xScale>  <yScale>           ← scales=1 (coords already in meters)
<id>  <x>  <y>  <isTerminal>  <dwellTime>  <desc>
```
Nodes 1 and 1503 are marked `isTerminal=1`.

**`linksFile_real_data.dat`** (1502 links) — the file training reads:
```
This is the link file of route1     (many tabs)
<count>  <lengthScale>  <speedScale>   ← both=1
<id>  <from>  <to>  <length_m>  <speed_mps>  <signalNo>  <grade_pct>  <curvature>
       <directions>  <speedVariation>  <hasCatenary>
```
- Link lengths are Euclidean distances between consecutive coordinates (meters), 34–66 m.
- **Speed is m/s** (confirmed in netlink.cpp: `length/freeFlowSpeed` gives seconds).
  Near-continuous, 1.42–21.16 m/s, 960 distinct values.
- **Grade is percent (%)**, range −3.49 … +8.49 (not spike-cleaned).
- `hasCatenary=1` (electric), `directions=2` (bidirectional).

**`trainsFile_rl.dat` / `train_return.dat`**:
```
Automatic Trains Definition
1                                           ← number of trains
1  <path>  <startTime>  <frictionCoef>  <loco_defs>  <car_defs>
```
Loco field order: `Count, Power(kW), TransmissionEff, NoOfAxles, AirDragCoeff, FrontalArea(m²), Length(m), GrossWeight(t), Type`
Car field order:  `Count, NoOfAxles, AirDragCoeff, FrontalArea(m²), Length(m), GrossWeight(t), TareWeight(t), Type`
Current config: ER9E EMU — 3 locos (1213 kW, 60 t, type 1 = electric) + 3 cars
(2×78 t + 37 t) = 373 t. `trainsFile_rl.dat` runs path 1→1500; `train_return.dat` runs
1500→1. Locomotive default max speed 33.33 m/s; route speed limits are the binding constraint.

## Architecture

### C++ Simulation Engine

**Entry point:** `src/NeTrainSimConsole/main.cpp` — QCommandLineParser, calls
`SimulatorAPI::ContinuousMode::createNewSimulationEnvironmentFromFiles()`, then
`sim->runSimulation()` (blocking).

**Core classes:**
- **`Simulator`** (`simulator.h/cpp`) — time-step loop: `runSimulation()` calls
  `runOneTimeStep()` → `playTrainOneTimeStep()` → writes one CSV row per train per step.
  Exposes `pauseSimulation()` / `resumeSimulation()` / `runOneTimeStep()` (Q_INVOKABLE).
- **`Network`** (`network/network.h`) — nodes + links + signals; train route is fixed.
- **`Train`** (`traindefinition/train.h`) — `getStepAcceleration()` → `moveTrain()`;
  holds `optimumThrottleLevels` queue (built-in optimizer injection point).
- **`Locomotive`** (`locomotive.h`) — `currentLocNotch` (int 0–8), `throttleLevel`
  (double 0–1). Notch-throttle mapping: quadratic `(N/8)^2`.

**Trajectory CSV columns** (written per timestep when `-e` flag used):
```
TrainNo, TStep_s, TravelledDistance_m, Acceleration_mps2, Speed_mps,
LinkMaxSpeed_mps, EnergyConsumption_KWH, DelayTimeToEach_s, DelayTime_s,
Stoppings, tractiveForce_N, ResistanceForces_N, CurrentUsedTractivePower_kw,
GradeAtTip_Perc, CurvatureAtTip_Perc, FirstLocoNotchPosition, optimizationEnabled
```
`EnergyConsumption_KWH` is the **per-step** net energy consumed this timestep (kWh), not cumulative.
It equals `train->energyStat = NEC - NER` where NEC/NER are reset at the start of each step
(`resetTrainEnergyConsumption`, train.cpp). NER ≡ 0 since regen removal.
Values range ~0–1.4 kWh depending on throttle; a full trip totals ~760–860 kWh (clean data).
The RL env tracks `_cum_energy_kwh` in Python by summing per-step values.
NOTE: energy is charged from *virtual* tractive power `(m·a + R)·v` — with impossible grades
(the raw v2 file's ±17% DEM spikes) this exceeded installed power 3×; clean data keeps it sane.

### Integration Strategy

**Phase 2 — interactive per-step control (IMPLEMENTED, the only mode in use):**
- `main.cpp` `-I/--interactive`: loops `read action JSON from stdin → set
  rlOverrideThrottle on every locomotive → runOneTimeStep() → write state JSON
  to stdout` (line-prefixed `NTS_JSON `).
- The RL override (`locomotive.cpp`) bypasses both the built-in A* optimizer and
  the discretized throttle governor: `getThrottleLevel()` returns
  `rlOverrideThrottle`, `updateLocNotch()` is a no-op, `train->optimize=false`.
  Notch→throttle mapping applied in main.cpp: `(notch/Nmax)²` — identical to
  `defineThrottleLevels()`.
- `NeTrainSimEnv.reset()` spawns the binary; a bootstrap `{"notch": 0}` fetches
  the initial state. A reader thread + queue (not select+readline) consumes
  stdout — immune to buffered-line races.
- Env-level action repeat: each `step()` holds the notch for CONTROL_INTERVAL=15
  simulator seconds (~430 decisions per on-schedule trip).

**stdout state JSON (simulator → Python):**
```json
{
  "timestep": 42, "speed_mps": 15.3, "position_m": 1230.0,
  "grade_perc": 1.2, "curvature_perc": 0.0, "remaining_dist_m": 73639.6,
  "energy_kwh": 0.4, "link_max_speed_mps": 11.1, "terminated": false, "notch": 3
}
```
`energy_kwh` is the energy of THIS step only (see trajectory CSV note above).
**stdin action JSON (Python → simulator):** `{"notch": 6}`

(Phase 1 — replaying a pre-computed trajectory CSV — is obsolete and removed.)

### Gymnasium Environment (`rl/train_env.py`)

**Observation space** (9 floats, `Box`, all normalized ~[-1,1]/[0,1]):
```
[speed_mps, position_m, grade_perc, curvature_perc,
 remaining_dist_m, energy_kwh, link_max_speed_mps,
 time_frac, behind_frac]
```
`time_frac` = step/DEADLINE_STEPS, `behind_frac` = fraction of route behind the
deadline pace. These make the clock visible — the pace penalty depends on the
step counter, so without them the reward would be non-Markov in the observation.
Normalization denominators (`_GRADE_MAX=4.5`, `_ENERGY_MAX=1.5`, …) are
calibrated to the CLEAN v2 data; recheck them whenever physics/data change.

**Action space:** `Discrete(9)` — notch 0–8, held for CONTROL_INTERVAL=15 sim-seconds
per decision (action repeat; credit assignment was intractable at 1 s decisions).

**Reward (per sim-second, summed over the 15-s decision)** — objective is
*minimize trip energy subject to the schedule deadline*:
```
r = -W_ENERGY * energy_kwh                          # energy this step       (W_ENERGY=3.0)
  - W_PACE * behind_frac                            # lagging deadline pace  (W_PACE=2.0; 0 when on pace)
  - W_OVERSPEED * max(0, speed - link_limit)        # safety net; sim caps speed  (1.0)
  + ARRIVAL_BONUS   on terminated                   # +200
  - TIMEOUT_PENALTY on truncated                    # -1500
plus, once per decision: -W_SMOOTH * |Δnotch|       # smoothness         (0.15)
```
The per-step pace penalty (not a terminal late fee) is what stops the
crawl-at-notch-1 failure mode; zero penalty when on/ahead of schedule preserves
the freedom to coast. Eco target = slowest on-schedule constant notch ≈ n3
(789 kWh / 6,442 s on clean data); the agent should beat it by modulating with
grade/speed zone.

**Episode boundaries:**
- `terminated=True`: simulator reports the train reached its destination, or
  `position_m >= TOTAL_ROUTE_LENGTH_M` (75,064.00 m — real_data)
- `truncated=True`: `_step_count >= MAX_STEPS` (12,000) without arriving

### Tianshou PPO (`rl/train.py`) — tianshou 0.5.1 API

`build_policy()` in `rl/train.py` is the single source of truth (imported by
`evaluate.py` so train/eval can never drift):

```python
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic  # use Actor, not DiscreteActor

# separate actor/critic nets, HIDDEN_SIZES=[256,128,64], OBS_SHAPE=(9,), 9 actions
policy = PPOPolicy(actor, critic, optim, dist_fn=torch.distributions.Categorical,
                   discount_factor=0.9999, eps_clip=0.2, ent_coef=0.004,
                   gae_lambda=0.95, vf_coef=0.5, max_grad_norm=0.5,
                   advantage_normalization=True, reward_normalization=True)
```
**Hard-won hyperparameter rationale (don't regress these):**
- `discount=0.9999`: episodes are ~6k sim-steps of energy SUM; at γ≤0.999 the
  discounting biases toward finishing fast (high notch). Near-undiscounted keeps
  the energy objective honest.
- `reward_normalization=True` required for stability at that γ (returns ~±1000).
- `ent_coef=0.004`: 0.008 leaves the argmax parked at the energy floor; 0.002
  collapses to a constant notch.
- `episode_per_collect` (not `step_per_collect`): full episodes per update.
- `train_fn` fires at epoch START, `test_fn` after the epoch's updates —
  checkpoints are saved in `test_fn` so labels match content.

## RL Design Decisions

| Dimension | Choice | Rationale |
|-----------|--------|-----------|
| Action space | `Discrete(9)` notch 0–8, 15 s action repeat | Matches simulator internals; coarse control makes long-horizon credit assignment tractable |
| State space | 9 normalized features (7 physics + 2 schedule) | Pace-penalty reward needs the clock in the obs (Markov) |
| Reward | Dense per-step energy + pace penalty + terminal | Terminal-only deadline was ignored; raw progress reward biased toward speed |
| Algorithm | PPO (`PPOPolicy`) | REINFORCE diverged; PPO + reward-norm stable |
| Parallelism | `SubprocVectorEnv(8)` | Each env = one C++ subprocess |
| Control mode | Phase 2 interactive (stdin/stdout JSON) | Real per-step physics control |

## C++ Modifications (Phase 2 — DONE)

Implemented in:
- `src/NeTrainSimConsole/main.cpp` — `-I` flag, JSON loop, notch→`(N/Nmax)²` throttle,
  sets `rlOverrideEnabled/rlOverrideThrottle` + `currentLocNotch` on all locomotives,
  `train->optimize=false`, drains Qt events each step.
- `src/NeTrainSim/traindefinition/locomotive.{h,cpp}` — override fields;
  `getThrottleLevel()` returns the override; `updateLocNotch()` no-op under override.
- Physics edits (2026-06-10): GOST/SI resistance `w0 = 1.1 + 0.01v + 0.000227v²`
  (v in km/h) in `Car::getResistance` and `Locomotive::getResistance`; grade term
  `trackGrade(%)×10×W_kN`; curve term `|curv°|×(700/1746.4)×W_kN`; regenerative
  braking removed (`getRegenerativeEffeciency` returns 0).

Rebuild after C++ changes: `cd NeTrainSim-adjusted && ./build-linux.sh`

## Key Data Facts

- Route (ACTIVE, `data/real_data/`): 75.064 km, 1503 nodes, 1502 links, 34–66 m per segment.
  Input files: `data/real_data/{nodesFile_real_data, linksFile_real_data}.dat`
  + `data/netrainsim_v2/trainsFile_rl.dat` (path 1→1503).
- Train: ER9E electric multiple unit — 3 locos (1213 kW, 60 t, type 1 = electric) +
  3 cars (2×78 t + 37 t) = 373 t; links have `hasCatenary=1`
- Speed limits: near-continuous, 1.42–21.16 m/s (5–76 km/h), 960 distinct values, mean
  12.31 m/s. NOT the four zones of the old data. Possibly a MEASURED speed profile rather
  than legal limits — unconfirmed, and it matters: the sim treats the column as a hard cap,
  so if it is measured, the agent can only ever drive slower than the real driver did.
- Grades: −3.49…+8.49 %, net elevation +181.4 m. Four links exceed |4 %|; the steepest is
  link 853 at +8.49 % (that is why `_GRADE_MAX` is 8.5 — at 4.5 those links clipped).
- SUPERSEDED (`data/netrainsim_v2/`, route files only): 74.89 km, 1500/1499, four speed
  zones, net +346.6 m, grades −2.7…+4.5 % after DEM-spike cleaning.
- Trip-time bounds (ACTIVE real_data): theoretical floor ≈ **7,100 s** (Σ length/limit; a
  kinematic pass with EMU accel/brake limits agrees at ~7,050 s). `DEADLINE_STEPS = 9,200`
  is PROVISIONAL — derived from that floor, not measured. **Re-set it from
  `rl/run_baselines.py` before trusting any training run**: it must exceed the fastest
  constant notch, else every episode is late by construction and the pace penalty carries
  no signal. `_ENERGY_MAX` likewise needs setting from the reported `maxE/step`.
- Constant-notch frontier for real_data: **NOT YET MEASURED**.
- (Superseded v2 route, for reference only — does NOT transfer: floor ≈ 5,025 s, n8=5,602 s/
  862 kWh, n3=6,442/789, n2=8,492/758, deadline was 6,500 s, eco target n3.)
- The simulator hard-caps speed at each link's freeFlowSpeed, so over-speeding is rare
- At standstill (v=0) the loco applies full adhesion force regardless of notch
  (original NeTrainSim behavior) — the train cannot be parked mid-route
- NeTrainSim speed unit in .dat files: **m/s** (confirmed in netlink.cpp)
- NeTrainSim .dat files: ASCII text, tab-separated, NOT binary

## Testing & Validation

```bash
# 1. Verify simulator runs with the active data (run from inside NeTrainSim-adjusted/)
cd NeTrainSim-adjusted && mkdir -p res
./build-linux/src/NeTrainSimConsole/NeTrainSim \
  -n ../data/real_data/nodesFile_real_data.dat \
  -l ../data/real_data/linksFile_real_data.dat \
  -t ../data/netrainsim_v2/trainsFile_rl.dat \
  -o res -e true -p 1.0        # note: -e takes 'true', not a bare flag
ls res/trainTrajectory_*.csv

# 2. Validate Gymnasium env
python -c "
import sys; sys.path.insert(0,'.')
from rl.train_env import NeTrainSimEnv
from gymnasium.utils.env_checker import check_env
check_env(NeTrainSimEnv())
print('OK')
"

# 3. Measure constant-notch baselines → results/baselines_<trip>.json
#    (make_plots.py reads this file directly; nothing is copy-pasted)
python rl/run_baselines.py
python rl/run_baselines.py --return-trip

# 4. Run training (use tmux for long runs). TEE THE LOG — make_plots reads it
#    for the training-progress figure; without it that figure cannot be drawn.
mkdir -p logs
python rl/train.py 2>&1 | tee logs/train_run_$(date +%Y%m%d_%H%M%S).log

# 5. Evaluate and plot
python rl/evaluate.py --stochastic   # → results/notch_profile_stochastic.csv
                                     #   (position, speed, notch, cumulative energy)
python rl/make_plots.py              # → results/plots/*.png + *.pdf
```
