# china-grant-rl-model

Reinforcement learning for energy-optimal train driving. A **Tianshou PPO** agent sets the
locomotive throttle (notch 0–8) along a real 75.1 km Uzbek rail route to minimise total
trip energy while arriving inside the scheduled time window.

The agent drives a modified build of **NeTrainSim**, a C++/Qt6 train simulator, one
simulated second at a time over a JSON pipe. The physics is real: GOST resistance, true
grades and speed limits, no regenerative braking.

---

## How the pieces fit

```
  rl/train.py            rl/train_env.py               NeTrainSim (C++)
 ┌──────────────┐       ┌──────────────────┐        ┌───────────────────┐
 │ PPO: choose  │──────>│ Gymnasium env:   │─stdin─>│ one train,        │
 │ notch 0–8    │       │ JSON bridge +    │  JSON  │ one second of     │
 │ from 9 obs   │<──────│ reward           │<stdout─│ physics per call  │
 └──────────────┘       └──────────────────┘  JSON  └───────────────────┘
```

The simulator is a **separate process**. Python launches it, sends `{"notch": 6}`, and reads
back a state line prefixed `NTS_JSON `. Training runs 8 such subprocesses in parallel.

| Component | Where |
|---|---|
| C++ simulator (forked, with `-I` interactive mode) | `NeTrainSim-adjusted/` |
| Gymnasium environment + reward | `rl/train_env.py` |
| PPO training | `rl/train.py` |
| Route data (active) | `data/real_data/` |
| Train consist + path | `data/netrainsim_v2/trainsFile_rl.dat` |

---

# Running the project from scratch

Follow these in order. Steps 1–5 are setup and take about 20 minutes; step 6 is a
decision point; step 7 is the long training run.

## 1. Install prerequisites

macOS:

```bash
brew install cmake qt python@3.12
```

You need **Python 3.10 or newer**. macOS ships 3.9 as `python3`, and the code uses PEP 604
unions (`int | None`) that raise `TypeError` on import under 3.9 — so always call the
interpreter explicitly (`python3.12`), never bare `python3`.

Linux: install `cmake`, `qt6-base-dev`, and a Python ≥ 3.10 through your package manager.

## 2. Build the C++ simulator

```bash
cd NeTrainSim-adjusted
./build-mac.sh        # macOS
# ./build-linux.sh    # Linux
cd ..
```

Confirm the binary exists before going on — nothing else works without it:

```bash
ls NeTrainSim-adjusted/build-mac/src/NeTrainSimConsole/NeTrainSim
```

## 3. Create the Python environment

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` at the repo root is the **only** dependency file.

## 4. Verify the environment

```bash
python -c "
import sys; sys.path.insert(0,'.')
from rl.train_env import NeTrainSimEnv
from gymnasium.utils.env_checker import check_env
check_env(NeTrainSimEnv()); print('OK')
"
```

This launches the simulator, runs a few steps, and checks the Gymnasium contract. It must
print `OK`. If it fails here, the problem is the build or the data paths — not the RL code.

## 5. Measure the constant-notch baseline

```bash
python rl/run_baselines.py                  # → results/baselines_forward.json
python rl/run_baselines.py --return-trip    # → results/baselines_return.json
```

This drives the train at each fixed notch (2,3,4,5,6,8) and records trip time and energy.
It is the frontier the learned policy has to beat, and every later figure references it.

## 6. ⏸️ Set the deadline from those numbers — do not skip

Open `rl/train_env.py` and set two constants from the baseline output:

- **`DEADLINE_STEPS`** — currently `9_200`, and **provisional**. It was derived from
  physics alone (the speed limits forbid arriving before ~7,100 s), not from a
  measurement. It **must** sit above the fastest constant notch's trip time. If it does
  not, every episode is late by construction, the pace penalty is permanently active, and
  the agent gets no learnable schedule signal — training will appear to run and teach
  nothing.
- **`_ENERGY_MAX`** — set to a little above the largest `maxE/step` the baseline reports.
  Too low and every heavy-power moment saturates to the same observation value.

## 7. Train

```bash
mkdir -p logs
python rl/train.py 2>&1 | tee logs/train_run_$(date +%Y%m%d_%H%M%S).log
```

Takes hours. Run it inside `tmux` (or `nohup`) so it survives closing the terminal.

**Keep the `| tee`.** `logs/train_run_*.log` is the only source for the training-progress
figure; without it that figure cannot be drawn.

Return trip, if wanted:

```bash
python rl/train_return.py 2>&1 | tee logs/train_return_$(date +%Y%m%d_%H%M%S).log
```

## 8. Evaluate and plot

```bash
python rl/evaluate.py --stochastic                  # → results/notch_profile_stochastic.csv
python rl/evaluate.py --stochastic --return-trip    # → results/return/…
python rl/make_plots.py                             # → results/plots/*.png + *.pdf
```

**Use `--stochastic`.** The deterministic argmax policy is degenerate across seeds; the
sampled policy is the deployable artifact (see *Known limitations*).

`make_plots.py` skips any figure whose inputs are missing and prints the command that
produces them, so running it early is safe and tells you what is left to do.

---

## Scripts

| File | Purpose |
|---|---|
| `rl/train_env.py` | Gymnasium env: subprocess bridge, 9-feature observation, reward |
| `rl/train.py` | PPO training. `build_policy()` is shared with evaluation so the two cannot drift |
| `rl/train_return.py` | Same trainer, reverse trip |
| `rl/evaluate.py` | Load a checkpoint, run one episode, export position/speed/notch/energy |
| `rl/run_baselines.py` | Measure the constant-notch energy/time frontier → JSON |
| `rl/make_plots.py` | The four figures below. Reads every number from a file — no pasted constants |
| `rl/reward_theory.py` | Closed-form analysis of the reward coefficients (not in the default flow) |

Nothing under `results/`, `checkpoints/` or `logs/` is committed — all three are produced
by the commands above.

### Figures

Each answers one question, in the order you would ask them:

| Figure | Question it answers |
|---|---|
| `fig_training_progress_<trip>` | **Did training work?** Reward, trip energy, trip time and arrival rate over the run |
| `fig_energy_time_tradeoff` | **Is the policy good?** The constant-notch frontier with the policy on it |
| `fig_speed_profile_<trip>` | **What does it do?** Speed vs limit, chosen notch, and the terrain driving it |
| `fig_energy_usage_<trip>` | **Where does the energy go?** Cumulative energy and kWh/km against grade |

If panel (b) of the training figure does not fall and panel (d) does not rise toward
100 %, the run failed — read that before anything else.

---

## The route and the train

Active dataset: **`data/real_data/`**

- **Route:** 75.064 km, 1503 nodes / 1502 links (34–66 m per segment)
- **Speed limits:** 1.42 – 21.16 m/s (5 – 76 km/h), **960 distinct values** — a
  near-continuous per-segment profile, not regulatory zones
- **Grades:** −3.49 % … +8.49 %, net elevation **+181.4 m**
- **Curvature:** 0 – 0.40
- **Train:** ER9E electric multiple unit — 3 × 1213 kW motor cars + 3 trailers, 373 t
- **Electrification:** catenary on every link

```
data/real_data/
  nodesFile_real_data.dat     1503 nodes (x, y in metres), terminals at 1 and 1503
  linksFile_real_data.dat     1502 links  ← training reads this

data/netrainsim_v2/
  trainsFile_rl.dat           consist + forward path (node 1 → 1503)
  train_return.dat            consist + return path  (node 1503 → 1)
```

> The two trains files still sit under `netrainsim_v2/` for historical reasons, but their
> paths were updated to `1→1503` and they belong to the **active** dataset. The other
> files in that folder are the superseded route and are not read by anything.

There is no data-generation step — the `.dat` files are the inputs, used as-is.

### Timing floor

The speed limits alone forbid completing the route faster than **~7,100 s**
(`Σ length/limit`; a kinematic pass with EMU acceleration and braking limits agrees at
~7,050 s). Any deadline below that is unreachable. This is why step 6 exists.

---

## Reward, in one block

Per simulated second, summed over each 15-second decision:

```
r = −3.0 · energy_kwh                        # the objective
  − 2.0 · behind_schedule_fraction           # zero when on or ahead of pace
  − 1.0 · overspeed_mps                      # safety net; the sim also caps speed
  + 200   on arrival
  − 1500  on timeout
  − 0.15 · |Δnotch|                          # once per decision, smoothness
```

Three design choices carry most of the weight, each learned the hard way:

- **The observation includes the clock** (`time_frac`, `behind_frac`). The pace penalty
  depends on the step counter, so without it the reward is not Markov in the observation.
- **`discount_factor = 0.9999`.** Episodes are thousands of steps of summed energy. At
  0.99 the agent optimises the next ~100 seconds and learns "finish fast", which burns
  energy.
- **Entropy is annealed to zero.** Held constant, the policy never stopped being uniform
  (entropy pinned at ln 9 for 100 epochs) because the entropy gradient exceeded the
  per-decision advantage gradient.

Actions repeat for **15 simulated seconds**, cutting decisions per episode roughly 15-fold
and making credit assignment tractable.

---

## Reference results — previous dataset, not this one

⚠️ These numbers come from the **superseded** `netrainsim_v2` route (74.89 km, four speed
zones, +346.6 m climb). The active `real_data` route is slower and differently graded, so
these do **not** transfer. They are kept only as an indication of what the method achieved
before; the real target is whatever step 5 measures.

| policy | trip time (s) | energy (kWh) |
|---|---|---|
| constant N8 | 5,602 | 862.4 |
| constant N3 *(eco, on-time)* | 6,442 | 788.6 |
| constant N2 | 8,492 *(late)* | 758.4 |
| **RL (w_P = 2, 5 seeds)** | **6,203 ± 440** | **819.7 ± 26.7** |

On that route the pace weight `w_P` controlled **schedule reliability, not the energy–time
price**: energy was statistically flat across `w_P ∈ {0,1,2,4,8}` (all p > 0.66) while
on-schedule arrivals rose from 60 % to 100 %. The campaign scripts that produced those
statistics have been removed (see git history for `rl/run_campaign.py`,
`rl/run_experiment.py`, `rl/aggregate_campaign.py`).

## Known limitations

- On the previous route the policy **matched but did not beat** the constant-notch
  frontier. Terrain-aware modulation that goes genuinely below it is the open problem.
- The **deterministic (argmax) policy is unreliable** — high-variance and often degenerate
  across seeds. Only the sampled policy is deployable.
- **Curvature units are unverified.** The GOST curve term assumes degrees of arc
  (R = 1746.4/D); the maximum in the data implies a very gentle minimum radius, which is
  worth confirming with the data provider.
- **The `real_data` speed column may be a measured profile rather than legal limits.** It
  has 960 distinct values on 1502 links. NeTrainSim treats it as a hard cap, so if these
  are speeds a real driver actually reached, the agent can only ever go *slower* — which
  closes off run-ahead-then-coast strategies and narrows what it can learn. Worth
  confirming with whoever supplied the file.

---

## Gotchas

1. **Use `python3.12`, not `python3`.** macOS's default 3.9 cannot import this code.
2. **`-e` takes a value.** `-e true` exports the trajectory CSV; a bare `-e` silently
   defaults to false and writes nothing.
3. **Speeds in `.dat` files are m/s**, not km/h (confirmed in `netlink.cpp`).
4. **At standstill the locomotive applies full adhesion regardless of notch** — original
   NeTrainSim behaviour, so the train cannot be parked mid-route.
5. **`DEADLINE_STEPS` must exceed the fastest constant notch.** Otherwise every episode is
   late and the schedule term teaches nothing.

`CLAUDE.md` holds the deeper architecture reference.
