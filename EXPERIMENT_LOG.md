# Train Energy-Optimization with RL — Methods, Changes & Results

A chronological, paper-oriented log of every change made to the simulator and the
RL layer, **why** each change was made, and **what result** it produced. Numbers
are from measured runs on the current configuration unless noted.

---

## 1. System under study

| Component | Description |
|---|---|
| **Simulator** | NeTrainSim (C++/Qt6 freight-train physics simulator), console build, driven per-second |
| **RL layer** | Python: Gymnasium env wrapping the simulator + Tianshou **PPO** agent (0.5.1) |
| **Route** | Toshkent → Ho'jakent line: **74.89 km**, 1500 nodes, 1499 links (~50 m/segment) |
| **Speed limits** | 11.11 / 16.67 / 19.44 / 22.22 m/s (40/60/70/80 km/h); distance per zone: 31.0 / 15.6 / 4.0 / 24.3 km |
| **Train** | ER9E electric multiple unit (3 power cars, 0.25 adhesion coefficient) |
| **Control** | Locomotive throttle **notch 0–8**; quadratic notch→throttle mapping `(N/Nmax)²` |
| **Objective** | Minimize total trip energy (kWh) subject to arriving within a schedule window |

The agent sets the locomotive notch; NeTrainSim integrates the longitudinal
dynamics (tractive effort vs. resistance, adhesion-limited) and reports per-step
state and energy.

---

## 2. Simulator physics modifications (NeTrainSim C++)

### 2.1 Davis resistance equation → GOST/SI form
**Files:** `src/NeTrainSim/traindefinition/car.cpp` (`Car::getResistance`),
`locomotive.cpp` (`Locomotive::getResistance`), doc comment in `traincomponent.h`.

**Before** — modified Davis equation in **US units** (lb, mph), with empirical
coefficients and unit-conversion factors:
```
R = (1.5 + 18/w_axle + 0.03·V_mph + K·A·V²/w)·W_lb + 20·W_lb·grade + |curv|·0.8·W_lb,  ×4.44822 → N
```

**After** — fully SI **GOST-based** specific-resistance formula (railway practice
in the region of study):
```
v   = speed·3.6                       [km/h]
W   = mass·9.81                       [kN]
w0  = 1.1 + 0.01·v + 0.000227·v²      [N/kN]   (basic running resistance)
R   = w0·W  +  grade‰·W  +  |curv|·(700/1746.4)·W   [N]
```
where `trackGrade` is stored in % (×10 → ‰) and curve resistance uses the GOST
`700/R_curve` rule with the standard degree-of-curve→radius conversion.

**Why:** the US-unit Davis form is not appropriate for the Uzbekistan rolling
stock/track standard; a transparent SI GOST formulation matches local engineering
practice and removes opaque imperial conversions, making the resistance model
defensible and reproducible for the paper.

**Result (measured):** grade and curve terms are numerically near-identical
between the two formulas (both ≈ m·g·grade), but the GOST **basic running
resistance is lower** than the US Davis term. Net effect at constant notch 6:
**981.4 → 909.3 kWh** (≈ −7%), trip time unchanged (5653 → 5654 s).

### 2.2 Removal of regenerative braking
**File:** `locomotive.cpp` (`Locomotive::getRegenerativeEffeciency`).

**Before:** for rechargeable power types (electric/hybrid), braking recovered
energy with efficiency `η = 1/exp(γ/|a|)`; the per-step net energy was
`consumed − regenerated`. The ER9E is **electric** (power-type index 1 ∈
`locomotiveRechargableTechnologies`), so it *was* regenerating.

**After:** `getRegenerativeEffeciency` returns **0** for all power types, so
braking is purely dissipative (only auxiliary power is drawn). This zeroes the
regenerated-energy term in both `getEnergyConsumption` and
`getEnergyConsumptionAtDCBus`; because braking then yields ≥0 energy, no negative
energy propagates to the battery/catenary recharge paths either.

**Why:** the study models energy *consumption* without regenerative recovery
(conservative energy accounting / matching the operating assumption for the line).

**Result:** removing regen *increases* energy, but on this route the §2.1 GOST
reduction dominates, so the net constant-notch-6 figure still falls to 909 kWh.
The two effects were separated by the constant-notch probe (below).

### 2.3 Adhesion-limited tractive force — unchanged (documented)
Tractive force is already adhesion-limited: `F = min(P·η/v, μ·m·g)` in
`Locomotive::getTractiveForce`, with μ the train friction coefficient (0.25). This
was reviewed and **intentionally left unchanged**; the simulator also caps speed at
each link's free-flow speed, so over-speeding is physically prevented.

### 2.4 Verification
- Independent C++ review (units, formula, compile-safety, completeness of regen
  removal) — no defects; confirmed `currentWeight` is in tonnes and `trackGrade`
  in %, so the GOST unit handling is correct.
- Rebuilt console binary (`build-linux.sh`), clean (exit 0).
- Smoke test (constant notch 6) confirmed the new energy figure (909 kWh).

---

### 2.3 Input-data cleaning — DEM grade-spike removal (2026-06-11)
**File:** `data/clean_grade_spikes.py` → `data/netrainsim_v2/linksFile_v2_clean.dat`
(now the active links file).

The raw v2 links file contained physically impossible grades — adjacent
opposite-sign pairs up to **−12.4% / +17.0% on 50 m segments** (links 887/888,
1204/1205, …), classic single-node DEM elevation noise. Consequences measured
before cleaning: on a +17% link the grade resistance (~620 kN) exceeded the
train's total adhesion (~440 kN), force balance broke (reported accel ≈ 0 with
R ≫ F), and the virtual-power energy accounting `(m·a + R)·v` charged up to
**3.76 kWh in one second (11.8 MW from a 3.6 MW consist)**.

Method: integrate grades to an elevation profile → 3× median-3 filter +
1× moving-average on interior nodes (endpoints pinned → net climb 346.6 m
preserved exactly) → re-derive grades. Grade range after: **−2.7…+4.5%**
(remaining >3% links are clustered near the mountainous route end, i.e.
plausible terrain). Max per-step energy after: 1.39 kWh (notch-8 acceleration
transient) — within installed power.

---

## 3. RL problem formulation

| Element | Definition |
|---|---|
| **Observation** | 9 floats, normalized ≈[0,1]: speed, position, grade, curvature, remaining distance, per-step energy, link speed limit, **time_frac, behind_frac** (schedule features added 2026-06-11 — the pace-penalty reward depends on the step counter, so without them the reward was non-Markov in the observation) |
| **Action** | `Discrete(9)` — notch 0–8 |
| **Transition** | persistent interactive NeTrainSim subprocess: send `{notch}` → run one (or several) sim-seconds → read state JSON |
| **Episode end** | `terminated` = reached terminus; `truncated` = step count ≥ MAX_STEPS (12,000) |
| **Algorithm** | PPO (separate actor/critic, GAE λ=0.95, clip 0.2), Tianshou 0.5.1 |

---

## 4. Correctness fixes (from code review, before tuning)

| # | Issue found | Fix | Why it mattered |
|---|---|---|---|
| 1 | `TARGET_STEPS=4500` deadline below the physical floor (~5,025 s) | set realistic deadline 6,500 s | every arrival was always "late"; the schedule term was meaningless |
| 2 | Train path `1,1499` stopped one node short of the 1500-node route | path `1,1500` | train now reaches the true terminus (74,891 m, was 74,889) |
| 3 | Obs normalization used 19.4 m/s max | 22.22 m/s | route actually reaches 80 km/h; features were mis-scaled |
| 4 | Eval rebuilt the policy with hyperparameters that could drift from training | shared `build_policy()` factory | guarantees the evaluated policy matches the trained one |
| 5 | Per-worker stderr log opened append-only, grew unbounded | per-PID log, truncated each reset | clean debugging |

---

## 5. Reward-design evolution (the core of the study)

PPO maximizes expected return; the reward is where the eco-driving objective is
encoded. The episode is long (~6,000 control steps at 1 s), which proved to be the
dominant difficulty. Each iteration below lists the **problem observed**, the
**change**, and the **measured result**.

### Iteration 0 — baseline (pre-existing)
Dense reward with a **speed-deficit penalty** (reward for staying near the speed
limit), terminal arrival/timeout, discount 0.99. **Result:** policy collapsed to a
near-constant high notch (used only notch 6–7); no energy optimization. The speed
target directly opposed coasting.

### Iteration 1 — remove speed target, add uniform time cost
**Change:** delete speed-deficit penalty; reward = `−energy − W_TIME·t − overspeed
+ progress_shaping + arrival`; discount 0.99 → 0.999.
**Why:** a *time cost* (not a speed target) should create the energy↔time trade-off.
**Result:** policy converged to **fast/high-energy (~965 kWh)**; best reward
plateaued. Diagnosis: at γ=0.999 over 6,000 steps the terminal arrival/timeout
signals are discounted to ≈0, and a discounted progress reward rewards *finishing
sooner* → bias to high notch.

### Iteration 2 — near-undiscounted; drop progress shaping
**Change:** discount 0.999 → **0.9999**; remove progress reward (a potential-based
variant was tried but its `(1−γ)·Φ` term corrupted the logged/selection reward).
**Why:** treat it as the near-undiscounted min-energy-to-goal problem it is.
**Result:** **diverged** — greedy test reward degraded −1063 → −2693 over 300
epochs; the policy drifted to slow-crawling. Large-magnitude returns (~−1,000 sums)
destabilized the value function.

### Iteration 3 — stabilize
**Change:** `reward_normalization=True`; `ent_coef` 0.01 → 0.005; `MAX_EPOCH`
300 → 60.
**Why:** normalize the large returns; calm exploration; stop before divergence.
**Result:** **stable** (no divergence) **but plateaued at fast ~900 kWh** —
essentially unchanged from start. The eco gain (~5% of episode reward) was too weak
a gradient to climb.

### Iteration 4 — amplify the energy signal
**Change:** `W_ENERGY` 1 → 3, `W_TIME` 0.05 → 0.03, `W_LATE` 0.10 → 0.30,
`ent_coef` → 0.008. Calibrated from the constant-notch curve so notch 3 is the clear
optimum (fast→eco reward gap ~205 vs ~41 before).
**Result:** stochastic behavior improved toward notch 4–5 (~882 kWh), **but the
greedy (deterministic) policy crawled** — notch 1, 820 kWh but **10,700 steps**
(massively late). A *terminal-only* late penalty cannot override an immediate
per-step energy gain over a long episode, so the policy's mode minimizes per-step
energy (lowest notch).

### Iteration 5 — per-step pace penalty
**Change:** replace the terminal late penalty with a **per-step pace penalty**:
```
behind = max(0, (t/DEADLINE)·route_len − position) / route_len
r_t = −W_ENERGY·energy − W_PACE·behind − overspeed   (+arrival / −timeout)
```
zero when on/ahead of schedule, so coasting is free where there is slack but
lagging is penalized *immediately*. `W_PACE=2`, `MAX_EPOCH=100`.
**Why:** make the schedule pressure local (per-step), not deferred to the terminus.
**Result:** crawl reduced (notch-1/10,700 → **notch-2/7,708 steps, 784 kWh**), but
still **~19% late** and the notch is **essentially constant** — not terrain-aware.
The reward is now *correct* (notch 3 strongly optimal), but PPO's deterministic mode
still lands one notch too slow: the delayed pace cost is under-weighted vs. the
immediate energy gain.

### Iteration 6 — coarse control interval (current run)
**Change:** `CONTROL_INTERVAL=15` — the agent chooses a notch every **15 s** (held
in between), reducing the decision horizon from ~6,000 to **~470 per trip**;
`STEP_PER_EPOCH` lowered to 3,000 accordingly.
**Why:** the recurring failure (constant notch, mode one step too slow) is a
**long-horizon credit-assignment** problem — crediting a 1-second throttle choice
for its delayed energy/schedule consequence is intractable at 6,000 steps. Action
repeat shortens the horizon ~15×, and at ~470 steps γ=0.9999 barely discounts, so
credit assignment becomes tractable — the prerequisite for learning terrain-aware
modulation.
**Result:** training was **stable and ~4× faster** (12.8 min for 100 epochs; ~15×
less per-transition overhead), best reward −2375.96 (best of any iteration). The
**stochastic policy now arrives on-schedule (6,113 s) at ≈887 kWh with the notch
genuinely spanning the full 0–8 range** — the first terrain-aware, on-schedule
policy obtained (decision histogram: N0×66, N1×135, N2×37, N3×27, N4×45, N5×38,
N6×27, N7×12, N8×21). Two honest caveats:
- **Energy parity, not superiority:** 887 kWh ≈ constant notch 4; it does **not** beat
  the constant-notch-3 reference (834 kWh). Part of the variation is genuine
  terrain response, part is sampling.
- **Deterministic argmax still degenerate:** the greedy policy collapses to notch 1
  and times out (12,000 s). The high entropy coefficient (0.008) parks the policy's
  *mode* at the lowest-energy notch while *sampling* supplies the on-schedule speed.

**Interpretation:** coarse control removed the long-horizon blocker — notch variation
and on-schedule arrival finally emerge together — but a clean, energy-optimal
*deterministic* policy needs the entropy lowered/annealed so the mode converges to the
on-schedule optimum (proposed next step).

### Iteration 7 — smoothness regularization + lower entropy → deterministic eco policy
**Motivation:** the iteration-6 *sampled* policy switched notch erratically (full 0–8
range, near-100% of decisions) and its deterministic argmax was degenerate.
**Change:** add a per-decision **notch-change penalty** `−W_SMOOTH·|Δnotch|` and lower
`ent_coef`. Two settings probed:
- `W_SMOOTH=0.5, ent_coef=0.002` → collapsed to **constant notch 5** (897 kWh, 5,716 s) —
  over-smoothed (smooth, but a sub-optimal notch).
- `W_SMOOTH=0.15, ent_coef=0.004` → the **deterministic policy converges to ~constant
  notch 3: 834 kWh, 6,337 s, on-schedule** (one notch change over the entire trip).

**Result:** the policy lands **exactly on the eco-optimal point of the constant-notch
Pareto frontier** — the lowest-energy schedule-feasible setting: **~8.6% below flat-out**
(notch 8 = 912 kWh) and **~6% below NeTrainSim's native driver** (888 kWh). The
deterministic policy is now clean, smooth and deployable (argmax no longer degenerate).

**Finding:** under any smoothness pressure the optimal policy is a *constant* notch,
because terrain-aware sub-trip modulation yields negligible energy savings on this route
(corroborated by the nearly straight constant-notch frontier and by NeTrainSim's own
*varied*-notch driver sitting *above* that frontier). The defensible contribution is
therefore that the RL agent **reliably identifies the energy-optimal, schedule-feasible
throttle setting end-to-end from the simulator**, rather than performing continuous
eco-modulation. (`W_SMOOTH` is the dial between an erratic profile and a flat one.)

### Iteration 8 — observation/data bug-fix pass (2026-06-11)
**Motivation:** a full-code review questioned whether the "no terrain modulation"
ceiling was real or self-inflicted. It found the agent had been **blind to the
terrain it was supposed to modulate against**:
1. `_GRADE_MAX=0.7` (stale v1 value) clipped the grade feature to a **sign bit on
   ~27% of the route** (v2 grades reach ±17% raw / ±4.5% clean).
2. `_ENERGY_MAX=0.25` vs real per-step max 1.39 kWh — the energy feature clipped
   on any heavy power draw.
3. The DEM grade spikes themselves (→ §2.3) distorted physics and energy.
4. The pace-penalty reward depends on the step counter, which was **not in the
   observation** (non-Markov reward) — the critic could not predict pace costs,
   the policy could not modulate with schedule slack. → obs 7→9 (`time_frac`,
   `behind_frac`).
5. A `select()`+buffered-`readline()` race in the env could falsely kill healthy
   episodes after a 30 s stall → reader thread + queue.
6. Checkpoint labels were one epoch ahead of their contents (tianshou `train_fn`
   fires at epoch *start*) → saving moved to `test_fn`.

Baselines re-measured on the clean data (→ §7.1): eco target = **const n3
= 788.6 kWh / 6,442 s** (deadline 6,500 s unchanged).

**Run 1 with all fixes (hyperparams = iteration 7):** stable, 901/901 arrivals,
best test −2470 (ep 89) — but `loss/ent` sat at **2.190 ≈ ln 9 for all 100
epochs**: the sampled histogram was flat (29–58 per notch) and the argmax parked
at **notch 0** (timeout). With a constant `ent_coef`, the entropy gradient
exceeded the per-decision advantage gradient — the policy never committed.

### Iteration 9 — entropy annealing → sharp policy (2026-06-11)
**Change:** linear anneal `ent_coef` 0.004 → 0 by epoch 70 (set in `train_fn`
via `policy._weight_ent`); last 30 epochs train on the pure objective.
**Result (16.1 min, 901/901 arrivals):** entropy **2.195 → 0.677**; best test
reward **−2323 (ep 84)** — best of any iteration. After the bonus reached zero,
test rewards drifted (−2323 → −4604 by ep 100), so `policy_best.pth` (ep 84) is
the artifact. Evaluation of best:
- **Sampled:** 6,708 s / **800.2 kWh** — mode notch 2 (287/448 decisions) with
  notch-1 stretches and notch-6/8 kicks; 3.2% late.
- **Deterministic argmax:** exactly **constant notch 2** (758.4 kWh / 8,492 s,
  31% late) — the n2 baseline point.

**Interpretation:** annealing fixed policy sharpness (uniform → committed).
The remaining gap is **reward calibration, not optimization**: at `W_PACE=2`,
being ~200 s late costs ≈210 reward ≈ 70 kWh-equivalents, so the n2/n3 boundary
trajectories (800 kWh, slightly late) score ≈ const-n3 (789 kWh, on-time:
≈ −2370) — the reward is now *indifferent* between them, and const n3 still
**Pareto-dominates** the learned policy. To force on-schedule eco operation the
lateness exchange rate must rise (e.g. `W_PACE` 2→4+) or the deadline must enter
as a hard constraint.

---

## 6. Training methodology (current)

| Hyperparameter | Value | Note |
|---|---|---|
| Algorithm | PPO (actor/critic) | Tianshou 0.5.1, CPU (tiny net) |
| Net | [256,128,64] | shared `build_policy()` for train+eval |
| Discount γ | 0.9999 | near-undiscounted; appropriate for long episodic min-cost |
| GAE λ | 0.95 | |
| Clip ε | 0.2 | |
| Entropy coef | 0.004 → 0, linear anneal by epoch 70 | constant coef pinned the policy at uniform (iteration 8); anneal lets it commit |
| Reward norm | True | stabilizes large-magnitude returns |
| Adv. norm | True | |
| Parallel envs | 8 (`SubprocVectorEnv`) | one C++ subprocess each |
| Control interval | 15 s (action repeat) | iteration 6 |
| Reward weights | W_ENERGY=3, W_PACE=2, W_OVERSPEED=1, arrival=+200, timeout=−1500 | |

---

## 7. Results

### 7.1 Constant-notch baselines (clean data, `rl/run_baselines.py`, 2026-06-11)
The energy–time Pareto frontier, obtained by driving each constant notch to
completion through the interactive simulator:

| Notch | Trip time (s) | Energy (kWh) | On schedule (≤6,500 s)? |
|---|---|---|---|
| 8 | 5,602 | 862.4 | yes |
| 6 | 5,652 | 853.5 | yes |
| 5 | 5,721 | 849.1 | yes |
| 4 | 5,910 | 834.4 | yes |
| **3** | **6,442** | **788.6** | **yes (eco optimum)** |
| 2 | 8,492 | 758.4 | no (31% late) |

→ **Constant notch 3 (788.6 kWh, on-time)** is the lowest-energy schedule-feasible
constant policy and serves as the reference eco target. Theoretical minimum trip
time (speed-limit-bound) ≈ 5,025 s.
(Pre-cleaning values — n8 912.5, n3 834.1/6,337 s, n2 783.5/7,706 s — are not
comparable: they included spike-link energy artifacts; see §2.3. Earlier
iterations' kWh numbers in §5 are on the old data.)

### 7.2 NeTrainSim native-driver baseline
Running NeTrainSim's own driver (no RL) on the clean data: **6,038 s, 848.2 kWh**,
with a genuinely **varied notch**. This point sits **above** the constant-notch
frontier (more energy than constant notch 4 at comparable time), i.e. the built-in
driver does not outperform simple constant-notch selection here. The RL sampled
policy (800.2 kWh / 6,708 s) uses **5.7% less energy** than the native driver,
at the cost of ~11 min more trip time. (Old spiky-data value was 6,023 s /
887.7 kWh — not comparable.)

### 7.3 RL policy outcomes by iteration

| Iter | Reward design | Greedy result | Verdict |
|---|---|---|---|
| 0 | speed-deficit | notch 6–7 only | no optimization |
| 1 | time cost, γ=0.999 | ~965 kWh, fast | plateaued (discount bias) |
| 2 | γ=0.9999, no shaping | — | diverged |
| 3 | + reward-norm, MAX_EPOCH 60 | ~900 kWh, fast | stable, no eco |
| 4 | W_ENERGY=3 | greedy crawl: 820 kWh / 10,700 s | off-schedule |
| 5 | per-step pace penalty | 784 kWh / 7,708 s, ~const notch 2 | low energy but ~19% late, not terrain-aware |
| 6 | coarse control (15 s) | sampled: **6,113 s / 887 kWh, notch spans 0–8**; argmax degenerate (notch 1, timeout) | **first terrain-aware + on-schedule** policy; energy ≈ const. notch 4; argmax needs lower entropy |
| 7 | + smoothness penalty + lower entropy | deterministic **~constant notch 3: 834 kWh / 6,337 s (on-schedule)** | **eco-optimal operating point**; clean/smooth/deployable; ~8.6% below flat-out, ~6% below NeTrainSim driver |
| 8 | obs/data bug-fix pass (clean grades, unclipped grade/energy obs, time features, obs 7→9) | constant ent_coef → policy stayed **uniform** (loss/ent ≈ ln 9 all run); argmax notch 0, timeout | fixes necessary but exposed: entropy gradient > advantage gradient |
| 9 | + entropy anneal 0.004→0 by ep 70 | best −2323 (ep 84); sampled **800.2 kWh / 6,708 s**, mode n2 + n6/8 kicks; argmax = const n2 (758 kWh, 31% late) | sharpest policy yet; const n3 still Pareto-dominates → raise lateness price (W_PACE) next |
| 10 | 2× budget: 200 epochs, anneal→0 by ep 140 (both trips) | fwd best −2254 (ep 196); sampled **790.0 kWh / 6,417 s ON-SCHEDULE = frontier parity** (interp. frontier ≈790.8); argmax const n1 (timeout). Return best −620 (ep 179); sampled ~const n1: 169.6 kWh but 8.9% late; argmax const n1 (18% late) | sampled fwd policy matches constant-notch selection on schedule — first time; argmax still parks 1 notch below feasible on both trips → W_PACE too low remains the blocker |

### 7.5 Return trip — Ho'jakent → Toshkent (B→A, added 2026-06-11)
New scenario: `data/netrainsim_v2/train_return.dat` (path 1500→1, same consist),
trained by `rl/train_return.py` (shared `run_training`), evaluated with
`rl/evaluate.py --return-trip`; outputs in `results/return/`, checkpoints in
`checkpoints/return/`. A direction bug had to be fixed first: the interactive
state emission took `grade.begin()` (always the forward-direction entry), so
the *observation* — not the physics, which resolves direction via path order
(`simulator.cpp:365-373`) — would have shown inverted grade signs on the return
trip. Verified post-fix: emitted grade == −(file grade) along the route, and the
forward trip is regression-identical.

Constant-notch frontier (net −346.6 m descent — far cheaper than forward):
| Notch | Trip time (s) | Energy (kWh) | On schedule (≤6,500 s)? |
|---|---|---|---|
| 8 | 5,605 | 348.3 | yes |
| 6 | 5,641 | 335.8 | yes |
| 5 | 5,682 | 327.4 | yes |
| 4 | 5,768 | 317.7 | yes |
| 3 | 5,950 | 293.9 | yes |
| **2** | **6,375** | **242.2** | **yes (eco optimum)** |
| 1 | 7,681 | 170.1 | no (18% late) |

RL run (same hyperparameters, 14.1 min, 901/901 arrivals): best reward
**−906.2 at epoch 100** — still improving when training ended. Deterministic
argmax = clean **constant notch 4 (317.7 kWh / 5,768 s, on-schedule)**;
stochastic ≈ same (319.7 / 5,827, mode n4). Not yet eco-optimal: the optimizer
was still descending the notch ladder when MAX_EPOCH hit. Entropy also stayed
high (2.03 vs 0.68 forward) — downhill, adjacent notches differ little.

**200-epoch rerun (anneal→0 by ep 140):** best **−620 (ep 179)**; the descent
continued *past* the eco point into the under-paced basin: sampled ≈ constant
n1 with rare kicks (**169.6 kWh but 7,076 s = 8.9% late**), argmax = const n1
(170.1 kWh / 7,681 s, 18% late). Same calibration conclusion as the forward
trip: at `W_PACE=2` the n2→n1 energy saving (72 kWh ≈ 216 reward) is not
decisively beaten by the lateness cost it incurs, so the mode settles below
the schedule-feasible point. Raise `W_PACE` for on-schedule convergence.

### 7.4 Figures (`results/plots/`, regenerate with `python rl/make_plots.py`)
- **`energy_time_tradeoff.png`** — constant-notch Pareto frontier (colored by notch)
  with the RL policy and NeTrainSim native driver overlaid.
- **`netrainsim_trajectory.png`** — NeTrainSim native-driver trajectory (the data the
  GUI plots): speed vs limit, notch, tractive/resistance force, cumulative energy,
  grade vs distance.
- **`rl_policy_profile.png`** — RL greedy policy: speed (vs limit), notch, grade vs
  distance — shows the constant-notch-2 behavior of iteration 5.
- **`training_curve.png`** — test/best reward per epoch.

---

## 8. Key findings & discussion

1. **Reward specification was solved; optimization is the bottleneck.** After
   iteration 5 the reward correctly makes the on-schedule eco policy (notch 3)
   strongly optimal. The agent nonetheless converges to a near-constant notch one
   step too slow.
2. **Root cause: long-horizon credit assignment.** At 1-second control (~6,000
   steps) PPO under-weights delayed schedule costs against immediate per-step energy
   gains, and discounting over such a horizon either erases terminal signals (low γ)
   or destabilizes value learning (high γ). Every reward tweak hit this same wall.
3. **Discounting/horizon interaction:** γ must be ≈0.9999 to keep the energy *sum*
   undistorted, which requires reward normalization for stability.
4. **The RL did not beat constant-notch selection** through iteration 5 — an
   honest negative result that motivates the structural change.
5. **Coarse control (iteration 6) confirmed the thesis:** shortening the decision
   horizon ~15× (one notch every 15 s) made credit assignment tractable — terrain-aware,
   on-schedule behaviour emerged for the first time (sampled policy: 6,113 s, 887 kWh,
   notch spanning 0–8) and training became ~4× faster. Two remaining items: (a) the
   policy's *mode* (argmax) still parks at the lowest notch under the entropy bonus, so
   a clean *deterministic* eco policy needs entropy annealing; (b) energy is at parity
   with constant notch 3 (≈834–887 kWh), not yet below it — converting notch *variation*
   into net energy *savings* is the next reward/curriculum question.
6. **The agent had been partially blind (iteration 8).** Stale normalization clipped
   the grade feature to a sign bit on ~27% of the route and the energy feature on any
   heavy draw; the input grades contained physics-breaking DEM spikes; and the
   pace-penalty reward depended on a step counter absent from the observation
   (non-Markov). Any terrain/schedule modulation was unlearnable *in principle*
   before these fixes — earlier "no modulation" conclusions partially confounded.
7. **Exploration–commitment trade-off is real here (iterations 8–9).** With a
   constant entropy coefficient the per-decision advantage gradient (tiny, because
   one 15 s notch choice changes the ~−2,500 return by little) is smaller than the
   entropy gradient — the policy stays uniform forever. Linear annealing to zero
   produced the sharpest, best-scoring policy of the study (−2323) and removed the
   degenerate-argmax pathology (argmax now a *sensible* constant notch, not 0/1).
8. **Current frontier gap is reward calibration:** the learned best (800 kWh,
   3.2% late) and const-n3 (789 kWh, on-time) score nearly equally because
   `W_PACE=2` prices ~200 s of lateness at only ≈70 kWh-equivalents. Const n3
   still Pareto-dominates. Next lever: raise `W_PACE` (≥4) or hard-constrain the
   deadline, then re-train — the machinery (sharp policy, visible schedule/terrain
   features, clean data) is now in place.

---

## 9. Reproducibility

```bash
# 1. Generate / rebuild simulator (after physics edits)
cd NeTrainSim-adjusted && ./build-linux.sh

# 2. Measure constant-notch baselines / verify physics
#    (constant-notch probe drives the interactive binary per notch)

# 3. Train
source venv/bin/activate
python rl/train.py 2>&1 | tee logs/train_run_<ts>.log

# 4. Evaluate best checkpoint → results/notch_profile.csv
python rl/evaluate.py

# 5. Plots
python rl/make_plots.py        # → results/plots/*.png
```

Key files: `rl/train_env.py` (env + reward, fully commented), `rl/train.py` (PPO +
shared `build_policy`), `rl/evaluate.py`, `rl/make_plots.py`; physics in
`NeTrainSim-adjusted/src/NeTrainSim/traindefinition/{car,locomotive}.cpp`. Checkpoints
of superseded runs are archived under `checkpoints/archive_*`.

---

*Status: iterations 0–10 complete (incl. the B→A return trip). High-water mark
(200-epoch forward run, `policy_best.pth` ep 196): the sampled policy reaches
**on-schedule frontier parity — 790.0 kWh / 6,417 s**, within ~1 kWh of the
interpolated constant-notch frontier, with a modulated (mode-n1 + kicks) profile.
The return trip converged past its eco point into "cheap but late" (169.6 kWh,
8.9% late). The one remaining systematic defect, seen on both trips: the policy
mode settles one notch below the schedule-feasible point because `W_PACE=2`
prices lateness too cheaply. Next experiment: `W_PACE` 2→4–6 on both trips;
then re-ask whether terrain-aware modulation can move *below* the frontier.*

---

## 11. Reviewer-response campaign (2026-06-15) — sensitivity, ablation, variance

Addresses journal feedback: (1) systematic reward sensitivity + ablation +
theoretical coefficient justification; (2) variance/CIs/significance over
single runs. Infrastructure added: per-RNG seeding (reproducible: same seed →
identical run, verified), env-var reward-weight overrides, `rl/run_experiment.py`
(one seeded run → results.json), `rl/run_campaign.py` (65-run matrix, resumable),
`rl/aggregate_campaign.py` (mean±95%CI + permutation p-values + figures),
`rl/reward_theory.py` (closed-form). 65 runs (w_P∈{0,1,2,4,8}×5 seeds×2 trips +
3 forward ablations×5 seeds), 100 epochs each, 9.5 h, **0 failures**.

**Closed-form coefficient theory (the key correction).** For constant-notch
policies R(n;w_P) = a(n) − w_P·Σβ(n) is linear in w_P (Σβ = integrated
behind-schedule fraction). The schedule-feasible eco notch is reward-optimal for
**w_P ≥ 0.07 (forward, N3) / 0.47 (return, N2)**. The chosen w_P=2 is 30×/4×
above threshold → coefficients justified analytically. **This OVERTURNS the
iteration-10 claim that lateness was an underpriced tradeoff**: that arithmetic
compared the final pace penalty, not its integral Σβ; done correctly the schedule
term already dominates the one-notch energy saving ~4:1.

**Sensitivity sweep (5 seeds, mean±95%CI).** Forward energy is FLAT across w_P
(819–825 kWh, all p>0.66 vs w_P=2) — exactly as theory predicts. What w_P controls
is schedule RELIABILITY: forward lateness 12.6%→0, on-sched 60%→100% as w_P 0→8;
return reaches 100% on-time only at w_P=8 (energy rises, p=0.007). So **w_P is a
schedule-reliability knob, not an energy–time price.**

**Headline (w_P=2, 5 seeds):** forward 819.7±26.7 kWh / 6203±440 s, 4/5 on-sched
(~1.3% above frontier, −5.0% vs flat-out, −3.4% vs native driver); return
221.2±21.9 kWh / 6523±143 s, 2/5 on-sched (eco energy, borderline timing). NOTE
these seeded means replace iteration-10's lucky single-run 790 kWh — the honest
number is ~820.

**Ablation (forward, 5 seeds):** energy robust to all (p>0.26); the PACE term is
the one essential for schedule (remove → 12.6%±22 late); no-time-features drops
on-sched 80→60% (Markov); constant-entropy worse reward; no-smoothness lowers
energy but breaks the argmax (deployability). argmax is high-variance/often
degenerate across seeds → sampled policy is the deployable artifact.

Paper (latex/main.tex) rewritten accordingly: new §"Reward-coefficient analysis"
(theory+sweep+ablation), seeded headline table, figures fig_reward_theory /
fig_sensitivity / fig_ablation; the "raise w_P / future work" framing removed.
Artifacts: results/campaign/, results/theory/. Open next step: more seeds to sharpen
return/ablation significance; make the deterministic argmax reliable.
