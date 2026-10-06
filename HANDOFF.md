# Session Handoff — 2026-06-11

(Replaces the 2026-06-09 handoff; that run finished — best was epoch 30, then diverged.
Its checkpoints are archived under `checkpoints/archive_*`.)

## What this session did (bug-fix pass + retrain)

A full-code review found and fixed bugs that were plausibly **the** reason PPO kept
converging to near-constant notch instead of terrain-aware modulation:

1. **Grade obs saturated** — `_GRADE_MAX` was 0.7 (stale v1 value) while v2 grades
   reach ±17% raw / ±4.5% clean → ~27% of the route was clipped to a sign bit.
   The agent literally could not see terrain magnitude. Now 4.5.
2. **Energy obs saturated** — `_ENERGY_MAX` was 0.25 kWh vs real per-step max 1.39
   (clean data). Now 1.5.
3. **Grade data had DEM noise spikes** (±12–17% on 50 m segments, links 887/888,
   1204/1205…). On those links resistance exceeded total adhesion (force balance
   broke) and the virtual-power energy model charged up to 3.76 kWh/s (11.8 MW from
   a 3.6 MW consist). Fixed elevation-preservingly: `data/clean_grade_spikes.py` →
   `data/netrainsim_v2/linksFile_v2_clean.dat` (now the active links file; net climb
   346.6 m preserved; grades now −2.7…+4.5%).
4. **Reward was non-Markov** — the pace penalty depends on the step counter, which
   was NOT in the observation. Added `time_frac` + `behind_frac` features (obs 7→9).
   The value function can now actually predict the pace cost; the policy can
   modulate with schedule slack.
5. **select()+readline race** in the env could kill healthy episodes after a 30 s
   false stall → replaced with a reader-thread + queue.
6. Checkpoint labels were one epoch ahead of content (train_fn fires at epoch
   START) → saving moved to test_fn. `policy_epochNNN.pth` now means "after NNN
   epochs".
7. Smaller: dead unreachable code in `_sim_step`, stale docstrings/banner in
   train.py, misleading `cum_energy` in evaluate.py, make_plots cumsum fallback
   (15× short under action repeat), hardcoded plot axes that would crop n2,
   stale error message, CLAUDE.md fully refreshed.

## New constant-notch baselines (CLEAN data — rl/run_baselines.py)

| notch | steps (s) | energy (kWh) |
|---|---|---|
| 8 | 5,602 | 862.4 |
| 6 | 5,652 | 853.5 |
| 5 | 5,721 | 849.1 |
| 4 | 5,910 | 834.4 |
| 3 | 6,442 | 788.6 |
| 2 | 8,492 | 758.4 |

Deadline 6,500 s unchanged → **eco target = constant n3 ≈ 789 kWh / 6,442 s**.
Success = arrive ≤6,500 with energy < 789 kWh and notch varying with grade/zone.

## Training results (both runs finished 2026-06-11)

**Run 1** (`logs/train_run_20260611_114046.log`, fixes only): stable, 901/901
arrivals, best −2470 — but entropy never moved (loss/ent ≈ ln 9 throughout);
the policy stayed UNIFORM; argmax = notch 0 (timeout). Constant ent_coef's
gradient exceeded the tiny per-decision advantage gradient.
Checkpoints: `checkpoints/archive_run1_uniform_20260611/`.

**Run 2** (`logs/train_run_20260611_115856.log`, + entropy anneal 0.004→0 by
ep 70): entropy 2.195→0.677, best **−2323 @ ep 84** (best of any iteration;
post-anneal epochs drift, so use `policy_best.pth`). Evaluation:
- sampled: **800.2 kWh / 6,708 s** — mode notch 2 + notch 6/8 kicks, 3.2% late;
  5.7% less energy than NeTrainSim's native driver (848.2 kWh / 6,038 s, clean data)
- argmax: constant notch 2 → 758.4 kWh / 8,492 s (31% late)
- const n3 (788.6 kWh / 6,442 s, on-time) still Pareto-dominates.

**Diagnosis & next lever:** reward calibration, not optimization — at W_PACE=2,
~200 s late costs only ≈70 kWh-equivalents, so n2-ish late trajectories score
≈ const-n3 on-time. Raise `W_PACE` to ~4–6 (and/or re-anneal from the run-2
checkpoint) and retrain; everything else (sharp policy, visible terrain/clock,
clean data) is in place. Full story: EXPERIMENT_LOG.md §iterations 8–9.

## Open items

- Curvature unit is unverified: the GOST curve term assumes US degrees of arc
  (R=1746.4/D). Data max 0.54° ⇒ min radius ~3.2 km — suspiciously gentle for this
  line. Ask the data provider; if it's 1/R or similar, fix `Car/Locomotive::getResistance`.
- Old 7-dim checkpoints: `checkpoints/archive_pre_obsfix_20260611/`.
- Working tree uncommitted — commit after judging the run.
