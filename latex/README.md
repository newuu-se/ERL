# Paper draft — Energy-Efficient Train Driving via Simulator-in-the-Loop Deep RL

Self-contained Overleaf project. **Deleting this folder changes nothing in the
repository** — every figure used by the paper is a *copy* in `figures/`, and the
scripts here only read from the project.

## Compile on Overleaf

1. Zip this folder's contents (`main.tex`, `figures/`).
2. Overleaf → New Project → Upload Project → select the zip.
3. Compiler: **pdfLaTeX** (default). No .bib file needed — the bibliography is
   embedded in `main.tex`. Only standard packages are used (graphicx, booktabs,
   amsmath, hyperref, xcolor, geometry, microtype, multirow).

## Before submitting

Search `main.tex` for `\todo{` (rendered red in the PDF):
- author affiliation
- grant acknowledgment (number/funder)
- verify the NeTrainSim reference details (volume/pages/year)
- curvature-unit confirmation footnote
- the planned W_PACE ablation result (Section: Return trip)

## Contents

| file | provenance |
|---|---|
| `main.tex` | the draft (journal revision; reward sensitivity + ablation + variance) |
| `figures/fig_tradeoff_forward.png` | copy of `results/plots/energy_time_tradeoff.png` (RL = 5-seed mean ± CI) |
| `figures/fig_profile_forward.png` | copy of `results/plots/rl_policy_profile.png` |
| `figures/fig_native_driver.png` | copy of `results/plots/netrainsim_trajectory.png` (available; unused) |
| `figures/fig_data_cleaning.png` | `make_paper_figures.py` |
| `figures/fig_tradeoff_return.png` | `make_paper_figures.py` |
| `figures/fig_profile_return.png` | `make_paper_figures.py` |
| `figures/fig_training_curves.png` | `make_paper_figures.py` |
| `figures/fig_reward_theory.png` | `rl/reward_theory.py` (closed-form coefficient justification) |
| `figures/fig_sensitivity.png` | `rl/aggregate_campaign.py` (w_P sweep, 5-seed CIs) |
| `figures/fig_ablation.png` | `rl/aggregate_campaign.py` (ablation, 5-seed CIs) |
| `make_paper_figures.py` | regenerates the four `make_paper_figures` figures (reads project files, writes only here) |

Numbers come from: `EXPERIMENT_LOG.md`; the campaign aggregates in
`results/campaign/{sweep_summary,ablation_summary,significance}.csv`; the
theory in `results/theory/reward_theory.json`; baselines in
`results/return/baselines.txt`. All measured on
`data/netrainsim_v2/linksFile_v2_clean.dat`. The reward-coefficient
figures (`fig_reward_theory`, `fig_sensitivity`, `fig_ablation`) are
produced by `rl/reward_theory.py` and `rl/aggregate_campaign.py`; rerun
those after any retraining and re-copy as needed.
