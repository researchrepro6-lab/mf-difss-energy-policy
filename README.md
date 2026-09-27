# Reproducibility bundle

**Mixed-frequency intuitionistic fuzzy decision support for benchmarking national
renewable-energy policy**

This bundle reproduces **every number, table and figure** of the manuscript from a
checksummed snapshot of public data. No result in the manuscript is typed by hand:
`run_all.py` writes `outputs/tex/numbers.tex` (one LaTeX macro for every number quoted in
the text) and `outputs/tex/tab_*.tex` (every table body); the manuscript contains these
files verbatim.

## Quick start

Requires Python 3.10 or newer. The same commands work on Windows, macOS and Linux.

**Option A: IDLE or any editor (simplest).** Open `reproduce_all.py` (or `run_all.py`) and
run it (in IDLE: *Run > Run Module* or F5). Missing packages, such as `xlrd`, are installed
automatically with pip the first time; the output then appears in the same window.

**Option B: terminal (Command Prompt on Windows).** Open a terminal in this folder and run

```bash
python -m pip install -r requirements.txt     # once; on Windows "py -m pip ..." also works
python reproduce_all.py                        # everything, about 2 minutes on a laptop
python run_all.py                              # analyses, tables and numbers only
python run_all.py --quick                      # smoke test with reduced Monte Carlo sizes
python run_all.py --no-verify                  # proceed even if a raw file fails its checksum
```

If an automatic installation is not possible (for example without internet access), the
script stops with a message naming the missing package and the exact command that installs
it. Set the environment variable `DIFSS_NO_AUTO_INSTALL=1` to switch automatic installation off.

`reproduce_all.py` runs, in order:

| Step | Command | Produces |
|---|---|---|
| 1 | `python -m pytest -q tests` | 10 unit tests of the theoretical results and the orthopair construction (skipped if pytest is absent) |
| 2 | `python run_all.py` | all analyses; CSV results; LaTeX tables; `numbers.tex`; Figures 5, 10, 11, 12 |
| 3 | `python scripts/build_tikz_figures.py` | Figures 1, 2, 3; icon PNGs |
| 4 | `python scripts/make_policy_and_map.py` | Figure 6 (world map), Figure 13, Table 17 and `tex/numbers_policy.tex` |
| 5 | `python scripts/make_result_charts.py` | Figures 4, 7, 8, 9 |

Step 3 needs LaTeX (TeX Live or MiKTeX, with TikZ and `pgf-blur`) and the poppler tools
`pdftoppm` and `pdftocairo`. Without them the step is skipped and the Figure 1-3 PNGs
shipped in `outputs/figures/` are kept; all other steps need only the Python packages.

`run_all.py` first verifies every raw file against the SHA-256 checksums listed in
`data/raw/SOURCES.md` and stops on a mismatch (use `--no-verify` to override, for example after
re-downloading newer data). All randomness derives from the master seed `20260923` in `run_all.py`, so repeated runs
produce identical outputs. `outputs/run_info.json` records package versions and runtime.
`requirements-exact.txt` lists the exact versions used for the shipped outputs.

## Layout

```
reproduce_all.py            runs every step below (Windows, macOS, Linux)
run_all.py                  data -> analyses -> CSV results, LaTeX tables, number macros, figures
download_data.py            re-downloads the public raw files (into data/raw_fresh/)
requirements.txt            Python dependencies (minimum versions)
requirements-exact.txt      exact versions used for the shipped outputs
src/difss/
  data.py                   loading, ISO3 matching, complete-case panel (Section 6)
  criteria.py               criteria, period sets T_j, hesitancy evidence: sub-indicator
                            disagreement and detrended volatility (Section 4.2)
  model.py                  orthopairs, entropies, IFWG/IFWA/q-rung operators, temporal and
                            criterion weights, pipeline, stability bounds (Sections 4-5)
  benchmarks.py             TOPSIS, weighted sum, IF-TOPSIS, IF-VIKOR, q-rung (Section 7.6)
  analyses.py               entropy options, sensitivity, real-data strategy comparison,
                            IF-WASPAS recency benchmark, SMAA-2, rank reversal,
                            perturbation, threshold subsets (Section 7)
  simulation.py             ground-truth simulation study, linear and step-change DGPs (Section 8)
  report.py                 figures, LaTeX tables, number macros
  figstyle.py               common figure style (fonts, sizes, 600-dpi PNG at the text width)
  check_env.py              checks that the required packages are installed
tests/test_model.py         feasibility, closure and numerical checks of Proposition 4.1,
                            Proposition 5.1, Lemma 5.2, Proposition 5.3 and Corollary 5.4
scripts/
  build_tikz_figures.py     TikZ figures and icons -> outputs/figures/*.png
  make_policy_and_map.py    world map, policy matrix, policy-profile table
  make_result_charts.py     data overview, entropy weights, robustness, benchmark charts
figures_tikz/               TikZ sources; icons.tex is an original icon library
data/raw/                   snapshot of the public sources + SOURCES.md (URLs, dates,
                            licences, SHA-256 checksums)
data/crosswalk/             country crosswalk (ISO3 and names in every source), CCPI matching
data/processed/             analytical panel, coverage counts, excluded countries
data/external/              Natural Earth 1:110m country boundaries (public domain), map only;
                            the Robinson projection is computed in make_policy_and_map.py
outputs/tables/             CSV results (full 132-country ranking, benchmark ranks, ...)
outputs/tex/                numbers.tex and table bodies contained in the manuscript
outputs/figures/            all 13 figures as 600-dpi RGB PNG, 16.46 cm wide (the text width)
fonts/                      Latin Modern Sans (GUST Font License) used for figure text
```

## Where each manuscript item comes from

| Manuscript item | Source (outputs/...) |
|---|---|
| All in-text numbers | `tex/numbers.tex`, `tex/numbers_policy.tex` |
| Fig. 1 research framework; Fig. 2 separable reduction; Fig. 3 flowchart | `figures_tikz/*.tex` -> `figures/fig_framework.png`, `fig_concept.png`, `fig_flowchart.png` |
| Table 1 comparative literature table | written in the manuscript (no computed values) |
| Table 2, Fig. 4 panel construction and data overview | `tables/panel_coverage.csv`, `tex/tab_coverage.tex`, `figures/fig_data_overview.png` |
| Table 3, Fig. 5 temporal weights | `tables/temporal_weights.csv`, `tex/tab_temporal_weights.tex`, `figures/fig_temporal_weights.png` |
| Fig. 6 world map; Table 4 (with rank ranges); Tables 19-20 full ranking (Appendix B) | `tables/full_ranking.csv`, `tex/tab_top_bottom.tex`, `tex/tab_full_ranking_a/b.tex`, `figures/fig_world_map.png` |
| Table 5, Fig. 7 entropy functionals and the shift-invariant standard-deviation variant | `tables/entropy_options.csv`, `tex/tab_entropy_options.tex`, `figures/fig_entropy_weights.png` |
| Table 6, Fig. 8 sensitivity (incl. non-hydro renewable share and CO2 goalposts [0, 25] t) | `tables/robustness.csv`, `tex/tab_robustness.tex`, `figures/fig_robustness.png` |
| Table 7 mixed-frequency strategies on the real panel (interpolation, alignment, omission) | `tables/strategy_comparison.csv`, `strategy_ranks_full.csv`, `tex/tab_strategy.tex` |
| Table 8, Fig. 9 benchmarks (incl. IF-WASPAS with exponential recency) and CCPI | `tables/benchmarks.csv`, `benchmark_ranks_full.csv`, `external_validation_ccpi.csv`, `tex/tab_benchmarks.tex`, `figures/fig_benchmarks.png` |
| Table 9, Fig. 10 SMAA-2 | `tables/smaa_rank_acceptability.csv`, `tex/tab_smaa.tex`, `figures/fig_top20_smaa.png` |
| Table 10 rank reversal (the summary CSV also holds the fixed-weight dominated-alternative check) | `tables/rank_reversal_summary.csv`, `rank_reversal_loo_*.csv`, `tex/tab_rank_reversal.tex` |
| Tables 11-12, Fig. 11 perturbation and bounds | `tables/perturbation.csv`, `entropy_margins.csv`, `tex/tab_perturbation.tex`, `tex/tab_margins.tex`, `figures/fig_perturbation.png` |
| Tables 13-14 sparse-data threshold | `tables/threshold_annual_subsets.csv`, `threshold_policy_subsets.csv`, `tex/tab_threshold*.tex` |
| Tables 15-17, Fig. 12 simulation (linear and step-change DGPs) | `tables/simulation_grid.csv`, `simulation_summary.csv`, `simulation_grid_step.csv`, `simulation_summary_step.csv`, `threshold_simulation.csv`, `tex/tab_simulation.tex`, `tex/tab_simulation_step.tex`, `tex/tab_threshold_sim.tex`, `figures/fig_simulation.png` |
| Table 18, Fig. 13 policy profiles | `tables/policy_groups.csv`, `tex/tab_policy_groups.tex`, `tex/numbers_policy.tex`, `figures/fig_policy_matrix.png` |

## Figure conventions

All figures are drawn at the text width of the Elsevier CAS single-column layout
(16.46 cm) and included at that width, so figure text appears at its nominal size: 8 pt for
axis labels and panel titles and 7 pt for tick labels, legends and annotations (6 pt minimum
in the diagrams). Text is set in Latin Modern Sans and every symbol in Computer Modern math
(LaTeX notation), matching the manuscript; panels are labelled (a), (b), ... in the same way
in every figure.

## Data

See `data/raw/SOURCES.md` for URLs, retrieval dates, licences and SHA-256 checksums.

* **Policy (e1):** World Bank RISE renewable-energy pillar (Data360), reference years
  2015, 2017, 2019, 2021 and 2023 (editions 2016-2024).
* **Renewable electricity share (e2):** Ember via Our World in Data; total generation,
  including nuclear, is the denominator; 2016-2024.
* **CO2 per capita (e3):** World Bank EN.GHG.CO2.PC.CE.AR5, 2016-2024.
* **Sensitivity checks only:** the non-hydro renewable share (Ember renewables minus hydro,
  via Our World in Data), the annual RISE series from Data360 (intermediate years are
  recalculated by RISE, not independently observed), the RISE 2016 database and the IMF Climate Change Indicators
  Dashboard renewable share, RE/(RE+fossil); its generation series ends in 2022, so that
  check uses 2016-2022 for both annual criteria and is compared with a window-matched baseline.
* **External comparison:** CCPI 2024 ranks (primary) and 2023 ranks, transcribed from a tabulation of the
  official results (`data/raw/ccpi_2023_2024_ranks.csv`).
* **Map only:** Natural Earth 1:110m admin-0 boundaries (public domain).

Upstream providers revise their data; `download_data.py` fetches current versions into a
separate folder so that they can be compared with the checksummed snapshot.

## Model settings (primary specification)

Goalpost normalisation (RISE and share on [0, 100]; CO2 on [0, 50] t per capita) ·
membership floor ε = 0.01 · hesitancy scale κ = 0.25, with the hesitancy withheld from the
membership (μ = μ0 − π, ν = 1 − μ0, π = κ h (μ0 − ε), so the score μ − ν = 2μ0 − 1 − π never rises
with hesitancy) · dispersion entropy for temporal and
criterion weights · sparse-data threshold θ = 3 · no staleness decay (λ = 0) · IFWG at both
stages. Hesitancy evidence uses detrended volatility (mean absolute deviation of year-on-year
changes from their own mean, over the pooled range), so steady growth is not treated as
uncertainty. Each country's rank range is obtained from its score interval [2μγ − 1, 1 − 2νγ].
All settings are fields of `difss.model.Config`; the period sets are defined once in
`difss/data.py` (`RISE_YEARS`, `ANNUAL_YEARS`), and every label, table and macro follows them.

## Licence

Code: MIT (see `LICENSE`). Derived outputs and figures: CC BY 4.0. Third-party raw data
remain under their providers' licences.

## Citation

The citation for this work will be added here after the paper is published.
