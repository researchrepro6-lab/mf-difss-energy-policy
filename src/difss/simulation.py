"""Monte Carlo simulation with known ground truth.

Compares four ways of handling mixed-frequency evidence:
  proposed      criterion-specific period sets T_j (this paper)
  interpolate   sparse criteria linearly interpolated to the annual grid
  omit          sparse criteria dropped
  align         only the common (nearest-year aligned) periods are used
All strategies use the same IFWG aggregation, hesitancy rule and entropy
weights, so differences isolate the treatment of reporting frequency.

Two data-generating processes are available:
  linear  smooth country-specific linear trends for every criterion
  step    as linear, but each sparse (policy-type) criterion also has, with
          probability STEP_PROB per country, one discrete reform in a random
          year of the annual window that shifts its latent level by U(STEP_LOW, STEP_HIGH)
The observation pattern mirrors the application: sparse criteria at RISE-type
reference years 2015-2023, annual criteria 2016-2024.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm

from .criteria import volatility_signal
from .model import Config, agg, entropy_weights, ENTROPIES, ranking, score

ANNUAL = np.arange(2016, 2025)
ALL_YEARS = np.arange(2015, 2025)
SPARSE_SETS = {1: [2015], 2: [2015, 2019], 3: [2015, 2019, 2023], 4: [2015, 2017, 2021, 2023],
               5: [2015, 2017, 2019, 2021, 2023]}
GRID_PS = (1, 2, 5)          # sparse observations per criterion in the main grid
STEP_PROB, STEP_LOW, STEP_HIGH = 0.5, 0.10, 0.30   # step DGP: reform probability and jump size


def _orthopairs_nan(x, h, eps, kappa):
    # goalpost normalisation on the known latent scale [0, 1] (as in the
    # empirical primary specification); noisy values are clipped
    # hesitancy withheld from the membership, as in model.orthopairs
    z = np.clip(x, 0, 1)
    mu0 = eps + (1 - 2 * eps) * z
    pi = kappa * h * (mu0 - eps)
    return mu0 - pi, 1 - mu0


def _reduce(mu, nu, cfg: Config, uniform=False):
    """Temporal reduction allowing country-level missing cells (weights are
    renormalised over each country's observed periods)."""
    p = mu.shape[1]
    if uniform or p <= cfg.theta:
        w = np.full(p, 1 / p)
    else:
        E = []
        for k in range(p):
            ok = ~np.isnan(mu[:, k])
            E.append(ENTROPIES[cfg.entropy](mu[ok, k], nu[ok, k], None))
        w = entropy_weights(np.array(E))
    obs = ~np.isnan(mu)
    W = np.where(obs, w, 0.0); W = W / W.sum(1, keepdims=True)
    lm = np.where(obs, np.log(np.where(obs, mu, 1)), 0)
    l1n = np.where(obs, np.log(1 - np.where(obs, nu, 0)), 0)
    return np.exp((W * lm).sum(1)), 1 - np.exp((W * l1n).sum(1))


def _final(Bm, Bn, cfg):
    f = ENTROPIES[cfg.entropy]
    E = np.array([f(Bm[:, j], Bn[:, j], None) for j in range(Bm.shape[1])])
    w = entropy_weights(E)
    gm, gn = agg(Bm, Bn, w, "ifwg")
    return score(gm, gn), gm, gn


def simulate_panel(rng, m, n_annual, n_sparse, p_sparse, sigma, miss, dgp="linear"):
    rho = 0.5
    f = rng.normal(size=m)
    n = n_annual + n_sparse
    lev = norm.cdf(rho * f[:, None] + np.sqrt(1 - rho ** 2) * rng.normal(size=(m, n)))
    slope = rng.normal(0, 0.02, size=(m, n))
    all_years = ALL_YEARS
    path = lev[:, :, None] + slope[:, :, None] * (all_years - all_years.mean())
    if dgp == "step":
        # one discrete reform per sparse criterion and country, with probability STEP_PROB
        reform = rng.random((m, n_sparse)) < STEP_PROB
        year = rng.integers(ANNUAL[0], ANNUAL[-1] + 1, size=(m, n_sparse))
        jump = rng.uniform(STEP_LOW, STEP_HIGH, size=(m, n_sparse))
        step = (all_years[None, None, :] >= year[:, :, None]) * (reform * jump)[:, :, None]
        path[:, n_annual:, :] += step
    elif dgp != "linear":
        raise ValueError(dgp)
    truth_path = np.clip(path, 0, 1)
    truth = truth_path.mean(axis=(1, 2))           # equal-weight latent performance
    annual = [truth_path[:, j, 1:] + rng.normal(0, sigma, (m, len(ANNUAL))) for j in range(n_annual)]
    if miss > 0:
        for a in annual:
            mask = rng.random(a.shape) < miss
            mask[np.all(mask, 1), 0] = False       # keep >= 1 obs per country
            a[mask] = np.nan
    yrs = SPARSE_SETS[p_sparse]
    sparse = []
    for j in range(n_annual, n):
        cols = [int(y - ALL_YEARS[0]) for y in yrs]
        sparse.append(truth_path[:, j, cols] + rng.normal(0, sigma, (m, len(cols))))
    return truth, annual, sparse, yrs


def _interp_row(xs, ys, grid):
    return np.interp(grid, xs, ys)                  # constant extrapolation at ends


def evaluate(rng, m, n_annual, n_sparse, p_sparse, sigma, miss, cfg: Config, dgp="linear"):
    truth, annual, sparse, yrs = simulate_panel(rng, m, n_annual, n_sparse, p_sparse, sigma, miss, dgp)
    t_rank = ranking(truth)
    hA = [volatility_signal(np.where(np.isnan(a), np.nanmean(a, 1, keepdims=True), a)) for a in annual]
    hS = [volatility_signal(s) if s.shape[1] > 1 else np.zeros_like(s) for s in sparse]
    out = {}

    def finish(Bm, Bn):
        S, gm, gn = _final(np.column_stack(Bm), np.column_stack(Bn), cfg)
        r = ranking(S, gm, gn)
        tau = kendalltau(t_rank, r).statistic
        top = len(set(np.where(t_rank <= 10)[0]) & set(np.where(r <= 10)[0])) / 10
        return tau, top

    # proposed (entropy) and proposed with uniform temporal weights
    for label, uni in [("proposed", False), ("proposed_uniform", True)]:
        Bm, Bn = [], []
        for x, h in zip(annual + sparse, hA + hS):
            mu, nu = _orthopairs_nan(x, h, cfg.eps, cfg.kappa)
            bm, bn = _reduce(mu, nu, cfg, uniform=uni)
            Bm.append(bm); Bn.append(bn)
        out[label] = finish(Bm, Bn)
    # interpolation to the annual grid (also fills missing annual cells)
    Bm, Bn = [], []
    filled_a = []
    for a in annual:
        f = a.copy()
        for i in range(m):
            ok = ~np.isnan(a[i])
            f[i] = _interp_row(ANNUAL[ok], a[i, ok], ANNUAL)
        filled_a.append(f)
    filled_s = [np.vstack([_interp_row(np.array(yrs), s[i], ANNUAL) for i in range(m)]) for s in sparse]
    for x in filled_a + filled_s:
        h = volatility_signal(x)
        mu, nu = _orthopairs_nan(x, h, cfg.eps, cfg.kappa)
        bm, bn = _reduce(mu, nu, cfg); Bm.append(bm); Bn.append(bn)
    out["interpolate"] = finish(Bm, Bn)
    # omission of sparse criteria
    Bm, Bn = [], []
    for x, h in zip(annual, hA):
        mu, nu = _orthopairs_nan(x, h, cfg.eps, cfg.kappa)
        bm, bn = _reduce(mu, nu, cfg); Bm.append(bm); Bn.append(bn)
    out["omit"] = finish(Bm, Bn)
    # ad hoc alignment: map each sparse year to nearest annual year, keep only those
    near = sorted({int(min(ANNUAL, key=lambda t: abs(t - y))) for y in yrs})
    cols = [int(t - ANNUAL[0]) for t in near]
    Bm, Bn = [], []
    for x in filled_a:
        xx = x[:, cols]
        mu, nu = _orthopairs_nan(xx, volatility_signal(xx) if len(cols) > 1 else np.zeros_like(xx), cfg.eps, cfg.kappa)
        bm, bn = _reduce(mu, nu, cfg); Bm.append(bm); Bn.append(bn)
    for s, h in zip(sparse, hS):
        mu, nu = _orthopairs_nan(s, h, cfg.eps, cfg.kappa)
        bm, bn = _reduce(mu, nu, cfg); Bm.append(bm); Bn.append(bn)
    out["align"] = finish(Bm, Bn)
    return out


def run_simulation(cfg: Config = Config(), reps: int = 50, seed: int = 20260923, dgp: str = "linear"):
    # separate streams keep the linear-DGP results identical to earlier runs
    rng = np.random.default_rng(seed + (10 if dgp == "linear" else 30))
    grid = itertools.product([40, 135], [2, 4], [1, 2], list(GRID_PS), [0.05, 0.15], [0.0, 0.1])
    rows = []
    for m, na, ns, ps, sg, ms in grid:
        acc = {}
        for _ in range(reps):
            res = evaluate(rng, m, na, ns, ps, sg, ms, cfg, dgp)
            for k, (tau, top) in res.items():
                acc.setdefault(k, []).append((tau, top))
        for k, v in acc.items():
            v = np.array(v)
            rows.append({"dgp": dgp, "m": m, "n_annual": na, "n_sparse": ns, "p_sparse": ps, "sigma": sg,
                         "missing": ms, "method": k, "tau": v[:, 0].mean(), "top10": v[:, 1].mean()})
    return pd.DataFrame(rows)


def run_threshold_simulation(cfg: Config = Config(), reps: int = 200, seed: int = 20260923):
    """Rank recovery of the proposed method when the sparse criteria have
    p in {2,3,4} observations, under fallback thresholds theta in {1,2,3,4}."""
    rng = np.random.default_rng(seed + 20)
    rows = []
    for ps in (2, 3, 4):
        for sg in (0.05, 0.15):
            seeds = rng.integers(0, 2 ** 32 - 1, reps)
            for theta in (1, 2, 3, 4):
                c = Config(**{**cfg.__dict__, "theta": theta})
                taus = []
                for s in seeds:
                    r2 = np.random.default_rng(int(s))
                    taus.append(evaluate(r2, 135, 2, 2, ps, sg, 0.0, c)["proposed"][0])
                rows.append({"p_sparse": ps, "sigma": sg, "theta": theta,
                             "entropy_active_for_sparse": ps > theta, "tau": np.mean(taus),
                             "se": np.std(taus) / np.sqrt(len(taus))})
    return pd.DataFrame(rows)
