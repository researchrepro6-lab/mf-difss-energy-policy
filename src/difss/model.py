"""Mixed-frequency entropy-weighted IFWG-DIFSS model.

Notation follows the manuscript: m alternatives, criterion j observed on its
own period set T_j (p_j = |T_j|). Orthopairs (mu, nu) with pi = 1 - mu - nu.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

YE_L = np.pi * (np.sqrt(2) + 1) / 2          # Lipschitz constant of Ye's entropy


# ---------------------------------------------------------------------------
# Normalisation and orthopair construction
# ---------------------------------------------------------------------------
def minmax_columns(x: np.ndarray, pooled: bool = False) -> np.ndarray:
    """Per-period (column-wise) or pooled min-max normalisation to [0, 1]."""
    x = np.asarray(x, float)
    if pooled:
        lo, hi = np.nanmin(x), np.nanmax(x)
        return (x - lo) / (hi - lo) if hi > lo else np.full_like(x, 0.5)
    lo, hi = np.nanmin(x, axis=0), np.nanmax(x, axis=0)
    rng = np.where(hi > lo, hi - lo, 1.0)
    z = (x - lo) / rng
    z[:, hi <= lo] = 0.5
    return z


def orthopairs(x, benefit: bool, h: np.ndarray, eps: float, kappa: float,
               pooled: bool = False, bounds=None):
    """mu0 = eps + (1 - 2 eps) z ;  pi = kappa * h * (mu0 - eps) ;  mu = mu0 - pi ;  nu = 1 - mu0.

    Hesitancy is withheld from the membership, so the score mu - nu = (2 mu0 - 1) - pi
    decreases with the hesitancy evidence, and mu >= eps is preserved.

    z is the benefit-oriented min-max value; h in [0, 1] is the evidence-based
    hesitancy signal (country- and possibly period-specific).
    """
    if bounds is not None:                       # fixed reference goalposts
        z = np.clip((np.asarray(x, float) - bounds[0]) / (bounds[1] - bounds[0]), 0, 1)
    else:
        z = minmax_columns(x, pooled)
    if not benefit:
        z = 1.0 - z
    mu0 = eps + (1.0 - 2.0 * eps) * z
    pi = kappa * np.clip(h, 0, 1) * (mu0 - eps)
    mu = mu0 - pi
    nu = 1.0 - mu0
    return mu, nu, z


# ---------------------------------------------------------------------------
# Entropy functionals (evaluated on one criterion-period, across alternatives)
# ---------------------------------------------------------------------------
def ent_dispersion(mu, nu, z=None):
    """Proposed dispersion entropy: normalised Shannon entropy of the
    cross-sectional distribution of s = (1 + mu - nu) / 2 (> 0)."""
    s = (1.0 + mu - nu) / 2.0
    q = s / s.sum()
    m = len(s)
    return float(-(q * np.log(q)).sum() / np.log(m))


def ent_ye(mu, nu, z=None):
    """Ye (2010) sine-based intuitionistic fuzzy entropy (mean over alternatives)."""
    g = (np.sin(np.pi * (1 + mu - nu) / 4) + np.sin(np.pi * (1 - mu + nu) / 4) - 1) / (np.sqrt(2) - 1)
    return float(np.mean(g))


def ent_sk(mu, nu, z=None):
    """Szmidt-Kacprzyk (2001) ratio entropy, hesitancy (pi) sensitive."""
    pi = 1 - mu - nu
    return float(np.mean((np.minimum(mu, nu) + pi) / (np.maximum(mu, nu) + pi)))


def ent_shannon_crisp(mu, nu, z):
    """Classical entropy-weight method on crisp normalised values (Zeleny)."""
    zz = np.clip(z, 1e-12, None)
    q = zz / zz.sum()
    return float(-(q * np.log(q)).sum() / np.log(len(q)))


def ent_sd(mu, nu, z=None):
    """Shift-invariant sensitivity variant: 1 - 2 sd(s) with s = (1 + mu - nu) / 2,
    so that 1 - E is proportional to the standard deviation of the scores
    (sd(s) <= 1/2 for s in [0, 1]). Unlike the dispersion entropy, it is
    unchanged when all scores are shifted by a constant."""
    s = (1.0 + mu - nu) / 2.0
    return float(1.0 - 2.0 * np.std(s))


ENTROPIES = {"dispersion": ent_dispersion, "ye": ent_ye, "sk": ent_sk,
             "shannon": ent_shannon_crisp, "sd": ent_sd}


def entropy_weights(E: np.ndarray) -> np.ndarray:
    d = 1.0 - np.asarray(E, float)
    d = np.clip(d, 0, None)
    return d / d.sum() if d.sum() > 0 else np.full(len(d), 1.0 / len(d))


# ---------------------------------------------------------------------------
# Aggregation operators (rows = alternatives, columns = aggregated items)
# ---------------------------------------------------------------------------
def agg(mu, nu, w, op: str = "ifwg"):
    w = np.asarray(w, float)
    if op == "ifwg":
        return np.prod(mu ** w, axis=1), 1 - np.prod((1 - nu) ** w, axis=1)
    if op == "ifwa":
        return 1 - np.prod((1 - mu) ** w, axis=1), np.prod(np.clip(nu, 1e-300, 1) ** w, axis=1)
    if op.startswith("q"):                 # q-rung orthopair weighted geometric
        q = float(op[1:])
        return np.prod(mu ** w, axis=1), (1 - np.prod((1 - nu ** q) ** w, axis=1)) ** (1 / q)
    raise ValueError(op)


def score(mu, nu, op: str = "ifwg"):
    if op.startswith("q"):
        q = float(op[1:])
        return mu ** q - nu ** q
    return mu - nu


# ---------------------------------------------------------------------------
# Configuration and pipeline
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    entropy: str = "dispersion"           # temporal- and criterion-entropy functional
    eps: float = 0.01                     # membership floor
    kappa: float = 0.25                   # hesitancy scale
    theta: int = 3                        # sparse fallback: uniform if p_j <= theta
    lam: float = 0.0                      # staleness decay rate (per year)
    t_ref: int = 2024                     # reference time for staleness (last annual year)
    temporal: str = "entropy"             # entropy | uniform | linear | exponential
    exp_base: float = 0.8
    pooled: bool = False                  # pooled (all-period) normalisation
    op: str = "ifwg"                      # ifwg | ifwa | q2 (Pythagorean) | q3
    crit_w: tuple | None = None           # fixed criterion weights (else entropy)
    hes_policy: str = "subindicator"      # subindicator | none
    hes_annual: str = "volatility"        # volatility | none
    reference: bool = True                # goalpost normalisation (primary); False = min-max


@dataclass
class Criterion:
    name: str
    x: np.ndarray            # m x p raw values
    years: list
    benefit: bool
    h: np.ndarray            # m x p hesitancy evidence in [0, 1]
    bounds: tuple = None     # reference goalposts (lo, hi) for cfg.reference


def temporal_weights(mu, nu, z, years, cfg: Config):
    p = mu.shape[1]
    if cfg.temporal == "uniform":
        w = np.full(p, 1 / p)
        E = np.full(p, np.nan)
    elif cfg.temporal == "linear":
        w = np.arange(1, p + 1, dtype=float); w /= w.sum(); E = np.full(p, np.nan)
    elif cfg.temporal == "exponential":
        w = cfg.exp_base ** np.arange(p - 1, -1, -1); w = w / w.sum(); E = np.full(p, np.nan)
    else:
        f = ENTROPIES[cfg.entropy]
        E = np.array([f(mu[:, k], nu[:, k], z[:, k]) for k in range(p)])
        w = np.full(p, 1 / p) if p <= cfg.theta else entropy_weights(E)
    if cfg.lam > 0:
        age = cfg.t_ref - np.asarray(years, float)
        w = w * np.exp(-cfg.lam * age)
        w = w / w.sum()
    return w, E


def build_parts(criteria: list[Criterion], cfg: Config = Config()):
    return [orthopairs(c.x, c.benefit, c.h, cfg.eps, cfg.kappa, cfg.pooled,
                       c.bounds if cfg.reference else None) for c in criteria]


def run(criteria: list[Criterion], cfg: Config = Config(), return_all: bool = True):
    return run_parts(build_parts(criteria, cfg), [c.years for c in criteria], cfg, return_all)


def run_parts(parts, years_list, cfg: Config = Config(), return_all: bool = True,
              fixed_tw=None):
    """Pipeline from orthopairs. fixed_tw: optional list of temporal-weight
    vectors that override the configured temporal rule."""
    red_mu, red_nu, red_z, tw, tE = [], [], [], [], []
    for j, ((mu, nu, z), years) in enumerate(zip(parts, years_list)):
        if fixed_tw is not None:
            w, E = np.asarray(fixed_tw[j], float), np.full(mu.shape[1], np.nan)
        else:
            w, E = temporal_weights(mu, nu, z, years, cfg)
        bm, bn = agg(mu, nu, w, cfg.op)
        red_mu.append(bm); red_nu.append(bn); red_z.append(z @ w)
        tw.append(w); tE.append(E)
    n = len(parts)
    B_mu = np.column_stack(red_mu); B_nu = np.column_stack(red_nu); Zbar = np.column_stack(red_z)
    if cfg.crit_w is not None:
        omega = np.asarray(cfg.crit_w, float); omega = omega / omega.sum(); CE = np.full(n, np.nan)
    else:
        f = ENTROPIES[cfg.entropy]
        CE = np.array([f(B_mu[:, j], B_nu[:, j], Zbar[:, j]) for j in range(n)])
        omega = entropy_weights(CE)
    g_mu, g_nu = agg(B_mu, B_nu, omega, cfg.op)
    S = score(g_mu, g_nu, cfg.op)
    out = {"score": S, "mu": g_mu, "nu": g_nu, "omega": omega, "crit_entropy": CE,
           "tw": tw, "tE": tE, "B_mu": B_mu, "B_nu": B_nu, "Zbar": Zbar}
    if return_all:
        out["parts"] = parts
    return out


def ranking(S, mu=None, nu=None):
    """Rank 1 = best; ties broken by larger accuracy mu + nu."""
    H = (mu + nu) if mu is not None else np.zeros_like(S)
    order = np.lexsort((-H, -S))
    r = np.empty(len(S), int); r[order] = np.arange(1, len(S) + 1)
    return r


# ---------------------------------------------------------------------------
# Theoretical bounds (Propositions in Section 5)
# ---------------------------------------------------------------------------
def weight_bound(D: float, p: int, Lam: float) -> float:
    """Tightened l1 bound: ||w - w~||_1 <= 2 p Lam / (D - p Lam) (inf if vacuous)."""
    den = D - p * Lam
    return 2 * p * Lam / den if den > 0 else np.inf


def dispersion_lipschitz(s_matrix: np.ndarray, delta: float) -> float:
    """Data-dependent Lipschitz factor Lambda/delta for the dispersion entropy
    (Lemma 5.2). For each period t with s = (1 + mu - nu)/2, total S and m
    alternatives, and any feasible perturbation with |ds_k| <= delta,
        |H - H~| <= delta * sum_k [ A_k + B ] / ((S - m delta) ln m),
    where a_k^- = m (s_k - delta)/(S + m delta), a_k^+ = m (s_k + delta)/(S - m delta),
    A_k = max(|ln a_k^-|, |ln a_k^+|) and B = max(0, ln max_k a_k^+).
    Returns the maximum factor over periods (inf if a denominator is <= 0)."""
    m = s_matrix.shape[0]
    Ls = []
    for t in range(s_matrix.shape[1]):
        s = s_matrix[:, t]; S = s.sum()
        if S - m * delta <= 0 or s.min() - delta <= 0:
            return np.inf
        am = m * (s - delta) / (S + m * delta)
        ap = m * (s + delta) / (S - m * delta)
        A = np.maximum(np.abs(np.log(am)), np.abs(np.log(ap)))
        B = max(0.0, float(np.log(ap.max())))
        Ls.append((A + B).sum() / ((S - m * delta) * np.log(m)))
    return float(max(Ls))


def dispersion_lipschitz_worstcase(m: int, eta: float) -> float:
    """Data-free constant: s_k >= eta, S >= m eta  =>  L = ln(m/eta) / (eta ln m)."""
    return np.log(m / eta) / (eta * np.log(m))
