"""Benchmark ranking methods. All benchmarks reuse the primary model's
normalised inputs, temporal weights and criterion weights, so differences
isolate the aggregation / representation step."""
from __future__ import annotations

import numpy as np

from .model import Config, agg, run_parts, score


def classical_topsis(Zbar, omega):
    """Crisp TOPSIS on the temporally reduced normalised matrix (benefit form)."""
    V = Zbar * omega
    best, worst = V.max(0), V.min(0)
    dp = np.sqrt(((V - best) ** 2).sum(1)); dn = np.sqrt(((V - worst) ** 2).sum(1))
    return dn / (dp + dn)


def weighted_sum(Zbar, omega):
    return Zbar @ omega


def _ifd(m1, n1, m2, n2, omega):
    """Weighted normalised Euclidean IF distance with hesitancy (Szmidt-Kacprzyk)."""
    p1, p2 = 1 - m1 - n1, 1 - m2 - n2
    return np.sqrt(0.5 * (omega * ((m1 - m2) ** 2 + (n1 - n2) ** 2 + (p1 - p2) ** 2)).sum(-1))


def if_topsis(B_mu, B_nu, omega):
    """IF-TOPSIS (Boran et al., 2009) on the reduced criterion-level IFNs."""
    pm, pn = B_mu.max(0), B_nu.min(0)
    nm, nn = B_mu.min(0), B_nu.max(0)
    dp = _ifd(B_mu, B_nu, pm, pn, omega); dn = _ifd(B_mu, B_nu, nm, nn, omega)
    return dn / (dp + dn)


def if_vikor(B_mu, B_nu, omega, v=0.5):
    """IF-VIKOR (Devi, 2011): returns -Q so that larger is better."""
    pm, pn = B_mu.max(0), B_nu.min(0)
    nm, nn = B_mu.min(0), B_nu.max(0)

    def d(m1, n1, m2, n2):
        p1, p2 = 1 - m1 - n1, 1 - m2 - n2
        return np.sqrt(0.5 * ((m1 - m2) ** 2 + (n1 - n2) ** 2 + (p1 - p2) ** 2))

    denom = d(pm, pn, nm, nn)
    denom = np.where(denom > 0, denom, 1.0)
    reg = omega * d(B_mu, B_nu, pm, pn) / denom
    S, R = reg.sum(1), reg.max(1)
    Q = v * (S - S.min()) / max(S.max() - S.min(), 1e-12) + (1 - v) * (R - R.min()) / max(R.max() - R.min(), 1e-12)
    return -Q


def orthopair_family(parts, years_list, base: dict, cfg: Config, op: str):
    """Re-aggregate the same orthopairs with operator `op` using the primary
    temporal and criterion weights."""
    c = Config(**{**cfg.__dict__, "op": op, "crit_w": tuple(base["omega"])})
    out = run_parts(parts, years_list, c, fixed_tw=base["tw"])
    return out["score"], out


def all_benchmarks(parts, years_list, base: dict, cfg: Config, crisp_geometric_score):
    B_mu, B_nu, Z, w = base["B_mu"], base["B_nu"], base["Zbar"], base["omega"]
    res = {
        "Classical TOPSIS (crisp)": classical_topsis(Z, w),
        "Weighted sum (crisp)": weighted_sum(Z, w),
        "Weighted geometric mean (crisp, kappa=0)": crisp_geometric_score,
        "IFWA-DIFSS (arithmetic)": orthopair_family(parts, years_list, base, cfg, "ifwa")[0],
        "IF-TOPSIS": if_topsis(B_mu, B_nu, w),
        "IF-VIKOR (v=0.5)": if_vikor(B_mu, B_nu, w),
        "Pythagorean fuzzy WG (q=2)": orthopair_family(parts, years_list, base, cfg, "q2")[0],
        "q-rung orthopair fuzzy WG (q=3)": orthopair_family(parts, years_list, base, cfg, "q3")[0],
    }
    return res
