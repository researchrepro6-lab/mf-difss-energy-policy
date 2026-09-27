"""Turn the processed panel into Criterion objects (with hesitancy evidence)."""
from __future__ import annotations

import numpy as np

from .data import ANNUAL_YEARS, LEGACY_SHARE_YEARS, RISE_ANNUAL_YEARS, RISE_YEARS
from .model import Criterion

# Reference goalposts for CO2 per capita (t CO2e): 0 and 50 (above the largest
# value observed in the panel; see the CoMaxRaw macro). Goalposts are the primary
# normalisation; Config(reference=False) switches to min-max.
CO2_GOALPOSTS = (0.0, 50.0)


def volatility_signal(x: np.ndarray) -> np.ndarray:
    """Country-level temporal instability of a series, in [0, 1], net of trend.

    With d_ik = x_i(t_k) - x_i(t_{k-1}) and dbar_i its country mean (the average
    trend), v_i = mean_k |d_ik - dbar_i| / (pooled range of the criterion) and
    h_i = min(1, v_i / q90(v)), broadcast to every observed period. Sustained
    growth or decline therefore does not count as instability; only irregular
    year-to-year movement around the country's own trend does.
    """
    x = np.asarray(x, float)
    rng = np.nanmax(x) - np.nanmin(x)
    d = np.diff(x, axis=1)
    if d.shape[1] == 0:
        return np.zeros_like(x)
    dev = d - np.nanmean(d, axis=1, keepdims=True)
    v = np.nanmean(np.abs(dev), axis=1) / (rng if rng > 0 else 1.0)
    q = np.nanquantile(v, 0.9)
    h = np.clip(v / q, 0, 1) if q > 0 else np.zeros_like(v)
    return np.repeat(h[:, None], x.shape[1], axis=1)


def subindicator_signal(sub: np.ndarray) -> np.ndarray:
    """Within-country disagreement among the five RISE renewable-energy
    sub-indicators at each survey year: sd / 50 (50 is an upper bound for the
    standard deviation of values in [0, 100])."""
    return np.clip(np.nanstd(sub, axis=2) / 50.0, 0, 1)


def build_criteria(data: dict, policy: str = "data360", share: str = "ember",
                   rise_idx=None, annual_idx=None, hes: bool = True,
                   rows=None, co2_goalposts=CO2_GOALPOSTS) -> list[Criterion]:
    """policy: 'data360' (T1 = RISE reference years), 'annual' (Data360 annual series) or 'rise2016' (single 2015 obs,
    legacy file); share: 'ember' (incl. nuclear in denominator) or 'legacy'."""
    r = slice(None) if rows is None else rows
    if share == "legacy" and annual_idx is None:     # IMF series is shorter: restrict both annual criteria
        annual_idx = [ANNUAL_YEARS.index(y) for y in LEGACY_SHARE_YEARS]
    ai = list(range(len(ANNUAL_YEARS))) if annual_idx is None else list(annual_idx)
    years_a = [ANNUAL_YEARS[k] for k in ai]

    if policy == "data360":
        ri = list(range(len(RISE_YEARS))) if rise_idx is None else list(rise_idx)
        x1 = data["rise"][r][:, ri]
        h1 = subindicator_signal(data["rise_sub"][r][:, ri, :])
        y1 = [RISE_YEARS[k] for k in ri]
    elif policy == "annual":  # Data360 annual RISE values (sensitivity)
        x1 = data["rise_annual"][r]
        h1 = subindicator_signal(data["rise_sub_annual"][r])
        y1 = list(RISE_ANNUAL_YEARS)
    else:  # legacy RISE 2016 workbook: one observation (2015 regulatory information)
        x1 = data["rise2016"][r][:, None]
        h1 = np.clip(np.nanstd(data["rise2016_ind"][r], axis=1) / 50.0, 0, 1)[:, None]
        y1 = [2015]

    if share == "ember":
        xs = data["share"][r][:, ai]
    elif share == "nonhydro":
        xs = data["share_nonhydro"][r][:, ai]
    else:
        la = [LEGACY_SHARE_YEARS.index(ANNUAL_YEARS[k]) for k in ai]
        xs = data["share_legacy"][r][:, la]
    x3 = data["co2"][r][:, ai]
    h2 = volatility_signal(xs)
    h3 = volatility_signal(x3)
    if not hes:
        h1, h2, h3 = np.zeros_like(h1), np.zeros_like(h2), np.zeros_like(h3)
    return [
        Criterion("Policy (RISE RE pillar)", x1, y1, True, h1, (0.0, 100.0)),
        Criterion("Renewable electricity share", xs, years_a, True, h2, (0.0, 100.0)),
        Criterion("CO2 per capita (cost)", x3, years_a, False, h3, tuple(co2_goalposts)),
    ]
