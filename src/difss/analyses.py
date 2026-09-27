"""Empirical analyses reported in the manuscript (Sections 7-8)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from . import benchmarks as bm
from .criteria import build_criteria
from .data import ANNUAL_YEARS, LEGACY_SHARE_YEARS, RISE_ANNUAL_YEARS, RISE_YEARS
from .model import (YE_L, Config, build_parts, dispersion_lipschitz, entropy_weights,
                    ranking, run, run_parts, weight_bound, agg, score)


def compare(base_scores, alt_scores, base_rank=None, alt_rank=None, k=10):
    br = ranking(base_scores) if base_rank is None else base_rank
    ar = ranking(alt_scores) if alt_rank is None else alt_rank
    tau = kendalltau(br, ar).statistic
    rho = spearmanr(br, ar).statistic
    top = len(set(np.where(br <= k)[0]) & set(np.where(ar <= k)[0]))
    return tau, rho, top


class Study:
    def __init__(self, data, cfg: Config = Config(), seed: int = 20260923):
        self.d = data
        self.cfg = cfg
        self.seed = seed
        self.C = build_criteria(data)
        self.years = [c.years for c in self.C]
        self.parts = build_parts(self.C, cfg)
        self.base = run_parts(self.parts, self.years, cfg)
        self.rank = ranking(self.base["score"], self.base["mu"], self.base["nu"])
        self.iso = np.array(data["iso"])
        self.names = np.array(data["names"].values)
        self.m = len(self.iso)

    # ------------------------------------------------------------------
    def primary_table(self):
        b = self.base
        df = pd.DataFrame({
            "Rank": self.rank, "ISO3": self.iso, "Country": self.names,
            "Score": b["score"], "mu": b["mu"], "nu": b["nu"], "pi": 1 - b["mu"] - b["nu"],
        })
        crit = ["Policy", "RenewableShare", "CO2"]
        for j, c in enumerate(crit):
            df[f"S_{c}"] = b["B_mu"][:, j] - b["B_nu"][:, j]
            df[f"mu_{c}"] = b["B_mu"][:, j]
            df[f"nu_{c}"] = b["B_nu"][:, j]
        # log-contribution decomposition of the aggregated membership
        lm = b["omega"] * np.log(b["B_mu"])
        ln1 = b["omega"] * np.log(1 - b["B_nu"])
        for j, c in enumerate(crit):
            df[f"logmu_contrib_{c}"] = lm[:, j]
            df[f"log1mnu_contrib_{c}"] = ln1[:, j]
        raw = self.d
        # first and last observed raw values (years: RISE_YEARS[0/-1], ANNUAL_YEARS[0/-1])
        df["RISE_first"], df["RISE_last"] = raw["rise"][:, 0], raw["rise"][:, -1]
        df["REshare_first"], df["REshare_last"] = raw["share"][:, 0], raw["share"][:, -1]
        df["CO2pc_first"], df["CO2pc_last"] = raw["co2"][:, 0], raw["co2"][:, -1]
        return df.sort_values("Rank").reset_index(drop=True)

    def weights_table(self):
        rows = []
        for j, c in enumerate(self.C):
            for y, w, E in zip(c.years, self.base["tw"][j], self.base["tE"][j]):
                rows.append({"criterion": c.name, "year": y, "entropy": E, "weight": w})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def entropy_comparison(self):
        """Options (a) Ye, (b) Shannon on crisp values, (c) dispersion (primary),
        plus the pi-sensitive Szmidt-Kacprzyk entropy."""
        rows, weights = [], {}
        for key, label in [("dispersion", "(c) Dispersion entropy (primary)"),
                           ("ye", "(a) Ye sine-based IF entropy"),
                           ("shannon", "(b) Shannon entropy on crisp values"),
                           ("sk", "Szmidt-Kacprzyk ratio entropy (pi-sensitive)"),
                           ("sd", "Standard deviation of scores (shift-invariant)")]:
            cfg = Config(**{**self.cfg.__dict__, "entropy": key})
            o = run_parts(self.parts, self.years, cfg)
            r = ranking(o["score"], o["mu"], o["nu"])
            tau, rho, top = compare(None, None, self.rank, r)
            rows.append({"specification": label, "key": key, "w_policy": o["omega"][0],
                         "w_share": o["omega"][1], "w_co2": o["omega"][2],
                         "tau": tau, "rho": rho, "top10": top})
            weights[key] = o
        return pd.DataFrame(rows), weights

    # ------------------------------------------------------------------
    def _alt(self, label, cfg=None, criteria=None, rows=None, fixed_tw=None):
        cfg = cfg or self.cfg
        if criteria is not None:
            parts = build_parts(criteria, cfg); years = [c.years for c in criteria]
        else:
            parts, years = build_parts(self.C, cfg), self.years
        o = run_parts(parts, years, cfg, fixed_tw=fixed_tw)
        r_alt = ranking(o["score"], o["mu"], o["nu"])
        if rows is None:
            tau, rho, top = compare(None, None, self.rank, r_alt)
        else:  # subset comparison: rank within subset
            sub_base = ranking(self.base["score"][rows], self.base["mu"][rows], self.base["nu"][rows])
            tau, rho, top = compare(None, None, sub_base, r_alt)
        return {"specification": label, "tau": tau, "rho": rho, "top10": top,
                "w_policy": o["omega"][0], "w_share": o["omega"][1], "w_co2": o["omega"][2]}, o, r_alt

    def robustness(self):
        rows, extra = [], {}
        c = self.cfg.__dict__
        specs = [
            ("Temporal: uniform", Config(**{**c, "temporal": "uniform"})),
            ("Temporal: linear recency", Config(**{**c, "temporal": "linear"})),
            ("Temporal: exponential recency (0.8)", Config(**{**c, "temporal": "exponential"})),
            ("Staleness decay lambda=0.10", Config(**{**c, "lam": 0.10})),
            ("Staleness decay lambda=0.25", Config(**{**c, "lam": 0.25})),
            ("Staleness decay lambda=0.50", Config(**{**c, "lam": 0.50})),
            ("Per-period min-max normalisation", Config(**{**c, "reference": False})),
            (f"Pooled min-max normalisation {ANNUAL_YEARS[0]}-{ANNUAL_YEARS[-1]}", Config(**{**c, "reference": False, "pooled": True})),
            ("Membership floor eps=0.001", Config(**{**c, "eps": 0.001})),
            ("Membership floor eps=0.05", Config(**{**c, "eps": 0.05})),
            ("Hesitancy kappa=0 (crisp geometric)", Config(**{**c, "kappa": 0.0})),
            ("Hesitancy kappa=0.10", Config(**{**c, "kappa": 0.10})),
            ("Hesitancy kappa=0.50", Config(**{**c, "kappa": 0.50})),
            ("Criterion weights equal", Config(**{**c, "crit_w": (1, 1, 1)})),
        ]
        for label, cfg in specs:
            row, o, r = self._alt(label, cfg)
            # bottom-10 overlap for zero-annihilation diagnostics
            row["bottom10"] = len(set(np.where(self.rank > self.m - 10)[0]) & set(np.where(r > self.m - 10)[0]))
            rows.append(row); extra[label] = (o, r)
        # Data-definition sensitivities
        ok = ~np.isnan(self.d["share_legacy"]).any(1)
        crit = build_criteria(self.d, share="legacy", rows=np.where(ok)[0])
        row, o, r = self._alt(f"Alternative share RE/(RE+fossil), IMF dashboard (m={ok.sum()})", criteria=crit, rows=np.where(ok)[0])
        row["bottom10"] = np.nan; rows.append(row); extra["legacy_share"] = (o, r, np.where(ok)[0])
        # window-matched baseline (Ember share, same shorter window and subset) isolates the definition
        crit_b = build_criteria(self.d, annual_idx=[ANNUAL_YEARS.index(y) for y in LEGACY_SHARE_YEARS], rows=np.where(ok)[0])
        ob = run(crit_b, self.cfg, return_all=False)
        extra["legacy_matched"] = ranking(ob["score"], ob["mu"], ob["nu"])
        ok2 = ~np.isnan(self.d["rise2016"]) & ~np.isnan(self.d["rise2016_ind"]).all(1)
        crit = build_criteria(self.d, policy="rise2016", rows=np.where(ok2)[0])
        row, o, r = self._alt(f"Policy = RISE 2016 database, p1=1 (m={ok2.sum()})", criteria=crit, rows=np.where(ok2)[0])
        row["bottom10"] = np.nan; rows.append(row); extra["rise2016"] = (o, r, np.where(ok2)[0])
        # policy observed annually (Data360 annual series) instead of survey years
        crit = build_criteria(self.d, policy="annual")
        row, o, r = self._alt(f"Policy observed annually {RISE_ANNUAL_YEARS[0]}-{RISE_ANNUAL_YEARS[-1]} (Data360 series)", criteria=crit)
        row["bottom10"] = len(set(np.where(self.rank > self.m - 10)[0]) & set(np.where(r > self.m - 10)[0]))
        rows.append(row); extra["rise_annual"] = (o, r)
        # non-hydro renewable share (natural hydropower endowment removed)
        okh = ~np.isnan(self.d["share_nonhydro"]).any(1)
        crit = build_criteria(self.d, share="nonhydro", rows=np.where(okh)[0])
        row, o, r = self._alt(f"Non-hydro renewable share (m={okh.sum()})", criteria=crit, rows=np.where(okh)[0])
        row["bottom10"] = np.nan; rows.append(row); extra["nonhydro"] = (o, r, np.where(okh)[0])
        # lower CO2 upper goalpost
        crit = build_criteria(self.d, co2_goalposts=(0.0, 25.0))
        row, o, r = self._alt("CO2 goalposts [0, 25] t", criteria=crit)
        row["bottom10"] = len(set(np.where(self.rank > self.m - 10)[0]) & set(np.where(r > self.m - 10)[0]))
        rows.append(row); extra["co2_25"] = (o, r)
        # log-transformed CO2 (skewness)
        crit = build_criteria(self.d)
        crit[2].x = np.log(crit[2].x)
        crit[2].bounds = (float(np.log(0.9 * np.nanmin(self.d["co2"]))), float(np.log(50.0)))
        row, o, r = self._alt("CO2 per capita log-transformed", criteria=crit)
        row["bottom10"] = len(set(np.where(self.rank > self.m - 10)[0]) & set(np.where(r > self.m - 10)[0]))
        rows.append(row); extra["logco2"] = (o, r)
        return pd.DataFrame(rows), extra

    # ------------------------------------------------------------------
    def benchmarks(self):
        c = self.cfg.__dict__
        cfg0 = Config(**{**c, "kappa": 0.0, "crit_w": tuple(self.base["omega"])})
        parts0 = build_parts(self.C, cfg0)
        crisp_g = run_parts(parts0, self.years, cfg0, fixed_tw=self.base["tw"])["score"]
        res = bm.all_benchmarks(self.parts, self.years, self.base, self.cfg, crisp_g)
        # IF-WASPAS with exponential recency temporal weights (generic dynamic benchmark):
        # Q = 0.5 S(IFWA_omega) + 0.5 S(IFWG_omega) on criteria reduced with recency weights
        cfg_r = Config(**{**c, "temporal": "exponential"})
        o_r = run_parts(self.parts, self.years, cfg_r)
        q1 = agg(o_r["B_mu"], o_r["B_nu"], o_r["omega"], "ifwa"); q2 = agg(o_r["B_mu"], o_r["B_nu"], o_r["omega"], "ifwg")
        res["IF-WASPAS, exponential recency (0.8)"] = 0.5 * score(*q1) + 0.5 * score(*q2)
        rows, ranks = [], {"IFWG-DIFSS (proposed)": self.rank}
        for k, s in res.items():
            r = ranking(s)
            tau, rho, top = compare(None, None, self.rank, r)
            rows.append({"method": k, "tau": tau, "rho": rho, "top10": top})
            ranks[k] = r
        rank_df = pd.DataFrame({"ISO3": self.iso, "Country": self.names, **ranks}).sort_values(
            "IFWG-DIFSS (proposed)")
        return pd.DataFrame(rows), rank_df

    # ------------------------------------------------------------------
    def external_validation(self, rank_df):
        cc = self.d["ccpi"]
        out = []
        for ed in ["2023", "2024"]:
            e = cc[(cc["edition"].astype(str) == ed) & cc["ISO3"].notna()].set_index("ISO3")["rank"]
            common = [i for i in self.iso if i in e.index]
            idx = [int(np.where(self.iso == i)[0][0]) for i in common]
            ext = e.loc[common].to_numpy(float)
            for method in rank_df.columns[2:]:
                ours = rank_df.set_index("ISO3").loc[common, method].to_numpy(float)
                rho = spearmanr(ours, ext).statistic
                n = len(common); q = int(np.ceil(n / 4))
                top_ours = set(np.argsort(ours)[:q]); top_ext = set(np.argsort(ext)[:q])
                out.append({"edition": f"CCPI {ed}", "method": method, "n_common": n,
                            "spearman": rho, "top_quartile_agreement": len(top_ours & top_ext) / q})
        return pd.DataFrame(out)

    # ------------------------------------------------------------------
    def smaa(self, n=10000):
        rng = np.random.default_rng(self.seed + 1)
        W = rng.dirichlet(np.ones(3), size=n)
        m = self.m
        acc = np.zeros((m, m))
        wsum = np.zeros((m, 3)); wcnt = np.zeros(m)
        for w in W:
            g_mu, g_nu = agg(self.base["B_mu"], self.base["B_nu"], w, self.cfg.op)
            r = ranking(score(g_mu, g_nu, self.cfg.op), g_mu, g_nu)
            acc[np.arange(m), r - 1] += 1
            first_i = int(np.where(r == 1)[0][0])
            wsum[first_i] += w; wcnt[first_i] += 1
        acc /= n
        exp_rank = acc @ np.arange(1, m + 1)
        top10 = acc[:, :10].sum(1)
        first = acc[:, 0]
        # central weight vector = mean weight among draws in which the country ranks first
        cw = {i: wsum[i] / wcnt[i] for i in range(m) if wcnt[i] > 0}
        df = pd.DataFrame({"ISO3": self.iso, "Country": self.names, "baseline_rank": self.rank,
                           "b1": first, "top10_acceptability": top10, "expected_rank": exp_rank})
        for k in range(10):
            df[f"b{k+1}"] = acc[:, k]
        df["central_w_policy"] = [cw.get(i, [np.nan] * 3)[0] for i in range(m)]
        df["central_w_share"] = [cw.get(i, [np.nan] * 3)[1] for i in range(m)]
        df["central_w_co2"] = [cw.get(i, [np.nan] * 3)[2] for i in range(m)]
        return df.sort_values("expected_rank").reset_index(drop=True), acc

    # ------------------------------------------------------------------
    def rank_reversal(self, cfg: Config | None = None):
        cfg = cfg or self.cfg
        base_o = run(self.C, cfg)
        base_rank = ranking(base_o["score"], base_o["mu"], base_o["nu"])
        rows = []
        for i in range(self.m):
            keep = np.array([k for k in range(self.m) if k != i])
            crit = build_criteria(self.d, rows=keep)
            o = run(crit, cfg)
            r = ranking(o["score"], o["mu"], o["nu"])
            b = ranking(base_o["score"][keep], base_o["mu"][keep], base_o["nu"][keep])
            diff = np.sign(b[:, None] - b[None, :]) != np.sign(r[:, None] - r[None, :])
            npairs = len(keep) * (len(keep) - 1)
            top_changed = not np.array_equal(np.argsort(b)[:10], np.argsort(r)[:10])
            rows.append({"removed": self.iso[i], "pair_reversal_share": diff.sum() / npairs,
                         "max_abs_rank_shift": int(np.abs(b - r).max()), "top10_order_changed": top_changed})
        loo = pd.DataFrame(rows)
        # Dominated alternative: worse than the observed minimum in every period
        d2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.d.items()}
        def worse(x, benefit):
            rng = np.nanmax(x, 0) - np.nanmin(x, 0)
            return (np.nanmin(x, 0) - 0.05 * rng) if benefit else (np.nanmax(x, 0) + 0.05 * rng)
        d2["rise"] = np.vstack([self.d["rise"], worse(self.d["rise"], True)])
        d2["share"] = np.vstack([self.d["share"], np.clip(worse(self.d["share"], True), 0, None)])
        d2["co2"] = np.vstack([self.d["co2"], worse(self.d["co2"], False)])
        d2["rise_sub"] = np.concatenate([self.d["rise_sub"], np.nanmedian(self.d["rise_sub"], 0)[None]], 0)
        crit = build_criteria(d2)
        o = run(crit, cfg)
        r = ranking(o["score"], o["mu"], o["nu"])[:-1]
        r = ranking(-r.astype(float))
        diff = np.sign(base_rank[:, None] - base_rank[None, :]) != np.sign(r[:, None] - r[None, :])
        dom = {"pair_reversal_share": diff.sum() / (self.m * (self.m - 1)),
               "max_abs_rank_shift": int(np.abs(base_rank - r).max()),
               "dominated_rank": int(ranking(o["score"], o["mu"], o["nu"])[-1])}
        # same test with the temporal and criterion weights fixed at their baseline values:
        # what remains is the effect of the panel-dependent hesitancy scaling (and of the
        # normalisation, if it is data-dependent)
        cfg_f = Config(**{**cfg.__dict__, "crit_w": tuple(base_o["omega"])})
        o_f = run_parts(build_parts(crit, cfg_f), [c.years for c in crit], cfg_f, fixed_tw=base_o["tw"])
        r_f = ranking(o_f["score"], o_f["mu"], o_f["nu"])[:-1]
        r_f = ranking(-r_f.astype(float))
        diff_f = np.sign(base_rank[:, None] - base_rank[None, :]) != np.sign(r_f[:, None] - r_f[None, :])
        dom["pair_reversal_share_fixed_weights"] = diff_f.sum() / (self.m * (self.m - 1))
        dom["max_abs_rank_shift_fixed_weights"] = int(np.abs(base_rank - r_f).max())
        return loo, dom

    # ------------------------------------------------------------------
    def perturbation(self, deltas=(0.01, 0.02, 0.03, 0.04, 0.05), reps=1000):
        rng = np.random.default_rng(self.seed + 2)
        eps = self.cfg.eps
        D = [float((1 - E).sum()) for E in self.base["tE"]]
        p = [len(y) for y in self.years]
        rows, drift = [], []
        for delta in deltas:
            taus, tops, dws, dS_w, dS_w_bound = [], [], [[] for _ in p], [], []
            for _ in range(reps):
                pp = []
                for (mu, nu, z) in self.parts:
                    mu2 = np.clip(mu + rng.uniform(-delta, delta, mu.shape), eps, 1.0)
                    nu2 = np.clip(nu + rng.uniform(-delta, delta, nu.shape), 0.0, 1.0 - mu2)
                    pp.append((mu2, nu2, z))
                o = run_parts(pp, self.years, self.cfg, return_all=False)
                r = ranking(o["score"], o["mu"], o["nu"])
                tau, _, top = compare(None, None, self.rank, r)
                taus.append(tau); tops.append(top)
                for j in range(len(p)):
                    dws[j].append(np.abs(o["tw"][j] - self.base["tw"][j]).sum())
                # score channel through temporal weights only (Prop. 5.3)
                o_w = run_parts(self.parts, self.years,
                                Config(**{**self.cfg.__dict__, "crit_w": tuple(self.base["omega"])}),
                                return_all=False, fixed_tw=o["tw"])
                dS_w.append(np.abs(o_w["score"] - self.base["score"]).max())
                dS_w_bound.append(np.log(1 / eps) * sum(self.base["omega"][j] * dws[j][-1] for j in range(len(p))))
            gaps = self.adjacent_gaps()
            row = {"delta": delta, "mean_tau": np.mean(taus), "sd_tau": np.std(taus),
                   "gaps_certified_observed": int((gaps > 2 * np.max(dS_w)).sum()),
                   "gaps_certified_bound": int((gaps > 2 * np.max(dS_w_bound)).sum()),
                   "min_tau": np.min(taus), "mean_top10": np.mean(tops) * 10, "min_top10": np.min(tops),
                   "max_dS_weight_channel": np.max(dS_w), "max_dS_bound": np.max(dS_w_bound),
                   "bound_holds_score": bool(np.all(np.array(dS_w) <= np.array(dS_w_bound) + 1e-12))}
            for j, name in enumerate(["policy", "share", "co2"]):
                mu, nu, _ = self.parts[j]
                s_mat = (1 + mu - nu) / 2
                Lam = dispersion_lipschitz(s_mat, delta) * delta
                bnd = weight_bound(D[j], p[j], Lam) if p[j] > self.cfg.theta else 0.0
                row[f"mean_dw_{name}"] = np.mean(dws[j]); row[f"max_dw_{name}"] = np.max(dws[j])
                row[f"bound_dw_{name}"] = bnd
            rows.append(row)
        # Ye-entropy comparison: margins and non-vacuity thresholds
        ye_rows = []
        for j, name in enumerate(["policy", "share", "co2"]):
            mu, nu, z = self.parts[j]
            from .model import ent_ye
            Eye = np.array([ent_ye(mu[:, k], nu[:, k]) for k in range(mu.shape[1])])
            Dye = float((1 - Eye).sum())
            s_mat = (1 + mu - nu) / 2
            # largest delta for which the l1 weight bound is informative (< 2)
            grid = np.linspace(1e-5, 0.2, 4000)
            ok = [g for g in grid if weight_bound(D[j], p[j], dispersion_lipschitz(s_mat, g) * g) < 2]
            Ld = dispersion_lipschitz(s_mat, 0.005)
            ye_rows.append({"criterion": name, "p": p[j], "D_dispersion": D[j],
                            "L_dispersion_at_0.005": Ld,
                            "eps_star_dispersion": max(ok) if ok else np.nan,
                            "D_ye": Dye, "eps_star_ye": Dye / (2 * p[j] * YE_L)})
        return pd.DataFrame(rows), pd.DataFrame(ye_rows)

    # ------------------------------------------------------------------
    def threshold_subsets(self):
        """All annual-year subsets of size p (not only the most recent) under
        fallback thresholds theta in {2,3,4}; policy years fixed at T1."""
        rows = []
        na, nr = len(ANNUAL_YEARS), len(RISE_YEARS)
        for p in range(2, na):
            for theta in (2, 3, 4):
                cfg = Config(**{**self.cfg.__dict__, "theta": theta})
                taus, tops = [], []
                for sub in itertools.combinations(range(na), p):
                    crit = build_criteria(self.d, annual_idx=sub)
                    o = run(crit, cfg, return_all=False)
                    r = ranking(o["score"], o["mu"], o["nu"])
                    tau, _, top = compare(None, None, self.rank, r)
                    taus.append(tau); tops.append(top)
                rows.append({"p_annual": p, "theta": theta, "n_subsets": len(taus),
                             "fallback_active": p <= theta, "mean_tau": np.mean(taus),
                             "min_tau": np.min(taus), "mean_top10": np.mean(tops)})
        # sparse policy criterion subsets
        prow = []
        for p in range(1, nr + 1):
            for theta in (2, 3, 4):
                cfg = Config(**{**self.cfg.__dict__, "theta": theta})
                taus = []
                for sub in itertools.combinations(range(nr), p):
                    crit = build_criteria(self.d, rise_idx=sub)
                    o = run(crit, cfg, return_all=False)
                    taus.append(compare(None, None, self.rank, ranking(o["score"], o["mu"], o["nu"]))[0])
                prow.append({"p_policy": p, "theta": theta, "n_subsets": len(taus),
                             "fallback_active": p <= theta, "mean_tau": np.mean(taus), "min_tau": np.min(taus)})
        return pd.DataFrame(rows), pd.DataFrame(prow)

    def strategy_comparison(self):
        """Real-data comparison of the ways of handling the sparse policy criterion:
        criterion-specific reduction (primary), linear interpolation of the policy
        criterion onto the annual grid, common-period alignment (nearest annual year),
        and omission of the policy criterion. Same orthopairs, IFWG and entropy weights."""
        from .criteria import subindicator_signal
        from .model import Criterion
        C = self.C
        pol, ann = C[0], C[1:]
        AY = np.array(ANNUAL_YEARS, float); RY = np.array(pol.years, float)
        out = {}
        # interpolation (constant beyond the last reference year)
        xi = np.vstack([np.interp(AY, RY, row) for row in pol.x])
        hi = np.vstack([np.interp(AY, RY, row) for row in pol.h])
        out["Linear interpolation of the policy criterion"] = [
            Criterion(pol.name, xi, list(ANNUAL_YEARS), True, hi, pol.bounds)] + list(ann)
        # common-period alignment: nearest annual year to each reference year
        near = sorted({int(AY[np.argmin(np.abs(AY - y))]) for y in RY})
        cols = [ANNUAL_YEARS.index(y) for y in near]
        crit_a = build_criteria(self.d, annual_idx=cols)
        crit_a[0] = Criterion(pol.name, pol.x, near, True, pol.h, pol.bounds)
        out["Common-period alignment"] = crit_a
        # omission of the sparse criterion
        out["Omission of the policy criterion"] = list(ann)
        rows, ranks = [], {}
        top_base = set(np.where(self.rank <= 10)[0])
        for label, crit in out.items():
            o = run(crit, self.cfg)
            r = ranking(o["score"], o["mu"], o["nu"])
            tau, rho, top = compare(None, None, self.rank, r)
            shift = np.abs(r - self.rank)
            top_alt = set(np.where(r <= 10)[0])
            rows.append({"strategy": label, "tau": tau, "rho": rho, "top10": top,
                         "n_shift5": int((shift >= 5).sum()), "max_shift": int(shift.max()),
                         "median_shift": float(np.median(shift)),
                         "enter_top10": "; ".join(self.names[sorted(top_alt - top_base, key=lambda i: r[i])]),
                         "leave_top10": "; ".join(self.names[sorted(top_base - top_alt, key=lambda i: self.rank[i])])})
            ranks[label] = r
        return pd.DataFrame(rows), ranks

    def adjacent_gaps(self):
        s = np.sort(self.base["score"])[::-1]
        g = s[:-1] - s[1:]
        return g
