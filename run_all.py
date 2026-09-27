#!/usr/bin/env python3
"""Reproduce every table, figure and in-text number of the manuscript.

    python run_all.py            # full run (about 2 minutes on a laptop)
    python run_all.py --quick    # reduced Monte Carlo sizes for a smoke test

Inputs : data/raw/            (snapshot of public sources; see data/raw/SOURCES.md)
Outputs: data/processed/, data/crosswalk/, outputs/tables/*.csv,
         outputs/figures/*.png, outputs/tex/*.tex (tables + numbers.tex)
All random numbers derive from the master seed below.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from difss.check_env import require                            # noqa: E402
require()                                                       # clear message if a package is missing
if hasattr(sys.stdout, "reconfigure"):                          # UTF-8 console output on Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

from difss import analyses, data, report, simulation            # noqa: E402
from difss.data import ANNUAL_YEARS, RISE_ANNUAL_YEARS, RISE_YEARS   # noqa: E402
from difss.model import Config, YE_L                            # noqa: E402
from difss.report import f3, tex_escape, write_table            # noqa: E402

MASTER_SEED = 20260923
TAB, FIG, TEX = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures", ROOT / "outputs" / "tex"


def verify_checksums(strict: bool = True):
    """Compare every raw file with the SHA-256 checksum listed in data/raw/SOURCES.md."""
    import hashlib
    import re
    raw = ROOT / "data" / "raw"
    listed = re.findall(r"^([0-9a-f]{64})\s+(\S+)$", (raw / "SOURCES.md").read_text(encoding="utf-8"), re.M)
    bad = []
    for digest, name in listed:
        f = raw / name
        got = hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else "missing"
        if got != digest:
            bad.append(f"{name}: expected {digest[:12]}..., found {got[:12]}")
    if bad:
        msg = "Raw-data checksum mismatch (the snapshot differs from the one analysed):\n  " + "\n  ".join(bad)
        if strict:
            raise SystemExit(msg + "\nRun with --no-verify to proceed anyway.")
        print("WARNING: " + msg)
    else:
        print(f"Checksums verified for {len(listed)} raw files.")


def main(quick: bool = False, verify: bool = True):
    t0 = time.time()
    verify_checksums(strict=verify)
    for p in (TAB, FIG, TEX):
        p.mkdir(parents=True, exist_ok=True)
    reps_pert = 100 if quick else 1000
    n_smaa = 2000 if quick else 10000
    reps_sim = 10 if quick else 50
    reps_th = 40 if quick else 200

    M = report.Macros()
    d = data.build_panel(verbose=True)
    cfg = Config()
    assert cfg.t_ref == ANNUAL_YEARS[-1], "staleness reference year must be the last annual year"
    S = analyses.Study(d, cfg, seed=MASTER_SEED)
    m = S.m
    M.add("NumCountries", m, "{:d}")
    words = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    M.add("YearRiseFirst", RISE_YEARS[0], "{:d}"); M.add("YearRiseLast", RISE_YEARS[-1], "{:d}")
    M.add("YearAnnualFirst", ANNUAL_YEARS[0], "{:d}"); M.add("YearAnnualLast", ANNUAL_YEARS[-1], "{:d}")
    M.add("YearRiseAnnualLast", RISE_ANNUAL_YEARS[-1], "{:d}")
    M.add("NRiseYears", len(RISE_YEARS), "{:d}"); M.add("NRiseYearsWord", words[len(RISE_YEARS)], "{}")
    M.add("NAnnualYears", len(ANNUAL_YEARS), "{:d}"); M.add("NRiseAnnualYears", len(RISE_ANNUAL_YEARS), "{:d}")
    M.add("RiseYearsList", ", ".join(map(str, RISE_YEARS[:-1])) + f" and {RISE_YEARS[-1]}", "{}")
    M.add("RiseGapYears", ", ".join(map(str, [y for y in ANNUAL_YEARS if y not in RISE_YEARS][:-1])) +
          f" and {[y for y in ANNUAL_YEARS if y not in RISE_YEARS][-1]}", "{}")
    cov = d["coverage"]; cov.to_csv(TAB / "panel_coverage.csv", index=False)
    write_table(TEX / "tab_coverage.tex", ["Selection stage", "Countries"],
                [[tex_escape(a), str(b)] for a, b in cov.itertuples(index=False)], "lr")

    # ---------------- primary results ----------------
    prim = S.primary_table(); prim.to_csv(TAB / "full_ranking.csv", index=False)
    wt = S.weights_table(); wt.to_csv(TAB / "temporal_weights.csv", index=False)
    om = S.base["omega"]; CE = S.base["crit_entropy"]
    for nm, v in zip(["WPolicy", "WShare", "WCOtwo"], om):
        M.add(nm, v)
    for nm, v in zip(["EPolicy", "EShare", "ECOtwo"], CE):
        M.add(nm, v)
    M.add("ScoreMax", prim.Score.max()); M.add("ScoreMin", prim.Score.min()); M.add("ScoreMean", prim.Score.mean())
    for k in range(5):
        M.add("Top" + "ABCDE"[k], prim.Country.iloc[k], "{}")
    for k in range(3):
        M.add("Bottom" + "ABC"[k], prim.Country.iloc[-1 - k], "{}")
    # temporal weights table: one row per year of T_1 or T_2 = T_3
    rows = []
    for y in sorted(set(RISE_YEARS) | set(ANNUAL_YEARS)):
        r = [str(y)]
        for j in range(3):
            q = wt[(wt.criterion == S.C[j].name) & (wt.year == y)]
            r.append(f"{q.weight.iloc[0]:.3f}" if len(q) else "--")
            r.append(f"{q.entropy.iloc[0]:.4f}" if len(q) else "--")
        rows.append(r)
    write_table(TEX / "tab_temporal_weights.tex",
                ["Year", r"$w^{(1)}$", r"$E_1(t)$", r"$w^{(2)}$", r"$E_2(t)$", r"$w^{(3)}$", r"$E_3(t)$"],
                rows, "lcccccc")
    pw = wt[wt.criterion.str.startswith("Policy")].weight.to_numpy()
    sw = wt[wt.criterion.str.startswith("Renew")].weight.to_numpy()
    cw = wt[wt.criterion.str.startswith("CO2")].weight.to_numpy()
    M.add("PolWFirst", pw[0]); M.add("PolWLast", pw[-1])
    M.add("ShWFirst", sw[0]); M.add("ShWLast", sw[-1])
    M.add("CoWMin", cw.min()); M.add("CoWMax", cw.max())
    cy = wt[wt.criterion.str.startswith("CO2")].year.to_numpy(); py = wt[wt.criterion.str.startswith("Policy")].year.to_numpy()
    M.add("CoWMinYear", int(cy[int(np.argmin(cw))]), "{:d}")
    M.add("PolWMin", pw.min()); M.add("PolWMinYear", int(py[int(np.argmin(pw))]), "{:d}")
    co22 = d["co2"][:, -1]
    M.add("CoMedian", float(np.median(co22)), "{:.2f}")
    M.add("CoShareHigh", 100 * float(np.mean(1 - co22 / 50 >= 0.8)), "{:.0f}")
    M.add("CoMaxRaw", float(np.nanmax(d["co2"])), "{:.1f}")
    M.add("DPolicy", float((1 - S.base["tE"][0]).sum()), "{:.4f}")
    M.add("DShare", float((1 - S.base["tE"][1]).sum()), "{:.4f}")
    M.add("DCOtwo", float((1 - S.base["tE"][2]).sum()), "{:.4f}")

    # hesitancy interval of each final score: pessimistic S_L = 2 mu - 1 (all hesitancy
    # counted against), optimistic S_U = 1 - 2 nu (all hesitancy counted in favour)
    from difss.model import ranking as _rank
    prim["rank_optimistic"] = _rank(1 - 2 * prim.nu.to_numpy())
    prim["rank_pessimistic"] = _rank(2 * prim.mu.to_numpy() - 1)
    prim["rank_lo"] = np.minimum(prim.rank_optimistic, prim.rank_pessimistic)
    prim["rank_hi"] = np.maximum(prim.rank_optimistic, prim.rank_pessimistic)
    prim.to_csv(TAB / "full_ranking.csv", index=False)
    width = (prim.rank_hi - prim.rank_lo).to_numpy()
    M.add("RangeMedWidth", float(np.median(width)), "{:.0f}"); M.add("RangeMaxWidth", int(width.max()), "{:d}")
    M.add("RangeWideN", int((width > 5).sum()), "{:d}")
    top10 = prim[prim.Rank <= 10]
    M.add("RangeTopStable", int(((top10.rank_hi <= 10)).sum()), "{:d}")
    M.add("PiMedian", float(np.median(prim.pi)), "{:.3f}")

    # top/bottom table
    rows = []
    for _, r in prim.head(15).iterrows():
        rows.append([str(r.Rank), tex_escape(r.Country), f3(r.Score), f3(r.S_Policy), f3(r.S_RenewableShare),
                     f3(r.S_CO2), f"{r.RISE_last:.1f}", f"{r.REshare_last:.1f}", f"{r.CO2pc_last:.2f}",
                     f"{int(r.rank_lo)}--{int(r.rank_hi)}"])
    rows.append("MIDRULE")
    for _, r in prim.tail(5).iterrows():
        rows.append([str(r.Rank), tex_escape(r.Country), f3(r.Score), f3(r.S_Policy), f3(r.S_RenewableShare),
                     f3(r.S_CO2), f"{r.RISE_last:.1f}", f"{r.REshare_last:.1f}", f"{r.CO2pc_last:.2f}",
                     f"{int(r.rank_lo)}--{int(r.rank_hi)}"])
    write_table(TEX / "tab_top_bottom.tex",
                ["Rank", "Country", r"$S(\gamma)$", r"$S(\beta_1)$", r"$S(\beta_2)$", r"$S(\beta_3)$",
                 f"RISE {RISE_YEARS[-1]}", rf"RE\% {ANNUAL_YEARS[-1]}", rf"CO$_2$ {ANNUAL_YEARS[-1]}", "Rank range"], rows, "rlrrrrrrrc")

    # real-data comparison of mixed-frequency strategies (decision sensitivity)
    stg, stg_ranks = S.strategy_comparison(); stg.to_csv(TAB / "strategy_comparison.csv", index=False)
    pd.DataFrame({"ISO3": d["iso"], "Country": d["names"].values, "proposed": S.rank, **stg_ranks}).to_csv(
        TAB / "strategy_ranks_full.csv", index=False)
    write_table(TEX / "tab_strategy.tex",
                ["Strategy", r"$\tau$", "Top-10", r"Shift $\ge5$", "Max.\\ shift", "Enter top 10", "Leave top 10"],
                [[r.strategy, f3(r.tau), f"{r.top10}/10", str(r.n_shift5), str(r.max_shift),
                  tex_escape(r.enter_top10) or "--", tex_escape(r.leave_top10) or "--"] for r in stg.itertuples()],
                ">{\\raggedright\\arraybackslash}p{3.0cm}cccc>{\\raggedright\\arraybackslash}p{3.0cm}>{\\raggedright\\arraybackslash}p{3.0cm}")
    for tag, lab in [("Interp", "Linear interpolation"), ("Align", "Common-period"), ("Omit", "Omission")]:
        q = stg[stg.strategy.str.startswith(lab)].iloc[0]
        M.add(f"Stg{tag}Tau", q.tau); M.add(f"Stg{tag}Top", int(q.top10), "{:d}")
        M.add(f"Stg{tag}Shift", int(q.n_shift5), "{:d}"); M.add(f"Stg{tag}Max", int(q.max_shift), "{:d}")
    # case diagnostics
    for iso, tag in [("UGA", "Uganda"), ("TCD", "Chad"), ("QAT", "Qatar"), ("BRA", "Brazil"),
                     ("FRA", "France"), ("CHE", "Switzerland"), ("SWE", "Sweden"), ("KEN", "Kenya"), ("FIN", "Finland")]:
        r = prim[prim.ISO3 == iso].iloc[0]
        M.add(f"Rank{tag}", int(r.Rank), "{:d}")
        M.add(f"Pol{tag}", r.S_Policy); M.add(f"Sh{tag}", r.S_RenewableShare); M.add(f"Co{tag}", r.S_CO2)
        M.add(f"RiseRaw{tag}", r.RISE_last, "{:.1f}"); M.add(f"ShareRaw{tag}", r.REshare_last, "{:.1f}")
        M.add(f"CoRaw{tag}", r.CO2pc_last, "{:.2f}")

    # full ranking appendix: two tables, each with two blocks of 34 rows
    n = len(prim); half = int(np.ceil(n / 2)); k = int(np.ceil(half / 2))
    M.add("RankHalf", half, "{:d}"); M.add("RankHalfNext", half + 1, "{:d}")
    for part, start in (("a", 0), ("b", half)):
        rows = []
        for i in range(k):
            r = []
            for bl in range(2):
                j = start + i + bl * k
                if j < min(n, start + half):
                    rr = prim.iloc[j]; r += [str(rr.Rank), tex_escape(rr.Country), f3(rr.Score)]
                else:
                    r += ["", "", ""]
            rows.append(r)
        write_table(TEX / f"tab_full_ranking_{part}.tex", ["Rank", "Country", r"$S(\gamma)$"] * 2, rows, "rlr|rlr")

    # ---------------- entropy options ----------------
    ENT_LABEL = {"dispersion": "Dispersion entropy (proposed)",
                 "shannon": "Shannon entropy on crisp values (entropy-weight method)",
                 "sk": r"Szmidt--Kacprzyk ratio entropy ($\pi$-sensitive)",
                 "ye": "Ye sine entropy",
                 "sd": "Standard deviation of scores (shift-invariant variant)"}
    ent, _ = S.entropy_comparison(); ent.to_csv(TAB / "entropy_options.csv", index=False)
    write_table(TEX / "tab_entropy_options.tex",
                ["Entropy functional", r"$\omega_1$", r"$\omega_2$", r"$\omega_3$", r"$\tau$", r"$\rho$", "Top-10"],
                [[ENT_LABEL[r.key], f3(r.w_policy), f3(r.w_share), f3(r.w_co2), f3(r.tau), f3(r.rho), f"{r.top10}/10"]
                 for r in ent.set_index("key", drop=False).loc[list(ENT_LABEL)].itertuples()], "lcccccc")
    for key, tag in [("ye", "Ye"), ("shannon", "Shannon"), ("sk", "SK"), ("sd", "SD")]:
        r = ent[ent.key == key].iloc[0]
        M.add(f"Tau{tag}", r.tau); M.add(f"Top{tag}", int(r.top10), "{:d}")
        M.add(f"WCo{tag}", r.w_co2); M.add(f"WPol{tag}", r.w_policy); M.add(f"WSh{tag}", r.w_share)

    # ---------------- robustness ----------------
    rb, extra = S.robustness(); rb.to_csv(TAB / "robustness.csv", index=False)
    write_table(TEX / "tab_robustness.tex",
                ["Alternative specification", r"$\omega_1$", r"$\omega_2$", r"$\omega_3$", r"$\tau$", r"$\rho$",
                 "Top-10", "Bottom-10"],
                [[report.robust_label(r.specification), f3(r.w_policy), f3(r.w_share), f3(r.w_co2), f3(r.tau), f3(r.rho),
                  f"{r.top10}/10", "--" if pd.isna(r.bottom10) else f"{int(r.bottom10)}/10"] for r in rb.itertuples()],
                "lccccccc")
    def rbv(pat, col="tau"):
        return rb[rb.specification.str.contains(pat, regex=False)].iloc[0][col]
    M.add("TauUniform", rbv("Temporal: uniform")); M.add("TauLinear", rbv("linear recency"))
    M.add("TauExp", rbv("exponential")); M.add("TauLamA", rbv("lambda=0.10")); M.add("TauLamC", rbv("lambda=0.50"))
    M.add("TauMinMax", rbv("Per-period min-max")); M.add("TauPooled", rbv("Pooled min-max"))
    M.add("TauEpsLow", rbv("eps=0.001")); M.add("TauEpsHigh", rbv("eps=0.05"))
    M.add("TauKappaZero", rbv("kappa=0 ")); M.add("TopKappaZero", int(rbv("kappa=0 ", "top10")), "{:d}"); M.add("TauKappaHigh", rbv("kappa=0.50"))
    M.add("TauEqualW", rbv("Criterion weights equal")); M.add("TauLegacyShare", rbv("Alternative share"))
    M.add("TopLegacyShare", int(rbv("Alternative share", "top10")), "{:d}")
    M.add("TauRiseOld", rbv("RISE 2016 database")); M.add("TopRiseOld", int(rbv("RISE 2016 database", "top10")), "{:d}")
    M.add("TauNonHydro", rbv("Non-hydro")); M.add("TopNonHydro", int(rbv("Non-hydro", "top10")), "{:d}")
    M.add("MNonHydro", len(extra["nonhydro"][2]), "{:d}")
    _o, _r, _idx = extra["nonhydro"]
    _nm = np.array(d["names"].values)[_idx][np.argsort(_r)[:5]]
    M.add("NonHydroTopNames", ", ".join(_nm[:-1]) + " and " + _nm[-1], "{}")
    M.add("TauCoTwentyFive", rbv("CO2 goalposts [0, 25]")); M.add("TopCoTwentyFive", int(rbv("CO2 goalposts [0, 25]", "top10")), "{:d}")
    M.add("WCoTwentyFive", rbv("CO2 goalposts [0, 25]", "w_co2"))
    M.add("NCoAboveTwentyFive", int((np.nanmax(d["co2"], axis=1) > 25).sum()), "{:d}")
    M.add("TauLogCo", rbv("log-transformed")); M.add("WCoLog", rbv("log-transformed", "w_co2"))
    M.add("BotEpsHigh", int(rbv("eps=0.05", "bottom10")), "{:d}"); M.add("BotEpsLow", int(rbv("eps=0.001", "bottom10")), "{:d}")
    o_leg, r_leg, idx_leg = extra["legacy_share"]
    iso_leg = np.array(d["iso"])[idx_leg]
    from difss.model import ranking as _rk
    sub_rank = extra["legacy_matched"]          # Ember share, same shorter window and subset
    from scipy.stats import kendalltau as _kt
    M.add("TauLegacyMatched", _kt(sub_rank, r_leg).statistic)
    M.add("YearLegacyLast", data.LEGACY_SHARE_YEARS[-1], "{:d}")
    for c, tag in [("FRA", "France"), ("CHE", "Switzerland"), ("SWE", "Sweden"), ("FIN", "Finland")]:
        k_ = np.where(iso_leg == c)[0][0]
        M.add(f"LegRank{tag}", int(r_leg[k_]), "{:d}")
        M.add(f"SubRank{tag}", int(sub_rank[k_]), "{:d}")
    for c, tag in [("CHE", "Switzerland"), ("SWE", "Sweden"), ("AUT", "Austria"), ("DNK", "Denmark")]:
        M.add(f"RiseFirst{tag}", d["rise"][list(d["iso"]).index(c), 0], "{:.1f}")
    M.add("RankGermany", int(prim[prim.ISO3 == "DEU"].Rank.iloc[0]), "{:d}")
    M.add("TauRiseAnnual", rbv("Policy observed annually")); M.add("TopRiseAnnual", int(rbv("Policy observed annually", "top10")), "{:d}")
    M.add("LegM", len(idx_leg), "{:d}")
    ichad = list(d["iso"]).index("TCD")
    M.add("ChadRiseFirst", d["rise"][ichad, 0], "{:.1f}"); M.add("ChadRiseSecond", d["rise"][ichad, 1], "{:.1f}")
    # alternative share definition: shares under both definitions
    leg = d["share_legacy"]; iso = np.array(d["iso"])
    for c, tag in [("FRA", "France"), ("BEL", "Belgium"), ("KEN", "Kenya")]:
        i = int(np.where(iso == c)[0][0])
        M.add(f"LegShare{tag}", leg[i, -1], "{:.1f}")
        M.add(f"NewShare{tag}", d["share"][i, ANNUAL_YEARS.index(data.LEGACY_SHARE_YEARS[-1])], "{:.1f}")

    # ---------------- benchmarks + external validation ----------------
    bmk, rank_df = S.benchmarks(); rank_df.to_csv(TAB / "benchmark_ranks_full.csv", index=False)
    ext = S.external_validation(rank_df); ext.to_csv(TAB / "external_validation_ccpi.csv", index=False)
    # primary external comparison: the most recent CCPI edition in the data snapshot
    CCPI_PRIMARY, CCPI_OTHER = "CCPI 2024", "CCPI 2023"
    e23 = ext[ext.edition == CCPI_PRIMARY].set_index("method")
    M.add("CcpiEd", CCPI_PRIMARY.split()[1], "{}"); M.add("CcpiOtherEd", CCPI_OTHER.split()[1], "{}")
    bmk.to_csv(TAB / "benchmarks.csv", index=False)
    rows = [["IFWG-DIFSS (proposed)", "1.000", "1.000", "10/10", f3(e23.loc["IFWG-DIFSS (proposed)", "spearman"]),
             f3(e23.loc["IFWG-DIFSS (proposed)", "top_quartile_agreement"])]]
    for r in bmk.itertuples():
        rows.append([tex_escape(r.method), f3(r.tau), f3(r.rho), f"{r.top10}/10", f3(e23.loc[r.method, "spearman"]),
                     f3(e23.loc[r.method, "top_quartile_agreement"])])
    write_table(TEX / "tab_benchmarks.tex",
                ["Method", r"$\tau$", r"$\rho$", "Top-10", r"$\rho_{\mathrm{CCPI}}$", "Top-quartile"], rows, "lccccc")
    b = bmk.set_index("method")
    M.add("TauTopsis", b.loc["Classical TOPSIS (crisp)", "tau"]); M.add("TauWsum", b.loc["Weighted sum (crisp)", "tau"])
    M.add("TauWgm", b.loc["Weighted geometric mean (crisp, kappa=0)", "tau"])
    M.add("TauIfwa", b.loc["IFWA-DIFSS (arithmetic)", "tau"]); M.add("TopIfwa", int(b.loc["IFWA-DIFSS (arithmetic)", "top10"]), "{:d}")
    M.add("TauIftopsis", b.loc["IF-TOPSIS", "tau"]); M.add("TauIfvikor", b.loc["IF-VIKOR (v=0.5)", "tau"])
    M.add("TauWaspas", b.loc["IF-WASPAS, exponential recency (0.8)", "tau"]); M.add("TopWaspas", int(b.loc["IF-WASPAS, exponential recency (0.8)", "top10"]), "{:d}")
    M.add("TauPfwg", b.loc["Pythagorean fuzzy WG (q=2)", "tau"]); M.add("TauQrof", b.loc["q-rung orthopair fuzzy WG (q=3)", "tau"])
    M.add("NCcpi", int(e23.iloc[0].n_common), "{:d}")
    M.add("RhoCcpi", e23.loc["IFWG-DIFSS (proposed)", "spearman"])
    M.add("RhoCcpiMin", e23.spearman.min()); M.add("RhoCcpiMax", e23.spearman.max())
    M.add("RhoCcpiNext", ext[(ext.edition == CCPI_OTHER) & (ext.method == "IFWG-DIFSS (proposed)")].spearman.iloc[0])
    M.add("NCcpiOther", int(ext[ext.edition == CCPI_OTHER].iloc[0].n_common), "{:d}")
    M.add("QuartCcpiMin", e23.top_quartile_agreement.min()); M.add("QuartCcpiMax", e23.top_quartile_agreement.max())
    M.add("QuartCcpi", e23.loc["IFWG-DIFSS (proposed)", "top_quartile_agreement"])
    ug = rank_df.set_index("ISO3").loc["UGA"]
    M.add("UgaTopsis", int(ug["Classical TOPSIS (crisp)"]), "{:d}"); M.add("UgaWsum", int(ug["Weighted sum (crisp)"]), "{:d}")

    # ---------------- SMAA ----------------
    sm, acc = S.smaa(n_smaa); sm.to_csv(TAB / "smaa_rank_acceptability.csv", index=False)
    np.save(TAB / "smaa_acceptability_matrix.npy", acc)
    rows = []
    for r in sm.head(12).itertuples():
        rows.append([tex_escape(r.Country), str(r.baseline_rank), f3(r.b1), f3(r.top10_acceptability), f"{r.expected_rank:.2f}",
                     f3(r.central_w_policy), f3(r.central_w_share), f3(r.central_w_co2)])
    write_table(TEX / "tab_smaa.tex", ["Country", "Rank", r"$b^1$", r"$\sum_{r\le10}b^r$", r"$E[r]$",
                                        r"$\omega^c_1$", r"$\omega^c_2$", r"$\omega^c_3$"], rows, "lrcccccc")
    M.add("NSmaa", f"{n_smaa:,}".replace(",", "{,}"), "{}")
    lead = sm.sort_values("b1", ascending=False).iloc[0]
    M.add("SmaaLeader", lead.Country, "{}"); M.add("SmaaLeaderBone", lead.b1)
    M.add("SmaaNFirst", int((sm.b1 > 0).sum()), "{:d}")
    M.add("SmaaNTopNine", int((sm.top10_acceptability >= 0.5).sum()), "{:d}")
    base_top = set(prim.head(10).ISO3)
    M.add("SmaaTopMin", sm[sm.ISO3.isin(base_top)].top10_acceptability.min())
    M.add("SmaaTopMax", sm[sm.ISO3.isin(base_top)].top10_acceptability.max())
    top_rows = sm[sm.ISO3.isin(base_top)]
    M.add("SmaaNRobust", int((top_rows.top10_acceptability >= 0.75).sum()), "{:d}")
    M.add("SmaaRobustMin", top_rows[top_rows.top10_acceptability >= 0.75].top10_acceptability.min())
    rob = prim[prim.ISO3.isin(top_rows[top_rows.top10_acceptability >= 0.75].ISO3)].sort_values("Rank")
    names_rob = list(rob.Country)
    M.add("SmaaRobustNames", (", ".join(names_rob[:-1]) + " and " + names_rob[-1]) if len(names_rob) > 1 else names_rob[0], "{}")
    report.fig_top20(prim, sm, FIG / "fig_top20_smaa")

    # ---------------- rank reversal ----------------
    rr_rows = []
    for label, c in [("Goalpost normalisation (primary)", cfg),
                     ("Per-period min-max", Config(**{**cfg.__dict__, "reference": False})),
                     ("Pooled min-max", Config(**{**cfg.__dict__, "reference": False, "pooled": True}))]:
        loo, dom = S.rank_reversal(c)
        loo.to_csv(TAB / f"rank_reversal_loo_{label.split()[0].lower()}.csv", index=False)
        rr_rows.append({"normalisation": label, "loo_mean_pair_reversal": loo.pair_reversal_share.mean(),
                        "loo_max_pair_reversal": loo.pair_reversal_share.max(),
                        "loo_max_rank_shift": loo.max_abs_rank_shift.max(),
                        "loo_runs_top10_order_changed": int(loo.top10_order_changed.sum()),
                        "dominated_pair_reversal": dom["pair_reversal_share"],
                        "dominated_max_rank_shift": dom["max_abs_rank_shift"],
                        "dominated_rank": dom["dominated_rank"],
                        "dominated_pair_reversal_fixed_weights": dom["pair_reversal_share_fixed_weights"],
                        "dominated_max_rank_shift_fixed_weights": dom["max_abs_rank_shift_fixed_weights"]})
    rr = pd.DataFrame(rr_rows); rr.to_csv(TAB / "rank_reversal_summary.csv", index=False)
    write_table(TEX / "tab_rank_reversal.tex",
                ["Normalisation", r"LOO mean (\%)", r"LOO max (\%)", "LOO max shift", r"Dominated (\%)", "Dominated max shift"],
                [[tex_escape(r.normalisation.replace("Goalpost normalisation", "Goalpost")), f"{100*r.loo_mean_pair_reversal:.2f}", f"{100*r.loo_max_pair_reversal:.2f}",
                  str(r.loo_max_rank_shift), f"{100*r.dominated_pair_reversal:.2f}", str(r.dominated_max_rank_shift)]
                 for r in rr.itertuples()], "lccccc")
    M.add("RrLooMean", 100 * rr.iloc[0].loo_mean_pair_reversal, "{:.2f}")
    M.add("RrLooShift", int(rr.iloc[0].loo_max_rank_shift), "{:d}")
    M.add("RrLooTop", int(rr.iloc[0].loo_runs_top10_order_changed), "{:d}")
    M.add("RrDomGoal", 100 * rr.iloc[0].dominated_pair_reversal, "{:.2f}")
    M.add("RrDomGoalShift", int(rr.iloc[0].dominated_max_rank_shift), "{:d}")
    M.add("RrDomGoalFixed", 100 * rr.iloc[0].dominated_pair_reversal_fixed_weights, "{:.2f}")
    M.add("RrDomGoalFixedShift", int(rr.iloc[0].dominated_max_rank_shift_fixed_weights), "{:d}")
    M.add("RrDomMinmax", 100 * rr.iloc[1].dominated_pair_reversal, "{:.2f}")
    M.add("RrLooMeanMinmax", 100 * rr.iloc[1].loo_mean_pair_reversal, "{:.2f}")
    M.add("RrLooShiftMinmax", int(rr.iloc[1].loo_max_rank_shift), "{:d}")
    M.add("RrLooTopMinmax", int(rr.iloc[1].loo_runs_top10_order_changed), "{:d}")
    M.add("RrDomMinmaxShift", int(rr.iloc[1].dominated_max_rank_shift), "{:d}")

    # ---------------- perturbation and bounds ----------------
    pr, mg = S.perturbation(reps=reps_pert)
    pr.to_csv(TAB / "perturbation.csv", index=False); mg.to_csv(TAB / "entropy_margins.csv", index=False)
    def bnd(x):  # an l1 distance between weight vectors never exceeds 2
        return r"$\infty$" if not np.isfinite(x) else (r"$\geq2$" if x >= 2 else f"{x:.3f}")
    write_table(TEX / "tab_perturbation.tex",
                [r"$\delta$", r"Mean $\tau$ (sd)", r"Min.\ $\tau$", r"Top-10 (\%)",
                 r"$\max\|\Delta w^{(1)}\|_1$", r"$\max\|\Delta w^{(2)}\|_1$", r"$\max\|\Delta w^{(3)}\|_1$",
                 r"$\max|\Delta S|$"],
                [[f"{r.delta:.2f}", f"{r.mean_tau:.3f} ({r.sd_tau:.3f})", f"{r.min_tau:.3f}", f"{r.mean_top10:.1f}",
                  f"{r.max_dw_policy:.3f} [{bnd(r.bound_dw_policy)}]", f"{r.max_dw_share:.3f} [{bnd(r.bound_dw_share)}]",
                  f"{r.max_dw_co2:.3f} [{bnd(r.bound_dw_co2)}]", f"{r.max_dS_weight_channel:.4f} [{r.max_dS_bound:.3f}]"]
                 for r in pr.itertuples()], "cccccccc")
    write_table(TEX / "tab_margins.tex",
                ["Criterion", r"$p_j$", r"$D_j$ (dispersion)", r"$\Lambda/\delta$ at $\delta{=}0.005$", r"$\varepsilon^*_j$ (dispersion)", r"$D_j$ (Ye)",
                 r"$\varepsilon^*_j$ (Ye)"],
                [[{"policy": "Policy (RISE)", "share": "Renewable share", "co2": r"CO$_2$ per capita"}[r.criterion],
                  str(r.p), f"{r.D_dispersion:.4f}", f"{getattr(r, '_4'):.3f}", f"{r.eps_star_dispersion:.4f}", f"{r.D_ye:.3f}", f"{r.eps_star_ye:.3f}"]
                 for r in mg.itertuples()], "lcccccc")
    last = pr.iloc[-1]
    M.add("PertTauLast", last.mean_tau); M.add("PertTauMinLast", last.min_tau)
    M.add("PertTopLast", last.mean_top10, "{:.1f}"); M.add("PertTauFirst", pr.iloc[0].mean_tau)
    M.add("PertReps", f"{reps_pert:,}".replace(",", "{,}"), "{}")
    M.add("PertDwMax", max(pr[["max_dw_policy", "max_dw_share", "max_dw_co2"]].max()))
    M.add("PertDsMax", last.max_dS_weight_channel, "{:.4f}"); M.add("PertDsBound", last.max_dS_bound)
    M.add("GapsCertFirst", int(pr.iloc[0].gaps_certified_observed), "{:d}")
    M.add("GapsCertLast", int(pr.iloc[-1].gaps_certified_observed), "{:d}")
    M.add("GapsCertBoundFirst", int(pr.iloc[0].gaps_certified_bound), "{:d}")
    M.add("NGaps", m - 1, "{:d}")
    M.add("EpsStarPolicy", mg.iloc[0].eps_star_dispersion, "{:.4f}"); M.add("EpsStarShare", mg.iloc[1].eps_star_dispersion, "{:.4f}")
    M.add("EpsStarCo", mg.iloc[2].eps_star_dispersion, "{:.4f}")
    M.add("EpsStarYeMin", mg.eps_star_ye.min()); M.add("EpsStarYeMax", mg.eps_star_ye.max())
    M.add("YeL", YE_L)
    M.add("BoundHolds", "all" if pr.bound_holds_score.all() else "not all", "{}")
    report.fig_perturbation(pr, FIG / "fig_perturbation")

    gaps = S.adjacent_gaps()
    M.add("GapMin", gaps.min(), "{:.5f}"); M.add("GapMedian", float(np.median(gaps)), "{:.4f}")
    M.add("GapTopTen", gaps[:9].min(), "{:.4f}")

    # ---------------- threshold (random subsets) ----------------
    th, thp = S.threshold_subsets(); th.to_csv(TAB / "threshold_annual_subsets.csv", index=False)
    thp.to_csv(TAB / "threshold_policy_subsets.csv", index=False)
    rows = []
    for p in sorted(th.p_annual.unique()):
        r = [str(p), str(int(th[th.p_annual == p].n_subsets.iloc[0]))]
        for t in (2, 3, 4):
            q = th[(th.p_annual == p) & (th.theta == t)].iloc[0]
            r.append(("\\textit{" if q.fallback_active else "") + f"{q.mean_tau:.3f}" + ("}" if q.fallback_active else ""))
        rows.append(r)
    write_table(TEX / "tab_threshold.tex", [r"$p$ (annual)", "Subsets", r"$\theta=2$", r"$\theta=3$", r"$\theta=4$"], rows, "ccccc")
    rows = []
    for p in sorted(thp.p_policy.unique()):
        r = [str(p), str(int(thp[thp.p_policy == p].n_subsets.iloc[0]))]
        for t in (2, 3, 4):
            q = thp[(thp.p_policy == p) & (thp.theta == t)].iloc[0]
            r.append(("\\textit{" if q.fallback_active else "") + f"{q.mean_tau:.3f}" + ("}" if q.fallback_active else ""))
        rows.append(r)
    write_table(TEX / "tab_threshold_policy.tex", [r"$p_1$ (RISE)", "Subsets", r"$\theta=2$", r"$\theta=3$", r"$\theta=4$"], rows, "ccccc")
    M.add("ThTauTwo", th[(th.p_annual == 2) & (th.theta == 3)].mean_tau.iloc[0])
    M.add("ThTauSix", th[(th.p_annual == 6) & (th.theta == 3)].mean_tau.iloc[0])
    M.add("ThPMax", int(th.p_annual.max()), "{:d}")
    M.add("ThTauMax", th[(th.p_annual == th.p_annual.max()) & (th.theta == 3)].mean_tau.iloc[0])
    M.add("ThPolOne", thp[(thp.p_policy == 1) & (thp.theta == 3)].mean_tau.iloc[0])

    # ---------------- simulation ----------------
    sim = simulation.run_simulation(cfg, reps=reps_sim, seed=MASTER_SEED); sim.to_csv(TAB / "simulation_grid.csv", index=False)
    summ = sim.groupby(["p_sparse", "sigma", "method"])[["tau", "top10"]].mean().reset_index()
    summ.to_csv(TAB / "simulation_summary.csv", index=False)
    order = ["proposed", "proposed_uniform", "interpolate", "align", "omit"]
    labels = {"proposed": "Proposed ($T_j$, entropy)", "proposed_uniform": "Proposed ($T_j$, uniform)",
              "interpolate": "Interpolation", "align": "Common-period alignment", "omit": "Omit sparse criteria"}
    rows = []
    for mth in order:
        r = [labels[mth]]
        for sg in (0.05, 0.15):
            for ps in simulation.GRID_PS:
                q = summ[(summ.method == mth) & (summ.sigma == sg) & (summ.p_sparse == ps)].iloc[0]
                r.append(f"{q.tau:.3f}")
        rows.append(r)
    write_table(TEX / "tab_simulation.tex",
                ["Strategy"] + [rf"$p_s{{=}}{ps}$" for ps in simulation.GRID_PS] * 2, rows, "lcccccc")
    ov = sim.groupby("method").tau.mean()
    for mth, tag in [("proposed", "Prop"), ("proposed_uniform", "PropU"), ("interpolate", "Interp"), ("align", "Align"), ("omit", "Omit")]:
        M.add(f"Sim{tag}", ov[mth])
    miss = sim[sim.missing > 0].groupby("method").tau.mean()
    M.add("SimPropMiss", miss["proposed"]); M.add("SimInterpMiss", miss["interpolate"])
    wins = sim.pivot_table(index=["m", "n_annual", "n_sparse", "p_sparse", "sigma", "missing"], columns="method", values="tau")
    M.add("SimNCells", len(wins), "{:d}")
    M.add("SimBeatOmit", int((wins["proposed"] > wins["omit"]).sum()), "{:d}")
    M.add("SimBeatInterp", int((wins["proposed"] > wins["interpolate"]).sum()), "{:d}")
    M.add("SimBeatAlign", int((wins["proposed"] > wins["align"]).sum()), "{:d}")
    M.add("SimReps", reps_sim, "{:d}")
    report.fig_simulation(sim, FIG / "fig_simulation")

    # step-change DGP: sparse (policy-type) criteria move through discrete reforms
    sst = simulation.run_simulation(cfg, reps=reps_sim, seed=MASTER_SEED, dgp="step")
    sst.to_csv(TAB / "simulation_grid_step.csv", index=False)
    sums = sst.groupby(["p_sparse", "sigma", "method"])[["tau", "top10"]].mean().reset_index()
    sums.to_csv(TAB / "simulation_summary_step.csv", index=False)
    rows = []
    for mth in order:
        r = [labels[mth]]
        for sg in (0.05, 0.15):
            for ps in simulation.GRID_PS:
                q = sums[(sums.method == mth) & (sums.sigma == sg) & (sums.p_sparse == ps)].iloc[0]
                r.append(f"{q.tau:.3f}")
        rows.append(r)
    write_table(TEX / "tab_simulation_step.tex",
                ["Strategy"] + [rf"$p_s{{=}}{ps}$" for ps in simulation.GRID_PS] * 2, rows, "lcccccc")
    ovs = sst.groupby("method").tau.mean()
    for mth, tag in [("proposed", "Prop"), ("proposed_uniform", "PropU"), ("interpolate", "Interp"), ("align", "Align"), ("omit", "Omit")]:
        M.add(f"SimStep{tag}", ovs[mth])
    ws = sst.pivot_table(index=["m", "n_annual", "n_sparse", "p_sparse", "sigma", "missing"], columns="method", values="tau")
    M.add("SimStepBeatInterp", int((ws["proposed"] > ws["interpolate"]).sum()), "{:d}")
    M.add("SimStepBeatAlign", int((ws["proposed"] > ws["align"]).sum()), "{:d}")
    M.add("SimStepBeatOmit", int((ws["proposed"] > ws["omit"]).sum()), "{:d}")
    dstep = (ws["proposed"] - ws["interpolate"]).groupby(level="sigma").mean()
    dlin = (wins["proposed"] - wins["interpolate"]).groupby(level="sigma").mean()
    M.add("SimStepGainLow", dstep[0.05], "{:.4f}"); M.add("SimStepGainHigh", dstep[0.15], "{:.4f}")
    M.add("SimLinGainLow", dlin[0.05], "{:.4f}"); M.add("SimLinGainHigh", dlin[0.15], "{:.4f}")
    M.add("SimStepGainMax", (ws["proposed"] - ws["interpolate"]).max(), "{:.3f}")
    M.add("StepProb", simulation.STEP_PROB, "{:.1f}")
    M.add("StepLow", simulation.STEP_LOW, "{:.2f}"); M.add("StepHigh", simulation.STEP_HIGH, "{:.2f}")

    ths = simulation.run_threshold_simulation(cfg, reps=reps_th, seed=MASTER_SEED)
    ths.to_csv(TAB / "threshold_simulation.csv", index=False)
    rows = []
    for ps in (2, 3, 4):
        for sg in (0.05, 0.15):
            r = [str(ps), f"{sg:.2f}"]
            for t in (1, 2, 3, 4):
                q = ths[(ths.p_sparse == ps) & (ths.sigma == sg) & (ths.theta == t)].iloc[0]
                r.append(("\\textit{" if not q.entropy_active_for_sparse else "") + f"{q.tau:.3f}" +
                         ("}" if not q.entropy_active_for_sparse else ""))
            rows.append(r)
    write_table(TEX / "tab_threshold_sim.tex", [r"$p_s$", r"$\sigma$", r"$\theta=1$", r"$\theta=2$", r"$\theta=3$", r"$\theta=4$"],
                rows, "cccccc")
    diff = []
    for ps in (2, 3, 4):
        for sg in (0.05, 0.15):
            g = ths[(ths.p_sparse == ps) & (ths.sigma == sg)].set_index("theta")
            diff.append(abs(g.tau.max() - g.tau.min()))
    M.add("ThSimMaxDiff", max(diff), "{:.4f}"); M.add("ThSimReps", reps_th, "{:d}")

    # ---------------- figures and remaining macros ----------------
    report.fig_temporal_weights(wt, FIG / "fig_temporal_weights")
    M.add("MasterSeed", MASTER_SEED, "{:d}")
    M.write(TEX / "numbers.tex")

    env = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
           "runtime_seconds": round(time.time() - t0, 1), "quick": quick, "master_seed": MASTER_SEED}
    (ROOT / "outputs" / "run_info.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(f"Done in {time.time() - t0:.1f}s. Outputs in {ROOT / 'outputs'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-verify", action="store_true", help="do not stop on raw-data checksum mismatches")
    a = ap.parse_args()
    main(a.quick, verify=not a.no_verify)
