"""Data loading, country matching and panel construction.

All raw files live in data/raw/ (see data/raw/SOURCES.md for provenance).
Countries are matched on ISO3 codes; the only name-based matching is for the
legacy RISE 2016 workbook (names only) and the CCPI ranking table, both
resolved through an explicit crosswalk written to data/crosswalk/.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
XWALK = ROOT / "data" / "crosswalk"

ANNUAL_YEARS = list(range(2016, 2025))          # T_2 = T_3 (latest year with near-complete coverage)
RISE_YEARS = [2015, 2017, 2019, 2021, 2023]     # T_1: RISE reference years (editions 2016-2024)
RISE_ANNUAL_YEARS = [y for y in ANNUAL_YEARS if y <= RISE_YEARS[-1]]   # Data360 annual series (sensitivity)
LEGACY_SHARE_YEARS = [y for y in ANNUAL_YEARS if y <= 2022]   # IMF generation series ends in 2022 (sensitivity)
def _span(years):
    return f"{years[0]}-{years[-1]}"

RISE_RE_SUB = ["WB_RISE_RE_ELEC", "WB_RISE_RE_GOV", "WB_RISE_RE_HE_CO",
               "WB_RISE_RE_LVL_PLYNG_FLD", "WB_RISE_RE_TRNS"]
RISE_RE_PILLAR = "WB_RISE_RE_ALL"

# Legacy RISE 2016 workbook: the seven renewable-energy pillar indicators.
RISE2016_RE_COLS = [
    "Indicator 1: Legal framework for renewable energy",
    "Indicator 2: Planning for renewable energy expansion",
    "Indicator 3: Incentives and regulatory support for renewable energy",
    "Indicator 4: Attributes of financial and regulatory incentives",
    "Indicator 5: Network connection and pricing",
    "Indicator 6: Counterparty risk",
    "indicator 7: Carbon Pricing and Monitoring Mechanism",
]


def clean_name(value) -> str:
    if pd.isna(value):
        return ""
    v = str(value).lower().strip().replace("&", "and")
    v = re.sub(r"[’'`´]", "", v)
    v = re.sub(r"[^a-z0-9]+", " ", v)
    return re.sub(r"\s+", " ", v).strip()


# Manual aliases (cleaned name -> ISO3) for names that do not match the
# World Bank country names after cleaning. Every entry is documented in the
# crosswalk output.
NAME_ALIASES = {
    "congo dem rep": "COD", "democratic republic of the congo": "COD", "dr congo": "COD",
    "congo rep": "COG", "republic of the congo": "COG",
    "cote divoire": "CIV", "cote d ivoire": "CIV", "ivory coast": "CIV",
    "egypt": "EGY", "iran": "IRN", "russia": "RUS", "south korea": "KOR", "korea": "KOR",
    "korea rep": "KOR", "vietnam": "VNM", "viet nam": "VNM", "turkey": "TUR", "turkiye": "TUR",
    "kyrgyzstan": "KGZ", "kyrgyz republic": "KGZ", "lao pdr": "LAO", "laos": "LAO",
    "slovakia": "SVK", "slovak republic": "SVK", "czech republic": "CZE", "czechia": "CZE",
    "yemen": "YEM", "venezuela": "VEN", "gambia": "GMB", "the gambia": "GMB",
    "bahamas": "BHS", "united states": "USA", "usa": "USA", "united kingdom": "GBR",
    "uk": "GBR", "taiwan": "TWN", "chinese taipei": "TWN", "european union": "EU",
    "eu": "EU", "micronesia": "FSM", "moldova": "MDA", "tanzania": "TZA",
    "west bank and gaza": "PSE", "palestine": "PSE", "bolivia": "BOL",
    "macedonia fyr": "MKD", "north macedonia": "MKD", "swaziland": "SWZ", "eswatini": "SWZ",
    "cabo verde": "CPV", "cape verde": "CPV", "brunei": "BRN", "syria": "SYR",
    "st lucia": "LCA", "st vincent and the grenadines": "VCT", "st kitts and nevis": "KNA",
}


def load_co2() -> pd.DataFrame:
    """World Bank EN.GHG.CO2.PC.CE.AR5 (t CO2e per capita, excl. LULUCF)."""
    raw = pd.read_excel(RAW / "API_EN.GHG.CO2.PC.CE.AR5_DS2_en_excel_v2_33405.xls",
                        sheet_name="Data", header=3)
    meta = pd.read_excel(RAW / "API_EN.GHG.CO2.PC.CE.AR5_DS2_en_excel_v2_33405.xls",
                         sheet_name="Metadata - Countries")
    countries = set(meta.loc[meta["Region"].notna(), "Country Code"])   # drops aggregates
    raw = raw[raw["Country Code"].isin(countries)]
    out = raw.set_index("Country Code")[[str(y) for y in ANNUAL_YEARS]]
    out.columns = ANNUAL_YEARS
    names = raw.set_index("Country Code")["Country Name"]
    return out.astype(float), names


def load_renewable_share() -> tuple[pd.DataFrame, pd.Series]:
    """Share of electricity generation from renewables (%), Ember via OWID.

    Denominator = total generation (fossil + nuclear + renewables); renewables
    include hydropower, wind, solar, bioenergy and other renewables (geothermal,
    marine), as defined by Ember/OWID; pumped-storage conventions follow Ember.
    """
    d = pd.read_csv(RAW / "owid-energy-data.csv", low_memory=False)
    d = d[d["iso_code"].notna() & ~d["iso_code"].astype(str).str.startswith("OWID")]
    piv = d.pivot(index="iso_code", columns="year", values="renewables_share_elec")
    names = d.drop_duplicates("iso_code").set_index("iso_code")["country"]
    return piv[ANNUAL_YEARS].astype(float), names


def load_nonhydro_share() -> pd.DataFrame:
    """Sensitivity only: non-hydro renewable share of electricity (%), Ember via OWID,
    renewables_share_elec - hydro_share_elec (same total-generation denominator)."""
    d = pd.read_csv(RAW / "owid-energy-data.csv", low_memory=False)
    d = d[d["iso_code"].notna() & ~d["iso_code"].astype(str).str.startswith("OWID")]
    re_ = d.pivot(index="iso_code", columns="year", values="renewables_share_elec")
    hy = d.pivot(index="iso_code", columns="year", values="hydro_share_elec")
    return (re_[ANNUAL_YEARS] - hy[ANNUAL_YEARS]).clip(lower=0).astype(float)


def load_renewable_share_legacy() -> pd.DataFrame:
    """Alternative definition (sensitivity analysis only): RE / (RE + fossil), from the
    IMF Climate Change Indicators Dashboard file (IRENA-based). Nuclear and
    geothermal are absent from this file. Used only as a sensitivity check."""
    df = pd.read_csv(RAW / "Renewable_Energy.csv")
    g = df[df["Indicator"].astype(str).str.strip().eq("Electricity Generation")]
    cols = [f"F{y}" for y in LEGACY_SHARE_YEARS]
    re_ = g[g["Energy_Type"].eq("Total Renewable")].groupby("ISO3")[cols].sum(min_count=1)
    fo = g[g["Energy_Type"].eq("Total Non-Renewable")].groupby("ISO3")[cols].sum(min_count=1)
    idx = re_.index.intersection(fo.index)
    share = 100 * re_.loc[idx] / (re_.loc[idx] + fo.loc[idx])
    share.columns = LEGACY_SHARE_YEARS
    return share[share.index.str.fullmatch(r"[A-Z]{3}")]


def load_rise() -> tuple[pd.DataFrame, dict, pd.Series]:
    """RISE renewable-energy pillar score (0-100) from World Bank Data360.

    Returns the pillar score at RISE_YEARS and, for hesitancy, the five
    renewable-energy sub-indicator scores at the same years.
    """
    d = pd.read_csv(RAW / "WB_RISE.csv", low_memory=False)
    names = d.drop_duplicates("REF_AREA").set_index("REF_AREA")["REF_AREA_LABEL"]
    pil = d[d["INDICATOR"] == RISE_RE_PILLAR].pivot(index="REF_AREA", columns="TIME_PERIOD",
                                                    values="OBS_VALUE")
    subs = {}
    for s in RISE_RE_SUB:
        subs[s] = d[d["INDICATOR"] == s].pivot(index="REF_AREA", columns="TIME_PERIOD",
                                                values="OBS_VALUE")
    return pil.astype(float), subs, names


def load_rise2016_legacy(name_to_iso: dict) -> pd.DataFrame:
    """Legacy RISE 2016 workbook (2015 regulatory information): the
    renewable-energy pillar score and its seven indicators, keyed by ISO3."""
    df = pd.read_excel(RAW / "rise-2016-database-for-download-1.xlsx", header=1)
    df = df[df["Country"].notna()].copy()
    df["ISO3"] = df["Country"].map(lambda n: name_to_iso.get(clean_name(n)))
    keep = ["ISO3", "Country", "Renewable Energy score"] + RISE2016_RE_COLS
    return df.loc[df["ISO3"].notna(), keep].set_index("ISO3")


def build_name_index(*series: pd.Series) -> dict:
    """cleaned country name -> ISO3 from several (ISO3 -> name) series."""
    idx = {}
    for s in series:
        for iso, nm in s.items():
            if isinstance(iso, str) and len(iso) == 3:
                idx.setdefault(clean_name(nm), iso)
    for k, v in NAME_ALIASES.items():
        idx.setdefault(k, v)
    return idx


def build_panel(verbose: bool = True) -> dict:
    """Construct the complete-case analytical panel and all side inputs."""
    PROC.mkdir(parents=True, exist_ok=True)
    XWALK.mkdir(parents=True, exist_ok=True)

    co2, co2_names = load_co2()
    share, owid_names = load_renewable_share()
    rise, rise_subs, rise_names = load_rise()
    name_idx = build_name_index(co2_names, owid_names, rise_names)

    ok_rise = set(rise[RISE_YEARS].dropna().index)
    ok_share = set(share.dropna().index)
    ok_co2 = set(co2.dropna().index)
    panel = sorted(ok_rise & ok_share & ok_co2)

    coverage = pd.DataFrame([
        ("RISE renewable-energy pillar observed in " + ", ".join(map(str, RISE_YEARS[:-1])) + f" and {RISE_YEARS[-1]}", len(ok_rise)),
        (f"Renewable electricity share complete {_span(ANNUAL_YEARS)} (Ember/OWID)", len(ok_share)),
        (f"CO2 per capita complete {_span(ANNUAL_YEARS)} (World Bank)", len(ok_co2)),
        ("RISE and renewable share", len(ok_rise & ok_share)),
        ("RISE and CO2", len(ok_rise & ok_co2)),
        ("Complete-case analytical panel (all three sources)", len(panel)),
    ], columns=["stage", "countries"])
    excluded = sorted(ok_rise - set(panel))

    # Sub-indicator arrays for policy hesitancy
    sub = np.stack([rise_subs[s].reindex(panel)[RISE_YEARS].to_numpy(float) for s in RISE_RE_SUB],
                   axis=2)  # m x p1 x 5

    # Legacy inputs (sensitivity analyses only)
    legacy_share = load_renewable_share_legacy().reindex(panel)
    nonhydro = load_nonhydro_share().reindex(panel)
    rise2016 = load_rise2016_legacy(name_idx).reindex(panel)

    # CCPI 2023 (external validation)
    ccpi = pd.read_csv(RAW / "ccpi_2023_2024_ranks.csv")
    ccpi["ISO3"] = ccpi["country"].map(lambda n: name_idx.get(clean_name(n)))

    names = pd.Series({iso: co2_names.get(iso, owid_names.get(iso)) for iso in panel})
    xw = pd.DataFrame({
        "ISO3": panel,
        "name_worldbank_co2": [co2_names.get(i) for i in panel],
        "name_owid_ember": [owid_names.get(i) for i in panel],
        "name_rise_data360": [rise_names.get(i) for i in panel],
        "name_rise2016_file": [rise2016["Country"].get(i) for i in panel],
    })
    xw.to_csv(XWALK / "country_crosswalk.csv", index=False)
    ccpi[["edition", "rank", "country", "ISO3"]].to_csv(XWALK / "ccpi_crosswalk.csv", index=False)
    coverage.to_csv(PROC / "panel_coverage.csv", index=False)
    pd.DataFrame({"ISO3": excluded, "name": [rise_names.get(i) for i in excluded]}).to_csv(
        PROC / "excluded_rise_countries.csv", index=False)

    data = {
        "iso": panel,
        "names": names,
        "rise": rise.reindex(panel)[RISE_YEARS].to_numpy(float),
        "rise_annual": rise.reindex(panel)[RISE_ANNUAL_YEARS].to_numpy(float),
        "rise_sub_annual": np.stack([rise_subs[s].reindex(panel)[RISE_ANNUAL_YEARS].to_numpy(float)
                                     for s in RISE_RE_SUB], axis=2),
        "rise_sub": sub,
        "share": share.reindex(panel)[ANNUAL_YEARS].to_numpy(float),
        "co2": co2.reindex(panel)[ANNUAL_YEARS].to_numpy(float),
        "share_legacy": legacy_share[LEGACY_SHARE_YEARS].to_numpy(float),
        "share_nonhydro": nonhydro[ANNUAL_YEARS].to_numpy(float),
        "rise2016": rise2016["Renewable Energy score"].to_numpy(float),
        "rise2016_ind": rise2016[RISE2016_RE_COLS].to_numpy(float),
        "ccpi": ccpi,
        "coverage": coverage,
    }
    # Save processed panel (long + wide) for inspection
    wide = pd.DataFrame({"ISO3": panel, "Country": names.values})
    for k, y in enumerate(RISE_YEARS):
        wide[f"RISE_RE_{y}"] = data["rise"][:, k]
    for k, y in enumerate(ANNUAL_YEARS):
        wide[f"REshare_{y}"] = data["share"][:, k]
    for k, y in enumerate(ANNUAL_YEARS):
        wide[f"CO2pc_{y}"] = data["co2"][:, k]
    wide.to_csv(PROC / "analytical_panel.csv", index=False)
    if verbose:
        print(coverage.to_string(index=False))
    return data
