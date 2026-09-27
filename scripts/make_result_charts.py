"""Result charts: data overview, entropy weights, robustness and benchmarks (600-dpi PNG).

Inputs (written by run_all.py): outputs/tables/full_ranking.csv, panel_coverage.csv,
  entropy_options.csv, robustness.csv, benchmarks.csv, external_validation_ccpi.csv
Icons: figures_tikz/icons_png/*.png (rendered from figures_tikz/icons.tex by
  scripts/build_tikz_figures.py)
Outputs: outputs/figures/fig_data_overview.png, fig_entropy_weights.png,
         fig_robustness.png, fig_benchmarks.png
Run from the bundle root after run_all.py:  python scripts/make_result_charts.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from difss import figstyle as fs                                    # noqa: E402
from difss.data import ANNUAL_YEARS, RISE_YEARS                    # noqa: E402
from difss.figstyle import C_POL, C_SHR, C_CO2, C_MAIN, C_GREY, C_AMBER, SMALL_PT   # noqa: E402
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.offsetbox import OffsetImage, AnnotationBbox       # noqa: E402

DATA, FIG = os.path.join(ROOT, "outputs", "tables"), os.path.join(ROOT, "outputs", "figures")
ICON = os.path.join(ROOT, "figures_tikz", "icons_png")


def add_icon(ax, name, xy, zoom):
    img = plt.imread(os.path.join(ICON, name + ".png"))
    ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=zoom), xy, xycoords="axes fraction", frameon=False))


# =========================================================== data overview (Figure 4)
df = pd.read_csv(os.path.join(DATA, "full_ranking.csv"))
cov = pd.read_csv(os.path.join(DATA, "panel_coverage.csv"))
fig = plt.figure(figsize=(fs.TEXTWIDTH_IN, 4.3), layout="none")
gs = fig.add_gridspec(1, 3, left=0.075, right=0.99, top=0.95, bottom=0.60, wspace=0.32)
gs2 = fig.add_gridspec(1, 1, left=0.29, right=0.95, top=0.43, bottom=0.09)
RY, AY = RISE_YEARS[-1], ANNUAL_YEARS[-1]
spec = [("RISE_last", f"(a) Policy (RISE), {RY}", "Pillar score (0–100)", C_POL, "survey", np.arange(0, 101, 10), "{:.1f}"),
        ("REshare_last", f"(b) Renewable share, {AY}", "Share of generation (%)", C_SHR, "turbine", np.arange(0, 101, 10), "{:.1f}"),
        ("CO2pc_last", rf"(c) CO$_2$ per capita, {AY}", "t per capita (log scale)", C_CO2, "factory",
         np.logspace(np.log10(0.03), np.log10(60), 14), "{:.2f}")]
for k, (col, title, xl, c, ic, bins, fmt) in enumerate(spec):
    ax = fig.add_subplot(gs[0, k])
    ax.hist(df[col], bins=bins, color=c, alpha=0.85, edgecolor="white", linewidth=0.5)
    med = df[col].median()
    ax.axvline(med, color="0.2", lw=0.8, ls="--")
    ax.text(0.03, 0.97, "Median " + fmt.format(med), transform=ax.transAxes, va="top", fontsize=SMALL_PT,
            color="0.2", zorder=5, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
    if col == "CO2pc_last":
        ax.set_xscale("log")
    ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel("Countries" if k == 0 else "")
    add_icon(ax, ic, xy=(0.15, 0.58) if ic == "factory" else (0.84, 0.78), zoom=0.085)
ax = fig.add_subplot(gs2[0, 0])
_n = {4: "four", 5: "five", 6: "six"}.get(len(RISE_YEARS), str(len(RISE_YEARS)))
labels = [f"RISE, {_n} reference years {RISE_YEARS[0]}–{RISE_YEARS[-1]}", f"Renewable share {ANNUAL_YEARS[0]}–{ANNUAL_YEARS[-1]}",
          rf"CO$_2$ per capita {ANNUAL_YEARS[0]}–{ANNUAL_YEARS[-1]}",
          "RISE and renewable share", r"RISE and CO$_2$", "Complete-case panel"]
cols = [C_POL, C_SHR, C_CO2, "#7A8CA8", "#7A8CA8", C_MAIN]
y = np.arange(len(cov))[::-1]
ax.barh(y, cov.countries, color=cols, height=0.62)
for yi, v in zip(y, cov.countries):
    ax.text(v + 2, yi, str(v), va="center", fontsize=SMALL_PT)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlim(0, 235); ax.set_xlabel("Countries with complete data")
ax.set_title("(d) Construction of the analytical panel")
fs.save(fig, os.path.join(FIG, "fig_data_overview.png"))

# =========================================================== entropy weights (Figure 7)
eo = pd.read_csv(os.path.join(DATA, "entropy_options.csv"))
order = ["dispersion", "shannon", "sk", "ye"]
names = {"dispersion": "Dispersion entropy\n(proposed)", "shannon": "Crisp Shannon entropy\n(entropy-weight method)",
         "sk": "Szmidt–Kacprzyk\nratio entropy", "ye": "Ye sine\nentropy"}
eo = eo.set_index("key").loc[order]
fig, ax = fs.figure(3.0)
x = np.arange(len(order)); w = 0.25
for k, (c, lab, colr) in enumerate([("w_policy", "Policy (RISE)", C_POL), ("w_share", "Renewable share", C_SHR),
                                     ("w_co2", r"CO$_2$ per capita", C_CO2)]):
    ax.bar(x + (k - 1) * w, eo[c], w, label=lab, color=colr)
    for xi, v in zip(x + (k - 1) * w, eo[c]):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=SMALL_PT)
ax.set_xticks(x)
ax.set_xticklabels([names[k] + "\n" + rf"$\tau={tv:.3f}$" for k, tv in zip(order, eo.tau)])
ax.set_ylabel(r"Criterion weight $\omega_j$"); ax.set_ylim(0, 0.68)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
ax.axvspan(-0.5, 1.5, color="#EEF4F1", zorder=0); ax.axvspan(1.5, 3.5, color="#FBEFEC", zorder=0)
ax.text(0.5, 0.635, "Discrimination-based weights", ha="center", fontsize=SMALL_PT, color="#2E7D4F", style="italic")
ax.text(2.5, 0.635, "Fuzziness-based weights", ha="center", fontsize=SMALL_PT, color=C_MAIN, style="italic")
ax.set_xlim(-0.5, 3.5)
fs.save(fig, os.path.join(FIG, "fig_entropy_weights.png"))

# =========================================================== robustness (Figure 8)
rb = pd.read_csv(os.path.join(DATA, "robustness.csv"))


def group(s):
    if s.startswith(("Temporal", "Staleness")):
        return "Temporal weighting"
    if "normalisation" in s:
        return "Normalisation"
    if s.startswith(("Membership", "Hesitancy", "Criterion weights")):
        return "Aggregation parameters"
    return "Data definitions"


from difss.report import robust_label                               # noqa: E402
pretty = {s: robust_label(s).replace("--", "–") for s in rb.specification}
gorder = ["Temporal weighting", "Aggregation parameters", "Normalisation", "Data definitions"]
gcol = {"Temporal weighting": C_POL, "Aggregation parameters": C_AMBER, "Normalisation": C_SHR,
        "Data definitions": C_MAIN}
rb["group"] = rb.specification.map(group)
rb["g"] = rb.group.map({g: i for i, g in enumerate(gorder)})
rb = rb.sort_values(["g", "tau"], ascending=[True, False]).reset_index(drop=True)
fig, ax = fs.figure(4.4)
ypos, yv, cur = [], 0.0, None
for r in rb.itertuples():
    if cur is not None and r.group != cur:
        yv += 0.8
    cur = r.group; ypos.append(-yv); yv += 1
ypos = np.array(ypos)
for yi, r in zip(ypos, rb.itertuples()):
    c = gcol[r.group]
    ax.hlines(yi, 0.5, r.tau, color=c, lw=1.2, alpha=0.6)
    ax.plot(r.tau, yi, "o", ms=5.5, color=c, mec="white", mew=0.6)
    ax.text(r.tau + 0.008, yi, f"{r.tau:.3f}", va="center", fontsize=SMALL_PT)
ax.set_yticks(ypos); ax.set_yticklabels([pretty[s] for s in rb.specification])
ax.set_xlim(0.5, 1.03); ax.set_xlabel(r"Kendall’s $\tau$ with the primary ranking")
ax.axvline(0.9, color="0.6", lw=0.7, ls=":")
for g in gorder:
    ax.plot([], [], "o", color=gcol[g], label=g)
ax.legend(loc="upper center", bbox_to_anchor=(0.4, 1.07), ncol=4, handletextpad=0.3)
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
fs.save(fig, os.path.join(FIG, "fig_robustness.png"))

# =========================================================== benchmarks (Figure 9)
bm = pd.read_csv(os.path.join(DATA, "benchmarks.csv"))
ev = pd.read_csv(os.path.join(DATA, "external_validation_ccpi.csv"))
short = {"Classical TOPSIS (crisp)": "TOPSIS (crisp)", "Weighted sum (crisp)": "Weighted sum (crisp)",
         "Weighted geometric mean (crisp, kappa=0)": "Weighted geometric mean (crisp)",
         "IFWA-DIFSS (arithmetic)": "IFWA (arithmetic)", "IF-TOPSIS": "IF-TOPSIS", "IF-VIKOR (v=0.5)": r"IF-VIKOR ($v=0.5$)",
         "Pythagorean fuzzy WG (q=2)": r"Pythagorean WG ($q=2$)", "q-rung orthopair fuzzy WG (q=3)": r"$q$-rung WG ($q=3$)",
         "IF-WASPAS, exponential recency (0.8)": "IF-WASPAS (recency)",
         "IFWG-DIFSS (proposed)": "Proposed (IFWG)"}
geo = {"Weighted geometric mean (crisp, kappa=0)", "Pythagorean fuzzy WG (q=2)", "q-rung orthopair fuzzy WG (q=3)"}
bm = bm.sort_values("tau", ascending=False)
methods = ["IFWG-DIFSS (proposed)"] + list(bm.method)          # top to bottom
tau = dict(zip(bm.method, bm.tau)); top = dict(zip(bm.method, bm.top10))
tau["IFWG-DIFSS (proposed)"] = 1.0; top["IFWG-DIFSS (proposed)"] = 10
yy = np.arange(len(methods))[::-1]
fig, axes = fs.figure(3.2, ncols=2, sharey=True, gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
colr = [C_MAIN if m == "IFWG-DIFSS (proposed)" else ("#2E8B57" if m in geo else C_GREY) for m in methods]
ax.barh(yy, [tau[m] for m in methods], color=colr, height=0.62)
for yi, m in zip(yy, methods):
    lab = "reference" if m == "IFWG-DIFSS (proposed)" else f"{tau[m]:.3f} ({top[m]}/10)"
    ax.text(tau[m] + 0.01, yi, lab, va="center", fontsize=SMALL_PT)
ax.set_yticks(yy); ax.set_yticklabels([short[m] for m in methods])
ax.get_yticklabels()[0].set_color(C_MAIN)
ax.set_xlim(0.5, 1.16); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); ax.set_xlabel(r"Kendall’s $\tau$ with the proposed ranking")
ax.set_title("(a) Agreement with benchmark methods")
h0, = ax.plot([], [], "s", color=C_MAIN, ms=6, label="Proposed")
h1, = ax.plot([], [], "s", color="#2E8B57", ms=6, label="Geometric aggregation")
h2, = ax.plot([], [], "s", color=C_GREY, ms=6, label="Arithmetic, hybrid or distance-based")
ax = axes[1]
piv = ev.pivot(index="method", columns="edition", values="spearman")
for yi, m in zip(yy, methods):
    r = piv.loc[m]
    ax.hlines(yi, r["CCPI 2024"], r["CCPI 2023"], color="0.8", lw=2)
    ax.plot(r["CCPI 2023"], yi, "o", color=C_POL, ms=5)
    ax.plot(r["CCPI 2024"], yi, "D", color=C_AMBER, ms=4.5)
ax.tick_params(axis="y", length=0)
lo, hi = float(piv.min().min()), float(piv.max().max())
ax.set_xlim(np.floor(lo * 20) / 20 - 0.02, np.ceil(hi * 20) / 20 + 0.02)
nce = ev.groupby("edition").n_common.first()
ax.set_xlabel(r"Spearman’s $\rho$ with the CCPI ranking")
ax.set_title("(b) Agreement with the CCPI")
h3, = ax.plot([], [], "o", color=C_POL, label=f"CCPI 2023 ($n={nce['CCPI 2023']}$)")
h4, = ax.plot([], [], "D", color=C_AMBER, label=f"CCPI 2024 ($n={nce['CCPI 2024']}$, primary)")
fig.legend(handles=[h0, h1, h2, h3, h4], loc="outside lower center", ncol=5)
fs.save(fig, os.path.join(FIG, "fig_benchmarks.png"))
print("done")
