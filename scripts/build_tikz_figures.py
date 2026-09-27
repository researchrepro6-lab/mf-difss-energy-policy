"""Build the TikZ figures (Figures 1-3) and the icon PNGs used in Figure 4.

Requires a LaTeX installation with TikZ and pgf-blur (TeX Live 2022+ or MiKTeX) and the
poppler utilities pdftoppm and pdftocairo (on Windows: install "poppler" and add its bin
folder to PATH). If these tools are missing, the step is skipped and the PNG files already
shipped in outputs/figures/ are kept.

Run from the bundle root:  python scripts/build_tikz_figures.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "figures_tikz"
FIG = ROOT / "outputs" / "figures"
ICONS = {"survey": "3B5BA5", "solar": "2E8B57", "turbine": "2E8B57", "factory": "8C5A3C"}
FIGURES = ["fig_framework", "fig_flowchart", "fig_concept"]


def run(cmd):
    subprocess.run(cmd, cwd=SRC, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    tools = {t: shutil.which(t) for t in ("pdflatex", "pdftoppm", "pdftocairo")}
    if not all(tools.values()):
        miss = ", ".join(t for t, p in tools.items() if not p)
        print(f"[build_tikz_figures] skipped: {miss} not found on PATH; "
              "keeping the shipped PNGs in outputs/figures/.")
        return 0
    FIG.mkdir(parents=True, exist_ok=True)
    (SRC / "icons_png").mkdir(exist_ok=True)
    for ic, col in ICONS.items():
        (SRC / "ic_tmp.tex").write_text(
            "\\documentclass[border=1pt]{standalone}\n\\usepackage{tikz}\\usetikzlibrary{calc,arrows.meta}\n"
            "\\input{icons}\n\\definecolor{cI}{HTML}{" + col + "}\n"
            "\\begin{document}\\begin{tikzpicture}\\pic[scale=2] at (0,0) {" + ic + "=cI};"
            "\\end{tikzpicture}\\end{document}\n", encoding="utf-8")
        run(["pdflatex", "-interaction=nonstopmode", "ic_tmp.tex"])
        run(["pdftocairo", "-png", "-transp", "-r", "400", "-singlefile", "ic_tmp.pdf", f"icons_png/{ic}"])
    from PIL import Image
    for f in FIGURES:
        run(["pdflatex", "-interaction=nonstopmode", f + ".tex"])
        run(["pdftoppm", "-r", "600", "-png", "-singlefile", f + ".pdf", str(FIG / f)])
        png = FIG / (f + ".png")
        Image.open(png).convert("RGB").save(png, dpi=(600, 600))
    for p in SRC.glob("*"):
        if p.suffix in {".aux", ".log", ".pdf"} or p.stem == "ic_tmp":
            p.unlink()
    print("[build_tikz_figures] Figures 1-3 and icons written to outputs/figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
