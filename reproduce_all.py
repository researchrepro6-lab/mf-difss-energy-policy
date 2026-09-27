#!/usr/bin/env python3
"""Reproduce every number, table and figure of the manuscript (Windows, macOS, Linux).

Run from a terminal:      python reproduce_all.py          (add --quick for a smoke test)
or open this file in IDLE and press F5. Missing packages are installed automatically.
All steps run inside this Python process, so the same interpreter is used throughout.
"""
import os
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from difss.check_env import require      # noqa: E402


def step(name):
    print(f"\n== {name}", flush=True)


def main(quick: bool = False):
    require()
    t0 = time.time()

    step("1/5 unit tests")
    try:
        import pytest
        code = pytest.main(["-q", "tests"])
        if code != 0:
            sys.exit(f"Unit tests failed (pytest exit code {code}).")
    except ImportError:
        print("pytest is not installed; unit tests skipped.")

    step("2/5 analyses, tables, numbers, Figures 5 and 10-12")
    import run_all
    run_all.main(quick)

    step("3/5 TikZ Figures 1-3 and icons")
    runpy.run_path(str(ROOT / "scripts" / "build_tikz_figures.py"))["main"]()

    step("4/5 world map, policy chart and policy table")
    runpy.run_path(str(ROOT / "scripts" / "make_policy_and_map.py"), run_name="__main__")

    step("5/5 Figures 4 and 7-9")
    runpy.run_path(str(ROOT / "scripts" / "make_result_charts.py"), run_name="__main__")

    print(f"\nAll outputs are in outputs/ (tables/, figures/, tex/). Total time {time.time() - t0:.0f} s.")


if __name__ == "__main__":
    main("--quick" in sys.argv)
