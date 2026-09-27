"""Check that the Python packages needed by the bundle are installed.

If a package is missing, it is installed automatically with pip (set the environment
variable DIFSS_NO_AUTO_INSTALL=1 to switch this off). If the installation fails, the script
stops with a message naming the package and the command that installs it.
"""
from __future__ import annotations

import importlib
import os
import site
import subprocess
import sys
from pathlib import Path

# import name -> (pip package name, what it is used for)
PACKAGES = {
    "numpy": ("numpy", "numerical computation"),
    "pandas": ("pandas", "data handling"),
    "scipy": ("scipy", "statistics (Kendall's tau, Spearman's rho)"),
    "matplotlib": ("matplotlib", "figures"),
    "openpyxl": ("openpyxl", "reading .xlsx source files"),
    "xlrd": ("xlrd", "reading the World Bank CO2 .xls source file"),
    "PIL": ("pillow", "writing 600-dpi PNG figures"),
}


def python_exe() -> str:
    """The console Python interpreter (IDLE runs scripts with pythonw.exe, which has no console)."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
        return str(exe.with_name("python.exe"))
    return str(exe)


def missing(names=None):
    out = []
    for mod in (names or PACKAGES):
        try:
            importlib.import_module(mod)
        except ImportError:
            out.append(mod)
    return out


def install(pip_names) -> bool:
    cmd = [python_exe(), "-m", "pip", "install", *pip_names]
    print("Installing missing package(s): " + ", ".join(pip_names))
    print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    tail = (res.stdout or "").strip().splitlines()[-3:]
    print("\n".join("  " + line for line in tail))
    if res.returncode != 0:
        print((res.stderr or "").strip()[-1500:])
        return False
    importlib.invalidate_caches()
    site.addsitedir(site.getusersitepackages())       # in case pip used --user
    return True


def require(names=None) -> None:
    miss = missing(names)
    if miss and os.environ.get("DIFSS_NO_AUTO_INSTALL") != "1":
        install([PACKAGES[m][0] for m in miss])
        miss = missing(names)
    if miss:
        exe = python_exe()
        pip_names = " ".join(PACKAGES[m][0] for m in miss)
        lines = ["", "Missing Python package(s):"]
        lines += [f"  - {PACKAGES[m][0]:12s} needed for {PACKAGES[m][1]}" for m in miss]
        lines += ["", "Open a Command Prompt (Windows) or terminal and run:",
                  f'  "{exe}" -m pip install {pip_names}',
                  "then run the script again (in IDLE: Run > Run Module, or F5).", ""]
        sys.exit("\n".join(lines))


if __name__ == "__main__":
    require()
    print("All required packages are installed.")
