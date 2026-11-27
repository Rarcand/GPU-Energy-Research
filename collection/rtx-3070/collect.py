"""Collect the matched RTX 3070 grid under Windows/WSL idle monitoring."""
import os
from pathlib import Path
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    driver = root / "collection/shared/run_timing_validation.py"
    os.execv(sys.executable, [sys.executable, str(driver), *sys.argv[1:],
                              "--root", str(root), '--device', 'RTX3070-WSL', '--desktop'])
