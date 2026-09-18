"""Reproduce the whole study: residuals, simulation, results tables, figures.

Usage:  .venv/Scripts/python.exe run_all.py
"""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
warnings.filterwarnings("ignore")


def main() -> None:
    t0 = time.time()

    import config as C
    import residuals as res
    import simulate as sim
    import experiments as E
    import figures as F

    print("[1/5] pressure residuals")
    for year in (C.TRAIN_YEAR, C.TEST_YEAR):
        R = res.daily_residuals(year)
        print(f"      {year}: {R.shape[0]} days x {R.shape[1]} sensors")

    print("[2/5] EPANET leak sensitivity matrix")
    S = sim.sensitivity_matrix()
    print(f"      {S.shape[0]} pipes x {S.shape[1]} sensors")

    print("[3/5] experiments")
    out = E.run()
    for row in out["detection"]:
        print(f"      {row['year']} ({row['role']}): "
              f"{row['detected']}/{row['leaks']} detected, "
              f"{row['false_alarms']} false alarms, "
              f"median delay {row['median_delay_days']:.0f} d")

    print("[4/5] figures")
    F.build_all()

    print("[5/5] project guide PDF")
    import make_pdf
    make_pdf.main()

    print(f"\ndone in {time.time() - t0:.0f}s")
    print(f"tables  -> {C.TABLES}")
    print(f"figures -> {C.FIGURES}")


if __name__ == "__main__":
    main()
