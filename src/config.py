"""Shared paths, constants and ground-truth leak table for the L-Town study."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"

for _d in (PROCESSED, MODELS, FIGURES, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

INP_FILE = RAW / "L-TOWN.inp"

# The SCADA record is 5-minute resolution for all of 2018 and 2019.
SAMPLE_MINUTES = 5
STEPS_PER_HOUR = 60 // SAMPLE_MINUTES
STEPS_PER_DAY = 24 * STEPS_PER_HOUR

TRAIN_YEAR = 2018  # 'historical' dataset in BattLeDIM terms
TEST_YEAR = 2019   # 'evaluation' dataset - never used for fitting

PRESSURE_SENSORS = [
    "n1", "n4", "n31", "n54", "n105", "n114", "n163", "n188", "n215", "n229",
    "n288", "n296", "n332", "n342", "n410", "n415", "n429", "n458", "n469",
    "n495", "n506", "n516", "n519", "n549", "n613", "n636", "n644", "n679",
    "n722", "n726", "n740", "n752", "n769",
]
FLOW_SENSORS = ["p227", "p235", "PUMP_1"]
LEVEL_SENSORS = ["T1"]


@dataclass(frozen=True)
class Leak:
    """One ground-truth leak event from dataset_configuration.yaml."""

    pipe: str
    start: datetime
    end: datetime
    diameter_m: float
    kind: str        # 'abrupt' or 'incipient'
    peak: datetime

    @property
    def year(self) -> int:
        return self.start.year

    @property
    def area_cm2(self) -> float:
        import math
        return math.pi * (self.diameter_m * 100 / 2) ** 2


def _p(s: str) -> datetime:
    return datetime.strptime(s.strip(), "%Y-%m-%d %H:%M")


# Transcribed from dataset_configuration.yaml (the competition ground truth).
LEAKS: list[Leak] = [
    Leak("p257", _p("2018-01-08 13:30"), _p("2019-12-31 23:55"), 0.011843, "incipient", _p("2018-01-25 08:30")),
    Leak("p461", _p("2018-01-23 04:25"), _p("2018-04-02 11:40"), 0.021320, "incipient", _p("2018-03-27 20:35")),
    Leak("p232", _p("2018-01-31 02:35"), _p("2018-02-10 09:20"), 0.020108, "incipient", _p("2018-02-03 16:05")),
    Leak("p427", _p("2018-02-13 08:25"), _p("2019-12-31 23:55"), 0.0090731, "incipient", _p("2018-05-14 19:25")),
    Leak("p673", _p("2018-03-05 15:45"), _p("2018-03-23 10:25"), 0.022916, "abrupt", _p("2018-03-05 15:45")),
    Leak("p810", _p("2018-07-28 03:05"), _p("2019-12-31 23:55"), 0.010028, "incipient", _p("2018-11-02 22:25")),
    Leak("p628", _p("2018-05-02 14:55"), _p("2018-05-29 21:20"), 0.022318, "incipient", _p("2018-05-16 08:00")),
    Leak("p538", _p("2018-05-18 08:35"), _p("2018-06-02 06:05"), 0.021731, "abrupt", _p("2018-05-18 08:35")),
    Leak("p866", _p("2018-06-01 09:05"), _p("2018-06-12 03:00"), 0.018108, "abrupt", _p("2018-06-01 09:05")),
    Leak("p31",  _p("2018-06-28 10:35"), _p("2018-08-12 17:30"), 0.016389, "incipient", _p("2018-08-03 02:45")),
    Leak("p654", _p("2018-07-05 03:40"), _p("2019-12-31 23:55"), 0.0087735, "incipient", _p("2018-09-16 21:05")),
    Leak("p183", _p("2018-08-07 02:35"), _p("2018-09-01 17:10"), 0.015853, "abrupt", _p("2018-08-07 02:35")),
    Leak("p158", _p("2018-10-06 02:35"), _p("2018-10-23 13:35"), 0.019364, "abrupt", _p("2018-10-06 02:35")),
    Leak("p369", _p("2018-10-26 02:05"), _p("2018-11-08 20:25"), 0.019363, "abrupt", _p("2018-10-26 02:05")),
    Leak("p523", _p("2019-01-15 23:00"), _p("2019-02-01 09:50"), 0.020246, "abrupt", _p("2019-01-15 23:00")),
    Leak("p827", _p("2019-01-24 18:30"), _p("2019-02-07 09:05"), 0.02025, "abrupt", _p("2019-01-24 18:30")),
    Leak("p280", _p("2019-02-10 13:05"), _p("2019-12-31 23:55"), 0.0095008, "abrupt", _p("2019-02-10 13:05")),
    Leak("p653", _p("2019-03-03 13:10"), _p("2019-05-05 12:10"), 0.016035, "incipient", _p("2019-04-21 19:00")),
    Leak("p710", _p("2019-03-24 14:15"), _p("2019-12-31 23:55"), 0.0092936, "abrupt", _p("2019-03-24 14:15")),
    Leak("p514", _p("2019-04-02 20:40"), _p("2019-05-23 14:55"), 0.014979, "abrupt", _p("2019-04-02 20:40")),
    Leak("p331", _p("2019-04-20 10:10"), _p("2019-12-31 23:55"), 0.014053, "abrupt", _p("2019-04-20 10:10")),
    Leak("p193", _p("2019-05-19 10:40"), _p("2019-12-31 23:55"), 0.01239, "incipient", _p("2019-07-25 03:20")),
    Leak("p277", _p("2019-05-30 21:55"), _p("2019-12-31 23:55"), 0.012089, "incipient", _p("2019-08-11 15:05")),
    Leak("p142", _p("2019-06-12 19:55"), _p("2019-07-17 09:25"), 0.019857, "abrupt", _p("2019-06-12 19:55")),
    Leak("p680", _p("2019-07-10 08:45"), _p("2019-12-31 23:55"), 0.0097197, "abrupt", _p("2019-07-10 08:45")),
    Leak("p586", _p("2019-07-26 14:40"), _p("2019-09-16 03:20"), 0.017184, "incipient", _p("2019-08-28 07:55")),
    Leak("p721", _p("2019-08-02 03:00"), _p("2019-12-31 23:55"), 0.01408, "incipient", _p("2019-09-23 05:40")),
    Leak("p800", _p("2019-08-16 14:00"), _p("2019-10-01 16:35"), 0.018847, "incipient", _p("2019-09-07 21:05")),
    Leak("p123", _p("2019-09-13 20:05"), _p("2019-12-31 23:55"), 0.011906, "incipient", _p("2019-11-29 22:10")),
    Leak("p455", _p("2019-10-03 14:00"), _p("2019-12-31 23:55"), 0.012722, "incipient", _p("2019-12-16 05:25")),
    Leak("p762", _p("2019-10-09 10:15"), _p("2019-12-31 23:55"), 0.01519, "incipient", _p("2019-12-03 01:15")),
    Leak("p426", _p("2019-10-25 13:25"), _p("2019-12-31 23:55"), 0.015008, "abrupt", _p("2019-10-25 13:25")),
    Leak("p879", _p("2019-11-20 11:55"), _p("2019-12-31 23:55"), 0.013195, "incipient", _p("2019-12-31 23:55")),
]


def leaks_for(year: int) -> list[Leak]:
    """Leak events that *begin* in the given year."""
    return [lk for lk in LEAKS if lk.year == year]
