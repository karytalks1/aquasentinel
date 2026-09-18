"""Rolling-reference pressure residuals - the core signal of this project.

Why not a single leak-free baseline?
------------------------------------
There is almost no leak-free data: only 2.2% of 2018 (8 days in January) and
*none* of 2019, where up to 16 leaks run concurrently. A model trained on one
fixed clean window would also have to extrapolate across a full year of
seasonal demand change.

Instead we refit a short-horizon 'expected pressure' model on a trailing
reference window and apply it to the current day. Leaks that were already
running are absorbed into the reference and stay silent, while a *new* leak
breaks the learned relationship and shows up as a step in the residual. This
turns an intractable 'is anything leaking?' question into the one a utility
actually cares about: 'has something *new* started leaking?'.

Model: pressure_i ~ Ridge(total inflow, inflow^2, tank level, time-of-day,
day-of-week), fitted per sensor on the trailing window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import config as C
import data_loader as dl

REF_DAYS = 21    # length of the trailing reference window
GAP_DAYS = 2     # skip the 2 days before 'today' so a leak that started
                 # yesterday does not contaminate its own reference
RIDGE_ALPHA = 1.0


def _design(years: int | tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hourly design matrix X and pressure targets P for one or more years."""
    if isinstance(years, int):
        years = (years,)
    s = pd.concat([dl.sensor_frame(y) for y in years]).sort_index()
    inflow = pd.concat([dl.total_inflow(y) for y in years]).sort_index()
    s = s.resample("1h").mean()
    inflow = inflow.resample("1h").mean()
    tf = dl.time_features(s.index)
    X = pd.concat(
        [
            inflow.rename("inflow"),
            (inflow ** 2).rename("inflow_sq"),
            s[["l_T1"]].rename(columns={"l_T1": "tank_level"}),
            tf,
        ],
        axis=1,
    )
    P = s[[c for c in s.columns if c.startswith("p_")]]
    P.columns = [c[2:] for c in P.columns]  # 'p_n1' -> 'n1'
    return X, P


def daily_residuals(year: int, use_cache: bool = True,
                    continuous: bool = True) -> pd.DataFrame:
    """Mean residual per pressure sensor per day (metres), for `year`.

    With `continuous=True` the reference window is allowed to reach back into
    the preceding year. This matters: a detector restarted every 1 January is
    blind for its first three weeks, which is exactly where two of the 2019
    leaks begin. A real utility never restarts, so neither do we.

    Returns a (days x 33) frame. Positive means the sensor read *higher* than
    the reference model expected.
    """
    cache = C.PROCESSED / f"residuals_{year}{'_cont' if continuous else ''}.csv"
    if use_cache and cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    years = (year - 1, year) if continuous and (year - 1) in (C.TRAIN_YEAR, C.TEST_YEAR) else year
    X, P = _design(years)
    days = pd.date_range(X.index[0].normalize(), X.index[-1].normalize(), freq="D")
    rows: dict[pd.Timestamp, pd.Series] = {}

    for day in days[REF_DAYS + GAP_DAYS:]:
        tr0 = day - pd.Timedelta(days=REF_DAYS + GAP_DAYS)
        tr1 = day - pd.Timedelta(days=GAP_DAYS)
        m_tr = (X.index >= tr0) & (X.index < tr1)
        m_te = (X.index >= day) & (X.index < day + pd.Timedelta(days=1))
        if m_tr.sum() < 100 or m_te.sum() < 12:
            continue
        model = Ridge(alpha=RIDGE_ALPHA).fit(X[m_tr], P[m_tr])
        rows[day] = (P[m_te] - model.predict(X[m_te])).mean()

    R = pd.DataFrame(rows).T
    R.index.name = "day"
    R = R[R.index.year == year]  # keep only the requested year
    R.to_csv(cache)
    return R


def normalised(R: pd.DataFrame, window: int = 28) -> pd.DataFrame:
    """Convert residuals to robust z-scores using a trailing MAD per sensor.

    A median-absolute-deviation scale is used rather than a standard deviation
    so that the leak we are trying to detect does not inflate its own threshold.
    """
    med = R.rolling(window, min_periods=10).median().shift(1)
    mad = (R - med).abs().rolling(window, min_periods=10).median().shift(1)
    scale = (1.4826 * mad).clip(lower=1e-3)  # 1.4826*MAD ~= sigma for a normal
    return (R - med) / scale


def signature(R: pd.DataFrame, day: pd.Timestamp, pre: int = 10, post: int = 5) -> pd.Series:
    """The 33-dim residual shift around `day` - the leak's spatial fingerprint."""
    before = R.loc[day - pd.Timedelta(days=pre): day - pd.Timedelta(days=1)]
    after = R.loc[day: day + pd.Timedelta(days=post)]
    if len(before) < 3 or len(after) < 3:
        return pd.Series(np.nan, index=R.columns)
    return after.mean() - before.mean()


# Windows considered by adaptive_signature(). Abrupt leaks show their whole
# effect within days; incipient ones ramp up over weeks, so no single window
# suits both (see reports/ - abrupt peaks at 5 d, incipient at 30 d).
CANDIDATE_WINDOWS = (5, 10, 20, 30)


def adaptive_signature(R: pd.DataFrame, day: pd.Timestamp, pre: int = 10,
                       windows: tuple[int, ...] = CANDIDATE_WINDOWS) -> pd.Series:
    """Signature using whichever post-window gives the strongest evidence.

    We cannot know at detection time whether a leak is abrupt or incipient, so
    we let the data choose: each candidate window is scored by the norm of its
    shift divided by the day-to-day residual noise over the same span, and the
    highest-scoring window wins. A slow leak naturally selects a long window
    once it has grown enough to beat the noise.
    """
    best, best_score = None, -np.inf
    for post in windows:
        sig = signature(R, day, pre=pre, post=post)
        if sig.isna().any():
            continue
        span = R.loc[day - pd.Timedelta(days=pre): day + pd.Timedelta(days=post)]
        noise = span.diff().std().mean() / np.sqrt(2)
        if not np.isfinite(noise) or noise <= 0:
            continue
        score = float(np.linalg.norm(sig.to_numpy())) / noise
        if score > best_score:
            best, best_score = sig, score
    if best is None:
        return signature(R, day, pre=pre, post=windows[0])
    return best
